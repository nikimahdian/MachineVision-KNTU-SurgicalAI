#!/usr/bin/env python3
"""Train DeepLabV3+ / UNet++ on CholecSeg8k with reproducible metadata."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_meta import build_run_meta
from src.datasets.cholecseg8k import NUM_CLASSES, CholecSeg8kDataset
from src.datasets.splits import split_counts
from src.models.losses import CombinedLoss
from src.models.metrics import confusion_matrix, metrics_from_confusion
from src.models.segmentation import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_weights(
    path: Path,
    device: torch.device,
    train_loader: DataLoader | None = None,
    power: float = 0.5,
    boost: dict[int, float] | None = None,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Inverse-frequency weights with optional sqrt compression (`power=0.5`)."""
    if path.exists() and not force_recompute:
        w = np.load(path)
        if np.any(w > 0):
            weights = w.astype(np.float64)
        else:
            weights = None
    else:
        weights = None

    if weights is None:
        if train_loader is None:
            return torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
        counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        for _, masks in train_loader:
            valid = masks != 255
            for c in range(NUM_CLASSES):
                counts[c] += ((masks == c) & valid).sum().item()
        freq = counts / max(counts.sum(), 1.0)
        weights = 1.0 / np.maximum(freq, 1e-8)
        weights = np.power(weights, power)
        weights = weights / weights.sum() * NUM_CLASSES
        if boost:
            for idx, mul in boost.items():
                i = int(idx)
                if 0 <= i < len(weights):
                    weights[i] *= float(mul)
            weights = weights / weights.sum() * NUM_CLASSES
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, weights)
    elif boost:
        # apply boost even on cached base weights, then renormalize once for this run
        weights = weights.astype(np.float64).copy()
        for idx, mul in boost.items():
            i = int(idx)
            if 0 <= i < len(weights):
                weights[i] *= float(mul)
        weights = weights / weights.sum() * NUM_CLASSES

    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds, masks, NUM_CLASSES)
    return metrics_from_confusion(cm)


def train_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--init-checkpoint", default=None, help="Warm-start from existing .pt")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data or cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    out_dir = Path(args.output or cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = PROJECT_ROOT / cfg["output"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    img_size = args.img_size or train_cfg["img_size"]
    amp = args.amp or train_cfg.get("amp", False)
    seed = int(train_cfg.get("seed", 42))
    aug_cfg = cfg.get("augmentation") or {}

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CholecSeg8kDataset(data_root, split="train", img_size=img_size, aug_cfg=aug_cfg)
    val_ds = CholecSeg8kDataset(data_root, split="val", img_size=img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
    )

    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        classes=cfg["model"]["classes"],
        architecture=cfg["model"].get("architecture", "deeplabv3plus"),
    ).to(device)

    init_ckpt = args.init_checkpoint or cfg.get("training", {}).get("init_checkpoint")
    if init_ckpt:
        init_path = Path(init_ckpt)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        warm = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(warm["model"])
        print(f"Warm-started from {init_path} (epoch={warm.get('epoch')})", flush=True)

    weights_path = PROJECT_ROOT / "outputs" / "metrics" / "class_weights.npy"
    cfg_wpath = train_cfg.get("class_weights_path")
    if cfg_wpath:
        weights_path = Path(cfg_wpath)
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
    use_weights = bool(train_cfg.get("use_class_weights", True))
    boost_raw = train_cfg.get("class_weight_boost") or {}
    boost = {int(k): float(v) for k, v in boost_raw.items()}
    if use_weights:
        class_weights = load_class_weights(
            weights_path,
            device,
            train_loader,
            power=float(train_cfg.get("class_weight_power", 0.5)),
            boost=boost or None,
            force_recompute=bool(train_cfg.get("force_recompute_class_weights", False)),
        )
        print(
            f"class_weights path={weights_path} power={train_cfg.get('class_weight_power')} "
            f"boost={boost} min={class_weights.min().item():.3f} max={class_weights.max().item():.3f}",
            flush=True,
        )
    else:
        class_weights = None

    criterion = CombinedLoss(
        class_weights=class_weights,
        ce_weight=train_cfg.get("loss_ce_weight", 0.5),
        dice_weight=train_cfg.get("loss_dice_weight", 0.5),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    writer = SummaryWriter(out_dir / "tb") if cfg["output"].get("tensorboard", True) else None

    best_miou = -1.0
    best_epoch = 0
    patience = int(train_cfg.get("early_stopping_patience", 0) or 0)
    min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    epochs_no_improve = 0
    run_name = out_dir.name
    log_file = log_dir / f"train_{run_name}.log"
    history = []

    split_info = split_counts(data_root)
    print(
        f"split train={len(train_ds)} val={len(val_ds)} "
        f"videos train={split_info['train_videos']} val={split_info['val_videos']} "
        f"{split_info['val_video_names']}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, amp)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        line = (
            f"epoch={epoch}/{epochs} loss={train_loss:.4f} "
            f"val_miou={val_metrics['miou_present']:.4f} "
            f"val_dice={val_metrics['dice_present']:.4f} "
            f"present={val_metrics['n_classes_present']}"
        )
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_miou_present": val_metrics["miou_present"],
                "val_dice_present": val_metrics["dice_present"],
                "val_miou_all": val_metrics["miou_all"],
            }
        )

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("metrics/val_miou_present", val_metrics["miou_present"], epoch)
            writer.add_scalar("metrics/val_dice_present", val_metrics["dice_present"], epoch)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "metrics": val_metrics,
            "config": str(cfg_path.relative_to(PROJECT_ROOT)),
            "seed": seed,
        }
        torch.save(payload, out_dir / "last.pt")

        improved = val_metrics["miou_present"] > best_miou + min_delta
        if improved:
            best_miou = val_metrics["miou_present"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(payload, out_dir / "best.pt")
            print(f"  -> new best mIoU_present {best_miou:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch={best_epoch}, best mIoU={best_miou:.4f}, patience={patience})",
                    flush=True,
                )
                break

    if writer:
        writer.close()

    summary = {
        "best_miou_present": best_miou,
        "best_epoch": best_epoch,
        "seed": seed,
        "split": split_info,
        "history": history,
        "meta": build_run_meta(
            PROJECT_ROOT,
            config_path=cfg_path,
            checkpoint=out_dir / "best.pt",
            seed=seed,
            extra={
                "phase": 1,
                "run": run_name,
                "architecture": cfg["model"].get("architecture", "deeplabv3plus"),
                "aug_strength": aug_cfg.get("strength", "basic"),
                "init_checkpoint": str(init_ckpt) if init_ckpt else None,
            },
        ),
    }
    (out_dir / "history.json").write_text(json.dumps(summary, indent=2))
    print(f"Training done. best mIoU_present={best_miou:.4f} @ epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
