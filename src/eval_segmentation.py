#!/usr/bin/env python3
"""Evaluate segmentation checkpoint with rich Phase-1 metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_meta import build_run_meta
from src.datasets.cholecseg8k import NUM_CLASSES, CholecSeg8kDataset
from src.datasets.splits import split_counts
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.metrics import CLASS_NAMES, confusion_matrix, metrics_from_confusion
from src.models.segmentation import build_model


def denorm(img: torch.Tensor) -> np.ndarray:
    x = img.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    return np.clip(x * std + mean, 0, 1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", default="outputs/metrics/seg_eval.json")
    parser.add_argument("--num-overlays", type=int, default=20)
    parser.add_argument("--overlay-dir", default=None)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(args.data or cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        encoder=cfg["model"]["encoder"],
        encoder_weights=None,
        classes=cfg["model"]["classes"],
        architecture=cfg["model"].get("architecture", "deeplabv3plus"),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_size = cfg["training"]["img_size"]
    ds = CholecSeg8kDataset(data_root, split=args.split, img_size=img_size)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else PROJECT_ROOT / "outputs" / "masks" / "predictions"
    if not overlay_dir.is_absolute():
        overlay_dir = PROJECT_ROOT / overlay_dir
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for images, masks in tqdm(loader, desc="eval"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        cm += confusion_matrix(preds, masks, NUM_CLASSES)

        if saved < args.num_overlays:
            for i in range(images.size(0)):
                if saved >= args.num_overlays:
                    break
                rgb = denorm(images[i])
                pred = preds[i].cpu().numpy()
                gt = masks[i].cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(rgb)
                ax[0].set_title("image")
                ax[1].imshow(gt, cmap="tab20", vmin=0, vmax=12)
                ax[1].set_title("gt")
                ax[2].imshow(pred, cmap="tab20", vmin=0, vmax=12)
                ax[2].set_title("pred")
                for a in ax:
                    a.axis("off")
                fig.tight_layout()
                fig.savefig(overlay_dir / f"overlay_{saved:03d}.png", dpi=120)
                plt.close(fig)
                saved += 1

    metrics = metrics_from_confusion(cm)
    metrics["split"] = args.split
    metrics["n_images"] = len(ds)
    metrics["checkpoint"] = str(ckpt_path.relative_to(PROJECT_ROOT)) if ckpt_path.is_relative_to(PROJECT_ROOT) else str(ckpt_path)
    metrics["epoch"] = ckpt.get("epoch")
    metrics["split_info"] = split_counts(data_root)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["meta"] = build_run_meta(
        PROJECT_ROOT,
        config_path=args.config,
        checkpoint=ckpt_path,
        seed=cfg["training"].get("seed"),
        extra={"phase": 1, "split": args.split},
    )

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_dir = out_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem

    ious = [metrics["per_class"][n]["iou"] or 0.0 for n in CLASS_NAMES]
    supports = [metrics["per_class"][n]["support"] for n in CLASS_NAMES]
    order = np.argsort(ious)[::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(range(NUM_CLASSES), [ious[i] for i in order])
    plt.xticks(range(NUM_CLASSES), [CLASS_NAMES[i] for i in order], rotation=45, ha="right")
    plt.ylabel("IoU")
    plt.title(f"Per-class IoU sorted ({args.split}) miou_present={metrics['miou_present']:.3f}")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_iou_sorted.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues")
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90, fontsize=7)
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=7)
    plt.title("Confusion (row-normalized)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_confusion.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(range(NUM_CLASSES), supports)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("GT pixels")
    plt.yscale("log")
    plt.title("Class support (log)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{stem}_support.png", dpi=120)
    plt.close()

    print(
        json.dumps(
            {
                "miou_present": metrics["miou_present"],
                "dice_present": metrics["dice_present"],
                "miou_all": metrics["miou_all"],
                "n_classes_present": metrics["n_classes_present"],
                "split": args.split,
                "epoch": metrics["epoch"],
            },
            indent=2,
        )
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
