#!/usr/bin/env python3
"""Train RGB BC policy from oracle demos (v1 tiny / v2 strong)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.bc_policy import build_policy


class DemoDS(Dataset):
    def __init__(self, rgb: np.ndarray, action: np.ndarray, augment: bool = False):
        self.rgb = torch.from_numpy(rgb)  # NCHW float [0,1]
        self.action = torch.from_numpy(action)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.rgb.shape[0])

    def __getitem__(self, i: int):
        x = self.rgb[i]
        y = self.action[i]
        if self.augment:
            # light photometric noise only — no geometric flip (action axes coupled)
            if torch.rand(1).item() < 0.5:
                x = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
            if torch.rand(1).item() < 0.3:
                x = (x * (0.9 + 0.2 * torch.rand(1).item())).clamp(0, 1)
        return x, y

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bc_oracle/demos.npz")
    ap.add_argument("--out", default="outputs/checkpoints/bc_rgb_v2/best.pt")
    ap.add_argument("--arch", default="strong", choices=["tiny", "strong"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", action="store_true", default=True)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--smooth-weight", type=float, default=0.05)
    args = ap.parse_args()
    use_aug = args.aug and not args.no_aug

    data_path = PROJECT_ROOT / args.data
    z = np.load(data_path)
    rgb, action = z["rgb"], z["action"]
    scale = float(np.max(np.abs(action)) + 1e-6)
    action_n = (action / scale).astype(np.float32)

    full = DemoDS(rgb, action_n, augment=False)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(range(len(full)), [n_train, n_val], generator=g)
    train_ds = DemoDS(rgb[list(train_idx.indices)], action_n[list(train_idx.indices)], augment=use_aug)
    val_ds = DemoDS(rgb[list(val_idx.indices)], action_n[list(val_idx.indices)], augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_policy(args.arch, action_dim=action.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()

    best_val = float("inf")
    history = []
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"arch={args.arch} steps={len(rgb)} aug={use_aug} device={device}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mse(pred, y)
            if args.smooth_weight > 0 and x.size(0) > 1:
                # penalize large actions a bit (stability)
                loss = loss + args.smooth_weight * (pred.pow(2).mean())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr.append(float(loss.item()))
        sched.step()
        model.eval()
        va = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                va.append(float(mse(model(x), y).item()))
        tr_m, va_m = float(np.mean(tr)), float(np.mean(va))
        history.append({"epoch": epoch, "train_mse": tr_m, "val_mse": va_m})
        print(f"epoch {epoch}: train_mse={tr_m:.5f} val_mse={va_m:.5f}", flush=True)
        if va_m < best_val:
            best_val = va_m
            torch.save(
                {
                    "model": model.state_dict(),
                    "img_size": int(rgb.shape[2]),
                    "action_dim": int(action.shape[1]),
                    "action_scale": scale,
                    "val_mse": best_val,
                    "n_steps": int(rgb.shape[0]),
                    "arch": args.arch,
                },
                out_path,
            )
            print(f"  saved best -> {out_path}", flush=True)

    meta = {
        "best_val_mse": best_val,
        "history": history,
        "action_scale": scale,
        "checkpoint": str(out_path),
        "arch": args.arch,
        "n_steps": int(rgb.shape[0]),
    }
    (out_path.parent / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", meta, flush=True)


if __name__ == "__main__":
    main()
