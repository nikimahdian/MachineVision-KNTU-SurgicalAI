"""Segmentation metrics: mIoU and Dice per class."""

from __future__ import annotations

import numpy as np
import torch


CLASS_NAMES = [
    "background",
    "abdominal_wall",
    "liver",
    "gi_tract",
    "fat",
    "grasper",
    "connective_tissue",
    "blood",
    "cystic_duct",
    "l_hook",
    "gallbladder",
    "hepatic_vein",
    "liver_ligament",
]


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 13,
    ignore_index: int = 255,
) -> np.ndarray:
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]
    cm = np.bincount(
        num_classes * targets.astype(int) + preds.astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Compute per-class and aggregate metrics.

    - miou_present / dice_present: mean over classes with GT support > 0
    - miou_macro / dice_macro: same as present (honest default for reporting)
    - miou_all / dice_all: mean over all classes (absent -> 0.0)
    """
    per_class = {}
    present_ious, present_dices = [], []
    all_ious, all_dices = [], []

    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else float("nan")
        dice = float(2 * tp / denom_dice) if denom_dice > 0 else float("nan")
        per_class[CLASS_NAMES[c]] = {
            "iou": None if np.isnan(iou) else iou,
            "dice": None if np.isnan(dice) else dice,
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "present": support > 0,
        }
        if support > 0:
            present_ious.append(iou)
            present_dices.append(dice)
            all_ious.append(iou)
            all_dices.append(dice)
        else:
            all_ious.append(0.0)
            all_dices.append(0.0)

    miou_present = float(np.mean(present_ious)) if present_ious else 0.0
    dice_present = float(np.mean(present_dices)) if present_dices else 0.0
    return {
        "miou_macro": miou_present,
        "dice_macro": dice_present,
        "miou_present": miou_present,
        "dice_present": dice_present,
        "miou_all": float(np.mean(all_ious)) if all_ious else 0.0,
        "dice_all": float(np.mean(all_dices)) if all_dices else 0.0,
        "n_classes_present": int(sum(1 for v in per_class.values() if v["present"])),
        "per_class": per_class,
    }
