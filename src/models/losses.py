"""Combined CE + Dice loss for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 13,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        targets = targets.clone()
        targets[~valid] = 0

        dice_sum = 0.0
        weight_sum = 0.0
        for c in range(self.num_classes):
            pred_c = probs[:, c][valid]
            tgt_c = (targets == c)[valid]
            if tgt_c.numel() == 0:
                continue
            inter = (pred_c * tgt_c.float()).sum()
            denom = pred_c.sum() + tgt_c.float().sum()
            dice_c = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
            w = 1.0
            if self.class_weights is not None:
                w = float(self.class_weights[c].item())
            dice_sum = dice_sum + dice_c * w
            weight_sum = weight_sum + w
        return dice_sum / max(weight_sum, 1.0)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        num_classes: int = 13,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )
