"""Albumentations transforms for CholecSeg8k."""

from __future__ import annotations

from typing import Any

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(img_size: int = 512, aug_cfg: dict[str, Any] | None = None) -> A.Compose:
    """Build train transforms. `aug_cfg` comes from YAML `augmentation:` block."""
    cfg = aug_cfg or {}
    strength = str(cfg.get("strength", "basic")).lower()

    ops: list[Any] = [A.Resize(img_size, img_size)]
    ops.append(A.HorizontalFlip(p=float(cfg.get("horizontal_flip", 0.5))))

    if strength in ("basic", "strong", "surgical"):
        ops.append(
            A.ColorJitter(
                brightness=float(cfg.get("brightness", 0.2)),
                contrast=float(cfg.get("contrast", 0.2)),
                saturation=float(cfg.get("saturation", 0.1)),
                hue=float(cfg.get("hue", 0.02)),
                p=0.8,
            )
        )

    if strength in ("strong", "surgical"):
        ops.extend(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                    ],
                    p=float(cfg.get("blur_p", 0.35)),
                ),
                A.GaussNoise(std_range=(0.02, 0.08), p=float(cfg.get("noise_p", 0.3))),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=float(cfg.get("clahe_p", 0.25))),
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.4,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=25,
                    val_shift_limit=20,
                    p=0.4,
                ),
                # Partial instrument/tissue occlusion
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(0.02, 0.12),
                    hole_width_range=(0.02, 0.12),
                    fill=0,
                    p=float(cfg.get("dropout_p", 0.35)),
                ),
            ]
        )

    if strength == "surgical":
        ops.append(
            A.OneOf(
                [
                    A.Downscale(scale_range=(0.6, 0.9), p=1.0),
                    A.ImageCompression(quality_range=(50, 90), p=1.0),
                ],
                p=0.25,
            )
        )

    ops.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(ops)


def val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
