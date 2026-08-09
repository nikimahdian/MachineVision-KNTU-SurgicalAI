"""CholecSeg8k PyTorch dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .mask_utils import decode_watershed_mask
from .splits import list_samples
from .transforms import train_transforms, val_transforms

NUM_CLASSES = 13


class CholecSeg8kDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 512,
        transform=None,
        aug_cfg: dict | None = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.samples = list_samples(self.root, split)

        if not self.samples:
            raise FileNotFoundError(
                f"No samples for split={split!r} under {self.root}. "
                "Run scripts/03_download_cholecseg8k.sh first."
            )

        if transform is None:
            if split == "train":
                transform = train_transforms(img_size, aug_cfg=aug_cfg)
            else:
                transform = val_transforms(img_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        mask = decode_watershed_mask(mask)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {image_path} or {mask_path}")

        augmented = self.transform(image=image, mask=mask)
        img_t = torch.from_numpy(augmented["image"]).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(augmented["mask"]).long()
        return img_t, mask_t
