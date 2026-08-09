"""Decode CholecSeg8k watershed masks (RGB gray-level encoding) to class IDs 0-12."""

from __future__ import annotations

import numpy as np

# Watershed masks store R=G=B gray levels derived from class hex colors (#111111 -> 11, etc.)
GRAY_TO_CLASS: dict[int, int] = {
    0: 0,
    50: 0,  # #505050 background
    11: 1,
    21: 2,
    13: 3,
    12: 4,
    31: 5,
    23: 6,
    24: 7,
    25: 8,
    32: 9,
    22: 10,
    33: 11,
    5: 12,
    255: 255,  # ignore / artifact pixels
}

LUT = np.full(256, 255, dtype=np.int64)
for gray, cls in GRAY_TO_CLASS.items():
    LUT[gray] = cls


def decode_watershed_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return LUT[mask.astype(np.uint8)]
