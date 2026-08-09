"""DeepLabV3+ / UNet++ segmentation models via SMP."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 13,
    architecture: str = "deeplabv3plus",
) -> nn.Module:
    arch = architecture.lower().replace("+", "plus").replace("_", "")
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    if arch in ("deeplabv3plus", "deeplabv3"):
        return smp.DeepLabV3Plus(**kwargs)
    if arch in ("unetplusplus", "unetpp", "unet++"):
        return smp.UnetPlusPlus(**kwargs)
    if arch == "unet":
        return smp.Unet(**kwargs)
    raise ValueError(
        f"Unknown architecture={architecture!r}. "
        "Use: deeplabv3plus | unetplusplus | unet"
    )
