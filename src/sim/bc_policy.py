"""RGB→action Behavior Cloning policies for TissueRetraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TinyRGBPolicy(nn.Module):
    """64×64 RGB → action(3). Fast v1."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrongRGBPolicy(nn.Module):
    """Stronger 64×64 RGB→action for Phase-2 BC."""

    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_policy(arch: str = "tiny", action_dim: int = 3) -> nn.Module:
    key = (arch or "tiny").lower()
    if key in ("strong", "v2", "bc_v2"):
        return StrongRGBPolicy(action_dim=action_dim)
    return TinyRGBPolicy(action_dim=action_dim)


def preprocess_rgb(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """HWC uint8/float → CHW float32 [0,1] at `size`."""
    import cv2

    if isinstance(rgb, dict):
        rgb = rgb.get("rgb", rgb.get("image"))
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    x = arr.astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))  # CHW


class BCController:
    """Closed-loop controller from trained RGB BC checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = Path(checkpoint)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.size = int(ckpt.get("img_size", 64))
        action_dim = int(ckpt.get("action_dim", 3))
        arch = str(ckpt.get("arch", "tiny"))
        self.model = build_policy(arch=arch, action_dim=action_dim).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.action_scale = float(ckpt.get("action_scale", 1.0))

    def reset(self, env) -> None:
        pass

    @torch.no_grad()
    def act(self, env, obs) -> np.ndarray:
        x = preprocess_rgb(obs, size=self.size)
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        a = self.model(t).squeeze(0).cpu().numpy().astype(np.float32)
        a = a * self.action_scale
        space = getattr(env, "action_space", None)
        if space is not None and hasattr(space, "low"):
            a = np.clip(a, space.low, space.high).astype(np.float32)
        shape = getattr(space, "shape", (3,))
        if a.shape != shape:
            out = np.zeros(shape, dtype=np.float32)
            n = min(a.size, out.size)
            out.flat[:n] = a.flat[:n]
            return out
        return a
