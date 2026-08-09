"""Vision-guided controllers: image features -> Cartesian velocity.

Honest bridge: does NOT fall back to SOFA ground-truth targets.
Re-run simulation on Linux/SOFA after this change — metrics will differ
from the old stub that cloned HeuristicController.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .controllers import BaseController, _max_action_magnitude


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CholecSeg8k class ids used by ModelVisionController
CLASS_GRASPER = 5
CLASS_L_HOOK = 9
CLASS_GALLBLADDER = 10
CLASS_LIVER = 2
CLASS_FAT = 4
CLASS_CONNECTIVE = 6


def _get_rgb(obs) -> np.ndarray | None:
    if isinstance(obs, dict):
        rgb = obs.get("rgb", obs.get("image"))
    else:
        rgb = obs
    if rgb is None or getattr(rgb, "ndim", 0) != 3:
        return None
    return np.asarray(rgb)


def extract_features(rgb: np.ndarray) -> dict:
    """Centroid features from HSV color masks on sim RGB frame."""
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    tool_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    tissue_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([20, 255, 255]))
    # purple-ish target marker often used in tissue-retraction scenes
    marker_mask = cv2.inRange(hsv, np.array([130, 50, 50]), np.array([170, 255, 255]))

    def centroid(mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    return {
        "tool_xy": centroid(tool_mask),
        "tissue_xy": centroid(tissue_mask),
        "marker_xy": centroid(marker_mask),
        "tool_mask_sum": int(tool_mask.sum()),
        "tissue_mask_sum": int(tissue_mask.sum()),
    }


def image_delta_to_action(
    tool_xy: tuple[float, float] | None,
    target_xy: tuple[float, float] | None,
    image_shape: tuple[int, ...],
    vmax: float,
    gain: float = 1.0,
    z_down: float = 0.15,
) -> np.ndarray | None:
    """Map 2D image offset (target - tool) to 3D Cartesian velocity.

    Camera assumed roughly top-down. Image +x -> world +x, image +y (down) -> world -y.
    Small constant -z helps approach tissue surface when features exist.
    """
    if tool_xy is None or target_xy is None:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    diag = float(np.hypot(w, h)) + 1e-6
    dx = (target_xy[0] - tool_xy[0]) / diag
    dy = (target_xy[1] - tool_xy[1]) / diag
    vec = np.array([dx, -dy, -abs(z_down) * 0.25], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    direction = vec / norm
    # stronger proportional speed so arm actually moves when offset is visible
    speed = float(np.clip(norm * gain * 6.0, 0.25, 1.0)) * vmax
    return (direction * speed).astype(np.float32)


def _pad_action(env, vec3: np.ndarray) -> np.ndarray:
    shape = env.action_space.shape
    out = np.zeros(shape, dtype=np.float32)
    n = min(3, int(np.prod(shape)))
    out.ravel()[:n] = vec3.ravel()[:n]
    return out


class VisionImageController(BaseController):
    """Base: RGB features -> action. No SOFA GT targets."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.last_features: dict | None = None

    def reset(self, env) -> None:
        self.last_features = None

    def _select_target(self, feats: dict, env) -> tuple[float, float] | None:
        phase = getattr(env, "_phase", None)
        phase_name = phase.name if phase is not None else "GRASPING"
        if phase_name == "RETRACTING":
            return feats.get("marker_xy") or feats.get("tissue_xy")
        return feats.get("tissue_xy") or feats.get("marker_xy")

    def act(self, env, obs) -> np.ndarray:
        rgb = _get_rgb(obs)
        if rgb is None:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        feats = extract_features(rgb)
        self.last_features = feats
        target = self._select_target(feats, env)
        # if tool missing but tissue exists, nudge from image center toward tissue
        tool = feats.get("tool_xy")
        if tool is None and target is not None:
            h, w = rgb.shape[:2]
            tool = (w * 0.5, h * 0.5)
        vmax = _max_action_magnitude(env)
        vec = image_delta_to_action(
            tool,
            target,
            rgb.shape,
            vmax=vmax,
            gain=self.gain,
        )
        if vec is None:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        return _pad_action(env, vec)


class SyntheticVisionController(VisionImageController):
    """HSV synthetic perception on sim frames -> velocity command."""

    pass


class ModelVisionController(VisionImageController):
    """DeepLabV3+ on sim RGB -> class centroids -> velocity command."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path = "configs/segmentation.yaml",
        gain: float = 1.0,
        img_size: int = 512,
    ):
        import torch

        super().__init__(gain=gain)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        from src.models.segmentation import build_model

        cfg_path = Path(config)
        if not cfg_path.is_absolute():
            # resolve relative to project root (…/src/sim -> parents[2])
            root = Path(__file__).resolve().parents[2]
            cfg_path = root / config
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.model = build_model(
            encoder=cfg["model"]["encoder"],
            encoder_weights=None,
            classes=cfg["model"]["classes"],
        ).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _preprocess(self, rgb: np.ndarray):
        import torch

        x = cv2.resize(rgb.astype(np.uint8), (self.img_size, self.img_size))
        x = x.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)
        return t

    @staticmethod
    def _centroid_of_classes(mask: np.ndarray, class_ids: list[int]):
        ys, xs = np.where(np.isin(mask, class_ids))
        if len(xs) == 0:
            return None
        # scale back later by caller using original image size
        return float(xs.mean()), float(ys.mean())

    def act(self, env, obs) -> np.ndarray:
        import torch

        rgb = _get_rgb(obs)
        if rgb is None:
            return np.zeros(env.action_space.shape, dtype=np.float32)

        # Prefer HSV when sim colors are strong; also run model for logging / hybrid
        feats = extract_features(rgb)
        x = self._preprocess(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(x).argmax(dim=1)[0].cpu().numpy()

        h, w = rgb.shape[:2]
        tool = self._centroid_of_classes(pred, [CLASS_GRASPER, CLASS_L_HOOK])
        tissue = self._centroid_of_classes(
            pred, [CLASS_GALLBLADDER, CLASS_LIVER, CLASS_FAT, CLASS_CONNECTIVE]
        )
        # map 512-space centroids to original image coords
        if tool is not None:
            tool = (tool[0] * w / self.img_size, tool[1] * h / self.img_size)
        if tissue is not None:
            tissue = (tissue[0] * w / self.img_size, tissue[1] * h / self.img_size)

        # Hybrid: model centroids if available, else HSV (sim domain gap is large)
        feats["tool_xy"] = tool or feats.get("tool_xy")
        feats["tissue_xy"] = tissue or feats.get("tissue_xy")
        self.last_features = feats

        target = self._select_target(feats, env)
        tool_xy = feats.get("tool_xy")
        if tool_xy is None and target is not None:
            tool_xy = (w * 0.5, h * 0.5)
        vmax = _max_action_magnitude(env)
        vec = image_delta_to_action(
            tool_xy,
            target,
            rgb.shape,
            vmax=vmax,
            gain=self.gain,
        )
        if vec is None:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        return _pad_action(env, vec)
