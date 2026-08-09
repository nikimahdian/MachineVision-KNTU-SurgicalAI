"""Controllers for sofa_env surgical tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def _max_action_magnitude(env) -> float:
    """Best-effort max action magnitude for velocity-like controllers."""
    v = getattr(env, "maximum_robot_velocity", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass

    space = getattr(env, "action_space", None)
    high = getattr(space, "high", None)
    if high is not None:
        try:
            return float(np.max(np.abs(high)))
        except Exception:
            pass

    return 1.0


class BaseController(ABC):
    @abstractmethod
    def reset(self, env) -> None:
        ...

    @abstractmethod
    def act(self, env, obs) -> np.ndarray:
        ...


class RandomController(BaseController):
    def reset(self, env) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:
        return env.action_space.sample().astype(np.float32)


class HeuristicController(BaseController):
    """Move gripper toward grasp point, then toward retract target (uses SOFA GT)."""

    def __init__(self, speed_scale: float = 0.5):
        self.speed_scale = speed_scale

    def reset(self, env) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:
        pos = env.end_effector.get_pose()[:3]
        phase = getattr(env, "_phase", None)
        phase_name = phase.name if phase is not None else "GRASPING"
        target = (
            env.get_end_position()
            if phase_name == "RETRACTING"
            else env.get_grasping_position()
        )
        delta = target - pos
        norm = np.linalg.norm(delta)
        if norm < 1e-6:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        direction = delta / norm
        vmax = _max_action_magnitude(env)
        return (direction * vmax * self.speed_scale).astype(np.float32)


class OracleController(HeuristicController):
    """GT targets at full speed — performance ceiling."""

    def __init__(self):
        super().__init__(speed_scale=1.0)


def get_controller(
    name: str,
    checkpoint: str | Path | None = None,
    config: str | Path = "configs/segmentation.yaml",
) -> BaseController:
    """Factory for all controllers including vision bridge."""
    key = name.lower().strip()
    if key == "random":
        return RandomController()
    if key == "heuristic":
        return HeuristicController()
    if key == "oracle":
        return OracleController()
    if key == "synthetic_vision":
        from .vision_bridge import SyntheticVisionController

        return SyntheticVisionController()
    if key == "model_vision":
        from .vision_bridge import ModelVisionController

        if checkpoint is None:
            raise ValueError("model_vision requires --checkpoint")
        return ModelVisionController(checkpoint, config=config)
    if key in ("bc", "bc_rgb", "rgb_bc"):
        from .bc_policy import BCController

        if checkpoint is None:
            raise ValueError("bc_rgb requires --checkpoint")
        return BCController(checkpoint)
    raise ValueError(
        f"Unknown controller: {name}. "
        "Use: random | heuristic | oracle | synthetic_vision | model_vision | bc_rgb"
    )
