"""Simulation environment factory."""

from __future__ import annotations

from typing import Any

from sofa_env.base import RenderFramework, RenderMode
from sofa_env.scenes.reach.reach_env import ReachEnv, ObservationType as ReachObs
from sofa_env.scenes.tissue_retraction.tissue_retraction_env import (
    ObservationType,
    TissueRetractionEnv,
)


def make_env(name: str, cfg: dict[str, Any]):
    env_cfg = cfg.get("env", cfg)
    env_name = (name or env_cfg.get("name", "tissue_retraction")).lower()
    image_shape = tuple(env_cfg.get("image_shape", [128, 128]))
    obs_type = env_cfg.get("observation_type", "rgb").lower()
    render_mode = RenderMode.HEADLESS
    framework = RenderFramework.PYGLET
    if env_cfg.get("render_framework", "pyglet").lower() == "pygame":
        framework = RenderFramework.PYGAME

    common = dict(
        image_shape=image_shape,
        render_mode=render_mode,
        render_framework=framework,
        time_step=float(env_cfg.get("time_step", 0.1)),
        frame_skip=int(env_cfg.get("frame_skip", 3)),
        settle_steps=int(env_cfg.get("settle_steps", 20)),
    )

    if env_name == "reach":
        observation_type = ReachObs.RGB if obs_type == "rgb" else ReachObs.STATE
        return ReachEnv(observation_type=observation_type, **common)

    if env_name in ("tissue_retraction", "tissue_retraction_env"):
        observation_type = ObservationType.RGB if obs_type == "rgb" else ObservationType.STATE
        return TissueRetractionEnv(
            observation_type=observation_type,
            observe_phase_state=(obs_type == "state"),
            **common,
        )

    raise ValueError(f"Unknown env: {env_name}")
