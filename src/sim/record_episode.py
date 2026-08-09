#!/usr/bin/env python3
"""Record one simulation episode to MP4."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.controllers import get_controller
from src.sim.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--controller", default="heuristic")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    env = make_env(args.env or cfg["env"]["name"], cfg)
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
    controller = get_controller(args.controller, checkpoint=checkpoint)
    obs, _ = env.reset(seed=42)
    controller.reset(env)

    frames = []
    for _ in range(args.max_steps):
        if isinstance(obs, dict):
            rgb = obs.get("rgb", obs.get("image"))
        else:
            rgb = obs
        if rgb is not None:
            frames.append(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        if len(frames) >= args.max_frames:
            break
        action = controller.act(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("No frames captured")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, fr in enumerate(frames):
            cv2.imwrite(str(td_path / f"frame_{i:05d}.png"), fr)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(td_path / "frame_%05d.png"),
            "-pix_fmt", "yuv420p", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
