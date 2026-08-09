#!/usr/bin/env python3
"""Run batch simulation evaluation with Phase-2 metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.run_meta import build_run_meta
from src.sim.controllers import get_controller
from src.sim.envs import make_env
from src.sim.stats import wilson_ci


def run_episodes(env, controller, episodes: int, max_steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rewards, lengths, successes = [], [], []
    failure_reasons = {"success": 0, "timeout": 0, "terminated_no_goal": 0}
    act_times_ms: list[float] = []
    total_steps = 0
    t_loop0 = time.perf_counter()

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        controller.reset(env)
        total_r, success = 0.0, False
        ended = "timeout"
        for step in range(max_steps):
            t0 = time.perf_counter()
            action = controller.act(env, obs)
            act_times_ms.append((time.perf_counter() - t0) * 1000.0)
            obs, r, terminated, truncated, _info = env.step(action)
            total_r += float(r)
            total_steps += 1
            if terminated:
                success = bool(getattr(env, "reward_info", {}).get("goal_reached", False))
                lengths.append(step + 1)
                ended = "success" if success else "terminated_no_goal"
                break
            if truncated:
                lengths.append(step + 1)
                ended = "timeout"
                break
        else:
            lengths.append(max_steps)
            ended = "timeout"
        rewards.append(total_r)
        successes.append(success)
        failure_reasons[ended] = failure_reasons.get(ended, 0) + 1

    wall = time.perf_counter() - t_loop0
    n_succ = int(np.sum(successes))
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "successes": n_succ,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "median_reward": float(np.median(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_length": float(np.median(lengths)) if lengths else 0.0,
        "failure_reasons": failure_reasons,
        "wilson_ci_95": wilson_ci(n_succ, episodes),
        "mean_act_ms": float(np.mean(act_times_ms)) if act_times_ms else 0.0,
        "steps_per_sec": float(total_steps / wall) if wall > 0 else 0.0,
        "total_steps": int(total_steps),
        "loop_wall_sec": float(wall),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    env_name = args.env or cfg["env"]["name"]
    t0 = time.time()
    env = make_env(env_name, cfg)

    ctrl_name = args.controller.lower()
    needs_ckpt = ctrl_name in ("model_vision", "bc", "bc_rgb", "rgb_bc")
    default_ckpt = (
        "outputs/checkpoints/bc_rgb/best.pt"
        if needs_ckpt and ctrl_name != "model_vision"
        else "outputs/checkpoints/run3_ft/best.pt"
    )
    ckpt = args.checkpoint or (default_ckpt if needs_ckpt else None)
    ckpt_path = PROJECT_ROOT / ckpt if ckpt else None
    if ckpt_path is not None and not ckpt_path.is_file() and ctrl_name == "model_vision":
        alt = PROJECT_ROOT / "outputs/checkpoints/run1/best.pt"
        if alt.is_file():
            ckpt_path = alt
    controller = get_controller(ctrl_name, checkpoint=ckpt_path)

    metrics = run_episodes(env, controller, args.episodes, args.max_steps, args.seed)
    metrics["controller"] = ctrl_name
    metrics["env"] = env_name
    metrics["wall_time_sec"] = time.time() - t0
    metrics["max_steps"] = args.max_steps
    metrics["seed"] = args.seed
    metrics["meta"] = build_run_meta(
        PROJECT_ROOT,
        config_path=args.config,
        checkpoint=ckpt_path,
        seed=args.seed,
        controller=ctrl_name,
        extra={"env": env_name, "episodes": args.episodes, "phase": 2},
    )
    env.close()

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
