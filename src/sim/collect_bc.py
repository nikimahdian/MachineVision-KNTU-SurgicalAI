#!/usr/bin/env python3
"""Collect oracle RGB→action demos for Behavior Cloning (slim)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim.bc_policy import preprocess_rgb
from src.sim.controllers import get_controller
from src.sim.envs import make_env


def _get_rgb(obs):
    if isinstance(obs, dict):
        return obs.get("rgb", obs.get("image"))
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/simulation.yaml")
    ap.add_argument("--out", default="data/bc_oracle")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--success-only", action="store_true", default=True)
    ap.add_argument("--keep-fail", action="store_true", help="Also keep failed episodes")
    ap.add_argument("--append", action="store_true", help="Merge with existing demos.npz")
    args = ap.parse_args()
    success_only = not args.keep_fail

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    existing_rgb = existing_act = existing_epid = existing_succ = None
    start_ep_offset = 0
    if args.append and (out / "demos.npz").exists():
        old = np.load(out / "demos.npz")
        existing_rgb = old["rgb"]
        existing_act = old["action"]
        existing_epid = old["ep_id"]
        existing_succ = old["success"]
        start_ep_offset = int(existing_epid.max()) + 1 if len(existing_epid) else 0
        print(f"APPEND existing steps={len(existing_rgb)} next_ep={start_ep_offset}", flush=True)

    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    env = make_env(cfg["env"]["name"], cfg)
    oracle = get_controller("oracle")

    rgbs, actions, ep_ids, successes = [], [], [], []
    n_ok = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        oracle.reset(env)
        ep_rgb, ep_act = [], []
        success = False
        for _ in range(args.max_steps):
            rgb = _get_rgb(obs)
            act = oracle.act(env, obs)
            ep_rgb.append(preprocess_rgb(rgb, size=args.img_size))
            ep_act.append(np.asarray(act, dtype=np.float32).reshape(-1))
            obs, _r, terminated, truncated, _info = env.step(act)
            if terminated:
                success = bool(getattr(env, "reward_info", {}).get("goal_reached", False))
                break
            if truncated:
                break
        if success_only and not success:
            print(f"ep {ep}: FAIL skip", flush=True)
            continue
        ep_tag = start_ep_offset + ep
        for x, a in zip(ep_rgb, ep_act):
            rgbs.append(x)
            actions.append(a)
            ep_ids.append(ep_tag)
            successes.append(int(success))
        n_ok += int(success)
        print(f"ep {ep}: success={success} steps={len(ep_rgb)} kept_ok={n_ok}", flush=True)

    env.close()
    if not rgbs and existing_rgb is None:
        raise SystemExit("No demos collected")

    if rgbs:
        rgbs_a = np.stack(rgbs, axis=0).astype(np.float32)
        acts_a = np.stack(actions, axis=0).astype(np.float32)
        ep_a = np.asarray(ep_ids, dtype=np.int32)
        succ_a = np.asarray(successes, dtype=np.int8)
        if existing_rgb is not None:
            rgbs_a = np.concatenate([existing_rgb, rgbs_a], axis=0)
            acts_a = np.concatenate([existing_act, acts_a], axis=0)
            ep_a = np.concatenate([existing_epid, ep_a], axis=0)
            succ_a = np.concatenate([existing_succ, succ_a], axis=0)
    else:
        rgbs_a, acts_a, ep_a, succ_a = existing_rgb, existing_act, existing_epid, existing_succ

    meta = {
        "episodes_requested": args.episodes,
        "episodes_kept": int(len(set(ep_a.tolist()))),
        "n_success_eps": int((succ_a > 0).sum() and len(set(ep_a[succ_a > 0].tolist()))),
        "n_steps": int(rgbs_a.shape[0]),
        "img_size": args.img_size,
        "action_dim": int(acts_a.shape[1]),
        "controller": "oracle",
        "success_only": success_only,
        "seed": args.seed,
        "append": bool(args.append),
    }
    np.savez_compressed(
        out / "demos.npz",
        rgb=rgbs_a,
        action=acts_a,
        ep_id=ep_a,
        success=succ_a,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved", out / "demos.npz", meta, flush=True)


if __name__ == "__main__":
    main()
