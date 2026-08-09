"""Stats helpers for simulation benchmarks."""

from __future__ import annotations

import math
from typing import Any


def wilson_ci(successes: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a binomial success rate."""
    if n <= 0:
        return {"low": 0.0, "high": 0.0, "center": 0.0}
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return {
        "center": float(center),
        "low": float(max(0.0, center - margin)),
        "high": float(min(1.0, center + margin)),
        "level": 0.95,
    }


def aggregate_seed_runs(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed JSON dicts into one Phase-2 summary."""
    import numpy as np

    if not seed_results:
        return {}

    total_successes = int(sum(r.get("successes", 0) for r in seed_results))
    total_episodes = int(sum(r.get("episodes", 0) for r in seed_results))
    per_seed_sr = [float(r.get("success_rate", 0.0)) for r in seed_results]
    per_seed_reward = [float(r.get("mean_reward", 0.0)) for r in seed_results]
    per_seed_length = [float(r.get("mean_length", 0.0)) for r in seed_results]
    act_ms = [float(r.get("mean_act_ms", 0.0)) for r in seed_results if "mean_act_ms" in r]
    fps = [float(r.get("steps_per_sec", 0.0)) for r in seed_results if "steps_per_sec" in r]

    fail_counts: dict[str, int] = {}
    for r in seed_results:
        for k, v in (r.get("failure_reasons") or {}).items():
            fail_counts[k] = fail_counts.get(k, 0) + int(v)

    return {
        "n_seeds": len(seed_results),
        "episodes_total": total_episodes,
        "successes_total": total_successes,
        "success_rate_pooled": float(total_successes / total_episodes) if total_episodes else 0.0,
        "success_rate_mean_over_seeds": float(np.mean(per_seed_sr)),
        "success_rate_std_over_seeds": float(np.std(per_seed_sr)),
        "wilson_ci_95": wilson_ci(total_successes, total_episodes),
        "mean_reward_mean_over_seeds": float(np.mean(per_seed_reward)),
        "mean_reward_std_over_seeds": float(np.std(per_seed_reward)),
        "mean_length_mean_over_seeds": float(np.mean(per_seed_length)),
        "mean_length_std_over_seeds": float(np.std(per_seed_length)),
        "mean_act_ms": float(np.mean(act_ms)) if act_ms else None,
        "steps_per_sec": float(np.mean(fps)) if fps else None,
        "failure_reasons": fail_counts,
        "per_seed": seed_results,
    }
