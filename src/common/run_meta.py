"""Collect reproducible run metadata for metric JSON files."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def build_run_meta(
    project_root: Path,
    *,
    config_path: Path | str | None = None,
    checkpoint: Path | str | None = None,
    seed: int | None = None,
    controller: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cfg = Path(config_path) if config_path else None
    if cfg is not None and not cfg.is_absolute():
        cfg = root / cfg
    ckpt = Path(checkpoint) if checkpoint else None
    if ckpt is not None and not ckpt.is_absolute():
        ckpt = root / ckpt

    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "torch": _pkg_version("torch"),
            "numpy": _pkg_version("numpy"),
            "cv2": _pkg_version("cv2"),
            "yaml": _pkg_version("yaml"),
            "segmentation_models_pytorch": _pkg_version("segmentation_models_pytorch"),
        },
    }
    if seed is not None:
        meta["seed"] = int(seed)
    if controller is not None:
        meta["controller"] = controller
    if cfg is not None:
        meta["config"] = str(cfg.relative_to(root)) if cfg.is_relative_to(root) else str(cfg)
        meta["config_sha256"] = _file_sha256(cfg)
    if ckpt is not None:
        try:
            rel = str(ckpt.relative_to(root))
        except ValueError:
            rel = str(ckpt)
        meta["checkpoint"] = rel
        meta["checkpoint_sha256"] = _file_sha256(ckpt)
    if extra:
        meta.update(extra)
    return meta
