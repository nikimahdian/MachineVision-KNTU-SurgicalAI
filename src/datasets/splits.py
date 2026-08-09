"""Video-level train/val split for CholecSeg8k (no frame leakage)."""

from __future__ import annotations

import re
from pathlib import Path

VAL_VIDEOS = frozenset({"video01", "video05", "video09", "video13", "video17"})

_VIDEO_RE = re.compile(r"^(video\d+)")


def video_from_folder(name: str) -> str | None:
    match = _VIDEO_RE.match(name)
    return match.group(1) if match else None


def iter_mask_paths(root: Path):
    for pattern in ("*_endo_watershed_mask.png", "*_watershed.png"):
        yield from root.rglob(pattern)


def videos_in_dataset(root: Path) -> set[str]:
    videos: set[str] = set()
    for mask_path in iter_mask_paths(root):
        video = video_from_folder(mask_path.parent.name)
        if video:
            videos.add(video)
    return videos


def val_videos_for_root(root: Path) -> frozenset[str]:
    return frozenset(VAL_VIDEOS & videos_in_dataset(root))


def image_from_mask(mask_path: Path) -> Path:
    name = mask_path.name
    if name.endswith("_endo_watershed_mask.png"):
        image_name = name.replace("_endo_watershed_mask.png", "_endo.png")
    elif name.endswith("_watershed.png"):
        image_name = name.replace("_watershed.png", ".png")
    else:
        raise ValueError(f"Unknown mask filename: {name}")
    return mask_path.parent / image_name


def list_samples(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs for train or val."""
    root = Path(root)
    val_set = val_videos_for_root(root)
    samples: list[tuple[Path, Path]] = []

    for mask_path in sorted(iter_mask_paths(root)):
        video = video_from_folder(mask_path.parent.name)
        if video is None:
            continue
        if split == "val":
            if video not in val_set:
                continue
        elif video in val_set:
            continue
        image_path = image_from_mask(mask_path)
        if image_path.exists():
            samples.append((image_path, mask_path))

    return samples


def split_counts(root: str | Path) -> dict[str, int]:
    root = Path(root)
    val_set = val_videos_for_root(root)
    counts = {"train": 0, "val": 0}
    videos: dict[str, set[str]] = {"train": set(), "val": set()}

    for mask_path in iter_mask_paths(root):
        video = video_from_folder(mask_path.parent.name)
        if video is None:
            continue
        key = "val" if video in val_set else "train"
        counts[key] += 1
        videos[key].add(video)

    counts["train_videos"] = len(videos["train"])
    counts["val_videos"] = len(videos["val"])
    counts["val_video_names"] = sorted(val_set)
    counts["sequence_folders"] = len({p.parent for p in iter_mask_paths(root)})
    return counts
