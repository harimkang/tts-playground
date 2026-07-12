#!/usr/bin/env python3
"""Prefetch the three Qwen checkpoints into the mounted Hugging Face cache."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tts_playground.common.models import QWEN_REPOSITORIES

MINIMUM_FREE_BYTES = 20 * 1024**3
DEFAULT_CACHE_DIRECTORY = Path("/cache/huggingface")


class DiskUsage(Protocol):
    free: int


def prefetch_models(
    *,
    cache_directory: Path = DEFAULT_CACHE_DIRECTORY,
    snapshot_download: Callable[..., str],
    disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
) -> list[str]:
    """Download each configured Qwen repository after checking cache capacity."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    available_bytes = disk_usage(cache_directory).free
    if available_bytes < MINIMUM_FREE_BYTES:
        available_gib = available_bytes / 1024**3
        raise RuntimeError(
            f"At least 20 GiB free is required under {cache_directory}; "
            f"only {available_gib:.2f} GiB is available."
        )

    downloaded_paths: list[str] = []
    for repository in QWEN_REPOSITORIES.values():
        print(f"Downloading {repository}")
        path = snapshot_download(repo_id=repository, cache_dir=str(cache_directory))
        downloaded_paths.append(path)
        print(f"Downloaded {repository} to {path}")
    return downloaded_paths


def main() -> int:
    from huggingface_hub import snapshot_download

    prefetch_models(snapshot_download=snapshot_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
