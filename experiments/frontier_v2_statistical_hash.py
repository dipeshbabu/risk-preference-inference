"""Canonical source hash for the v2 statistical calibration artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


STATISTICAL_IMPLEMENTATION_FILES = (
    "experiments/frontier_v2_statistical_hash.py",
    "experiments/anytime_familywise_router.py",
    "experiments/anytime_familywise_calibration.py",
    "experiments/familywise_policy_baselines.py",
    "experiments/familywise_close_comparators.py",
    "experiments/familywise_policy_comparison.py",
)


def statistical_implementation_sha256(
    repository_root: Path | None = None,
) -> str:
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else repository_root
    )
    digest = hashlib.sha256()
    for relative in STATISTICAL_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"statistical implementation file is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()
