from __future__ import annotations

from experiments.frontier_v2_statistical_hash import (
    STATISTICAL_IMPLEMENTATION_FILES,
    statistical_implementation_sha256,
)


def test_statistical_implementation_hash_covers_the_complete_pipeline() -> None:
    assert set(STATISTICAL_IMPLEMENTATION_FILES) == {
        "experiments/frontier_v2_statistical_hash.py",
        "experiments/anytime_familywise_router.py",
        "experiments/anytime_familywise_calibration.py",
        "experiments/familywise_policy_baselines.py",
        "experiments/familywise_close_comparators.py",
        "experiments/familywise_policy_comparison.py",
    }
    digest = statistical_implementation_sha256()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
