from __future__ import annotations

import json
from collections import Counter

import riskshiftbench.experiments.prospective_full_vs_absolute as base
from riskshiftbench.experiments.verifier_preservation_study import (
    CONFIG,
    pairs,
    source_sha256,
    validate_config,
)


def test_maintrack_design_has_36_balanced_updates_and_two_model_families() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    rows = pairs(config)
    assert len(rows) == 36
    assert Counter(row["cohort"] for row in rows) == {"natural": 18, "stress": 18}
    assert len({row["model_id"] for row in rows}) == 2
    assert len({row["design"] for row in rows}) == 6


def test_maintrack_coding_tasks_use_the_ten_step_slice() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tasks = base._generate_split(
        config, split="test", count=2, seed=991, shift="low"
    )
    coding = [row for row in tasks if row["family"] == "coding"]
    assert coding
    assert all(row["long_horizon"] for row in coding)
    assert config["maximum_steps"]["coding"] == 10


def test_maintrack_lock_hash_covers_the_new_method_source() -> None:
    digest = source_sha256()
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
