from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.frontier_v2_external_design import canonical_sha256
from experiments.frontier_v2_protocol_lock import (
    STATISTICAL_ARTIFACT_FILES,
    byte_sha256,
    finalize_registration,
    validate_protocol,
)
from experiments.frontier_v2_statistical_readiness import (
    BENTKUS_NULL_FILE,
    CLOSE_COMPARATOR_NULL_FILE,
)


def _draft(tmp_path: Path) -> Path:
    design = {
        "protocol_id": "test",
        "source_manifest": [],
        "statistical_artifacts": [],
        "baseline_manifests": [],
        "router_lock": {"path": "missing", "byte_sha256": "missing"},
    }
    design_path = tmp_path / "design.json"
    design_path.write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    wrapper = {
        "status": "awaiting_external_registration",
        "locked_design_path": design_path.as_posix(),
        "locked_design_byte_sha256": byte_sha256(design_path),
        "locked_design_canonical_sha256": canonical_sha256(design),
        "registration": {"provider": None, "url": None, "registered_at": None},
    }
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return draft


def test_finalize_registration_binds_design_hash_and_metadata(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    output = tmp_path / "registered.json"
    wrapper = finalize_registration(
        draft,
        output,
        provider="OSF",
        url="https://doi.org/10.17605/OSF.IO/EXAMPLE",
        registered_at="2026-07-20T01:00:00Z",
    )
    assert wrapper["status"] == "externally_registered_locked"
    assert wrapper["registration"]["registered_design_sha256"] == wrapper[
        "locked_design_byte_sha256"
    ]
    assert output.is_file()


def test_finalize_registration_requires_timezone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone"):
        finalize_registration(
            _draft(tmp_path),
            tmp_path / "registered.json",
            provider="OSF",
            url="https://osf.io/example",
            registered_at="2026-07-20T01:00:00",
        )


def test_validate_protocol_refuses_unregistered_draft_before_artifact_reads(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="publicly registered"):
        validate_protocol(_draft(tmp_path), require_registration=True)


def test_protocol_lock_binds_close_comparator_calibrations() -> None:
    assert BENTKUS_NULL_FILE in STATISTICAL_ARTIFACT_FILES
    assert CLOSE_COMPARATOR_NULL_FILE in STATISTICAL_ARTIFACT_FILES
