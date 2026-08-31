import gzip
import json

from riskshiftbench.experiments import verifier_preservation_study as study
from riskshiftbench.experiments.real_agent_validation import canonical_sha256


EXPECTED_SOURCE_SHA256 = "84a318bc2840036c0143ce0fbce3e99d003f3014d50c0b096c3ba7e8ad32bd1f"
EXPECTED_TARGET_MANIFEST_SHA256 = (
    "787766ed1fb124686df68bda0d72f509c2859b2c122e00b2e5fd76f7ab969ef1"
)
TARGET_MANIFEST_GZIP = study.TARGET_MANIFEST.with_suffix(".json.gz")


def _without(payload: dict, digest_key: str) -> dict:
    return {key: value for key, value in payload.items() if key != digest_key}


def test_a100_bundle_preserves_the_prospective_locks() -> None:
    config = json.loads(study.CONFIG.read_text(encoding="utf-8"))
    development_manifest = json.loads(
        study.DEVELOPMENT_MANIFEST.read_text(encoding="utf-8")
    )
    with gzip.open(study.DEVELOPMENT_RAW, "rt", encoding="utf-8") as handle:
        development_raw = json.load(handle)
    development = json.loads(study.DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    with gzip.open(TARGET_MANIFEST_GZIP, "rt", encoding="utf-8") as handle:
        target_manifest = json.load(handle)

    assert study.source_sha256() == EXPECTED_SOURCE_SHA256
    assert development_manifest["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert development_raw["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert development["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert target_manifest["source_sha256"] == EXPECTED_SOURCE_SHA256

    assert development_manifest["manifest_sha256"] == canonical_sha256(
        _without(development_manifest, "manifest_sha256")
    )
    assert development_raw["raw_sha256"] == canonical_sha256(
        _without(development_raw, "raw_sha256")
    )
    assert development["result_sha256"] == canonical_sha256(
        _without(development, "result_sha256")
    )
    assert target_manifest["manifest_sha256"] == canonical_sha256(
        _without(target_manifest, "manifest_sha256")
    )
    assert target_manifest["manifest_sha256"] == EXPECTED_TARGET_MANIFEST_SHA256
    assert target_manifest["development_result_sha256"] == development["result_sha256"]
    assert target_manifest["config_sha256"] == canonical_sha256(config)

    assert len(development_raw["completed_runs"]) == 42
    assert len(development_raw["trajectories"]) == 12_600
    assert len(target_manifest["pairs"]) == 36
    assert len(target_manifest["tasks"]) == 12_000
