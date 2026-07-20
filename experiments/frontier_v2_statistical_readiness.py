"""Audit the development-only statistical evidence required before v2 lock."""

from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.familywise_policy_comparison import (
    DEFAULT_METHODS,
    METHOD_ASSUMPTIONS,
)
from experiments.frontier_v2_statistical_hash import (
    STATISTICAL_IMPLEMENTATION_FILES,
    statistical_implementation_sha256,
)
from experiments.frontier_v2_resolution_bound_check import (
    RESOLUTION_BOUND_FILE,
    audit_resolution_bound_check,
)
from experiments.frontier_v2_proof_certificate import (
    PROOF_CERTIFICATE_FILE,
    audit_proof_certificate,
)


PRIMARY_NULL_FILE = "global_null_betting_certified_10000_current.json"
PREDICTABLE_NULL_FILE = "global_null_predictable_10000_current.json"
BENTKUS_NULL_FILE = "global_null_bentkus_stitched_10000_current.json"
CLOSE_COMPARATOR_NULL_FILE = "global_null_close_comparators_10000_current.json"
METHOD_COMPARISON_FILE = "paired_method_comparison_mixed_300_complete_current.json"


def _load_current_payload(path: Path, *, design: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("design") != design:
        raise RuntimeError(f"unexpected statistical artifact design: {path}")
    if payload.get("statistical_implementation_files") != list(
        STATISTICAL_IMPLEMENTATION_FILES
    ):
        raise RuntimeError(f"statistical implementation file set changed: {path}")
    if (
        payload.get("statistical_implementation_sha256")
        != statistical_implementation_sha256()
    ):
        raise RuntimeError(f"statistical implementation hash changed: {path}")
    return payload


def _audit_null_payload(
    payload: dict,
    *,
    expected_methods: set[tuple[str, str]],
) -> list[dict]:
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        raise RuntimeError("null calibration summaries are missing")
    observed = {
        (str(summary.get("e_process_method")), str(summary.get("strategy")))
        for summary in summaries
    }
    if observed != expected_methods:
        raise RuntimeError("null calibration method coverage changed")
    audited = []
    for summary in summaries:
        if summary.get("scenario") != "global_null":
            raise RuntimeError("null calibration scenario changed")
        if int(summary.get("trials", -1)) < 10_000:
            raise RuntimeError("null calibration has fewer than 10,000 families")
        if not math.isclose(
            float(summary.get("familywise_alpha", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("null calibration familywise alpha changed")
        task_means = summary.get("task_means")
        if not isinstance(task_means, dict) or not task_means or any(
            float(mean) != 0.0 for mean in task_means.values()
        ):
            raise RuntimeError("global-null task family changed")
        interval = summary.get("familywise_false_accept_wilson_95_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not math.isfinite(float(value)) for value in interval)
            or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 0.05
        ):
            raise RuntimeError("global-null Wilson interval exceeds 0.05")
        rate = float(summary.get("familywise_false_accept_rate", math.nan))
        if not math.isfinite(rate) or not 0.0 <= rate <= 0.05:
            raise RuntimeError("global-null false-accept rate exceeds 0.05")
        audited.append(
            {
                "e_process_method": summary["e_process_method"],
                "strategy": summary["strategy"],
                "trials": int(summary["trials"]),
                "familywise_false_accept_rate": rate,
                "familywise_false_accept_wilson_95_ci": [
                    float(interval[0]),
                    float(interval[1]),
                ],
            }
        )
    return sorted(audited, key=lambda item: (item["e_process_method"], item["strategy"]))


def _audit_method_comparison(payload: dict) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("paired method-comparison summary is missing")
    if summary.get("scenario") != "mixed_effects":
        raise RuntimeError("paired method-comparison scenario changed")
    if int(summary.get("trials", -1)) < 300:
        raise RuntimeError("paired comparison has fewer than 300 trials")
    if summary.get("reference_method") != "betting_uniform":
        raise RuntimeError("paired comparison reference method changed")
    method_summaries = summary.get("method_summaries")
    if not isinstance(method_summaries, dict) or set(method_summaries) != set(
        DEFAULT_METHODS
    ):
        raise RuntimeError("paired comparison method coverage changed")
    for method, method_summary in method_summaries.items():
        if method_summary.get("assumption") != METHOD_ASSUMPTIONS[method]:
            raise RuntimeError(f"method assumption changed: {method}")
        rate = float(
            method_summary.get("familywise_false_accept_rate", math.nan)
        )
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise RuntimeError(f"invalid method false-accept rate: {method}")
        observations = float(method_summary.get("mean_total_observations", math.nan))
        if not math.isfinite(observations) or observations <= 0.0:
            raise RuntimeError(f"invalid method observation count: {method}")
    if int(summary.get("positive_task_count", 0)) <= 0:
        raise RuntimeError("paired comparison has no positive tasks")
    return {
        "trials": int(summary["trials"]),
        "method_count": len(method_summaries),
        "reference_method": summary["reference_method"],
        "performance_threshold_used": False,
    }


def _audit_bentkus_null(payload: dict) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Bentkus null summary is missing")
    if summary.get("scenario") != "global_null":
        raise RuntimeError("Bentkus null scenario changed")
    if int(summary.get("trials", -1)) < 10_000:
        raise RuntimeError("Bentkus null calibration has fewer than 10,000 families")
    if summary.get("reference_method") != "bentkus_stitched_racing":
        raise RuntimeError("Bentkus null reference method changed")
    task_means = summary.get("task_means")
    if not isinstance(task_means, dict) or not task_means or any(
        float(mean) != 0.0 for mean in task_means.values()
    ):
        raise RuntimeError("Bentkus global-null task family changed")
    method_summaries = summary.get("method_summaries")
    if not isinstance(method_summaries, dict) or set(method_summaries) != {
        "bentkus_stitched_racing"
    }:
        raise RuntimeError("Bentkus null method coverage changed")
    method_summary = method_summaries["bentkus_stitched_racing"]
    if (
        method_summary.get("assumption")
        != METHOD_ASSUMPTIONS["bentkus_stitched_racing"]
    ):
        raise RuntimeError("Bentkus null assumption changed")
    interval = method_summary.get("familywise_false_accept_wilson_95_ci")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(not math.isfinite(float(value)) for value in interval)
        or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 0.05
    ):
        raise RuntimeError("Bentkus global-null Wilson interval exceeds 0.05")
    rate = float(method_summary.get("familywise_false_accept_rate", math.nan))
    if not math.isfinite(rate) or not 0.0 <= rate <= 0.05:
        raise RuntimeError("Bentkus global-null false-accept rate exceeds 0.05")
    return {
        "method": "bentkus_stitched_racing",
        "assumption": METHOD_ASSUMPTIONS["bentkus_stitched_racing"],
        "trials": int(summary["trials"]),
        "familywise_false_accept_rate": rate,
        "familywise_false_accept_wilson_95_ci": [
            float(interval[0]),
            float(interval[1]),
        ],
    }


def _audit_close_comparator_null(payload: dict) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("close-comparator null summary is missing")
    if summary.get("scenario") != "global_null":
        raise RuntimeError("close-comparator null scenario changed")
    if int(summary.get("trials", -1)) < 10_000:
        raise RuntimeError(
            "close-comparator null calibration has fewer than 10,000 families"
        )
    if summary.get("reference_method") != "altt_agrapa_bonferroni":
        raise RuntimeError("close-comparator null reference method changed")
    task_means = summary.get("task_means")
    if not isinstance(task_means, dict) or not task_means or any(
        float(mean) != 0.0 for mean in task_means.values()
    ):
        raise RuntimeError("close-comparator global-null task family changed")
    expected = {"altt_agrapa_bonferroni", "eholm_agrapa"}
    method_summaries = summary.get("method_summaries")
    if not isinstance(method_summaries, dict) or set(method_summaries) != expected:
        raise RuntimeError("close-comparator null method coverage changed")

    audited = []
    for method in sorted(expected):
        method_summary = method_summaries[method]
        if method_summary.get("assumption") != METHOD_ASSUMPTIONS[method]:
            raise RuntimeError(f"close-comparator assumption changed: {method}")
        interval = method_summary.get("familywise_false_accept_wilson_95_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not math.isfinite(float(value)) for value in interval)
            or not 0.0 <= float(interval[0]) <= float(interval[1]) <= 0.05
        ):
            raise RuntimeError(
                f"close-comparator global-null Wilson interval exceeds 0.05: {method}"
            )
        rate = float(method_summary.get("familywise_false_accept_rate", math.nan))
        if not math.isfinite(rate) or not 0.0 <= rate <= 0.05:
            raise RuntimeError(
                f"close-comparator global-null false-accept rate exceeds 0.05: {method}"
            )
        audited.append(
            {
                "method": method,
                "assumption": METHOD_ASSUMPTIONS[method],
                "trials": int(summary["trials"]),
                "familywise_false_accept_rate": rate,
                "familywise_false_accept_wilson_95_ci": [
                    float(interval[0]),
                    float(interval[1]),
                ],
            }
        )
    return {"methods": audited, "performance_threshold_used": False}


def audit_statistical_readiness(root: Path) -> dict:
    primary = _load_current_payload(
        root / PRIMARY_NULL_FILE,
        design="riskshiftbench-v2-anytime-familywise-synthetic-calibration",
    )
    predictable = _load_current_payload(
        root / PREDICTABLE_NULL_FILE,
        design="riskshiftbench-v2-anytime-familywise-synthetic-calibration",
    )
    bentkus = _load_current_payload(
        root / BENTKUS_NULL_FILE,
        design="riskshiftbench-v2-paired-familywise-method-comparison",
    )
    close_comparators = _load_current_payload(
        root / CLOSE_COMPARATOR_NULL_FILE,
        design="riskshiftbench-v2-paired-familywise-method-comparison",
    )
    comparison = _load_current_payload(
        root / METHOD_COMPARISON_FILE,
        design="riskshiftbench-v2-paired-familywise-method-comparison",
    )
    resolution_bound = json.loads(
        (root / RESOLUTION_BOUND_FILE).read_text(encoding="utf-8")
    )
    proof_certificate = json.loads(
        (root / PROOF_CERTIFICATE_FILE).read_text(encoding="utf-8")
    )
    return {
        "statistical_implementation_sha256": statistical_implementation_sha256(),
        "primary_null": _audit_null_payload(
            primary,
            expected_methods={("betting_mixture", "certified")},
        ),
        "predictable_null": _audit_null_payload(
            predictable,
            expected_methods={
                ("predictable_betting", "uniform"),
                ("predictable_betting", "resolution"),
            },
        ),
        "bentkus_iid_null": _audit_bentkus_null(bentkus),
        "close_comparator_null": _audit_close_comparator_null(close_comparators),
        "paired_method_comparison": _audit_method_comparison(comparison),
        "nonbinding_resolution_bound": audit_resolution_bound_check(
            resolution_bound
        ),
        "proof_certificate": audit_proof_certificate(proof_certificate),
    }
