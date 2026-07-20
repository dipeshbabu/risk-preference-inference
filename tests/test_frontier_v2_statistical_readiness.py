from __future__ import annotations

import pytest

from experiments.frontier_v2_statistical_readiness import (
    _audit_bentkus_null,
    _audit_close_comparator_null,
    _audit_null_payload,
)


def _summary() -> dict:
    return {
        "e_process_method": "betting_mixture",
        "strategy": "certified",
        "scenario": "global_null",
        "trials": 10_000,
        "familywise_alpha": 0.05,
        "task_means": {"null": 0.0},
        "familywise_false_accept_rate": 0.01,
        "familywise_false_accept_wilson_95_ci": [0.008, 0.012],
    }


def test_null_audit_requires_current_method_and_conservative_interval() -> None:
    audited = _audit_null_payload(
        {"summaries": [_summary()]},
        expected_methods={("betting_mixture", "certified")},
    )
    assert audited[0]["trials"] == 10_000


def test_null_audit_rejects_interval_above_familywise_level() -> None:
    summary = _summary()
    summary["familywise_false_accept_wilson_95_ci"] = [0.04, 0.06]
    with pytest.raises(RuntimeError, match="exceeds"):
        _audit_null_payload(
            {"summaries": [summary]},
            expected_methods={("betting_mixture", "certified")},
        )


def _bentkus_payload() -> dict:
    return {
        "summary": {
            "scenario": "global_null",
            "trials": 10_000,
            "reference_method": "bentkus_stitched_racing",
            "task_means": {"null": 0.0},
            "method_summaries": {
                "bentkus_stitched_racing": {
                    "assumption": "bounded IID task streams with anytime stopping",
                    "familywise_false_accept_rate": 0.01,
                    "familywise_false_accept_wilson_95_ci": [0.008, 0.012],
                }
            },
        }
    }


def test_bentkus_null_audit_requires_iid_label_and_ten_thousand_families() -> None:
    audited = _audit_bentkus_null(_bentkus_payload())
    assert audited["trials"] == 10_000
    assert "IID" in audited["assumption"]


def test_bentkus_null_audit_rejects_weak_monte_carlo_bound() -> None:
    payload = _bentkus_payload()
    payload["summary"]["method_summaries"]["bentkus_stitched_racing"][
        "familywise_false_accept_wilson_95_ci"
    ] = [0.04, 0.06]
    with pytest.raises(RuntimeError, match="exceeds"):
        _audit_bentkus_null(payload)


def _close_comparator_payload() -> dict:
    summaries = {}
    for method, assumption in {
        "altt_agrapa_bonferroni": (
            "bounded conditional-mean task streams; anytime Bonferroni FWER"
        ),
        "eholm_agrapa": (
            "bounded conditional-mean task streams; always-valid e-Holm strong FWER"
        ),
        "nscore11_bonferroni": (
            "bounded conditional-mean task streams; N-SCORE multiplier; "
            "anytime Bonferroni FWER"
        ),
    }.items():
        summaries[method] = {
            "assumption": assumption,
            "familywise_false_accept_rate": 0.01,
            "familywise_false_accept_wilson_95_ci": [0.008, 0.012],
        }
    return {
        "summary": {
            "scenario": "global_null",
            "trials": 10_000,
            "reference_method": "altt_agrapa_bonferroni",
            "task_means": {"null": 0.0},
            "method_summaries": summaries,
        }
    }


def test_close_comparator_null_audit_requires_all_close_methods() -> None:
    audited = _audit_close_comparator_null(_close_comparator_payload())
    assert len(audited["methods"]) == 3
    assert audited["performance_threshold_used"] is False


def test_close_comparator_null_audit_rejects_weak_monte_carlo_bound() -> None:
    payload = _close_comparator_payload()
    payload["summary"]["method_summaries"]["eholm_agrapa"][
        "familywise_false_accept_wilson_95_ci"
    ] = [0.04, 0.06]
    with pytest.raises(RuntimeError, match="exceeds"):
        _audit_close_comparator_null(payload)
