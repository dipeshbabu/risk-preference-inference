from __future__ import annotations

import pytest

from riskshiftbench.experiments.verifier_preservation import (
    PreservationEvidence,
    certify_preservation_family,
    operational_drift_decomposition,
    preservation_component_pvalues,
)


def evidence(
    *,
    count: int = 5_000,
    candidate_confidence: float = 0.9,
    fallback_confidence: float = 0.9,
    candidate_failures: int = 100,
    fallback_failures: int = 100,
) -> PreservationEvidence:
    c_fail = tuple(index < candidate_failures for index in range(count))
    f_fail = tuple(index < fallback_failures for index in range(count))
    return PreservationEvidence(
        effects=tuple(0.5 for _ in range(count)),
        candidate_operational_failures={0.7: c_fail, 0.8: c_fail, 0.9: c_fail},
        fallback_operational_failures={0.7: f_fail, 0.8: f_fail, 0.9: f_fail},
        candidate_confidences=tuple(candidate_confidence for _ in range(count)),
        candidate_failures=c_fail,
        fallback_confidences=tuple(fallback_confidence for _ in range(count)),
        fallback_failures=f_fail,
        selective_thresholds={0.1: (0.7, 0.7), 0.3: (0.7, 0.7), 0.5: (0.7, 0.7)},
    )


KWARGS = {
    "mean_margin": 0.02,
    "material_loss_threshold": 0.5,
    "maximum_material_loss_probability": 0.10,
    "maximum_candidate_high_confidence_failure_probability": 0.15,
    "maximum_operational_drift": 0.05,
    "maximum_selective_drift": 0.05,
    "minimum_coverage_retention": 0.8,
    "minimum_selected_episodes": 50,
}


def test_clear_update_passes_all_preservation_components() -> None:
    pvalues = preservation_component_pvalues(
        evidence(), target="preserve", **KWARGS
    )
    assert max(pvalues.values()) < 0.05


def test_confidence_collapse_passes_operational_but_not_preserve() -> None:
    collapsed = evidence(candidate_confidence=0.2)
    operational = preservation_component_pvalues(
        collapsed, target="operational", **KWARGS
    )
    preserve = preservation_component_pvalues(
        collapsed, target="preserve", **KWARGS
    )
    assert max(operational.values()) < 0.05
    assert max(preserve.values()) == 1.0
    assert any(name.startswith("coverage:") for name in preserve)


def test_selective_risk_shift_is_unique_to_preservation_target() -> None:
    count = 5_000
    candidate_selected = tuple(index < 500 for index in range(count))
    fallback_selected = tuple(index < 2_500 for index in range(count))
    candidate_failures = tuple(index < 250 for index in range(count))
    fallback_failures = tuple(index < 250 for index in range(count))
    shifted = PreservationEvidence(
        effects=tuple(0.5 for _ in range(count)),
        candidate_operational_failures={
            threshold: tuple(s and f for s, f in zip(candidate_selected, candidate_failures))
            for threshold in (0.7, 0.8, 0.9)
        },
        fallback_operational_failures={
            threshold: tuple(s and f for s, f in zip(fallback_selected, fallback_failures))
            for threshold in (0.7, 0.8, 0.9)
        },
        candidate_confidences=tuple(0.9 if value else 0.2 for value in candidate_selected),
        candidate_failures=candidate_failures,
        fallback_confidences=tuple(0.9 if value else 0.2 for value in fallback_selected),
        fallback_failures=fallback_failures,
        selective_thresholds={0.1: (0.7, 0.7), 0.3: (0.7, 0.7), 0.5: (0.7, 0.7)},
    )
    operational = preservation_component_pvalues(
        shifted, target="operational", **KWARGS
    )
    preserve = preservation_component_pvalues(
        shifted, target="preserve", **KWARGS
    )
    assert max(operational.values()) < 0.05
    assert any(
        name.startswith("selective:") and value > 0.05
        for name, value in preserve.items()
    )


def test_task_iut_certifies_family_without_component_flattening() -> None:
    decisions = certify_preservation_family(
        {"a": evidence(), "b": evidence()},
        target="preserve",
        family_alpha=0.05,
        **KWARGS,
    )
    assert {value.reason for value in decisions.values()} == {"promote"}


def test_operational_decomposition_is_exact() -> None:
    result = operational_drift_decomposition(
        candidate_coverage=0.6,
        fallback_coverage=0.4,
        candidate_selective_risk=0.3,
        fallback_selective_risk=0.2,
    )
    assert result["operational_drift"] == pytest.approx(0.10)
    assert result["coverage_component"] == pytest.approx(0.04)
    assert result["selective_risk_component"] == pytest.approx(0.06)
