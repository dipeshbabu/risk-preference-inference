"""Prospective task-IUT certification of verifier preservation.

The module separates three population targets:

* Absolute: utility, downside, and absolute high-confidence failure;
* Operational: Absolute plus fixed-threshold operational drift;
* Preserve: Operational plus matched-coverage selective drift and retained
  confidence coverage.

All thresholds must be fixed without using verification outcomes.  The caller
is responsible for enforcing that evidence split.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from riskshiftbench.experiments.vdc_efficient import (
    TaskIUTDecision,
    binomial_pvalue_less,
    eb_pvalue_greater,
    eb_pvalue_less,
    task_iut_family_decisions,
)
from riskshiftbench.experiments.verifier_drift_control import _binomial_bounds


CERTIFICATION_TARGETS = ("absolute", "operational", "preserve")


@dataclass(frozen=True)
class PreservationEvidence:
    effects: tuple[float, ...]
    candidate_operational_failures: Mapping[float, tuple[bool, ...]]
    fallback_operational_failures: Mapping[float, tuple[bool, ...]]
    candidate_confidences: tuple[float, ...]
    candidate_failures: tuple[bool, ...]
    fallback_confidences: tuple[float, ...]
    fallback_failures: tuple[bool, ...]
    selective_thresholds: Mapping[float, tuple[float, float]]

    def __post_init__(self) -> None:
        count = len(self.effects)
        if count < 2:
            raise ValueError("preservation evidence requires at least two pairs")
        aligned = (
            self.candidate_confidences,
            self.candidate_failures,
            self.fallback_confidences,
            self.fallback_failures,
        )
        if any(len(values) != count for values in aligned):
            raise ValueError("confidence and failure sequences must align with effects")
        if set(self.candidate_operational_failures) != set(
            self.fallback_operational_failures
        ):
            raise ValueError("operational threshold grids must match")
        if not self.candidate_operational_failures:
            raise ValueError("at least one operational threshold is required")
        for values in (
            *self.candidate_operational_failures.values(),
            *self.fallback_operational_failures.values(),
        ):
            if len(values) != count:
                raise ValueError("operational indicators must align with effects")
        coverages = tuple(self.selective_thresholds)
        if coverages != tuple(sorted(set(coverages))) or not coverages:
            raise ValueError("selective coverage grid must be nonempty and sorted")


def _rejects_selective_drift(
    candidate_failures: Sequence[bool],
    fallback_failures: Sequence[bool],
    *,
    boundary: float,
    alpha: float,
) -> bool:
    candidate_count = sum(bool(value) for value in candidate_failures)
    fallback_count = sum(bool(value) for value in fallback_failures)
    _, candidate_upper = _binomial_bounds(
        candidate_count, len(candidate_failures), alpha / 2.0
    )
    fallback_lower, _ = _binomial_bounds(
        fallback_count, len(fallback_failures), alpha / 2.0
    )
    return candidate_upper - fallback_lower < boundary


def selective_drift_pvalue_less(
    candidate_failures: Sequence[bool],
    fallback_failures: Sequence[bool],
    *,
    boundary: float,
    minimum_selected: int,
) -> float:
    """Test that candidate selective risk minus fallback risk is below a limit.

    Candidate and fallback selected sets may have different sizes.  Conditional
    Clopper--Pearson bounds are combined with a two-tail union bound.  Returning
    one for a small selected set prevents a coverage collapse from certifying
    selective preservation through an undefined or unstable conditional risk.
    """

    if len(candidate_failures) < minimum_selected or len(fallback_failures) < minimum_selected:
        return 1.0
    if not _rejects_selective_drift(
        candidate_failures,
        fallback_failures,
        boundary=boundary,
        alpha=1.0 - 1e-12,
    ):
        return 1.0
    low, high = 1e-12, 1.0 - 1e-12
    for _ in range(60):
        middle = (low + high) / 2.0
        if _rejects_selective_drift(
            candidate_failures,
            fallback_failures,
            boundary=boundary,
            alpha=middle,
        ):
            high = middle
        else:
            low = middle
    return high


def preservation_component_pvalues(
    evidence: PreservationEvidence,
    *,
    target: str,
    mean_margin: float,
    material_loss_threshold: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_operational_drift: float,
    maximum_selective_drift: float,
    minimum_coverage_retention: float,
    minimum_selected_episodes: int,
) -> dict[str, float]:
    if target not in CERTIFICATION_TARGETS:
        raise ValueError(f"unknown certification target: {target}")
    if not 0.0 < minimum_coverage_retention <= 1.0:
        raise ValueError("coverage retention must lie in (0, 1]")
    pvalues = {
        "mean": eb_pvalue_greater(
            evidence.effects, boundary=mean_margin, range_width=2.0
        ),
        "downside": binomial_pvalue_less(
            [float(value) < -material_loss_threshold for value in evidence.effects],
            maximum_material_loss_probability,
        ),
    }
    operational_thresholds = sorted(evidence.candidate_operational_failures)
    minimum_threshold = operational_thresholds[0]
    pvalues[f"absolute:{minimum_threshold}"] = binomial_pvalue_less(
        evidence.candidate_operational_failures[minimum_threshold],
        maximum_candidate_high_confidence_failure_probability,
    )
    if target in {"operational", "preserve"}:
        for threshold in operational_thresholds:
            candidate = evidence.candidate_operational_failures[threshold]
            fallback = evidence.fallback_operational_failures[threshold]
            pvalues[f"operational:{threshold}"] = eb_pvalue_less(
                [float(c) - float(f) for c, f in zip(candidate, fallback)],
                boundary=maximum_operational_drift,
                range_width=2.0,
            )
    if target == "preserve":
        for coverage, (candidate_threshold, fallback_threshold) in sorted(
            evidence.selective_thresholds.items()
        ):
            candidate_selected = [
                confidence >= candidate_threshold
                for confidence in evidence.candidate_confidences
            ]
            fallback_selected = [
                confidence >= fallback_threshold
                for confidence in evidence.fallback_confidences
            ]
            candidate_failures = [
                failure
                for failure, selected in zip(
                    evidence.candidate_failures, candidate_selected
                )
                if selected
            ]
            fallback_failures = [
                failure
                for failure, selected in zip(
                    evidence.fallback_failures, fallback_selected
                )
                if selected
            ]
            pvalues[f"selective:{coverage}"] = selective_drift_pvalue_less(
                candidate_failures,
                fallback_failures,
                boundary=maximum_selective_drift,
                minimum_selected=minimum_selected_episodes,
            )
            pvalues[f"coverage:{coverage}"] = eb_pvalue_greater(
                [
                    float(candidate_value)
                    - minimum_coverage_retention * float(fallback_value)
                    for candidate_value, fallback_value in zip(
                        candidate_selected, fallback_selected
                    )
                ],
                boundary=0.0,
                range_width=1.0 + minimum_coverage_retention,
            )
    return pvalues


def certify_preservation_family(
    tasks: Mapping[str, PreservationEvidence],
    *,
    target: str,
    family_alpha: float,
    mean_margin: float,
    material_loss_threshold: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_operational_drift: float,
    maximum_selective_drift: float,
    minimum_coverage_retention: float,
    minimum_selected_episodes: int,
) -> dict[str, TaskIUTDecision]:
    components = {
        task: preservation_component_pvalues(
            evidence,
            target=target,
            mean_margin=mean_margin,
            material_loss_threshold=material_loss_threshold,
            maximum_material_loss_probability=maximum_material_loss_probability,
            maximum_candidate_high_confidence_failure_probability=(
                maximum_candidate_high_confidence_failure_probability
            ),
            maximum_operational_drift=maximum_operational_drift,
            maximum_selective_drift=maximum_selective_drift,
            minimum_coverage_retention=minimum_coverage_retention,
            minimum_selected_episodes=minimum_selected_episodes,
        )
        for task, evidence in tasks.items()
    }
    return task_iut_family_decisions(components, family_alpha=family_alpha)


def operational_drift_decomposition(
    *, candidate_coverage: float, fallback_coverage: float,
    candidate_selective_risk: float, fallback_selective_risk: float
) -> dict[str, float]:
    coverage_component = (
        candidate_coverage - fallback_coverage
    ) * fallback_selective_risk
    selective_component = candidate_coverage * (
        candidate_selective_risk - fallback_selective_risk
    )
    total = (
        candidate_coverage * candidate_selective_risk
        - fallback_coverage * fallback_selective_risk
    )
    if abs(total - coverage_component - selective_component) > 1e-12:
        raise RuntimeError("operational drift decomposition failed")
    return {
        "operational_drift": total,
        "coverage_component": coverage_component,
        "selective_risk_component": selective_component,
    }


def point_preservation_metrics(evidence: PreservationEvidence) -> dict:
    operational_profile = {}
    for threshold in sorted(evidence.candidate_operational_failures):
        operational_profile[str(threshold)] = (
            fmean(evidence.candidate_operational_failures[threshold])
            - fmean(evidence.fallback_operational_failures[threshold])
        )
    selective_profile = {}
    coverage_profile = {}
    decomposition = {}
    for coverage, (candidate_threshold, fallback_threshold) in sorted(
        evidence.selective_thresholds.items()
    ):
        candidate_selected = [
            value >= candidate_threshold for value in evidence.candidate_confidences
        ]
        fallback_selected = [
            value >= fallback_threshold for value in evidence.fallback_confidences
        ]
        candidate_coverage = fmean(candidate_selected)
        fallback_coverage = fmean(fallback_selected)
        candidate_failures = [
            failure
            for failure, selected in zip(evidence.candidate_failures, candidate_selected)
            if selected
        ]
        fallback_failures = [
            failure
            for failure, selected in zip(evidence.fallback_failures, fallback_selected)
            if selected
        ]
        candidate_risk = fmean(candidate_failures) if candidate_failures else 1.0
        fallback_risk = fmean(fallback_failures) if fallback_failures else 1.0
        selective_profile[str(coverage)] = candidate_risk - fallback_risk
        coverage_profile[str(coverage)] = {
            "candidate": candidate_coverage,
            "fallback": fallback_coverage,
            "retained_ratio": (
                candidate_coverage / fallback_coverage
                if fallback_coverage > 0.0
                else 1.0
            ),
        }
        decomposition[str(coverage)] = operational_drift_decomposition(
            candidate_coverage=candidate_coverage,
            fallback_coverage=fallback_coverage,
            candidate_selective_risk=candidate_risk,
            fallback_selective_risk=fallback_risk,
        )
    return {
        "mean_effect": fmean(evidence.effects),
        "material_loss_probability": fmean(
            float(value) < -0.5 for value in evidence.effects
        ),
        "candidate_high_confidence_failure": max(
            fmean(values)
            for values in evidence.candidate_operational_failures.values()
        ),
        "operational_drift_profile": operational_profile,
        "operational_drift": max(operational_profile.values()),
        "selective_drift_profile": selective_profile,
        "selective_drift": max(selective_profile.values()),
        "coverage_profile": coverage_profile,
        "decomposition": decomposition,
    }
