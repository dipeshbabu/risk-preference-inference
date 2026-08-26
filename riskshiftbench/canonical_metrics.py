"""Canonical population quantities for candidate--fallback verification.

This module is the single source for the score effect, material downside,
absolute high-confidence-failure risk, and operational verifier drift used by
new RiskShiftBench studies.  Historical locked implementations remain
immutable; invariant tests compare their inputs and outputs with these
functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CanonicalUpdateMetrics:
    mean_effect: float
    material_loss_probability: float
    candidate_high_confidence_failure: float
    verifier_drift_max: float
    drift_profile: tuple[float, ...]


@dataclass(frozen=True)
class SelectiveDriftMetrics:
    coverages: tuple[float, ...]
    candidate_selective_risk: tuple[float, ...]
    fallback_selective_risk: tuple[float, ...]
    drift_profile: tuple[float, ...]
    verifier_drift_max: float


def paired_effects(
    fallback_scores: Sequence[float], candidate_scores: Sequence[float]
) -> tuple[float, ...]:
    if len(fallback_scores) != len(candidate_scores) or not fallback_scores:
        raise ValueError("score sequences must be nonempty and paired")
    effects = tuple(float(c) - float(f) for f, c in zip(fallback_scores, candidate_scores))
    if any(not -1.0 <= value <= 1.0 for value in effects):
        raise ValueError("paired effects must lie in [-1, 1]")
    return effects


def material_loss_indicators(
    effects: Sequence[float], material_loss_threshold: float
) -> tuple[bool, ...]:
    if material_loss_threshold < 0.0:
        raise ValueError("material-loss threshold must be nonnegative")
    return tuple(float(value) < -material_loss_threshold for value in effects)


def threshold_profile(
    candidate_failures: Mapping[float, Sequence[bool]],
    fallback_failures: Mapping[float, Sequence[bool]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if set(candidate_failures) != set(fallback_failures) or not candidate_failures:
        raise ValueError("candidate and fallback threshold grids must match")
    thresholds = tuple(sorted(candidate_failures))
    candidate_rates = []
    fallback_rates = []
    for threshold in thresholds:
        candidate = candidate_failures[threshold]
        fallback = fallback_failures[threshold]
        if len(candidate) != len(fallback) or not candidate:
            raise ValueError("threshold indicators must be nonempty and paired")
        candidate_rates.append(fmean(bool(value) for value in candidate))
        fallback_rates.append(fmean(bool(value) for value in fallback))
    drift = tuple(c - f for c, f in zip(candidate_rates, fallback_rates))
    return tuple(candidate_rates), tuple(fallback_rates), drift


def selective_risk_at_coverage(
    confidences: Sequence[float], failures: Sequence[bool], coverage: float
) -> float:
    """Failure risk among the most confident observations at fixed coverage.

    Confidence ties use original observation order, which makes the rule
    deterministic and keeps candidate and fallback selection independent.
    """

    if len(confidences) != len(failures) or not confidences:
        raise ValueError("confidence and failure sequences must be nonempty and aligned")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must lie in (0, 1]")
    count = max(1, min(len(confidences), math.ceil(coverage * len(confidences))))
    ranked = sorted(
        range(len(confidences)), key=lambda index: (-float(confidences[index]), index)
    )
    return fmean(bool(failures[index]) for index in ranked[:count])


def selective_verifier_drift(
    candidate_confidences: Sequence[float],
    candidate_failures: Sequence[bool],
    fallback_confidences: Sequence[float],
    fallback_failures: Sequence[bool],
    *,
    coverages: Sequence[float],
) -> SelectiveDriftMetrics:
    """Compare selective failure risk at development-fixed coverage levels."""

    if not coverages:
        raise ValueError("at least one coverage is required")
    grid = tuple(float(value) for value in coverages)
    if grid != tuple(sorted(set(grid))):
        raise ValueError("coverages must be unique and sorted")
    candidate = tuple(
        selective_risk_at_coverage(candidate_confidences, candidate_failures, value)
        for value in grid
    )
    fallback = tuple(
        selective_risk_at_coverage(fallback_confidences, fallback_failures, value)
        for value in grid
    )
    drift = tuple(c - f for c, f in zip(candidate, fallback))
    return SelectiveDriftMetrics(
        coverages=grid,
        candidate_selective_risk=candidate,
        fallback_selective_risk=fallback,
        drift_profile=drift,
        verifier_drift_max=max(drift),
    )


def canonical_update_metrics(
    effects: Sequence[float],
    candidate_failures: Mapping[float, Sequence[bool]],
    fallback_failures: Mapping[float, Sequence[bool]],
    *,
    material_loss_threshold: float,
) -> CanonicalUpdateMetrics:
    if not effects:
        raise ValueError("at least one paired effect is required")
    candidate_rates, _, drift = threshold_profile(
        candidate_failures, fallback_failures
    )
    if any(len(values) != len(effects) for values in candidate_failures.values()):
        raise ValueError("confidence-failure indicators must align with effects")
    downside = material_loss_indicators(effects, material_loss_threshold)
    return CanonicalUpdateMetrics(
        mean_effect=fmean(float(value) for value in effects),
        material_loss_probability=fmean(downside),
        candidate_high_confidence_failure=max(candidate_rates),
        verifier_drift_max=max(drift),
        drift_profile=drift,
    )


def is_eligible(
    metrics: CanonicalUpdateMetrics,
    *,
    mean_margin: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_verifier_drift: float,
) -> bool:
    return (
        metrics.mean_effect > mean_margin
        and metrics.material_loss_probability < maximum_material_loss_probability
        and metrics.candidate_high_confidence_failure
        < maximum_candidate_high_confidence_failure_probability
        and metrics.verifier_drift_max < maximum_verifier_drift
    )
