"""Verifier Drift Detection and Control for candidate--fallback updates.

The implementation is outcome-free method code. It operates on bounded paired
pilot summaries and does not execute the prospective agent study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from statistics import fmean
from typing import Mapping, Sequence


class VDCAction(str, Enum):
    DEPLOY = "deploy"
    VERIFY_MORE = "verify-more"
    RECALIBRATE = "recalibrate"
    RETAIN = "retain"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class VDCPlan:
    confidence_thresholds: tuple[float, ...]
    family_tasks: int
    declared_looks: int
    family_alpha: float = 0.05
    mean_margin: float = 0.025
    material_loss_threshold: float = 0.25
    maximum_material_loss_probability: float = 0.10
    maximum_candidate_high_confidence_failure_probability: float = 0.10
    maximum_verifier_drift: float = 0.05

    def __post_init__(self) -> None:
        if not self.confidence_thresholds:
            raise ValueError("at least one confidence threshold is required")
        if tuple(sorted(set(self.confidence_thresholds))) != self.confidence_thresholds:
            raise ValueError("confidence thresholds must be unique and sorted")
        if not all(0.0 < value < 1.0 for value in self.confidence_thresholds):
            raise ValueError("confidence thresholds must lie in (0, 1)")
        if self.family_tasks <= 0 or self.declared_looks <= 0:
            raise ValueError("family_tasks and declared_looks must be positive")
        if not 0.0 < self.family_alpha < 1.0:
            raise ValueError("family_alpha must lie in (0, 1)")
        if not 0.0 < self.maximum_material_loss_probability < 1.0:
            raise ValueError("maximum material-loss probability must lie in (0, 1)")
        if not 0.0 < self.maximum_candidate_high_confidence_failure_probability < 1.0:
            raise ValueError(
                "maximum candidate high-confidence failure probability must lie in (0, 1)"
            )
        if self.maximum_verifier_drift < 0.0:
            raise ValueError("maximum verifier drift must be nonnegative")

    @property
    def tail_alpha(self) -> float:
        # Two directions for mean, downside, candidate risk, and drift.
        return self.family_alpha / (
            2
            * self.family_tasks
            * self.declared_looks
            * (2 * len(self.confidence_thresholds) + 2)
        )


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class VDCBounds:
    mean: Interval
    downside: Interval
    candidate_confidence_risk: Interval
    candidate_risk_by_threshold: Mapping[float, Interval]
    verifier_drift: Interval
    drift_by_threshold: Mapping[float, Interval]
    observations: int
    tail_alpha: float


@dataclass(frozen=True)
class VDCDecision:
    action: VDCAction
    reason: str
    bounds: VDCBounds


def hoeffding_radius(observations: int, alpha: float, range_width: float) -> float:
    """One-sided Hoeffding radius for a bounded sample mean."""

    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if range_width <= 0.0:
        raise ValueError("range_width must be positive")
    return range_width * math.sqrt(math.log(1.0 / alpha) / (2.0 * observations))


def _interval(values: Sequence[float], *, alpha: float, range_width: float) -> Interval:
    estimate = fmean(values)
    radius = hoeffding_radius(len(values), alpha, range_width)
    return Interval(estimate=estimate, lower=estimate - radius, upper=estimate + radius)


def _binomial_cdf(count: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return float(count >= trials)
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    log_total = -math.inf
    for index in range(count + 1):
        log_term = (
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + index * log_probability
            + (trials - index) * log_complement
        )
        if log_total == -math.inf:
            log_total = log_term
        elif log_term > log_total:
            log_total = log_term + math.log1p(math.exp(log_total - log_term))
        else:
            log_total = log_total + math.log1p(math.exp(log_term - log_total))
    return min(1.0, math.exp(log_total))


@lru_cache(maxsize=None)
def _binomial_bounds(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """One-sided Clopper--Pearson bounds at a fixed declared look."""

    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, successes / trials
        for _ in range(70):
            mid = (lo + hi) / 2.0
            upper_tail = 1.0 - _binomial_cdf(successes - 1, trials, mid)
            if upper_tail > alpha:
                hi = mid
            else:
                lo = mid
        lower = (lo + hi) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        lo, hi = successes / trials, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2.0
            if _binomial_cdf(successes, trials, mid) > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return lower, upper


def _bernoulli_interval(values: Sequence[bool | float], *, alpha: float) -> Interval:
    successes = sum(bool(value) for value in values)
    lower, upper = _binomial_bounds(successes, len(values), alpha)
    return Interval(estimate=successes / len(values), lower=lower, upper=upper)


def estimate_vdc_bounds(
    effects: Sequence[float],
    candidate_high_confidence_failures: Mapping[float, Sequence[bool]],
    fallback_high_confidence_failures: Mapping[float, Sequence[bool]],
    plan: VDCPlan,
) -> VDCBounds:
    """Return simultaneous utility, downside, and verifier-drift bounds."""

    observations = len(effects)
    if observations == 0:
        raise ValueError("at least one paired observation is required")
    if any(not -1.0 <= float(value) <= 1.0 for value in effects):
        raise ValueError("paired effects must lie in [-1, 1]")
    expected_thresholds = set(plan.confidence_thresholds)
    if set(candidate_high_confidence_failures) != expected_thresholds:
        raise ValueError("candidate threshold grid does not match the plan")
    if set(fallback_high_confidence_failures) != expected_thresholds:
        raise ValueError("fallback threshold grid does not match the plan")

    tail_alpha = plan.tail_alpha
    mean_interval = _interval(effects, alpha=tail_alpha, range_width=2.0)
    downside_values = [value < -plan.material_loss_threshold for value in effects]
    downside_interval = _bernoulli_interval(downside_values, alpha=tail_alpha)

    drift_intervals: dict[float, Interval] = {}
    candidate_risk_intervals: dict[float, Interval] = {}
    for threshold in plan.confidence_thresholds:
        candidate = candidate_high_confidence_failures[threshold]
        fallback = fallback_high_confidence_failures[threshold]
        if len(candidate) != observations or len(fallback) != observations:
            raise ValueError("every threshold must contain one value per paired episode")
        candidate_interval = _bernoulli_interval(candidate, alpha=tail_alpha)
        fallback_interval = _bernoulli_interval(fallback, alpha=tail_alpha)
        candidate_risk_intervals[threshold] = candidate_interval
        drift_intervals[threshold] = Interval(
            estimate=candidate_interval.estimate - fallback_interval.estimate,
            lower=candidate_interval.lower - fallback_interval.upper,
            upper=candidate_interval.upper - fallback_interval.lower,
        )

    drift = Interval(
        estimate=max(row.estimate for row in drift_intervals.values()),
        lower=max(row.lower for row in drift_intervals.values()),
        upper=max(row.upper for row in drift_intervals.values()),
    )
    candidate_risk = Interval(
        estimate=max(row.estimate for row in candidate_risk_intervals.values()),
        lower=max(row.lower for row in candidate_risk_intervals.values()),
        upper=max(row.upper for row in candidate_risk_intervals.values()),
    )
    return VDCBounds(
        mean=mean_interval,
        downside=downside_interval,
        candidate_confidence_risk=candidate_risk,
        candidate_risk_by_threshold=candidate_risk_intervals,
        verifier_drift=drift,
        drift_by_threshold=drift_intervals,
        observations=observations,
        tail_alpha=tail_alpha,
    )


def decide_vdc(
    effects: Sequence[float],
    candidate_high_confidence_failures: Mapping[float, Sequence[bool]],
    fallback_high_confidence_failures: Mapping[float, Sequence[bool]],
    plan: VDCPlan,
    *,
    at_task_cap: bool,
    recalibration_indicated: bool = False,
    legacy_recalibration_label: bool = False,
) -> VDCDecision:
    """Choose a fallback-preserving route from pilot evidence.

    Unresolved verifier evidence requests more verification by default.
    Recalibration is a distinct action and requires an externally established,
    development-only repair indication.  The legacy flag reproduces studies
    that historically called every confidence-limited route "recalibrate".
    """

    bounds = estimate_vdc_bounds(
        effects,
        candidate_high_confidence_failures,
        fallback_high_confidence_failures,
        plan,
    )
    if bounds.mean.upper <= plan.mean_margin:
        return VDCDecision(VDCAction.RETAIN, "retain-utility", bounds)
    if bounds.downside.lower >= plan.maximum_material_loss_probability:
        return VDCDecision(VDCAction.RETAIN, "retain-downside", bounds)
    if (
        bounds.candidate_confidence_risk.lower
        >= plan.maximum_candidate_high_confidence_failure_probability
    ):
        return VDCDecision(VDCAction.RETAIN, "retain-confidence-risk", bounds)
    if bounds.verifier_drift.lower >= plan.maximum_verifier_drift:
        return VDCDecision(VDCAction.RETAIN, "retain-verifier-drift", bounds)

    utility_passes = bounds.mean.lower > plan.mean_margin
    downside_passes = (
        bounds.downside.upper < plan.maximum_material_loss_probability
    )
    candidate_risk_passes = (
        bounds.candidate_confidence_risk.upper
        < plan.maximum_candidate_high_confidence_failure_probability
    )
    drift_passes = bounds.verifier_drift.upper < plan.maximum_verifier_drift
    if utility_passes and downside_passes and candidate_risk_passes and drift_passes:
        return VDCDecision(VDCAction.DEPLOY, "promote", bounds)
    if at_task_cap and utility_passes and downside_passes:
        if recalibration_indicated:
            return VDCDecision(
                VDCAction.RECALIBRATE, "recalibration-indicated", bounds
            )
        if legacy_recalibration_label:
            return VDCDecision(
                VDCAction.RECALIBRATE, "recalibrate-verifier", bounds
            )
        return VDCDecision(
            VDCAction.VERIFY_MORE, "verifier-evidence-unresolved", bounds
        )
    return VDCDecision(VDCAction.UNRESOLVED, "unresolved", bounds)


def sufficient_resolution_samples(
    plan: VDCPlan,
    *,
    mean_margin: float,
    downside_margin: float,
    candidate_risk_margin: float,
    drift_margin: float,
) -> int:
    """Sufficient all-Hoeffding size for interval-certified eligibility.

    The factor eight accounts for both sampling deviation and the reported
    confidence radius.  For drift, the conservative construction also
    subtracts separate candidate and fallback endpoints.
    """

    if min(mean_margin, downside_margin, candidate_risk_margin, drift_margin) <= 0.0:
        raise ValueError("all resolution margins must be positive")
    return max(
        sufficient_interval_resolution_samples(
            tail_alpha=plan.tail_alpha, margin=mean_margin, range_width=2.0
        ),
        sufficient_interval_resolution_samples(
            tail_alpha=plan.tail_alpha, margin=downside_margin, range_width=1.0
        ),
        sufficient_interval_resolution_samples(
            tail_alpha=plan.tail_alpha,
            margin=candidate_risk_margin,
            range_width=1.0,
        ),
        sufficient_interval_resolution_samples(
            tail_alpha=plan.tail_alpha, margin=drift_margin, range_width=2.0
        ),
    )


def sufficient_interval_resolution_samples(
    *, tail_alpha: float, margin: float, range_width: float
) -> int:
    """All-Hoeffding size for one confidence endpoint to clear a margin."""

    if not 0.0 < tail_alpha < 1.0:
        raise ValueError("tail_alpha must lie in (0, 1)")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    if range_width <= 0.0:
        raise ValueError("range_width must be positive")
    return math.ceil(
        2.0 * range_width**2 * math.log(1.0 / tail_alpha) / margin**2
    )


def sufficient_drift_detection_samples(
    *, threshold_count: int, miss_probability: float, drift_separation: float
) -> int:
    """Sufficient size for drift detection using separate rate endpoints."""

    if threshold_count <= 0:
        raise ValueError("threshold_count must be positive")
    if not 0.0 < miss_probability < 1.0:
        raise ValueError("miss_probability must lie in (0, 1)")
    if drift_separation <= 0.0:
        raise ValueError("drift_separation must be positive")
    return math.ceil(
        8.0
        * math.log(2.0 * threshold_count / miss_probability)
        / drift_separation**2
    )
