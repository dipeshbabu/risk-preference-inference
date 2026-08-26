"""Fixed-look VDC variants with validity-matched component tests.

VDC-Full uses empirical-Bernstein mean and paired-drift p-values, exact
binomial downside and absolute-risk p-values, and Holm's fixed-look step-down
procedure. VDC-Absolute uses the same machinery but omits the relative-drift
null. This gives a validity-matched comparison of the extra drift condition.
VDC-A(split) uses a disjoint guidance sample to set verification counts, so the
fixed-look Holm guarantee applies conditionally on the guidance data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

from riskshiftbench.experiments.verifier_drift_control import _binomial_cdf


@dataclass(frozen=True)
class EfficientDecision:
    reason: str
    rejected_components: tuple[str, ...]
    component_pvalues: Mapping[str, float]


@dataclass(frozen=True)
class TaskIUTDecision:
    """Decision from a task-level intersection--union test."""

    reason: str
    task_pvalue: float
    adjusted_task_pvalue: float
    component_pvalues: Mapping[str, float]


def empirical_variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = fmean(values)
    return sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)


def empirical_bernstein_radius(
    values: Sequence[float], *, alpha: float, range_width: float
) -> float:
    if len(values) <= 1:
        raise ValueError("empirical-Bernstein bounds require at least two values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if range_width <= 0.0:
        raise ValueError("range width must be positive")
    log_term = math.log(2.0 / alpha)
    return math.sqrt(
        2.0 * empirical_variance(values) * log_term / len(values)
    ) + 7.0 * range_width * log_term / (3.0 * (len(values) - 1))


def _rejects_greater(
    values: Sequence[float], boundary: float, alpha: float, range_width: float
) -> bool:
    return (
        fmean(values)
        - empirical_bernstein_radius(
            values, alpha=alpha, range_width=range_width
        )
        > boundary
    )


def eb_pvalue_greater(
    values: Sequence[float], *, boundary: float, range_width: float
) -> float:
    """Invert a one-sided empirical-Bernstein lower bound."""

    if not _rejects_greater(values, boundary, 1.0 - 1e-12, range_width):
        return 1.0
    low, high = 1e-12, 1.0 - 1e-12
    for _ in range(60):
        middle = (low + high) / 2.0
        if _rejects_greater(values, boundary, middle, range_width):
            high = middle
        else:
            low = middle
    return high


def eb_pvalue_less(
    values: Sequence[float], *, boundary: float, range_width: float
) -> float:
    return eb_pvalue_greater(
        [-float(value) for value in values],
        boundary=-boundary,
        range_width=range_width,
    )


def binomial_pvalue_less(values: Sequence[bool], boundary: float) -> float:
    successes = sum(bool(value) for value in values)
    return _binomial_cdf(successes, len(values), boundary)


def holm_rejections(
    pvalues: Mapping[str, float], family_alpha: float
) -> set[str]:
    ordered = sorted(pvalues.items(), key=lambda row: (row[1], row[0]))
    rejected = set()
    total = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        if pvalue <= family_alpha / (total - index):
            rejected.add(name)
        else:
            break
    return rejected


def holm_adjusted_pvalues(
    pvalues: Mapping[str, float], family_alpha: float = 0.05
) -> dict[str, float]:
    """Return Holm-adjusted p-values for a fixed family.

    ``family_alpha`` is validated here for symmetry with ``holm_rejections``;
    adjusted values themselves do not depend on the chosen rejection level.
    """

    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")
    ordered = sorted(pvalues.items(), key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, float(pvalue) * (total - index)))
        adjusted[name] = running
    return adjusted


def task_iut_family_decisions(
    component_pvalues: Mapping[str, Mapping[str, float]],
    *,
    family_alpha: float,
) -> dict[str, TaskIUTDecision]:
    """Apply Holm across task-level intersection--union p-values.

    A task is eligible only if every component alternative holds.  Under the
    task-level union null, at least one component null is true, so the maximum
    component p-value is super-uniform.  Holm then controls the probability of
    deploying any task whose eligibility null is true.
    """

    if not component_pvalues:
        raise ValueError("at least one task is required")
    task_pvalues = {}
    for task, values in component_pvalues.items():
        if not values:
            raise ValueError(f"task has no component p-values: {task}")
        task_pvalues[task] = max(float(value) for value in values.values())
    adjusted = holm_adjusted_pvalues(task_pvalues, family_alpha)
    return {
        task: TaskIUTDecision(
            reason="promote" if adjusted[task] <= family_alpha else "unresolved",
            task_pvalue=task_pvalues[task],
            adjusted_task_pvalue=adjusted[task],
            component_pvalues=values,
        )
        for task, values in component_pvalues.items()
    }


def efficient_component_pvalues(
    effects: Sequence[float],
    candidate_failures: Mapping[float, Sequence[bool]],
    fallback_failures: Mapping[float, Sequence[bool]],
    *,
    mean_margin: float,
    material_loss_threshold: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_verifier_drift: float,
    include_relative_drift: bool = True,
    absolute_minimum_only: bool = False,
) -> dict[str, float]:
    pvalues = {
        "mean": eb_pvalue_greater(
            effects, boundary=mean_margin, range_width=2.0
        ),
        "downside": binomial_pvalue_less(
            [float(value) < -material_loss_threshold for value in effects],
            maximum_material_loss_probability,
        ),
    }
    thresholds = sorted(candidate_failures)
    for threshold in thresholds:
        candidate = candidate_failures[threshold]
        fallback = fallback_failures[threshold]
        if not absolute_minimum_only or threshold == thresholds[0]:
            pvalues[f"absolute:{threshold}"] = binomial_pvalue_less(
                candidate, maximum_candidate_high_confidence_failure_probability
            )
        if include_relative_drift:
            pvalues[f"drift:{threshold}"] = eb_pvalue_less(
                [float(c) - float(f) for c, f in zip(candidate, fallback)],
                boundary=maximum_verifier_drift,
                range_width=2.0,
            )
    return pvalues


def efficient_family_decisions(
    tasks: Mapping[
        str,
        tuple[
            Sequence[float],
            Mapping[float, Sequence[bool]],
            Mapping[float, Sequence[bool]],
        ],
    ],
    *,
    family_alpha: float,
    mean_margin: float,
    material_loss_threshold: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_verifier_drift: float,
    include_relative_drift: bool = True,
    absolute_minimum_only: bool = False,
    confidence_unresolved_reason: str = "verifier-evidence-unresolved",
) -> dict[str, EfficientDecision]:
    family_pvalues = {}
    by_task = {}
    for task_id, (effects, candidate, fallback) in tasks.items():
        pvalues = efficient_component_pvalues(
            effects,
            candidate,
            fallback,
            mean_margin=mean_margin,
            material_loss_threshold=material_loss_threshold,
            maximum_material_loss_probability=maximum_material_loss_probability,
            maximum_candidate_high_confidence_failure_probability=maximum_candidate_high_confidence_failure_probability,
            maximum_verifier_drift=maximum_verifier_drift,
            include_relative_drift=include_relative_drift,
            absolute_minimum_only=absolute_minimum_only,
        )
        by_task[task_id] = pvalues
        family_pvalues.update(
            {f"{task_id}::{component}": value for component, value in pvalues.items()}
        )
    rejected = holm_rejections(family_pvalues, family_alpha)
    decisions = {}
    for task_id, pvalues in by_task.items():
        names = {f"{task_id}::{component}" for component in pvalues}
        rejected_components = tuple(
            sorted(name.split("::", 1)[1] for name in names if name in rejected)
        )
        if names <= rejected:
            reason = "promote"
        elif {f"{task_id}::mean", f"{task_id}::downside"} <= rejected:
            reason = confidence_unresolved_reason
        else:
            reason = "unresolved"
        decisions[task_id] = EfficientDecision(
            reason=reason,
            rejected_components=rejected_components,
            component_pvalues=pvalues,
        )
    return decisions


def absolute_family_decisions(
    tasks: Mapping[
        str,
        tuple[
            Sequence[float],
            Mapping[float, Sequence[bool]],
            Mapping[float, Sequence[bool]],
        ],
    ],
    *,
    family_alpha: float,
    mean_margin: float,
    material_loss_threshold: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    absolute_minimum_only: bool = False,
    confidence_unresolved_reason: str = "verifier-evidence-unresolved",
) -> dict[str, EfficientDecision]:
    """Run the Holm-valid VDC-Absolute rule on a fixed task family."""

    return efficient_family_decisions(
        tasks,
        family_alpha=family_alpha,
        mean_margin=mean_margin,
        material_loss_threshold=material_loss_threshold,
        maximum_material_loss_probability=maximum_material_loss_probability,
        maximum_candidate_high_confidence_failure_probability=(
            maximum_candidate_high_confidence_failure_probability
        ),
        maximum_verifier_drift=0.0,
        include_relative_drift=False,
        absolute_minimum_only=absolute_minimum_only,
        confidence_unresolved_reason=confidence_unresolved_reason,
    )


def normalized_boundary_distance(
    effects: Sequence[float],
    candidate_failures: Mapping[float, Sequence[bool]],
    fallback_failures: Mapping[float, Sequence[bool]],
    *,
    mean_margin: float,
    maximum_material_loss_probability: float,
    maximum_candidate_high_confidence_failure_probability: float,
    maximum_verifier_drift: float,
) -> float:
    observations = len(effects)
    downside = fmean(float(value) < -0.5 for value in effects)
    candidate_rates = [fmean(values) for values in candidate_failures.values()]
    drift = [
        fmean(float(c) - float(f) for c, f in zip(candidate_failures[key], fallback_failures[key]))
        for key in candidate_failures
    ]
    margins = (
        fmean(effects) - mean_margin,
        maximum_material_loss_probability - downside,
        maximum_candidate_high_confidence_failure_probability - max(candidate_rates),
        maximum_verifier_drift - max(drift),
    )
    widths = (
        2.0 / math.sqrt(observations),
        1.0 / math.sqrt(observations),
        1.0 / math.sqrt(observations),
        2.0 / math.sqrt(observations),
    )
    return min(abs(margin) / width for margin, width in zip(margins, widths))
