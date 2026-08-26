import math

import pytest

from riskshiftbench.experiments.verifier_drift_control import (
    VDCAction,
    VDCPlan,
    decide_vdc,
    estimate_vdc_bounds,
    sufficient_drift_detection_samples,
    sufficient_interval_resolution_samples,
    sufficient_resolution_samples,
)


def plan() -> VDCPlan:
    return VDCPlan(
        confidence_thresholds=(0.7, 0.8, 0.9),
        family_tasks=30,
        declared_looks=3,
    )


def indicators(count: int, positives: int) -> list[bool]:
    return [True] * positives + [False] * (count - positives)


def grid(count: int, positives: int) -> dict[float, list[bool]]:
    return {threshold: indicators(count, positives) for threshold in plan().confidence_thresholds}


def test_clear_stable_update_is_deployed() -> None:
    count = 240
    result = decide_vdc(
        [0.5] * count,
        grid(count, 0),
        grid(count, 0),
        plan(),
        at_task_cap=True,
    )
    assert result.action is VDCAction.DEPLOY
    assert result.reason == "promote"


def test_resolved_drift_retains_fallback() -> None:
    count = 20_000
    result = decide_vdc(
        [0.5] * count,
        grid(count, 1_800),
        grid(count, 0),
        plan(),
        at_task_cap=True,
    )
    assert result.action is VDCAction.RETAIN
    assert result.reason == "retain-verifier-drift"
    assert result.bounds.verifier_drift.lower > plan().maximum_verifier_drift


def test_boundary_drift_requests_more_verifier_evidence_at_cap() -> None:
    count = 20_000
    result = decide_vdc(
        [0.5] * count,
        grid(count, 1_000),
        grid(count, 0),
        plan(),
        at_task_cap=True,
    )
    assert result.action is VDCAction.VERIFY_MORE
    assert result.reason == "verifier-evidence-unresolved"


def test_recalibration_requires_an_explicit_repair_indication() -> None:
    count = 20_000
    result = decide_vdc(
        [0.5] * count,
        grid(count, 1_000),
        grid(count, 0),
        plan(),
        at_task_cap=True,
        recalibration_indicated=True,
    )
    assert result.action is VDCAction.RECALIBRATE
    assert result.reason == "recalibration-indicated"


def test_boundary_drift_remains_unresolved_before_cap() -> None:
    count = 20_000
    result = decide_vdc(
        [0.5] * count,
        grid(count, 1_000),
        grid(count, 0),
        plan(),
        at_task_cap=False,
    )
    assert result.action is VDCAction.UNRESOLVED


def test_drift_is_worst_threshold_increase() -> None:
    count = 20_000
    candidate = {
        0.7: indicators(count, 1_000),
        0.8: indicators(count, 2_000),
        0.9: indicators(count, 3_000),
    }
    fallback = grid(count, 0)
    bounds = estimate_vdc_bounds([0.5] * count, candidate, fallback, plan())
    assert bounds.verifier_drift.estimate == pytest.approx(0.15)


def test_sample_bound_grows_as_margin_shrinks() -> None:
    large_margin = sufficient_resolution_samples(
        plan(),
        mean_margin=0.2,
        downside_margin=0.1,
        candidate_risk_margin=0.1,
        drift_margin=0.1,
    )
    small_margin = sufficient_resolution_samples(
        plan(),
        mean_margin=0.1,
        downside_margin=0.05,
        candidate_risk_margin=0.05,
        drift_margin=0.05,
    )
    assert small_margin > large_margin


def test_resolution_bound_accounts_for_sampling_error_and_interval_radius() -> None:
    active_plan = plan()
    bound = sufficient_resolution_samples(
        active_plan,
        mean_margin=0.2,
        downside_margin=0.1,
        candidate_risk_margin=0.1,
        drift_margin=0.1,
    )
    denominator = min(0.2**2, 4 * 0.1**2, 4 * 0.1**2, 0.1**2)
    expected = math.ceil(8 * math.log(1 / active_plan.tail_alpha) / denominator)
    assert bound == expected


def test_component_resolution_bound_matches_endpoint_derivation() -> None:
    bound = sufficient_interval_resolution_samples(
        tail_alpha=0.01, margin=0.05, range_width=1.0
    )
    assert bound == math.ceil(2 * math.log(100) / 0.05**2)


def test_drift_detection_bound_grows_with_grid_and_smaller_separation() -> None:
    baseline = sufficient_drift_detection_samples(
        threshold_count=3, miss_probability=0.05, drift_separation=0.10
    )
    finer_grid = sufficient_drift_detection_samples(
        threshold_count=9, miss_probability=0.05, drift_separation=0.10
    )
    smaller_gap = sufficient_drift_detection_samples(
        threshold_count=3, miss_probability=0.05, drift_separation=0.05
    )
    assert finer_grid > baseline
    assert smaller_gap > baseline


def test_drift_detection_bound_uses_separate_endpoint_constant() -> None:
    bound = sufficient_drift_detection_samples(
        threshold_count=3, miss_probability=0.05, drift_separation=0.10
    )
    expected = math.ceil(8 * math.log(2 * 3 / 0.05) / 0.10**2)
    assert bound == expected


def test_mismatched_threshold_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold grid"):
        estimate_vdc_bounds(
            [0.5],
            {0.8: [False]},
            {0.8: [False]},
            plan(),
        )
