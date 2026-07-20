from __future__ import annotations

import math

import pytest

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
    RouteDecision,
    betting_component_growth_lower_bound,
    betting_resolution_sample_bound,
)


def test_uniform_alpha_weights_sum_to_familywise_level() -> None:
    plan = AnytimeFamilywisePlan(task_names=("a", "b", "c"))
    assert sum(plan.acceptance_alpha(task) for task in plan.task_names) == pytest.approx(
        plan.familywise_alpha
    )


def test_development_weights_are_normalized_without_changing_familywise_level() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("a", "b"),
        task_weights=(("a", 3.0), ("b", 1.0)),
    )
    assert plan.acceptance_alpha("a") == pytest.approx(0.0375)
    assert plan.acceptance_alpha("b") == pytest.approx(0.0125)
    assert sum(plan.acceptance_alpha(task) for task in plan.task_names) == pytest.approx(
        0.05
    )


def test_positive_bounded_stream_crosses_acceptance_boundary() -> None:
    plan = AnytimeFamilywisePlan(task_names=("positive",), maximum_observations_per_task=200)
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(200):
        evidence = router.update("positive", 1.0)
        if evidence.decision is not RouteDecision.UNDECIDED:
            break
    assert evidence is not None
    assert evidence.decision is RouteDecision.ACCEPT_CANDIDATE
    assert evidence.acceptance_log_e >= evidence.acceptance_log_threshold


@pytest.mark.parametrize(
    "method",
    ["hoeffding_mixture", "betting_mixture", "predictable_betting"],
)
def test_each_e_process_method_accepts_clear_positive_stream(method: str) -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("positive",),
        maximum_observations_per_task=200,
        e_process_method=method,
    )
    router = AnytimeFamilywiseRouter(plan)
    while router.evidence("positive").decision is RouteDecision.UNDECIDED:
        router.update("positive", 1.0)
    assert router.evidence("positive").decision is RouteDecision.ACCEPT_CANDIDATE


def test_predictable_betting_uses_no_current_outcome_in_its_stake() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("task",),
        e_process_method="predictable_betting",
        maximum_observations_per_task=200,
    )
    positive = AnytimeFamilywiseRouter(plan)
    negative = AnytimeFamilywiseRouter(plan)
    positive_first = positive.update("task", 1.0)
    negative_first = negative.update("task", -1.0)
    assert positive_first.acceptance_log_e == pytest.approx(0.0)
    assert negative_first.acceptance_log_e == pytest.approx(0.0)
    assert positive.update("task", 1.0).acceptance_log_e > 0.0


def test_predictable_betting_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="prior strengths"):
        AnytimeFamilywisePlan(
            task_names=("task",),
            e_process_method="predictable_betting",
            predictable_prior_strengths=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="maximum fraction"):
        AnytimeFamilywisePlan(
            task_names=("task",),
            e_process_method="predictable_betting",
            predictable_max_fraction=1.0,
        )


def test_negative_bounded_stream_crosses_futility_boundary() -> None:
    plan = AnytimeFamilywisePlan(task_names=("negative",), maximum_observations_per_task=200)
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(200):
        evidence = router.update("negative", -1.0)
        if evidence.decision is not RouteDecision.UNDECIDED:
            break
    assert evidence is not None
    assert evidence.decision is RouteDecision.REJECT_TO_FALLBACK
    assert evidence.futility_log_e >= evidence.futility_log_threshold


def test_null_stream_does_not_gain_evidence_from_time_alone() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("null",),
        maximum_observations_per_task=20,
    )
    router = AnytimeFamilywiseRouter(plan)
    evidence = None
    for _ in range(20):
        evidence = router.update("null", 0.0)
    assert evidence is not None
    assert evidence.decision is RouteDecision.BUDGET_EXHAUSTED
    assert evidence.acceptance_log_e < 0.0
    assert evidence.futility_log_e < 0.0


def test_resolution_allocation_prioritizes_clear_task_after_forced_sampling() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("ambiguous", "clear"),
        maximum_observations_per_task=200,
    )
    router = AnytimeFamilywiseRouter(plan)
    router.update("ambiguous", 0.01)
    router.update("clear", 0.9)
    assert (
        router.next_task(strategy="resolution", forced_initial_observations=1)
        == "clear"
    )


def test_uniform_allocation_uses_sample_count_then_name() -> None:
    plan = AnytimeFamilywisePlan(task_names=("b", "a"))
    router = AnytimeFamilywiseRouter(plan)
    assert router.next_task(strategy="uniform") == "a"
    router.update("a", 0.0)
    assert router.next_task(strategy="uniform") == "b"


def test_observation_bounds_are_enforced() -> None:
    router = AnytimeFamilywiseRouter(AnytimeFamilywisePlan(task_names=("a",)))
    with pytest.raises(ValueError, match="outside"):
        router.update("a", 1.01)
    with pytest.raises(ValueError, match="finite"):
        router.update("a", math.nan)


def test_terminal_task_cannot_be_sampled_again() -> None:
    router = AnytimeFamilywiseRouter(
        AnytimeFamilywisePlan(task_names=("a",), maximum_observations_per_task=200)
    )
    while router.evidence("a").decision is RouteDecision.UNDECIDED:
        router.update("a", 1.0)
    with pytest.raises(RuntimeError, match="terminal"):
        router.update("a", 1.0)


def test_invalid_weight_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly"):
        AnytimeFamilywisePlan(
            task_names=("a", "b"),
            task_weights=(("a", 1.0),),
        )


def test_betting_resolution_bound_shrinks_with_effect_gap() -> None:
    small, _small_fraction = betting_resolution_sample_bound(
        0.2, alpha=0.01, beta=0.01
    )
    large, _large_fraction = betting_resolution_sample_bound(
        0.6, alpha=0.01, beta=0.01
    )
    assert large < small
    assert betting_component_growth_lower_bound(0.6, 0.2) > 0.0


def test_certified_allocation_completes_shortest_frozen_quota_first() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("hard", "easy"),
        planning_effect_gaps=(("hard", 0.2), ("easy", 0.6)),
        maximum_observations_per_task=2_000,
    )
    router = AnytimeFamilywiseRouter(plan)
    assert router.next_task(strategy="certified", forced_initial_observations=1) == "easy"
    router.update("easy", 0.0)
    assert router.next_task(strategy="certified", forced_initial_observations=1) == "hard"
    router.update("hard", 0.0)
    assert router.next_task(strategy="certified", forced_initial_observations=1) == "easy"
    targets = {target.task: target for target in router.certified_sample_targets()}
    assert targets["easy"].required_observations < targets["hard"].required_observations
    assert not targets["easy"].clipped_by_task_cap


def test_certified_allocation_preserves_gap_order_when_targets_are_capped() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("a_hard", "z_easy"),
        planning_effect_gaps=(("a_hard", 0.05), ("z_easy", 0.5)),
        maximum_observations_per_task=20,
    )
    router = AnytimeFamilywiseRouter(plan)
    router.update("a_hard", 0.0)
    router.update("z_easy", 0.0)

    targets = {target.task: target for target in router.certified_sample_targets()}
    assert targets["a_hard"].clipped_by_task_cap
    assert targets["z_easy"].clipped_by_task_cap
    assert (
        targets["z_easy"].required_observations
        < targets["a_hard"].required_observations
    )
    assert (
        router.next_task(strategy="certified", forced_initial_observations=1)
        == "z_easy"
    )


def test_certified_allocation_requires_planning_gaps() -> None:
    router = AnytimeFamilywiseRouter(AnytimeFamilywisePlan(task_names=("task",)))
    with pytest.raises(RuntimeError, match="planning effect gaps"):
        router.next_task(strategy="certified", forced_initial_observations=1)
