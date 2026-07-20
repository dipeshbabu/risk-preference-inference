from __future__ import annotations

import pytest

from experiments.anytime_familywise_router import AnytimeFamilywisePlan, RouteDecision
from experiments.familywise_policy_baselines import (
    AlphaSpendingFamilywiseRouter,
    BentkusStitchedFamilywiseRouter,
    alpha_spending_radius,
    bentkus_binomial_p2_bound,
    bentkus_stitched_boundary,
    bonferroni_rejections,
    exact_sign_p_value,
    fixed_sample_hoeffding_p_value,
    fixed_sample_rejections,
    holm_rejections,
    task_resolution_bound,
)


def test_fixed_sample_hoeffding_p_value_orders_clear_effects() -> None:
    null_p = fixed_sample_hoeffding_p_value(
        [0.0] * 20,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    positive_p = fixed_sample_hoeffding_p_value(
        [1.0] * 20,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    assert null_p == 1.0
    assert positive_p < null_p


def test_exact_sign_p_value_matches_unanimous_probability() -> None:
    assert exact_sign_p_value([1.0] * 9) == pytest.approx(1.0 / 512.0)
    assert exact_sign_p_value([0.0] * 9) == 1.0


def test_bonferroni_and_holm_rejections() -> None:
    p_values = {"a": 0.001, "b": 0.02, "c": 0.5}
    assert bonferroni_rejections(p_values, 0.05) == {"a"}
    assert holm_rejections(p_values, 0.05) == {"a", "b"}


def test_fixed_sample_dispatch() -> None:
    observations = {"a": [1.0] * 20, "b": [-1.0] * 20}
    rejected = fixed_sample_rejections(
        observations,
        test="hoeffding",
        correction="bonferroni",
        familywise_alpha=0.05,
        null_mean=0.0,
        lower=-1.0,
        upper=1.0,
    )
    assert rejected == {"a"}


def test_alpha_spending_radius_shrinks_and_resolution_bound_separates() -> None:
    assert alpha_spending_radius(100, alpha=0.05, width=2.0) < alpha_spending_radius(
        10, alpha=0.05, width=2.0
    )
    small_gap = task_resolution_bound(0.2, alpha=0.05, width=2.0)
    large_gap = task_resolution_bound(0.5, alpha=0.05, width=2.0)
    assert large_gap < small_gap


def test_alpha_spending_router_accepts_and_rejects_clear_streams() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("positive", "negative"),
        maximum_observations_per_task=2_000,
    )
    router = AlphaSpendingFamilywiseRouter(plan)
    while router.evidence("positive").decision is RouteDecision.UNDECIDED:
        router.update("positive", 1.0)
    while router.evidence("negative").decision is RouteDecision.UNDECIDED:
        router.update("negative", -1.0)
    assert router.evidence("positive").decision is RouteDecision.ACCEPT_CANDIDATE
    assert router.evidence("negative").decision is RouteDecision.REJECT_TO_FALLBACK


def test_alpha_spending_race_samples_widest_interval() -> None:
    plan = AnytimeFamilywisePlan(task_names=("a", "b"))
    router = AlphaSpendingFamilywiseRouter(plan)
    assert router.next_task() == "a"
    router.update("a", 0.0)
    assert router.next_task() == "b"


def test_bentkus_binomial_p2_matches_one_trial_closed_form() -> None:
    assert bentkus_binomial_p2_bound(
        0.75,
        trials=1,
        success_probability=0.5,
    ) == pytest.approx(0.8)
    assert bentkus_binomial_p2_bound(
        1.0,
        trials=1,
        success_probability=0.5,
    ) == pytest.approx(0.5)


def test_bentkus_stitching_controls_exact_null_crossing_probability() -> None:
    alpha = 0.05
    horizon = 128
    alive = {0: 1.0}
    crossing_probability = 0.0
    for observations in range(1, horizon + 1):
        boundary = bentkus_stitched_boundary(
            observations,
            alpha=alpha,
            observation_lower=-1.0,
            observation_upper=1.0,
            null_mean=0.0,
        )
        next_alive: dict[int, float] = {}
        for total, probability in alive.items():
            for increment in (-1, 1):
                updated = total + increment
                branch_probability = probability / 2.0
                if updated >= boundary:
                    crossing_probability += branch_probability
                else:
                    next_alive[updated] = (
                        next_alive.get(updated, 0.0) + branch_probability
                    )
        alive = next_alive
    assert crossing_probability <= alpha

    bentkus_boundary = bentkus_stitched_boundary(
        64,
        alpha=alpha,
        observation_lower=-1.0,
        observation_upper=1.0,
        null_mean=0.0,
    )
    hoeffding_boundary = 64 * alpha_spending_radius(
        64,
        alpha=alpha,
        width=2.0,
    )
    assert bentkus_boundary < hoeffding_boundary


@pytest.mark.parametrize("null_mean", [-0.3, 0.25])
def test_bentkus_stitching_controls_noncentered_two_point_nulls(
    null_mean: float,
) -> None:
    alpha = 0.05
    success_probability = (null_mean + 1.0) / 2.0
    alive = {0: 1.0}
    crossing_probability = 0.0
    for observations in range(1, 129):
        boundary = bentkus_stitched_boundary(
            observations,
            alpha=alpha,
            observation_lower=-1.0,
            observation_upper=1.0,
            null_mean=null_mean,
        )
        next_alive: dict[int, float] = {}
        for successes, probability in alive.items():
            for increment, branch_weight in (
                (0, 1.0 - success_probability),
                (1, success_probability),
            ):
                updated_successes = successes + increment
                branch_probability = probability * branch_weight
                centered_sum = 2.0 * (
                    updated_successes - observations * success_probability
                )
                if centered_sum >= boundary:
                    crossing_probability += branch_probability
                else:
                    next_alive[updated_successes] = (
                        next_alive.get(updated_successes, 0.0)
                        + branch_probability
                    )
        alive = next_alive
    assert crossing_probability <= alpha


def test_bentkus_router_accepts_and_rejects_clear_iid_streams() -> None:
    plan = AnytimeFamilywisePlan(
        task_names=("positive", "negative"),
        maximum_observations_per_task=200,
    )
    router = BentkusStitchedFamilywiseRouter(plan)
    while router.evidence("positive").decision is RouteDecision.UNDECIDED:
        router.update("positive", 1.0)
    while router.evidence("negative").decision is RouteDecision.UNDECIDED:
        router.update("negative", -1.0)
    assert router.evidence("positive").decision is RouteDecision.ACCEPT_CANDIDATE
    assert router.evidence("negative").decision is RouteDecision.REJECT_TO_FALLBACK
