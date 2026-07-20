from __future__ import annotations

from math import log

import pytest

from experiments.familywise_close_comparators import (
    AgrapaFamilywiseRouter,
    NSCORE_NUMERICAL_TRUNCATION,
    NScoreFamilywiseRouter,
    altt_bonferroni_rejections,
    e_holm_rejections,
    nscore_kde_fraction,
)


def test_e_holm_can_reject_when_anytime_bonferroni_cannot() -> None:
    current = {"strong": log(35.0), "weak": log(10.0)}
    assert e_holm_rejections(current, familywise_alpha=0.05) == {"strong"}
    assert altt_bonferroni_rejections(
        current, familywise_alpha=0.05
    ) == set()


@pytest.mark.parametrize("selection_rule", ["altt_bonferroni", "e_holm"])
def test_agrapa_router_accepts_a_clear_positive_stream(selection_rule: str) -> None:
    router = AgrapaFamilywiseRouter(
        ("positive",),
        selection_rule=selection_rule,
        maximum_observations_per_task=100,
    )
    first = router.update("positive", 1.0)
    assert first.last_betting_fraction == 0.0
    while not router.evidence("positive").accepted:
        router.update("positive", 1.0)
    assert router.total_observations() <= 100


def test_agrapa_stake_is_predictable_and_outcomes_are_bounded() -> None:
    positive = AgrapaFamilywiseRouter(("task",))
    negative = AgrapaFamilywiseRouter(("task",))
    positive.update("task", 1.0)
    negative.update("task", -1.0)
    assert positive.evidence("task").last_betting_fraction == 0.0
    assert negative.evidence("task").last_betting_fraction == 0.0
    with pytest.raises(ValueError, match="outside"):
        positive.update("task", 1.01)


def test_epsilon_greedy_acquisition_is_seed_deterministic() -> None:
    first = AgrapaFamilywiseRouter(("a", "b", "c"), acquisition_seed=19)
    second = AgrapaFamilywiseRouter(("a", "b", "c"), acquisition_seed=19)
    first_sequence = []
    second_sequence = []
    for _ in range(12):
        first_task = first.next_task()
        second_task = second.next_task()
        assert first_task is not None and second_task is not None
        first_sequence.append(first_task)
        second_sequence.append(second_task)
        first.update(first_task, 0.0)
        second.update(second_task, 0.0)
    assert first_sequence == second_sequence


def test_nscore_fraction_uses_only_past_histograms() -> None:
    assert nscore_kde_fraction((0, 0), (0, 0)) == 0.0
    favorable = nscore_kde_fraction((10, 0), (0, 10))
    unfavorable = nscore_kde_fraction((0, 10), (10, 0))
    assert favorable == pytest.approx(NSCORE_NUMERICAL_TRUNCATION)
    assert unfavorable == 0.0


def test_nscore_router_accepts_a_clear_paired_improvement() -> None:
    router = NScoreFamilywiseRouter(
        ("positive",),
        maximum_observations_per_task=100,
    )
    first = router.update(
        "positive",
        fallback_score=0.0,
        candidate_score=1.0,
    )
    assert first.last_betting_fraction == 0.0
    while not router.evidence("positive").accepted:
        router.update(
            "positive",
            fallback_score=0.0,
            candidate_score=1.0,
        )
    assert router.total_observations() <= 100


def test_nscore_router_rejects_unbounded_policy_scores() -> None:
    router = NScoreFamilywiseRouter(("task",))
    with pytest.raises(ValueError, match="outside"):
        router.update("task", fallback_score=0.0, candidate_score=1.01)
