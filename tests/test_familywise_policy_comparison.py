from __future__ import annotations

from experiments.anytime_familywise_calibration import SyntheticScenario
from experiments.familywise_policy_comparison import (
    DEFAULT_METHODS,
    METHOD_ASSUMPTIONS,
    balanced_fixed_allocation,
    generate_task_streams,
    run_method_trial,
    summarize_paired_methods,
)


SMALL_SCENARIO = SyntheticScenario(
    name="small",
    task_means=(("negative", -0.5), ("null", 0.0), ("positive", 0.5)),
    planning_effect_gaps=(("negative", 0.4), ("null", 0.1), ("positive", 0.4)),
)


def test_balanced_fixed_allocation_uses_budget_and_cap() -> None:
    allocation = balanced_fixed_allocation(
        ("a", "b", "c"), total_budget=8, maximum_per_task=3, seed=7
    )
    assert sum(allocation.values()) == 8
    assert max(allocation.values()) <= 3
    assert max(allocation.values()) - min(allocation.values()) <= 1

    capped = balanced_fixed_allocation(
        ("a", "b"), total_budget=20, maximum_per_task=4, seed=7
    )
    assert capped == {"a": 4, "b": 4}


def test_generated_streams_are_deterministic() -> None:
    lengths = {"negative": 3, "null": 3, "positive": 3}
    assert generate_task_streams(SMALL_SCENARIO, seed=11, lengths=lengths) == (
        generate_task_streams(SMALL_SCENARIO, seed=11, lengths=lengths)
    )


def test_every_method_respects_the_global_budget() -> None:
    for method in DEFAULT_METHODS:
        result = run_method_trial(
            SMALL_SCENARIO,
            method=method,
            seed=17,
            familywise_alpha=0.05,
            effect_margin=0.0,
            maximum_observations_per_task=12,
            global_observation_budget=24,
            forced_initial_observations=1,
        )
        assert result.total_observations <= 24
        assert result.positive_task_count == 1

    assert "bentkus_stitched_racing" in DEFAULT_METHODS
    assert "IID" in METHOD_ASSUMPTIONS["bentkus_stitched_racing"]
    assert "altt_agrapa_bonferroni" in DEFAULT_METHODS
    assert "eholm_agrapa" in DEFAULT_METHODS
    assert "nscore11_bonferroni" in DEFAULT_METHODS
    assert "always-valid" in METHOD_ASSUMPTIONS["eholm_agrapa"]


def test_paired_summary_is_deterministic_and_reports_contrasts() -> None:
    methods = (
        "fixed_hoeffding_holm",
        "alpha_spending_racing",
        "betting_uniform",
        "betting_resolution",
    )
    first = summarize_paired_methods(
        SMALL_SCENARIO,
        methods=methods,
        reference_method="betting_uniform",
        trials=4,
        seed=21,
        maximum_observations_per_task=10,
        global_observation_budget=24,
        forced_initial_observations=1,
    )
    second = summarize_paired_methods(
        SMALL_SCENARIO,
        methods=methods,
        reference_method="betting_uniform",
        trials=4,
        seed=21,
        maximum_observations_per_task=10,
        global_observation_budget=24,
        forced_initial_observations=1,
    )
    assert first == second
    assert set(first["method_summaries"]) == set(methods)
    assert "betting_resolution" in first["paired_differences_from_reference"]
