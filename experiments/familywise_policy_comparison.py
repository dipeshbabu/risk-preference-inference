"""Paired, budget-matched synthetic comparisons for v2 routing methods.

This module is development-only. Every method sees the same deterministic
task-specific latent streams within a trial. The comparisons therefore isolate
the routing and testing rules rather than Monte Carlo stream differences.
"""

from __future__ import annotations

import argparse
import json
import random
from math import sqrt
from pathlib import Path
from statistics import fmean

from experiments.anytime_familywise_calibration import (
    SyntheticScenario,
    SyntheticTrialResult,
    bounded_two_point_observation,
    default_scenarios,
    run_synthetic_trial,
    task_stream_seed,
    wilson_interval,
)
from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    RouteDecision,
)
from experiments.familywise_policy_baselines import (
    AlphaSpendingFamilywiseRouter,
    BentkusStitchedFamilywiseRouter,
    fixed_sample_rejections,
)
from experiments.frontier_v2_statistical_hash import (
    STATISTICAL_IMPLEMENTATION_FILES,
    statistical_implementation_sha256,
)


DEFAULT_METHODS = (
    "fixed_hoeffding_bonferroni",
    "fixed_hoeffding_holm",
    "fixed_sign_bonferroni",
    "fixed_sign_holm",
    "alpha_spending_racing",
    "bentkus_stitched_racing",
    "hoeffding_uniform",
    "betting_uniform",
    "betting_resolution",
    "betting_certified",
    "predictable_uniform",
    "predictable_resolution",
)

METHOD_ASSUMPTIONS = {
    "fixed_hoeffding_bonferroni": "bounded fixed-sample conditional-mean null",
    "fixed_hoeffding_holm": "bounded fixed-sample conditional-mean null",
    "fixed_sign_bonferroni": "fixed-sample independent sign null",
    "fixed_sign_holm": "fixed-sample independent sign null",
    "alpha_spending_racing": "bounded anytime conditional-mean null",
    "bentkus_stitched_racing": "bounded IID task streams with anytime stopping",
    "hoeffding_uniform": "bounded anytime conditional-mean null",
    "betting_uniform": "bounded anytime conditional-mean null",
    "betting_resolution": "bounded anytime conditional-mean null",
    "betting_certified": "bounded anytime conditional-mean null",
    "predictable_uniform": "bounded anytime conditional-mean null",
    "predictable_resolution": "bounded anytime conditional-mean null",
}


def balanced_fixed_allocation(
    task_names: tuple[str, ...],
    *,
    total_budget: int,
    maximum_per_task: int,
    seed: int,
) -> dict[str, int]:
    """Outcome-blind near-equal allocation that uses the available budget."""

    if not task_names:
        raise ValueError("at least one task is required")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if maximum_per_task <= 0:
        raise ValueError("maximum_per_task must be positive")
    usable = min(total_budget, len(task_names) * maximum_per_task)
    quotient, remainder = divmod(usable, len(task_names))
    allocation = {task: quotient for task in task_names}
    remainder_order = sorted(
        task_names,
        key=lambda task: (task_stream_seed(seed, f"fixed-allocation:{task}"), task),
    )
    for task in remainder_order[:remainder]:
        allocation[task] += 1
    return allocation


def generate_task_streams(
    scenario: SyntheticScenario,
    *,
    seed: int,
    lengths: dict[str, int],
    lower: float = -1.0,
    upper: float = 1.0,
) -> dict[str, list[float]]:
    means = dict(scenario.task_means)
    if set(lengths) != set(means):
        raise ValueError("stream lengths must cover exactly the scenario tasks")
    streams: dict[str, list[float]] = {}
    for task in sorted(means):
        if lengths[task] < 0:
            raise ValueError("stream lengths cannot be negative")
        rng = random.Random(task_stream_seed(seed, task))
        streams[task] = [
            bounded_two_point_observation(
                means[task],
                lower=lower,
                upper=upper,
                rng=rng,
            )
            for _ in range(lengths[task])
        ]
    return streams


def _trial_result(
    scenario: SyntheticScenario,
    *,
    accepted: set[str],
    rejected_count: int,
    unresolved_count: int,
    total_observations: int,
    effect_margin: float,
) -> SyntheticTrialResult:
    means = dict(scenario.task_means)
    false_accepted = {task for task in accepted if means[task] <= effect_margin}
    true_accepted = {task for task in accepted if means[task] > effect_margin}
    positive_tasks = {task for task, mean in means.items() if mean > effect_margin}
    return SyntheticTrialResult(
        false_accept=bool(false_accepted),
        false_accept_count=len(false_accepted),
        true_accept_count=len(true_accepted),
        positive_task_count=len(positive_tasks),
        rejected_count=rejected_count,
        unresolved_count=unresolved_count,
        total_observations=total_observations,
        expected_equal_task_improvement=(
            sum(means[task] for task in accepted) / len(means)
        ),
    )


def run_fixed_sample_trial(
    scenario: SyntheticScenario,
    *,
    seed: int,
    test: str,
    correction: str,
    familywise_alpha: float,
    effect_margin: float,
    maximum_observations_per_task: int,
    global_observation_budget: int,
) -> SyntheticTrialResult:
    task_names = tuple(sorted(dict(scenario.task_means)))
    allocation = balanced_fixed_allocation(
        task_names,
        total_budget=global_observation_budget,
        maximum_per_task=maximum_observations_per_task,
        seed=seed,
    )
    streams = generate_task_streams(scenario, seed=seed, lengths=allocation)
    accepted = fixed_sample_rejections(
        streams,
        test=test,
        correction=correction,
        familywise_alpha=familywise_alpha,
        null_mean=effect_margin,
        lower=-1.0,
        upper=1.0,
    )
    return _trial_result(
        scenario,
        accepted=accepted,
        rejected_count=len(task_names) - len(accepted),
        unresolved_count=0,
        total_observations=sum(allocation.values()),
        effect_margin=effect_margin,
    )


def run_alpha_spending_trial(
    scenario: SyntheticScenario,
    *,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    maximum_observations_per_task: int,
    global_observation_budget: int,
) -> SyntheticTrialResult:
    means = dict(scenario.task_means)
    plan = AnytimeFamilywisePlan(
        task_names=tuple(sorted(means)),
        familywise_alpha=familywise_alpha,
        futility_familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        maximum_observations_per_task=maximum_observations_per_task,
    )
    router = AlphaSpendingFamilywiseRouter(plan)
    task_rngs = {
        task: random.Random(task_stream_seed(seed, task)) for task in plan.task_names
    }
    while router.total_observations() < global_observation_budget:
        task = router.next_task()
        if task is None:
            break
        observation = bounded_two_point_observation(
            means[task],
            lower=plan.observation_lower,
            upper=plan.observation_upper,
            rng=task_rngs[task],
        )
        router.update(task, observation)

    decisions = router.decisions()
    accepted = {
        task
        for task, evidence in decisions.items()
        if evidence.decision is RouteDecision.ACCEPT_CANDIDATE
    }
    rejected_count = sum(
        evidence.decision is RouteDecision.REJECT_TO_FALLBACK
        for evidence in decisions.values()
    )
    unresolved_count = sum(
        evidence.decision
        in {RouteDecision.UNDECIDED, RouteDecision.BUDGET_EXHAUSTED}
        for evidence in decisions.values()
    )
    return _trial_result(
        scenario,
        accepted=accepted,
        rejected_count=rejected_count,
        unresolved_count=unresolved_count,
        total_observations=router.total_observations(),
        effect_margin=effect_margin,
    )


def run_bentkus_stitched_trial(
    scenario: SyntheticScenario,
    *,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    maximum_observations_per_task: int,
    global_observation_budget: int,
) -> SyntheticTrialResult:
    means = dict(scenario.task_means)
    plan = AnytimeFamilywisePlan(
        task_names=tuple(sorted(means)),
        familywise_alpha=familywise_alpha,
        futility_familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        maximum_observations_per_task=maximum_observations_per_task,
    )
    router = BentkusStitchedFamilywiseRouter(plan)
    task_rngs = {
        task: random.Random(task_stream_seed(seed, task)) for task in plan.task_names
    }
    while router.total_observations() < global_observation_budget:
        task = router.next_task()
        if task is None:
            break
        observation = bounded_two_point_observation(
            means[task],
            lower=plan.observation_lower,
            upper=plan.observation_upper,
            rng=task_rngs[task],
        )
        router.update(task, observation)

    decisions = router.decisions()
    accepted = {
        task
        for task, evidence in decisions.items()
        if evidence.decision is RouteDecision.ACCEPT_CANDIDATE
    }
    rejected_count = sum(
        evidence.decision is RouteDecision.REJECT_TO_FALLBACK
        for evidence in decisions.values()
    )
    unresolved_count = sum(
        evidence.decision
        in {RouteDecision.UNDECIDED, RouteDecision.BUDGET_EXHAUSTED}
        for evidence in decisions.values()
    )
    return _trial_result(
        scenario,
        accepted=accepted,
        rejected_count=rejected_count,
        unresolved_count=unresolved_count,
        total_observations=router.total_observations(),
        effect_margin=effect_margin,
    )


def run_method_trial(
    scenario: SyntheticScenario,
    *,
    method: str,
    seed: int,
    familywise_alpha: float,
    effect_margin: float,
    maximum_observations_per_task: int,
    global_observation_budget: int,
    forced_initial_observations: int,
) -> SyntheticTrialResult:
    if method not in DEFAULT_METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    if method.startswith("fixed_"):
        _fixed, test, correction = method.split("_")
        return run_fixed_sample_trial(
            scenario,
            seed=seed,
            test=test,
            correction=correction,
            familywise_alpha=familywise_alpha,
            effect_margin=effect_margin,
            maximum_observations_per_task=maximum_observations_per_task,
            global_observation_budget=global_observation_budget,
        )
    if method == "alpha_spending_racing":
        return run_alpha_spending_trial(
            scenario,
            seed=seed,
            familywise_alpha=familywise_alpha,
            effect_margin=effect_margin,
            maximum_observations_per_task=maximum_observations_per_task,
            global_observation_budget=global_observation_budget,
        )
    if method == "bentkus_stitched_racing":
        return run_bentkus_stitched_trial(
            scenario,
            seed=seed,
            familywise_alpha=familywise_alpha,
            effect_margin=effect_margin,
            maximum_observations_per_task=maximum_observations_per_task,
            global_observation_budget=global_observation_budget,
        )
    e_process_method, strategy = method.split("_")
    router_method = (
        "predictable_betting"
        if e_process_method == "predictable"
        else f"{e_process_method}_mixture"
    )
    return run_synthetic_trial(
        scenario,
        strategy=strategy,
        seed=seed,
        familywise_alpha=familywise_alpha,
        effect_margin=effect_margin,
        e_process_method=router_method,
        maximum_observations_per_task=maximum_observations_per_task,
        global_observation_budget=global_observation_budget,
        forced_initial_observations=forced_initial_observations,
    )


def normal_mean_interval(
    values: list[float], z_value: float = 1.959963984540054
) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    mean = fmean(values)
    if len(values) == 1:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    radius = z_value * sqrt(variance / len(values))
    return mean - radius, mean + radius


def summarize_paired_methods(
    scenario: SyntheticScenario,
    *,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    reference_method: str = "betting_uniform",
    trials: int,
    seed: int,
    familywise_alpha: float = 0.05,
    effect_margin: float = 0.0,
    maximum_observations_per_task: int = 200,
    global_observation_budget: int = 1_150,
    forced_initial_observations: int = 2,
) -> dict:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    if any(method not in DEFAULT_METHODS for method in methods):
        raise ValueError("methods contain an unknown comparison method")
    if reference_method not in methods:
        raise ValueError("reference_method must be included in methods")

    results = {
        method: [
            run_method_trial(
                scenario,
                method=method,
                seed=seed + trial_index,
                familywise_alpha=familywise_alpha,
                effect_margin=effect_margin,
                maximum_observations_per_task=maximum_observations_per_task,
                global_observation_budget=global_observation_budget,
                forced_initial_observations=forced_initial_observations,
            )
            for trial_index in range(trials)
        ]
        for method in methods
    }
    positive_task_count = results[methods[0]][0].positive_task_count
    method_summaries = {}
    for method, method_results in results.items():
        false_accept_families = sum(result.false_accept for result in method_results)
        method_summaries[method] = {
            "assumption": METHOD_ASSUMPTIONS[method],
            "familywise_false_accept_rate": false_accept_families / trials,
            "familywise_false_accept_wilson_95_ci": list(
                wilson_interval(false_accept_families, trials)
            ),
            "mean_false_accept_count": fmean(
                result.false_accept_count for result in method_results
            ),
            "mean_true_accept_count": fmean(
                result.true_accept_count for result in method_results
            ),
            "mean_positive_acceptance_rate": (
                fmean(result.true_accept_count for result in method_results)
                / positive_task_count
                if positive_task_count
                else None
            ),
            "mean_rejected_count": fmean(
                result.rejected_count for result in method_results
            ),
            "mean_unresolved_count": fmean(
                result.unresolved_count for result in method_results
            ),
            "mean_total_observations": fmean(
                result.total_observations for result in method_results
            ),
            "mean_expected_equal_task_improvement": fmean(
                result.expected_equal_task_improvement for result in method_results
            ),
        }

    reference_results = results[reference_method]
    paired_differences = {}
    for method in methods:
        if method == reference_method:
            continue
        method_results = results[method]
        acceptance_rate_differences = [
            (candidate.true_accept_count - reference.true_accept_count)
            / positive_task_count
            for candidate, reference in zip(method_results, reference_results, strict=True)
        ] if positive_task_count else []
        improvement_differences = [
            candidate.expected_equal_task_improvement
            - reference.expected_equal_task_improvement
            for candidate, reference in zip(method_results, reference_results, strict=True)
        ]
        episode_savings = [
            reference.total_observations - candidate.total_observations
            for candidate, reference in zip(method_results, reference_results, strict=True)
        ]
        paired_differences[method] = {
            "positive_acceptance_rate_difference": (
                fmean(acceptance_rate_differences)
                if acceptance_rate_differences
                else None
            ),
            "positive_acceptance_rate_difference_normal_95_ci": (
                list(normal_mean_interval(acceptance_rate_differences))
                if acceptance_rate_differences
                else None
            ),
            "expected_equal_task_improvement_difference": fmean(
                improvement_differences
            ),
            "expected_equal_task_improvement_difference_normal_95_ci": list(
                normal_mean_interval(improvement_differences)
            ),
            "mean_episode_savings": fmean(episode_savings),
            "episode_savings_normal_95_ci": list(
                normal_mean_interval(episode_savings)
            ),
        }

    return {
        "scope": "Paired synthetic development comparison; no confirmation artifact is read.",
        "scenario": scenario.name,
        "task_means": dict(scenario.task_means),
        "trials": trials,
        "seed": seed,
        "familywise_alpha": familywise_alpha,
        "effect_margin": effect_margin,
        "maximum_observations_per_task": maximum_observations_per_task,
        "global_observation_budget": global_observation_budget,
        "forced_initial_observations": forced_initial_observations,
        "positive_task_count": positive_task_count,
        "reference_method": reference_method,
        "method_summaries": method_summaries,
        "paired_differences_from_reference": paired_differences,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=tuple(default_scenarios()), default="mixed_effects"
    )
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--reference-method", choices=DEFAULT_METHODS, default="betting_uniform")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2_026_071_900)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--effect-margin", type=float, default=0.0)
    parser.add_argument("--maximum-observations-per-task", type=int, default=200)
    parser.add_argument("--global-observation-budget", type=int, default=1_150)
    parser.add_argument("--forced-initial-observations", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_paired_methods(
        default_scenarios()[args.scenario],
        methods=tuple(args.methods),
        reference_method=args.reference_method,
        trials=args.trials,
        seed=args.seed,
        familywise_alpha=args.familywise_alpha,
        effect_margin=args.effect_margin,
        maximum_observations_per_task=args.maximum_observations_per_task,
        global_observation_budget=args.global_observation_budget,
        forced_initial_observations=args.forced_initial_observations,
    )
    payload = {
        "design": "riskshiftbench-v2-paired-familywise-method-comparison",
        "statistical_implementation_files": list(STATISTICAL_IMPLEMENTATION_FILES),
        "statistical_implementation_sha256": statistical_implementation_sha256(),
        "summary": summary,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
