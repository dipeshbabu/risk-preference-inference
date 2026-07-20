"""Familywise-valid baselines for RiskShiftBench v2 development.

The fixed-sample tests and alpha-spending confidence sequences in this module
use only bounded paired score differences. They provide transparent reference
points for the more powerful betting-mixture router.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, comb, exp, floor, log, nextafter, pi, sqrt
from statistics import fmean

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    RouteDecision,
)


def fixed_sample_hoeffding_p_value(
    observations: list[float] | tuple[float, ...],
    *,
    null_mean: float,
    lower: float,
    upper: float,
) -> float:
    """One-sided bounded-mean p-value for a fixed sample size."""

    if not observations:
        raise ValueError("at least one observation is required")
    if lower >= upper:
        raise ValueError("observation bounds must be ordered")
    if any(not lower <= value <= upper for value in observations):
        raise ValueError("observation lies outside the declared bounds")
    advantage = max(fmean(observations) - null_mean, 0.0)
    width = upper - lower
    return min(1.0, exp(-2.0 * len(observations) * advantage**2 / width**2))


def exact_sign_p_value(
    observations: list[float] | tuple[float, ...],
    *,
    null_value: float = 0.0,
) -> float:
    """Exact one-sided sign-test p-value after discarding ties."""

    nonzero = [value for value in observations if value != null_value]
    if not nonzero:
        return 1.0
    positive = sum(value > null_value for value in nonzero)
    return sum(comb(len(nonzero), count) for count in range(positive, len(nonzero) + 1)) / (
        2 ** len(nonzero)
    )


def bonferroni_rejections(
    p_values: dict[str, float], familywise_alpha: float = 0.05
) -> set[str]:
    if not p_values:
        return set()
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must lie between zero and one")
    local_alpha = familywise_alpha / len(p_values)
    return {task for task, p_value in p_values.items() if p_value <= local_alpha}


def holm_rejections(
    p_values: dict[str, float], familywise_alpha: float = 0.05
) -> set[str]:
    if not p_values:
        return set()
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    if any(not 0.0 <= value <= 1.0 for _task, value in ordered):
        raise ValueError("p-values must lie between zero and one")
    rejected: set[str] = set()
    family_size = len(ordered)
    for index, (task, p_value) in enumerate(ordered):
        if p_value > familywise_alpha / (family_size - index):
            break
        rejected.add(task)
    return rejected


def fixed_sample_rejections(
    observations: dict[str, list[float] | tuple[float, ...]],
    *,
    test: str,
    correction: str,
    familywise_alpha: float,
    null_mean: float,
    lower: float,
    upper: float,
) -> set[str]:
    if test == "hoeffding":
        p_values = {
            task: fixed_sample_hoeffding_p_value(
                values,
                null_mean=null_mean,
                lower=lower,
                upper=upper,
            )
            for task, values in observations.items()
        }
    elif test == "sign":
        p_values = {
            task: exact_sign_p_value(values, null_value=null_mean)
            for task, values in observations.items()
        }
    else:
        raise ValueError("test must be 'hoeffding' or 'sign'")

    if correction == "bonferroni":
        return bonferroni_rejections(p_values, familywise_alpha)
    if correction == "holm":
        return holm_rejections(p_values, familywise_alpha)
    raise ValueError("correction must be 'bonferroni' or 'holm'")


def alpha_spending_radius(
    observations: int,
    *,
    alpha: float,
    width: float,
) -> float:
    """One-sided time-uniform Hoeffding radius using 1/n^2 spending."""

    if observations <= 0:
        return float("inf")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if width <= 0.0:
        raise ValueError("width must be positive")
    spending_probability = 6.0 * alpha / (pi**2 * observations**2)
    return width * sqrt(log(1.0 / spending_probability) / (2.0 * observations))


def task_resolution_bound(
    gap: float,
    *,
    alpha: float,
    width: float,
    search_limit: int = 10_000_000,
) -> int:
    """First n for which simultaneous coverage forces a correct decision.

    On the confidence-sequence coverage event, the empirical mean can differ
    from the true mean by one radius. The decision bound uses another radius,
    so ``2 * radius < gap`` is sufficient.
    """

    if gap <= 0.0:
        raise ValueError("gap must be positive")
    if search_limit <= 0:
        raise ValueError("search_limit must be positive")
    for observations in range(1, search_limit + 1):
        if 2.0 * alpha_spending_radius(
            observations, alpha=alpha, width=width
        ) < gap:
            return observations
    raise RuntimeError("resolution bound exceeds search_limit")


@lru_cache(maxsize=None)
def _binomial_tail_moments(
    trials: int,
    success_probability: float,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return stable binomial tail probability and first two tail moments."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0.0 < success_probability < 1.0:
        raise ValueError("success_probability must lie strictly between zero and one")

    mode = min(trials, floor((trials + 1) * success_probability))
    weights = [0.0] * (trials + 1)
    weights[mode] = 1.0
    odds = success_probability / (1.0 - success_probability)
    for count in range(mode + 1, trials + 1):
        weights[count] = (
            weights[count - 1]
            * (trials - count + 1)
            / count
            * odds
        )
    inverse_odds = 1.0 / odds
    for count in range(mode - 1, -1, -1):
        weights[count] = (
            weights[count + 1]
            * (count + 1)
            / (trials - count)
            * inverse_odds
        )
    normalizer = sum(weights)
    probabilities = [weight / normalizer for weight in weights]

    tail_probability = [0.0] * (trials + 1)
    tail_first = [0.0] * (trials + 1)
    tail_second = [0.0] * (trials + 1)
    running_probability = 0.0
    running_first = 0.0
    running_second = 0.0
    for count in range(trials, -1, -1):
        probability = probabilities[count]
        running_probability += probability
        running_first += count * probability
        running_second += count * count * probability
        tail_probability[count] = running_probability
        tail_first[count] = running_first
        tail_second[count] = running_second
    return (
        tuple(tail_probability),
        tuple(tail_first),
        tuple(tail_second),
    )


def bentkus_binomial_p2_bound(
    threshold: float,
    *,
    trials: int,
    success_probability: float,
) -> float:
    """Compute the Bentkus--Pinelis ``P2`` bound for a binomial variable.

    This is the computable bound in Proposition 2 of Kuchibhotla and Zheng
    (ICML 2021, supplement C). The minimizer is analytic on every interval
    between adjacent points in the binomial support, so enumerating those
    candidates is exact apart from conservative floating-point rounding.
    """

    tails, first_moments, second_moments = _binomial_tail_moments(
        trials, success_probability
    )
    mean = trials * success_probability
    if threshold <= mean:
        return 1.0
    if threshold > trials:
        return 0.0
    if threshold == trials:
        return min(1.0, nextafter(tails[trials], 1.0))

    best = 1.0

    def consider(candidate: float, tail_index: int) -> None:
        nonlocal best
        if not candidate < threshold:
            return
        probability = tails[tail_index]
        first = first_moments[tail_index]
        second = second_moments[tail_index]
        numerator = second - 2.0 * candidate * first + candidate**2 * probability
        denominator = (threshold - candidate) ** 2
        if denominator <= 0.0:
            return
        best = min(best, max(0.0, numerator / denominator))

    variance = trials * success_probability * (1.0 - success_probability)
    negative_candidate = mean - variance / (threshold - mean)
    if negative_candidate <= 0.0:
        consider(negative_candidate, 0)
    consider(0.0, 0)

    for tail_index in range(1, trials + 1):
        lower_endpoint = float(tail_index - 1)
        upper_endpoint = float(tail_index)
        consider(lower_endpoint, tail_index)
        if upper_endpoint < threshold:
            consider(upper_endpoint, tail_index)
        denominator = (
            first_moments[tail_index]
            - threshold * tails[tail_index]
        )
        if abs(denominator) <= 1e-18:
            continue
        candidate = (
            second_moments[tail_index]
            - threshold * first_moments[tail_index]
        ) / denominator
        if lower_endpoint <= candidate <= upper_endpoint:
            consider(candidate, tail_index)

    return min(1.0, nextafter(best, 1.0))


@lru_cache(maxsize=None)
def _bentkus_fixed_horizon_boundary(
    horizon: int,
    *,
    alpha: float,
    observation_lower: float,
    observation_upper: float,
    null_mean: float,
) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not observation_lower < null_mean < observation_upper:
        raise ValueError("null_mean must lie strictly inside the observation bounds")

    width = observation_upper - observation_lower
    success_probability = (null_mean - observation_lower) / width
    maximum_centered_sum = horizon * (observation_upper - null_mean)
    extreme_probability = success_probability**horizon
    if alpha < extreme_probability:
        return nextafter(maximum_centered_sum + width, float("inf"))

    lower_binomial = horizon * success_probability
    upper_binomial = float(horizon)
    for _iteration in range(80):
        midpoint = (lower_binomial + upper_binomial) / 2.0
        probability_bound = bentkus_binomial_p2_bound(
            midpoint,
            trials=horizon,
            success_probability=success_probability,
        )
        if probability_bound <= alpha:
            upper_binomial = midpoint
        else:
            lower_binomial = midpoint
    centered_boundary = width * (
        upper_binomial - horizon * success_probability
    )
    return nextafter(centered_boundary, float("inf"))


def bentkus_stitched_boundary(
    observations: int,
    *,
    alpha: float,
    observation_lower: float,
    observation_upper: float,
    null_mean: float,
    spacing: float = 2.0,
) -> float:
    """Piecewise-constant anytime boundary from maximal Bentkus stitching.

    Epoch ``k`` receives error ``alpha / ((k + 1) * (k + 2))``; the
    reciprocal stitching weights telescope to one. This comparator assumes
    independent bounded observations within each task. It must not be used as
    support for the primary router's stronger conditional-mean claim.
    """

    if observations <= 0:
        return float("inf")
    if spacing <= 1.0:
        raise ValueError("spacing must exceed one")
    epoch = 0
    while not (
        ceil(spacing**epoch)
        <= observations
        <= floor(spacing ** (epoch + 1))
    ):
        epoch += 1
    horizon = floor(spacing ** (epoch + 1))
    stitching_weight = (epoch + 1) * (epoch + 2)
    return _bentkus_fixed_horizon_boundary(
        horizon,
        alpha=alpha / stitching_weight,
        observation_lower=observation_lower,
        observation_upper=observation_upper,
        null_mean=null_mean,
    )


@dataclass(frozen=True)
class ConfidenceSequenceEvidence:
    task: str
    decision: RouteDecision
    observations: int
    mean_difference: float
    lower_confidence_bound: float
    upper_confidence_bound: float


class AlphaSpendingFamilywiseRouter:
    """Successive-elimination router with explicit time-uniform bounds."""

    def __init__(self, plan: AnytimeFamilywisePlan):
        self.plan = plan
        self._counts = {task: 0 for task in plan.task_names}
        self._sums = {task: 0.0 for task in plan.task_names}
        self._decisions = {
            task: RouteDecision.UNDECIDED for task in plan.task_names
        }

    def _validate_task(self, task: str) -> None:
        if task not in self._counts:
            raise KeyError(f"unknown proposal task: {task}")

    def evidence(self, task: str) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        count = self._counts[task]
        mean = self._sums[task] / count if count else 0.0
        lower_radius = alpha_spending_radius(
            count,
            alpha=self.plan.acceptance_alpha(task),
            width=self.plan.observation_width,
        )
        upper_radius = alpha_spending_radius(
            count,
            alpha=self.plan.futility_alpha(task),
            width=self.plan.observation_width,
        )
        return ConfidenceSequenceEvidence(
            task=task,
            decision=self._decisions[task],
            observations=count,
            mean_difference=mean,
            lower_confidence_bound=mean - lower_radius,
            upper_confidence_bound=mean + upper_radius,
        )

    def update(self, task: str, paired_score_difference: float) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        if self._decisions[task] is not RouteDecision.UNDECIDED:
            raise RuntimeError(f"task {task} already has a terminal route decision")
        value = float(paired_score_difference)
        if not self.plan.observation_lower <= value <= self.plan.observation_upper:
            raise ValueError("observation lies outside the declared bounds")
        self._counts[task] += 1
        self._sums[task] += value
        evidence = self.evidence(task)
        if (
            evidence.observations >= self.plan.minimum_observations
            and evidence.lower_confidence_bound > self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.ACCEPT_CANDIDATE
        elif (
            evidence.observations >= self.plan.minimum_observations
            and evidence.upper_confidence_bound < self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.REJECT_TO_FALLBACK
        elif evidence.observations >= self.plan.maximum_observations_per_task:
            self._decisions[task] = RouteDecision.BUDGET_EXHAUSTED
        return self.evidence(task)

    def next_task(self) -> str | None:
        unresolved = [
            task
            for task in self.plan.task_names
            if self._decisions[task] is RouteDecision.UNDECIDED
        ]
        if not unresolved:
            return None

        def priority(task: str) -> tuple[float, str]:
            evidence = self.evidence(task)
            width = (
                evidence.upper_confidence_bound
                - evidence.lower_confidence_bound
            )
            return (-width, task)

        return min(unresolved, key=priority)

    def decisions(self) -> dict[str, ConfidenceSequenceEvidence]:
        return {task: self.evidence(task) for task in self.plan.task_names}

    def total_observations(self) -> int:
        return sum(self._counts.values())


class BentkusStitchedFamilywiseRouter:
    """IID-only racing comparator using stitched maximal Bentkus bounds."""

    def __init__(self, plan: AnytimeFamilywisePlan):
        self.plan = plan
        self._counts = {task: 0 for task in plan.task_names}
        self._sums = {task: 0.0 for task in plan.task_names}
        self._decisions = {
            task: RouteDecision.UNDECIDED for task in plan.task_names
        }
        self._acceptance_alphas = {
            task: plan.acceptance_alpha(task) for task in plan.task_names
        }
        self._futility_alphas = {
            task: plan.futility_alpha(task) for task in plan.task_names
        }
        self._acceptance_boundaries = {task: {} for task in plan.task_names}
        self._futility_boundaries = {task: {} for task in plan.task_names}

    def _validate_task(self, task: str) -> None:
        if task not in self._counts:
            raise KeyError(f"unknown proposal task: {task}")

    def _boundary(self, task: str, observations: int, *, acceptance: bool) -> float:
        cache = (
            self._acceptance_boundaries[task]
            if acceptance
            else self._futility_boundaries[task]
        )
        if observations not in cache:
            if acceptance:
                cache[observations] = bentkus_stitched_boundary(
                    observations,
                    alpha=self._acceptance_alphas[task],
                    observation_lower=self.plan.observation_lower,
                    observation_upper=self.plan.observation_upper,
                    null_mean=self.plan.effect_margin,
                )
            else:
                cache[observations] = bentkus_stitched_boundary(
                    observations,
                    alpha=self._futility_alphas[task],
                    observation_lower=-self.plan.observation_upper,
                    observation_upper=-self.plan.observation_lower,
                    null_mean=-self.plan.effect_margin,
                )
        return cache[observations]

    def evidence(self, task: str) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        count = self._counts[task]
        mean = self._sums[task] / count if count else 0.0
        acceptance_boundary = self._boundary(task, count, acceptance=True)
        futility_boundary = self._boundary(task, count, acceptance=False)
        return ConfidenceSequenceEvidence(
            task=task,
            decision=self._decisions[task],
            observations=count,
            mean_difference=mean,
            lower_confidence_bound=(
                mean - acceptance_boundary / count
                if count
                else float("-inf")
            ),
            upper_confidence_bound=(
                mean + futility_boundary / count
                if count
                else float("inf")
            ),
        )

    def update(self, task: str, paired_score_difference: float) -> ConfidenceSequenceEvidence:
        self._validate_task(task)
        if self._decisions[task] is not RouteDecision.UNDECIDED:
            raise RuntimeError(f"task {task} already has a terminal route decision")
        value = float(paired_score_difference)
        if not self.plan.observation_lower <= value <= self.plan.observation_upper:
            raise ValueError("observation lies outside the declared bounds")
        self._counts[task] += 1
        self._sums[task] += value
        evidence = self.evidence(task)
        if (
            evidence.observations >= self.plan.minimum_observations
            and evidence.lower_confidence_bound > self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.ACCEPT_CANDIDATE
        elif (
            evidence.observations >= self.plan.minimum_observations
            and evidence.upper_confidence_bound < self.plan.effect_margin
        ):
            self._decisions[task] = RouteDecision.REJECT_TO_FALLBACK
        elif evidence.observations >= self.plan.maximum_observations_per_task:
            self._decisions[task] = RouteDecision.BUDGET_EXHAUSTED
        return self.evidence(task)

    def next_task(self) -> str | None:
        unresolved = [
            task
            for task in self.plan.task_names
            if self._decisions[task] is RouteDecision.UNDECIDED
        ]
        if not unresolved:
            return None

        def priority(task: str) -> tuple[float, str]:
            evidence = self.evidence(task)
            width = (
                evidence.upper_confidence_bound
                - evidence.lower_confidence_bound
            )
            return (-width, task)

        return min(unresolved, key=priority)

    def decisions(self) -> dict[str, ConfidenceSequenceEvidence]:
        return {task: self.evidence(task) for task in self.plan.task_names}

    def total_observations(self) -> int:
        return sum(self._counts.values())
