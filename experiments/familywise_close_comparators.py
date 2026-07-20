"""Close sequential-MHT comparators for the RiskShiftBench v2 router.

The aLTT comparator follows the published ICML 2025 construction: a bounded
aGRAPA e-process for every proposal, anytime p-values obtained from running
e-process maxima, Bonferroni FWER control, and epsilon-greedy acquisition.
The e-Holm comparator keeps the same task e-processes and acquisition policy
but applies the always-valid e-Holm closed test of Hartog and Lei (2026) to
the current e-values.  Both are development-only and consume no confirmation
artifact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import exp, isfinite, log, log1p


ALTT_EPSILON = 0.25
AGRAPA_TRUNCATION = 0.95


def altt_bonferroni_rejections(
    maximum_log_e_values: dict[str, float],
    *,
    familywise_alpha: float,
) -> set[str]:
    """Return the aLTT Bonferroni rejection set from anytime p-values."""

    if not maximum_log_e_values:
        raise ValueError("at least one e-process is required")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    threshold = log(len(maximum_log_e_values) / familywise_alpha)
    return {
        task
        for task, maximum_log_e in maximum_log_e_values.items()
        if maximum_log_e >= threshold
    }


def e_holm_rejections(
    log_e_values: dict[str, float],
    *,
    familywise_alpha: float,
) -> set[str]:
    """Return the e-Holm closed-testing rejection set in linear time.

    This is Theorem 4.2 of Hartog and Lei (2026).  Computation stays in the
    log domain for candidate e-values; only insignificant values, which are
    bounded above by ``1 / alpha``, are exponentiated.
    """

    if not log_e_values:
        raise ValueError("at least one e-value is required")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must lie in (0, 1)")
    base_threshold = 1.0 / familywise_alpha
    base_log_threshold = log(base_threshold)
    deficit = sum(
        base_threshold - exp(log_e)
        for log_e in log_e_values.values()
        if log_e < base_log_threshold
    )
    log_threshold = log(base_threshold + deficit)
    return {
        task for task, log_e in log_e_values.items() if log_e >= log_threshold
    }


@dataclass(frozen=True)
class CloseComparatorEvidence:
    task: str
    observations: int
    current_log_e: float
    maximum_log_e: float
    last_betting_fraction: float
    accepted: bool


class AgrapaFamilywiseRouter:
    """aGRAPA evidence with aLTT or e-Holm familywise selection.

    Observations are mapped to ``Y in [0, 1]`` and the task null is the
    conditional-mean statement ``E[Y_t | F_{t-1}] <= q``.  The nonnegative,
    predictable aGRAPA fraction makes

        product_t (1 + lambda_t (Y_t - q))

    a test supermartingale under that null.  The regularized mean and variance
    estimates are the standard half-observation/quarter-variance predictable
    estimates used in the bounded-mean betting reference.
    """

    def __init__(
        self,
        task_names: tuple[str, ...],
        *,
        familywise_alpha: float = 0.05,
        effect_margin: float = 0.0,
        observation_lower: float = -1.0,
        observation_upper: float = 1.0,
        maximum_observations_per_task: int = 200,
        selection_rule: str = "altt_bonferroni",
        epsilon: float = ALTT_EPSILON,
        truncation: float = AGRAPA_TRUNCATION,
        acquisition_seed: int = 0,
    ) -> None:
        if not task_names:
            raise ValueError("at least one task is required")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique")
        if not 0.0 < familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie in (0, 1)")
        if not observation_lower < effect_margin < observation_upper:
            raise ValueError("effect_margin must lie strictly within the bounds")
        if maximum_observations_per_task <= 0:
            raise ValueError("maximum_observations_per_task must be positive")
        if selection_rule not in {"altt_bonferroni", "e_holm"}:
            raise ValueError("selection_rule must be 'altt_bonferroni' or 'e_holm'")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if not 0.0 < truncation < 1.0:
            raise ValueError("truncation must lie in (0, 1)")

        self.task_names = task_names
        self.familywise_alpha = familywise_alpha
        self.effect_margin = effect_margin
        self.observation_lower = observation_lower
        self.observation_upper = observation_upper
        self.maximum_observations_per_task = maximum_observations_per_task
        self.selection_rule = selection_rule
        self.epsilon = epsilon
        self.truncation = truncation
        self._null_mean = (
            effect_margin - observation_lower
        ) / (observation_upper - observation_lower)
        self._rng = random.Random(acquisition_seed)
        self._observations = {task: [] for task in task_names}
        self._normalized_sums = {task: 0.0 for task in task_names}
        self._variance_residual_sums = {task: 0.0 for task in task_names}
        self._current_log_e = {task: 0.0 for task in task_names}
        self._maximum_log_e = {task: 0.0 for task in task_names}
        self._last_betting_fraction = {task: 0.0 for task in task_names}
        self._accepted: set[str] = set()

    def _validate_observation(self, value: float) -> float:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("paired score differences must be finite")
        if not self.observation_lower <= numeric <= self.observation_upper:
            raise ValueError("paired score difference lies outside the frozen bounds")
        return numeric

    def _predictable_agrapa_fraction(self, task: str) -> float:
        count = len(self._observations[task])
        estimated_mean = (
            0.5 + self._normalized_sums[task]
        ) / (count + 1.0)
        estimated_variance = (
            0.25 + self._variance_residual_sums[task]
        ) / (count + 1.0)
        gap = estimated_mean - self._null_mean
        unconstrained = gap / (estimated_variance + gap * gap)
        return max(
            0.0,
            min(self.truncation / self._null_mean, unconstrained),
        )

    def _refresh_rejections(self) -> None:
        if self.selection_rule == "altt_bonferroni":
            rejected = altt_bonferroni_rejections(
                self._maximum_log_e,
                familywise_alpha=self.familywise_alpha,
            )
        else:
            rejected = e_holm_rejections(
                self._current_log_e,
                familywise_alpha=self.familywise_alpha,
            )
        self._accepted.update(rejected)

    def update(self, task: str, paired_score_difference: float) -> CloseComparatorEvidence:
        if task not in self._observations:
            raise KeyError(f"unknown task: {task}")
        if task in self._accepted:
            raise RuntimeError(f"task {task} has already been accepted")
        if len(self._observations[task]) >= self.maximum_observations_per_task:
            raise RuntimeError(f"task {task} has exhausted its observation cap")

        value = self._validate_observation(paired_score_difference)
        normalized = (
            value - self.observation_lower
        ) / (self.observation_upper - self.observation_lower)
        fraction = self._predictable_agrapa_fraction(task)
        self._last_betting_fraction[task] = fraction
        self._current_log_e[task] += log1p(
            fraction * (normalized - self._null_mean)
        )
        self._maximum_log_e[task] = max(
            self._maximum_log_e[task], self._current_log_e[task]
        )

        self._observations[task].append(value)
        self._normalized_sums[task] += normalized
        count = len(self._observations[task])
        updated_mean = (0.5 + self._normalized_sums[task]) / (count + 1.0)
        self._variance_residual_sums[task] += (normalized - updated_mean) ** 2
        self._refresh_rejections()
        return self.evidence(task)

    def next_task(self) -> str | None:
        eligible = [
            task
            for task in self.task_names
            if task not in self._accepted
            and len(self._observations[task]) < self.maximum_observations_per_task
        ]
        if not eligible:
            return None
        ordered = sorted(eligible)
        if self._rng.random() < self.epsilon:
            return ordered[self._rng.randrange(len(ordered))]
        return min(
            ordered,
            key=lambda task: (-self._current_log_e[task], task),
        )

    def evidence(self, task: str) -> CloseComparatorEvidence:
        if task not in self._observations:
            raise KeyError(f"unknown task: {task}")
        return CloseComparatorEvidence(
            task=task,
            observations=len(self._observations[task]),
            current_log_e=self._current_log_e[task],
            maximum_log_e=self._maximum_log_e[task],
            last_betting_fraction=self._last_betting_fraction[task],
            accepted=task in self._accepted,
        )

    def accepted_tasks(self) -> tuple[str, ...]:
        return tuple(task for task in self.task_names if task in self._accepted)

    def total_observations(self) -> int:
        return sum(len(values) for values in self._observations.values())

