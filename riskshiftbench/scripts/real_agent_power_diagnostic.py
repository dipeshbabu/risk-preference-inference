"""Post-lock power and drift-decomposition diagnostics for the real-agent run.

This script uses the observed reference pool as an empirical population.  Its
Monte Carlo streams are independent conditional on that pool, but the analysis
is post-lock and does not estimate generalization beyond the observed task
generator.  It is a power diagnostic, not new confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import beta

from riskshiftbench.experiments.real_agent_validation import (
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
    load_locked_manifest,
    load_raw,
)
from riskshiftbench.experiments.verifier_drift_control import VDCPlan


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "riskshiftbench" / "configs" / "real_agent_validation_v1.json"
MANIFEST = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "task_manifest.json"
RAW = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "raw_trajectories.json"
RESULTS = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "results_amended_v2.json"
OUTPUT = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "postlock_power_diagnostic.json"


BUDGETS = (50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000)
SIMULATIONS = 500
SEED = 924_001


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_amended() -> dict:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(body):
        raise RuntimeError("amended result digest mismatch")
    return payload


def _cp_bounds(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha, successes + 1, trials - successes)
    )
    return lower, upper


def _fast_decision(
    counts: np.ndarray, patterns: np.ndarray, observations: int, plan: VDCPlan
) -> str:
    estimates = counts @ patterns / observations
    effect = float(estimates[0])
    mean_radius = 2.0 * math.sqrt(math.log(1.0 / plan.tail_alpha) / (2.0 * observations))
    mean_lower, mean_upper = effect - mean_radius, effect + mean_radius
    downside_count = int(round(float(counts @ patterns[:, 1])))
    downside_lower, downside_upper = _cp_bounds(
        downside_count, observations, plan.tail_alpha
    )

    candidate_intervals = []
    drift_intervals = []
    threshold_count = len(plan.confidence_thresholds)
    for index in range(threshold_count):
        candidate_count = int(round(float(counts @ patterns[:, 2 + index])))
        fallback_count = int(
            round(float(counts @ patterns[:, 2 + threshold_count + index]))
        )
        c_lower, c_upper = _cp_bounds(candidate_count, observations, plan.tail_alpha)
        f_lower, f_upper = _cp_bounds(fallback_count, observations, plan.tail_alpha)
        candidate_intervals.append((c_lower, c_upper))
        drift_intervals.append((c_lower - f_upper, c_upper - f_lower))

    candidate_lower = max(row[0] for row in candidate_intervals)
    candidate_upper = max(row[1] for row in candidate_intervals)
    drift_lower = max(row[0] for row in drift_intervals)
    drift_upper = max(row[1] for row in drift_intervals)

    if mean_upper <= plan.mean_margin:
        return "retain"
    if downside_lower >= plan.maximum_material_loss_probability:
        return "retain"
    if candidate_lower >= plan.maximum_candidate_high_confidence_failure_probability:
        return "retain"
    if drift_lower >= plan.maximum_verifier_drift:
        return "retain"
    utility_passes = mean_lower > plan.mean_margin
    downside_passes = downside_upper < plan.maximum_material_loss_probability
    candidate_passes = (
        candidate_upper < plan.maximum_candidate_high_confidence_failure_probability
    )
    drift_passes = drift_upper < plan.maximum_verifier_drift
    if utility_passes and downside_passes and candidate_passes and drift_passes:
        return "promote"
    if utility_passes and downside_passes:
        return "recalibrate"
    return "unresolved"


def _calibrated_reference(raw: dict, amended: dict) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for episode in raw["trajectories"]:
        grouped.setdefault((episode["pair_id"], episode["role"]), []).append(episode)
    output = {}
    for update in amended["updates"]:
        pair_id = update["pair_id"]
        temperature = float(update["fallback_temperature"])
        fallback = [
            calibrate_episode(row, temperature)
            for row in grouped[(pair_id, "fallback")]
            if row["split"] == "reference"
        ]
        candidate = [
            calibrate_episode(row, temperature)
            for row in grouped[(pair_id, "candidate")]
            if row["split"] == "reference"
        ]
        thresholds = [float(value) for value in update["thresholds"]]
        rows = []
        for f_row, c_row in zip(fallback, candidate):
            rows.append(
                (
                    c_row["score"] - f_row["score"],
                    float(c_row["score"] - f_row["score"] < -0.5),
                    *[float(high_confidence_failure(c_row, tau)) for tau in thresholds],
                    *[float(high_confidence_failure(f_row, tau)) for tau in thresholds],
                )
            )
        pattern_counts = Counter(rows)
        patterns = np.asarray(list(pattern_counts), dtype=float)
        probabilities = np.asarray(list(pattern_counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        output[pair_id] = {
            "patterns": patterns,
            "probabilities": probabilities,
            "fallback": fallback,
            "candidate": candidate,
            "thresholds": thresholds,
        }
    return output


def _episode_coverage(episode: dict, threshold: float) -> bool:
    return episode["episode_confidence"] >= threshold


def _commitment_failure(episode: dict) -> bool:
    return any(row["commitment_failure"] for row in episode["actions"])


def _workflow_decomposition(data: dict) -> list[dict]:
    rows = []
    for threshold, coverage_target in zip(data["thresholds"], (0.1, 0.3, 0.5)):
        record = {"coverage_target": coverage_target, "threshold": threshold}
        for role in ("fallback", "candidate"):
            episodes = data[role]
            coverage = np.mean([_episode_coverage(row, threshold) for row in episodes])
            h = np.mean([high_confidence_failure(row, threshold) for row in episodes])
            record[f"{role}_coverage"] = float(coverage)
            record[f"{role}_h"] = float(h)
            record[f"{role}_selective_commitment_risk"] = (
                float(h / coverage) if coverage else None
            )
            record[f"{role}_commitment_failure"] = float(
                np.mean([_commitment_failure(row) for row in episodes])
            )
        record["drift"] = record["candidate_h"] - record["fallback_h"]
        rows.append(record)
    return rows


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = load_locked_manifest(config, MANIFEST)
    raw = load_raw(config, manifest, RAW)
    amended = _load_amended()
    data = _calibrated_reference(raw, amended)
    coverages = tuple(float(value) for value in config["threshold_development_coverages"])
    plan = VDCPlan(
        confidence_thresholds=coverages,
        family_tasks=len(amended["updates"]),
        declared_looks=int(config["declared_looks"]),
        family_alpha=float(config["family_alpha"]),
        mean_margin=float(config["mean_margin"]),
        material_loss_threshold=float(config["material_loss_threshold"]),
        maximum_material_loss_probability=float(config["maximum_material_loss_probability"]),
        maximum_candidate_high_confidence_failure_probability=float(
            config["maximum_candidate_high_confidence_failure_probability"]
        ),
        maximum_verifier_drift=float(config["maximum_verifier_drift"]),
    )
    eligible = {
        row["pair_id"] for row in amended["updates"] if row["frozen_eligible"]
    }
    final_effect = {}
    for update in amended["updates"]:
        pair_id = update["pair_id"]
        fallback_final = [
            row["score"]
            for row in raw["trajectories"]
            if row["pair_id"] == pair_id
            and row["role"] == "fallback"
            and row["split"] == "final"
        ]
        candidate_final = [
            row["score"]
            for row in raw["trajectories"]
            if row["pair_id"] == pair_id
            and row["role"] == "candidate"
            and row["split"] == "final"
        ]
        final_effect[pair_id] = float(np.mean(candidate_final) - np.mean(fallback_final))
    rng = np.random.default_rng(SEED)
    curve = []
    for budget in BUDGETS:
        trials = []
        for _ in range(SIMULATIONS):
            routes = {}
            for update in amended["updates"]:
                pair_id = update["pair_id"]
                task = data[pair_id]
                counts = rng.multinomial(budget, task["probabilities"])
                routes[pair_id] = _fast_decision(
                    counts, task["patterns"], budget, plan
                )
            promoted = {key for key, value in routes.items() if value == "promote"}
            trials.append(
                {
                    "gain": sum(final_effect[key] for key in promoted)
                    / len(amended["updates"]),
                    "eligible_recall": len(promoted & eligible) / len(eligible),
                    "any_ineligible": bool(promoted - eligible),
                    "promotions": len(promoted),
                    "unresolved": sum(value == "unresolved" for value in routes.values()),
                    "recalibrate": sum(value == "recalibrate" for value in routes.values()),
                }
            )
        curve.append(
            {
                "pairs_per_update": budget,
                "total_pilot_pairs": budget * len(amended["updates"]),
                "mean_gain": float(np.mean([row["gain"] for row in trials])),
                "eligible_recall": float(
                    np.mean([row["eligible_recall"] for row in trials])
                ),
                "probability_any_ineligible": float(
                    np.mean([row["any_ineligible"] for row in trials])
                ),
                "mean_promotions": float(
                    np.mean([row["promotions"] for row in trials])
                ),
                "mean_unresolved": float(
                    np.mean([row["unresolved"] for row in trials])
                ),
                "mean_recalibrate": float(
                    np.mean([row["recalibrate"] for row in trials])
                ),
            }
        )

    sensitivity = []
    for kappa in (0.15, 0.20, 0.22, 0.25):
        for eta in (0.03, 0.05, 0.08):
            labels = []
            for update in amended["updates"]:
                metrics = update["frozen_reference"]
                passes_first_three = (
                    metrics["mean_effect"] > float(config["mean_margin"])
                    and metrics["material_loss_probability"]
                    < float(config["maximum_material_loss_probability"])
                    and metrics["candidate_high_confidence_failure"] < kappa
                )
                labels.append(
                    {
                        "pair_id": update["pair_id"],
                        "passes_first_three": passes_first_three,
                        "drift_only_failure": passes_first_three
                        and metrics["verifier_drift_max"] >= eta,
                    }
                )
            sensitivity.append(
                {
                    "kappa": kappa,
                    "eta": eta,
                    "eligible_without_drift": sum(
                        row["passes_first_three"] for row in labels
                    ),
                    "drift_only_failures": [
                        row["pair_id"] for row in labels if row["drift_only_failure"]
                    ],
                }
            )

    payload = {
        "diagnostic_status": (
            "post-lock Monte Carlo conditional on the observed reference pool; "
            "not preregistered and not independent external evidence"
        ),
        "source_sha256": source_sha256(),
        "amended_result_sha256": amended["result_sha256"],
        "seed": SEED,
        "simulations_per_budget": SIMULATIONS,
        "budget_curve": curve,
        "workflow_memory_decomposition": _workflow_decomposition(
            data["workflow-memory"]
        ),
        "threshold_sensitivity": sensitivity,
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
