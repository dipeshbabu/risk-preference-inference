"""Execute the controlled 12-update verifier-drift mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from math import sqrt
from statistics import fmean, stdev

from riskshiftbench.experiments.verifier_drift_control import (
    VDCAction,
    VDCPlan,
    decide_vdc,
)


CONFIG = Path("riskshiftbench/configs/controlled_verifier_drift_v1.json")
OUTPUT = Path("riskshiftbench/artifacts/controlled_verifier_drift_v1/results.json")


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_config(config: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-controlled-verifier-drift-v1":
        raise RuntimeError("unexpected controlled-study protocol")
    tasks = config["tasks"]
    if len(tasks) != 12 or len({row["id"] for row in tasks}) != 12:
        raise RuntimeError("the controlled study requires 12 unique updates")
    domains = Counter(row["domain"] for row in tasks)
    regimes = Counter(row["regime"] for row in tasks)
    if set(domains.values()) != {4} or len(domains) != 3:
        raise RuntimeError("each controlled domain must contain four updates")
    if set(regimes.values()) != {3} or len(regimes) != 4:
        raise RuntimeError("each controlled regime must contain three updates")
    if config["reference_mode"] != "known_generator_population":
        raise RuntimeError("controlled reference labels must use generator truth")
    if not config["shared_behavior_uniform_within_candidate_fallback_pair"]:
        raise RuntimeError("candidate and fallback outcomes must be paired")
    if int(config["pilot_pairs_per_task"]) != 240:
        raise RuntimeError("the controlled pilot uses 240 pairs per task")
    if int(config["pilot_streams"]) < 100:
        raise RuntimeError("the controlled pilot requires at least 100 streams")


def population_record(task: dict, config: dict) -> dict:
    fallback_success = float(task["fallback_success"])
    candidate_success = float(task["candidate_success"])
    fallback_hcf = (
        (1.0 - fallback_success)
        * float(config["fallback_failure_high_confidence_probability"])
    )
    candidate_hcf = (
        (1.0 - candidate_success)
        * float(task["candidate_failure_high_confidence_probability"])
    )
    mean = candidate_success - fallback_success
    downside = max(fallback_success - candidate_success, 0.0)
    drift = candidate_hcf - fallback_hcf
    eligible = (
        mean > float(config["mean_margin"])
        and downside < float(config["maximum_material_loss_probability"])
        and candidate_hcf
        < float(config["maximum_candidate_high_confidence_failure_probability"])
        and drift < float(config["maximum_verifier_drift"])
    )
    failed_conditions = []
    if mean <= float(config["mean_margin"]):
        failed_conditions.append("mean")
    if downside >= float(config["maximum_material_loss_probability"]):
        failed_conditions.append("downside")
    if candidate_hcf >= float(
        config["maximum_candidate_high_confidence_failure_probability"]
    ):
        failed_conditions.append("absolute_confidence_risk")
    if drift >= float(config["maximum_verifier_drift"]):
        failed_conditions.append("verifier_drift")
    return {
        "task": task["id"],
        "domain": task["domain"],
        "intended_regime": task["regime"],
        "mean_effect": mean,
        "material_loss_probability": downside,
        "candidate_high_confidence_failure_probability": candidate_hcf,
        "fallback_high_confidence_failure_probability": fallback_hcf,
        "verifier_drift": drift,
        "eligible": eligible,
        "failed_conditions": failed_conditions,
    }


def simulate_pilot(task: dict, config: dict, seed: int) -> dict:
    rng = random.Random(seed)
    effects: list[float] = []
    candidate_failures = {
        float(threshold): [] for threshold in config["confidence_thresholds"]
    }
    fallback_failures = {
        float(threshold): [] for threshold in config["confidence_thresholds"]
    }
    high_confidence = float(config["confidence_values"]["high"])
    low_confidence = float(config["confidence_values"]["low"])
    count = int(config["pilot_pairs_per_task"])
    for _ in range(count):
        shared_behavior = rng.random()
        fallback_success = shared_behavior < float(task["fallback_success"])
        candidate_success = shared_behavior < float(task["candidate_success"])
        effects.append(float(candidate_success) - float(fallback_success))

        candidate_confidence = high_confidence if (
            not candidate_success
            and rng.random()
            < float(task["candidate_failure_high_confidence_probability"])
        ) else low_confidence
        fallback_confidence = high_confidence if (
            not fallback_success
            and rng.random()
            < float(config["fallback_failure_high_confidence_probability"])
        ) else low_confidence
        for threshold in candidate_failures:
            candidate_failures[threshold].append(
                not candidate_success and candidate_confidence >= threshold
            )
            fallback_failures[threshold].append(
                not fallback_success and fallback_confidence >= threshold
            )
    return {
        "effects": effects,
        "candidate_high_confidence_failures": candidate_failures,
        "fallback_high_confidence_failures": fallback_failures,
    }


def _plugin_decision(pilot: dict, config: dict, *, absolute_risk: bool) -> str:
    effects = pilot["effects"]
    mean = fmean(effects)
    downside = fmean(
        value < -float(config["material_loss_threshold"]) for value in effects
    )
    if mean <= float(config["mean_margin"]):
        return "retain-utility"
    if downside >= float(config["maximum_material_loss_probability"]):
        return "retain-downside"
    if absolute_risk:
        risk = max(
            fmean(values)
            for values in pilot["candidate_high_confidence_failures"].values()
        )
        if risk >= float(
            config["maximum_candidate_high_confidence_failure_probability"]
        ):
            return "retain-confidence-risk"
    return "promote"


def _vdc_plan(config: dict) -> VDCPlan:
    return VDCPlan(
        confidence_thresholds=tuple(float(x) for x in config["confidence_thresholds"]),
        family_tasks=len(config["tasks"]),
        declared_looks=1,
        family_alpha=float(config["family_alpha"]),
        mean_margin=float(config["mean_margin"]),
        material_loss_threshold=float(config["material_loss_threshold"]),
        maximum_material_loss_probability=float(
            config["maximum_material_loss_probability"]
        ),
        maximum_candidate_high_confidence_failure_probability=float(
            config["maximum_candidate_high_confidence_failure_probability"]
        ),
        maximum_verifier_drift=float(config["maximum_verifier_drift"]),
    )


def evaluate_route(decisions: dict[str, str], population: dict[str, dict]) -> dict:
    promoted = [task for task, reason in decisions.items() if reason == "promote"]
    eligible = [task for task in promoted if population[task]["eligible"]]
    ineligible = [task for task in promoted if not population[task]["eligible"]]
    actions = Counter(
        "recalibrate" if reason == "recalibrate-verifier"
        else "deploy" if reason == "promote"
        else "unresolved" if reason == "unresolved"
        else "retain"
        for reason in decisions.values()
    )
    return {
        "deployment_gain": sum(population[task]["mean_effect"] for task in promoted)
        / len(population),
        "promotions": len(promoted),
        "eligible_promotions": len(eligible),
        "ineligible_promotions": len(ineligible),
        "any_ineligible_promotion": bool(ineligible),
        "actions": dict(sorted(actions.items())),
        "decisions": decisions,
    }


def mean_interval(values: list[float]) -> list[float]:
    center = fmean(values)
    if len(values) < 2 or max(values) == min(values):
        return [center, center]
    half = 1.972 * stdev(values) / sqrt(len(values))
    return [center - half, center + half]


def wilson_interval(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    half = z * sqrt(
        probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return [lower, upper]


def run(config: dict) -> dict:
    validate_config(config)
    population_rows = [population_record(task, config) for task in config["tasks"]]
    population = {row["task"]: row for row in population_rows}
    methods = {name: [] for name in ("outcome_only", "frozen_verifier", "vdc")}
    vdc_plan = _vdc_plan(config)
    for stream in range(int(config["pilot_streams"])):
        routes = {name: {} for name in methods}
        for task_index, task in enumerate(config["tasks"]):
            seed = int(config["pilot_seed_start"]) + stream * 1009 + task_index * 37
            pilot = simulate_pilot(task, config, seed)
            routes["outcome_only"][task["id"]] = _plugin_decision(
                pilot, config, absolute_risk=False
            )
            routes["frozen_verifier"][task["id"]] = _plugin_decision(
                pilot, config, absolute_risk=True
            )
            vdc = decide_vdc(
                pilot["effects"],
                pilot["candidate_high_confidence_failures"],
                pilot["fallback_high_confidence_failures"],
                vdc_plan,
                at_task_cap=True,
                legacy_recalibration_label=True,
            )
            routes["vdc"][task["id"]] = vdc.reason
        for method, decisions in routes.items():
            methods[method].append(evaluate_route(decisions, population))

    aggregate = {}
    for method, rows in methods.items():
        unsafe_events = sum(row["any_ineligible_promotion"] for row in rows)
        aggregate[method] = {
            "mean_deployment_gain": fmean(row["deployment_gain"] for row in rows),
            "deployment_gain_t_interval_95": mean_interval(
                [row["deployment_gain"] for row in rows]
            ),
            "mean_promotions": fmean(row["promotions"] for row in rows),
            "mean_eligible_promotions": fmean(
                row["eligible_promotions"] for row in rows
            ),
            "mean_ineligible_promotions": fmean(
                row["ineligible_promotions"] for row in rows
            ),
            "probability_any_ineligible_promotion": fmean(
                row["any_ineligible_promotion"] for row in rows
            ),
            "any_ineligible_promotion_wilson_interval_95": wilson_interval(
                unsafe_events, len(rows)
            ),
            "mean_actions": {
                action: fmean(row["actions"].get(action, 0) for row in rows)
                for action in ("deploy", "recalibrate", "retain", "unresolved")
            },
        }
    payload = {
        "protocol_id": config["protocol_id"],
        "evidential_status": config["evidential_status"],
        "config_sha256": canonical_sha256(config),
        "tasks": population_rows,
        "reference_category_counts": dict(
            sorted(Counter(row["intended_regime"] for row in population_rows).items())
        ),
        "population_eligible_tasks": sum(row["eligible"] for row in population_rows),
        "pilot_streams": int(config["pilot_streams"]),
        "pilot_pairs_per_task": int(config["pilot_pairs_per_task"]),
        "methods": aggregate,
        "trials": methods,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
