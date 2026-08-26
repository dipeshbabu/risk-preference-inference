"""Post-lock VDC-Conservative, VDC-Efficient, and guided allocation study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from riskshiftbench.canonical_metrics import (
    canonical_update_metrics,
    paired_effects,
)
from riskshiftbench.experiments.real_agent_validation import (
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
)
from riskshiftbench.experiments.vdc_efficient import (
    efficient_family_decisions,
    normalized_boundary_distance,
)
from riskshiftbench.experiments.verifier_drift_control import VDCPlan, decide_vdc


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "riskshiftbench" / "configs" / "real_agent_high_budget_extension_v1.json"
PARENT_RAW = ROOT / "riskshiftbench" / "artifacts" / "real_agent_high_budget_v1" / "raw_trajectories.json"
EXTENSION_RAW = ROOT / "riskshiftbench" / "artifacts" / "real_agent_high_budget_extension_v1" / "raw_trajectories.json"
RESULTS = ROOT / "riskshiftbench" / "artifacts" / "real_agent_high_budget_extension_v1" / "results.json"
OUTPUT = ROOT / "riskshiftbench" / "artifacts" / "real_agent_high_budget_extension_v1" / "vdc_efficiency_study.json"


BUDGETS = (500, 1_000, 1_500, 2_000, 2_500, 5_000)


def source_sha256() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(
        (ROOT / "riskshiftbench" / "experiments" / "vdc_efficient.py").read_bytes()
    )
    return digest.hexdigest()


def _load_hashed(path: Path, digest_key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != digest_key}
    if payload[digest_key] != canonical_sha256(body):
        raise RuntimeError(f"digest mismatch: {path}")
    return payload


def _task_arrays(config: dict, parent: dict, extension: dict) -> dict[str, dict]:
    trajectories = parent["trajectories"] + extension["trajectories"]
    output = {}
    for pair in config["agent_pairs"]:
        pair_id, family = pair["id"], pair["family"]
        temperature = float(config["confidence"][family]["temperature"])
        thresholds = [
            float(value) for value in config["confidence"][family]["thresholds"]
        ]
        fallback = [
            calibrate_episode(row, temperature)
            for row in trajectories
            if row["run_id"] == f"{family}-fallback-shared"
        ]
        candidate = [
            calibrate_episode(row, temperature)
            for row in trajectories
            if row["run_id"] == pair_id and row["role"] == "candidate"
        ]
        effects = paired_effects(
            [row["score"] for row in fallback],
            [row["score"] for row in candidate],
        )
        output[pair_id] = {
            "effects": list(effects),
            "candidate": {
                threshold: [high_confidence_failure(row, threshold) for row in candidate]
                for threshold in thresholds
            },
            "fallback": {
                threshold: [high_confidence_failure(row, threshold) for row in fallback]
                for threshold in thresholds
            },
        }
    return output


def _slice_task(task: dict, start: int, count: int) -> tuple:
    end = start + count
    return (
        task["effects"][start:end],
        {key: value[start:end] for key, value in task["candidate"].items()},
        {key: value[start:end] for key, value in task["fallback"].items()},
    )


def _summarize(route: dict[str, str], config: dict, pilot_pairs: int) -> dict:
    promoted = {key for key, value in route.items() if value == "promote"}
    eligible = {key for key, value in config["reference_eligibility"].items() if value}
    ineligible = promoted - eligible
    return {
        "route": route,
        "promotions": len(promoted),
        "promoted_updates": sorted(promoted),
        "ineligible_promotions": len(ineligible),
        "eligible_recall": len(promoted & eligible) / len(eligible),
        "deployment_gain": sum(config["final_effects"][key] for key in promoted)
        / len(config["agent_pairs"]),
        "unresolved": sum(value == "unresolved" for value in route.values()),
        "recalibrations": sum(
            value == "recalibrate-verifier" for value in route.values()
        ),
        "pilot_pairs": pilot_pairs,
    }


def _conservative_route(tasks: dict[str, tuple], config: dict) -> dict[str, str]:
    plan = VDCPlan(
        confidence_thresholds=(0.1, 0.3, 0.5),
        family_tasks=6,
        declared_looks=1,
        family_alpha=0.05,
        mean_margin=0.02,
        material_loss_threshold=0.5,
        maximum_material_loss_probability=0.10,
        maximum_candidate_high_confidence_failure_probability=0.15,
        maximum_verifier_drift=0.05,
    )
    route = {}
    for task_id, (effects, candidate, fallback) in tasks.items():
        candidate_grid = {
            coverage: candidate[threshold]
            for coverage, threshold in zip((0.1, 0.3, 0.5), sorted(candidate))
        }
        fallback_grid = {
            coverage: fallback[threshold]
            for coverage, threshold in zip((0.1, 0.3, 0.5), sorted(fallback))
        }
        route[task_id] = decide_vdc(
            effects,
            candidate_grid,
            fallback_grid,
            plan,
            at_task_cap=True,
            legacy_recalibration_label=True,
        ).reason
    return route


def _efficient_route(tasks: dict[str, tuple], config: dict) -> dict[str, str]:
    decisions = efficient_family_decisions(
        tasks,
        family_alpha=0.05,
        mean_margin=0.02,
        material_loss_threshold=0.5,
        maximum_material_loss_probability=0.10,
        maximum_candidate_high_confidence_failure_probability=0.15,
        maximum_verifier_drift=0.05,
        confidence_unresolved_reason="recalibrate-verifier",
    )
    return {key: value.reason for key, value in decisions.items()}


def _plugin_routes(tasks: dict[str, tuple]) -> tuple[dict[str, str], dict[str, str]]:
    outcome = {}
    frozen = {}
    for task_id, (effects, candidate, fallback) in tasks.items():
        metrics = canonical_update_metrics(
            effects,
            candidate,
            fallback,
            material_loss_threshold=0.5,
        )
        outcome[task_id] = "promote" if (
            metrics.mean_effect > 0.02
            and metrics.material_loss_probability < 0.10
        ) else "retain"
        frozen[task_id] = "promote" if (
            metrics.mean_effect > 0.02
            and metrics.material_loss_probability < 0.10
            and metrics.candidate_high_confidence_failure < 0.15
            and metrics.verifier_drift_max < 0.05
        ) else "retain"
    return outcome, frozen


def _guided_allocation(
    arrays: dict[str, dict], config: dict, total_pilot_pairs: int
) -> dict:
    task_ids = sorted(arrays)
    guidance_counts = {key: 100 for key in task_ids}
    guidance_total = 1_800
    block = 100
    while sum(guidance_counts.values()) < guidance_total:
        scores = {}
        for key in task_ids:
            if guidance_counts[key] >= 500:
                continue
            task = _slice_task(arrays[key], 0, guidance_counts[key])
            scores[key] = normalized_boundary_distance(
                *task,
                mean_margin=0.02,
                maximum_material_loss_probability=0.10,
                maximum_candidate_high_confidence_failure_probability=0.15,
                maximum_verifier_drift=0.05,
            )
        selected = min(scores, key=lambda key: (scores[key], key))
        guidance_counts[selected] += block

    guidance_scores = {}
    for key in task_ids:
        guidance_scores[key] = normalized_boundary_distance(
            *_slice_task(arrays[key], 0, guidance_counts[key]),
            mean_margin=0.02,
            maximum_material_loss_probability=0.10,
            maximum_candidate_high_confidence_failure_probability=0.15,
            maximum_verifier_drift=0.05,
        )

    guidance_tasks = {
        key: _slice_task(arrays[key], 0, guidance_counts[key]) for key in task_ids
    }
    guidance_route = _conservative_route(guidance_tasks, config)
    resolved_guidance = {
        key for key, reason in guidance_route.items() if reason.startswith("retain-")
    }

    verification_total = total_pilot_pairs - guidance_total
    if verification_total < 200 * len(task_ids):
        raise ValueError("total pilot budget is too small for verification minima")
    verification_counts = {key: 200 for key in task_ids}
    while sum(verification_counts.values()) < verification_total:
        candidates = {
            key: guidance_scores[key] * math.sqrt(verification_counts[key] / 200)
            for key in task_ids
            if verification_counts[key] < 4_500 and key not in resolved_guidance
        }
        if not candidates:
            break
        selected = min(candidates, key=lambda key: (candidates[key], key))
        verification_counts[selected] += block
    verification_tasks = {
        key: _slice_task(arrays[key], 500, verification_counts[key])
        for key in task_ids
    }
    route = _efficient_route(verification_tasks, config)
    for key in resolved_guidance:
        if route[key] != "promote":
            route[key] = "retain-guidance"
    return {
        "total_pilot_pairs": total_pilot_pairs,
        "guidance_counts": guidance_counts,
        "guidance_scores": guidance_scores,
        "guidance_route": guidance_route,
        "resolved_guidance": sorted(resolved_guidance),
        "verification_counts": verification_counts,
        "summary": _summarize(
            route, config, total_pilot_pairs
        ),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = _load_hashed(PARENT_RAW, "raw_sha256")
    extension = _load_hashed(EXTENSION_RAW, "raw_sha256")
    cumulative = _load_hashed(RESULTS, "result_sha256")
    arrays = _task_arrays(config, parent, extension)
    budgets = []
    for budget in BUDGETS:
        tasks = {key: _slice_task(value, 0, budget) for key, value in arrays.items()}
        outcome, frozen = _plugin_routes(tasks)
        budgets.append(
            {
                "pairs_per_update": budget,
                "methods": {
                    "always_fallback": _summarize(
                        {key: "retain" for key in tasks}, config, 0
                    ),
                    "outcome_only": _summarize(
                        outcome, config, budget * len(tasks)
                    ),
                    "frozen_verifier": _summarize(
                        frozen, config, budget * len(tasks)
                    ),
                    "vdc_conservative": _summarize(
                        _conservative_route(tasks, config),
                        config,
                        budget * len(tasks),
                    ),
                    "vdc_efficient": _summarize(
                        _efficient_route(tasks, config),
                        config,
                        budget * len(tasks),
                    ),
                },
            }
        )
    payload = {
        "diagnostic_status": (
            "post-lock fixed-look VDC-Efficient and sample-split guided "
            "allocation study on locked cumulative trajectories"
        ),
        "source_sha256": source_sha256(),
        "cumulative_result_sha256": cumulative["result_sha256"],
        "budgets": budgets,
        "vdc_a_split": [
            _guided_allocation(arrays, config, total)
            for total in (6_000, 9_000, 12_000, 15_000)
        ],
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
