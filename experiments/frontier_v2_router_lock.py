"""Freeze the outcome-free RiskShiftBench v2 confirmation proposal family.

The lock consumes only complete, audited development and calibration rehearsals.
It never instantiates an environment and never reads confirmation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
)
from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    all_tasks,
    canonical_sha256,
    domain_tasks,
    outcome_implementation_sha256,
    task_manifest_sha256,
    task_sha256,
)
from experiments.frontier_v2_rehearsal_audit import audit_split_coverage_payloads


EXPECTED_EPISODES_PER_POLICY = 20
FAMILYWISE_ALPHA = 0.05
FUTILITY_FAMILYWISE_ALPHA = 0.05
EFFECT_MARGIN = 0.0
MINIMUM_OBSERVATIONS = 2
MAXIMUM_OBSERVATIONS_PER_TASK = 200
GLOBAL_OBSERVATION_BUDGET = 3_960
FORCED_INITIAL_OBSERVATIONS = 2
RESOLUTION_FAMILYWISE_BETA = 0.05
MINIMUM_PLANNING_GAP = 0.05
MAXIMUM_PLANNING_GAP = 0.50


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_split_payloads(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    paths = tuple(sorted(root.glob(f"RSBv2-*-{split}-*-v0.json")))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    audit_split_coverage_payloads(
        payloads,
        split=split,
        expected_episodes_per_policy=EXPECTED_EPISODES_PER_POLICY,
    )
    artifacts = [
        {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    return payloads, artifacts


def paired_policy_effect(payload: dict, candidate: str, fallback: str) -> float:
    """Return the paired episode mean after checking the common seed schedule."""

    try:
        candidate_rows = payload["outcomes"][candidate]
        fallback_rows = payload["outcomes"][fallback]
    except KeyError as error:
        raise RuntimeError("proposal policy is absent from a rehearsal artifact") from error
    if len(candidate_rows) != len(fallback_rows) or not candidate_rows:
        raise RuntimeError("paired policy outcomes must be nonempty and equally sized")
    differences = []
    for candidate_row, fallback_row in zip(
        candidate_rows, fallback_rows, strict=True
    ):
        if candidate_row["seed"] != fallback_row["seed"]:
            raise RuntimeError("candidate and fallback outcomes do not use common seeds")
        differences.append(float(candidate_row["score"]) - float(fallback_row["score"]))
    return fmean(differences)


def select_domain_candidate(
    domain: str,
    development_payloads: list[dict],
) -> tuple[str, dict[str, dict[str, float] | float]]:
    """Select one candidate per domain using development outcomes only."""

    spec = DOMAIN_SPECS[domain]
    domain_payloads = sorted(
        (payload for payload in development_payloads if payload["domain"] == domain),
        key=lambda payload: payload["task"],
    )
    expected_names = {task.name for task in domain_tasks(domain, "development")}
    if {payload["task"] for payload in domain_payloads} != expected_names:
        raise RuntimeError(f"development coverage changed for {domain}")
    summaries: dict[str, dict[str, dict[str, float] | float]] = {}
    for candidate in spec.candidate_policies:
        task_effects = {
            payload["task"]: paired_policy_effect(
                payload, candidate, spec.fallback_policy
            )
            for payload in domain_payloads
        }
        summaries[candidate] = {
            "equal_task_mean_effect": fmean(task_effects.values()),
            "task_effects": task_effects,
        }
    selected = max(
        spec.candidate_policies,
        key=lambda candidate: (
            float(summaries[candidate]["equal_task_mean_effect"]),
            -spec.candidate_policies.index(candidate),
        ),
    )
    return selected, summaries


def planning_gap(observed_effect: float) -> float:
    """Apply the frozen calibration-only gap transform used for allocation."""

    return min(
        MAXIMUM_PLANNING_GAP,
        max(MINIMUM_PLANNING_GAP, abs(float(observed_effect) - EFFECT_MARGIN)),
    )


def build_router_lock(
    development_root: Path,
    calibration_root: Path,
) -> dict:
    development_payloads, development_artifacts = _load_split_payloads(
        development_root, "development"
    )
    calibration_payloads, calibration_artifacts = _load_split_payloads(
        calibration_root, "calibration"
    )
    calibration_by_task = {
        payload["task"]: payload for payload in calibration_payloads
    }

    selected_by_domain: dict[str, str] = {}
    selection_diagnostics = {}
    proposals = []
    for domain, spec in DOMAIN_SPECS.items():
        selected, diagnostics = select_domain_candidate(
            domain, development_payloads
        )
        selected_by_domain[domain] = selected
        selection_diagnostics[domain] = {
            "selected_candidate": selected,
            "candidates": diagnostics,
        }
        calibration_tasks = domain_tasks(domain, "calibration")
        confirmation_tasks = domain_tasks(domain, "confirmation")
        for index, confirmation_task in enumerate(confirmation_tasks):
            calibration_task = calibration_tasks[index]
            effect = paired_policy_effect(
                calibration_by_task[calibration_task.name],
                selected,
                spec.fallback_policy,
            )
            proposals.append(
                {
                    "task": confirmation_task.name,
                    "domain": domain,
                    "task_index": index,
                    "task_sha256": task_sha256(confirmation_task),
                    "fallback_policy": spec.fallback_policy,
                    "candidate_policy": selected,
                    "calibration_analogue": calibration_task.name,
                    "calibration_paired_mean_effect": effect,
                    "planning_effect_gap": planning_gap(effect),
                    "task_weight": 1.0 / len(all_tasks("confirmation")),
                }
            )

    proposals.sort(key=lambda proposal: proposal["task"])
    task_names = tuple(proposal["task"] for proposal in proposals)
    task_weights = tuple(
        (proposal["task"], proposal["task_weight"]) for proposal in proposals
    )
    planning_effect_gaps = tuple(
        (proposal["task"], proposal["planning_effect_gap"])
        for proposal in proposals
    )
    plan = AnytimeFamilywisePlan(
        task_names=task_names,
        familywise_alpha=FAMILYWISE_ALPHA,
        futility_familywise_alpha=FUTILITY_FAMILYWISE_ALPHA,
        effect_margin=EFFECT_MARGIN,
        observation_lower=-1.0,
        observation_upper=1.0,
        minimum_observations=MINIMUM_OBSERVATIONS,
        maximum_observations_per_task=MAXIMUM_OBSERVATIONS_PER_TASK,
        task_weights=task_weights,
        e_process_method="betting_mixture",
        planning_effect_gaps=planning_effect_gaps,
        resolution_familywise_beta=RESOLUTION_FAMILYWISE_BETA,
    )
    certified_targets = {
        target.task: target
        for target in AnytimeFamilywiseRouter(plan).certified_sample_targets()
    }
    for proposal in proposals:
        target = certified_targets[proposal["task"]]
        proposal["certified_resolution_target"] = {
            "required_observations": target.required_observations,
            "scheduled_observations": target.scheduled_observations,
            "betting_fraction": target.betting_fraction,
            "clipped_by_task_cap": target.clipped_by_task_cap,
        }

    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-router-lock-v1",
        "scope": (
            "Outcome-free proposal and allocation lock derived only from complete "
            "development and calibration rehearsals. Confirmation remains prohibited."
        ),
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "input_splits": {
            "development": {
                "task_manifest_sha256": task_manifest_sha256(
                    all_tasks("development")
                ),
                "artifacts": development_artifacts,
            },
            "calibration": {
                "task_manifest_sha256": task_manifest_sha256(
                    all_tasks("calibration")
                ),
                "artifacts": calibration_artifacts,
            },
        },
        "confirmation_task_manifest_sha256": task_manifest_sha256(
            all_tasks("confirmation")
        ),
        "proposal_family_size": len(proposals),
        "candidate_selection": {
            "split": "development",
            "unit": "one candidate per domain",
            "criterion": "largest equal-task paired mean score effect",
            "tie_break": "candidate order in the frozen domain specification",
            "selected_by_domain": selected_by_domain,
            "diagnostics": selection_diagnostics,
        },
        "planning_gap_rule": {
            "split": "calibration",
            "analogue": "same frozen within-domain task index",
            "formula": (
                "clip(abs(calibration_paired_mean_effect - effect_margin), "
                f"{MINIMUM_PLANNING_GAP}, {MAXIMUM_PLANNING_GAP})"
            ),
            "validity_dependency": (
                "None. Planning gaps change allocation only; acceptance uses the "
                "unchanged anytime-valid e-process thresholds."
            ),
        },
        "anytime_plan": {
            "familywise_alpha": plan.familywise_alpha,
            "futility_familywise_alpha": plan.futility_familywise_alpha,
            "effect_margin": plan.effect_margin,
            "observation_bounds": [
                plan.observation_lower,
                plan.observation_upper,
            ],
            "minimum_observations": plan.minimum_observations,
            "maximum_observations_per_task": plan.maximum_observations_per_task,
            "global_observation_budget": GLOBAL_OBSERVATION_BUDGET,
            "forced_initial_observations": FORCED_INITIAL_OBSERVATIONS,
            "e_process_method": plan.e_process_method,
            "betting_fraction_grid": list(plan.betting_fraction_grid),
            "allocation": "certified",
            "allocation_priority": (
                "smallest remaining scheduled quota, then smallest theoretical "
                "required-observation target, then task name"
            ),
            "certified_target_summary": {
                "required_observations_total": sum(
                    target.required_observations
                    for target in certified_targets.values()
                ),
                "scheduled_observations_total": sum(
                    target.scheduled_observations
                    for target in certified_targets.values()
                ),
                "clipped_task_count": sum(
                    target.clipped_by_task_cap
                    for target in certified_targets.values()
                ),
            },
            "resolution_familywise_beta": plan.resolution_familywise_beta,
            "multiplicity": "prespecified weighted Bonferroni e-value thresholds",
            "common_random_numbers": True,
        },
        "cost_accounting": {
            "pilot_unit": "one candidate-plus-fallback paired episode",
            "paired_observation_budget": GLOBAL_OBSERVATION_BUDGET,
            "policy_episode_budget": 2 * GLOBAL_OBSERVATION_BUDGET,
            "comparison_rule": (
                "Every allocation and gate comparison receives the same paired "
                "observation budget and task-indexed latent episode streams."
            ),
        },
        "proposals": proposals,
    }
    payload["router_lock_canonical_sha256"] = canonical_sha256(payload)
    return payload


def write_router_lock(
    development_root: Path,
    calibration_root: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite router lock: {output}")
    payload = build_router_lock(development_root, calibration_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def audit_router_lock(
    path: Path,
    *,
    development_root: Path,
    calibration_root: Path,
) -> dict:
    """Recompute the lock from its frozen inputs and require exact equality."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = build_router_lock(development_root, calibration_root)
    if payload != expected:
        raise RuntimeError("router lock differs from the deterministic frozen design")
    return {
        "protocol_id": payload["protocol_id"],
        "router_lock_canonical_sha256": payload[
            "router_lock_canonical_sha256"
        ],
        "proposal_family_size": payload["proposal_family_size"],
        "domain_count": len(payload["candidate_selection"]["selected_by_domain"]),
        "paired_observation_budget": payload["cost_accounting"][
            "paired_observation_budget"
        ],
        "clipped_target_count": payload["anytime_plan"][
            "certified_target_summary"
        ]["clipped_task_count"],
        "confirmation_task_manifest_sha256": payload[
            "confirmation_task_manifest_sha256"
        ],
        "outcome_implementation_sha256": payload[
            "outcome_implementation_sha256"
        ],
        "confirmation_execution_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-root",
        type=Path,
        default=Path(
            "artifacts/frontier_v2_full_rehearsal/"
            "development_portable_20ep_episode_lifetime"
        ),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path(
            "artifacts/frontier_v2_full_rehearsal/"
            "calibration_portable_20ep_episode_lifetime"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/frontier_v2_router_lock/router_lock.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_router_lock(
        args.development_root,
        args.calibration_root,
        args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
