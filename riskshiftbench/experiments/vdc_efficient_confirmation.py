"""Run the prospectively locked full-family VDC-Efficient confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import riskshiftbench.experiments.real_agent_high_budget_extension as cached_runner
from riskshiftbench.canonical_metrics import canonical_update_metrics, paired_effects
from riskshiftbench.experiments.real_agent_high_budget import _run_specs
from riskshiftbench.experiments.real_agent_validation import (
    _coding_task,
    _research_task,
    _workflow_task,
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
    source_sha256 as engine_source_sha256,
)
from riskshiftbench.experiments.vdc_efficient import efficient_family_decisions


CONFIG = Path("riskshiftbench/configs/vdc_efficient_confirmation_v1.json")
ARTIFACT_DIR = Path("riskshiftbench/artifacts/vdc_efficient_confirmation_v1")
MANIFEST = ARTIFACT_DIR / "task_manifest.json"
RAW = ARTIFACT_DIR / "raw_trajectories.json"
OUTPUT = ARTIFACT_DIR / "results.json"


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for path in (root / "canonical_metrics.py", root / "experiments" / "vdc_efficient.py"):
        digest.update(path.read_bytes())
    digest.update(engine_source_sha256().encode("utf-8"))
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-vdc-efficient-confirmation-v1":
        raise RuntimeError("unexpected confirmation protocol")
    if not config["design_locked_before_inference"] or config["outcomes_observed_at_lock"]:
        raise RuntimeError("confirmation must be locked before outcome access")
    if int(config["tasks_per_family"]) != 2_000 or len(config["agent_pairs"]) != 6:
        raise RuntimeError("confirmation fixes six updates and 2,000 pairs/update")
    if config["declared_looks"] != 1 or not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("one fixed look and no replacement are required")


def generate_tasks(config: dict) -> list[dict]:
    validate_config(config)
    builders = {
        "coding": _coding_task,
        "workflow": _workflow_task,
        "research": _research_task,
    }
    rng = random.Random(int(config["task_seed"]))
    tasks = []
    for family, builder in builders.items():
        for index in range(int(config["tasks_per_family"])):
            payload = builder(rng, index, config["shift"], config)
            tasks.append(
                {
                    "id": f"efficient-confirmation-{family}-{index:05d}",
                    "split": "fresh-confirmation",
                    "family": family,
                    "shift": config["shift"],
                    "option_seed": rng.randrange(2**31),
                    **payload,
                }
            )
    return tasks


def prepare_manifest(config: dict) -> dict:
    payload = {
        "protocol_id": config["protocol_id"],
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "tasks": generate_tasks(config),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_manifest(config: dict) -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise RuntimeError("confirmation manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("confirmation config changed after lock")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("confirmation source changed after lock")
    return payload


def execute(config: dict, manifest: dict) -> dict:
    original_raw = cached_runner.RAW
    original_source = cached_runner.source_sha256
    cached_runner.RAW = RAW
    cached_runner.source_sha256 = source_sha256
    try:
        return cached_runner.execute(config, manifest)
    finally:
        cached_runner.RAW = original_raw
        cached_runner.source_sha256 = original_source


def load_raw(manifest: dict) -> dict:
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "raw_sha256"}
    if payload["raw_sha256"] != canonical_sha256(body):
        raise RuntimeError("confirmation raw digest mismatch")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("confirmation raw data use another manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("confirmation raw data use another source")
    return payload


def _summary(route: dict[str, str], config: dict, pilot_pairs: int) -> dict:
    promoted = {key for key, value in route.items() if value == "promote"}
    eligible = {key for key, value in config["reference_eligibility"].items() if value}
    return {
        "route": route,
        "promoted_updates": sorted(promoted),
        "promotions": len(promoted),
        "ineligible_promotions": len(promoted - eligible),
        "eligible_recall": len(promoted & eligible) / len(eligible),
        "deployment_gain": sum(config["final_effects"][key] for key in promoted)
        / len(config["agent_pairs"]),
        "pilot_pairs": pilot_pairs,
        "recalibrations": sum(value == "recalibrate-verifier" for value in route.values()),
        "unresolved": sum(value == "unresolved" for value in route.values()),
    }


def analyze(config: dict, manifest: dict, raw: dict) -> dict:
    trajectories = raw["trajectories"]
    tasks = {}
    point_metrics = {}
    for pair in config["agent_pairs"]:
        pair_id, family = pair["id"], pair["family"]
        temperature = float(config["confidence"][family]["temperature"])
        thresholds = [float(value) for value in config["confidence"][family]["thresholds"]]
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
            [row["score"] for row in fallback], [row["score"] for row in candidate]
        )
        candidate_h = {
            threshold: [high_confidence_failure(row, threshold) for row in candidate]
            for threshold in thresholds
        }
        fallback_h = {
            threshold: [high_confidence_failure(row, threshold) for row in fallback]
            for threshold in thresholds
        }
        tasks[pair_id] = (effects, candidate_h, fallback_h)
        metrics = canonical_update_metrics(
            effects, candidate_h, fallback_h, material_loss_threshold=0.5
        )
        point_metrics[pair_id] = {
            "mean_effect": metrics.mean_effect,
            "material_loss_probability": metrics.material_loss_probability,
            "candidate_high_confidence_failure": metrics.candidate_high_confidence_failure,
            "verifier_drift_max": metrics.verifier_drift_max,
        }
    efficient = efficient_family_decisions(
        tasks,
        family_alpha=0.05,
        mean_margin=0.02,
        material_loss_threshold=0.5,
        maximum_material_loss_probability=0.10,
        maximum_candidate_high_confidence_failure_probability=0.15,
        maximum_verifier_drift=0.05,
        confidence_unresolved_reason="recalibrate-verifier",
    )
    efficient_route = {key: value.reason for key, value in efficient.items()}
    outcome_route = {}
    frozen_route = {}
    for key, metrics in point_metrics.items():
        outcome_route[key] = "promote" if (
            metrics["mean_effect"] > 0.02
            and metrics["material_loss_probability"] < 0.10
        ) else "retain"
        frozen_route[key] = "promote" if (
            metrics["mean_effect"] > 0.02
            and metrics["material_loss_probability"] < 0.10
            and metrics["candidate_high_confidence_failure"] < 0.15
            and metrics["verifier_drift_max"] < 0.05
        ) else "retain"
    total = len(config["agent_pairs"]) * int(config["tasks_per_family"])
    methods = {
        "always_fallback": _summary(
            {key: "retain" for key in tasks}, config, 0
        ),
        "outcome_only": _summary(outcome_route, config, total),
        "frozen_verifier": _summary(frozen_route, config, total),
        "vdc_efficient": _summary(efficient_route, config, total),
        "oracle": _summary(
            {
                key: "promote" if value else "retain"
                for key, value in config["reference_eligibility"].items()
            },
            config,
            0,
        ),
    }
    payload = {
        "protocol_id": config["protocol_id"],
        "evidential_status": config["evidential_status"],
        "selection_disclosure": config["selection_disclosure"],
        "config_sha256": canonical_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": source_sha256(),
        "raw_sha256": raw["raw_sha256"],
        "executed_agent_episodes": len(trajectories),
        "executed_actions": sum(len(row["actions"]) for row in trajectories),
        "point_metrics": point_metrics,
        "component_pvalues": {
            key: value.component_pvalues for key, value in efficient.items()
        },
        "methods": methods,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.prepare_only:
        payload = prepare_manifest(config)
        print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))
        return
    manifest = load_manifest(config)
    raw = load_raw(manifest) if args.analyze_only else execute(config, manifest)
    payload = analyze(config, manifest, raw)
    print(json.dumps({key: value for key, value in payload.items() if key != "component_pvalues"}, indent=2))


if __name__ == "__main__":
    main()
