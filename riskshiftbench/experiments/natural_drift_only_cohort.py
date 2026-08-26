"""Run the locked ten-update natural drift-only cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import riskshiftbench.experiments.real_agent_high_budget_extension as cached_runner
import riskshiftbench.experiments.real_agent_validation as engine
from riskshiftbench.canonical_metrics import canonical_update_metrics, paired_effects
from riskshiftbench.experiments.real_agent_validation import (
    _coding_task,
    _research_task,
    _workflow_task,
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
    source_sha256 as engine_source_sha256,
)


CONFIG = Path("riskshiftbench/configs/natural_drift_only_cohort_v1.json")
ARTIFACT_DIR = Path("riskshiftbench/artifacts/natural_drift_only_cohort_v1")
MANIFEST = ARTIFACT_DIR / "task_manifest.json"
RAW = ARTIFACT_DIR / "raw_trajectories.json"
OUTPUT = ARTIFACT_DIR / "results.json"


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update((root / "canonical_metrics.py").read_bytes())
    digest.update(engine_source_sha256().encode("utf-8"))
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-natural-drift-only-cohort-v1":
        raise RuntimeError("unexpected cohort protocol")
    if not config["design_locked_before_inference"] or config["outcomes_observed_at_lock"]:
        raise RuntimeError("cohort must be locked before outcome access")
    if len(config["agent_pairs"]) != 10 or int(config["tasks_per_family"]) != 1_000:
        raise RuntimeError("cohort requires ten updates and 1,000 tasks/family")
    if len({row["id"] for row in config["agent_pairs"]}) != 10:
        raise RuntimeError("cohort update identifiers must be unique")
    custom = {
        row["id"]
        for row in config["agent_pairs"]
        if row["id"].startswith("cohort-")
    }
    if custom != set(config["candidate_system_prompts"]):
        raise RuntimeError("every custom update must have one frozen system prompt")
    if not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("cohort requires a no-replacement rule")


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
                    "id": f"drift-cohort-{family}-{index:05d}",
                    "split": "fresh-drift-cohort",
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
        raise RuntimeError("cohort manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("cohort config changed after lock")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("cohort source changed after lock")
    return payload


def execute(config: dict, manifest: dict) -> dict:
    old_prompts = dict(engine.CANDIDATE_SYSTEM)
    old_raw = cached_runner.RAW
    old_source = cached_runner.source_sha256
    engine.CANDIDATE_SYSTEM.update(config["candidate_system_prompts"])
    cached_runner.RAW = RAW
    cached_runner.source_sha256 = source_sha256
    try:
        return cached_runner.execute(config, manifest)
    finally:
        engine.CANDIDATE_SYSTEM.clear()
        engine.CANDIDATE_SYSTEM.update(old_prompts)
        cached_runner.RAW = old_raw
        cached_runner.source_sha256 = old_source


def load_raw(manifest: dict) -> dict:
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "raw_sha256"}
    if payload["raw_sha256"] != canonical_sha256(body):
        raise RuntimeError("cohort raw digest mismatch")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("cohort raw data use another manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("cohort raw data use another source")
    return payload


def analyze(config: dict, manifest: dict, raw: dict) -> dict:
    trajectories = raw["trajectories"]
    updates = []
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
        metrics = canonical_update_metrics(
            effects, candidate_h, fallback_h, material_loss_threshold=0.5
        )
        first_three = (
            metrics.mean_effect > float(config["mean_margin"])
            and metrics.material_loss_probability
            < float(config["maximum_material_loss_probability"])
            and metrics.candidate_high_confidence_failure
            < float(config["maximum_candidate_high_confidence_failure_probability"])
        )
        drift_only = first_three and metrics.verifier_drift_max >= float(
            config["maximum_verifier_drift"]
        )
        eligible = first_three and not drift_only
        updates.append(
            {
                "pair_id": pair_id,
                "family": family,
                "update": pair["update"],
                "mean_effect": metrics.mean_effect,
                "material_loss_probability": metrics.material_loss_probability,
                "candidate_high_confidence_failure": metrics.candidate_high_confidence_failure,
                "verifier_drift_max": metrics.verifier_drift_max,
                "drift_profile": metrics.drift_profile,
                "passes_first_three": first_three,
                "drift_only_failure": drift_only,
                "eligible": eligible,
            }
        )
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
        "updates": updates,
        "drift_only_updates": [
            row["pair_id"] for row in updates if row["drift_only_failure"]
        ],
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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
