"""Run the locked full-family 2,500-pair real-agent VDC validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from statistics import fmean

from riskshiftbench.canonical_metrics import (
    canonical_update_metrics,
    is_eligible,
    paired_effects,
)
from riskshiftbench.experiments.real_agent_validation import (
    _coding_task,
    _research_task,
    _workflow_task,
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
    run_trajectories,
    source_sha256 as engine_source_sha256,
)
from riskshiftbench.experiments.verifier_drift_control import (
    VDCPlan,
    decide_vdc,
)


CONFIG = Path("riskshiftbench/configs/real_agent_high_budget_v1.json")
AMENDED_REFERENCE = Path(
    "riskshiftbench/artifacts/real_agent_validation_v1/results_amended_v2.json"
)
ARTIFACT_DIR = Path("riskshiftbench/artifacts/real_agent_high_budget_v1")
MANIFEST = ARTIFACT_DIR / "task_manifest.json"
RAW = ARTIFACT_DIR / "raw_trajectories.json"
OUTPUT = ARTIFACT_DIR / "results.json"


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        root / "canonical_metrics.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(engine_source_sha256().encode("utf-8"))
    return digest.hexdigest()


def _load_amended_reference() -> dict:
    payload = json.loads(AMENDED_REFERENCE.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(body):
        raise RuntimeError("amended reference result digest mismatch")
    return payload


def validate_config(config: dict, amended: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-real-agent-high-budget-v1":
        raise RuntimeError("unexpected high-budget protocol")
    if not config["design_locked_before_pilot_inference"]:
        raise RuntimeError("high-budget design must be locked before inference")
    if config["pilot_outcomes_observed_at_lock"]:
        raise RuntimeError("lock cannot follow fresh pilot outcome access")
    if not config["shared_fallback_within_family"]:
        raise RuntimeError("the execution plan requires shared family fallbacks")
    if len(config["agent_pairs"]) != 6:
        raise RuntimeError("the full six-update family is required")
    if not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("the no-replacement rule is required")
    expected_labels = {
        row["pair_id"]: bool(row["frozen_eligible"])
        for row in amended["updates"]
    }
    if config["reference_eligibility"] != expected_labels:
        raise RuntimeError("frozen reference labels do not match amended v2")


def generate_tasks(config: dict, amended: dict) -> list[dict]:
    validate_config(config, amended)
    builders = {
        "coding": _coding_task,
        "workflow": _workflow_task,
        "research": _research_task,
    }
    rng = random.Random(int(config["pilot_seed"]))
    tasks = []
    for family, builder in builders.items():
        for index in range(int(config["pilot_tasks_per_family"])):
            payload = builder(rng, index, config["shift"], config)
            tasks.append(
                {
                    "id": f"high-budget-{family}-{index:05d}",
                    "split": "fresh-pilot",
                    "family": family,
                    "shift": config["shift"],
                    "option_seed": rng.randrange(2**31),
                    **payload,
                }
            )
    return tasks


def prepare_manifest(config: dict, amended: dict, path: Path = MANIFEST) -> dict:
    tasks = generate_tasks(config, amended)
    payload = {
        "protocol_id": config["protocol_id"],
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "amended_reference_result_sha256": amended["result_sha256"],
        "tasks": tasks,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_manifest(config: dict, amended: dict, path: Path = MANIFEST) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise RuntimeError("high-budget manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("high-budget config changed after lock")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("high-budget source changed after lock")
    if payload["amended_reference_result_sha256"] != amended["result_sha256"]:
        raise RuntimeError("reference labels changed after lock")
    return payload


def _run_specs(config: dict) -> list[dict]:
    specs = []
    by_family = {}
    for pair in config["agent_pairs"]:
        by_family.setdefault(pair["family"], []).append(pair)
    for family, pairs in by_family.items():
        specs.append(
            {
                "run_id": f"{family}-fallback-shared",
                "pair": pairs[0],
                "role": "fallback",
                "family": family,
            }
        )
        for pair in pairs:
            specs.append(
                {
                    "run_id": pair["id"],
                    "pair": pair,
                    "role": "candidate",
                    "family": family,
                }
            )
    return specs


def execute(config: dict, manifest: dict, output: Path = RAW) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"], revision=model_spec["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.float16,
    ).to(config["device"])
    model.eval()
    trajectories = []
    completed = set()
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        body = {key: value for key, value in prior.items() if key != "raw_sha256"}
        if (
            prior.get("raw_sha256") == canonical_sha256(body)
            and prior.get("manifest_sha256") == manifest["manifest_sha256"]
            and prior.get("source_sha256") == source_sha256()
        ):
            trajectories = prior["trajectories"]
            completed = set(prior["completed_runs"])
    for spec in _run_specs(config):
        if spec["run_id"] in completed:
            print(f"{spec['run_id']}: restored from checkpoint")
            continue
        tasks = [row for row in manifest["tasks"] if row["family"] == spec["family"]]
        started = time.time()
        rows = run_trajectories(
            model, tokenizer, tasks, spec["pair"], spec["role"], config
        )
        for row in rows:
            row["run_id"] = spec["run_id"]
        trajectories.extend(rows)
        completed.add(spec["run_id"])
        payload = {
            "protocol_id": config["protocol_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": source_sha256(),
            "completed_runs": sorted(completed),
            "trajectories": trajectories,
        }
        payload["raw_sha256"] = canonical_sha256(payload)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"{spec['run_id']}: {len(rows)} episodes in "
            f"{time.time() - started:.1f}s"
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_raw(manifest: dict, path: Path = RAW) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "raw_sha256"}
    if payload["raw_sha256"] != canonical_sha256(body):
        raise RuntimeError("high-budget raw digest mismatch")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("high-budget raw data use a different manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("high-budget raw data use a different source")
    return payload


def _interval_payload(interval) -> dict:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


def _route_summary(route: dict, config: dict) -> dict:
    promoted = {key for key, value in route.items() if value == "promote"}
    ineligible = {
        key for key in promoted if not config["reference_eligibility"][key]
    }
    return {
        "route": route,
        "promotions": len(promoted),
        "promoted_updates": sorted(promoted),
        "ineligible_promotions": len(ineligible),
        "ineligible_updates": sorted(ineligible),
        "deployment_gain": sum(config["final_effects"][key] for key in promoted)
        / len(config["agent_pairs"]),
        "pilot_pairs": 0 if not promoted and set(route.values()) == {"retain-no-pilot"}
        else len(config["agent_pairs"]) * int(config["pilot_tasks_per_family"]),
    }


def analyze(config: dict, manifest: dict, raw: dict, output: Path = OUTPUT) -> dict:
    trajectories = raw["trajectories"]
    fallback_by_family = {
        family: [
            row
            for row in trajectories
            if row["run_id"] == f"{family}-fallback-shared"
        ]
        for family in ("coding", "workflow", "research")
    }
    plan = VDCPlan(
        confidence_thresholds=(0.1, 0.3, 0.5),
        family_tasks=len(config["agent_pairs"]),
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
    updates = []
    routes = {
        "always_fallback": {},
        "outcome_only": {},
        "frozen_verifier": {},
        "vdc": {},
        "oracle": {},
    }
    for pair in config["agent_pairs"]:
        pair_id, family = pair["id"], pair["family"]
        temperature = float(config["confidence"][family]["temperature"])
        thresholds = [
            float(value) for value in config["confidence"][family]["thresholds"]
        ]
        fallback = [
            calibrate_episode(row, temperature)
            for row in fallback_by_family[family]
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
        candidate_h_threshold = {
            threshold: [high_confidence_failure(row, threshold) for row in candidate]
            for threshold in thresholds
        }
        fallback_h_threshold = {
            threshold: [high_confidence_failure(row, threshold) for row in fallback]
            for threshold in thresholds
        }
        canonical = canonical_update_metrics(
            effects,
            candidate_h_threshold,
            fallback_h_threshold,
            material_loss_threshold=float(config["material_loss_threshold"]),
        )
        canonical_eligible = is_eligible(
            canonical,
            mean_margin=float(config["mean_margin"]),
            maximum_material_loss_probability=float(
                config["maximum_material_loss_probability"]
            ),
            maximum_candidate_high_confidence_failure_probability=float(
                config["maximum_candidate_high_confidence_failure_probability"]
            ),
            maximum_verifier_drift=float(config["maximum_verifier_drift"]),
        )
        candidate_h = {
            coverage: candidate_h_threshold[threshold]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        fallback_h = {
            coverage: fallback_h_threshold[threshold]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        decision = decide_vdc(
            effects,
            candidate_h,
            fallback_h,
            plan,
            at_task_cap=True,
            legacy_recalibration_label=True,
        )
        bounds = decision.bounds
        updates.append(
            {
                "pair_id": pair_id,
                "family": family,
                "pilot_metrics": {
                    "mean_effect": canonical.mean_effect,
                    "material_loss_probability": canonical.material_loss_probability,
                    "candidate_high_confidence_failure": canonical.candidate_high_confidence_failure,
                    "verifier_drift_max": canonical.verifier_drift_max,
                    "drift_profile": canonical.drift_profile,
                },
                "pilot_point_eligible": canonical_eligible,
                "reference_eligible": config["reference_eligibility"][pair_id],
                "vdc_reason": decision.reason,
                "bounds": {
                    "mean": _interval_payload(bounds.mean),
                    "downside": _interval_payload(bounds.downside),
                    "candidate_confidence_risk": _interval_payload(
                        bounds.candidate_confidence_risk
                    ),
                    "verifier_drift": _interval_payload(bounds.verifier_drift),
                },
            }
        )
        routes["always_fallback"][pair_id] = "retain-no-pilot"
        routes["outcome_only"][pair_id] = "promote" if (
            canonical.mean_effect > float(config["mean_margin"])
            and canonical.material_loss_probability
            < float(config["maximum_material_loss_probability"])
        ) else "retain"
        routes["frozen_verifier"][pair_id] = (
            "promote" if canonical_eligible else "retain"
        )
        routes["vdc"][pair_id] = decision.reason
        routes["oracle"][pair_id] = (
            "promote" if config["reference_eligibility"][pair_id] else "retain"
        )

    summaries = {
        method: _route_summary(route, config) for method, route in routes.items()
    }
    summaries["oracle"]["pilot_pairs"] = 0
    payload = {
        "protocol_id": config["protocol_id"],
        "evidential_status": config["evidential_status"],
        "selection_disclosure": config["selection_disclosure"],
        "config_sha256": canonical_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": source_sha256(),
        "raw_sha256": raw["raw_sha256"],
        "logical_paired_pilots": len(config["agent_pairs"])
        * int(config["pilot_tasks_per_family"]),
        "executed_agent_episodes": len(trajectories),
        "executed_actions": sum(len(row["actions"]) for row in trajectories),
        "updates": updates,
        "methods": summaries,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    amended = _load_amended_reference()
    if args.prepare_only:
        payload = prepare_manifest(config, amended)
        print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))
        return
    manifest = load_manifest(config, amended)
    raw = load_raw(manifest) if args.analyze_only else execute(config, manifest)
    payload = analyze(config, manifest, raw)
    print(json.dumps({key: value for key, value in payload.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()
