"""Prospective multi-model study of verifier preservation under agent updates.

The study has two locks.  The design and development manifest are committed
before new inference.  Development outcomes fix calibration and policy-specific
coverage thresholds.  A second manifest freezes those values and fresh target
tasks before any target outcome is opened.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

import riskshiftbench.experiments.prospective_full_vs_absolute as base
import riskshiftbench.experiments.real_agent_validation as engine
from riskshiftbench.canonical_metrics import selective_verifier_drift
from riskshiftbench.experiments.natural_agent_updates import (
    fit_temperature,
    thresholds_at_coverages,
)
from riskshiftbench.experiments.real_agent_validation import (
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
)
from riskshiftbench.experiments.verifier_preservation import (
    PreservationEvidence,
    certify_preservation_family,
    point_preservation_metrics,
)


CONFIG = Path("riskshiftbench/configs/verifier_preservation_v1.json")
ARTIFACT_DIR = Path("riskshiftbench/artifacts/verifier_preservation_v1")
DEVELOPMENT_MANIFEST = ARTIFACT_DIR / "development_manifest.json"
DEVELOPMENT_RAW = ARTIFACT_DIR / "development_trajectories.json.gz"
DEVELOPMENT_RESULTS = ARTIFACT_DIR / "development_results.json"
TARGET_MANIFEST = ARTIFACT_DIR / "target_manifest.json"
TARGET_RAW = ARTIFACT_DIR / "target_trajectories.json.gz"
TARGET_RESULTS = ARTIFACT_DIR / "results.json"
FEASIBILITY = ARTIFACT_DIR / "feasibility.json"


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        root / "experiments" / "verifier_preservation.py",
        root / "experiments" / "prospective_full_vs_absolute.py",
        root / "experiments" / "vdc_efficient.py",
        root / "canonical_metrics.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pairs(config: dict) -> list[dict]:
    return base._pairs(config)


def validate_config(config: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-verifier-preservation-v1":
        raise RuntimeError("unexpected verifier-preservation protocol")
    if not config["design_locked_before_inference"] or config["outcomes_observed_at_lock"]:
        raise RuntimeError("the study must be locked before inference")
    if len(config["models"]) != 2 or len(config["update_designs"]) != 6:
        raise RuntimeError("the study requires two models and six update mechanisms")
    rows = pairs(config)
    if len(rows) != 36 or len({row["id"] for row in rows}) != 36:
        raise RuntimeError("the study requires 36 unique candidate--fallback updates")
    if Counter(row["cohort"] for row in rows) != {"natural": 18, "stress": 18}:
        raise RuntimeError("natural and stress cohorts must each contain 18 updates")
    if config["certification_targets"] != ["absolute", "operational", "preserve"]:
        raise RuntimeError("certification target family changed")
    if int(config["pilot_pairs_per_update"]) != 2_000:
        raise RuntimeError("the target pilot budget is fixed at 2,000 pairs per update")
    if not config["long_horizon_coding"] or int(config["maximum_steps"]["coding"]) != 10:
        raise RuntimeError("the coding slice must use the locked ten-step environment")
    if not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("the update cohort has a no-replacement rule")
    if config["primary_update_unit"] != "candidate_fallback_update":
        raise RuntimeError("the independent study unit must be the agent update")
    expected_methods = [
        "always_fallback",
        "outcome_only",
        "task_iut_absolute",
        "task_iut_operational",
        "task_iut_preserve",
        "oracle_preserve",
    ]
    if config["primary_methods"] != expected_methods:
        raise RuntimeError("primary method family changed")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))


def _read_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_development_manifest(config: dict) -> dict:
    validate_config(config)
    payload = {
        "protocol_id": config["protocol_id"],
        "phase": "development",
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "pairs": pairs(config),
        "tasks": base._generate_split(
            config,
            split="development",
            count=int(config["development_tasks_per_family"]),
            seed=int(config["development_seed"]),
            shift=config["development_shift"],
        ),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(DEVELOPMENT_MANIFEST, payload)
    return payload


def _load_manifest(path: Path, config: dict, phase: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise RuntimeError(f"{phase} manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError(f"{phase} manifest uses another configuration")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError(f"{phase} source changed after lock")
    return payload


def _load_raw(path: Path, manifest: dict) -> dict:
    payload = _read_gzip(path)
    body = {key: value for key, value in payload.items() if key != "raw_sha256"}
    if payload["raw_sha256"] != canonical_sha256(body):
        raise RuntimeError("raw trajectory digest mismatch")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("raw trajectories use another manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("raw trajectories use another source")
    return payload


def validate_complete_raw(manifest: dict, raw: dict) -> None:
    model_families = {
        (row["model_id"], row["family"]) for row in manifest["pairs"]
    }
    expected_runs = {
        *(f"{model}::{family}::fallback" for model, family in model_families),
        *(row["id"] for row in manifest["pairs"]),
    }
    actual_runs = {row["run_id"] for row in raw["trajectories"]}
    if expected_runs != actual_runs:
        missing = sorted(expected_runs - actual_runs)
        raise RuntimeError(f"trajectory collection is incomplete: {missing[:3]}")
    base.validate_complete_target_raw(manifest, raw)


def execute(config: dict, manifest: dict, output: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trajectories = []
    completed = set()
    if output.exists():
        prior = _load_raw(output, manifest)
        trajectories = prior["trajectories"]
        completed = set(prior["completed_runs"])
    for model_spec in config["models"]:
        load_path = model_spec.get("local_path", model_spec["repository"])
        local_model = "local_path" in model_spec
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            revision=None if local_model else model_spec["revision"],
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        dtype = getattr(torch, str(config["torch_dtype"]))
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            revision=None if local_model else model_spec["revision"],
            torch_dtype=dtype,
            local_files_only=True,
        ).to(config["device"])
        model.eval()
        original_score = engine._score_prompt_batch
        cache: dict[tuple[str, str], list[float]] = {}

        def cached_score(model_arg, tokenizer_arg, prompts, config_arg):
            missing = []
            seen = set()
            for prompt in prompts:
                if prompt not in cache and prompt not in seen:
                    missing.append(prompt)
                    seen.add(prompt)
            if missing:
                scored = original_score(model_arg, tokenizer_arg, missing, config_arg)
                cache.update(zip(missing, scored))
            return [cache[prompt] for prompt in prompts]

        engine._score_prompt_batch = cached_score
        try:
            model_pairs = [
                row for row in manifest["pairs"] if row["model_id"] == model_spec["id"]
            ]
            for family in config["families"]:
                tasks = [row for row in manifest["tasks"] if row["family"] == family]
                fallback_id = f"{model_spec['id']}::{family}::fallback"
                run_specs = [
                    {
                        "id": fallback_id,
                        "family": family,
                        "memory_mode": "none",
                        "system_prompt": engine.BASE_SYSTEM[family],
                        "role": "fallback",
                    },
                    *(
                        {**row, "role": "candidate"}
                        for row in model_pairs
                        if row["family"] == family
                    ),
                ]
                for spec in run_specs:
                    if spec["id"] in completed:
                        continue
                    started = time.time()
                    rows = base._run_trajectories(
                        model,
                        tokenizer,
                        tasks,
                        spec,
                        spec["role"],
                        config,
                        int(model_spec["batch_size"]),
                    )
                    for row in rows:
                        row["run_id"] = spec["id"]
                        row["model_id"] = model_spec["id"]
                    trajectories.extend(rows)
                    completed.add(spec["id"])
                    payload = {
                        "protocol_id": config["protocol_id"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "source_sha256": source_sha256(),
                        "completed_runs": sorted(completed),
                        "trajectories": trajectories,
                    }
                    payload["raw_sha256"] = canonical_sha256(payload)
                    _write_gzip(output, payload)
                    print(
                        f"{spec['id']}: {len(rows)} episodes, {len(cache)} cached prompts, "
                        f"{time.time() - started:.1f}s"
                    )
        finally:
            engine._score_prompt_batch = original_score
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return _load_raw(output, manifest)


def smoke_models(config: dict) -> dict:
    """Check checkpoint loading and four-option scoring without study tasks."""

    import math
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    for model_spec in config["models"]:
        load_path = model_spec.get("local_path", model_spec["repository"])
        local_model = "local_path" in model_spec
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            revision=None if local_model else model_spec["revision"],
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            revision=None if local_model else model_spec["revision"],
            torch_dtype=getattr(torch, str(config["torch_dtype"])),
            local_files_only=True,
        ).to(config["device"])
        model.eval()
        logits = engine._score_prompt_batch(
            model,
            tokenizer,
            [
                (
                    "You are checking an option-scoring interface. Return exactly one option letter.",
                    "Choose the first option. A: first; B: second; C: third; D: fourth.",
                )
            ],
            config,
        )[0]
        if len(logits) != 4 or not all(math.isfinite(value) for value in logits):
            raise RuntimeError(f"invalid option logits for {model_spec['id']}")
        rows.append(
            {
                "model_id": model_spec["id"],
                "option_logits_finite": True,
                "option_count": len(logits),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0,
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    payload = {
        "status": "feasibility_only_no_study_tasks",
        "evidential_status": "checkpoint and option-interface smoke before study lock; no study task used",
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "models": rows,
    }
    payload["feasibility_sha256"] = canonical_sha256(payload)
    _write_json(FEASIBILITY, payload)
    return payload


def _episode_failed(row: dict) -> bool:
    return any(action["commitment_failure"] for action in row["actions"])


def _ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["split"], row["task_id"]))


def _preservation_evidence(
    fallback: list[dict],
    candidate: list[dict],
    operational_thresholds: list[float],
    selective_thresholds: dict[str, list[float]],
) -> PreservationEvidence:
    fallback = _ordered(fallback)
    candidate = _ordered(candidate)
    if [(row["split"], row["task_id"]) for row in fallback] != [
        (row["split"], row["task_id"]) for row in candidate
    ]:
        raise RuntimeError("candidate and fallback evidence is not paired")
    return PreservationEvidence(
        effects=tuple(c["score"] - f["score"] for f, c in zip(fallback, candidate)),
        candidate_operational_failures={
            threshold: tuple(high_confidence_failure(row, threshold) for row in candidate)
            for threshold in operational_thresholds
        },
        fallback_operational_failures={
            threshold: tuple(high_confidence_failure(row, threshold) for row in fallback)
            for threshold in operational_thresholds
        },
        candidate_confidences=tuple(row["episode_confidence"] for row in candidate),
        candidate_failures=tuple(_episode_failed(row) for row in candidate),
        fallback_confidences=tuple(row["episode_confidence"] for row in fallback),
        fallback_failures=tuple(_episode_failed(row) for row in fallback),
        selective_thresholds={
            float(coverage): (float(values[0]), float(values[1]))
            for coverage, values in selective_thresholds.items()
        },
    )


def analyze_development(config: dict, manifest: dict, raw: dict) -> dict:
    validate_complete_raw(manifest, raw)
    verification = {}
    calibrated_point = {}
    for model in config["models"]:
        for family in config["families"]:
            fallback_id = f"{model['id']}::{family}::fallback"
            fallback_raw = _ordered(
                [row for row in raw["trajectories"] if row["run_id"] == fallback_id]
            )
            temperature = fit_temperature(
                [action for row in fallback_raw for action in row["actions"]],
                [float(value) for value in config["temperature_grid"]],
            )
            fallback = [calibrate_episode(row, temperature) for row in fallback_raw]
            operational_thresholds = thresholds_at_coverages(
                [row["episode_confidence"] for row in fallback],
                [float(value) for value in config["threshold_development_coverages"]],
            )
            fallback_selective = thresholds_at_coverages(
                [row["episode_confidence"] for row in fallback],
                [float(value) for value in config["selective_drift_coverages"]],
            )
            for pair in manifest["pairs"]:
                if pair["model_id"] != model["id"] or pair["family"] != family:
                    continue
                candidate = [
                    calibrate_episode(row, temperature)
                    for row in _ordered(
                        [row for row in raw["trajectories"] if row["run_id"] == pair["id"]]
                    )
                ]
                candidate_selective = thresholds_at_coverages(
                    [row["episode_confidence"] for row in candidate],
                    [float(value) for value in config["selective_drift_coverages"]],
                )
                selective = {
                    str(coverage): [candidate_threshold, fallback_threshold]
                    for coverage, candidate_threshold, fallback_threshold in zip(
                        config["selective_drift_coverages"],
                        candidate_selective,
                        fallback_selective,
                    )
                }
                evidence = _preservation_evidence(
                    fallback, candidate, operational_thresholds, selective
                )
                verification[pair["id"]] = {
                    "temperature": temperature,
                    "operational_thresholds": operational_thresholds,
                    "selective_thresholds": selective,
                }
                calibrated_point[pair["id"]] = point_preservation_metrics(evidence)
    payload = {
        "protocol_id": config["protocol_id"],
        "phase": "development",
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_sha256": raw["raw_sha256"],
        "selected_primary_pairs_per_update": int(config["pilot_pairs_per_update"]),
        "verification": verification,
        "point_metrics": calibrated_point,
        "executed_agent_episodes": len(raw["trajectories"]),
        "executed_actions": sum(len(row["actions"]) for row in raw["trajectories"]),
    }
    payload["result_sha256"] = canonical_sha256(payload)
    _write_json(DEVELOPMENT_RESULTS, payload)
    return payload


def _load_development(config: dict, manifest: dict, raw: dict) -> dict:
    payload = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(body):
        raise RuntimeError("development result digest mismatch")
    for key, expected in (
        ("config_sha256", canonical_sha256(config)),
        ("source_sha256", source_sha256()),
        ("manifest_sha256", manifest["manifest_sha256"]),
        ("raw_sha256", raw["raw_sha256"]),
    ):
        if payload[key] != expected:
            raise RuntimeError(f"development result mismatch: {key}")
    return payload


def prepare_target_manifest(config: dict, development: dict) -> dict:
    tasks = []
    for split, count in (
        ("pilot", int(config["pilot_pairs_per_update"])),
        ("reference", int(config["reference_tasks_per_family"])),
        ("final", int(config["final_tasks_per_family"])),
    ):
        tasks.extend(
            base._generate_split(
                config,
                split=split,
                count=count,
                seed=int(config["target_seeds"][split]),
                shift=config["target_shift"],
            )
        )
    payload = {
        "protocol_id": config["protocol_id"],
        "phase": "target",
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "development_result_sha256": development["result_sha256"],
        "pairs": pairs(config),
        "verification": development["verification"],
        "tasks": tasks,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    _write_json(TARGET_MANIFEST, payload)
    return payload


def _component_kwargs(config: dict) -> dict:
    return {
        "family_alpha": float(config["family_alpha"]),
        "mean_margin": float(config["mean_margin"]),
        "material_loss_threshold": float(config["material_loss_threshold"]),
        "maximum_material_loss_probability": float(
            config["maximum_material_loss_probability"]
        ),
        "maximum_candidate_high_confidence_failure_probability": float(
            config["maximum_candidate_high_confidence_failure_probability"]
        ),
        "maximum_operational_drift": float(
            config["maximum_operational_verifier_drift"]
        ),
        "maximum_selective_drift": float(config["maximum_selective_verifier_drift"]),
        "minimum_coverage_retention": float(config["minimum_coverage_retention"]),
        "minimum_selected_episodes": int(config["minimum_selected_episodes"]),
    }


def _point_eligible(metrics: dict, target: str, config: dict) -> bool:
    eligible = (
        metrics["mean_effect"] > float(config["mean_margin"])
        and metrics["material_loss_probability"]
        < float(config["maximum_material_loss_probability"])
        and metrics["candidate_high_confidence_failure"]
        < float(config["maximum_candidate_high_confidence_failure_probability"])
    )
    if target in {"operational", "preserve"}:
        eligible = eligible and metrics["operational_drift"] < float(
            config["maximum_operational_verifier_drift"]
        )
    if target == "preserve":
        eligible = (
            eligible
            and metrics["selective_drift"]
            < float(config["maximum_selective_verifier_drift"])
            and all(
                row["retained_ratio"] >= float(config["minimum_coverage_retention"])
                for row in metrics["coverage_profile"].values()
            )
        )
    return eligible


def _summary(
    route: dict[str, str],
    reference_eligible: dict[str, bool],
    final_effects: dict[str, float],
    *,
    pilot_pairs: int,
) -> dict:
    promoted = {task for task, reason in route.items() if reason == "promote"}
    eligible = {task for task, value in reference_eligible.items() if value}
    return {
        "route": route,
        "promoted_updates": sorted(promoted),
        "promotions": len(promoted),
        "reference_ineligible_promotions": len(promoted - eligible),
        "eligible_recall": len(promoted & eligible) / len(eligible) if eligible else 0.0,
        "deployment_gain": sum(final_effects[task] for task in promoted) / len(route),
        "unresolved": sum(reason != "promote" for reason in route.values()),
        "pilot_pairs": pilot_pairs,
    }


def _depth_profile(
    candidate: list[dict],
    fallback: list[dict],
    operational_thresholds: list[float],
    config: dict,
) -> dict:
    candidate = _ordered(candidate)
    fallback = _ordered(fallback)
    threshold = min(float(value) for value in operational_thresholds)
    profile = {}
    for position in config["depth_analysis_positions"]:
        index = int(position) - 1
        candidate_actions = [
            row["actions"][index] for row in candidate if len(row["actions"]) > index
        ]
        fallback_actions = [
            row["actions"][index] for row in fallback if len(row["actions"]) > index
        ]
        candidate_mass = fmean(
            len(row["actions"]) > index
            and row["actions"][index]["confidence"] >= threshold
            and row["actions"][index]["commitment_failure"]
            for row in candidate
        )
        fallback_mass = fmean(
            len(row["actions"]) > index
            and row["actions"][index]["confidence"] >= threshold
            and row["actions"][index]["commitment_failure"]
            for row in fallback
        )
        selective = None
        if len(candidate_actions) >= 20 and len(fallback_actions) >= 20:
            values = selective_verifier_drift(
                [row["confidence"] for row in candidate_actions],
                [row["commitment_failure"] for row in candidate_actions],
                [row["confidence"] for row in fallback_actions],
                [row["commitment_failure"] for row in fallback_actions],
                coverages=[float(value) for value in config["selective_drift_coverages"]],
            )
            selective = values.verifier_drift_max
        profile[str(position)] = {
            "candidate_reach": len(candidate_actions) / len(candidate),
            "fallback_reach": len(fallback_actions) / len(fallback),
            "operational_drift": candidate_mass - fallback_mass,
            "selective_drift_within_reached_actions": selective,
        }
    return profile


def analyze_target(config: dict, manifest: dict, raw: dict) -> dict:
    validate_complete_raw(manifest, raw)
    evidence_by_split = {split: {} for split in ("pilot", "reference", "final")}
    point = {split: {} for split in evidence_by_split}
    depth = {}
    step_counts = {}
    for pair in manifest["pairs"]:
        verification = manifest["verification"][pair["id"]]
        temperature = float(verification["temperature"])
        fallback_id = f"{pair['model_id']}::{pair['family']}::fallback"
        fallback_all = [
            calibrate_episode(row, temperature)
            for row in raw["trajectories"]
            if row["run_id"] == fallback_id
        ]
        candidate_all = [
            calibrate_episode(row, temperature)
            for row in raw["trajectories"]
            if row["run_id"] == pair["id"]
        ]
        step_counts[pair["id"]] = {
            "fallback": fmean(row["steps"] for row in fallback_all),
            "candidate": fmean(row["steps"] for row in candidate_all),
        }
        for split in evidence_by_split:
            fallback = [row for row in fallback_all if row["split"] == split]
            candidate = [row for row in candidate_all if row["split"] == split]
            evidence = _preservation_evidence(
                fallback,
                candidate,
                [float(value) for value in verification["operational_thresholds"]],
                verification["selective_thresholds"],
            )
            evidence_by_split[split][pair["id"]] = evidence
            point[split][pair["id"]] = point_preservation_metrics(evidence)
        if pair["family"] == "coding":
            depth[pair["id"]] = _depth_profile(
                [row for row in candidate_all if row["split"] == "reference"],
                [row for row in fallback_all if row["split"] == "reference"],
                [float(value) for value in verification["operational_thresholds"]],
                config,
            )

    pilot = evidence_by_split["pilot"]
    decisions = {
        target: certify_preservation_family(
            pilot, target=target, **_component_kwargs(config)
        )
        for target in config["certification_targets"]
    }
    eligibility = {
        target: {
            task: _point_eligible(metrics, target, config)
            for task, metrics in point["reference"].items()
        }
        for target in config["certification_targets"]
    }
    final_effects = {
        task: metrics["mean_effect"] for task, metrics in point["final"].items()
    }
    cost = int(config["pilot_pairs_per_update"]) * len(manifest["pairs"])
    methods = {
        "always_fallback": _summary(
            {task: "retain" for task in pilot},
            eligibility["preserve"],
            final_effects,
            pilot_pairs=0,
        ),
        "outcome_only": _summary(
            {
                task: "promote"
                if point["pilot"][task]["mean_effect"] > float(config["mean_margin"])
                else "retain"
                for task in pilot
            },
            eligibility["preserve"],
            final_effects,
            pilot_pairs=cost,
        ),
    }
    for target in config["certification_targets"]:
        methods[f"task_iut_{target}"] = _summary(
            {task: value.reason for task, value in decisions[target].items()},
            eligibility[target],
            final_effects,
            pilot_pairs=cost,
        )
    methods["oracle_preserve"] = _summary(
        {
            task: "promote" if value else "retain"
            for task, value in eligibility["preserve"].items()
        },
        eligibility["preserve"],
        final_effects,
        pilot_pairs=0,
    )

    pair_specs = {row["id"]: row for row in manifest["pairs"]}
    prevalence = {}
    for group_name, group_values in (
        ("overall", {"all"}),
        ("cohort", {row["cohort"] for row in manifest["pairs"]}),
        ("mechanism", {row["design"] for row in manifest["pairs"]}),
        ("model", {row["model_id"] for row in manifest["pairs"]}),
        ("domain", {row["family"] for row in manifest["pairs"]}),
    ):
        prevalence[group_name] = {}
        for value in sorted(group_values):
            selected = [
                task
                for task, spec in pair_specs.items()
                if value == "all"
                or (group_name == "cohort" and spec["cohort"] == value)
                or (group_name == "mechanism" and spec["design"] == value)
                or (group_name == "model" and spec["model_id"] == value)
                or (group_name == "domain" and spec["family"] == value)
            ]
            useful = [
                task
                for task in selected
                if point["reference"][task]["mean_effect"] > float(config["mean_margin"])
            ]
            prevalence[group_name][value] = {
                "updates": len(selected),
                "useful": len(useful),
                "operational_drift": sum(
                    point["reference"][task]["operational_drift"]
                    >= float(config["maximum_operational_verifier_drift"])
                    for task in useful
                ),
                "selective_drift": sum(
                    point["reference"][task]["selective_drift"]
                    >= float(config["maximum_selective_verifier_drift"])
                    for task in useful
                ),
                "coverage_failure": sum(
                    any(
                        row["retained_ratio"] < float(config["minimum_coverage_retention"])
                        for row in point["reference"][task]["coverage_profile"].values()
                    )
                    for task in useful
                ),
            }

    payload = {
        "protocol_id": config["protocol_id"],
        "evidential_status": "prospective target analysis under the frozen verifier-preservation design",
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_sha256": raw["raw_sha256"],
        "executed_agent_episodes": len(raw["trajectories"]),
        "executed_actions": sum(len(row["actions"]) for row in raw["trajectories"]),
        "point_metrics": point,
        "reference_eligibility": eligibility,
        "methods": methods,
        "component_pvalues": {
            target: {
                task: dict(value.component_pvalues)
                for task, value in family.items()
            }
            for target, family in decisions.items()
        },
        "task_pvalues": {
            target: {
                task: {
                    "raw": value.task_pvalue,
                    "holm_adjusted": value.adjusted_task_pvalue,
                }
                for task, value in family.items()
            }
            for target, family in decisions.items()
        },
        "prevalence": prevalence,
        "step_counts": step_counts,
        "depth_profiles": depth,
    }
    payload["result_sha256"] = canonical_sha256(payload)
    _write_json(TARGET_RESULTS, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare-development",
            "smoke-models",
            "run-development",
            "analyze-development",
            "lock-target",
            "run-target",
            "analyze-target",
            "validate",
        ),
    )
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    if args.action == "smoke-models":
        payload = smoke_models(config)
    elif args.action == "prepare-development":
        payload = prepare_development_manifest(config)
    else:
        development_manifest = _load_manifest(DEVELOPMENT_MANIFEST, config, "development")
        if args.action == "run-development":
            payload = execute(config, development_manifest, DEVELOPMENT_RAW)
        else:
            development_raw = _load_raw(DEVELOPMENT_RAW, development_manifest)
            if args.action == "analyze-development":
                payload = analyze_development(config, development_manifest, development_raw)
            else:
                development = _load_development(
                    config, development_manifest, development_raw
                )
                if args.action == "lock-target":
                    payload = prepare_target_manifest(config, development)
                else:
                    target_manifest = _load_manifest(TARGET_MANIFEST, config, "target")
                    if target_manifest["development_result_sha256"] != development["result_sha256"]:
                        raise RuntimeError("target lock uses another development result")
                    if args.action == "run-target":
                        payload = execute(config, target_manifest, TARGET_RAW)
                    else:
                        target_raw = _load_raw(TARGET_RAW, target_manifest)
                        payload = analyze_target(config, target_manifest, target_raw)
                        if args.action == "validate":
                            print("validated verifier-preservation result")
    summary = {
        key: value
        for key, value in payload.items()
        if key not in {"tasks", "trajectories", "point_metrics", "component_pvalues"}
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
