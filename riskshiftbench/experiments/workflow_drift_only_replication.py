"""Execute the locked targeted workflow-memory drift-only replication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from statistics import fmean

from riskshiftbench.experiments.controlled_verifier_drift import (
    mean_interval,
    wilson_interval,
)
from riskshiftbench.experiments.real_agent_validation import (
    _workflow_task,
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
    paired_metrics,
    run_trajectories,
    source_sha256 as engine_source_sha256,
)
from riskshiftbench.experiments.verifier_drift_control import VDCPlan, decide_vdc


CONFIG = Path("riskshiftbench/configs/workflow_drift_only_replication_v1.json")
BASE_CONFIG = Path("riskshiftbench/configs/real_agent_validation_v1.json")
ARTIFACT_DIR = Path("riskshiftbench/artifacts/workflow_drift_only_replication_v1")
MANIFEST = ARTIFACT_DIR / "task_manifest.json"
RAW = ARTIFACT_DIR / "raw_trajectories.json"
OUTPUT = ARTIFACT_DIR / "results.json"


def source_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(engine_source_sha256().encode("utf-8"))
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-workflow-drift-only-replication-v1":
        raise RuntimeError("unexpected replication protocol")
    if not config["design_locked_before_replication_inference"]:
        raise RuntimeError("replication must be locked before inference")
    if config["replication_outcomes_observed_at_lock"]:
        raise RuntimeError("lock cannot follow replication outcome access")
    if config["pair"]["id"] != "workflow-memory":
        raise RuntimeError("this replication fixes the workflow-memory pair")
    if int(config["task_count"]) != int(config["independent_streams"]) * int(
        config["stream_size"]
    ):
        raise RuntimeError("independent streams must partition the task manifest")
    if not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("the no-replacement rule is required")


def generate_tasks(config: dict, base: dict) -> list[dict]:
    validate_config(config)
    rng = random.Random(int(config["task_seed"]))
    tasks = []
    for index in range(int(config["task_count"])):
        payload = _workflow_task(rng, index, config["shift"], base)
        tasks.append(
            {
                "id": f"replication-workflow-{index:05d}",
                "split": "replication",
                "family": "workflow",
                "shift": config["shift"],
                "stream": index // int(config["stream_size"]),
                "option_seed": rng.randrange(2**31),
                **payload,
            }
        )
    return tasks


def prepare_manifest(config: dict, base: dict, path: Path = MANIFEST) -> dict:
    payload = {
        "protocol_id": config["protocol_id"],
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "base_config_sha256": canonical_sha256(base),
        "source_sha256": source_sha256(),
        "tasks": generate_tasks(config, base),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_manifest(config: dict, base: dict, path: Path = MANIFEST) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise RuntimeError("replication manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("replication config changed after lock")
    if payload["base_config_sha256"] != canonical_sha256(base):
        raise RuntimeError("base generator config changed after lock")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("replication source changed after lock")
    return payload


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
            completed = set(prior["completed_roles"])
    pair = config["pair"]
    for role in ("fallback", "candidate"):
        if role in completed:
            continue
        started = time.time()
        rows = run_trajectories(
            model, tokenizer, manifest["tasks"], pair, role, config
        )
        trajectories.extend(rows)
        completed.add(role)
        payload = {
            "protocol_id": config["protocol_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": source_sha256(),
            "completed_roles": sorted(completed),
            "trajectories": trajectories,
        }
        payload["raw_sha256"] = canonical_sha256(payload)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{role}: {len(rows)} episodes in {time.time() - started:.1f}s")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_raw(manifest: dict, path: Path = RAW) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "raw_sha256"}
    if payload["raw_sha256"] != canonical_sha256(body):
        raise RuntimeError("replication raw digest mismatch")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("replication raw data use a different manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("replication raw data use a different source")
    return payload


def score_downside_metrics(fallback: list[dict], candidate: list[dict], thresholds: list[float]) -> dict:
    metrics = paired_metrics(fallback, candidate, thresholds)
    metrics["material_loss_probability"] = fmean(
        c_row["score"] - f_row["score"] < -0.5
        for f_row, c_row in zip(fallback, candidate)
    )
    return metrics


def analyze(config: dict, manifest: dict, raw: dict, output: Path = OUTPUT) -> dict:
    temperature = float(config["frozen_fallback_temperature"])
    thresholds = [float(value) for value in config["frozen_confidence_thresholds"]]
    fallback = [
        calibrate_episode(row, temperature)
        for row in raw["trajectories"]
        if row["role"] == "fallback"
    ]
    candidate = [
        calibrate_episode(row, temperature)
        for row in raw["trajectories"]
        if row["role"] == "candidate"
    ]
    metrics = score_downside_metrics(fallback, candidate, thresholds)
    drift_only = (
        metrics["mean_effect"] > float(config["mean_margin"])
        and metrics["material_loss_probability"]
        < float(config["maximum_material_loss_probability"])
        and metrics["candidate_high_confidence_failure"]
        < float(config["maximum_candidate_high_confidence_failure_probability"])
        and metrics["verifier_drift_max"] >= float(config["maximum_verifier_drift"])
    )
    plan = VDCPlan(
        confidence_thresholds=(0.1, 0.3, 0.5),
        family_tasks=1,
        declared_looks=1,
        family_alpha=float(config["family_alpha"]),
        mean_margin=float(config["mean_margin"]),
        material_loss_threshold=float(config["material_loss_threshold"]),
        maximum_material_loss_probability=float(config["maximum_material_loss_probability"]),
        maximum_candidate_high_confidence_failure_probability=float(
            config["maximum_candidate_high_confidence_failure_probability"]
        ),
        maximum_verifier_drift=float(config["maximum_verifier_drift"]),
    )
    streams = []
    size = int(config["stream_size"])
    for stream in range(int(config["independent_streams"])):
        start, end = stream * size, (stream + 1) * size
        f_stream, c_stream = fallback[start:end], candidate[start:end]
        stream_metrics = score_downside_metrics(f_stream, c_stream, thresholds)
        effects = [c["score"] - f["score"] for f, c in zip(f_stream, c_stream)]
        candidate_h = {
            coverage: [high_confidence_failure(row, threshold) for row in c_stream]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        fallback_h = {
            coverage: [high_confidence_failure(row, threshold) for row in f_stream]
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
        streams.append(
            {
                "stream": stream,
                "metrics": stream_metrics,
                "vdc_reason": decision.reason,
                "point_drift_detected": stream_metrics["verifier_drift_max"]
                >= float(config["maximum_verifier_drift"]),
            }
        )

    rng = random.Random(int(config["bootstrap_seed"]))
    drift_draws = []
    mean_draws = []
    for _ in range(int(config["bootstrap_draws"])):
        indices = [rng.randrange(len(fallback)) for _ in range(len(fallback))]
        draw = score_downside_metrics(
            [fallback[index] for index in indices],
            [candidate[index] for index in indices],
            thresholds,
        )
        drift_draws.append(draw["verifier_drift_max"])
        mean_draws.append(draw["mean_effect"])
    drift_draws.sort()
    mean_draws.sort()
    lower_index = int(0.025 * len(drift_draws))
    upper_index = int(0.975 * len(drift_draws)) - 1
    payload = {
        "protocol_id": config["protocol_id"],
        "evidential_status": config["evidential_status"],
        "selection_disclosure": config["selection_disclosure"],
        "config_sha256": canonical_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": source_sha256(),
        "raw_sha256": raw["raw_sha256"],
        "task_count": len(fallback),
        "metrics": metrics,
        "drift_only_point_label": drift_only,
        "bootstrap_mean_interval_95": [
            mean_draws[lower_index],
            mean_draws[upper_index],
        ],
        "bootstrap_drift_interval_95": [
            drift_draws[lower_index],
            drift_draws[upper_index],
        ],
        "independent_streams": streams,
        "stream_summary": {
            "streams": len(streams),
            "point_drift_detection_frequency": fmean(
                row["point_drift_detected"] for row in streams
            ),
            "point_drift_detection_wilson_interval_95": wilson_interval(
                sum(row["point_drift_detected"] for row in streams), len(streams)
            ),
            "vdc_reasons": {
                reason: sum(row["vdc_reason"] == reason for row in streams)
                for reason in sorted({row["vdc_reason"] for row in streams})
            },
            "mean_effect_interval_across_streams": mean_interval(
                [row["metrics"]["mean_effect"] for row in streams]
            ),
        },
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
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    if args.prepare_only:
        payload = prepare_manifest(config, base)
        print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))
        return
    manifest = load_manifest(config, base)
    raw = load_raw(manifest) if args.analyze_only else execute(config, manifest)
    payload = analyze(config, manifest, raw)
    print(json.dumps({key: value for key, value in payload.items() if key != "independent_streams"}, indent=2))


if __name__ == "__main__":
    main()
