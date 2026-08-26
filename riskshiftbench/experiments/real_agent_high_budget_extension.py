"""Extend the locked full-family VDC validation to 5,000 cumulative pairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import riskshiftbench.experiments.real_agent_validation as engine
from riskshiftbench.experiments.real_agent_high_budget import (
    _run_specs,
    analyze as analyze_family,
    source_sha256 as parent_source_sha256,
)
from riskshiftbench.experiments.real_agent_validation import (
    _coding_task,
    _research_task,
    _workflow_task,
    canonical_sha256,
)


CONFIG = Path("riskshiftbench/configs/real_agent_high_budget_extension_v1.json")
PARENT_RAW = Path("riskshiftbench/artifacts/real_agent_high_budget_v1/raw_trajectories.json")
PARENT_RESULT = Path("riskshiftbench/artifacts/real_agent_high_budget_v1/results.json")
ARTIFACT_DIR = Path("riskshiftbench/artifacts/real_agent_high_budget_extension_v1")
MANIFEST = ARTIFACT_DIR / "task_manifest.json"
RAW = ARTIFACT_DIR / "raw_trajectories.json"
OUTPUT = ARTIFACT_DIR / "results.json"


def source_sha256() -> str:
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(parent_source_sha256().encode("utf-8"))
    return digest.hexdigest()


def _load_hashed(path: Path, digest_key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != digest_key}
    if payload[digest_key] != canonical_sha256(body):
        raise RuntimeError(f"digest mismatch: {path}")
    return payload


def validate_config(config: dict, parent_raw: dict, parent_result: dict) -> None:
    if config["protocol_id"] != "riskshiftbench-real-agent-high-budget-extension-v1":
        raise RuntimeError("unexpected extension protocol")
    if not config["design_locked_before_extension_inference"]:
        raise RuntimeError("extension must be locked before inference")
    if config["extension_outcomes_observed_at_lock"]:
        raise RuntimeError("lock cannot follow extension outcome access")
    if config["parent_result_sha256"] != parent_result["result_sha256"]:
        raise RuntimeError("parent result changed")
    if config["parent_raw_sha256"] != parent_raw["raw_sha256"]:
        raise RuntimeError("parent raw data changed")
    if int(config["additional_tasks_per_family"]) * 2 != int(
        config["cumulative_tasks_per_family"]
    ):
        raise RuntimeError("extension must double the parent task count")
    if not config["no_replacement_after_outcome_access"]:
        raise RuntimeError("no-replacement rule is required")


def generate_tasks(config: dict, parent_raw: dict, parent_result: dict) -> list[dict]:
    validate_config(config, parent_raw, parent_result)
    builders = {
        "coding": _coding_task,
        "workflow": _workflow_task,
        "research": _research_task,
    }
    rng = random.Random(int(config["extension_seed"]))
    tasks = []
    for family, builder in builders.items():
        for index in range(int(config["additional_tasks_per_family"])):
            payload = builder(rng, index, config["shift"], config)
            tasks.append(
                {
                    "id": f"high-budget-extension-{family}-{index:05d}",
                    "split": "fresh-pilot-extension",
                    "family": family,
                    "shift": config["shift"],
                    "option_seed": rng.randrange(2**31),
                    **payload,
                }
            )
    return tasks


def prepare_manifest(config: dict, parent_raw: dict, parent_result: dict) -> dict:
    payload = {
        "protocol_id": config["protocol_id"],
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": canonical_sha256(config),
        "source_sha256": source_sha256(),
        "parent_raw_sha256": parent_raw["raw_sha256"],
        "parent_result_sha256": parent_result["result_sha256"],
        "tasks": generate_tasks(config, parent_raw, parent_result),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def load_manifest(config: dict, parent_raw: dict, parent_result: dict) -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise RuntimeError("extension manifest digest mismatch")
    if payload["config_sha256"] != canonical_sha256(config):
        raise RuntimeError("extension config changed after lock")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("extension source changed after lock")
    if payload["parent_raw_sha256"] != parent_raw["raw_sha256"]:
        raise RuntimeError("extension parent raw changed")
    if payload["parent_result_sha256"] != parent_result["result_sha256"]:
        raise RuntimeError("extension parent result changed")
    return payload


def execute(config: dict, manifest: dict) -> dict:
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
    if RAW.exists():
        prior = _load_hashed(RAW, "raw_sha256")
        if (
            prior.get("manifest_sha256") == manifest["manifest_sha256"]
            and prior.get("source_sha256") == source_sha256()
        ):
            trajectories = prior["trajectories"]
            completed = set(prior["completed_runs"])

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
            scored = original_score(
                model_arg, tokenizer_arg, missing, config_arg
            )
            cache.update(zip(missing, scored))
        return [cache[prompt] for prompt in prompts]

    engine._score_prompt_batch = cached_score
    try:
        for spec in _run_specs(config):
            if spec["run_id"] in completed:
                continue
            tasks = [
                row for row in manifest["tasks"] if row["family"] == spec["family"]
            ]
            started = time.time()
            rows = engine.run_trajectories(
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
                "prompt_cache_entries": len(cache),
                "trajectories": trajectories,
            }
            payload["raw_sha256"] = canonical_sha256(payload)
            RAW.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"{spec['run_id']}: {len(rows)} episodes, {len(cache)} cached prompts, "
                f"{time.time() - started:.1f}s"
            )
    finally:
        engine._score_prompt_batch = original_score
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def load_extension_raw(manifest: dict) -> dict:
    payload = _load_hashed(RAW, "raw_sha256")
    if payload["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("extension raw data use another manifest")
    if payload["source_sha256"] != source_sha256():
        raise RuntimeError("extension raw data use another source")
    return payload


def analyze(config: dict, manifest: dict, parent_raw: dict, extension_raw: dict) -> dict:
    cumulative_raw = {
        "raw_sha256": canonical_sha256(
            {
                "parent": parent_raw["raw_sha256"],
                "extension": extension_raw["raw_sha256"],
            }
        ),
        "trajectories": parent_raw["trajectories"]
        + extension_raw["trajectories"],
    }
    analysis_config = dict(config)
    analysis_config["pilot_tasks_per_family"] = int(
        config["cumulative_tasks_per_family"]
    )
    payload = analyze_family(analysis_config, manifest, cumulative_raw, OUTPUT)
    payload.pop("result_sha256", None)
    payload.update(
        {
            "protocol_id": config["protocol_id"],
            "evidential_status": config["evidential_status"],
            "selection_disclosure": config["selection_disclosure"],
            "config_sha256": canonical_sha256(config),
            "source_sha256": source_sha256(),
            "parent_raw_sha256": parent_raw["raw_sha256"],
            "extension_raw_sha256": extension_raw["raw_sha256"],
            "raw_sha256": cumulative_raw["raw_sha256"],
        }
    )
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
    parent_raw = _load_hashed(PARENT_RAW, "raw_sha256")
    parent_result = _load_hashed(PARENT_RESULT, "result_sha256")
    if args.prepare_only:
        payload = prepare_manifest(config, parent_raw, parent_result)
        print(json.dumps({key: value for key, value in payload.items() if key != "tasks"}, indent=2))
        return
    manifest = load_manifest(config, parent_raw, parent_result)
    extension_raw = load_extension_raw(manifest) if args.analyze_only else execute(config, manifest)
    payload = analyze(config, manifest, parent_raw, extension_raw)
    print(json.dumps({key: value for key, value in payload.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()
