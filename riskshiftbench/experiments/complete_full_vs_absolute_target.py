"""Complete the missing 1.5B target runs for the locked comparison.

This is an execution-only amendment. It does not change the locked config,
update mechanisms, target budget, or analysis. Existing 1.5B coding runs are
left untouched; missing workflow/research runs are written as independently
hashed per-run bundles and can be resumed safely after interruption.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import time
from pathlib import Path

import riskshiftbench.experiments.real_agent_validation as engine
from riskshiftbench.experiments import prospective_full_vs_absolute as primary


CONFIG = primary.CONFIG
MANIFEST = primary.TARGET_MANIFEST
OUT_DIR = primary.ARTIFACT_DIR / "target_qwen15_runs_v3"
MODEL_ID = "qwen2.5-1.5b-instruct"
# Execution-only throughput amendment. The model, prompts, tasks, seeds, and
# decision rule remain locked; padding/attention masks make this batching
# change prediction-preserving for the same prompt sequence.
EXECUTION_BATCH_SIZE = 64


def _write_run(path: Path, manifest: dict, run_id: str, rows: list[dict]) -> None:
    payload = {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "primary_source_sha256": primary.source_sha256(),
        "amendment_source_sha256": primary.source_sha256(),
        "run_id": run_id,
        "execution_batch_size": EXECUTION_BATCH_SIZE,
        "trajectories": rows,
    }
    payload["run_sha256"] = primary.canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))


def _read_run(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    body = {key: value for key, value in payload.items() if key != "run_sha256"}
    if payload["run_sha256"] != primary.canonical_sha256(body):
        raise RuntimeError(f"run digest mismatch: {path}")
    return payload


def _run_path(run_id: str) -> Path:
    safe = run_id.replace("::", "__")
    return OUT_DIR / f"{safe}.json.gz"


def validate_manifest(manifest: dict) -> None:
    pairs = [row for row in manifest["pairs"] if row["model_id"] == MODEL_ID]
    expected = {f"{MODEL_ID}::{family}::fallback" for family in {row["family"] for row in pairs}}
    expected.update(row["id"] for row in pairs)
    missing = [run_id for run_id in sorted(expected) if not _run_path(run_id).exists()]
    print(json.dumps({"expected_runs": len(expected), "already_present": len(expected) - len(missing), "missing": missing}, indent=2))


def execute(config: dict, manifest: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = [row for row in manifest["pairs"] if row["model_id"] == MODEL_ID]
    families = sorted({row["family"] for row in pairs})
    target_tasks = {
        family: [row for row in manifest["tasks"] if row["family"] == family]
        for family in families
    }
    model_spec = next(row for row in config["models"] if row["id"] == MODEL_ID)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.float16,
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
        ordered = []
        for family in families:
            ordered.append(
                {
                    "id": f"{MODEL_ID}::{family}::fallback",
                    "family": family,
                    "memory_mode": "none",
                    "system_prompt": engine.BASE_SYSTEM[family],
                    "role": "fallback",
                }
            )
            ordered.extend(
                {
                    **row,
                    "role": "candidate",
                }
                for row in pairs
                if row["family"] == family
            )

        for pair in ordered:
            path = _run_path(pair["id"])
            if path.exists():
                _read_run(path)
                print(f"skip {pair['id']} (validated existing run)", flush=True)
                continue
            started = time.time()
            rows = primary._run_trajectories(
                model,
                tokenizer,
                target_tasks[pair["family"]],
                pair,
                pair["role"],
                config,
                EXECUTION_BATCH_SIZE,
            )
            for row in rows:
                row["run_id"] = pair["id"]
                row["model_id"] = MODEL_ID
            _write_run(path, manifest, pair["id"], rows)
            print(
                f"completed {pair['id']}: {len(rows)} episodes, "
                f"{len(cache)} cached prompts, {time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        engine._score_prompt_batch = original_score
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run"))
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.action == "validate":
        validate_manifest(manifest)
    else:
        execute(config, manifest)
        validate_manifest(manifest)


if __name__ == "__main__":
    main()
