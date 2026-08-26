"""Merge completed target bundles and analyze the locked comparison.

The parent target manifest is preserved. This creates an explicit v3 amendment
manifest and result bundle whose hashes identify the source and all raw runs.
It refuses to analyze until every expected model/family/update run is present.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

from riskshiftbench.experiments import prospective_full_vs_absolute as primary
from riskshiftbench.experiments.complete_full_vs_absolute_target import (
    OUT_DIR,
    _read_run,
)


V2_MANIFEST = primary.TARGET_MANIFEST
V3_MANIFEST = primary.ARTIFACT_DIR / "target_manifest_v3.json"
V3_RAW = primary.ARTIFACT_DIR / "target_trajectories_v3.json.gz"
V3_RESULTS = primary.ARTIFACT_DIR / "results_v3.json"
QWEN05_RAW = primary.ARTIFACT_DIR / "target_trajectories_v2.json.gz"


def _load_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(parent: dict) -> dict:
    body = {key: value for key, value in parent.items() if key != "manifest_sha256"}
    body.update(
        {
            "amendment_status": "completed_target_collection_from_resumable_runs",
            "parent_manifest_sha256": parent["manifest_sha256"],
            "source_sha256": primary.source_sha256(),
            "manifest_version": "v3",
        }
    )
    body["manifest_sha256"] = primary.canonical_sha256(body)
    V3_MANIFEST.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return body


def merge_runs(manifest: dict) -> dict:
    primary.validate_complete_target_raw  # keep the canonical validator in scope
    qwen05 = _load_gzip(QWEN05_RAW)
    trajectories = list(qwen05["trajectories"])
    completed = set(qwen05.get("completed_runs", []))
    expected_qwen15 = {
        *(f"qwen2.5-1.5b-instruct::{family}::fallback"
          for family in {row["family"] for row in manifest["pairs"]
                         if row["model_id"] == "qwen2.5-1.5b-instruct"}),
        *(row["id"] for row in manifest["pairs"]
          if row["model_id"] == "qwen2.5-1.5b-instruct"),
    }
    for run_id in sorted(expected_qwen15):
        path = OUT_DIR / f"{run_id.replace('::', '__')}.json.gz"
        if not path.exists():
            raise RuntimeError(f"missing 1.5B run bundle: {run_id}")
        payload = _read_run(path)
        if payload["manifest_sha256"] != manifest["parent_manifest_sha256"]:
            raise RuntimeError(f"run bundle uses another parent manifest: {run_id}")
        trajectories.extend(payload["trajectories"])
        completed.add(run_id)
    body = {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": primary.source_sha256(),
        "completed_runs": sorted(completed),
        "trajectories": trajectories,
    }
    body["raw_sha256"] = primary.canonical_sha256(body)
    with gzip.open(V3_RAW, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(body, handle, sort_keys=True, separators=(",", ":"))
    return body


def analyze(config: dict, manifest: dict, raw: dict) -> dict:
    primary.validate_complete_target_raw(manifest, raw)
    # analyze_target writes the canonical payload; retain an explicitly named
    # v3 copy so the amendment cannot be confused with the incomplete v2 run.
    payload = primary.analyze_target(config, manifest, raw)
    shutil.copyfile(primary.TARGET_RESULTS, V3_RESULTS)
    return payload


def main() -> None:
    config = json.loads(primary.CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    manifest = build_manifest(parent)
    raw = merge_runs(manifest)
    payload = analyze(config, manifest, raw)
    print(json.dumps({
        "result": str(V3_RESULTS),
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_sha256": raw["raw_sha256"],
        "executed_agent_episodes": payload["executed_agent_episodes"],
        "executed_actions": payload["executed_actions"],
    }, indent=2))


if __name__ == "__main__":
    main()
