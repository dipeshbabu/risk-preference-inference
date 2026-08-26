"""Apply the score-downside analysis correction to the locked real-agent run.

The locked v1 analysis accidentally classified a material loss when either the
paired score effect crossed ``-gamma`` or the candidate alone triggered an
unsafe-action flag.  The protocol and manuscript define material downside only
as ``Pr(Z < -gamma)``.  This amendment changes that one derived quantity and
recomputes reference labels and routes from the unchanged raw trajectories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import riskshiftbench.experiments.real_agent_validation as locked


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "riskshiftbench" / "configs" / "real_agent_validation_v1.json"
MANIFEST = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "task_manifest.json"
RAW = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "raw_trajectories.json"
ORIGINAL = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "results.json"
OUTPUT = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "results_amended_v2.json"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


_ORIGINAL_PAIRED_METRICS = locked.paired_metrics


def score_downside_metrics(
    fallback: list[dict], candidate: list[dict], thresholds: list[float]
) -> dict:
    metrics = _ORIGINAL_PAIRED_METRICS(fallback, candidate, thresholds)
    metrics["material_loss_probability"] = fmean(
        candidate_row["score"] - fallback_row["score"] < -0.5
        for fallback_row, candidate_row in zip(fallback, candidate)
    )
    return metrics


def _never_deploy_trials(updates: list[dict], streams: int) -> list[dict]:
    route = {row["pair_id"]: "retain" for row in updates}
    return [
        {
            "gain": 0.0,
            "promotions": 0,
            "ineligible_promotions": 0,
            "any_ineligible_promotion": False,
            "actions": {"retain": len(updates)},
            "route": route,
        }
        for _ in range(streams)
    ]


def _summarize_never_deploy(trials: list[dict]) -> dict:
    return {
        "mean_gain": 0.0,
        "gain_interval_95": [0.0, 0.0],
        "mean_promotions": 0.0,
        "mean_ineligible_promotions": 0.0,
        "probability_any_ineligible_promotion": 0.0,
        "mean_actions": {
            "deploy": 0.0,
            "recalibrate": 0.0,
            "retain": float(len(trials[0]["route"])),
            "unresolved": 0.0,
        },
        "mean_pilot_pairs": 0.0,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = locked.load_locked_manifest(config, MANIFEST)
    raw = locked.load_raw(config, manifest, RAW)
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    original_body = {
        key: value for key, value in original.items() if key != "result_sha256"
    }
    if original["result_sha256"] != locked.canonical_sha256(original_body):
        raise RuntimeError("original result digest mismatch")

    locked.paired_metrics = score_downside_metrics
    try:
        payload = locked.analyze(config, manifest, raw, OUTPUT)
    finally:
        locked.paired_metrics = _ORIGINAL_PAIRED_METRICS

    payload.pop("result_sha256", None)
    streams = int(config["pilot_streams"])
    never_trials = _never_deploy_trials(payload["updates"], streams)
    payload["trials"]["always_fallback"] = never_trials
    payload["methods"]["always_fallback"] = _summarize_never_deploy(never_trials)

    fixed_cost = float(
        len(payload["updates"]) * int(config["pilot_episodes_per_update"])
    )
    for method, result in payload["methods"].items():
        result.pop("any_ineligible_promotion_wilson_interval_95", None)
        if method not in {"always_fallback", "oracle_drift"}:
            result["mean_pilot_pairs"] = fixed_cost
        elif method == "oracle_drift":
            result["mean_pilot_pairs"] = 0.0

    payload["protocol_id"] = "riskshiftbench-real-agent-validation-v1-analysis-amendment-v2"
    payload["analysis_amendment"] = {
        "status": "post-lock mechanical correction",
        "reason": (
            "The v1 analysis included candidate-only unsafe actions in q, while "
            "the locked protocol defines q as Pr(Z < -gamma)."
        ),
        "unchanged": [
            "raw trajectories",
            "model outputs",
            "task manifest",
            "confidence calibration",
            "threshold grid",
            "VDC score-downside decision rule",
        ],
        "original_result_sha256": original["result_sha256"],
        "amendment_source_sha256": source_sha256(),
    }
    payload["route_frequency_interpretation"] = (
        "conditional bootstrap stability on the fixed 100-episode pilot pool; "
        "not independent-stream population inference"
    )
    payload["common_reference_check"] = {
        "frozen_and_recalibrated_labels_identical": all(
            row["frozen_eligible"] == row["recalibrated_eligible"]
            for row in payload["updates"]
        )
    }
    payload["result_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result_sha256": payload["result_sha256"],
                "eligible_updates": [
                    row["pair_id"] for row in payload["updates"] if row["frozen_eligible"]
                ],
                "methods": payload["methods"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
