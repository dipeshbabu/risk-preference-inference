"""Efficient analysis amendment for the locked workflow replication.

The locked runner completed inference but its bootstrap called the quadratic
AUROC diagnostic inside every resample.  That analysis was interrupted before
it emitted a result.  This amendment leaves the raw trajectories and all
scientific definitions unchanged and vectorizes only the prespecified mean and
drift bootstrap statistics.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from statistics import fmean

import numpy as np

from riskshiftbench.experiments.controlled_verifier_drift import (
    mean_interval,
    wilson_interval,
)
from riskshiftbench.experiments.real_agent_validation import (
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
)
from riskshiftbench.experiments.verifier_drift_control import VDCPlan, decide_vdc
from riskshiftbench.experiments.workflow_drift_only_replication import (
    BASE_CONFIG,
    CONFIG,
    MANIFEST,
    RAW,
    load_manifest,
    load_raw,
    score_downside_metrics,
)


OUTPUT = Path(
    "riskshiftbench/artifacts/workflow_drift_only_replication_v1/results_amended_v2.json"
)


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    manifest = load_manifest(config, base, MANIFEST)
    raw = load_raw(manifest, RAW)
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

    effects = np.asarray(
        [c["score"] - f["score"] for f, c in zip(fallback, candidate)], dtype=float
    )
    paired_drift = np.asarray(
        [
            [
                float(high_confidence_failure(c, threshold))
                - float(high_confidence_failure(f, threshold))
                for threshold in thresholds
            ]
            for f, c in zip(fallback, candidate)
        ],
        dtype=float,
    )
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    mean_draws = []
    drift_draws = []
    for _ in range(int(config["bootstrap_draws"])):
        indices = rng.integers(0, len(effects), size=len(effects))
        mean_draws.append(float(effects[indices].mean()))
        drift_draws.append(float(paired_drift[indices].mean(axis=0).max()))
    mean_draws.sort()
    drift_draws.sort()
    lower_index = int(0.025 * len(mean_draws))
    upper_index = int(0.975 * len(mean_draws)) - 1

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
        stream_effects = [
            c["score"] - f["score"] for f, c in zip(f_stream, c_stream)
        ]
        candidate_h = {
            coverage: [high_confidence_failure(row, threshold) for row in c_stream]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        fallback_h = {
            coverage: [high_confidence_failure(row, threshold) for row in f_stream]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        decision = decide_vdc(
            stream_effects,
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

    detections = sum(row["point_drift_detected"] for row in streams)
    payload = {
        "protocol_id": "riskshiftbench-workflow-drift-only-replication-v1-analysis-amendment-v2",
        "evidential_status": config["evidential_status"],
        "selection_disclosure": config["selection_disclosure"],
        "analysis_amendment": {
            "status": "post-lock computational correction before result access",
            "reason": (
                "The locked analysis recomputed quadratic AUROC inside each "
                "bootstrap draw. It was interrupted before emitting a result."
            ),
            "unchanged": [
                "raw trajectories",
                "task manifest",
                "confidence mapping",
                "thresholds",
                "eligibility conditions",
                "stream partition",
                "bootstrap draws and seed",
            ],
            "amendment_source_sha256": source_sha256(),
        },
        "config_sha256": canonical_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
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
            "point_drift_detection_frequency": detections / len(streams),
            "point_drift_detection_wilson_interval_95": wilson_interval(
                detections, len(streams)
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
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "independent_streams"}, indent=2))


if __name__ == "__main__":
    main()
