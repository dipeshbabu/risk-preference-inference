"""Post-lock component and alternative-drift diagnostics for VDC."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from riskshiftbench.experiments.real_agent_validation import (
    calibrate_episode,
    canonical_sha256,
    high_confidence_failure,
)
from riskshiftbench.experiments.verifier_drift_control import (
    VDCPlan,
    estimate_vdc_bounds,
    sufficient_interval_resolution_samples,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "riskshiftbench" / "configs" / "real_agent_validation_v1.json"
RAW = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "raw_trajectories.json"
RESULTS = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "results_amended_v2.json"
OUTPUT = ROOT / "riskshiftbench" / "artifacts" / "real_agent_validation_v1" / "vdc_component_diagnostics.json"


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_hashed(path: Path, digest_key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != digest_key}
    if payload[digest_key] != canonical_sha256(body):
        raise RuntimeError(f"digest mismatch: {path}")
    return payload


def _commitment_failure(episode: dict) -> bool:
    return any(row["commitment_failure"] for row in episode["actions"])


def _aurc(episodes: list[dict]) -> float:
    ordered = sorted(
        episodes, key=lambda row: row["episode_confidence"], reverse=True
    )
    failure = [float(_commitment_failure(row)) for row in ordered]
    coverages = [index / 20 for index in range(1, 21)]
    selective_risk = []
    for coverage in coverages:
        count = max(1, math.ceil(coverage * len(failure)))
        selective_risk.append(fmean(failure[:count]))
    return fmean(selective_risk)


def _sufficient_samples(alpha: float, margin: float, range_width: float) -> int | None:
    if margin <= 0:
        return None
    return sufficient_interval_resolution_samples(
        tail_alpha=alpha, margin=margin, range_width=range_width
    )


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    raw_body = {key: value for key, value in raw.items() if key != "raw_sha256"}
    if raw["raw_sha256"] != canonical_sha256(raw_body):
        raise RuntimeError("raw trajectory digest mismatch")
    results = _load_hashed(RESULTS, "result_sha256")
    trajectories = raw["trajectories"]
    plan = VDCPlan(
        confidence_thresholds=(0.1, 0.3, 0.5),
        family_tasks=6,
        declared_looks=1,
        family_alpha=0.05,
        mean_margin=0.02,
        material_loss_threshold=0.5,
        maximum_material_loss_probability=0.10,
        maximum_candidate_high_confidence_failure_probability=0.15,
        maximum_verifier_drift=0.05,
    )
    component_rows = []
    alternative_rows = []
    for update in results["updates"]:
        pair_id = update["pair_id"]
        temperature = float(update["fallback_temperature"])
        thresholds = [float(value) for value in update["thresholds"]]
        calibrated = {
            role: [
                calibrate_episode(row, temperature)
                for row in trajectories
                if row["pair_id"] == pair_id
                and row["role"] == role
                and row["split"] in {"pilot", "reference"}
            ]
            for role in ("fallback", "candidate")
        }
        pilot = {
            role: [row for row in rows if row["split"] == "pilot"]
            for role, rows in calibrated.items()
        }
        reference = {
            role: [row for row in rows if row["split"] == "reference"]
            for role, rows in calibrated.items()
        }
        effects = [
            c["score"] - f["score"]
            for f, c in zip(pilot["fallback"], pilot["candidate"])
        ]
        candidate_h = {
            coverage: [high_confidence_failure(row, threshold) for row in pilot["candidate"]]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        fallback_h = {
            coverage: [high_confidence_failure(row, threshold) for row in pilot["fallback"]]
            for coverage, threshold in zip((0.1, 0.3, 0.5), thresholds)
        }
        bounds = estimate_vdc_bounds(effects, candidate_h, fallback_h, plan)
        blockers = []
        if bounds.mean.lower <= 0.02:
            blockers.append("mean")
        if bounds.downside.upper >= 0.10:
            blockers.append("downside")
        if bounds.candidate_confidence_risk.upper >= 0.15:
            blockers.append("absolute-risk")
        if bounds.verifier_drift.upper >= 0.05:
            blockers.append("drift")

        metrics = update["frozen_reference"]
        if update["frozen_eligible"]:
            margins = {
                "mean": metrics["mean_effect"] - 0.02,
                "downside": 0.10 - metrics["material_loss_probability"],
                "absolute-risk": 0.15
                - metrics["candidate_high_confidence_failure"],
                "drift": 0.05 - metrics["verifier_drift_max"],
            }
            full_alpha = plan.tail_alpha
            taskwise_alpha = 0.05 / (2 * 1 * 1 * 8)
            single_threshold_alpha = 0.05 / (2 * 6 * 1 * 4)
            sample_bounds = {
                "full": {
                    key: _sufficient_samples(
                        full_alpha, value, 2.0 if key in {"mean", "drift"} else 1.0
                    )
                    for key, value in margins.items()
                },
                "taskwise_no_family": {
                    key: _sufficient_samples(
                        taskwise_alpha,
                        value,
                        2.0 if key in {"mean", "drift"} else 1.0,
                    )
                    for key, value in margins.items()
                },
                "single_threshold": {
                    key: _sufficient_samples(
                        single_threshold_alpha,
                        value,
                        2.0 if key in {"mean", "drift"} else 1.0,
                    )
                    for key, value in margins.items()
                },
                "full_without_drift": {
                    key: _sufficient_samples(
                        0.05 / (2 * 6 * 1 * 7),
                        value,
                        2.0 if key == "mean" else 1.0,
                    )
                    for key, value in margins.items()
                    if key != "drift"
                },
            }
        else:
            margins = None
            sample_bounds = None
        component_rows.append(
            {
                "pair_id": pair_id,
                "reference_eligible": update["frozen_eligible"],
                "pilot_observations": len(effects),
                "blocking_components": blockers,
                "bounds": {
                    "mean": [bounds.mean.lower, bounds.mean.upper],
                    "downside": [bounds.downside.lower, bounds.downside.upper],
                    "absolute-risk": [
                        bounds.candidate_confidence_risk.lower,
                        bounds.candidate_confidence_risk.upper,
                    ],
                    "drift": [
                        bounds.verifier_drift.lower,
                        bounds.verifier_drift.upper,
                    ],
                },
                "reference_margins": margins,
                "sufficient_sample_bounds": sample_bounds,
            }
        )
        alternative_rows.append(
            {
                "pair_id": pair_id,
                "max_threshold_drift": metrics["verifier_drift_max"],
                "average_threshold_drift": metrics["verifier_drift_average"],
                "candidate_aurc": _aurc(reference["candidate"]),
                "fallback_aurc": _aurc(reference["fallback"]),
                "aurc_drift": _aurc(reference["candidate"])
                - _aurc(reference["fallback"]),
                **(
                    lambda index, threshold: {
                        "max_drift_threshold": threshold,
                        "candidate_coverage": fmean(
                            row["episode_confidence"] >= threshold
                            for row in reference["candidate"]
                        ),
                        "fallback_coverage": fmean(
                            row["episode_confidence"] >= threshold
                            for row in reference["fallback"]
                        ),
                        "candidate_selective_risk": (
                            fmean(
                                high_confidence_failure(row, threshold)
                                for row in reference["candidate"]
                            )
                            / max(
                                fmean(
                                    row["episode_confidence"] >= threshold
                                    for row in reference["candidate"]
                                ),
                                1e-12,
                            )
                        ),
                        "fallback_selective_risk": (
                            fmean(
                                high_confidence_failure(row, threshold)
                                for row in reference["fallback"]
                            )
                            / max(
                                fmean(
                                    row["episode_confidence"] >= threshold
                                    for row in reference["fallback"]
                                ),
                                1e-12,
                            )
                        ),
                    }
                )(
                    max(
                        range(len(metrics["drift_profile"])),
                        key=lambda index: metrics["drift_profile"][index],
                    ),
                    thresholds[
                        max(
                            range(len(metrics["drift_profile"])),
                            key=lambda index: metrics["drift_profile"][index],
                        )
                    ],
                ),
            }
        )
        row = alternative_rows[-1]
        row["coverage_change"] = (
            row["candidate_coverage"] - row["fallback_coverage"]
        )
        row["selective_risk_change"] = (
            row["candidate_selective_risk"] - row["fallback_selective_risk"]
        )
        if row["coverage_change"] > 0 and row["selective_risk_change"] > 0:
            row["mechanism"] = "mixed-dangerous"
        elif row["coverage_change"] > 0:
            row["mechanism"] = "coverage-driven"
        elif row["selective_risk_change"] > 0:
            row["mechanism"] = "selective-risk-driven"
        else:
            row["mechanism"] = "both-improve"
    payload = {
        "diagnostic_status": "post-lock component and construct diagnostic",
        "source_sha256": source_sha256(),
        "amended_result_sha256": results["result_sha256"],
        "component_resolution": component_rows,
        "alternative_drift": alternative_rows,
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
