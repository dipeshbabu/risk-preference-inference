"""Recompute the 24-update deployment p-values from raw pilot trajectories."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from riskshiftbench.experiments import prospective_full_vs_absolute as study
from riskshiftbench.experiments.real_agent_validation import calibrate_episode
from riskshiftbench.experiments.vdc_efficient import (
    absolute_family_decisions,
    efficient_family_decisions,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "riskshiftbench/artifacts/prospective_full_vs_absolute_v1"


def main() -> None:
    config = json.loads(study.CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT / "target_manifest_v3.json").read_text(encoding="utf-8"))
    result = json.loads((ARTIFACT / "results_v3.json").read_text(encoding="utf-8"))
    with gzip.open(ARTIFACT / "target_trajectories_v3.json.gz", "rt", encoding="utf-8") as handle:
        raw = json.load(handle)

    evidence = {}
    for model in config["models"]:
        for family in config["families"]:
            calibration = manifest["calibration"][f"{model['id']}::{family}"]
            temperature = float(calibration["temperature"])
            thresholds = [float(value) for value in calibration["thresholds"]]
            fallback_id = f"{model['id']}::{family}::fallback"
            fallback = [
                calibrate_episode(row, temperature)
                for row in raw["trajectories"]
                if row["run_id"] == fallback_id and row["split"] == "pilot"
            ]
            for pair in manifest["pairs"]:
                if pair["model_id"] != model["id"] or pair["family"] != family:
                    continue
                candidate = [
                    calibrate_episode(row, temperature)
                    for row in raw["trajectories"]
                    if row["run_id"] == pair["id"] and row["split"] == "pilot"
                ]
                evidence[pair["id"]] = study._evidence(fallback, candidate, thresholds)

    kwargs = study._vdc_kwargs(config)
    absolute = absolute_family_decisions(
        evidence, **kwargs, confidence_unresolved_reason="recalibrate-verifier"
    )
    full = efficient_family_decisions(
        evidence,
        **kwargs,
        maximum_verifier_drift=float(config["maximum_operational_verifier_drift"]),
        confidence_unresolved_reason="recalibrate-verifier",
    )
    for task in evidence:
        if absolute[task].component_pvalues != result["component_pvalues"]["vdc_absolute"][task]:
            raise RuntimeError(f"VDC-Absolute p-value mismatch: {task}")
        if full[task].component_pvalues != result["component_pvalues"]["vdc_full"][task]:
            raise RuntimeError(f"VDC-Full p-value mismatch: {task}")
        if absolute[task].reason != result["methods"]["vdc_absolute"]["route"][task]:
            raise RuntimeError(f"VDC-Absolute route mismatch: {task}")
        if full[task].reason != result["methods"]["vdc_full"]["route"][task]:
            raise RuntimeError(f"VDC-Full route mismatch: {task}")

    payload = {
        "protocol_id": result["protocol_id"],
        "updates_checked": len(evidence),
        "absolute_pvalues_match": True,
        "full_pvalues_match": True,
        "routes_match": True,
        "route_disagreements": result["routes_changed_by_relative_drift"],
    }
    (ARTIFACT / "pvalue_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
