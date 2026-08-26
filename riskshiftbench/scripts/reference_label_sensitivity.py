"""Reference-label sensitivity for the completed 24-update comparison."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from riskshiftbench.experiments import prospective_full_vs_absolute as study
from riskshiftbench.experiments.real_agent_validation import calibrate_episode
from riskshiftbench.experiments.verifier_drift_control import VDCPlan, estimate_vdc_bounds


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "riskshiftbench/artifacts/prospective_full_vs_absolute_v1"
OUTPUT = ARTIFACT / "reference_label_sensitivity.json"
CSV_OUTPUT = ARTIFACT / "reference_label_sensitivity.csv"
TABLE_OUTPUT = ROOT / "paper/tables/reference_label_sensitivity.tex"


def labels(bounds, plan: VDCPlan) -> tuple[str, str]:
    absolute_eligible = (
        bounds.mean.lower > plan.mean_margin
        and bounds.downside.upper < plan.maximum_material_loss_probability
        and bounds.candidate_confidence_risk.upper
        < plan.maximum_candidate_high_confidence_failure_probability
    )
    absolute_ineligible = (
        bounds.mean.upper <= plan.mean_margin
        or bounds.downside.lower >= plan.maximum_material_loss_probability
        or bounds.candidate_confidence_risk.lower
        >= plan.maximum_candidate_high_confidence_failure_probability
    )
    absolute = (
        "eligible" if absolute_eligible else "ineligible" if absolute_ineligible else "uncertain"
    )
    full_eligible = absolute_eligible and bounds.verifier_drift.upper < plan.maximum_verifier_drift
    full_ineligible = absolute_ineligible or bounds.verifier_drift.lower >= plan.maximum_verifier_drift
    full = "eligible" if full_eligible else "ineligible" if full_ineligible else "uncertain"
    return absolute, full


def method_summary(route: dict[str, str], label_map: dict[str, str], final_effects: dict[str, float]) -> dict:
    promoted = {task for task, action in route.items() if action == "promote"}
    eligible = {task for task, label in label_map.items() if label == "eligible"}
    ineligible = {task for task, label in label_map.items() if label == "ineligible"}
    uncertain = {task for task, label in label_map.items() if label == "uncertain"}
    return {
        "promotions": len(promoted),
        "resolved_ineligible_promotions": len(promoted & ineligible),
        "uncertain_promotions": len(promoted & uncertain),
        "resolved_eligible_recall": len(promoted & eligible) / len(eligible) if eligible else 0.0,
        "deployment_gain": sum(final_effects[task] for task in promoted) / len(route),
    }


def main() -> None:
    config = json.loads(study.CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT / "target_manifest_v3.json").read_text(encoding="utf-8"))
    result = json.loads((ARTIFACT / "results_v3.json").read_text(encoding="utf-8"))
    with gzip.open(ARTIFACT / "target_trajectories_v3.json.gz", "rt", encoding="utf-8") as handle:
        raw = json.load(handle)

    rows = []
    for model in config["models"]:
        for family in config["families"]:
            calibration = manifest["calibration"][f"{model['id']}::{family}"]
            temperature = float(calibration["temperature"])
            thresholds = tuple(float(value) for value in calibration["thresholds"])
            plan = VDCPlan(
                confidence_thresholds=thresholds,
                family_tasks=24,
                declared_looks=1,
                family_alpha=float(config["family_alpha"]),
                mean_margin=float(config["mean_margin"]),
                material_loss_threshold=float(config["material_loss_threshold"]),
                maximum_material_loss_probability=float(config["maximum_material_loss_probability"]),
                maximum_candidate_high_confidence_failure_probability=float(
                    config["maximum_candidate_high_confidence_failure_probability"]
                ),
                maximum_verifier_drift=float(config["maximum_operational_verifier_drift"]),
            )
            fallback_id = f"{model['id']}::{family}::fallback"
            fallback = [
                calibrate_episode(row, temperature)
                for row in raw["trajectories"]
                if row["run_id"] == fallback_id and row["split"] == "reference"
            ]
            for pair in manifest["pairs"]:
                if pair["model_id"] != model["id"] or pair["family"] != family:
                    continue
                candidate = [
                    calibrate_episode(row, temperature)
                    for row in raw["trajectories"]
                    if row["run_id"] == pair["id"] and row["split"] == "reference"
                ]
                evidence = study._evidence(fallback, candidate, list(thresholds))
                bounds = estimate_vdc_bounds(*evidence, plan)
                absolute, full = labels(bounds, plan)
                rows.append(
                    {
                        "task_id": pair["id"],
                        "absolute_interval_label": absolute,
                        "full_interval_label": full,
                        "mean_lower": bounds.mean.lower,
                        "mean_upper": bounds.mean.upper,
                        "downside_lower": bounds.downside.lower,
                        "downside_upper": bounds.downside.upper,
                        "absolute_risk_lower": bounds.candidate_confidence_risk.lower,
                        "absolute_risk_upper": bounds.candidate_confidence_risk.upper,
                        "drift_lower": bounds.verifier_drift.lower,
                        "drift_upper": bounds.verifier_drift.upper,
                    }
                )

    absolute_labels = {row["task_id"]: row["absolute_interval_label"] for row in rows}
    full_labels = {row["task_id"]: row["full_interval_label"] for row in rows}
    final_effects = {
        task: values["mean_effect"] for task, values in result["point_metrics"]["final"].items()
    }
    payload = {
        "protocol_id": result["protocol_id"],
        "reference_pairs_per_update": 1000,
        "family_alpha": float(config["family_alpha"]),
        "tail_alpha": VDCPlan(
            confidence_thresholds=(0.1, 0.3, 0.5), family_tasks=24, declared_looks=1
        ).tail_alpha,
        "absolute_label_counts": dict(Counter(absolute_labels.values())),
        "full_label_counts": dict(Counter(full_labels.values())),
        "vdc_absolute_under_absolute_labels": method_summary(
            result["methods"]["vdc_absolute"]["route"], absolute_labels, final_effects
        ),
        "vdc_full_under_full_labels": method_summary(
            result["methods"]["vdc_full"]["route"], full_labels, final_effects
        ),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    absolute_counts = Counter(absolute_labels.values())
    full_counts = Counter(full_labels.values())
    a = payload["vdc_absolute_under_absolute_labels"]
    f = payload["vdc_full_under_full_labels"]
    table = rf"""\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Reference target & Eligible & Ineligible & Uncertain & Ineligible/uncertain promotions & Recall \\
\midrule
Absolute $(\mu,q,H^C)$ & {absolute_counts['eligible']} & {absolute_counts['ineligible']} & {absolute_counts['uncertain']} & {a['resolved_ineligible_promotions']}/{a['uncertain_promotions']} & {a['resolved_eligible_recall']:.3f} \\
Full $(\mu,q,H^C,D^V)$ & {full_counts['eligible']} & {full_counts['ineligible']} & {full_counts['uncertain']} & {f['resolved_ineligible_promotions']}/{f['uncertain_promotions']} & {f['resolved_eligible_recall']:.3f} \\
\bottomrule
\end{{tabular}}
\caption{{Sensitivity to simultaneous reference intervals on the independent 1,000-pair reference split. ``Ineligible/uncertain'' reports promotions against resolved-ineligible and reference-uncertain labels. The primary point-label analysis remains unchanged.}}
\label{{tab:reference-label-sensitivity}}
\end{{table}}
"""
    TABLE_OUTPUT.write_text(table, encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
