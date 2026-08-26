"""Render route and component audits for the completed 24-update study."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "riskshiftbench/artifacts/prospective_full_vs_absolute_v1"
RESULT = ARTIFACT / "results_v3.json"
MANIFEST = ARTIFACT / "target_manifest_v3.json"
RAW = ARTIFACT / "target_trajectories_v3.json.gz"
CSV_OUTPUT = ARTIFACT / "full_vs_absolute_route_audit.csv"
ESTIMATE_OUTPUT = ROOT / "paper/tables/full_vs_absolute_route_estimates.tex"
EVIDENCE_OUTPUT = ROOT / "paper/tables/full_vs_absolute_disagreement_evidence.tex"


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - index)))
        adjusted[name] = running
    return adjusted


def fmt(value: float) -> str:
    value = float(value)
    if value < 0.0005:
        return "<.001"
    if value >= 0.9995:
        return "1.000"
    return f"{value:.3f}".removeprefix("0")


def triple(values: list[float]) -> str:
    return "/".join(fmt(value) for value in values)


def short_update(pair: dict) -> str:
    model = "0.5B" if "0.5b" in pair["model_id"] else "1.5B"
    return f"{model} {pair['family']} {pair['design']}"


def paired_hoeffding_intervals(raw: dict, manifest: dict) -> dict[str, dict]:
    """Compute two-sided 95% intervals on disjoint final paired effects."""

    trajectories = raw["trajectories"]
    intervals = {}
    for pair in manifest["pairs"]:
        fallback_id = f"{pair['model_id']}::{pair['family']}::fallback"
        fallback = sorted(
            (
                row
                for row in trajectories
                if row["run_id"] == fallback_id and row["split"] == "final"
            ),
            key=lambda row: row["task_id"],
        )
        candidate = sorted(
            (
                row
                for row in trajectories
                if row["run_id"] == pair["id"] and row["split"] == "final"
            ),
            key=lambda row: row["task_id"],
        )
        if [row["task_id"] for row in fallback] != [row["task_id"] for row in candidate]:
            raise RuntimeError(f"unpaired final stream: {pair['id']}")
        effects = [
            float(candidate_row["score"]) - float(fallback_row["score"])
            for fallback_row, candidate_row in zip(fallback, candidate)
        ]
        count = len(effects)
        if count == 0:
            raise RuntimeError(f"empty final stream: {pair['id']}")
        estimate = sum(effects) / count
        radius = 2.0 * math.sqrt(math.log(2.0 / 0.05) / (2.0 * count))
        intervals[pair["id"]] = {
            "final_pair_count": count,
            "final_effect": estimate,
            "final_effect_ci_lower": max(-1.0, estimate - radius),
            "final_effect_ci_upper": min(1.0, estimate + radius),
            "final_effect_ci_method": "two-sided 95% paired Hoeffding, range [-1,1]",
        }
    return intervals


def build_rows(result: dict, manifest: dict, intervals: dict[str, dict]) -> list[dict]:
    pairs = {pair["id"]: pair for pair in manifest["pairs"]}
    full_p = result["component_pvalues"]["vdc_full"]
    absolute_p = result["component_pvalues"]["vdc_absolute"]
    full_adjusted = holm_adjust(
        {
            f"{task}::{component}": value
            for task, values in full_p.items()
            for component, value in values.items()
        }
    )
    absolute_adjusted = holm_adjust(
        {
            f"{task}::{component}": value
            for task, values in absolute_p.items()
            for component, value in values.items()
        }
    )
    rows = []
    for task in sorted(pairs):
        pair = pairs[task]
        ref = result["point_metrics"]["reference"][task]
        pilot = result["point_metrics"]["pilot"][task]
        final = result["point_metrics"]["final"][task]
        shared = absolute_p[task]
        full = full_p[task]
        thresholds = sorted(
            float(key.split(":", 1)[1])
            for key in full
            if key.startswith("absolute:")
        )
        row = {
            "task_id": task,
            "update": short_update(pair),
            "model": pair["model_id"],
            "family": pair["family"],
            "design": pair["design"],
            "cohort": pair["cohort"],
            "reference_mean": ref["mean_effect"],
            "reference_downside": ref["material_loss_probability"],
            "reference_absolute_risk": ref["candidate_high_confidence_failure"],
            "reference_drift": ref["operational_verifier_drift"],
            "pilot_mean": pilot["mean_effect"],
            "pilot_downside": pilot["material_loss_probability"],
            "pilot_absolute_risk": pilot["candidate_high_confidence_failure"],
            "pilot_drift": pilot["operational_verifier_drift"],
            "reference_full_eligible": result["reference_full_eligibility"][task],
            "reference_absolute_eligible": result["reference_absolute_eligibility"][task],
            "final_gain": final["mean_effect"],
            **intervals[task],
            "absolute_route": result["methods"]["vdc_absolute"]["route"][task],
            "full_route": result["methods"]["vdc_full"]["route"][task],
            "raw_mean_p": shared["mean"],
            "absolute_holm_mean_p": absolute_adjusted[f"{task}::mean"],
            "full_holm_mean_p": full_adjusted[f"{task}::mean"],
            "raw_downside_p": shared["downside"],
            "absolute_holm_downside_p": absolute_adjusted[f"{task}::downside"],
            "full_holm_downside_p": full_adjusted[f"{task}::downside"],
            "thresholds": thresholds,
            "raw_absolute_p": [shared[f"absolute:{value}"] for value in thresholds],
            "absolute_holm_absolute_p": [
                absolute_adjusted[f"{task}::absolute:{value}"] for value in thresholds
            ],
            "full_holm_absolute_p": [
                full_adjusted[f"{task}::absolute:{value}"] for value in thresholds
            ],
            "raw_drift_p": [full[f"drift:{value}"] for value in thresholds],
            "full_holm_drift_p": [
                full_adjusted[f"{task}::drift:{value}"] for value in thresholds
            ],
        }
        if abs(row["final_gain"] - row["final_effect"]) > 1e-12:
            raise RuntimeError(f"final effect mismatch: {task}")
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, list)}
        for key in (
            "thresholds",
            "raw_absolute_p",
            "absolute_holm_absolute_p",
            "full_holm_absolute_p",
            "raw_drift_p",
            "full_holm_drift_p",
        ):
            for index, value in enumerate(row[key], start=1):
                flat[f"{key}_{index}"] = value
        flat_rows.append(flat)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def write_estimate_table(rows: list[dict]) -> None:
    disagreements = [
        row for row in rows if row["absolute_route"] != row["full_route"]
    ]
    if len(disagreements) != 1:
        raise RuntimeError("expected exactly one Absolute--Full route disagreement")
    disagreement = disagreements[0]
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcrrrrrcrrr}",
        r"\toprule",
        r"Update & Coh. & $\hat\mu_{ref}$ & $\hat q_{ref}$ & $\hat H^C_{ref}$ & $\hat D^V_{ref}$ & Ref. & $\hat D^V_{pilot}$ & Abs. & Full & Final $\Delta$ \\",
        r"\midrule",
    ]
    for row in rows:
        label = "eligible" if row["reference_full_eligible"] else "ineligible"
        lines.append(
            f"{row['update']} & {row['cohort'][0].upper()} & "
            f"{row['reference_mean']:.3f} & {row['reference_downside']:.3f} & "
            f"{row['reference_absolute_risk']:.3f} & {row['reference_drift']:.3f} & "
            f"{label} & {row['pilot_drift']:.3f} & {row['absolute_route']} & "
            f"{row['full_route']} & {row['final_gain']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{Reference estimates, pilot drift, and routes for all 24 prospectively fixed updates. Reference labels use point thresholds on the independent 1,000-pair reference split; the audit contains eight eligible and sixteen ineligible updates and no reference-uncertain class. Cohort labels are N (natural) and S (stress). For the sole route disagreement, the disjoint final paired effect is {disagreement['final_effect']:.3f} with a two-sided 95\% Hoeffding interval [{disagreement['final_effect_ci_lower']:.3f}, {disagreement['final_effect_ci_upper']:.3f}] ($n={disagreement['final_pair_count']:,}$ pairs).}}",
            r"\label{tab:full-vs-absolute-estimates}",
            r"\end{table}",
        ]
    )
    ESTIMATE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evidence_table(rows: list[dict]) -> None:
    disagreements = [
        row for row in rows if row["absolute_route"] != row["full_route"]
    ]
    if len(disagreements) != 1:
        raise RuntimeError("expected exactly one locked route disagreement")
    row = disagreements[0]
    task_raw = {
        item["task_id"]: max(
            item["raw_mean_p"],
            item["raw_downside_p"],
            *item["raw_absolute_p"],
            *item["raw_drift_p"],
        )
        for item in rows
    }
    task_adjusted = holm_adjust(task_raw)
    drift = "; ".join(
        f"{fmt(raw)}/{fmt(adjusted)}"
        for raw, adjusted in zip(row["raw_drift_p"], row["full_holm_drift_p"])
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Update & $\tilde p_\mu^{F}$ & $\tilde p_q^{F}$ & $\tilde p_H^{F}$ & $p_D/\tilde p_D^{F}$ & $p_t^{\rm IUT}/\tilde p_t$ & Locked routes \\",
        r"\midrule",
        f"{row['update']} & {fmt(row['full_holm_mean_p'])} & "
        f"{fmt(row['full_holm_downside_p'])} & "
        f"{fmt(max(row['full_holm_absolute_p']))} & {drift} & "
        f"{fmt(task_raw[row['task_id']])}/{fmt(task_adjusted[row['task_id']])} & "
        r"Abs.: deploy; Full: V-unresolved \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Route-determining evidence for the sole locked Full--Absolute disagreement. Flat Full Holm leaves two drift components unresolved. Task-IUT instead rejects the task-level ineligibility null. The complete 24-update component audit remains in the released CSV.}",
        r"\label{tab:full-vs-absolute-disagreement}",
        r"\end{table}",
    ]
    EVIDENCE_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    intervals = paired_hoeffding_intervals(raw, manifest)
    rows = build_rows(result, manifest, intervals)
    write_csv(rows)
    write_estimate_table(rows)
    write_evidence_table(rows)
    print(CSV_OUTPUT)
    print(ESTIMATE_OUTPUT)
    print(EVIDENCE_OUTPUT)


if __name__ == "__main__":
    main()
