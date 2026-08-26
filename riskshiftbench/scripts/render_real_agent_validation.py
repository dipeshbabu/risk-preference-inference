"""Render the locked multi-step real-agent tables and utility--drift map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_validation_v1"
    / "results_amended_v2.json"
)
POWER = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_validation_v1"
    / "postlock_power_diagnostic.json"
)
REPLICATION = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "workflow_drift_only_replication_v1"
    / "results_amended_v2.json"
)
COMPONENTS = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_validation_v1"
    / "vdc_component_diagnostics.json"
)
HIGH_2500 = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_high_budget_v1"
    / "results.json"
)
HIGH_5000 = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_high_budget_extension_v1"
    / "results.json"
)
EFFICIENCY = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "real_agent_high_budget_extension_v1"
    / "vdc_efficiency_study.json"
)
EFFICIENT_CONFIRMATION = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "vdc_efficient_confirmation_v1"
    / "results.json"
)
DRIFT_ONLY_COHORT = (
    ROOT
    / "riskshiftbench"
    / "artifacts"
    / "natural_drift_only_cohort_v1"
    / "results.json"
)
UPDATE_TABLE = ROOT / "paper" / "tables" / "real_agent_updates.tex"
METHOD_TABLE = ROOT / "paper" / "tables" / "real_agent_methods.tex"
REPLICATION_TABLE = ROOT / "paper" / "tables" / "workflow_drift_replication.tex"
COMPONENT_TABLE = ROOT / "paper" / "tables" / "vdc_component_resolution.tex"
ALTERNATIVE_TABLE = ROOT / "paper" / "tables" / "alternative_drift_metrics.tex"
HIGH_METHOD_TABLE = ROOT / "paper" / "tables" / "real_agent_high_budget_methods.tex"
EXECUTED_BUDGET_TABLE = ROOT / "paper" / "tables" / "vdc_executed_budgets.tex"
EFFICIENCY_TABLE = ROOT / "paper" / "tables" / "vdc_efficiency_methods.tex"
EFFICIENCY_BUDGET_TABLE = ROOT / "paper" / "tables" / "vdc_efficiency_budgets.tex"
CONFIRMATION_TABLE = ROOT / "paper" / "tables" / "vdc_efficient_confirmation.tex"
DRIFT_ONLY_TABLE = ROOT / "paper" / "tables" / "natural_drift_only_cohort.tex"
FIGURE = ROOT / "paper" / "figures" / "real_agent_drift.pdf"
POWER_FIGURE = ROOT / "paper" / "figures" / "real_agent_power_curve.pdf"
MECHANISM_FIGURE = ROOT / "paper" / "figures" / "drift_mechanisms.pdf"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_results() -> dict:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(body):
        raise RuntimeError("real-agent result digest mismatch")
    return payload


def _load_hashed(path: Path, digest_key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != digest_key}
    if payload[digest_key] != canonical_sha256(body):
        raise RuntimeError(f"digest mismatch: {path}")
    return payload


def _delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "--"
    return f"{value - baseline:.3f}"


def render_update_table(payload: dict) -> None:
    labels = {
        "coding-planner": "Coding & Planner scaffold",
        "coding-memory": "Coding & Patch memory",
        "workflow-router": "Workflow & Tool router",
        "workflow-memory": "Workflow & Record memory",
        "research-reflection": "Research & Reflection",
        "research-retrieval": "Research & Retrieval",
    }
    rows = []
    for update in payload["updates"]:
        metrics = update["frozen_reference"]
        family, update_name = labels[update["pair_id"]].split(" & ")
        rows.append(
            f"{family} & {update_name} & {metrics['fallback_success']:.3f} & "
            f"{metrics['candidate_success']:.3f} & {metrics['mean_effect']:.3f} & "
            f"{metrics['verifier_drift_max']:.3f} & "
            f"{_delta(metrics['candidate_auroc'], metrics['fallback_auroc'])} & "
            f"{metrics['candidate_ece'] - metrics['fallback_ece']:.3f} & "
            f"{'yes' if update['frozen_eligible'] else 'no'} \\\\"
        )
    UPDATE_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.8pt}",
                r"\begin{tabular}{llrrrrrrc}",
                r"\toprule",
                r"Agent & Update & Fall. & Cand. & $\Delta S$ & $D^V$ & $\Delta$AUC & $\Delta$ECE & Elig. \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Disjoint reference results for six multi-step agent updates. Fall. and Cand. are episode success rates. Positive $D^V$ and $\Delta$ECE, and negative $\Delta$AUC, indicate verifier degradation.}",
                r"\label{tab:real-agent-updates}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_method_table(payload: dict) -> None:
    labels = {
        "always_fallback": "Always fallback",
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "always_recalibrate": "Always recalibrate",
        "vdc": "VDC",
        "oracle_drift": "Oracle drift",
    }
    rows = []
    for method, label in labels.items():
        result = payload["methods"][method]
        rows.append(
            f"{label} & {result['mean_gain']:.3f} & {result['mean_promotions']:.2f} & "
            f"{result['mean_ineligible_promotions']:.2f} & "
            f"{result['probability_any_ineligible_promotion']:.3f} & "
            f"{result['mean_actions']['unresolved']:.2f} & "
            f"{result['mean_pilot_pairs']:.0f} \\\\"
        )
    METHOD_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.1pt}",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Method & Gain & Promote & Ineligible & Any inelig. & Unres. & Pilot pairs \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Frozen-route stability over 50 bootstrap resamples of the fixed pilot pool. Frequencies are conditional diagnostics, not confidence estimates for independent future streams. Ineligibility uses the reference split and gain uses the disjoint final split.}",
                r"\label{tab:real-agent-methods}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_replication_table(payload: dict, replication: dict) -> None:
    original = next(
        row for row in payload["updates"] if row["pair_id"] == "workflow-memory"
    )["frozen_reference"]
    fresh = replication["metrics"]
    rows = [
        (
            "Original reference (250) & "
            f"{original['mean_effect']:.3f} & {original['material_loss_probability']:.3f} & "
            f"{original['candidate_high_confidence_failure']:.3f} & "
            f"{original['verifier_drift_max']:.3f} & -- \\\\"
        ),
        (
            "Fresh targeted replication (2,000) & "
            f"{fresh['mean_effect']:.3f} & {fresh['material_loss_probability']:.3f} & "
            f"{fresh['candidate_high_confidence_failure']:.3f} & "
            f"{fresh['verifier_drift_max']:.3f} & "
            f"{'yes' if replication['drift_only_point_label'] else 'no'} \\\\"
        ),
    ]
    REPLICATION_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{lrrrrc}",
                r"\toprule",
                r"Cohort & $\mu$ & $q$ & $H^C$ & $D^V$ & Drift-only \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Workflow-memory replication. The fresh cohort confirms drift but misses the locked drift-only condition because $H^C=0.243>\kappa=0.22$. Cohort selection was informed by the original result; fresh outcomes were locked separately.}",
                r"\label{tab:workflow-drift-replication}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_component_tables(components: dict) -> None:
    labels = {
        "coding-planner": "Coding planner",
        "coding-memory": "Coding memory",
        "research-retrieval": "Research retrieval",
    }
    rows = []
    for record in components["component_resolution"]:
        if not record["reference_eligible"]:
            continue
        bounds = record["sufficient_sample_bounds"]
        rows.append(
            f"{labels[record['pair_id']]} & {', '.join(record['blocking_components'])} & "
            f"{max(bounds['full'].values()):,} & "
            f"{max(bounds['taskwise_no_family'].values()):,} & "
            f"{max(bounds['single_threshold'].values()):,} & "
            f"{max(bounds['full_without_drift'].values()):,} \\\\"
        )
    COMPONENT_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.4pt}",
                r"\begin{tabular}{llrrrr}",
                r"\toprule",
                r"Eligible update & Blocking at $n=100$ & Full & No family & One $\tau$ & No drift \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Post-lock component diagnostic. Numeric columns are corrected all-Hoeffding interval-resolution bounds, including sampling deviation and the reported endpoint radius. Each column takes the maximum over included component margins. `No family' sets family size to one; `One $\tau$' fixes the middle threshold; `No drift' omits the relative-drift condition.}",
                r"\label{tab:vdc-component-resolution}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_high_budget_tables(
    primary: dict, high_2500: dict, high_5000: dict, components: dict
) -> None:
    labels = {
        "always_fallback": "Always fallback",
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "vdc": "VDC",
        "oracle": "Oracle",
    }
    rows = []
    for method, label in labels.items():
        result = high_5000["methods"][method]
        rows.append(
            f"{label} & {result['deployment_gain']:.3f} & {result['promotions']} & "
            f"{result['ineligible_promotions']} & {result['pilot_pairs']:,} \\\\"
        )
    HIGH_METHOD_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{5pt}",
                r"\begin{tabular}{lrrrr}",
                r"\toprule",
                r"Method & Gain & Promotions & Ineligible & Pilot pairs \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Fresh cumulative 5,000-pair/update route over the full six-update family. Reference eligibility and final gain use the previously frozen disjoint streams. The extension was selected after the 2,500-pair VDC route remained unresolved.}",
                r"\label{tab:real-agent-high-budget}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_efficiency_tables(efficiency: dict, high_5000: dict) -> None:
    uniform_2000 = next(
        row for row in efficiency["budgets"] if row["pairs_per_update"] == 2_000
    )["methods"]
    adaptive_12000 = next(
        row for row in efficiency["vdc_a_split"] if row["total_pilot_pairs"] == 12_000
    )["summary"]
    rows_data = [
        ("Always fallback", uniform_2000["always_fallback"]),
        ("Outcome only", uniform_2000["outcome_only"]),
        ("Frozen verifier", uniform_2000["frozen_verifier"]),
        ("VDC-Conservative", uniform_2000["vdc_conservative"]),
        ("VDC-Efficient", uniform_2000["vdc_efficient"]),
        ("VDC-A(split)", adaptive_12000),
        (
            "Oracle",
            {
                "deployment_gain": high_5000["methods"]["oracle"]["deployment_gain"],
                "ineligible_promotions": 0,
                "eligible_recall": 1.0,
                "pilot_pairs": 0,
                "recalibrations": 0,
            },
        ),
    ]
    rows = [
        f"{label} & {row['ineligible_promotions']} & {row['deployment_gain']:.3f} & "
        f"{row['eligible_recall']:.3f} & {row['pilot_pairs']:,} & "
        f"{row['recalibrations']} \\\\"
        for label, row in rows_data
    ]
    EFFICIENCY_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Method & Ineligible & Gain & Eligible recall & Pilot pairs & V-unres. \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Post-lock method diagnostic at 12,000 total pilot pairs. Uniform methods use 2,000 pairs/update; VDC-A(split) uses 1,800 guidance pairs and 10,200 independently scored verification pairs. Empirical safety is descriptive; fixed-look familywise validity follows from component validity and Holm correction.}",
                r"\label{tab:vdc-efficiency}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_prospective_tables(
    confirmation: dict, cohort: dict, efficiency: dict
) -> None:
    confirmation_labels = {
        "always_fallback": "Always fallback",
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "vdc_efficient": "VDC-Efficient",
        "oracle": "Oracle",
    }
    confirmation_rows = []
    for method, label in confirmation_labels.items():
        row = confirmation["methods"][method]
        confirmation_rows.append(
            f"{label} & {row['deployment_gain']:.3f} & {row['promotions']} & "
            f"{row['ineligible_promotions']} & {row['eligible_recall']:.3f} & "
            f"{row['unresolved']} & {row['pilot_pairs']:,} \\\\"
        )
    CONFIRMATION_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                r"Method & Gain & Promote & Ineligible & Recall & Unresolved & Pilot pairs \\",
                r"\midrule",
                *confirmation_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Prospectively locked VDC-Efficient confirmation on one new 2,000-pair/update cohort. The frozen verifier safely promotes coding-memory; Holm-corrected VDC-Efficient leaves every route unresolved.}",
                r"\label{tab:vdc-efficient-confirmation}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    labels = {
        "coding-memory": "Coding memory",
        "cohort-coding-review": "Coding review",
        "cohort-coding-risk": "Coding risk-aware",
        "workflow-memory": "Workflow memory",
        "cohort-workflow-confirm": "Workflow confirm",
        "cohort-workflow-direct": "Workflow direct",
        "cohort-workflow-audit": "Workflow audit",
        "research-retrieval": "Research retrieval",
        "cohort-research-doublecheck": "Research double-check",
        "cohort-research-skeptical": "Research skeptical",
    }
    cohort_rows = []
    for row in cohort["updates"]:
        cohort_rows.append(
            f"{labels[row['pair_id']]} & {row['mean_effect']:.3f} & "
            f"{row['material_loss_probability']:.3f} & "
            f"{row['candidate_high_confidence_failure']:.3f} & "
            f"{row['verifier_drift_max']:.3f} & "
            f"{'yes' if row['passes_first_three'] else 'no'} & "
            f"{'yes' if row['drift_only_failure'] else 'no'} \\\\"
        )
    DRIFT_ONLY_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3.3pt}",
                r"\begin{tabular}{lrrrrcc}",
                r"\toprule",
                r"Update & $\mu$ & $q$ & $H^C$ & $D^V$ & First three & Drift-only \\",
                r"\midrule",
                *cohort_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Locally locked ten-update natural cohort. Three updates pass utility, downside, and absolute confidence risk; none fails only the relative drift condition. All committed updates remain in the table.}",
                r"\label{tab:natural-drift-only-cohort}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    budget_rows = []
    for record in efficiency["budgets"]:
        budget = record["pairs_per_update"]
        for method, label in (
            ("vdc_conservative", "Conservative"),
            ("vdc_efficient", "Efficient"),
        ):
            row = record["methods"][method]
            budget_rows.append(
                f"{budget:,} & {label} & {row['deployment_gain']:.3f} & "
                f"{row['eligible_recall']:.3f} & {row['ineligible_promotions']} & "
                f"{row['recalibrations']} \\\\"
            )
    EFFICIENCY_BUDGET_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{rlrrrr}",
                r"\toprule",
                r"Pairs/update & VDC & Gain & Eligible recall & Ineligible & V-unres. \\",
                r"\midrule",
                *budget_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Post-lock uniform-budget comparison on prefixes of the locked cumulative pilot stream. The historical route is confidence-limited at 1,500 pairs; under the revised semantics this means verifier evidence unresolved. The method promotes at 2,000 pairs and makes no observed ineligible promotion.}",
                r"\label{tab:vdc-efficiency-budgets}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_mechanism_figure(
    components: dict, primary: dict, high_2500: dict, high_5000: dict
) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "coding-planner": "C-P",
        "coding-memory": "C-M",
        "workflow-router": "W-R",
        "workflow-memory": "W-M",
        "research-reflection": "R-F",
        "research-retrieval": "R-G",
    }
    colors = {"coding": "#0072B2", "workflow": "#D55E00", "research": "#009E73"}
    fig, axis = plt.subplots(figsize=(5.6, 3.4))
    for row in components["alternative_drift"]:
        family = row["pair_id"].split("-", 1)[0]
        x, y = row["coverage_change"], row["selective_risk_change"]
        axis.scatter(x, y, s=58, color=colors[family], edgecolor="black", linewidth=0.4)
        axis.annotate(labels[row["pair_id"]], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Candidate minus fallback confidence coverage")
    axis.set_ylabel("Candidate minus fallback selective risk")
    axis.grid(alpha=0.20, linewidth=0.5)
    axis.margins(0.15)
    fig.tight_layout()
    fig.savefig(MECHANISM_FIGURE, bbox_inches="tight")
    plt.close(fig)
    primary_vdc = primary["methods"]["vdc"]
    budget_rows = [
        f"100 & Conditional bootstrap & {primary_vdc['mean_gain']:.3f} & "
        f"{primary_vdc['mean_promotions']:.2f} & {primary_vdc['mean_ineligible_promotions']:.2f} & "
        f"{primary_vdc['mean_actions']['unresolved']:.2f} \\\\"
    ]
    for budget, payload in ((2_500, high_2500), (5_000, high_5000)):
        result = payload["methods"]["vdc"]
        unresolved = sum(value == "unresolved" for value in result["route"].values())
        budget_rows.append(
            f"{budget:,} & Fresh cumulative route & {result['deployment_gain']:.3f} & "
            f"{result['promotions']} & {result['ineligible_promotions']} & {unresolved} \\\\"
        )
    EXECUTED_BUDGET_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{rlrrrr}",
                r"\toprule",
                r"Pairs/update & Evidence & Gain & Promote & Ineligible & Unresolved \\",
                r"\midrule",
                *budget_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Observed VDC budget behavior. The 100-pair row averages conditional bootstrap routes; 2,500 and 5,000 are successively locked fresh-data routes.}",
                r"\label{tab:vdc-executed-budgets}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    alt_labels = {
        "coding-planner": "Coding planner",
        "coding-memory": "Coding memory",
        "workflow-router": "Workflow router",
        "workflow-memory": "Workflow memory",
        "research-reflection": "Research reflection",
        "research-retrieval": "Research retrieval",
    }
    alt_rows = [
        f"{alt_labels[row['pair_id']]} & {row['max_threshold_drift']:.3f} & "
        f"{row['average_threshold_drift']:.3f} & {row['aurc_drift']:.3f} \\\\"
        for row in components["alternative_drift"]
    ]
    ALTERNATIVE_TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{5pt}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Update & Max threshold & Average threshold & $\Delta$AURC \\",
                r"\midrule",
                *alt_rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Post-lock construct diagnostic. $\Delta$AURC is candidate minus fallback area under the selective-risk--coverage curve; positive values are worse. The summaries answer different operational questions and need not agree.}",
                r"\label{tab:alternative-drift}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_figure(payload: dict) -> None:
    import matplotlib.pyplot as plt

    colors = {"coding": "#0072B2", "workflow": "#D55E00", "research": "#009E73"}
    markers = {
        "coding-planner": "o",
        "coding-memory": "s",
        "workflow-router": "o",
        "workflow-memory": "s",
        "research-reflection": "o",
        "research-retrieval": "s",
    }
    abbreviations = {
        "coding-planner": "C-P",
        "coding-memory": "C-M",
        "workflow-router": "W-R",
        "workflow-memory": "W-M",
        "research-reflection": "R-F",
        "research-retrieval": "R-G",
    }
    fig, axis = plt.subplots(figsize=(5.8, 3.45))
    for update in payload["updates"]:
        metrics = update["frozen_reference"]
        axis.scatter(
            metrics["mean_effect"],
            metrics["verifier_drift_max"],
            color=colors[update["family"]],
            marker=markers[update["pair_id"]],
            s=64,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        axis.annotate(
            abbreviations[update["pair_id"]],
            (metrics["mean_effect"], metrics["verifier_drift_max"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(0.02, color="black", linestyle="--", linewidth=0.85)
    axis.axhline(0.05, color="black", linestyle=":", linewidth=0.85)
    axis.set_xlabel("Candidate minus fallback episode success")
    axis.set_ylabel(r"Maximum verifier drift $D^V$")
    axis.grid(alpha=0.20, linewidth=0.6)
    axis.margins(x=0.18, y=0.20)
    fig.tight_layout()
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def render_power_figure(power: dict) -> None:
    import matplotlib.pyplot as plt

    rows = power["budget_curve"]
    x = [row["pairs_per_update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True)
    series = (
        ("mean_gain", "Deployment gain", "#0072B2"),
        ("probability_any_ineligible", "Any ineligible promotion", "#D55E00"),
        ("eligible_recall", "Eligible-update recall", "#009E73"),
        ("mean_unresolved", "Mean unresolved routes", "#CC79A7"),
    )
    for axis, (key, ylabel, color) in zip(axes.flat, series):
        axis.plot(x, [row[key] for row in rows], marker="o", color=color, linewidth=1.4, markersize=3.5)
        axis.set_xscale("log")
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.tick_params(labelsize=7)
        if key == "mean_gain":
            axis.set_ylim(bottom=0.0)
        elif key == "probability_any_ineligible":
            axis.set_ylim(0.0, 0.05)
        elif key == "eligible_recall":
            axis.set_ylim(0.0, 1.02)
        elif key == "mean_unresolved":
            axis.set_ylim(0.0, 6.2)
    axes[1, 0].set_xlabel("Paired episodes per update", fontsize=8)
    axes[1, 1].set_xlabel("Paired episodes per update", fontsize=8)
    fig.tight_layout(pad=0.8)
    fig.savefig(POWER_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = load_results()
    power = _load_hashed(POWER, "diagnostic_sha256")
    replication = _load_hashed(REPLICATION, "result_sha256")
    components = _load_hashed(COMPONENTS, "diagnostic_sha256")
    high_2500 = _load_hashed(HIGH_2500, "result_sha256")
    high_5000 = _load_hashed(HIGH_5000, "result_sha256")
    efficiency = _load_hashed(EFFICIENCY, "diagnostic_sha256")
    confirmation = _load_hashed(EFFICIENT_CONFIRMATION, "result_sha256")
    cohort = _load_hashed(DRIFT_ONLY_COHORT, "result_sha256")
    render_update_table(payload)
    render_method_table(payload)
    render_replication_table(payload, replication)
    render_component_tables(components)
    render_high_budget_tables(payload, high_2500, high_5000, components)
    render_efficiency_tables(efficiency, high_5000)
    render_prospective_tables(confirmation, cohort, efficiency)
    render_figure(payload)
    render_power_figure(power)
    render_mechanism_figure(components, payload, high_2500, high_5000)


if __name__ == "__main__":
    main()
