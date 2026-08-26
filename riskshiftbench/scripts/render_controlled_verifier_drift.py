"""Render the controlled verifier-drift table and Pareto figure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "riskshiftbench" / "artifacts" / "controlled_verifier_drift_v1" / "results.json"
TABLE = ROOT / "paper" / "riskshiftbench" / "tables" / "controlled_verifier_drift.tex"
TASK_TABLE = ROOT / "paper" / "riskshiftbench" / "tables" / "controlled_verifier_drift_tasks.tex"
FIGURE = ROOT / "paper" / "riskshiftbench" / "figures" / "controlled_verifier_drift_pareto.pdf"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_results() -> dict:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    content = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(content):
        raise RuntimeError("controlled verifier-drift result digest mismatch")
    return payload


def render_table(payload: dict) -> None:
    labels = {
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "vdc": "VDC",
    }
    rows = []
    for method in ("outcome_only", "frozen_verifier", "vdc"):
        result = payload["methods"][method]
        gain_low, gain_high = result["deployment_gain_t_interval_95"]
        risk_low, risk_high = result["any_ineligible_promotion_wilson_interval_95"]
        rows.append(
            f"{labels[method]} & "
            f"{result['mean_deployment_gain']:.3f} [{gain_low:.3f}, {gain_high:.3f}] & "
            f"{result['mean_promotions']:.2f} & "
            f"{result['mean_eligible_promotions']:.2f} & "
            f"{result['mean_ineligible_promotions']:.2f} & "
            f"{result['probability_any_ineligible_promotion']:.3f} "
            f"[{risk_low:.3f}, {risk_high:.3f}] & "
            f"{result['mean_actions'].get('recalibrate', 0.0):.2f} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.7pt}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Method & Gain [95\%] & Prom. & Elig. & Inelig. & $\Pr(\mathrm{any\ inelig.})$ [95\%] & V-unres. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Controlled 12-update study over 200 pilot streams with 240 pairs per task. Population labels come from the fixed generators. Gain intervals are stream-level $t$ intervals and event intervals are Wilson intervals. V-unres. denotes historical confidence-limited routes, not demonstrated recalibration.}",
            r"\label{tab:controlled-vdc}",
            r"\end{table}",
            "",
        ]
    )
    TABLE.write_text(text, encoding="utf-8")


def render_task_table(payload: dict) -> None:
    rows = []
    for task in payload["tasks"]:
        label = "eligible" if task["eligible"] else "+".join(task["failed_conditions"])
        rows.append(
            f"{task['task']} & {task['intended_regime'].replace('_', ' ')} & "
            f"{task['mean_effect']:.3f} & "
            f"{task['candidate_high_confidence_failure_probability']:.3f} & "
            f"{task['verifier_drift']:.3f} & {label.replace('_', ' ')} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{llrrrl}",
            r"\toprule",
            r"Task & Intended regime & $\mu$ & $H^C$ & $D^V$ & Population label \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Generator-truth geometry of the controlled study. The three drift-only tasks improve mean utility and satisfy the absolute confidence-risk limit but violate only the verifier-drift condition.}",
            r"\label{tab:controlled-vdc-tasks}",
            r"\end{table}",
            "",
        ]
    )
    TASK_TABLE.write_text(text, encoding="utf-8")


def render_figure(payload: dict) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "vdc": "VDC",
    }
    colors = {
        "outcome_only": "#d55e00",
        "frozen_verifier": "#cc79a7",
        "vdc": "#0072b2",
    }
    fig, axis = plt.subplots(figsize=(5.8, 3.1))
    for method in ("outcome_only", "frozen_verifier", "vdc"):
        result = payload["methods"][method]
        x = result["probability_any_ineligible_promotion"]
        y = result["mean_deployment_gain"]
        x_low, x_high = result["any_ineligible_promotion_wilson_interval_95"]
        y_low, y_high = result["deployment_gain_t_interval_95"]
        axis.errorbar(
            x,
            y,
            xerr=[[x - x_low], [x_high - x]],
            yerr=[[y - y_low], [y_high - y]],
            fmt="o",
            color=colors[method],
            capsize=3,
            markersize=6,
        )
        x_offset = 0.018 if method != "outcome_only" else -0.20
        axis.annotate(labels[method], (x, y), xytext=(x + x_offset, y + 0.012))
    axis.set_xlabel("Probability of any reference-ineligible promotion (lower is safer)")
    axis.set_ylabel("Deployment gain (higher is better)")
    axis.set_xlim(-0.03, 1.05)
    axis.set_ylim(0.08, 0.39)
    axis.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = load_results()
    render_table(payload)
    render_task_table(payload)
    render_figure(payload)


if __name__ == "__main__":
    main()
