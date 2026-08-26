"""Render natural-agent update tables and drift figure from locked results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "riskshiftbench" / "artifacts" / "natural_agent_updates_v1" / "results.json"
UPDATE_TABLE = ROOT / "paper" / "tables" / "natural_agent_updates.tex"
METHOD_TABLE = ROOT / "paper" / "tables" / "natural_agent_methods.tex"
FIGURE = ROOT / "paper" / "figures" / "natural_agent_drift.pdf"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_results() -> dict:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload["result_sha256"] != canonical_sha256(body):
        raise RuntimeError("natural-agent result digest mismatch")
    return payload


def _value(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def render_update_table(payload: dict) -> None:
    model_labels = {
        "smollm2-360m-instruct": "SmolLM2-360M",
        "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B",
    }
    update_labels = {
        "prompt_checklist": "Prompt checklist",
        "retrieved_memory": "Retrieved memory",
        "direct_tool_policy": "Direct tool policy",
    }
    rows = []
    for update in payload["updates"]:
        frozen = update["frozen_reference"]
        delta_auroc = None
        if frozen["candidate_auroc"] is not None and frozen["fallback_auroc"] is not None:
            delta_auroc = frozen["candidate_auroc"] - frozen["fallback_auroc"]
        delta_ece = frozen["candidate_ece"] - frozen["fallback_ece"]
        rows.append(
            f"{model_labels[update['model']]} & {update_labels[update['update_type']]} & "
            f"{frozen['mean_effect']:.3f} & {frozen['verifier_drift_max']:.3f} & "
            f"{frozen['verifier_drift_average']:.3f} & {_value(delta_auroc)} & "
            f"{delta_ece:.3f} & {'yes' if update['frozen_eligible'] else 'no'} & "
            f"{'yes' if update['recalibrated_eligible'] else 'no'} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular}{llrrrrrrr}",
            r"\toprule",
            r"Model & Update & $\Delta$acc. & $D^V_{\max}$ & $D^V_{\rm avg}$ & $\Delta$AUROC & $\Delta$ECE & Frozen elig. & Recal. elig. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Six open-weight model updates on disjoint reference items. Positive drift and ECE differences are worse; negative AUROC differences are worse. Eligibility uses the frozen or candidate-recalibrated confidence mapping indicated by the final columns.}",
            r"\label{tab:natural-agent-updates}",
            r"\end{table}",
            "",
        ]
    )
    UPDATE_TABLE.write_text(text, encoding="utf-8")


def render_method_table(payload: dict) -> None:
    labels = {
        "outcome_only": "Outcome only",
        "frozen_verifier": "Frozen verifier",
        "always_recalibrate": "Always recalibrate",
        "vdc": "VDC",
        "oracle_drift": "Oracle drift",
    }
    rows = []
    for method in labels:
        result = payload["methods"][method]
        rows.append(
            f"{labels[method]} & {result['mean_gain']:.3f} & "
            f"{result['mean_promotions']:.2f} & {result['mean_ineligible_promotions']:.2f} & "
            f"{result['probability_any_ineligible_promotion']:.3f} & "
            f"{result['mean_actions'].get('recalibrate', 0.0):.2f} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.4pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & Gain & Promotions & Ineligible & Any inelig. & Recal. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Conditional route stability across bootstrap resamples in the six-update single-step diagnostic. Always-recalibrate eligibility uses candidate-specific calibration; other non-oracle methods use frozen fallback calibration.}",
            r"\label{tab:natural-agent-methods}",
            r"\end{table}",
            "",
        ]
    )
    METHOD_TABLE.write_text(text, encoding="utf-8")


def render_figure(payload: dict) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "smollm2-360m-instruct": "#0072b2",
        "qwen2.5-0.5b-instruct": "#d55e00",
    }
    markers = {
        "prompt_checklist": "o",
        "retrieved_memory": "s",
        "direct_tool_policy": "^",
    }
    fig, axis = plt.subplots(figsize=(5.8, 3.4))
    for update in payload["updates"]:
        metrics = update["frozen_reference"]
        axis.scatter(
            metrics["mean_effect"],
            metrics["verifier_drift_max"],
            color=colors[update["model"]],
            marker=markers[update["update_type"]],
            s=55,
            edgecolor="black",
            linewidth=0.4,
        )
        short_model = "S" if update["model"].startswith("smol") else "Q"
        short_update = {
            "prompt_checklist": "P",
            "retrieved_memory": "M",
            "direct_tool_policy": "T",
        }[update["update_type"]]
        axis.annotate(f"{short_model}-{short_update}", (
            metrics["mean_effect"], metrics["verifier_drift_max"]
        ), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axvline(0.02, color="black", linestyle="--", linewidth=0.8, label="utility margin")
    axis.axhline(0.05, color="black", linestyle=":", linewidth=0.8, label="drift limit")
    axis.set_xlabel("Candidate minus fallback accuracy")
    axis.set_ylabel("Maximum verifier drift $D^V$")
    axis.margins(x=0.15, y=0.18)
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    payload = load_results()
    render_update_table(payload)
    render_method_table(payload)
    render_figure(payload)


if __name__ == "__main__":
    main()
