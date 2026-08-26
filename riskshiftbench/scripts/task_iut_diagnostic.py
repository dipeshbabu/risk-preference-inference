"""Post-lock task-level intersection--union diagnostic for the 24-update study."""

from __future__ import annotations

import json
from pathlib import Path

from riskshiftbench.experiments.real_agent_validation import canonical_sha256
from riskshiftbench.experiments.vdc_efficient import task_iut_family_decisions


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "riskshiftbench/artifacts/prospective_full_vs_absolute_v1"
RESULT = ARTIFACT / "results_v3.json"
SENSITIVITY = ARTIFACT / "reference_label_sensitivity.json"
OUTPUT = ARTIFACT / "task_iut_diagnostic.json"
TABLE = ROOT / "paper/tables/task_iut_diagnostic.tex"


def summarize(
    *,
    routes: dict[str, str],
    reference_eligible: dict[str, bool],
    interval_labels: dict[str, str],
    final_effects: dict[str, float],
) -> dict:
    promoted = sorted(task for task, reason in routes.items() if reason == "promote")
    eligible = [task for task, value in reference_eligible.items() if value]
    return {
        "promotions": len(promoted),
        "promoted_updates": promoted,
        "deployment_gain": sum(final_effects[task] for task in promoted) / len(routes),
        "point_reference_ineligible_promotions": sum(
            not reference_eligible[task] for task in promoted
        ),
        "point_eligible_recall": sum(reference_eligible[task] for task in promoted)
        / len(eligible),
        "resolved_ineligible_promotions": sum(
            interval_labels[task] == "ineligible" for task in promoted
        ),
        "reference_uncertain_promotions": sum(
            interval_labels[task] == "uncertain" for task in promoted
        ),
        "unresolved": sum(reason != "promote" for reason in routes.values()),
    }


def build_payload(result: dict, sensitivity: dict) -> dict:
    interval_rows = {row["task_id"]: row for row in sensitivity["rows"]}
    final_effects = {
        task: float(values["mean_effect"])
        for task, values in result["point_metrics"]["final"].items()
    }
    methods = {}
    for short, result_key, eligibility_key, label_key in (
        ("absolute", "vdc_absolute", "reference_absolute_eligibility", "absolute_interval_label"),
        ("full", "vdc_full", "reference_full_eligibility", "full_interval_label"),
    ):
        component = result["component_pvalues"][result_key]
        decisions = task_iut_family_decisions(component, family_alpha=0.05)
        routes = {task: decision.reason for task, decision in decisions.items()}
        labels = {task: interval_rows[task][label_key] for task in routes}
        summary = summarize(
            routes=routes,
            reference_eligible=result[eligibility_key],
            interval_labels=labels,
            final_effects=final_effects,
        )
        locked_routes = result["methods"][result_key]["route"]
        methods[short] = {
            **summary,
            "task_pvalues": {
                task: {
                    "raw": decision.task_pvalue,
                    "holm_adjusted": decision.adjusted_task_pvalue,
                }
                for task, decision in decisions.items()
            },
            "route": routes,
            "changes_from_locked_flat_holm": sorted(
                task for task in routes if routes[task] != locked_routes[task]
            ),
            "deployment_changes_from_locked_flat_holm": sorted(
                task
                for task in routes
                if (routes[task] == "promote") != (locked_routes[task] == "promote")
            ),
        }
    disagreement = "qwen2.5-1.5b-instruct::workflow::compressed-planner"
    payload = {
        "protocol_id": result["protocol_id"],
        "evidential_status": "post-lock diagnostic using stored component p-values; no new agent inference",
        "family_alpha": 0.05,
        "construction": "maximum component p-value per task, followed by Holm across 24 tasks",
        "methods": methods,
        "disagreement_task": disagreement,
        "disagreement_full_task_pvalue": methods["full"]["task_pvalues"][disagreement],
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    return payload


def render_table(payload: dict, result: dict) -> None:
    rows = []
    for label, key in (
        ("Locked flat Holm Absolute", "vdc_absolute"),
        ("Locked flat Holm Full", "vdc_full"),
    ):
        method = result["methods"][key]
        rows.append(
            f"{label} & {method['promotions']} & {method['deployment_gain']:.3f} & "
            f"{method['ineligible_promotions']} & {method['eligible_recall']:.3f} & -- \\\\"
        )
    for label, key in (("Task-IUT Absolute", "absolute"), ("Task-IUT Full", "full")):
        method = payload["methods"][key]
        rows.append(
            f"{label} & {method['promotions']} & {method['deployment_gain']:.3f} & "
            f"{method['point_reference_ineligible_promotions']} & "
            f"{method['point_eligible_recall']:.3f} & "
            f"{len(method['deployment_changes_from_locked_flat_holm'])} \\\\"
        )
    disputed = payload["disagreement_full_task_pvalue"]
    TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                "Procedure & Promoted & Gain & Point-ineligible & Recall & Deploy changes \\\\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{Post-lock task-level intersection--union diagnostic. Each task uses the maximum component $p$ value, followed by Holm correction across 24 tasks. Task-IUT Full promotes the previously disputed 1.5B workflow compressed-planner route: its task value is {disputed['raw']:.4f} and its Holm-adjusted value is {disputed['holm_adjusted']:.4f}. The diagnostic uses stored component evidence and does not replace the locked primary analysis.}}",
                r"\label{tab:task-iut-diagnostic}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    payload = build_payload(result, sensitivity)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_table(payload, result)
    print(json.dumps({
        "output": str(OUTPUT),
        "table": str(TABLE),
        "full": payload["methods"]["full"],
    }, indent=2))


if __name__ == "__main__":
    main()
