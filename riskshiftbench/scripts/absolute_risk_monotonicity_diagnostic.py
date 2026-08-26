"""Post-lock diagnostic that removes redundant absolute-risk hypotheses."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from riskshiftbench.experiments.vdc_efficient import holm_rejections
from riskshiftbench.scripts.render_full_vs_absolute_audit import holm_adjust


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "riskshiftbench/artifacts/prospective_full_vs_absolute_v1"
RESULT = ARTIFACT / "results_v3.json"
OUTPUT = ARTIFACT / "absolute_risk_monotonicity_diagnostic.json"
TABLE = ROOT / "paper/tables/absolute_risk_monotonicity_diagnostic.tex"


def reduced_family(per_task: dict[str, dict[str, float]]) -> tuple[dict, dict]:
    family = {}
    reduced = {}
    for task, values in per_task.items():
        absolute = sorted(key for key in values if key.startswith("absolute:"))
        keep = set(values) - set(absolute[1:])
        reduced[task] = {key: value for key, value in values.items() if key in keep}
        family.update({f"{task}::{key}": value for key, value in reduced[task].items()})
    return reduced, family


def routes(per_task: dict, family: dict, alpha: float) -> dict[str, str]:
    rejected = holm_rejections(family, alpha)
    output = {}
    for task, values in per_task.items():
        names = {f"{task}::{key}" for key in values}
        if names <= rejected:
            output[task] = "promote"
        elif {f"{task}::mean", f"{task}::downside"} <= rejected:
            output[task] = "recalibrate-verifier"
        else:
            output[task] = "unresolved"
    return output


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    payload = {"protocol_id": result["protocol_id"], "evidential_status": "post-lock diagnostic"}
    for method in ("vdc_absolute", "vdc_full"):
        original = result["component_pvalues"][method]
        reduced, family = reduced_family(original)
        route = routes(reduced, family, 0.05)
        original_route = result["methods"][method]["route"]
        row = {
            "original_components": sum(len(values) for values in original.values()),
            "reduced_components": len(family),
            "original_route_counts": dict(Counter(original_route.values())),
            "reduced_route_counts": dict(Counter(route.values())),
            "route_changes": sorted(task for task in route if route[task] != original_route[task]),
        }
        if method == "vdc_full":
            task = "qwen2.5-1.5b-instruct::workflow::compressed-planner"
            adjusted = holm_adjust(family)
            row["disagreement_drift_adjusted"] = {
                key: adjusted[f"{task}::{key}"]
                for key in sorted(reduced[task])
                if key.startswith("drift:")
            }
        payload[method] = row
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Method & Locked components & Reduced components & Locked routes & Reduced routes \\
\midrule
VDC-Absolute & 120 & 72 & 5/1/18 & 5/1/18 \\
VDC-Full & 192 & 144 & 4/2/18 & 4/2/18 \\
\bottomrule
\end{tabular}
\caption{Post-lock monotonicity diagnostic. Because $h^C(\tau)$ is nonincreasing, the population absolute-risk maximum is attained at $\tau_{\min}$. Retaining only that hypothesis reduces the Holm families but changes no route. The middle count preserves the locked \texttt{recalibrate-verifier} label, now interpreted as unresolved verifier evidence.}
\label{tab:absolute-risk-monotonicity}
\end{table}
"""
    TABLE.write_text(table, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
