"""Controlled counterexample: operational VDC does not prevent confidence collapse."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from statistics import fmean

from riskshiftbench.experiments.real_agent_validation import canonical_sha256
from riskshiftbench.experiments.verifier_drift_control import VDCPlan, decide_vdc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "riskshiftbench/artifacts/controlled_verifier_drift_v1/confidence_collapse_diagnostic.json"
TABLE = ROOT / "paper/tables/confidence_collapse_diagnostic.tex"

FALLBACK_SUCCESS = 0.55
CANDIDATE_SUCCESS = 0.75
THRESHOLDS = (0.7, 0.8, 0.9)
PAIRS = 2_000
STREAMS = 200


def simulate(seed: int) -> tuple[list[float], dict[float, list[bool]], dict[float, list[bool]], float, float]:
    rng = random.Random(seed)
    effects = []
    candidate_hcf = {threshold: [] for threshold in THRESHOLDS}
    fallback_hcf = {threshold: [] for threshold in THRESHOLDS}
    candidate_coverage = []
    fallback_coverage = []
    for _ in range(PAIRS):
        shared = rng.random()
        fallback_success = shared < FALLBACK_SUCCESS
        candidate_success = shared < CANDIDATE_SUCCESS
        effects.append(float(candidate_success) - float(fallback_success))

        # The fallback score perfectly separates success (0.95) from failure
        # (0.20).  The candidate emits the constant low score 0.20.
        fallback_confidence = 0.95 if fallback_success else 0.20
        candidate_confidence = 0.20
        fallback_coverage.append(fallback_confidence >= THRESHOLDS[0])
        candidate_coverage.append(candidate_confidence >= THRESHOLDS[0])
        for threshold in THRESHOLDS:
            candidate_hcf[threshold].append(
                not candidate_success and candidate_confidence >= threshold
            )
            fallback_hcf[threshold].append(
                not fallback_success and fallback_confidence >= threshold
            )
    return (
        effects,
        candidate_hcf,
        fallback_hcf,
        fmean(candidate_coverage),
        fmean(fallback_coverage),
    )


def main() -> None:
    plan = VDCPlan(
        confidence_thresholds=THRESHOLDS,
        family_tasks=1,
        declared_looks=1,
        family_alpha=0.05,
        mean_margin=0.025,
        material_loss_threshold=0.25,
        maximum_material_loss_probability=0.10,
        maximum_candidate_high_confidence_failure_probability=0.10,
        maximum_verifier_drift=0.05,
    )
    actions = Counter()
    empirical = []
    for stream in range(STREAMS):
        effects, candidate_hcf, fallback_hcf, candidate_cov, fallback_cov = simulate(
            91_000 + stream
        )
        decision = decide_vdc(
            effects,
            candidate_hcf,
            fallback_hcf,
            plan,
            at_task_cap=True,
        )
        actions[decision.reason] += 1
        empirical.append(
            {
                "mean_effect": fmean(effects),
                "candidate_coverage": candidate_cov,
                "fallback_coverage": fallback_cov,
                "candidate_high_confidence_failure": max(
                    fmean(values) for values in candidate_hcf.values()
                ),
                "operational_drift": max(
                    fmean(candidate_hcf[t]) - fmean(fallback_hcf[t])
                    for t in THRESHOLDS
                ),
            }
        )
    payload = {
        "evidential_status": "post-review controlled counterexample; no agent inference",
        "pairs_per_stream": PAIRS,
        "streams": STREAMS,
        "population": {
            "mean_effect": CANDIDATE_SUCCESS - FALLBACK_SUCCESS,
            "material_downside_probability": 0.0,
            "candidate_high_confidence_failure": 0.0,
            "operational_drift": 0.0,
            "fallback_coverage": FALLBACK_SUCCESS,
            "candidate_coverage": 0.0,
            "fallback_type2_auroc": 1.0,
            "candidate_type2_auroc": 0.5,
            "operationally_eligible": True,
            "coverage_guardrail_passes": False,
        },
        "mean_empirical": {
            key: fmean(row[key] for row in empirical) for key in empirical[0]
        },
        "vdc_route_frequency": {
            key: value / STREAMS for key, value in sorted(actions.items())
        },
        "interpretation": "VDC certifies bounded high-confidence failure mass, not retained confidence coverage or discrimination.",
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{lrrrrrr}",
                r"\toprule",
                "Case & $\\mu$ & $H^C$ & $D^{\\mathrm{op}}$ & Coverage $F\\to C$ & AUROC $F\\to C$ & VDC deploy \\\\ ",
                r"\midrule",
                rf"Confidence collapse & {payload['population']['mean_effect']:.2f} & 0.00 & 0.00 & {FALLBACK_SUCCESS:.2f}$\to$0.00 & 1.00$\to$0.50 & {payload['vdc_route_frequency'].get('promote', 0.0):.2f} \\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{Post-review controlled counterexample. The candidate improves utility but moves every confidence score below the frozen grid. Operational high-confidence failure mass remains zero, so VDC deploys; confidence coverage and discrimination collapse. A coverage guardrail would reject this case. This diagnostic limits the interpretation of VDC and is not agent evidence.}",
                r"\label{tab:confidence-collapse}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
