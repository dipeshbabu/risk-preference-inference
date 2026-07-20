"""Build, register, and validate the prospective RiskShiftBench v2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from experiments.frontier_v2_baseline_design import (
    COMPETITIVE_BASELINES,
    baseline_design_summary,
)
from experiments.frontier_v2_external_design import (
    all_tasks,
    canonical_sha256,
    design_summary,
    outcome_implementation_sha256,
    task_manifest_sha256,
)
from experiments.frontier_v2_readiness import registration_readiness
from experiments.frontier_v2_router_lock import audit_router_lock
from experiments.frontier_v2_statistical_readiness import (
    BENTKUS_NULL_FILE,
    CLOSE_COMPARATOR_NULL_FILE,
    METHOD_COMPARISON_FILE,
    PREDICTABLE_NULL_FILE,
    PRIMARY_NULL_FILE,
)
from experiments.frontier_v2_resolution_bound_check import RESOLUTION_BOUND_FILE
from experiments.frontier_v2_proof_certificate import PROOF_CERTIFICATE_FILE


LOCKED_PROTOCOL_FILES = (
    "experiments/frontier_v2_router_lock.py",
    "experiments/frontier_v2_confirmation_runtime.py",
    "experiments/frontier_v2_competitive_evaluation.py",
    "experiments/frontier_v2_proof_certificate.py",
    "experiments/frontier_v2_protocol_lock.py",
    "experiments/frontier_v2_readiness.py",
)

STATISTICAL_ARTIFACT_FILES = (
    PRIMARY_NULL_FILE,
    PREDICTABLE_NULL_FILE,
    BENTKUS_NULL_FILE,
    CLOSE_COMPARATOR_NULL_FILE,
    METHOD_COMPARISON_FILE,
    RESOLUTION_BOUND_FILE,
    PROOF_CERTIFICATE_FILE,
)


def canonical_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def byte_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_manifest(repository_root: Path = Path(".")) -> list[dict[str, str]]:
    records = []
    for relative in LOCKED_PROTOCOL_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise RuntimeError(f"locked v2 protocol source is missing: {path}")
        records.append(
            {"path": relative, "canonical_sha256": canonical_file_sha256(path)}
        )
    return records


def _artifact_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required v2 artifact is missing: {path}")
    return {"path": path.as_posix(), "byte_sha256": byte_sha256(path)}


def _baseline_manifest_records(baseline_root: Path) -> list[dict[str, str]]:
    return [
        _artifact_record(
            baseline_root / baseline.domain / baseline.name / "manifest.json"
        )
        for baseline in COMPETITIVE_BASELINES
    ]


def build_locked_design(
    *,
    environment_source_root: Path,
    baseline_source_root: Path,
    rehearsal_root: Path,
    baseline_root: Path,
    statistical_root: Path,
    router_lock_path: Path,
    v1_development_root: Path,
    v1_router_root: Path,
    confirmation_output_root: Path,
) -> dict:
    readiness = registration_readiness(
        environment_source_root=environment_source_root,
        baseline_source_root=baseline_source_root,
        rehearsal_root=rehearsal_root,
        baseline_root=baseline_root,
        statistical_root=statistical_root,
        router_lock_path=router_lock_path,
        v1_development_root=v1_development_root,
        v1_router_root=v1_router_root,
    )
    if readiness.get("ready_for_registration") is not True:
        raise RuntimeError(
            "v2 protocol cannot be locked before every registration-readiness gate passes"
        )
    router_audit = audit_router_lock(
        router_lock_path,
        development_root=(
            rehearsal_root / "development_portable_20ep_episode_lifetime"
        ),
        calibration_root=(
            rehearsal_root / "calibration_portable_20ep_episode_lifetime"
        ),
    )
    router_lock = json.loads(router_lock_path.read_text(encoding="utf-8"))
    statistical_artifacts = [
        _artifact_record(statistical_root / filename)
        for filename in STATISTICAL_ARTIFACT_FILES
    ]
    return {
        "protocol_id": "riskshiftbench-frontier-v2-confirmation-v1",
        "purpose": (
            "Prospective nine-domain confirmation of an anytime-valid, "
            "familywise policy router under a fixed pilot episode budget."
        ),
        "registration_readiness": readiness,
        "external_design": design_summary(),
        "competitive_baseline_design": baseline_design_summary(),
        "artifact_roots": {
            "environment_source_root": environment_source_root.as_posix(),
            "baseline_source_root": baseline_source_root.as_posix(),
            "rehearsal_root": rehearsal_root.as_posix(),
            "baseline_root": baseline_root.as_posix(),
            "statistical_root": statistical_root.as_posix(),
            "v1_development_root": v1_development_root.as_posix(),
            "v1_router_root": v1_router_root.as_posix(),
        },
        "router_lock": {
            **_artifact_record(router_lock_path),
            **router_audit,
            "content": router_lock,
        },
        "statistical_artifacts": statistical_artifacts,
        "baseline_manifests": _baseline_manifest_records(baseline_root),
        "confirmation": {
            "task_count": len(all_tasks("confirmation")),
            "task_manifest_sha256": task_manifest_sha256(
                all_tasks("confirmation")
            ),
            "output_root": confirmation_output_root.as_posix(),
            "pilot": {
                **router_lock["anytime_plan"],
                "paired_observation_budget": router_lock["cost_accounting"][
                    "paired_observation_budget"
                ],
                "policy_episode_budget": router_lock["cost_accounting"][
                    "policy_episode_budget"
                ],
                "stream": "pilot",
                "resume_rule": (
                    "Authenticate and replay the complete hash-chained pilot log; "
                    "the next task must equal the registered adaptive scheduler."
                ),
            },
            "final": {
                "stream": "final",
                "episodes_per_task_policy": 100,
                "policies": (
                    "Evaluate the frozen candidate and fallback on every task; "
                    "deployed route is selected only from frozen pilot decisions."
                ),
                "pilot_seed_disjointness": True,
                "common_random_numbers": True,
                "execution_order": "task name, then candidate before fallback",
            },
            "competitive_references": {
                "evaluation_episodes_per_task": 100,
                "learned_evaluation_unit": (
                    "100 episodes per task for each of five calibration-selected "
                    "training-seed checkpoints; aggregate with equal checkpoint-seed weight"
                ),
                "checkpoint_rule": (
                    "Use only the calibration-selected checkpoint recorded in each "
                    "registered baseline manifest."
                ),
            },
        },
        "analysis": {
            "primary_estimand": (
                "Equal-domain mean candidate-router minus fallback normalized score "
                "effect across the nine fixed external domains."
            ),
            "primary_hypothesis": "one-sided equal-domain mean effect greater than zero",
            "confirmatory_alpha": 0.05,
            "confirmatory_test": (
                "One-sided weighted Hoeffding test of the fixed-family "
                "equal-domain route effect, using all independently seeded final "
                "paired episode differences in [-1, 1], conditional on the "
                "pilot-frozen route. Reject only when the one-sided p-value is "
                "at most 0.05 and the observed effect is positive."
            ),
            "confirmatory_test_assumption": (
                "Final paired episode units are independent conditional on the "
                "pilot-frozen route; task distributions may differ."
            ),
            "task_effect": "mean paired candidate-route score minus fallback score",
            "domain_weighting": "equal domain; equal task within domain",
            "task_stratified_bootstrap_replicates": 10_000,
            "domain_resampling_bootstrap_replicates": 10_000,
            "task_level_sign_flip_replicates": 100_000,
            "task_level_sign_flip_role": (
                "secondary symmetry-based sensitivity analysis, not the "
                "confirmatory test"
            ),
            "random_seed": 62_000_001,
            "secondary_outputs": [
                "total paired pilot observations",
                "accepted, rejected, budget-exhausted, and harmful accepted routes",
                "per-domain effects",
                "candidate-everywhere and fallback-only",
                "all registered familywise and allocation comparisons",
                "all registered competitive learned and nonlearned references",
                "route-held-fixed score sensitivity",
                "leave-one-domain-out effects",
            ],
            "performance_threshold_used_for_method_selection": False,
        },
        "source_manifest": _source_manifest(),
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "execution_guard": (
            "No confirmation task may be reset until this exact design is publicly "
            "registered and the wrapper is finalized with its immutable URL and time."
        ),
        "reporting_commitment": (
            "Report all tasks, pilot observations, route decisions, final outcomes, "
            "baselines, nulls, harmful routes, and sensitivity analyses regardless of "
            "direction; do not revise the method on these confirmation suites."
        ),
    }


def write_registration_draft(
    *,
    locked_design_path: Path,
    draft_path: Path,
    **design_arguments,
) -> dict:
    if locked_design_path.exists() or draft_path.exists():
        raise FileExistsError("refusing to overwrite a v2 design or registration draft")
    design = build_locked_design(**design_arguments)
    locked_design_path.parent.mkdir(parents=True, exist_ok=True)
    locked_design_path.write_bytes(_json_bytes(design))
    wrapper = {
        "status": "awaiting_external_registration",
        "locked_design_path": locked_design_path.as_posix(),
        "locked_design_byte_sha256": byte_sha256(locked_design_path),
        "locked_design_canonical_sha256": canonical_sha256(design),
        "registration": {
            "provider": None,
            "url": None,
            "registered_at": None,
        },
        "execution_guard": (
            "Confirmation execution refuses this draft. Register the locked design "
            "publicly, then finalize a separate wrapper."
        ),
    }
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_bytes(_json_bytes(wrapper))
    return wrapper


def finalize_registration(
    draft_path: Path,
    output_path: Path,
    *,
    provider: str,
    url: str,
    registered_at: str,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite registered wrapper: {output_path}")
    if not provider.strip():
        raise ValueError("registration provider cannot be blank")
    if not url.startswith("https://"):
        raise ValueError("registration URL must use HTTPS")
    timestamp = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("registration timestamp must include a timezone")
    wrapper = json.loads(draft_path.read_text(encoding="utf-8"))
    if wrapper.get("status") != "awaiting_external_registration":
        raise RuntimeError("only an unregistered v2 draft can be finalized")
    design_path = Path(wrapper["locked_design_path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if byte_sha256(design_path) != wrapper["locked_design_byte_sha256"]:
        raise RuntimeError("locked v2 design bytes changed after draft creation")
    if canonical_sha256(design) != wrapper["locked_design_canonical_sha256"]:
        raise RuntimeError("locked v2 design content changed after draft creation")
    wrapper["status"] = "externally_registered_locked"
    wrapper["registration"] = {
        "provider": provider,
        "url": url,
        "registered_at": registered_at,
        "registered_design_sha256": wrapper["locked_design_byte_sha256"],
    }
    wrapper["execution_guard"] = (
        "Registration metadata is present; runtime must still validate every lock."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_json_bytes(wrapper))
    return wrapper


def _validate_artifact_record(record: dict) -> None:
    path = Path(record["path"])
    if not path.is_file() or byte_sha256(path) != record["byte_sha256"]:
        raise RuntimeError(f"locked v2 artifact changed: {path}")


def validate_protocol(
    protocol_path: Path,
    *,
    require_registration: bool,
) -> tuple[dict, dict]:
    wrapper = json.loads(protocol_path.read_text(encoding="utf-8"))
    design_path = Path(wrapper["locked_design_path"])
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if byte_sha256(design_path) != wrapper["locked_design_byte_sha256"]:
        raise RuntimeError("locked v2 design bytes do not match the wrapper")
    if canonical_sha256(design) != wrapper["locked_design_canonical_sha256"]:
        raise RuntimeError("locked v2 design content does not match the wrapper")
    if require_registration:
        if wrapper.get("status") != "externally_registered_locked":
            raise RuntimeError(
                "v2 confirmation is blocked until the design is publicly registered"
            )
        registration = wrapper.get("registration", {})
        required = ("provider", "url", "registered_at", "registered_design_sha256")
        if any(not registration.get(key) for key in required):
            raise RuntimeError("v2 registration metadata is incomplete")
        if not str(registration["url"]).startswith("https://"):
            raise RuntimeError("v2 registration URL must use HTTPS")
        if registration["registered_design_sha256"] != wrapper[
            "locked_design_byte_sha256"
        ]:
            raise RuntimeError("registered v2 design hash does not match the wrapper")
    for record in design["source_manifest"]:
        path = Path(record["path"])
        if (
            not path.is_file()
            or canonical_file_sha256(path) != record["canonical_sha256"]
        ):
            raise RuntimeError(f"locked v2 protocol source changed: {path}")
    for record in design["statistical_artifacts"]:
        _validate_artifact_record(record)
    for record in design["baseline_manifests"]:
        _validate_artifact_record(record)
    _validate_artifact_record(design["router_lock"])
    roots = design["artifact_roots"]
    readiness = registration_readiness(
        environment_source_root=Path(roots["environment_source_root"]),
        baseline_source_root=Path(roots["baseline_source_root"]),
        rehearsal_root=Path(roots["rehearsal_root"]),
        baseline_root=Path(roots["baseline_root"]),
        statistical_root=Path(roots["statistical_root"]),
        router_lock_path=Path(design["router_lock"]["path"]),
        v1_development_root=Path(roots["v1_development_root"]),
        v1_router_root=Path(roots["v1_router_root"]),
    )
    if readiness != design["registration_readiness"]:
        raise RuntimeError("v2 registration-readiness evidence changed")
    if readiness.get("ready_for_registration") is not True:
        raise RuntimeError("v2 registration-readiness gate no longer passes")
    if design.get("outcome_implementation_sha256") != outcome_implementation_sha256():
        raise RuntimeError("v2 outcome implementation changed after registration")
    if design["confirmation"]["task_manifest_sha256"] != task_manifest_sha256(
        all_tasks("confirmation")
    ):
        raise RuntimeError("v2 confirmation task manifest changed after registration")
    return wrapper, design


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_sources"),
    )
    parser.add_argument(
        "--baseline-source-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baseline_sources"),
    )
    parser.add_argument(
        "--rehearsal-root",
        type=Path,
        default=Path("artifacts/frontier_v2_full_rehearsal"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("artifacts/frontier_v2_baselines"),
    )
    parser.add_argument(
        "--statistical-root",
        type=Path,
        default=Path("artifacts/frontier_v2_development"),
    )
    parser.add_argument(
        "--router-lock",
        type=Path,
        default=Path("artifacts/frontier_v2_router_lock/router_lock.json"),
    )
    parser.add_argument(
        "--v1-development-root",
        type=Path,
        default=Path("artifacts/external_development_v1"),
    )
    parser.add_argument(
        "--v1-router-root",
        type=Path,
        default=Path("artifacts/external_router_lock_v1"),
    )
    parser.add_argument(
        "--confirmation-output-root",
        type=Path,
        default=Path("artifacts/frontier_v2_confirmation"),
    )
    parser.add_argument(
        "--locked-design",
        type=Path,
        default=Path("configs/frontier_v2_confirmation_locked_design.json"),
    )
    parser.add_argument(
        "--draft",
        type=Path,
        default=Path("configs/frontier_v2_confirmation.registration-draft.json"),
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--url")
    parser.add_argument("--registered-at")
    parser.add_argument(
        "--registered-output",
        type=Path,
        default=Path("configs/frontier_v2_confirmation.registered.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize:
        if not all((args.provider, args.url, args.registered_at)):
            raise ValueError("finalization requires provider, URL, and registration time")
        wrapper = finalize_registration(
            args.draft,
            args.registered_output,
            provider=args.provider,
            url=args.url,
            registered_at=args.registered_at,
        )
    else:
        wrapper = write_registration_draft(
            locked_design_path=args.locked_design,
            draft_path=args.draft,
            environment_source_root=args.environment_source_root,
            baseline_source_root=args.baseline_source_root,
            rehearsal_root=args.rehearsal_root,
            baseline_root=args.baseline_root,
            statistical_root=args.statistical_root,
            router_lock_path=args.router_lock,
            v1_development_root=args.v1_development_root,
            v1_router_root=args.v1_router_root,
            confirmation_output_root=args.confirmation_output_root,
        )
    print(json.dumps(wrapper, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
