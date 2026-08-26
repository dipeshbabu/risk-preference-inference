"""Registered seed-validation amendment for external confirmation v1.

This module changes only the expected pilot seed recorded for PointGoal tasks.
The environment adapter stores the seed actually passed to ``reset``:

    layout_seed_base + pilot_seed_base + episode

The v1 gate validator omitted ``layout_seed_base`` when checking that metadata.
All score calculations, familywise gate rules, routes, and final analyses remain
delegated to the registered v1 implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from experiments import external_confirmation_evaluation as v1
from experiments.external_familywise_verifier import (
    FamilywisePilotPlan,
    verify_familywise_promotion,
)
from experiments.external_study_design import DOMAINS, POLICY_LIBRARIES, domain_tasks


DEFAULT_AMENDMENT = Path(
    "configs/external_confirmation_seed_validation_amendment_v1_1.registration-draft.json"
)


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pilot_artifact_set(design: dict) -> list[dict]:
    root = Path(design["evaluation"]["output_root"])
    records = []
    for mode in ("proposal", "random"):
        for domain in DOMAINS:
            for batch_index in design["pilot"]["batch_indices"]:
                path = root / f"pilot_{mode}" / domain / f"batch_{batch_index:02d}.csv"
                if not path.is_file():
                    raise RuntimeError(f"amendment pilot checkpoint is missing: {path}")
                records.append(
                    {
                        "path": path.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_bytes(path),
                    }
                )
    return records


def pilot_artifact_set_sha256(design: dict) -> str:
    return _canonical_sha256(_pilot_artifact_set(design))


def validate_amendment(
    wrapper_path: Path,
    *,
    require_registration: bool,
) -> tuple[dict, dict, dict]:
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    amendment_path = Path(wrapper["amendment_path"])
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if _sha256_bytes(amendment_path) != wrapper["amendment_sha256"]:
        raise RuntimeError("seed-validation amendment bytes do not match its wrapper")
    if _canonical_sha256(amendment) != wrapper["amendment_canonical_sha256"]:
        raise RuntimeError("seed-validation amendment canonical hash changed")
    amended_source = amendment["amended_source"]
    source_path = Path(amended_source["path"])
    if _sha256_bytes(source_path) != amended_source["sha256"]:
        raise RuntimeError("amended evaluator source changed")
    base_wrapper_path = Path(amendment["base_protocol"]["wrapper_path"])
    base_wrapper, design = v1.validate_protocol(
        base_wrapper_path,
        require_registration=True,
    )
    base = amendment["base_protocol"]
    if base_wrapper["locked_design_sha256"] != base["locked_design_sha256"]:
        raise RuntimeError("amendment base design byte hash changed")
    if (
        base_wrapper["locked_design_canonical_sha256"]
        != base["locked_design_canonical_sha256"]
    ):
        raise RuntimeError("amendment base design canonical hash changed")
    observed_pilot_hash = pilot_artifact_set_sha256(design)
    if observed_pilot_hash != amendment["pilot_state"]["artifact_set_sha256"]:
        raise RuntimeError("pilot checkpoint artifact set changed after the amendment")
    if require_registration:
        if wrapper.get("status") != "externally_registered_amendment_locked":
            raise RuntimeError(
                "gate and final execution are blocked until the amendment is externally registered"
            )
        registration = wrapper.get("registration") or {}
        required = ("url", "provider", "registered_at", "registered_amendment_sha256")
        if any(not registration.get(field) for field in required):
            raise RuntimeError("amendment registration metadata is incomplete")
        if registration["registered_amendment_sha256"] != wrapper["amendment_sha256"]:
            raise RuntimeError("registered amendment hash does not match the local wrapper")
    return wrapper, amendment, design


def _recorded_seed_offset(task_name: str) -> int:
    for domain in DOMAINS:
        for task in domain_tasks(domain, "confirmation"):
            if task.name == task_name:
                return int(dict(task.parameters).get("layout_seed_base", 0))
    raise KeyError(task_name)


def _gate_mode(design: dict, mode: str) -> list[dict]:
    selected = v1._allocation_tasks(design, mode)
    pilot = design["pilot"]
    plan = FamilywisePilotPlan(
        proposal_family_size=len(selected),
        familywise_alpha=float(pilot["familywise_alpha"]),
        episodes_per_batch=int(pilot["episodes_per_batch"]),
        min_mean_advantage=float(pilot["minimum_mean_advantage"]),
    )
    if plan.required_unanimous_batches != int(pilot["required_unanimous_batches"]):
        raise RuntimeError("locked familywise batch count is internally inconsistent")
    candidates_by_task = design["router_lock"]["candidate_policy_by_task"]
    task_indices = v1._task_index()
    rows = []
    for domain in DOMAINS:
        domain_selected = sorted(
            task.name
            for task in domain_tasks(domain, "confirmation")
            if task.name in selected
        )
        if not domain_selected:
            continue
        advantages = {task: [] for task in domain_selected}
        for batch_index in range(plan.required_unanimous_batches):
            batch_rows = v1.read_csv(v1._batch_path(design, mode, domain, batch_index))
            expected_rows = 2 * plan.episodes_per_batch * len(domain_selected)
            if len(batch_rows) != expected_rows:
                raise RuntimeError(f"unexpected row count in {mode} pilot batch for {domain}")
            if any(
                row["domain"] != domain or row["task"] not in advantages
                for row in batch_rows
            ):
                raise RuntimeError(f"unexpected task in {mode} pilot batch for {domain}")
            for task in domain_selected:
                candidate = [
                    row
                    for row in batch_rows
                    if row["task"] == task and row["role"] == "candidate"
                ]
                fallback = [
                    row
                    for row in batch_rows
                    if row["task"] == task and row["role"] == "fallback"
                ]
                expected = plan.episodes_per_batch
                if len(candidate) != expected or len(fallback) != expected:
                    raise RuntimeError(f"incomplete {mode} pilot batch for {task}")
                expected_seed_base = (
                    int(pilot["seed_base"])
                    + task_indices[task] * int(pilot["task_seed_stride"])
                    + batch_index * int(pilot["batch_seed_stride"])
                    + _recorded_seed_offset(task)
                )
                expected_seeds = list(range(expected_seed_base, expected_seed_base + expected))
                candidate_seeds = [int(row["seed"]) for row in candidate]
                fallback_seeds = [int(row["seed"]) for row in fallback]
                if candidate_seeds != expected_seeds or fallback_seeds != expected_seeds:
                    raise RuntimeError(f"common-random-number pairing changed for {task}")
                if any(row["policy"] != candidates_by_task[task] for row in candidate):
                    raise RuntimeError(f"candidate policy changed in pilot batch for {task}")
                fallback_policy = POLICY_LIBRARIES[domain].fallback
                if any(row["policy"] != fallback_policy for row in fallback):
                    raise RuntimeError(f"fallback policy changed in pilot batch for {task}")
                advantages[task].append(v1._score(candidate) - v1._score(fallback))
        for task, values in advantages.items():
            result = verify_familywise_promotion(values, plan)
            rows.append(
                {
                    "mode": mode,
                    "task": task,
                    **asdict(result),
                    "batch_advantages": json.dumps(values, separators=(",", ":")),
                }
            )
    return rows


def lock_gates(design: dict) -> Path:
    rows = _gate_mode(design, "proposal") + _gate_mode(design, "random")
    path = Path(design["evaluation"]["output_root"]) / "gate_decisions.csv"
    v1.write_csv_once(path, rows)
    return path


def combine_results(design: dict) -> tuple[Path, Path]:
    original = v1._gate_mode
    v1._gate_mode = _gate_mode
    try:
        return v1.combine_results(design)
    finally:
        v1._gate_mode = original


def finalize_registration(
    draft_path: Path,
    output_path: Path,
    *,
    registration_url: str,
    registered_at: str,
    provider: str,
) -> Path:
    wrapper, _amendment, _design = validate_amendment(
        draft_path,
        require_registration=False,
    )
    if wrapper["status"] != "awaiting_external_amendment_registration":
        raise RuntimeError("only an amendment registration draft can be finalized")
    registered = dict(wrapper)
    registered["status"] = "externally_registered_amendment_locked"
    registered["registration"] = {
        "url": registration_url,
        "provider": provider,
        "registered_at": registered_at,
        "registered_amendment_sha256": wrapper["amendment_sha256"],
    }
    v1.write_json_once(output_path, registered)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dry-run")
    finalize = commands.add_parser("finalize-registration")
    finalize.add_argument("--url", required=True)
    finalize.add_argument("--registered-at", required=True)
    finalize.add_argument("--provider", default="Open Science Framework (OSF)")
    finalize.add_argument("--output", type=Path, required=True)
    commands.add_parser("lock-gates")
    final = commands.add_parser("final")
    final.add_argument("--domain", choices=DOMAINS, required=True)
    final.add_argument("--seed-index", type=int, required=True)
    final.add_argument("--environment-source", type=Path, required=True)
    commands.add_parser("combine")
    args = parser.parse_args()
    require_registration = args.command not in {"dry-run", "finalize-registration"}
    wrapper, _amendment, design = validate_amendment(
        args.amendment,
        require_registration=require_registration,
    )
    if args.command == "dry-run":
        print(f"amendment_status={wrapper['status']}")
        print("base_protocol_hashes_valid=true")
        print("amended_source_hash_valid=true")
        allowed = wrapper["status"] == "externally_registered_amendment_locked"
        print(f"confirmation_execution_allowed={str(allowed).lower()}")
    elif args.command == "finalize-registration":
        print(
            "registered_wrapper="
            f"{finalize_registration(args.amendment, args.output, registration_url=args.url, registered_at=args.registered_at, provider=args.provider)}"
        )
    elif args.command == "lock-gates":
        print(f"gate_output={lock_gates(design)}")
    elif args.command == "final":
        print(
            "final_output="
            f"{v1.run_final_seed(design, args.domain, args.seed_index, args.environment_source)}"
        )
    elif args.command == "combine":
        task_path, summary_path = combine_results(design)
        print(f"combined_tasks={task_path}")
        print(f"combined_summary={summary_path}")


if __name__ == "__main__":
    main()
