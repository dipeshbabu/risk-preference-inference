"""Guarded, resumable confirmation state for the registered v2 protocol.

This module does not make confirmation tasks executable by itself. It defines
the deterministic pilot state machine that a separately registered wrapper
must validate before collecting or finalizing any confirmation outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean

from experiments.anytime_familywise_router import (
    AnytimeFamilywisePlan,
    AnytimeFamilywiseRouter,
)
from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    V2ExternalTask,
    all_tasks,
    canonical_episode_seed_base,
    canonical_sha256,
    domain_tasks,
    expected_episode_seeds,
    outcome_implementation_sha256,
    task_manifest_sha256,
)
from experiments.frontier_v2_external_adapters import (
    outcome_rows,
    run_gymnasium_task,
    run_minigrid_task,
    run_or_gym_task,
    run_safety_gymnasium_task,
)
from experiments.frontier_v2_protocol_lock import validate_protocol
from experiments.frontier_v2_source_audit import SOURCE_DIRECTORIES


PILOT_RECORD_PROTOCOL = "riskshiftbench-frontier-v2-pilot-record-v1"


def _proposal_index(router_lock: dict) -> dict[str, dict]:
    proposals = router_lock.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise RuntimeError("router lock has no proposal family")
    index = {proposal["task"]: proposal for proposal in proposals}
    if len(index) != len(proposals):
        raise RuntimeError("router lock contains duplicate proposal tasks")
    expected_tasks = {task.name for task in all_tasks("confirmation")}
    if set(index) != expected_tasks:
        raise RuntimeError("router lock no longer covers the confirmation suite")
    return index


def validate_router_lock_for_confirmation(router_lock: dict) -> None:
    if router_lock.get("protocol_id") != "riskshiftbench-frontier-v2-router-lock-v1":
        raise RuntimeError("unexpected v2 router-lock protocol")
    expected_hash = router_lock.get("router_lock_canonical_sha256")
    unsigned = dict(router_lock)
    unsigned.pop("router_lock_canonical_sha256", None)
    if expected_hash != canonical_sha256(unsigned):
        raise RuntimeError("router-lock canonical hash changed")
    if router_lock.get("outcome_implementation_sha256") != outcome_implementation_sha256():
        raise RuntimeError("outcome implementation changed after the router lock")
    if router_lock.get("confirmation_task_manifest_sha256") != task_manifest_sha256(
        all_tasks("confirmation")
    ):
        raise RuntimeError("confirmation task manifest changed after the router lock")
    proposals = _proposal_index(router_lock)
    if int(router_lock.get("proposal_family_size", -1)) != len(proposals):
        raise RuntimeError("proposal-family size changed")


def plan_from_router_lock(router_lock: dict) -> AnytimeFamilywisePlan:
    validate_router_lock_for_confirmation(router_lock)
    proposals = sorted(router_lock["proposals"], key=lambda item: item["task"])
    settings = router_lock["anytime_plan"]
    if settings.get("e_process_method") != "betting_mixture":
        raise RuntimeError("registered primary e-process is not the betting mixture")
    if settings.get("allocation") != "certified":
        raise RuntimeError("registered primary allocation is not certified")
    bounds = settings["observation_bounds"]
    return AnytimeFamilywisePlan(
        task_names=tuple(proposal["task"] for proposal in proposals),
        familywise_alpha=float(settings["familywise_alpha"]),
        futility_familywise_alpha=float(settings["futility_familywise_alpha"]),
        effect_margin=float(settings["effect_margin"]),
        observation_lower=float(bounds[0]),
        observation_upper=float(bounds[1]),
        minimum_observations=int(settings["minimum_observations"]),
        maximum_observations_per_task=int(
            settings["maximum_observations_per_task"]
        ),
        task_weights=tuple(
            (proposal["task"], float(proposal["task_weight"]))
            for proposal in proposals
        ),
        e_process_method=settings["e_process_method"],
        betting_fraction_grid=tuple(
            float(value) for value in settings["betting_fraction_grid"]
        ),
        planning_effect_gaps=tuple(
            (proposal["task"], float(proposal["planning_effect_gap"]))
            for proposal in proposals
        ),
        resolution_familywise_beta=float(
            settings["resolution_familywise_beta"]
        ),
    )


def pilot_record_sha256(record: dict) -> str:
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    return canonical_sha256(unsigned)


def _confirmation_task_index() -> dict[str, object]:
    return {task.name: task for task in all_tasks("confirmation")}


def _validate_outcome_identity(
    outcome: dict,
    *,
    task: str,
    domain: str,
    policy: str,
    seed: int,
) -> None:
    if outcome.get("task") != task or outcome.get("domain") != domain:
        raise RuntimeError("pilot outcome task identity changed")
    if outcome.get("policy") != policy:
        raise RuntimeError("pilot outcome policy changed")
    if int(outcome.get("seed", -1)) != seed:
        raise RuntimeError("pilot outcome seed changed")
    score = float(outcome["score"])
    if not 0.0 <= score <= 1.0:
        raise RuntimeError("pilot outcome score lies outside [0, 1]")


def build_pilot_record(
    router_lock: dict,
    *,
    sequence_index: int,
    within_task_index: int,
    task_name: str,
    candidate_outcome: dict,
    fallback_outcome: dict,
    previous_record_sha256: str | None,
) -> dict:
    proposals = _proposal_index(router_lock)
    try:
        proposal = proposals[task_name]
        task = _confirmation_task_index()[task_name]
    except KeyError as error:
        raise KeyError(f"unknown confirmation proposal: {task_name}") from error
    seed_base = canonical_episode_seed_base(task, stream="pilot") + within_task_index
    expected_seed = expected_episode_seeds(
        task,
        episodes=1,
        seed_base=seed_base,
    )[0]
    _validate_outcome_identity(
        candidate_outcome,
        task=task_name,
        domain=proposal["domain"],
        policy=proposal["candidate_policy"],
        seed=expected_seed,
    )
    _validate_outcome_identity(
        fallback_outcome,
        task=task_name,
        domain=proposal["domain"],
        policy=proposal["fallback_policy"],
        seed=expected_seed,
    )
    difference = float(candidate_outcome["score"]) - float(
        fallback_outcome["score"]
    )
    record = {
        "protocol_id": PILOT_RECORD_PROTOCOL,
        "sequence_index": int(sequence_index),
        "within_task_index": int(within_task_index),
        "task": task_name,
        "domain": proposal["domain"],
        "candidate_policy": proposal["candidate_policy"],
        "fallback_policy": proposal["fallback_policy"],
        "seed": expected_seed,
        "candidate_outcome": candidate_outcome,
        "fallback_outcome": fallback_outcome,
        "paired_score_difference": difference,
        "previous_record_sha256": previous_record_sha256,
    }
    record["record_sha256"] = pilot_record_sha256(record)
    return record


def replay_pilot_records(
    router_lock: dict,
    records: list[dict],
) -> AnytimeFamilywiseRouter:
    """Replay and authenticate a possibly partial primary pilot log."""

    plan = plan_from_router_lock(router_lock)
    router = AnytimeFamilywiseRouter(plan)
    settings = router_lock["anytime_plan"]
    budget = int(settings["global_observation_budget"])
    forced = int(settings["forced_initial_observations"])
    if len(records) > budget:
        raise RuntimeError("pilot log exceeds the registered global budget")
    previous_hash = None
    task_index = _confirmation_task_index()
    proposals = _proposal_index(router_lock)
    for sequence_index, record in enumerate(records):
        if record.get("protocol_id") != PILOT_RECORD_PROTOCOL:
            raise RuntimeError("unexpected pilot-record protocol")
        if int(record.get("sequence_index", -1)) != sequence_index:
            raise RuntimeError("pilot sequence is not contiguous")
        if record.get("previous_record_sha256") != previous_hash:
            raise RuntimeError("pilot record hash chain is broken")
        if record.get("record_sha256") != pilot_record_sha256(record):
            raise RuntimeError("pilot record content hash changed")
        expected_task = router.next_task(
            "certified", forced_initial_observations=forced
        )
        if expected_task is None or record.get("task") != expected_task:
            raise RuntimeError("pilot allocation order differs from the registered rule")
        evidence = router.evidence(expected_task)
        if int(record.get("within_task_index", -1)) != evidence.observations:
            raise RuntimeError("within-task pilot index changed")
        proposal = proposals[expected_task]
        task = task_index[expected_task]
        expected_seed = expected_episode_seeds(
            task,
            episodes=1,
            seed_base=(
                canonical_episode_seed_base(task, stream="pilot")
                + evidence.observations
            ),
        )[0]
        if int(record.get("seed", -1)) != expected_seed:
            raise RuntimeError("pilot seed schedule changed")
        _validate_outcome_identity(
            record["candidate_outcome"],
            task=expected_task,
            domain=proposal["domain"],
            policy=proposal["candidate_policy"],
            seed=expected_seed,
        )
        _validate_outcome_identity(
            record["fallback_outcome"],
            task=expected_task,
            domain=proposal["domain"],
            policy=proposal["fallback_policy"],
            seed=expected_seed,
        )
        observed_difference = float(record["candidate_outcome"]["score"]) - float(
            record["fallback_outcome"]["score"]
        )
        if float(record.get("paired_score_difference")) != observed_difference:
            raise RuntimeError("paired pilot difference changed")
        router.update(expected_task, observed_difference)
        previous_hash = record["record_sha256"]
    return router


def next_pilot_request(router_lock: dict, records: list[dict]) -> dict | None:
    """Return the unique next registered request, or ``None`` at termination."""

    router = replay_pilot_records(router_lock, records)
    settings = router_lock["anytime_plan"]
    if router.total_observations() >= int(settings["global_observation_budget"]):
        return None
    task_name = router.next_task(
        "certified",
        forced_initial_observations=int(settings["forced_initial_observations"]),
    )
    if task_name is None:
        return None
    task = _confirmation_task_index()[task_name]
    proposal = _proposal_index(router_lock)[task_name]
    within_task_index = router.evidence(task_name).observations
    seed_base = canonical_episode_seed_base(task, stream="pilot") + within_task_index
    return {
        "sequence_index": len(records),
        "within_task_index": within_task_index,
        "task": task_name,
        "domain": proposal["domain"],
        "candidate_policy": proposal["candidate_policy"],
        "fallback_policy": proposal["fallback_policy"],
        "seed_base": seed_base,
        "expected_seed": expected_episode_seeds(
            task,
            episodes=1,
            seed_base=seed_base,
        )[0],
        "previous_record_sha256": (
            records[-1]["record_sha256"] if records else None
        ),
    }


def pilot_decision_summary(router_lock: dict, records: list[dict]) -> dict:
    router = replay_pilot_records(router_lock, records)
    settings = router_lock["anytime_plan"]
    next_request = next_pilot_request(router_lock, records)
    if next_request is not None:
        raise RuntimeError("pilot decisions cannot be frozen before termination")
    decisions = {
        task: asdict(evidence) for task, evidence in router.decisions().items()
    }
    return {
        "protocol_id": "riskshiftbench-frontier-v2-pilot-decisions-v1",
        "router_lock_canonical_sha256": router_lock[
            "router_lock_canonical_sha256"
        ],
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "confirmation_task_manifest_sha256": task_manifest_sha256(
            all_tasks("confirmation")
        ),
        "global_observation_budget": int(settings["global_observation_budget"]),
        "total_observations": router.total_observations(),
        "pilot_terminated_by": (
            "all_tasks_resolved"
            if router.next_task(
                "certified",
                forced_initial_observations=int(
                    settings["forced_initial_observations"]
                ),
            )
            is None
            else "global_budget"
        ),
        "accepted_tasks": list(router.accepted_tasks()),
        "decisions": decisions,
        "pilot_record_count": len(records),
        "pilot_chain_tip_sha256": (
            records[-1]["record_sha256"] if records else None
        ),
    }


def load_pilot_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank line in pilot log at {line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid pilot JSON record at line {line_number}"
                ) from error
    return records


def append_pilot_record(path: Path, record: dict) -> None:
    """Durably append one authenticated record without rewriting prior outcomes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite confirmation result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _registered_adapter_task(task: V2ExternalTask) -> V2ExternalTask:
    """Create the capability-bearing view consumed by development adapters.

    The environment identity, parameters, features, and task name are unchanged.
    Only the execution label is changed after the registered wrapper has passed
    every protocol check. This is the separately registered wrapper anticipated
    by the development adapters' confirmation guard.
    """

    if task.split != "confirmation":
        raise RuntimeError("registered adapter capability requires a confirmation task")
    return replace(task, split="calibration")


def _run_registered_policy(
    task: V2ExternalTask,
    policy: str,
    *,
    episodes: int,
    seed_base: int,
    source_root: Path,
) -> list[dict]:
    executable = _registered_adapter_task(task)
    source = source_root / SOURCE_DIRECTORIES[DOMAIN_SPECS[task.domain].codebase]
    codebase = DOMAIN_SPECS[task.domain].codebase
    if codebase == "gymnasium":
        outcomes = run_gymnasium_task(executable, policy, episodes, seed_base, source)
    elif codebase == "minigrid":
        outcomes = run_minigrid_task(executable, policy, episodes, seed_base, source)
    elif codebase == "or_gym":
        outcomes = run_or_gym_task(executable, policy, episodes, seed_base, source)
    elif codebase == "safety_gymnasium":
        outcomes = run_safety_gymnasium_task(
            executable, policy, episodes, seed_base, source
        )
    else:
        raise KeyError(codebase)
    return outcome_rows(outcomes)


def collect_registered_pair(
    design: dict,
    request: dict,
    *,
    codebase: str,
) -> dict:
    """Collect exactly one registered pilot pair inside its pinned environment."""

    router_lock = design["router_lock"]["content"]
    proposals = _proposal_index(router_lock)
    tasks = _confirmation_task_index()
    task_name = request["task"]
    try:
        task = tasks[task_name]
        proposal = proposals[task_name]
    except KeyError as error:
        raise RuntimeError("worker received an unknown confirmation task") from error
    if DOMAIN_SPECS[task.domain].codebase != codebase:
        raise RuntimeError("worker received a task from another codebase")
    within_task_index = int(request["within_task_index"])
    expected_seed_base = (
        canonical_episode_seed_base(task, stream="pilot") + within_task_index
    )
    if int(request["seed_base"]) != expected_seed_base:
        raise RuntimeError("worker pilot seed base differs from the registered schedule")
    if request["candidate_policy"] != proposal["candidate_policy"] or request[
        "fallback_policy"
    ] != proposal["fallback_policy"]:
        raise RuntimeError("worker pilot policy pair differs from the router lock")
    source_root = Path(design["artifact_roots"]["environment_source_root"])
    candidate = _run_registered_policy(
        task,
        proposal["candidate_policy"],
        episodes=1,
        seed_base=expected_seed_base,
        source_root=source_root,
    )[0]
    fallback = _run_registered_policy(
        task,
        proposal["fallback_policy"],
        episodes=1,
        seed_base=expected_seed_base,
        source_root=source_root,
    )[0]
    return {"candidate_outcome": candidate, "fallback_outcome": fallback}


def collect_registered_final_task(
    design: dict,
    task_name: str,
    *,
    codebase: str,
) -> dict:
    """Collect the frozen candidate and fallback on one disjoint final stream."""

    router_lock = design["router_lock"]["content"]
    proposals = _proposal_index(router_lock)
    tasks = _confirmation_task_index()
    try:
        task = tasks[task_name]
        proposal = proposals[task_name]
    except KeyError as error:
        raise RuntimeError("worker received an unknown final task") from error
    if DOMAIN_SPECS[task.domain].codebase != codebase:
        raise RuntimeError("worker received a final task from another codebase")
    episodes = int(design["confirmation"]["final"]["episodes_per_task_policy"])
    seed_base = canonical_episode_seed_base(task, stream="final")
    source_root = Path(design["artifact_roots"]["environment_source_root"])
    candidate = _run_registered_policy(
        task,
        proposal["candidate_policy"],
        episodes=episodes,
        seed_base=seed_base,
        source_root=source_root,
    )
    fallback = _run_registered_policy(
        task,
        proposal["fallback_policy"],
        episodes=episodes,
        seed_base=seed_base,
        source_root=source_root,
    )
    expected_seeds = list(
        expected_episode_seeds(task, episodes=episodes, seed_base=seed_base)
    )
    if [int(row["seed"]) for row in candidate] != expected_seeds or [
        int(row["seed"]) for row in fallback
    ] != expected_seeds:
        raise RuntimeError("final common-random-number seed schedule changed")
    return {
        "protocol_id": "riskshiftbench-frontier-v2-final-task-v1",
        "task": task_name,
        "domain": task.domain,
        "candidate_policy": proposal["candidate_policy"],
        "fallback_policy": proposal["fallback_policy"],
        "episodes_per_policy": episodes,
        "canonical_seed_base": seed_base,
        "candidate_outcomes": candidate,
        "fallback_outcomes": fallback,
    }


def authenticate_worker_request(
    router_lock: dict,
    pilot_log_path: Path,
    request: dict,
) -> None:
    """Require the worker request to equal the unique next hash-chained step."""

    records = load_pilot_records(pilot_log_path)
    expected_request = next_pilot_request(router_lock, records)
    observed_request = {key: request.get(key) for key in expected_request or {}}
    if expected_request is None or observed_request != expected_request:
        raise RuntimeError(
            "worker request is not the unique next authenticated pilot step"
        )


def _validated_pilot_decisions(design: dict, pilot_log_path: Path) -> dict:
    router_lock = design["router_lock"]["content"]
    records = load_pilot_records(pilot_log_path)
    expected = pilot_decision_summary(router_lock, records)
    decisions_path = pilot_log_path.parent / "primary_decisions.json"
    if not decisions_path.is_file():
        raise RuntimeError("final execution is blocked until pilot decisions are frozen")
    observed = json.loads(decisions_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("frozen pilot decisions differ from the authenticated log")
    return observed


def _next_final_task(design: dict) -> str | None:
    output_root = Path(design["confirmation"]["output_root"]) / "final" / "tasks"
    for task in sorted(all_tasks("confirmation"), key=lambda item: item.name):
        if not (output_root / f"{task.name}.json").is_file():
            return task.name
    return None


def authenticate_final_worker_request(
    design: dict,
    pilot_log_path: Path,
    request: dict,
) -> None:
    _validated_pilot_decisions(design, pilot_log_path)
    expected_task = _next_final_task(design)
    if expected_task is None or request.get("task") != expected_task:
        raise RuntimeError("worker request is not the unique next registered final task")


def run_worker(protocol_path: Path, codebase: str, pilot_log_path: Path) -> None:
    """Serve registered episode requests over newline-delimited JSON."""

    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    if codebase not in set(SOURCE_DIRECTORIES):
        raise KeyError(codebase)
    print(json.dumps({"status": "ready", "codebase": codebase}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                print(json.dumps({"status": "closed", "codebase": codebase}), flush=True)
                return
            if request.get("command") not in {"pilot_pair", "final_task"}:
                raise RuntimeError("unknown registered worker command")
            if request["command"] == "pilot_pair":
                authenticate_worker_request(
                    design["router_lock"]["content"], pilot_log_path, request
                )
                result = collect_registered_pair(design, request, codebase=codebase)
            else:
                authenticate_final_worker_request(design, pilot_log_path, request)
                result = {
                    "final_task": collect_registered_final_task(
                        design, request["task"], codebase=codebase
                    )
                }
            print(json.dumps({"status": "ok", **result}, sort_keys=True), flush=True)
        except Exception as error:  # worker boundary must return an actionable failure
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return


class RegisteredWorkerPool:
    def __init__(
        self,
        protocol_path: Path,
        interpreters: dict[str, Path],
        pilot_log_path: Path,
    ):
        self.protocol_path = protocol_path
        self.interpreters = interpreters
        self.pilot_log_path = pilot_log_path
        self.processes: dict[str, subprocess.Popen[str]] = {}

    def _start(self, codebase: str) -> subprocess.Popen[str]:
        try:
            interpreter = self.interpreters[codebase]
        except KeyError as error:
            raise RuntimeError(f"no registered worker interpreter for {codebase}") from error
        if not interpreter.is_file():
            raise RuntimeError(f"registered worker interpreter is missing: {interpreter}")
        process = subprocess.Popen(
            [
                str(interpreter),
                "-m",
                "experiments.frontier_v2_confirmation_runtime",
                "worker",
                "--protocol",
                str(self.protocol_path),
                "--codebase",
                codebase,
                "--pilot-log",
                str(self.pilot_log_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("registered worker stdout was not created")
        ready_line = process.stdout.readline()
        if not ready_line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"registered worker failed to start: {stderr}")
        ready = json.loads(ready_line)
        if ready != {"status": "ready", "codebase": codebase}:
            raise RuntimeError(f"registered worker returned invalid readiness: {ready}")
        self.processes[codebase] = process
        return process

    def request(self, codebase: str, request: dict, *, command: str = "pilot_pair") -> dict:
        process = self.processes.get(codebase) or self._start(codebase)
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("registered worker pipes are unavailable")
        process.stdin.write(json.dumps({"command": command, **request}) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"registered worker exited without a response: {stderr}")
        response = json.loads(response_line)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"registered worker failed: {response.get('error_type')}: "
                f"{response.get('error')}"
            )
        return response

    def close(self) -> None:
        for process in self.processes.values():
            if process.poll() is not None:
                continue
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "close"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=30)

    def __enter__(self) -> RegisteredWorkerPool:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def run_registered_pilot(
    protocol_path: Path,
    *,
    interpreters: dict[str, Path],
) -> dict:
    """Resume or complete the registered primary adaptive pilot."""

    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    router_lock = design["router_lock"]["content"]
    output_root = Path(design["confirmation"]["output_root"])
    log_path = output_root / "pilot" / "primary_records.jsonl"
    decisions_path = output_root / "pilot" / "primary_decisions.json"
    records = load_pilot_records(log_path)
    replay_pilot_records(router_lock, records)
    if decisions_path.exists():
        observed = json.loads(decisions_path.read_text(encoding="utf-8"))
        if observed != pilot_decision_summary(router_lock, records):
            raise RuntimeError("frozen pilot decisions changed")
        return observed

    with RegisteredWorkerPool(protocol_path, interpreters, log_path) as workers:
        while True:
            request = next_pilot_request(router_lock, records)
            if request is None:
                break
            task = _confirmation_task_index()[request["task"]]
            codebase = DOMAIN_SPECS[task.domain].codebase
            outcome = workers.request(codebase, request)
            record = build_pilot_record(
                router_lock,
                sequence_index=request["sequence_index"],
                within_task_index=request["within_task_index"],
                task_name=request["task"],
                candidate_outcome=outcome["candidate_outcome"],
                fallback_outcome=outcome["fallback_outcome"],
                previous_record_sha256=request["previous_record_sha256"],
            )
            append_pilot_record(log_path, record)
            records.append(record)
    summary = pilot_decision_summary(router_lock, records)
    write_json_once(decisions_path, summary)
    return summary


def _byte_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_registered_final(
    protocol_path: Path,
    *,
    interpreters: dict[str, Path],
) -> dict:
    """Resume or complete the registered disjoint final task evaluation."""

    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    output_root = Path(design["confirmation"]["output_root"])
    pilot_log_path = output_root / "pilot" / "primary_records.jsonl"
    decisions = _validated_pilot_decisions(design, pilot_log_path)
    final_root = output_root / "final"
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decisions_path = output_root / "pilot" / "primary_decisions.json"
        if _byte_sha256(decisions_path) != manifest["pilot_decisions_byte_sha256"]:
            raise RuntimeError("pilot decisions changed after final evaluation")
        for record in manifest["task_artifacts"]:
            path = Path(record["path"])
            if not path.is_file() or _byte_sha256(path) != record["byte_sha256"]:
                raise RuntimeError(f"frozen final task artifact changed: {path}")
        return manifest

    with RegisteredWorkerPool(
        protocol_path, interpreters, pilot_log_path
    ) as workers:
        while True:
            task_name = _next_final_task(design)
            if task_name is None:
                break
            task = _confirmation_task_index()[task_name]
            codebase = DOMAIN_SPECS[task.domain].codebase
            response = workers.request(
                codebase,
                {"task": task_name},
                command="final_task",
            )
            payload = response["final_task"]
            decision = decisions["decisions"][task_name]["decision"]
            proposal = _proposal_index(design["router_lock"]["content"])[task_name]
            payload["pilot_decision"] = decision
            payload["deployed_policy"] = (
                proposal["candidate_policy"]
                if decision == "accept_candidate"
                else proposal["fallback_policy"]
            )
            write_json_once(final_root / "tasks" / f"{task_name}.json", payload)

    task_records = []
    for task in sorted(all_tasks("confirmation"), key=lambda item: item.name):
        path = final_root / "tasks" / f"{task.name}.json"
        if not path.is_file():
            raise RuntimeError(f"registered final task artifact is missing: {task.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("pilot_decision") != decisions["decisions"][task.name][
            "decision"
        ]:
            raise RuntimeError("final task route differs from the frozen pilot decision")
        _validate_final_task_payload(design, payload)
        task_records.append({"path": path.as_posix(), "byte_sha256": _byte_sha256(path)})
    decisions_path = output_root / "pilot" / "primary_decisions.json"
    manifest = {
        "protocol_id": "riskshiftbench-frontier-v2-final-manifest-v1",
        "confirmation_task_manifest_sha256": task_manifest_sha256(
            all_tasks("confirmation")
        ),
        "router_lock_canonical_sha256": design["router_lock"][
            "router_lock_canonical_sha256"
        ],
        "pilot_decisions_byte_sha256": _byte_sha256(decisions_path),
        "task_count": len(task_records),
        "episodes_per_task_policy": int(
            design["confirmation"]["final"]["episodes_per_task_policy"]
        ),
        "task_artifacts": task_records,
    }
    write_json_once(manifest_path, manifest)
    return manifest


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sample")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _domain_task_effects(task_effects: dict[str, float]) -> dict[str, list[float]]:
    expected = _confirmation_task_index()
    if set(task_effects) != set(expected):
        raise RuntimeError("analysis task effects do not cover the confirmation suite")
    return {
        domain: [
            float(task_effects[task.name])
            for task in domain_tasks(domain, "confirmation")
        ]
        for domain in DOMAIN_SPECS
    }


def _equal_domain_mean(effects: dict[str, list[float]]) -> float:
    return fmean(fmean(effects[domain]) for domain in DOMAIN_SPECS)


def primary_analysis_from_task_effects(
    route_effects: dict[str, float],
    candidate_effects: dict[str, float],
    *,
    task_bootstrap_replicates: int,
    domain_bootstrap_replicates: int,
    sign_flip_replicates: int,
    seed: int,
) -> dict:
    if (
        task_bootstrap_replicates <= 0
        or domain_bootstrap_replicates <= 0
        or sign_flip_replicates <= 0
    ):
        raise ValueError("analysis replicate counts must be positive")
    routes = _domain_task_effects(route_effects)
    candidates = _domain_task_effects(candidate_effects)
    observed = _equal_domain_mean(routes)
    rng = random.Random(seed)
    fixed_bootstrap = []
    domain_bootstrap = []
    domains = tuple(DOMAIN_SPECS)
    for _replicate in range(task_bootstrap_replicates):
        fixed_draw = {
            domain: [rng.choice(routes[domain]) for _ in routes[domain]]
            for domain in domains
        }
        fixed_bootstrap.append(_equal_domain_mean(fixed_draw))
    for _replicate in range(domain_bootstrap_replicates):
        sampled_domains = [rng.choice(domains) for _ in domains]
        domain_bootstrap.append(
            fmean(
                fmean(rng.choice(routes[domain]) for _ in routes[domain])
                for domain in sampled_domains
            )
        )
    exceedances = 0
    for _replicate in range(sign_flip_replicates):
        randomized = {
            domain: [value * rng.choice((-1.0, 1.0)) for value in routes[domain]]
            for domain in domains
        }
        exceedances += _equal_domain_mean(randomized) >= observed
    domain_effects = {
        domain: fmean(routes[domain]) for domain in domains
    }
    return {
        "equal_domain_mean_route_effect": observed,
        "equal_domain_mean_candidate_everywhere_effect": _equal_domain_mean(
            candidates
        ),
        "fixed_domain_task_bootstrap_95_ci": [
            _quantile(fixed_bootstrap, 0.025),
            _quantile(fixed_bootstrap, 0.975),
        ],
        "domain_resampling_bootstrap_95_ci": [
            _quantile(domain_bootstrap, 0.025),
            _quantile(domain_bootstrap, 0.975),
        ],
        "secondary_one_sided_task_sign_flip_p": (exceedances + 1)
        / (sign_flip_replicates + 1),
        "per_domain_route_effects": domain_effects,
        "leave_one_domain_out_route_effects": {
            omitted: fmean(
                domain_effects[domain] for domain in domains if domain != omitted
            )
            for omitted in domains
        },
        "task_bootstrap_replicates": task_bootstrap_replicates,
        "domain_bootstrap_replicates": domain_bootstrap_replicates,
        "sign_flip_replicates": sign_flip_replicates,
        "random_seed": seed,
    }


def fixed_family_bounded_mean_test(
    candidate_episode_differences: dict[str, list[float]],
    accepted_tasks: set[str],
    *,
    alpha: float = 0.05,
) -> dict:
    """Test the frozen router's fixed-family mean with a Hoeffding bound.

    Conditional on the pilot-frozen route, each accepted task contributes its
    separately seeded final paired differences in [-1, 1].  Fallback-routed
    tasks contribute the constant zero.  The weighted Hoeffding inequality is
    valid for independent, non-identically distributed final episode pairs.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    expected = _confirmation_task_index()
    if set(candidate_episode_differences) != set(expected):
        raise RuntimeError("episode differences do not cover the confirmation suite")
    unknown_accepted = accepted_tasks - set(expected)
    if unknown_accepted:
        raise RuntimeError(f"accepted task family contains unknown tasks: {unknown_accepted}")

    domain_count = len(DOMAIN_SPECS)
    observed = 0.0
    accepted_task_weight = 0.0
    weighted_range_square_sum = 0.0
    accepted_episode_count = 0
    for domain in DOMAIN_SPECS:
        tasks = domain_tasks(domain, "confirmation")
        task_weight = 1.0 / (domain_count * len(tasks))
        for task in tasks:
            differences = [
                float(value) for value in candidate_episode_differences[task.name]
            ]
            if not differences:
                raise RuntimeError(f"final episode differences are empty: {task.name}")
            if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in differences):
                raise RuntimeError(
                    f"final paired score difference lies outside [-1, 1]: {task.name}"
                )
            if task.name not in accepted_tasks:
                continue
            episode_weight = task_weight / len(differences)
            observed += task_weight * fmean(differences)
            accepted_task_weight += task_weight
            accepted_episode_count += len(differences)
            weighted_range_square_sum += len(differences) * (
                2.0 * episode_weight
            ) ** 2

    if weighted_range_square_sum == 0.0:
        p_value = 1.0
        lower_bound = 0.0
    else:
        positive_statistic = max(observed, 0.0)
        p_value = math.exp(
            -2.0 * positive_statistic * positive_statistic
            / weighted_range_square_sum
        )
        radius = math.sqrt(
            0.5 * weighted_range_square_sum * math.log(1.0 / alpha)
        )
        lower_bound = max(-accepted_task_weight, observed - radius)

    return {
        "test": "one-sided weighted Hoeffding test",
        "null": "fixed-family equal-domain deployed-router mean <= 0",
        "alternative": "fixed-family equal-domain deployed-router mean > 0",
        "alpha": alpha,
        "score_difference_bounds": [-1.0, 1.0],
        "observed_equal_domain_mean_route_effect": observed,
        "one_sided_p": p_value,
        "one_sided_lower_confidence_bound": lower_bound,
        "reject_null": observed > 0.0 and p_value <= alpha,
        "accepted_task_count": len(accepted_tasks),
        "accepted_episode_count": accepted_episode_count,
        "accepted_task_weight": accepted_task_weight,
        "weighted_range_square_sum": weighted_range_square_sum,
        "conditioning": "pilot-frozen routes",
        "assumption": (
            "independent bounded final paired episode units conditional on the "
            "pilot-frozen route; identical distributions are not required"
        ),
    }


def _validate_final_task_payload(
    design: dict, payload: dict
) -> tuple[float, float, list[float]]:
    task_name = payload["task"]
    task = _confirmation_task_index()[task_name]
    proposal = _proposal_index(design["router_lock"]["content"])[task_name]
    episodes = int(design["confirmation"]["final"]["episodes_per_task_policy"])
    candidate = payload["candidate_outcomes"]
    fallback = payload["fallback_outcomes"]
    if len(candidate) != episodes or len(fallback) != episodes:
        raise RuntimeError("final task artifact has an incomplete episode schedule")
    expected_seeds = expected_episode_seeds(
        task,
        episodes=episodes,
        seed_base=canonical_episode_seed_base(task, stream="final"),
    )
    for index, (candidate_row, fallback_row, seed) in enumerate(
        zip(candidate, fallback, expected_seeds, strict=True)
    ):
        _validate_outcome_identity(
            candidate_row,
            task=task_name,
            domain=task.domain,
            policy=proposal["candidate_policy"],
            seed=seed,
        )
        _validate_outcome_identity(
            fallback_row,
            task=task_name,
            domain=task.domain,
            policy=proposal["fallback_policy"],
            seed=seed,
        )
        if int(candidate_row.get("episode", index)) != index or int(
            fallback_row.get("episode", index)
        ) != index:
            raise RuntimeError("final task episode indices changed")
    candidate_differences = [
        float(candidate_row["score"]) - float(fallback_row["score"])
        for candidate_row, fallback_row in zip(candidate, fallback, strict=True)
    ]
    candidate_effect = fmean(candidate_differences)
    route_effect = (
        candidate_effect if payload["pilot_decision"] == "accept_candidate" else 0.0
    )
    expected_deployed = (
        proposal["candidate_policy"]
        if payload["pilot_decision"] == "accept_candidate"
        else proposal["fallback_policy"]
    )
    if payload["deployed_policy"] != expected_deployed:
        raise RuntimeError("final deployed policy differs from the frozen pilot route")
    return route_effect, candidate_effect, candidate_differences


def analyze_registered_final(protocol_path: Path) -> dict:
    _wrapper, design = validate_protocol(protocol_path, require_registration=True)
    output_root = Path(design["confirmation"]["output_root"])
    pilot_log_path = output_root / "pilot" / "primary_records.jsonl"
    decisions = _validated_pilot_decisions(design, pilot_log_path)
    manifest = run_registered_final_manifest_only(design)
    route_effects = {}
    candidate_effects = {}
    candidate_episode_differences = {}
    harmful_accepted = []
    for record in manifest["task_artifacts"]:
        path = Path(record["path"])
        if _byte_sha256(path) != record["byte_sha256"]:
            raise RuntimeError(f"final task artifact changed before analysis: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("pilot_decision") != decisions["decisions"][payload["task"]][
            "decision"
        ]:
            raise RuntimeError("final task route changed after gate freezing")
        route_effect, candidate_effect, episode_differences = (
            _validate_final_task_payload(design, payload)
        )
        route_effects[payload["task"]] = route_effect
        candidate_effects[payload["task"]] = candidate_effect
        candidate_episode_differences[payload["task"]] = episode_differences
        if payload["pilot_decision"] == "accept_candidate" and candidate_effect < 0.0:
            harmful_accepted.append(payload["task"])
    settings = design["analysis"]
    result = {
        "protocol_id": "riskshiftbench-frontier-v2-primary-analysis-v1",
        **primary_analysis_from_task_effects(
            route_effects,
            candidate_effects,
            task_bootstrap_replicates=int(
                settings["task_stratified_bootstrap_replicates"]
            ),
            domain_bootstrap_replicates=int(
                settings["domain_resampling_bootstrap_replicates"]
            ),
            sign_flip_replicates=int(settings["task_level_sign_flip_replicates"]),
            seed=int(settings["random_seed"]),
        ),
        "confirmatory_fixed_family_test": fixed_family_bounded_mean_test(
            candidate_episode_differences,
            set(decisions["accepted_tasks"]),
            alpha=float(settings["confirmatory_alpha"]),
        ),
        "accepted_route_count": len(decisions["accepted_tasks"]),
        "harmful_accepted_route_count": len(harmful_accepted),
        "harmful_accepted_tasks": sorted(harmful_accepted),
        "task_route_effects": route_effects,
        "task_candidate_effects": candidate_effects,
    }
    path = output_root / "analysis" / "primary.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != result:
            raise RuntimeError("frozen primary analysis changed")
        return observed
    write_json_once(path, result)
    return result


def run_registered_final_manifest_only(design: dict) -> dict:
    path = Path(design["confirmation"]["output_root"]) / "final" / "manifest.json"
    if not path.is_file():
        raise RuntimeError("primary analysis is blocked until final evaluation completes")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("task_count") != len(all_tasks("confirmation")):
        raise RuntimeError("final manifest task coverage changed")
    return manifest


def _default_interpreters() -> dict[str, Path]:
    return {
        "gymnasium": Path("artifacts/frontier_v2_envs/gymnasium/Scripts/python.exe"),
        "or_gym": Path("artifacts/frontier_v2_envs/or-gym/Scripts/python.exe"),
        "safety_gymnasium": Path(
            "artifacts/frontier_v2_envs/safety-gymnasium/Scripts/python.exe"
        ),
        "minigrid": Path("artifacts/frontier_v2_envs/minigrid/Scripts/python.exe"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--protocol", type=Path, required=True)
    worker.add_argument("--codebase", choices=tuple(SOURCE_DIRECTORIES), required=True)
    worker.add_argument("--pilot-log", type=Path, required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--protocol", type=Path, required=True)
    final = commands.add_parser("final")
    final.add_argument("--protocol", type=Path, required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--protocol", type=Path, required=True)
    for command in (pilot, final):
        for codebase, path in _default_interpreters().items():
            command.add_argument(
                f"--{codebase.replace('_', '-')}-python",
                type=Path,
                default=path,
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "worker":
        run_worker(args.protocol, args.codebase, args.pilot_log)
        return
    if args.command == "analyze":
        print(json.dumps(analyze_registered_final(args.protocol), indent=2, sort_keys=True))
        return
    interpreters = {
        codebase: getattr(args, f"{codebase}_python")
        for codebase in SOURCE_DIRECTORIES
    }
    if args.command == "pilot":
        summary = run_registered_pilot(args.protocol, interpreters=interpreters)
    else:
        summary = run_registered_final(args.protocol, interpreters=interpreters)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
