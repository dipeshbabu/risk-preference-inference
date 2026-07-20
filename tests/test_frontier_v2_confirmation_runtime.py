from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.frontier_v2_confirmation_runtime import (
    append_pilot_record,
    authenticate_worker_request,
    build_pilot_record,
    fixed_family_bounded_mean_test,
    load_pilot_records,
    next_pilot_request,
    replay_pilot_records,
    primary_analysis_from_task_effects,
)
from experiments.frontier_v2_external_design import (
    DOMAIN_SPECS,
    all_tasks,
    canonical_sha256,
    outcome_implementation_sha256,
    task_manifest_sha256,
    task_sha256,
)


def _router_lock(*, budget: int = 4) -> dict:
    tasks = all_tasks("confirmation")
    proposals = [
        {
            "task": task.name,
            "domain": task.domain,
            "task_sha256": task_sha256(task),
            "candidate_policy": "candidate",
            "fallback_policy": "fallback",
            "planning_effect_gap": 0.2,
            "task_weight": 1.0 / len(tasks),
        }
        for task in tasks
    ]
    payload = {
        "protocol_id": "riskshiftbench-frontier-v2-router-lock-v1",
        "outcome_implementation_sha256": outcome_implementation_sha256(),
        "confirmation_task_manifest_sha256": task_manifest_sha256(tasks),
        "proposal_family_size": len(tasks),
        "anytime_plan": {
            "familywise_alpha": 0.05,
            "futility_familywise_alpha": 0.05,
            "effect_margin": 0.0,
            "observation_bounds": [-1.0, 1.0],
            "minimum_observations": 1,
            "maximum_observations_per_task": 3,
            "global_observation_budget": budget,
            "forced_initial_observations": 1,
            "e_process_method": "betting_mixture",
            "betting_fraction_grid": [0.2, 0.4],
            "allocation": "certified",
            "resolution_familywise_beta": 0.05,
        },
        "proposals": proposals,
    }
    payload["router_lock_canonical_sha256"] = canonical_sha256(payload)
    return payload


def _outcome(request: dict, policy: str, score: float) -> dict:
    return {
        "task": request["task"],
        "domain": request["domain"],
        "policy": policy,
        "seed": request["expected_seed"],
        "score": score,
    }


def _append(lock: dict, records: list[dict], difference: float = 0.1) -> None:
    request = next_pilot_request(lock, records)
    assert request is not None
    fallback_score = 0.4
    records.append(
        build_pilot_record(
            lock,
            sequence_index=request["sequence_index"],
            within_task_index=request["within_task_index"],
            task_name=request["task"],
            candidate_outcome=_outcome(
                request, request["candidate_policy"], fallback_score + difference
            ),
            fallback_outcome=_outcome(
                request, request["fallback_policy"], fallback_score
            ),
            previous_record_sha256=request["previous_record_sha256"],
        )
    )


def test_pilot_state_machine_replays_registered_order_and_budget() -> None:
    lock = _router_lock(budget=3)
    records: list[dict] = []
    for _ in range(3):
        _append(lock, records)
    router = replay_pilot_records(lock, records)
    assert router.total_observations() == 3
    assert next_pilot_request(lock, records) is None


def test_pilot_hash_chain_rejects_edited_outcome() -> None:
    lock = _router_lock()
    records: list[dict] = []
    _append(lock, records)
    edited = deepcopy(records)
    edited[0]["candidate_outcome"]["score"] = 0.9
    with pytest.raises(RuntimeError, match="content hash"):
        replay_pilot_records(lock, edited)


def test_pilot_replay_rejects_nonregistered_adaptive_order() -> None:
    lock = _router_lock()
    records: list[dict] = []
    _append(lock, records)
    edited = deepcopy(records)
    edited[0]["task"] = all_tasks("confirmation")[-1].name
    edited[0]["record_sha256"] = canonical_sha256(
        {key: value for key, value in edited[0].items() if key != "record_sha256"}
    )
    with pytest.raises(RuntimeError, match="allocation order"):
        replay_pilot_records(lock, edited)


def test_router_lock_hash_is_required_before_pilot_replay() -> None:
    lock = _router_lock()
    lock["anytime_plan"]["global_observation_budget"] += 1
    with pytest.raises(RuntimeError, match="canonical hash"):
        replay_pilot_records(lock, [])


def test_pilot_jsonl_round_trip_preserves_hash_chain(tmp_path) -> None:
    lock = _router_lock()
    records: list[dict] = []
    _append(lock, records)
    path = tmp_path / "pilot.jsonl"
    append_pilot_record(path, records[0])
    loaded = load_pilot_records(path)
    assert loaded == records
    assert replay_pilot_records(lock, loaded).total_observations() == 1


def test_worker_only_accepts_unique_next_authenticated_step(tmp_path) -> None:
    lock = _router_lock()
    request = next_pilot_request(lock, [])
    assert request is not None
    authenticate_worker_request(lock, tmp_path / "missing.jsonl", request)
    request["within_task_index"] += 1
    with pytest.raises(RuntimeError, match="unique next"):
        authenticate_worker_request(lock, tmp_path / "missing.jsonl", request)


def test_primary_analysis_preserves_equal_domain_weighting() -> None:
    tasks = all_tasks("confirmation")
    route_effects = {task.name: 0.1 for task in tasks}
    candidate_effects = {task.name: 0.2 for task in tasks}
    result = primary_analysis_from_task_effects(
        route_effects,
        candidate_effects,
        task_bootstrap_replicates=20,
        domain_bootstrap_replicates=30,
        sign_flip_replicates=20,
        seed=7,
    )
    assert result["equal_domain_mean_route_effect"] == pytest.approx(0.1)
    assert result["equal_domain_mean_candidate_everywhere_effect"] == pytest.approx(0.2)
    assert set(result["leave_one_domain_out_route_effects"]) == set(DOMAIN_SPECS)
    assert result["task_bootstrap_replicates"] == 20
    assert result["domain_bootstrap_replicates"] == 30


def test_fixed_family_bounded_test_uses_final_episode_units() -> None:
    tasks = all_tasks("confirmation")
    differences = {task.name: [1.0] * 100 for task in tasks}
    result = fixed_family_bounded_mean_test(
        differences,
        {task.name for task in tasks},
    )
    assert result["observed_equal_domain_mean_route_effect"] == pytest.approx(1.0)
    assert result["accepted_episode_count"] == 3_600
    assert result["weighted_range_square_sum"] == pytest.approx(4.0 / 3_600)
    assert result["one_sided_lower_confidence_bound"] > 0.95
    assert result["reject_null"] is True


def test_fixed_family_bounded_test_conditions_on_frozen_fallback_routes() -> None:
    tasks = all_tasks("confirmation")
    differences = {task.name: [1.0] * 5 for task in tasks}
    result = fixed_family_bounded_mean_test(differences, set())
    assert result["observed_equal_domain_mean_route_effect"] == 0.0
    assert result["accepted_episode_count"] == 0
    assert result["one_sided_p"] == 1.0
    assert result["one_sided_lower_confidence_bound"] == 0.0
    assert result["reject_null"] is False


def test_fixed_family_bounded_test_rejects_out_of_range_differences() -> None:
    tasks = all_tasks("confirmation")
    differences = {task.name: [0.0] for task in tasks}
    differences[tasks[0].name] = [1.01]
    with pytest.raises(RuntimeError, match="outside"):
        fixed_family_bounded_mean_test(
            differences,
            {tasks[0].name},
        )
