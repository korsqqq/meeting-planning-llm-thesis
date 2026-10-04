# tests/test_stage_extraction.py
"""Offline tests for the stage and proposal extraction.

Every test here corresponds to a defect the smoke test against the development
documents actually surfaced, or to a rule the extraction must not quietly relax:
the C3 tie resolving to worker A, the C5 winner coming from the recorded index, the
C2 cycle assignment, and the discipline that an unknown is empty and never zero.

No held-out document is read. Every fixture is synthetic.
"""

from __future__ import annotations

from scripts.extract_stage_metrics import (
    UNKNOWN,
    c3_reconstruction,
    call_stages,
    cost_of,
    extract_run,
    proposal_outcome,
    run_level_plan,
    worker_final_plans,
)

INFO = {"block": "A", "band": "high", "n_people": 8, "optimum": 4,
        "complexity_metric": 12}


def plan(*people):
    return [{"person_id": p, "start_time": 100 + i} for i, p in enumerate(people)]


def proposal(role, step, people, *, valid=True, accepted=True, in_pool=None,
             parsed=True, attempt_index=None):
    p = {"role": role, "step": step, "parsed": parsed, "plan": plan(*people),
         "n_meetings": len(people), "valid": valid,
         "accepted_into_best_plan": accepted, "reasons": [] if valid else ["overlap"]}
    if in_pool is not None:
        p["in_pool"] = in_pool
    if attempt_index is not None:
        p["attempt_index"] = attempt_index
    return p


def call(role, **kw):
    base = {"role": role, "input_tokens": 10, "thinking_tokens": 20,
            "answer_tokens": 5, "latency_seconds": 1.0, "finish_reason": "stop",
            "context_limited": False}
    base.update(kw)
    return base


def document(condition, calls, proposals, final_people, *, diagnostics=None,
             total=None):
    n = len(calls)
    return {
        "schema_version": "harness_slice/1.0",
        "condition": condition,
        "agent": {"termination": "agent_finish", "proposals": proposals,
                  "diagnostics": diagnostics or {}},
        "run_result": {
            "run_id": f"{condition}__i__cap64000__M", "instance_id": "i",
            "condition": condition, "cap": 64000, "model": "M",
            "score": {"satisfaction": 0.5, "valid": True, "solver_optimum": 4},
            "tokens": {"total": total if total is not None else n * 35},
            "calls": calls, "final_plan": {"meetings": plan(*final_people)},
        },
    }


# --------------------------------------------------------------------------- #
# C3: the aggregator rule, copied from hierarchical.aggregate_node.
# --------------------------------------------------------------------------- #
def test_a_tie_between_workers_resolves_to_worker_a():
    """The defect the development documents exposed: 14 runs disagreed on ties."""
    proposals = [proposal("worker_a", 3, ["p1", "p2"]),
                 proposal("worker_b", 4, ["p7", "p8"])]
    recon = c3_reconstruction({}, proposals)
    assert recon["fallback_from"] == "worker_a"
    assert recon["fallback_size"] == 2
    assert [m["person_id"] for m in recon["fallback_plan"]] == ["p1", "p2"]


def test_the_longer_worker_wins_when_there_is_no_tie():
    proposals = [proposal("worker_a", 3, ["p1"]),
                 proposal("worker_b", 4, ["p7", "p8"])]
    assert c3_reconstruction({}, proposals)["fallback_from"] == "worker_b"


def test_the_candidate_pool_is_the_union_of_the_final_worker_plans():
    """Union of person_id over the accepted plans -- no prompt text is parsed."""
    proposals = [
        proposal("worker_a", 2, ["p1"]),                       # superseded
        proposal("worker_a", 5, ["p1", "p2"]),                 # final for A
        proposal("worker_b", 6, ["p2", "p9"]),                 # final for B
    ]
    recon = c3_reconstruction({}, proposals)
    assert recon["pool_size"] == 3                              # p1, p2, p9
    assert recon["worker_a_meetings"] == 2 and recon["worker_b_meetings"] == 2


def test_a_worker_that_never_had_a_proposal_accepted_ends_empty():
    proposals = [proposal("worker_a", 2, ["p1"], valid=False, accepted=False)]
    recon = c3_reconstruction({}, proposals)
    assert recon["worker_a_meetings"] == 0 and recon["fallback_size"] == 0
    assert worker_final_plans(proposals)["worker_b"] == []


def test_the_c3_critic_gain_is_measured_against_the_fallback():
    calls = [call("worker_a"), call("worker_b"), call("critic"), call("finalize")]
    proposals = [proposal("worker_a", 3, ["p1", "p2"]),
                 proposal("worker_b", 4, ["p3"]),
                 proposal("critic", 0, ["p1", "p2", "p3"], in_pool=True)]
    stages, _, checks = extract_run(
        document("c3_mas", calls, proposals, ["p1", "p2", "p3"]), INFO)
    agg = next(s for s in stages if s["stage_role"] == "aggregate")
    critic = next(s for s in stages if s["stage_role"] == "critic")
    assert agg["best_meetings_before"] == 0 and agg["best_meetings_after"] == 2
    assert critic["best_meetings_before"] == 2      # the fallback, not the workers
    assert critic["best_meetings_after"] == 3 and critic["meetings_added"] == 1
    assert checks["final_plan_matches"]


def test_worker_quality_is_labelled_as_sub_instance():
    """A worker's best is its own sub-instance best, not the run's."""
    calls = [call("worker_a"), call("worker_b"), call("finalize")]
    proposals = [proposal("worker_a", 3, ["p1", "p2"]),
                 proposal("worker_b", 4, ["p3"])]
    stages, _, _ = extract_run(document("c3_mas", calls, proposals, ["p1", "p2"]), INFO)
    sources = {s["stage_role"]: s["quality_source"] for s in stages}
    assert sources["worker_a"] == "reconstructed_subinstance"
    assert sources["worker_b"] == "reconstructed_subinstance"
    assert sources["aggregate"] == "reconstructed"


# --------------------------------------------------------------------------- #
# C5: the winner is recorded, not inferred.
# --------------------------------------------------------------------------- #
def test_the_c5_winner_comes_from_the_recorded_index():
    """The second defect: the last accepted proposal is not the selected product."""
    diagnostics = {
        "winner_attempt_index": 0, "n_distinct_products": 2,
        "attempt_products": [
            {"attempt_index": 0, "plan": plan("p1", "p2"), "n_meetings": 2},
            {"attempt_index": 1, "plan": plan("p8"), "n_meetings": 1},
            {"attempt_index": 2, "plan": [], "n_meetings": 0},
        ],
    }
    proposals = [proposal("bon_attempt_0", 1, ["p1", "p2"], attempt_index=0),
                 proposal("bon_attempt_1", 2, ["p8"], attempt_index=1)]
    assert run_level_plan("c5_best_of_3", proposals, diagnostics, {}) \
        == diagnostics["attempt_products"][0]["plan"]

    calls = [call("bon_attempt_0"), call("bon_attempt_1"), call("bon_attempt_2"),
             call("finalize")]
    stages, _, checks = extract_run(
        document("c5_best_of_3", calls, proposals, ["p1", "p2"],
                 diagnostics=diagnostics), INFO)
    per_attempt = {s["stage_role"]: s for s in stages
                   if s["stage_role"].startswith("bon_attempt_")}
    assert per_attempt["bon_attempt_2"]["best_meetings_after"] == 0
    assert per_attempt["bon_attempt_2"]["quality_source"] == "direct"
    assert per_attempt["bon_attempt_0"]["stage_outcome"] == "selected_winner"
    assert checks["final_plan_matches"]


# --------------------------------------------------------------------------- #
# C2: cycles, and the refusal to guess.
# --------------------------------------------------------------------------- #
def test_one_proposal_per_revise_cycle_is_assigned_not_blanked():
    calls = [call("react_step"), call("verify"), call("revise"),
             call("verify"), call("revise"), call("finalize")]
    proposals = [proposal("react_step", 2, ["p1"]),
                 proposal("revise", 6, ["p1", "p2"]),
                 proposal("revise", 9, ["p1", "p2", "p3"])]
    stages, _, checks = extract_run(
        document("c2_verify_revise", calls, proposals, ["p1", "p2", "p3"]), INFO)
    revises = [s for s in stages if s["stage_role"] == "revise"]
    assert [s["cycle_index"] for s in revises] == [0, 1]
    assert [s["n_proposals"] for s in revises] == [1, 1]
    assert all(s["quality_source"] == "reconstructed" for s in revises)
    assert checks["ambiguous_stage_quality"] == 0


def test_an_unresolvable_cycle_assignment_is_unknown_and_not_zero():
    """Two cycles, three proposals: the mapping is not determined, so nothing is claimed.

    The cycles are separated by the `verify` call between them -- consecutive calls of
    one role are a single cycle, which is what `call_stages` segments on.
    """
    calls = [call("react_step"), call("verify"), call("revise"),
             call("verify"), call("revise"), call("finalize")]
    proposals = [proposal("react_step", 2, ["p1"]),
                 proposal("revise", 4, ["p1", "p2"]),
                 proposal("revise", 5, ["p1", "p2"]),
                 proposal("revise", 6, ["p1", "p2", "p3"])]
    stages, _, checks = extract_run(
        document("c2_verify_revise", calls, proposals, ["p1", "p2", "p3"]), INFO)
    revises = [s for s in stages if s["stage_role"] == "revise"]
    assert all(s["quality_source"] == "unknown" for s in revises)
    for s in revises:
        assert s["best_meetings_after"] == UNKNOWN != 0
        assert s["sat_after"] == UNKNOWN
    assert checks["ambiguous_stage_quality"] == len(revises)


def test_a_non_proposing_stage_is_not_applicable_rather_than_zero():
    calls = [call("react_step"), call("verify"), call("finalize")]
    stages, _, _ = extract_run(
        document("c2_verify_revise", calls, [proposal("react_step", 2, ["p1"])],
                 ["p1"]), INFO)
    for role in ("verify", "finalize"):
        row = next(s for s in stages if s["stage_role"] == role)
        assert row["quality_source"] == "not_applicable"
        assert row["best_meetings_after"] == UNKNOWN


# --------------------------------------------------------------------------- #
# The proposal taxonomy.
# --------------------------------------------------------------------------- #
def test_the_five_proposal_outcomes_are_distinguished():
    assert proposal_outcome({"parsed": False}) == "parse_failed"
    assert proposal_outcome({"parsed": True, "valid": False}) == "invalid_constraints"
    assert proposal_outcome({"parsed": True, "valid": True, "in_pool": False,
                             "accepted_into_best_plan": False}) == "out_of_pool"
    assert proposal_outcome({"parsed": True, "valid": True,
                             "accepted_into_best_plan": False}) == "valid_not_longer"
    assert proposal_outcome({"parsed": True, "valid": True,
                             "accepted_into_best_plan": True}) == "accepted"


def test_out_of_pool_cannot_arise_where_the_gate_never_recorded_it():
    """Only the C3 and C4 critics carry `in_pool`; elsewhere the label is impossible."""
    assert proposal_outcome({"parsed": True, "valid": True,
                             "accepted_into_best_plan": False}) != "out_of_pool"


# --------------------------------------------------------------------------- #
# Direct diagnostics, and checking the reconstruction against them.
# --------------------------------------------------------------------------- #
C4_CALLS = [call("planner"), call("critic"), call("finalize")]


def c4_document(*, critic_improved, critic_people, planner_people=("p1",)):
    diagnostics = {"critic_reached": True, "critic_improved": critic_improved,
                   "pool_size": 3, "fallback_size": 1, "headroom": 500,
                   "critic_room": 400, "planner_quota": 4000, "planner_spent": 3900,
                   "evidence_people_covered": 3, "evidence_travel_coverage": 0.25,
                   "known_pairs_shown": 3, "unknown_pairs_shown": 9}
    proposals = [proposal("planner", 3, list(planner_people))]
    if critic_people is not None:
        proposals.append(proposal("critic", 0, list(critic_people), in_pool=True,
                                  accepted=len(critic_people) > len(planner_people)))
    final = critic_people if (critic_people and
                              len(critic_people) > len(planner_people)) else planner_people
    return document("c4_planner_critic", C4_CALLS, proposals, list(final),
                    diagnostics=diagnostics)


def test_every_recorded_diagnostics_key_reaches_the_row():
    """The omission that made the C4 critic claim uncheckable must not recur."""
    stages, _, _ = extract_run(
        c4_document(critic_improved=True, critic_people=["p1", "p2"]), INFO)
    critic = next(s for s in stages if s["stage_role"] == "critic")
    assert critic["critic_reached"] == 1 and critic["critic_improved"] == 1
    assert critic["pool_size"] == 3 and critic["evidence_travel_coverage"] == 0.25
    assert critic["information_source"] == "direct"


def test_booleans_are_written_as_zero_or_one():
    stages, _, _ = extract_run(
        c4_document(critic_improved=False, critic_people=["p9"]), INFO)
    critic = next(s for s in stages if s["stage_role"] == "critic")
    assert critic["critic_improved"] == 0 and critic["critic_improved"] is not False


def test_the_rebuilt_improvement_is_compared_with_the_recorded_one():
    _, _, agree = extract_run(
        c4_document(critic_improved=True, critic_people=["p1", "p2"]), INFO)
    assert agree["critic_improved_checkable"] and agree["critic_improved_agrees"]

    # A recorded improvement the reconstruction does not see must be reported as a
    # disagreement, not smoothed over.
    _, _, clash = extract_run(
        c4_document(critic_improved=True, critic_people=["p9"]), INFO)
    assert clash["critic_improved_checkable"] and not clash["critic_improved_agrees"]


def test_a_condition_without_the_field_reports_not_checkable():
    calls = [call("worker_a"), call("worker_b"), call("critic"), call("finalize")]
    proposals = [proposal("worker_a", 3, ["p1"]), proposal("worker_b", 4, ["p2"]),
                 proposal("critic", 0, ["p1", "p2"], in_pool=True)]
    stages, _, checks = extract_run(
        document("c3_mas", calls, proposals, ["p1", "p2"]), INFO)
    assert checks["critic_improved_checkable"] is False
    assert checks["critic_improved_agrees"] == UNKNOWN
    critic = next(s for s in stages if s["stage_role"] == "critic")
    assert critic["critic_improved"] == UNKNOWN      # absent, not 0


def test_the_c5_selection_fields_reach_the_row():
    diagnostics = {"winner_attempt_index": 2, "n_distinct_products": 3,
                   "attempt_products": [
                       {"attempt_index": 0, "plan": plan("p1"), "n_meetings": 1},
                       {"attempt_index": 1, "plan": [], "n_meetings": 0},
                       {"attempt_index": 2, "plan": plan("p1", "p2"), "n_meetings": 2}]}
    calls = [call("bon_attempt_0"), call("bon_attempt_1"), call("bon_attempt_2"),
             call("finalize")]
    stages, _, _ = extract_run(
        document("c5_best_of_3", calls, [proposal("bon_attempt_2", 1, ["p1", "p2"],
                                                  attempt_index=2)],
                 ["p1", "p2"], diagnostics=diagnostics), INFO)
    row = next(s for s in stages if s["stage_role"] == "bon_attempt_2")
    assert row["winner_attempt_index"] == 2 and row["n_distinct_products"] == 3


# --------------------------------------------------------------------------- #
# Cost and segmentation.
# --------------------------------------------------------------------------- #
def test_consecutive_same_role_calls_form_one_cycle():
    stages = call_stages([call("react_step"), call("react_step"), call("verify"),
                          call("react_step")])
    assert [(s["stage_role"], s["cycle_index"]) for s in stages] == [
        ("react_step", 0), ("verify", 0), ("react_step", 1)]


def test_stage_cost_sums_the_calls_of_that_stage_only():
    c = cost_of([call("planner", input_tokens=100, thinking_tokens=200,
                      answer_tokens=7, latency_seconds=2.5, context_limited=True),
                 call("planner")])
    assert c["n_calls"] == 2 and c["stage_tokens"] == 100 + 200 + 7 + 35
    assert c["n_context_limited"] == 1 and c["latency_seconds"] == 3.5


def test_token_and_call_totals_reconcile_with_the_document():
    calls = [call("planner"), call("planner"), call("critic"), call("finalize")]
    doc = document("c4_planner_critic", calls,
                   [proposal("planner", 3, ["p1"])], ["p1"], total=4 * 35)
    stages, _, checks = extract_run(doc, INFO)
    assert checks["tokens_match"] and checks["calls_match"]
    assert sum(s["n_calls"] for s in stages) == 4
