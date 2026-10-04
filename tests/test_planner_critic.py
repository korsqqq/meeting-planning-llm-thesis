# tests/test_planner_critic.py
"""Offline tests for C4 -- one full-instance planner + one fresh-context critic.

The design lives or dies on one property: the critic must be able to improve the plan
without receiving a single task fact the planner did not buy through a tool. Most of the
tests below attack that property from one side or the other -- a person mentioned but never
queried, a travel pair never asked for, a pool that leaks the validator's verdict -- and
the rest pin the gate, the budget and the failure semantics to C3's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agents.multi_agent.worker import WorkerBudgetView
from src.agents.single_agent.best_of_3 import AttemptBudgetView
from src.agents.single_agent.planner_critic import (
    CONDITION,
    PLANNER_SHARE_DEN,
    PLANNER_SHARE_NUM,
    UNKNOWN,
    Evidence,
    PlannerBudgetView,
    build_draft,
    candidate_pool,
    collect_evidence,
    critic_messages,
    planner_quota,
    run_planner_critic,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger
from src.schemas import (
    AnswerContract,
    Condition,
    GeneratorParams,
    Instance,
    Person,
    TravelStructure,
)


def make_instance() -> Instance:
    """Four people at four locations, generous windows, symmetric travel.

    S->X is 10 for every X and X<->Y is 20, so `p0@10` then `p1@60` is feasible while
    `p0@10, p1@10` is not. Four people give the pool room to be strictly larger than a
    two-meeting fallback.
    """
    locs = ["S", "A", "B", "C", "D"]
    travel = {a: {b: (0 if a == b else 10 if "S" in (a, b) else 20) for b in locs}
              for a in locs}
    return Instance(
        instance_id="c4-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=4, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=600,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=locs,
        people=[
            Person(person_id="p0", location="A", window_start=0, window_end=300),
            Person(person_id="p1", location="B", window_start=0, window_end=400),
            Person(person_id="p2", location="C", window_start=0, window_end=500),
            Person(person_id="p3", location="D", window_start=0, window_end=550),
        ],
        travel_times=travel,
    )


# --------------------------------------------------------------------------- #
# A scripted client. Separate scripts for the planner turns and the critic call.
# --------------------------------------------------------------------------- #
@dataclass
class ScriptedClient:
    planner: list[str] = field(default_factory=list)
    critic: str = "Action: finish"
    seen: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 40
    _i: int = 0

    def count_input(self, messages, enable_thinking: bool = True) -> int:
        return self.input_tokens

    def complete(self, messages, *, max_tokens: int, enable_thinking: bool = True,
                 guided_json=None, stop=None, seed: int | None = None):
        if guided_json is not None:
            text = '{"meetings": []}'
            kind = "finalize"
        elif any("meeting-plan reviewer" in m.get("content", "") for m in messages):
            text = self.critic
            kind = "critic"
        else:
            text = self.planner[self._i] if self._i < len(self.planner) else "Action: finish"
            self._i += 1
            kind = "planner"
        self.seen.append({"kind": kind, "messages": messages, "max_tokens": max_tokens,
                          "thinking": enable_thinking, "text": text})
        thinking = min(3, max_tokens) if enable_thinking else 0
        return _Resp(answer=text, input_tokens=self.input_tokens,
                     thinking_tokens=thinking,
                     answer_tokens=min(max(1, len(text) // 4),
                                       max(0, max_tokens - thinking)),
                     requested_max_tokens=max_tokens)


@dataclass
class _Resp:
    answer: str
    input_tokens: int
    thinking_tokens: int
    answer_tokens: int
    requested_max_tokens: int
    raw_text: str = ""
    finish_reason: str = "stop"
    effective_max_tokens: int | None = None
    context_limited: bool = False


def _executable_source(path) -> str:
    """A module's source with comments and string literals removed.

    Guard tests must look at what the code DOES, not at what its prose says. Matching
    docstrings makes a test that fails on an honest explanation and passes on a silent
    change -- exactly backwards.
    """
    import tokenize
    with open(path, "rb") as fh:
        return "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))


def act(text: str) -> str:
    return f"Thought: ok\n{text}"


def critic_input(client: ScriptedClient) -> str:
    call = next(c for c in client.seen if c["kind"] == "critic")
    return "\n".join(m["content"] for m in call["messages"])


# --------------------------------------------------------------------------- #
# 1, 2. The planner sees the full instance; nothing is decomposed.
# --------------------------------------------------------------------------- #
def test_the_planner_receives_the_whole_instance():
    inst = make_instance()
    client = ScriptedClient(planner=[act("Action: list_people[]"), act("Action: finish")])
    run_planner_critic(inst, client, cap=8000, max_steps=6)
    observations = [m["content"] for c in client.seen if c["kind"] == "planner"
                    for m in c["messages"] if m["role"] == "user"]
    assert any("p0" in o and "p1" in o and "p2" in o and "p3" in o for o in observations)


def test_nothing_is_decomposed():
    """No sub-instance is built and the supervisor is never involved."""
    from pathlib import Path
    src = _executable_source(Path(__file__).resolve().parents[1] / "src" / "agents"
                             / "single_agent" / "planner_critic.py")
    for banned in ("split_instance", "split_people", "make_sub_instance", "run_worker",
                   "sub_instance", "cluster"):
        assert banned not in src, f"C4 source references {banned!r}"


# --------------------------------------------------------------------------- #
# 3, 4. Budget.
# --------------------------------------------------------------------------- #
def test_the_planner_quota_is_three_quarters_of_the_working_budget():
    assert (PLANNER_SHARE_NUM, PLANNER_SHARE_DEN) == (3, 4)
    for cap in (2000, 8000, 16000, 32000, 64000):
        w = cap - DEFAULT_FINALIZATION_RESERVE
        assert planner_quota(cap, DEFAULT_FINALIZATION_RESERVE) == w * 3 // 4


def test_at_cap_64000_the_arithmetic_matches_the_registered_design():
    cap, reserve = 64000, DEFAULT_FINALIZATION_RESERVE
    w = cap - reserve
    assert (w, planner_quota(cap, reserve), w - planner_quota(cap, reserve)) == \
        (63744, 47808, 15936)


def test_one_shared_ledger_carries_planner_critic_and_finalisation():
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"), act("Action: propose[p0@10]"),
                 act("Action: finish")],
        critic=act("Action: propose[p0@10]"),
    )
    result = run_planner_critic(inst, client, cap=8000, max_steps=8)
    booked = sum(c.input_tokens + c.thinking_tokens + c.answer_tokens
                 for c in result.calls)
    assert result.tokens.total == booked <= 8000
    assert {c.role for c in result.calls} <= {"planner", "critic", "finalize"}


def test_the_planner_cannot_spend_beyond_its_quota():
    view = PlannerBudgetView(shared=BudgetLedger(cap=8000), quota=1000)
    for _ in range(50):
        room = view.max_new_tokens(40)
        if room <= 0:
            break
        view.record_call(input_tokens=40, thinking_tokens=min(3, room),
                         answer_tokens=room - min(3, room))
    assert view.spent <= 1000


def test_all_three_quota_views_agree_arithmetically():
    """C3's worker, C5's attempt and C4's planner share one discipline stated three
    times; each belongs to a frozen condition, so equivalence is proven not assumed."""
    for cap, quota, spent, inp in [(8000, 2000, 0, 40), (8000, 2000, 1500, 40),
                                   (64000, 47808, 40000, 500)]:
        views = [PlannerBudgetView(shared=BudgetLedger(cap=cap), quota=quota, spent=spent),
                 AttemptBudgetView(shared=BudgetLedger(cap=cap), quota=quota, spent=spent),
                 WorkerBudgetView(shared=BudgetLedger(cap=cap), quota=quota, spent=spent)]
        assert len({v.max_new_tokens(inp) for v in views}) == 1
        assert len({v.should_finalize(inp) for v in views}) == 1


def test_the_planner_may_not_issue_a_finalising_call():
    view = PlannerBudgetView(shared=BudgetLedger(cap=8000), quota=1000)
    with pytest.raises(ValueError, match="cannot issue a finalising call"):
        view.max_new_tokens(40, finalizing=True)


# --------------------------------------------------------------------------- #
# 5, 6, 7. One finalisation; the critic is fresh and has no tools.
# --------------------------------------------------------------------------- #
def test_exactly_one_finalisation():
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"), act("Action: propose[p0@10]"),
                 act("Action: finish")],
        critic=act("Action: propose[p0@10]"),
    )
    result = run_planner_critic(inst, client, cap=8000, max_steps=8)
    assert sum(1 for c in client.seen if c["kind"] == "finalize") == 1
    assert [c.role for c in result.calls].count("finalize") == 1


def test_the_critic_context_is_fresh_and_carries_no_planner_transcript():
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"),
                 act("Action: propose[p0@10]"),
                 act("Action: finish")],
        critic=act("Action: propose[p0@10]"),
    )
    run_planner_critic(inst, client, cap=8000, max_steps=8)
    call = next(c for c in client.seen if c["kind"] == "critic")
    assert len(call["messages"]) == 2                      # system + user, nothing else
    assert "Observation:" not in critic_input(client)
    assert "Thought:" not in critic_input(client)


def test_the_critic_is_called_exactly_once_and_offered_no_tools():
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"), act("Action: propose[p0@10]"),
                 act("Action: finish")],
        critic=act("Action: propose[p0@10]"),
    )
    run_planner_critic(inst, client, cap=8000, max_steps=8)
    assert sum(1 for c in client.seen if c["kind"] == "critic") == 1
    text = critic_input(client)
    for tool in ("list_people", "get_availability", "get_travel_time"):
        assert tool not in text


# --------------------------------------------------------------------------- #
# 8, 9, 10, 11. The evidence rule -- the heart of design E.
# --------------------------------------------------------------------------- #
def test_a_proposal_mention_alone_reveals_nothing():
    inst = make_instance()
    transcript = [{"role": "assistant", "content": act("Action: propose[p3@10]")}]
    ev = collect_evidence(inst, transcript)
    assert ev.windows == {} and ev.travel == {} and ev.people_listed == set()


def test_only_get_availability_exposes_a_window():
    inst = make_instance()
    ev = collect_evidence(inst, [
        {"role": "assistant", "content": act("Action: get_availability[p1]")}])
    assert set(ev.windows) == {"p1"}
    assert ev.windows["p1"] == {"location": "B", "window_start": 0, "window_end": 400}


def test_list_people_exposes_ids_but_no_windows():
    inst = make_instance()
    ev = collect_evidence(inst, [
        {"role": "assistant", "content": act("Action: list_people[]")}])
    assert ev.people_listed == {"p0", "p1", "p2", "p3"}
    assert ev.windows == {}


def test_only_the_requested_travel_entries_appear_and_direction_matters():
    inst = make_instance()
    ev = collect_evidence(inst, [
        {"role": "assistant", "content": act("Action: get_travel_time[S, A]")}])
    assert ev.travel == {("S", "A"): 10}
    assert ("A", "S") not in ev.travel


def test_a_failed_tool_call_exposes_nothing():
    inst = make_instance()
    ev = collect_evidence(inst, [
        {"role": "assistant", "content": act("Action: get_availability[p99]")},
        {"role": "assistant", "content": act("Action: get_travel_time[S, ZZZ]")}])
    assert ev.windows == {} and ev.travel == {}


def test_unknown_travel_pairs_stay_unknown_in_the_critic_prompt():
    inst = make_instance()
    ev = Evidence(windows={
        "p0": {"location": "A", "window_start": 0, "window_end": 300},
        "p1": {"location": "B", "window_start": 0, "window_end": 400}},
        travel={("S", "A"): 10})
    msgs, known, unknown = critic_messages(inst, AnswerContract.empty(),
                                           {"p0", "p1"}, ev)
    text = "\n".join(m["content"] for m in msgs)
    assert "A=10" in text                      # the pair the planner did ask for
    assert f"B={UNKNOWN}" in text              # the ones it did not
    assert known == 1 and unknown == 5         # 3 locations -> 6 ordered pairs
    for value in ("=20", "=0,"):
        assert f"B{value}" not in text         # no real distance leaked for B


def test_the_critic_never_sees_a_person_outside_the_pool():
    inst = make_instance()
    ev = collect_evidence(inst, [
        {"role": "assistant", "content": act("Action: get_availability[p0]")},
        {"role": "assistant", "content": act("Action: get_availability[p2]")}])
    msgs, _, _ = critic_messages(inst, AnswerContract.empty(), {"p0"}, ev)
    text = "\n".join(m["content"] for m in msgs)
    assert "p0 @ A" in text
    assert "p2" not in text                    # queried, but not in the pool


# --------------------------------------------------------------------------- #
# 12, 13. The pool rule.
# --------------------------------------------------------------------------- #
def _row(ids, *, parsed=True, valid=True):
    return {"parsed": parsed, "valid": valid,
            "plan": [{"person_id": p, "start_time": 10} for p in ids] if parsed else None}


def test_the_pool_is_proposed_intersect_availability_exposed():
    ev = Evidence(windows={"p0": {}, "p1": {}, "p2": {}})
    pool = candidate_pool([_row(["p0", "p1"]), _row(["p3"])], ev)
    assert pool == {"p0", "p1"}                # p3 proposed but never queried
    assert "p2" not in pool                    # p2 queried but never proposed


def test_the_pool_does_not_depend_on_the_validator_verdict():
    """A pool built from valid proposals only would encode the hidden gate."""
    ev = Evidence(windows={"p0": {}, "p1": {}})
    invalid_only = candidate_pool([_row(["p0", "p1"], valid=False)], ev)
    valid_only = candidate_pool([_row(["p0", "p1"], valid=True)], ev)
    assert invalid_only == valid_only == {"p0", "p1"}


def test_an_unparseable_proposal_contributes_nothing():
    ev = Evidence(windows={"p0": {}})
    assert candidate_pool([_row([], parsed=False)], ev) == set()


# --------------------------------------------------------------------------- #
# The draft.
# --------------------------------------------------------------------------- #
def test_the_draft_takes_the_latest_time_per_person_and_orders_deterministically():
    draft = build_draft([_row(["p1"]), {"parsed": True, "valid": False,
                                        "plan": [{"person_id": "p1", "start_time": 200},
                                                 {"person_id": "p0", "start_time": 50}]}])
    assert [(m.person_id, m.start_time) for m in draft.meetings] == \
        [("p0", 50), ("p1", 200)]


def test_the_draft_is_not_repaired():
    """It may be invalid on the full instance -- that is what makes it raw material."""
    from src.oracle import is_valid
    inst = make_instance()
    draft = build_draft([{"parsed": True, "valid": False,
                          "plan": [{"person_id": "p0", "start_time": 10},
                                   {"person_id": "p1", "start_time": 10}]}])
    assert len(draft.meetings) == 2
    assert not is_valid(inst, draft)


# --------------------------------------------------------------------------- #
# 14-19. Fallback, gate and failure semantics.
# --------------------------------------------------------------------------- #
def _run(critic: str, cap: int = 20000):
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"), act("Action: get_availability[p1]"),
                 act("Action: get_travel_time[S, A]"), act("Action: get_travel_time[A, B]"),
                 act("Action: propose[p0@10]"), act("Action: propose[p1@10, p0@10]"),
                 act("Action: finish")],
        critic=critic,
    )
    return inst, client, run_planner_critic(inst, client, cap=cap, max_steps=12)


def test_the_fallback_is_the_planner_best_plan_and_survives_a_useless_critic():
    _, _, result = _run(critic="")                      # empty post-think -> no-op
    assert [m.person_id for m in result.best_plan_so_far.meetings] == ["p0"]
    assert result.diagnostics["critic_improved"] is False
    assert result.diagnostics["fallback_size"] == 1


def test_a_longer_valid_critic_plan_is_adopted():
    _, _, result = _run(critic=act("Action: propose[p0@10, p1@60]"))
    assert len(result.best_plan_so_far.meetings) == 2
    assert result.diagnostics["critic_improved"] is True


def test_an_out_of_pool_critic_plan_is_refused_even_when_feasible():
    _, _, result = _run(critic=act("Action: propose[p0@10, p2@60]"))
    assert [m.person_id for m in result.best_plan_so_far.meetings] == ["p0"]
    row = [p for p in result.proposals if p["role"] == "critic"][0]
    assert row["in_pool"] is False
    assert row["valid"] is True                          # feasible, and still refused
    assert row["accepted_into_best_plan"] is False


def test_an_invalid_critic_plan_is_rejected_silently():
    _, client, result = _run(critic=act("Action: propose[p0@10, p1@10]"))
    assert [m.person_id for m in result.best_plan_so_far.meetings] == ["p0"]
    row = [p for p in result.proposals if p["role"] == "critic"][0]
    assert row["valid"] is False and row["reasons"]
    # The verdict never reaches any model: the critic was called once and told nothing.
    assert sum(1 for c in client.seen if c["kind"] == "critic") == 1


def test_a_critic_plan_that_is_not_strictly_longer_is_refused():
    _, _, result = _run(critic=act("Action: propose[p1@10]"))
    assert [m.person_id for m in result.best_plan_so_far.meetings] == ["p0"]
    assert result.diagnostics["critic_improved"] is False


def test_an_unparseable_critic_answer_is_a_no_op():
    _, _, result = _run(critic="I think the plan looks fine, honestly.")
    assert [m.person_id for m in result.best_plan_so_far.meetings] == ["p0"]
    assert result.diagnostics["critic_reached"] is True
    assert result.diagnostics["critic_improved"] is False


def test_the_critic_is_skipped_when_its_input_does_not_fit():
    inst = make_instance()
    client = ScriptedClient(
        planner=[act("Action: get_availability[p0]"), act("Action: propose[p0@10]"),
                 act("Action: finish")],
        critic=act("Action: propose[p0@10, p1@60]"),
        input_tokens=400,
    )
    result = run_planner_critic(inst, client, cap=900, max_steps=4)
    assert result.diagnostics["critic_reached"] is False
    assert sum(1 for c in client.seen if c["kind"] == "critic") == 0


def test_an_empty_pool_skips_the_critic():
    inst = make_instance()
    client = ScriptedClient(planner=[act("Action: finish")],
                            critic=act("Action: propose[p0@10]"))
    result = run_planner_critic(inst, client, cap=8000, max_steps=4)
    assert result.diagnostics["pool_size"] == 0
    assert result.diagnostics["critic_reached"] is False


def test_the_critic_gets_no_retry():
    _, client, _ = _run(critic="nonsense")
    assert sum(1 for c in client.seen if c["kind"] == "critic") == 1


# --------------------------------------------------------------------------- #
# 20. No oracle information anywhere near the agents.
# --------------------------------------------------------------------------- #
def test_no_oracle_vocabulary_reaches_any_message():
    inst = make_instance()
    _, client, result = _run(critic=act("Action: propose[p0@10, p1@60]"))
    everything = "\n".join(
        m["content"] for c in client.seen for m in c["messages"]
    ) + "\n".join(m["content"] for m in result.transcript)
    for banned in ("optimum", "CP-SAT", "cp_sat", "solver", "oracle", "satisfaction",
                   "valid plan", "validator"):
        assert banned.lower() not in everything.lower(), f"leak: {banned!r}"


def test_the_condition_never_imports_the_solver():
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "src" / "agents" / "single_agent"
            / "planner_critic.py")
    code = _executable_source(path)
    for banned in ("solve", "optimum", "score_plan", "OracleSolution", "brute_force"):
        assert banned not in code, f"C4 source references {banned!r}"


# --------------------------------------------------------------------------- #
# Diagnostics.
# --------------------------------------------------------------------------- #
def test_the_diagnostics_describe_the_run_without_shaping_it():
    _, _, result = _run(critic=act("Action: propose[p0@10, p1@60]"))
    d = result.diagnostics
    assert d["pool_size"] == 2 and d["fallback_size"] == 1 and d["headroom"] == 1
    assert d["critic_reached"] is True and d["critic_improved"] is True
    assert d["evidence_people_covered"] == 2
    assert d["known_pairs_shown"] == 2 and d["unknown_pairs_shown"] == 4
    assert d["evidence_travel_coverage"] == round(2 / 6, 4)
    assert d["planner_quota"] == planner_quota(20000, DEFAULT_FINALIZATION_RESERVE)
    assert 0 < d["planner_spent"] <= d["planner_quota"]
    assert d["pool"] == ["p0", "p1"]


def test_proposal_rows_carry_their_role():
    _, _, result = _run(critic=act("Action: propose[p0@10, p1@60]"))
    roles = {p["role"] for p in result.proposals}
    assert roles == {"planner", "critic"}
    assert all(p["condition"] == CONDITION for p in result.proposals)


# --------------------------------------------------------------------------- #
# 21. The other conditions are untouched.
# --------------------------------------------------------------------------- #
def test_every_condition_is_dispatchable_and_c4_is_no_longer_refused():
    from src.harness.runner import SUPPORTED_CONDITIONS, _resolve_condition
    assert set(SUPPORTED_CONDITIONS) == set(Condition)
    assert _resolve_condition("c4_planner_critic") is Condition.C4_PLANNER_CRITIC


def test_react_core_knows_nothing_about_c4():
    from pathlib import Path
    src = _executable_source(Path(__file__).resolve().parents[1] / "src" / "agents"
                             / "react_core.py")
    for banned in ("planner_critic", "c4_", "Evidence", "candidate_pool"):
        assert banned not in src, f"react_core mentions {banned!r}"


def test_the_other_runners_are_unchanged():
    from src.agents.multi_agent.hierarchical import run_hierarchical
    from src.agents.single_agent.best_of_3 import run_best_of_3
    from src.agents.single_agent.react import run_react
    from src.agents.single_agent.react_verify_revise import run_verify_revise
    from src.harness.runner import SUPPORTED_CONDITIONS
    assert SUPPORTED_CONDITIONS[Condition.C1_REACT] is run_react
    assert SUPPORTED_CONDITIONS[Condition.C2_VERIFY_REVISE] is run_verify_revise
    assert SUPPORTED_CONDITIONS[Condition.C3_MAS] is run_hierarchical
    assert SUPPORTED_CONDITIONS[Condition.C5_BEST_OF_3] is run_best_of_3
