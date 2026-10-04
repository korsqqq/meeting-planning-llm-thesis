# tests/test_hierarchical.py
"""Offline tests for C3 (hierarchical MAS): full LangGraph wiring with a scripted
fake client, endpoint-free.

Checks the locked C3 behaviour (THESIS_DECISIONS section 4, C3-1..C3-7):
  * fixed supervisor -> worker A -> worker B -> aggregate -> critic -> finalize flow;
  * workers use the shared C1 template over their sub-instances, in isolated
    conversations (no cross-cluster data);
  * budget: the shared cap is never exceeded, quotas are symmetric (worker A
    cannot starve worker B), quota exhaustion is not global exhaustion;
  * empty outputs: an empty worker ends only its own loop, both-empty skips the
    critic, an empty/unparseable critic is a no-op -- the merged plan survives;
  * the critic proposal passes the same hidden gate (validator + strictly longer);
  * finalisation is the shared react_core node (identity), same terminal emit;
  * the runner dispatches c3_mas end-to-end with a clean transcript sanity block.
"""

from __future__ import annotations

from typing import Any

from src.agents import react_core
from src.agents.multi_agent import hierarchical
from src.agents.multi_agent.hierarchical import (
    WORKER_SHARE_DEN,
    WORKER_SHARE_NUM,
    run_hierarchical,
)
from src.agents.multi_agent.supervisor import split_instance
from src.agents.multi_agent.worker import WorkerBudgetView
from src.core import BudgetLedger, LLMResponse
from src.harness.runner import run_single_instance
from src.harness.sanity import transcript_sanity
from src.schemas import (
    Condition,
    GeneratorParams,
    Instance,
    Person,
    TravelStructure,
)


# --------------------------------------------------------------------------- #
# Instance and client helpers.
# --------------------------------------------------------------------------- #
def _travel(locs: list[str], pairs: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    t = {a: {b: 0 for b in locs} for a in locs}
    for (a, b), d in pairs.items():
        t[a][b] = d
        t[b][a] = d
    return t


def _instance() -> Instance:
    """Two geographic clusters: {p0, p1} near the depot, {p2, p3} far, all four
    meetable in one day (merge keeps everything)."""
    locs = ["S", "A1", "A2", "B1", "B2"]
    return Instance(
        instance_id="c3-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=4, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.CLUSTERED,
        ),
        start_location="S",
        start_time=0,
        end_of_day=480,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=locs,
        people=[
            Person(person_id="p0", location="A1", window_start=0, window_end=200),
            Person(person_id="p1", location="A2", window_start=0, window_end=200),
            Person(person_id="p2", location="B1", window_start=200, window_end=480),
            Person(person_id="p3", location="B2", window_start=200, window_end=480),
        ],
        travel_times=_travel(locs, {
            ("S", "A1"): 10, ("S", "A2"): 10, ("A1", "A2"): 10,
            ("S", "B1"): 60, ("S", "B2"): 60, ("B1", "B2"): 10,
            ("A1", "B1"): 60, ("A1", "B2"): 60, ("A2", "B1"): 60, ("A2", "B2"): 60,
        }),
    )


class _FakeClient:
    """Scripted completions in order; flat token counts; records every prompt.

    `answer_tokens` is capped by the granted `max_tokens`, so the budget-guard
    arithmetic (quotas included) is exercised realistically.
    """

    def __init__(self, answers: list[str], *, input_tokens: int = 50, answer_tokens: int = 5):
        self._answers = answers
        self.i = 0
        self._input = input_tokens
        self._answer = answer_tokens
        self.seen: list[list[dict[str, str]]] = []

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        return self._input

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        self.seen.append([dict(m) for m in messages])
        text = self._answers[self.i]
        self.i += 1
        return LLMResponse(
            thinking="", answer=text, raw_text=text,
            input_tokens=self._input, thinking_tokens=0,
            answer_tokens=min(self._answer, max_tokens),
            finish_reason="stop",
        )


_A_PLAN = "Action: propose[p0@10, p1@50]"
_B_PLAN = "Action: propose[p2@200, p3@240]"
_FULL_JSON = (
    '{"meetings": [{"person_id": "p0", "start_time": 10}, '
    '{"person_id": "p1", "start_time": 50}, '
    '{"person_id": "p2", "start_time": 200}, '
    '{"person_id": "p3", "start_time": 240}]}'
)


def _full_pipeline_client(critic_answer: str) -> _FakeClient:
    return _FakeClient([
        _A_PLAN, "Action: finish",       # worker A
        _B_PLAN, "Action: finish",       # worker B
        critic_answer,                   # critic
        _FULL_JSON,                      # finalize
    ])


# --------------------------------------------------------------------------- #
# Full wiring.
# --------------------------------------------------------------------------- #
def test_full_pipeline_roles_plans_and_budget() -> None:
    # The 4-meeting combination exists only if the CRITIC proposes it: the
    # aggregator supplies the fallback (best single sub-plan, 2 meetings) and a
    # raw draft; coordination gains must come from the critic's counted call.
    client = _full_pipeline_client("Action: propose[p0@10, p1@50, p2@200, p3@240]")
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert client.i == 6  # exactly the scripted calls, nothing re-run
    roles = [c.role for c in result.calls]
    assert roles == ["worker_a", "worker_a", "worker_b", "worker_b", "critic", "finalize"]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p0", 10), ("p1", 50), ("p2", 200), ("p3", 240)
    ]
    assert result.final_plan is not None
    assert [(m.person_id, m.start_time) for m in result.final_plan.meetings] == [
        ("p0", 10), ("p1", 50), ("p2", 200), ("p3", 240)
    ]
    assert result.n_steps == 2 + 2 + 1
    assert result.tokens.total <= 8000
    assert not result.tokens.budget_exhausted


def test_workers_use_shared_c1_template_in_isolated_conversations() -> None:
    inst = _instance()
    client = _FakeClient([
        "Action: list_people[]",           # A: sees only its own cluster
        "Action: get_availability[p3]",    # A: cross-cluster lookup must fail
        _A_PLAN, "Action: finish",         # A: normal end
        _B_PLAN, "Action: finish",         # B
        "keep it",                         # critic, no-op
        _FULL_JSON,
    ])
    run_hierarchical(inst, client, cap=8000)  # type: ignore[arg-type]

    sub_a, sub_b = split_instance(inst)
    assert sub_a is not None and sub_b is not None
    # First prompt of each worker == the shared react_core template over ITS
    # sub-instance: the identical template function, by construction.
    assert client.seen[0] == react_core.initial_state(sub_a)["messages"]
    assert client.seen[4] == react_core.initial_state(sub_b)["messages"]

    # Tool boundary: list_people inside worker A returns ONLY cluster A, and a
    # cross-cluster availability lookup yields an error, never p3's window data.
    a_last_prompt = " ".join(m["content"] for m in client.seen[3])
    assert "People: p0, p1" in a_last_prompt
    assert "People: p0, p1, p2" not in a_last_prompt
    p3_obs = [m["content"] for m in client.seen[3]
              if m["role"] == "user" and "p3" in m["content"] and "Observation" in m["content"]]
    assert p3_obs and "Error" in p3_obs[-1] and "window" not in p3_obs[-1]

    # Location isolation (location names never appear in the shared template):
    # worker A never sees B1/B2, worker B never sees A1/A2.
    a_text = " ".join(m["content"] for m in client.seen[3])
    b_text = " ".join(m["content"] for m in client.seen[5])
    assert "B1" not in a_text and "B2" not in a_text
    assert "A1" not in b_text and "A2" not in b_text

    # The critic, by contrast, sees the full instance dump.
    critic_text = " ".join(m["content"] for m in client.seen[6])
    for token in ("p0", "p1", "p2", "p3", "A1", "A2", "B1", "B2"):
        assert token in critic_text


def test_quota_symmetry_worker_a_cannot_starve_worker_b() -> None:
    # Worker A never finishes: it burns tool calls until its quota is gone.
    # Worker B must still run under its own untouched quota.
    cap = 2000
    client = _FakeClient(
        ["Action: list_people[]"] * 8      # A: loops until the quota guard stops it
        + [_B_PLAN, "Action: finish",      # B: normal
           "unparseable critique",         # critic: no-op
           '{"meetings": [{"person_id": "p2", "start_time": 200}, '
           '{"person_id": "p3", "start_time": 240}]}'],
        answer_tokens=100,
    )
    result = run_hierarchical(_instance(), client, cap=cap)  # type: ignore[arg-type]

    roles = [c.role for c in result.calls]
    assert "worker_b" in roles and "critic" in roles and "finalize" in roles

    quota = (cap - 256) * WORKER_SHARE_NUM // WORKER_SHARE_DEN
    spent_a = sum(
        c.input_tokens + c.thinking_tokens + c.answer_tokens
        for c in result.calls if c.role == "worker_a"
    )
    assert 0 < spent_a <= quota                # the quota bound held exactly
    assert result.tokens.total <= cap          # the shared cap held
    # Quota exhaustion is an internal allocation boundary, not global exhaustion.
    assert not result.tokens.budget_exhausted
    # B's plan survived A's starvation: the fallback is B's sub-plan.
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p2", 200), ("p3", 240)
    ]


def test_empty_worker_a_ends_only_its_loop() -> None:
    client = _FakeClient([
        "",                                # worker A: empty post-think -> abort
        _B_PLAN, "Action: finish",         # worker B still runs
        "fine as is",                      # critic runs (pool non-empty), no-op
        '{"meetings": [{"person_id": "p2", "start_time": 200}, '
        '{"person_id": "p3", "start_time": 240}]}',
    ])
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert client.i == 5
    roles = [c.role for c in result.calls]
    assert roles == ["worker_a", "worker_b", "worker_b", "critic", "finalize"]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p2", 200), ("p3", 240)
    ]


def test_both_workers_empty_skips_critic() -> None:
    # Candidate pool empty -> the critic would be a fresh solo attempt (hidden
    # retry) -> straight to finalize (C3-6).
    client = _FakeClient(["", "", '{"meetings": []}'])
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert client.i == 3
    assert [c.role for c in result.calls] == ["worker_a", "worker_b", "finalize"]
    assert result.final_plan is not None and result.final_plan.meetings == []
    assert result.best_plan_so_far.meetings == []


def test_critic_empty_and_unparseable_are_noops() -> None:
    for critic_answer in ("", "The draft looks strong overall."):
        client = _full_pipeline_client(critic_answer)
        result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
        assert client.i == 6  # exactly one critic call, never retried
        assert [c.role for c in result.calls].count("critic") == 1
        # The fallback (best single sub-plan; A on the 2-2 tie) survives every
        # critic failure mode -- the raw draft is never scored.
        assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
            ("p0", 10), ("p1", 50)
        ]


def test_critic_gate_rejects_worse_shorter_or_invalid() -> None:
    # Shorter, same-length, and out-of-pool/invalid proposals must all be
    # rejected by the hidden gate; the fallback (A's sub-plan on the tie) stands.
    for critic_answer in (
        "Action: propose[p0@10]",                              # shorter
        "Action: propose[p2@200, p3@240]",                     # same length as fallback
        "Action: propose[p0@10, p1@50, p2@200, p3@240, p9@300]",  # p9 outside pool + invalid
    ):
        client = _full_pipeline_client(critic_answer)
        result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
        assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
            ("p0", 10), ("p1", 50)
        ]


def test_critic_out_of_pool_person_is_rejected_even_if_valid_and_longer() -> None:
    # Worker A never proposed p1, so p1 is outside the candidate pool. The
    # critic's 4-meeting plan is full-instance VALID and longer than the
    # fallback, but the subset conjunct of the gate must reject it.
    client = _FakeClient([
        "Action: propose[p0@10]", "Action: finish",        # A -> pool gets p0 only
        _B_PLAN, "Action: finish",                         # B -> p2, p3
        "Action: propose[p0@10, p1@50, p2@200, p3@240]",   # p1 not in pool
        '{"meetings": [{"person_id": "p2", "start_time": 200}, '
        '{"person_id": "p3", "start_time": 240}]}',
    ])
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
    # Fallback = B's 2-meeting sub-plan (longer than A's single meeting).
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p2", 200), ("p3", 240)
    ]
    # And the critic never even saw p1's data: no data line, no location.
    critic_prompt = " ".join(m["content"] for m in client.seen[4])
    assert "p1 @" not in critic_prompt and "A2" not in critic_prompt


def test_critic_improvement_within_pool_is_adopted() -> None:
    # Workers surface 3 candidates; the critic coordinates them into a valid
    # 3-meeting route (> fallback 2) using only pool people -> adopted.
    client = _FakeClient([
        "Action: propose[p0@10]", "Action: finish",   # worker A (partial)
        _B_PLAN, "Action: finish",                    # worker B
        "Action: propose[p0@10, p2@200, p3@240]",     # critic stitches the pool
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p2", "start_time": 200}, '
        '{"person_id": "p3", "start_time": 240}]}',
    ])
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p0", 10), ("p2", 200), ("p3", 240)
    ]


def test_fallback_is_best_single_worker_subplan() -> None:
    # A has 1 meeting, B has 2; with a no-op critic the scored plan must be B's
    # sub-plan -- C3's floor is the best half, by selection, never construction.
    client = _FakeClient([
        "Action: propose[p0@10]", "Action: finish",
        _B_PLAN, "Action: finish",
        "the draft covers everyone reachable",        # critic no-op
        '{"meetings": [{"person_id": "p2", "start_time": 200}, '
        '{"person_id": "p3", "start_time": 240}]}',
    ])
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p2", 200), ("p3", 240)
    ]


def test_tight_cap_skips_all_llm_stages_but_still_finalises() -> None:
    # cap barely above the reserve: neither worker call fits, merged is empty,
    # the critic is skipped -- only the finalisation emit runs (same as C1/C2).
    client = _FakeClient(['{"meetings": []}'])
    result = run_hierarchical(_instance(), client, cap=300)  # type: ignore[arg-type]

    assert client.i == 1
    assert [c.role for c in result.calls] == ["finalize"]
    assert result.final_plan is not None and result.final_plan.meetings == []
    assert result.tokens.budget_exhausted  # the GLOBAL guard is what fired here


def test_single_person_instance_skips_worker_b() -> None:
    locs = ["S", "A1"]
    inst = Instance(
        instance_id="c3-single", seed=0,
        generator_params=GeneratorParams(
            n_people=1, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S", start_time=0, end_of_day=480, meeting_duration=30,
        waiting_allowed=True, tie_break="earliest_start",
        locations=locs,
        people=[Person(person_id="p0", location="A1", window_start=0, window_end=480)],
        travel_times=_travel(locs, {("S", "A1"): 10}),
    )
    client = _FakeClient([
        "Action: propose[p0@10]", "Action: finish",
        "keep it",                                    # critic (merged non-empty), no-op
        '{"meetings": [{"person_id": "p0", "start_time": 10}]}',
    ])
    result = run_hierarchical(inst, client, cap=8000)  # type: ignore[arg-type]

    roles = [c.role for c in result.calls]
    assert roles == ["worker_a", "worker_a", "critic", "finalize"]  # no worker_b
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [
        ("p0", 10)
    ]


def test_finalize_is_the_shared_react_core_node() -> None:
    # Identity, not similarity: C3 finalisation IS the C1/C2 node (C3-7).
    assert hierarchical.finalize_step is react_core.finalize_step


def test_transcript_sanity_clean_including_critic() -> None:
    client = _full_pipeline_client("Action: propose[p0@10, p1@50, p2@200, p3@240]")
    result = run_hierarchical(_instance(), client, cap=8000)  # type: ignore[arg-type]
    sanity = transcript_sanity(result.transcript)
    assert sanity["clean"], sanity


# --------------------------------------------------------------------------- #
# WorkerBudgetView arithmetic (pure).
# --------------------------------------------------------------------------- #
def test_worker_budget_view_min_of_global_and_quota() -> None:
    shared = BudgetLedger(cap=2000, finalization_reserve=256)
    view = WorkerBudgetView(shared=shared, quota=300)
    # Quota is the tighter bound here: min(2000-256-50, 300-50) = 250.
    assert view.max_new_tokens(50) == 250
    view.record_call(input_tokens=50, thinking_tokens=0, answer_tokens=100)
    assert shared.total == 150 and view.spent == 150
    # Now quota room is 300-150-50 = 100.
    assert view.max_new_tokens(50) == 100
    # Quota block does NOT set the global flag.
    view.record_call(input_tokens=50, thinking_tokens=0, answer_tokens=50)
    assert view.should_finalize(250)  # quota_remaining=50 < input 250
    assert not view.global_cut
    # A worker may never issue a finalising call.
    try:
        view.max_new_tokens(10, finalizing=True)
        raise AssertionError("finalizing call must raise")
    except ValueError:
        pass


def test_worker_budget_view_flags_global_cut() -> None:
    shared = BudgetLedger(cap=400, finalization_reserve=256)
    view = WorkerBudgetView(shared=shared, quota=1000)  # quota looser than global
    assert view.should_finalize(200)  # global: 400-256-200 < 0
    assert view.global_cut


# --------------------------------------------------------------------------- #
# Runner dispatch (end-to-end slice through CP-SAT + scorer).
# --------------------------------------------------------------------------- #
def test_runner_dispatches_c3_end_to_end() -> None:
    locs = ["S", "A1", "B1"]
    inst = Instance(
        instance_id="c3-runner", seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=0.1, overlap=0.1,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S", start_time=0, end_of_day=480, meeting_duration=30,
        waiting_allowed=True, tie_break="earliest_start",
        locations=locs,
        people=[
            Person(person_id="p0", location="A1", window_start=0, window_end=480),
            Person(person_id="p1", location="B1", window_start=0, window_end=480),
        ],
        travel_times=_travel(locs, {("S", "A1"): 10, ("S", "B1"): 60, ("A1", "B1"): 60}),
    )
    client = _FakeClient([
        "Action: propose[p0@10]", "Action: finish",   # worker A ({p0})
        "Action: propose[p1@70]", "Action: finish",   # worker B ({p1})
        "Action: propose[p0@10, p1@100]",             # critic (same length, no-op)
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p1", "start_time": 100}]}',
    ])
    run = run_single_instance(
        instance=inst, client=client, condition="c3_mas", cap=4000,
        model_label="offline-smoke-client",
    )
    assert run.run_result.condition is Condition.C3_MAS
    # The 2-meeting combination is the CRITIC's coordination (pool {p0, p1},
    # valid, longer than the 1-meeting fallback) -- never a mechanical merge.
    assert [(m.person_id, m.start_time) for m in run.run_result.best_plan_so_far.meetings] == [
        ("p0", 10), ("p1", 100)
    ]
    assert run.run_result.score.valid
    assert run.run_result.score.satisfaction == 1.0  # optimum 2, both met
    assert run.sanity["clean"], run.sanity
    assert run.run_result.tokens.total <= 4000
