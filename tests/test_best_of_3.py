# tests/test_best_of_3.py
"""Offline tests for C5 -- Validated Best-of-3 single-agent sampling.

C5 can go wrong in ways that still produce a plausible run: three finalisations instead of
one, a quota that silently transfers and makes the result order-dependent, a remainder
handed to some attempt, a selector that peeks at the oracle, or a seed sequence that never
actually reaches the endpoint. Each has a test here, on scripted clients, so a failure
points at the rule rather than at the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agents.multi_agent.worker import WorkerBudgetView
from src.agents.single_agent.best_of_3 import (
    BASE_SEED,
    CONDITION,
    N_ATTEMPTS,
    AttemptBudgetView,
    AttemptOutcome,
    attempt_quota,
    attempt_seed,
    run_best_of_3,
    select_best,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger
from src.schemas import (
    AnswerContract,
    Condition,
    GeneratorParams,
    Instance,
    Meeting,
    Person,
    TravelStructure,
)


def make_instance() -> Instance:
    """Three people, one depot, generous windows.

    Travel S->A/B/C is 10 and A<->B is 20, so `p0@10` then `p1@60` is feasible while
    `p0@10, p1@10` is not: the route cannot be in two places at once. That gives the
    selection tests a real valid plan, a real longer valid plan, and a real infeasible
    one without depending on the generator.
    """
    return Instance(
        instance_id="bon-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=3, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=400,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B", "C"],
        people=[
            Person(person_id="p0", location="A", window_start=0, window_end=200),
            Person(person_id="p1", location="B", window_start=0, window_end=300),
            Person(person_id="p2", location="C", window_start=0, window_end=350),
        ],
        travel_times={
            "S": {"S": 0, "A": 10, "B": 10, "C": 10},
            "A": {"S": 10, "A": 0, "B": 20, "C": 20},
            "B": {"S": 10, "A": 20, "B": 0, "C": 20},
            "C": {"S": 10, "A": 20, "B": 20, "C": 0},
        },
    )


# --------------------------------------------------------------------------- #
# A scripted client that records the seed of every call.
# --------------------------------------------------------------------------- #
@dataclass
class ScriptedClient:
    """Replies from a per-call script; counts input crudely but consistently."""

    scripts: dict[int, list[str]] = field(default_factory=dict)
    default: list[str] = field(default_factory=list)
    seen: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 40
    _cursor: dict[int | None, int] = field(default_factory=dict)

    def count_input(self, messages, enable_thinking: bool = True) -> int:
        return self.input_tokens

    def complete(self, messages, *, max_tokens: int, enable_thinking: bool = True,
                 guided_json=None, stop=None, seed: int | None = None):
        script = self.scripts.get(seed, self.default)
        i = self._cursor.get(seed, 0)
        text = script[i] if i < len(script) else "Action: finish"
        self._cursor[seed] = i + 1
        self.seen.append({"seed": seed, "max_tokens": max_tokens,
                          "thinking": enable_thinking, "guided": guided_json is not None,
                          "text": text})
        # A real endpoint cannot generate beyond `max_tokens`, and the quota invariant
        # depends on that: the guard grants `quota_remaining - input`, so thinking plus
        # answer must fit inside the grant or the ledger would book more than the quota.
        thinking_tokens = min(3, max_tokens) if enable_thinking else 0
        answer_tokens = min(max(1, len(text) // 4), max(0, max_tokens - thinking_tokens))
        return _Resp(answer=text, input_tokens=self.input_tokens,
                     thinking_tokens=thinking_tokens, answer_tokens=answer_tokens,
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


def propose(*pairs: str) -> str:
    return "Thought: ok\nAction: propose[" + ", ".join(pairs) + "]"


# --------------------------------------------------------------------------- #
# 3. The seed sequence.
# --------------------------------------------------------------------------- #
def test_the_seed_sequence_is_exactly_42_43_44():
    assert BASE_SEED == 42
    assert [attempt_seed(i) for i in range(N_ATTEMPTS)] == [42, 43, 44]


def test_every_search_call_carries_its_attempt_seed_and_the_finalisation_does_not():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    run_best_of_3(inst, client, cap=8000, max_steps=4)

    search = [c for c in client.seen if not c["guided"]]
    final = [c for c in client.seen if c["guided"]]
    assert {c["seed"] for c in search} == {42, 43, 44}
    assert len(final) == 1
    # The terminal emit goes through the unwrapped client: no seed override at all.
    assert final[0]["seed"] is None


def test_the_default_seed_of_the_other_conditions_is_untouched():
    """C1/C2/C3 never pass a seed, so the client keeps its single fixed one."""
    from src.agents.single_agent.react import run_react
    inst = make_instance()
    client = ScriptedClient(default=["Action: finish"])
    run_react(inst, client, cap=8000, max_steps=4)
    assert {c["seed"] for c in client.seen} == {None}


# --------------------------------------------------------------------------- #
# 1, 4, 5. Budget accounting, non-transferable quotas, unused remainder.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap", [2000, 4000, 8000, 16000, 32000, 64000])
def test_quota_is_floor_of_working_budget_over_three(cap):
    w = cap - DEFAULT_FINALIZATION_RESERVE
    q = attempt_quota(cap, DEFAULT_FINALIZATION_RESERVE)
    assert q == w // N_ATTEMPTS
    assert 0 <= w - N_ATTEMPTS * q <= N_ATTEMPTS - 1     # remainder is 0, 1 or 2


def test_the_remainder_is_left_unused_and_given_to_no_attempt():
    # W = 64000 - 256 = 63744; 63744 / 3 = 21248 exactly, so pick a cap with a remainder.
    cap, reserve = 2002, 0
    w = cap - reserve
    q = attempt_quota(cap, reserve)
    assert w % N_ATTEMPTS != 0                    # this cap really has a remainder
    assert N_ATTEMPTS * q < w                     # and it is not distributed
    assert w - N_ATTEMPTS * q == w % N_ATTEMPTS


def test_total_spend_never_exceeds_the_cap_and_the_reserve_survives():
    inst = make_instance()
    # A long script so every attempt genuinely tries to spend its whole quota.
    client = ScriptedClient(default=[propose("p0@10")] * 200)
    result = run_best_of_3(inst, client, cap=8000, max_steps=200)
    assert result.tokens.total <= 8000
    assert result.tokens.total == (result.tokens.input_tokens
                                   + result.tokens.thinking_tokens
                                   + result.tokens.answer_tokens)


def test_an_attempt_cannot_spend_more_than_its_quota():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10")] * 200)
    cap, reserve = 8000, DEFAULT_FINALIZATION_RESERVE
    quota = attempt_quota(cap, reserve)
    ledger = BudgetLedger(cap=cap, finalization_reserve=reserve)
    view = AttemptBudgetView(shared=ledger, quota=quota)
    for _ in range(50):
        room = view.max_new_tokens(40)
        if room <= 0:
            break
        # thinking + answer == the whole grant: the worst case the endpoint can produce.
        view.record_call(input_tokens=40, thinking_tokens=min(3, room),
                         answer_tokens=room - min(3, room))
    assert view.spent <= quota


def test_an_unspent_quota_does_not_transfer_to_the_next_attempt():
    """The order-independence rule: attempt 1's room must not grow because attempt 0
    stopped early."""
    cap, reserve = 8000, DEFAULT_FINALIZATION_RESERVE
    quota = attempt_quota(cap, reserve)
    ledger = BudgetLedger(cap=cap, finalization_reserve=reserve)

    first = AttemptBudgetView(shared=ledger, quota=quota)
    first.record_call(input_tokens=10, thinking_tokens=1, answer_tokens=1)   # stops early
    second = AttemptBudgetView(shared=ledger, quota=quota)
    assert second.quota_remaining == quota           # not quota + first's leftover
    assert second.quota == quota


def test_the_attempt_view_matches_the_frozen_c3_worker_arithmetic():
    """The local mirror is proven equivalent rather than assumed: C3 must not acquire a
    dependency on a condition specified after it."""
    for cap, quota, spent, inp in [(8000, 2000, 0, 40), (8000, 2000, 1500, 40),
                                   (2000, 500, 400, 100), (64000, 21248, 20000, 500)]:
        a = AttemptBudgetView(shared=BudgetLedger(cap=cap), quota=quota, spent=spent)
        w = WorkerBudgetView(shared=BudgetLedger(cap=cap), quota=quota, spent=spent)
        assert a.max_new_tokens(inp) == w.max_new_tokens(inp)
        assert a.should_finalize(inp) == w.should_finalize(inp)
        assert a.quota_remaining == w.quota_remaining


def test_an_attempt_may_not_issue_a_finalising_call():
    view = AttemptBudgetView(shared=BudgetLedger(cap=8000), quota=2000)
    with pytest.raises(ValueError, match="cannot issue a finalising call"):
        view.max_new_tokens(40, finalizing=True)


# --------------------------------------------------------------------------- #
# 2, 10, 11. Three trajectories, one finalisation, call roles and metadata.
# --------------------------------------------------------------------------- #
def test_exactly_three_search_trajectories_and_exactly_one_finalisation():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    result = run_best_of_3(inst, client, cap=8000, max_steps=4)

    roles = [c.role for c in result.calls]
    assert roles.count("finalize") == 1
    assert {r for r in roles if r != "finalize"} == {
        f"bon_attempt_{i}" for i in range(N_ATTEMPTS)}
    assert sum(1 for c in client.seen if c["guided"]) == 1


def test_call_records_carry_the_attempt_index_and_the_seed_that_ran():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    result = run_best_of_3(inst, client, cap=8000, max_steps=4)
    for c in result.calls:
        if c.role == "finalize":
            assert c.attempt_index is None and c.sampling_seed is None
        else:
            i = int(c.role.rsplit("_", 1)[1])
            assert c.attempt_index == i
            assert c.sampling_seed == attempt_seed(i)


def test_proposal_rows_are_attributable_to_their_attempt():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    result = run_best_of_3(inst, client, cap=8000, max_steps=4)
    assert result.proposals
    for row in result.proposals:
        assert row["condition"] == CONDITION
        assert row["role"].startswith("bon_attempt_")
        assert row["attempt_index"] == int(row["role"].rsplit("_", 1)[1])


# --------------------------------------------------------------------------- #
# 6, 7, 8, 9. Selection.
# --------------------------------------------------------------------------- #
def _outcome(index: int, plan: AnswerContract) -> AttemptOutcome:
    return AttemptOutcome(index=index, seed=attempt_seed(index), plan=plan, calls=[],
                          transcript=[], proposals=[], n_steps=0, aborted=False,
                          empty_turns=0, global_budget_cut=False)


def _plan(*pairs: tuple[str, int]) -> AnswerContract:
    return AnswerContract(meetings=[Meeting(person_id=p, start_time=t) for p, t in pairs])


def test_the_candidate_with_most_meetings_wins():
    inst = make_instance()
    outs = [_outcome(0, _plan(("p0", 10))),
            _outcome(1, _plan(("p0", 10), ("p1", 60))),
            _outcome(2, AnswerContract.empty())]
    idx, plan = select_best(inst, outs)
    assert idx == 1 and len(plan.meetings) == 2


def test_a_tie_is_broken_by_the_lowest_attempt_index():
    inst = make_instance()
    same = _plan(("p0", 10))
    idx, _ = select_best(inst, [_outcome(0, same), _outcome(1, same), _outcome(2, same)])
    assert idx == 0
    # And the rule is about the index, not the list order.
    idx, _ = select_best(inst, [_outcome(2, same), _outcome(1, same), _outcome(0, same)])
    assert idx == 0


def test_an_invalid_candidate_cannot_win_however_long_it_is():
    inst = make_instance()
    bogus = _plan(("p0", 10), ("p1", 10), ("p2", 10))   # cannot be in three places
    good = _plan(("p0", 10))
    idx, plan = select_best(inst, [_outcome(0, bogus), _outcome(1, good),
                                   _outcome(2, AnswerContract.empty())])
    assert idx == 1 and plan == good


def test_when_nothing_is_valid_attempt_zero_is_taken():
    inst = make_instance()
    bogus = _plan(("p0", 10), ("p1", 10))
    outs = [_outcome(0, bogus), _outcome(1, bogus), _outcome(2, bogus)]
    idx, plan = select_best(inst, outs)
    assert idx == 0 and plan is outs[0].plan


def test_selection_is_deterministic_across_repeats():
    inst = make_instance()
    outs = [_outcome(0, _plan(("p0", 10))), _outcome(1, _plan(("p1", 60))),
            _outcome(2, _plan(("p0", 10)))]
    assert {select_best(inst, outs)[0] for _ in range(20)} == {0}


def test_the_selector_never_consults_the_solver_optimum():
    """Guards the oracle-isolation invariant: C5 selects on structure, never on score."""
    import tokenize
    from pathlib import Path
    path = (Path(__file__).resolve().parents[1] / "src" / "agents" / "single_agent"
            / "best_of_3.py")
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))
    for banned in ("solve", "optimum", "score_plan", "OracleSolution", "satisfaction",
                   "brute_force"):
        assert banned not in code, f"C5 source references {banned!r}"
    # The one oracle import allowed is the same harness-side gate every condition uses.
    assert "is_valid" in code


def test_selection_costs_nothing():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    before = len(client.seen)
    select_best(inst, [_outcome(0, _plan(("p0", 10))), _outcome(1, AnswerContract.empty()),
                       _outcome(2, AnswerContract.empty())])
    assert len(client.seen) == before


# --------------------------------------------------------------------------- #
# End-to-end behaviour of the whole condition.
# --------------------------------------------------------------------------- #
def test_the_best_of_the_three_trajectories_is_what_gets_finalised():
    inst = make_instance()
    client = ScriptedClient(
        scripts={
            42: [propose("p0@10"), "Action: finish"],
            43: [propose("p0@10"), propose("p0@10, p1@60"), "Action: finish"],
            44: ["Action: finish"],
        },
        default=["Action: finish"],
    )
    result = run_best_of_3(inst, client, cap=8000, max_steps=6)
    assert len(result.best_plan_so_far.meetings) == 2       # attempt 1's richer plan
    final = [c for c in client.seen if c["guided"]][0]
    assert "p1@60" in final["text"] or len(result.final_plan.meetings) == 2


def test_c5_is_not_three_full_runs_at_cap_over_three():
    """One reserve, not three: the attempts share a single instance-level reserve."""
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10")] * 200)
    cap, reserve = 8000, DEFAULT_FINALIZATION_RESERVE
    result = run_best_of_3(inst, client, cap=cap, max_steps=200)
    # Three separate C1 runs at cap/3 would each hold back a reserve, i.e. 3 * reserve
    # withheld in total. Here the working budget is cap - reserve once.
    assert 3 * attempt_quota(cap, reserve) == (cap - reserve) - ((cap - reserve) % 3)
    assert result.tokens.total <= cap


def test_attempts_do_not_individually_finalise():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    run_best_of_3(inst, client, cap=8000, max_steps=4)
    guided = [c for c in client.seen if c["guided"]]
    thinking_off = [c for c in client.seen if not c["thinking"]]
    assert len(guided) == 1 and len(thinking_off) == 1


def test_transcript_marks_each_attempt_separately():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    result = run_best_of_3(inst, client, cap=8000, max_steps=4)
    markers = [m["content"] for m in result.transcript
               if m["content"].startswith("[bon_attempt_")]
    assert markers == ["[bon_attempt_0 conversation]", "[bon_attempt_1 conversation]",
                       "[bon_attempt_2 conversation]"]


# --------------------------------------------------------------------------- #
# 12. Harness, run-id and schema compatibility.
# --------------------------------------------------------------------------- #
def test_the_condition_is_registered_and_dispatchable():
    from src.harness.runner import SUPPORTED_CONDITIONS
    assert Condition.C5_BEST_OF_3.value == "c5_best_of_3"
    assert Condition.C5_BEST_OF_3 in SUPPORTED_CONDITIONS


def test_the_run_id_formula_is_unchanged_and_covers_c5():
    from src.harness.runner import run_id_for
    assert run_id_for(Condition.C5_BEST_OF_3, "inst-1", 64000, "Qwen/Qwen3-32B-AWQ") == \
        "c5_best_of_3__inst-1__cap64000__Qwen-Qwen3-32B-AWQ"
    assert run_id_for(Condition.C1_REACT, "inst-1", 64000, "Qwen/Qwen3-32B-AWQ") == \
        "c1_react__inst-1__cap64000__Qwen-Qwen3-32B-AWQ"


def test_a_call_record_without_the_new_fields_is_still_valid():
    """Backwards compatibility: every document written before C5 existed still parses."""
    from src.schemas import CallRecord
    old = CallRecord(role="react_step", input_tokens=10, thinking_tokens=1,
                     answer_tokens=2, latency_seconds=0.5)
    assert old.attempt_index is None and old.sampling_seed is None
    assert "attempt_index" in old.model_dump()


def test_an_unknown_field_is_still_refused():
    from pydantic import ValidationError
    from src.schemas import CallRecord
    with pytest.raises(ValidationError):
        CallRecord(role="x", input_tokens=0, thinking_tokens=0, answer_tokens=0,
                   latency_seconds=0.0, nonsense=1)


def test_an_unknown_condition_is_still_a_value_error():
    from src.harness.runner import _resolve_condition
    with pytest.raises(ValueError, match="unknown condition"):
        _resolve_condition("c9_wat")


# --------------------------------------------------------------------------- #
# 13. The other conditions are untouched.
# --------------------------------------------------------------------------- #
def test_react_core_is_not_modified_by_c5():
    """C5 wires the shared nodes; it must not have edited them."""
    import inspect
    from src.agents import react_core
    src = inspect.getsource(react_core)
    for banned in ("bon_attempt", "best_of_3", "attempt_index", "c5_"):
        assert banned not in src, f"react_core mentions {banned!r}"


def test_c1_and_c2_and_c3_still_dispatch_to_their_own_runners():
    from src.agents.multi_agent.hierarchical import run_hierarchical
    from src.agents.single_agent.react import run_react
    from src.agents.single_agent.react_verify_revise import run_verify_revise
    from src.harness.runner import SUPPORTED_CONDITIONS
    assert SUPPORTED_CONDITIONS[Condition.C1_REACT] is run_react
    assert SUPPORTED_CONDITIONS[Condition.C2_VERIFY_REVISE] is run_verify_revise
    assert SUPPORTED_CONDITIONS[Condition.C3_MAS] is run_hierarchical


# --------------------------------------------------------------------------- #
# Log-only diversity diagnostics: they must describe the run, never shape it.
# --------------------------------------------------------------------------- #
def test_the_three_trajectory_products_and_the_winner_are_preserved():
    inst = make_instance()
    client = ScriptedClient(
        scripts={
            42: [propose("p0@10"), "Action: finish"],
            43: [propose("p0@10"), propose("p0@10, p1@60"), "Action: finish"],
            44: ["Action: finish"],
        },
        default=["Action: finish"],
    )
    result = run_best_of_3(inst, client, cap=8000, max_steps=6)
    d = result.diagnostics
    assert [p["attempt_index"] for p in d["attempt_products"]] == [0, 1, 2]
    assert [p["seed"] for p in d["attempt_products"]] == [42, 43, 44]
    assert [p["n_meetings"] for p in d["attempt_products"]] == [1, 2, 0]
    assert d["winner_attempt_index"] == 1
    assert d["n_distinct_products"] == 3


def test_identical_trajectories_are_reported_as_one_distinct_product():
    inst = make_instance()
    client = ScriptedClient(default=[propose("p0@10"), "Action: finish"])
    d = run_best_of_3(inst, client, cap=8000, max_steps=4).diagnostics
    assert d["n_distinct_products"] == 1
    assert d["winner_attempt_index"] == 0          # a tie goes to the lowest index


def test_the_diagnostics_agree_with_reconstruction_from_the_proposal_rows():
    """The explicit record must match what reconstruction would have produced, so the
    two never disagree in the analysis layer."""
    inst = make_instance()
    client = ScriptedClient(
        scripts={
            42: [propose("p0@10"), "Action: finish"],
            43: [propose("p0@10"), propose("p0@10, p1@60"), "Action: finish"],
            44: ["Action: finish"],
        },
        default=["Action: finish"],
    )
    result = run_best_of_3(inst, client, cap=8000, max_steps=6)
    for product in result.diagnostics["attempt_products"]:
        i = product["attempt_index"]
        accepted = [r for r in result.proposals
                    if r["attempt_index"] == i and r["accepted_into_best_plan"]]
        expected = accepted[-1]["plan"] if accepted else []
        assert product["plan"] == expected


def test_the_diagnostics_do_not_change_tokens_calls_or_the_selected_plan():
    """Instrumentation only: the run must be identical with and without reading it."""
    inst = make_instance()
    script = {42: [propose("p0@10"), "Action: finish"],
              43: [propose("p0@10"), propose("p0@10, p1@60"), "Action: finish"],
              44: ["Action: finish"]}
    a = run_best_of_3(inst, ScriptedClient(scripts=dict(script),
                                           default=["Action: finish"]),
                      cap=8000, max_steps=6)
    b = run_best_of_3(inst, ScriptedClient(scripts=dict(script),
                                           default=["Action: finish"]),
                      cap=8000, max_steps=6)
    assert a.tokens.model_dump() == b.tokens.model_dump()
    assert [c.role for c in a.calls] == [c.role for c in b.calls]
    assert a.best_plan_so_far == b.best_plan_so_far
    assert a.diagnostics == b.diagnostics


def test_conditions_without_diagnostics_emit_an_empty_block():
    from src.agents.single_agent.react import run_react
    inst = make_instance()
    result = run_react(inst, ScriptedClient(default=["Action: finish"]),
                       cap=8000, max_steps=4)
    assert result.diagnostics == {}
