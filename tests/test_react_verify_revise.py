# tests/test_react_verify_revise.py
"""Offline tests for C2 (ReAct + verify/revise).

Two layers, both endpoint-free:
  * the routers are pure functions of the state dict -> exercised directly;
  * the whole LangGraph wiring is exercised with a scripted fake client, so the fixed
    Draft -> Verify -> Revise -> Verify -> Revise -> Stop structure, the revision cap,
    the hidden best-plan bookkeeping, and the budget-skip path are all checked without a
    live model.
"""

from __future__ import annotations

from typing import Any

from src.agents.single_agent.react_verify_revise import (
    MAX_REVISIONS,
    route_after_agent,
    route_after_revise,
    route_after_verify,
    run_verify_revise,
)
from src.core import LLMResponse
from src.schemas import GeneratorParams, Instance, Person, TravelStructure


def _instance() -> Instance:
    return Instance(
        instance_id="c2-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=300,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=[
            Person(person_id="p0", location="A", window_start=0, window_end=100),
            Person(person_id="p1", location="B", window_start=0, window_end=200),
        ],
        travel_times={
            "S": {"S": 0, "A": 10, "B": 10},
            "A": {"S": 10, "A": 0, "B": 20},
            "B": {"S": 10, "A": 20, "B": 0},
        },
    )


class _FakeClient:
    """Returns scripted completions in order; flat token counts keep the budget simple."""

    def __init__(self, answers: list[str], *, input_tokens: int = 50, answer_tokens: int = 5):
        self._answers = answers
        self.i = 0
        self._input = input_tokens
        self._answer = answer_tokens

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        return self._input

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        text = self._answers[self.i]
        self.i += 1
        return LLMResponse(
            thinking="", answer=text, raw_text=text,
            input_tokens=self._input, thinking_tokens=0, answer_tokens=self._answer,
            finish_reason="stop",
        )


# --------------------------------------------------------------------------- #
# Routers (pure).
# --------------------------------------------------------------------------- #
def test_route_after_agent() -> None:
    # Only a NORMALLY finished draft enters verification.
    assert route_after_agent({"finalize": True, "via_budget": False}) == "verify"
    assert route_after_agent({"finalize": True, "via_budget": True}) == "finalize"
    # An aborted draft (empty post-think / second format failure) must NOT be
    # converted into extra verify/revise turns -- the section-2 empty->finalise
    # rule is identical on every condition.
    assert route_after_agent(
        {"finalize": True, "via_budget": False, "aborted": True}
    ) == "finalize"
    assert route_after_agent({"pending_tool": {"tool": "x"}}) == "tool"
    assert route_after_agent({}) == "agent"


def test_route_after_verify() -> None:
    assert route_after_verify({"vr_done": True}) == "finalize"
    assert route_after_verify({"vr_done": False}) == "revise"


def test_route_after_revise() -> None:
    assert route_after_revise({"vr_done": True, "revision_count": 0}) == "finalize"
    assert route_after_revise({"vr_done": False, "revision_count": MAX_REVISIONS}) == "finalize"
    assert route_after_revise({"vr_done": False, "revision_count": 1}) == "verify"


def test_max_revisions_is_two() -> None:
    assert MAX_REVISIONS == 2


# --------------------------------------------------------------------------- #
# Full wiring with a scripted fake client.
# --------------------------------------------------------------------------- #
def test_full_verify_revise_cycle() -> None:
    # Draft proposes p0, finishes; verify nudges; revise adds p1; second cycle keeps it;
    # finalize serialises the best plan.
    client = _FakeClient([
        "Action: propose[p0@10]",                       # draft step 1
        "Action: finish",                                # draft step 2 -> verify
        "You could still add p1.",                       # verify 1
        "Action: propose[p0@10, p1@60]",                 # revise 1 (count -> 1)
        "The plan looks complete.",                      # verify 2
        "Action: propose[p0@10, p1@60]",                 # revise 2 (count -> 2 -> stop)
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p1", "start_time": 60}]}',       # finalize
    ])
    result = run_verify_revise(_instance(), client, cap=4000)  # type: ignore[arg-type]

    assert client.i == 7  # exactly the scripted calls, no extra loops
    assert result.final_plan is not None
    assert [(m.person_id, m.start_time) for m in result.final_plan.meetings] == [("p0", 10), ("p1", 60)]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [("p0", 10), ("p1", 60)]

    roles = [c.role for c in result.calls]
    assert roles.count("verify") == 2
    assert roles.count("revise") == MAX_REVISIONS
    assert roles.count("finalize") == 1
    assert not result.tokens.budget_exhausted


def test_revise_keeps_best_when_revision_is_worse() -> None:
    # Draft finds the 2-meeting plan; revise then proposes a worse (1-meeting) plan.
    # best_plan_so_far must not regress (hidden validity bookkeeping).
    client = _FakeClient([
        "Action: propose[p0@10, p1@60]",                 # draft step 1 (best = 2)
        "Action: finish",                                # draft step 2 -> verify
        "Maybe drop p1.",                                # verify 1
        "Action: propose[p0@10]",                        # revise 1 -> worse, must be ignored
        "Keep it.",                                       # verify 2
        "Action: propose[p0@10]",                        # revise 2 -> still worse
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p1", "start_time": 60}]}',       # finalize from best_plan_so_far
    ])
    result = run_verify_revise(_instance(), client, cap=4000)  # type: ignore[arg-type]
    assert [(m.person_id, m.start_time) for m in result.best_plan_so_far.meetings] == [("p0", 10), ("p1", 60)]


def test_budget_exhaustion_skips_verify_revise() -> None:
    # cap barely above the reserve: no reasoning call ever fits, so the draft routes
    # straight to finalisation; verify/revise never run.
    client = _FakeClient(['{"meetings": []}'])
    result = run_verify_revise(_instance(), client, cap=300, finalization_reserve=256)  # type: ignore[arg-type]

    roles = [c.role for c in result.calls]
    assert "verify" not in roles and "revise" not in roles
    assert result.final_plan is not None and result.final_plan.meetings == []
    assert result.tokens.budget_exhausted


# --------------------------------------------------------------------------- #
# Aborted drafts skip verify/revise (section-2 empty->finalise, same rule as C1).
# --------------------------------------------------------------------------- #
class _ScriptedClient:
    """Returns pre-built LLMResponse objects in order (full control over the
    thinking/answer/raw_text split, unlike _FakeClient's plain answers)."""

    def __init__(self, responses: list[LLMResponse], *, input_tokens: int = 50):
        self._responses = list(responses)
        self.i = 0
        self._input = input_tokens

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        return self._input

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        resp = self._responses[self.i]
        self.i += 1
        return resp


def _resp(*, thinking: str = "", answer: str = "", raw_text: str | None = None) -> LLMResponse:
    if raw_text is None:
        raw_text = (f"<think>{thinking}</think>" if thinking else "") + answer
    return LLMResponse(
        thinking=thinking, answer=answer, raw_text=raw_text,
        input_tokens=50, thinking_tokens=10 if thinking else 0,
        answer_tokens=5 if answer else 0, finish_reason="stop",
    )


def test_empty_draft_skips_verify_revise() -> None:
    # The draft's first decide step returns nothing at all -> abort -> finalise.
    # verify/revise must NOT run (they would be extra turns C1 does not get).
    client = _ScriptedClient([
        _resp(answer="", raw_text=""),
        _resp(answer='{"meetings": []}'),   # terminal emit
    ])
    result = run_verify_revise(_instance(), client, cap=4000)  # type: ignore[arg-type]

    assert client.i == 2
    roles = [c.role for c in result.calls]
    assert "verify" not in roles and "revise" not in roles
    assert roles == ["react_step", "finalize"]
    assert result.final_plan is not None and result.final_plan.meetings == []


def test_truncated_think_draft_skips_verify_revise() -> None:
    # The draft spent its whole grant inside <think> (empty post-think content).
    # Same rule as C1: straight to finalise; the rehearsed propose inside the think
    # block is never parsed, and no thinking text enters the conversation.
    client = _ScriptedClient([
        _resp(thinking="I could Action: propose[p0@10] here",
              answer="", raw_text="<think>I could Action: propose[p0@10] here"),
        _resp(answer='{"meetings": []}'),   # terminal emit
    ])
    result = run_verify_revise(_instance(), client, cap=4000)  # type: ignore[arg-type]

    assert client.i == 2
    roles = [c.role for c in result.calls]
    assert "verify" not in roles and "revise" not in roles
    # The rehearsed propose was not executed: best_plan_so_far stayed empty.
    assert result.best_plan_so_far.meetings == []
    # No thinking text was re-fed into the history.
    assert not any("think" in m["content"] for m in result.transcript)
