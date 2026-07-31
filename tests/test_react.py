# tests/test_react.py
"""Offline tests for C1.

Pure helpers first (these lock in the two bugs found by reading the first live
transcripts: quoted tool arguments and robust JSON extraction), then the LangGraph
wiring with a scripted fake client -- in particular the section-2 termination policy:
empty post-think content aborts straight to finalisation, a truncated <think> block
is never parsed for Actions nor re-fed as history, a non-empty format failure gets
exactly one retry, and finalise serialises best_plan_so_far, not the trajectory.
"""

from __future__ import annotations

from typing import Any

from src.agents.single_agent.react import (
    _execute_tool,
    _parse_action,
    _parse_answer_plan,
    _plan_from_args,
    run_react,
)
from src.core import LLMResponse
from src.schemas import GeneratorParams, Instance, Person, TravelStructure


def _instance() -> Instance:
    return Instance(
        instance_id="react-test",
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


# --------------------------------------------------------------------------- #
# Action parsing.
# --------------------------------------------------------------------------- #
def test_parse_plain_action() -> None:
    assert _parse_action("Thought: list them\nAction: list_people[]") == {
        "tool": "list_people", "args": []
    }


def test_parse_strips_quotes_around_args() -> None:
    # The exact bug from the first transcript.
    assert _parse_action("Action: get_travel_time['start', 'loc_0']") == {
        "tool": "get_travel_time", "args": ["start", "loc_0"]
    }


def test_parse_ignores_trailing_junk() -> None:
    # Weak models append noise after the action; the first [...] still parses.
    parsed = _parse_action('Action: get_availability[p0]   {"id": 0, "name": "x"}')
    assert parsed == {"tool": "get_availability", "args": ["p0"]}


def test_parse_finish_with_and_without_brackets() -> None:
    assert _parse_action("Action: finish")["tool"] == "finish"
    assert _parse_action("Action: finish[]")["tool"] == "finish"


def test_parse_no_action_is_none() -> None:
    assert _parse_action("I am still thinking about the route.") is None
    assert _parse_action("") is None


# --------------------------------------------------------------------------- #
# Tool dispatch (errors are text Observations, never raises).
# --------------------------------------------------------------------------- #
def test_execute_list_people() -> None:
    assert _execute_tool(_instance(), {"tool": "list_people", "args": []}) == "People: p0, p1"


def test_execute_availability_and_travel() -> None:
    inst = _instance()
    avail = _execute_tool(inst, {"tool": "get_availability", "args": ["p0"]})
    assert avail.startswith("availability(p0)") and "'location': 'A'" in avail
    out = _execute_tool(inst, {"tool": "get_travel_time", "args": ["S", "A"]})
    assert out == "travel_time(S -> A) = 10 min"


def test_execute_unknown_person_is_text_error() -> None:
    out = _execute_tool(_instance(), {"tool": "get_availability", "args": ["zzz"]})
    assert out.startswith("Error:")


def test_execute_wrong_arity_is_text_error() -> None:
    out = _execute_tool(_instance(), {"tool": "get_travel_time", "args": ["S"]})
    assert "expects 2 arguments" in out


def test_execute_unknown_tool_is_text_error() -> None:
    out = _execute_tool(_instance(), {"tool": "teleport", "args": []})
    assert "unknown tool" in out


# --------------------------------------------------------------------------- #
# Terminal-emit JSON parsing.
# --------------------------------------------------------------------------- #
def test_parse_answer_clean_json() -> None:
    plan = _parse_answer_plan('{"meetings": [{"person_id": "p0", "start_time": 10}]}')
    assert plan is not None
    assert [(m.person_id, m.start_time) for m in plan.meetings] == [("p0", 10)]


def test_parse_answer_with_surrounding_text_and_fence() -> None:
    text = 'Here is the plan:\n```json\n{"meetings": [{"person_id": "p1", "start_time": 60}]}\n```'
    plan = _parse_answer_plan(text)
    assert plan is not None and plan.meetings[0].person_id == "p1"


def test_parse_answer_empty_meetings() -> None:
    plan = _parse_answer_plan('{"meetings": []}')
    assert plan is not None and plan.meetings == []


def test_parse_answer_malformed_is_none() -> None:
    assert _parse_answer_plan("no json here") is None
    assert _parse_answer_plan('{"meetings": [{"person_id": "p0"}]}') is None  # missing start_time


# --------------------------------------------------------------------------- #
# Proposed-plan parsing (`p0@10, p1@60`) for best_plan_so_far tracking.
# --------------------------------------------------------------------------- #
def test_plan_from_args_ok() -> None:
    plan = _plan_from_args(["p0@10", "p1@60"])
    assert plan is not None
    assert [(m.person_id, m.start_time) for m in plan.meetings] == [("p0", 10), ("p1", 60)]


def test_plan_from_args_empty_is_empty_plan() -> None:
    plan = _plan_from_args([])
    assert plan is not None and plan.meetings == []


def test_plan_from_args_bad_format_is_none() -> None:
    assert _plan_from_args(["p0-10"]) is None      # missing '@'
    assert _plan_from_args(["p0@soon"]) is None     # non-integer start
    assert _plan_from_args(["@10"]) is None          # missing person


# --------------------------------------------------------------------------- #
# LangGraph wiring with a scripted client (section-2 termination policy).
# --------------------------------------------------------------------------- #
class _ScriptedClient:
    """Returns pre-built LLMResponse objects in order; flat input counts keep the
    budget arithmetic trivial. Records the messages of the terminal emit call so a
    test can assert finalise is a short serialisation, not the trajectory."""

    def __init__(self, responses: list[LLMResponse], *, input_tokens: int = 50):
        self._responses = list(responses)
        self.i = 0
        self._input = input_tokens
        self.finalize_messages: list[dict[str, str]] | None = None

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        return self._input

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        if guided_json is not None or not enable_thinking:
            self.finalize_messages = messages
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


def test_truncated_think_routes_immediately_to_finalize() -> None:
    # The grant was spent inside <think> (no closing tag, empty post-think answer).
    # Rule: finalise immediately; the rehearsed Action inside the think block must
    # NOT be parsed/executed, and the think text must NOT enter the conversation.
    client = _ScriptedClient([
        _resp(thinking="I should call Action: list_people[] next",
              answer="", raw_text="<think>I should call Action: list_people[] next"),
        _resp(answer='{"meetings": []}'),  # terminal emit
    ])
    result = run_react(_instance(), client, cap=4000)  # type: ignore[arg-type]

    assert client.i == 2  # one aborted decide step + the terminal emit, nothing else
    assert [c.role for c in result.calls] == ["react_step", "finalize"]
    # The rehearsed tool call was not executed (no Observation ever appeared).
    assert not any("People:" in m["content"] for m in result.transcript)
    # No thinking text was re-fed into the history.
    assert not any("think" in m["content"] for m in result.transcript)
    assert result.final_plan is not None and result.final_plan.meetings == []


def test_fully_empty_output_routes_immediately_to_finalize() -> None:
    client = _ScriptedClient([
        _resp(answer="", raw_text=""),      # e.g. Ollama returning nothing at all
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=4000)  # type: ignore[arg-type]
    assert client.i == 2
    assert [c.role for c in result.calls] == ["react_step", "finalize"]


def test_nonempty_format_failure_gets_exactly_one_retry() -> None:
    client = _ScriptedClient([
        _resp(answer="Let me look at the people first."),   # slip 1 -> one nudge
        _resp(answer="Still narrating, no action line."),   # slip 2 -> abort
        _resp(answer='{"meetings": []}'),                    # terminal emit
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert client.i == 3
    assert [c.role for c in result.calls] == ["react_step", "react_step", "finalize"]
    nudges = [m for m in result.transcript
              if m["role"] == "user" and "No valid Action found" in m["content"]]
    assert len(nudges) == 1  # exactly the one permitted retry


def test_finalize_accepts_a_faithful_emit() -> None:
    # The emit round-trips best_plan_so_far exactly -> accepted, no mismatch recorded.
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10, p1@60]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": [{"person_id": "p0", "start_time": 10}, '
                     '{"person_id": "p1", "start_time": 60}]}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert result.finalization_mismatch is False
    assert result.final_plan == result.best_plan_so_far
    assert [(m.person_id, m.start_time) for m in result.final_plan.meetings] == [
        ("p0", 10), ("p1", 60)
    ]


def test_finalize_rejects_any_altered_emit() -> None:
    """Faithful serialisation is ENFORCED: a parseable emit that does not round-trip
    best_plan_so_far is discarded in favour of the structural plan, and the divergence is
    recorded. Guided JSON constrains the emit's shape, not its values, so all four kinds of
    drift must be caught: changed time, changed person, added meeting, reordered route."""
    best = "Action: propose[p0@10, p1@60]"
    altered_emits = {
        "changed time": '{"meetings": [{"person_id": "p0", "start_time": 10}, '
                        '{"person_id": "p1", "start_time": 61}]}',
        "changed person": '{"meetings": [{"person_id": "p0", "start_time": 10}, '
                          '{"person_id": "p0", "start_time": 60}]}',
        "dropped meeting": '{"meetings": [{"person_id": "p0", "start_time": 10}]}',
        "added meeting": '{"meetings": [{"person_id": "p0", "start_time": 10}, '
                         '{"person_id": "p1", "start_time": 60}, '
                         '{"person_id": "p1", "start_time": 90}]}',
        "reordered": '{"meetings": [{"person_id": "p1", "start_time": 60}, '
                     '{"person_id": "p0", "start_time": 10}]}',
    }
    for label, emit in altered_emits.items():
        client = _ScriptedClient([
            _resp(answer=best), _resp(answer="Action: finish"), _resp(answer=emit),
        ])
        result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

        assert result.finalization_mismatch is True, label
        # The scored artefact is the structural plan, byte-for-byte.
        assert result.final_plan == result.best_plan_so_far, label
        assert [(m.person_id, m.start_time) for m in result.final_plan.meetings] == [
            ("p0", 10), ("p1", 60)
        ], label


def test_finalize_serialises_best_plan_not_trajectory() -> None:
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": [{"person_id": "p0", "start_time": 10}]}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    fmsgs = client.finalize_messages
    assert fmsgs is not None
    # A short stateless serialisation prompt: system + user, never the trajectory.
    assert len(fmsgs) == 2
    assert "p0@10" in fmsgs[1]["content"]  # the decided plan is the payload
    assert all("Observation" not in m["content"] for m in fmsgs)
    assert result.final_plan is not None
    assert [(m.person_id, m.start_time) for m in result.final_plan.meetings] == [("p0", 10)]
