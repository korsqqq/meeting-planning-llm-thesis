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
from src.agents.react_core import PROPOSAL_SCHEMA_VERSION
from src.schemas import (
    GeneratorParams,
    Instance,
    InvalidReason,
    Person,
    TravelStructure,
)


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
        "tool": "list_people", "args": [], "raw": ""
    }


def test_parse_strips_quotes_around_args() -> None:
    # The exact bug from the first transcript. `raw` keeps the argument text as
    # written, for the proposal taxonomy; `args` stays the cleaned parse.
    assert _parse_action("Action: get_travel_time['start', 'loc_0']") == {
        "tool": "get_travel_time", "args": ["start", "loc_0"], "raw": "'start', 'loc_0'"
    }


def test_parse_ignores_trailing_junk() -> None:
    # Weak models append noise after the action; the first [...] still parses.
    parsed = _parse_action('Action: get_availability[p0]   {"id": 0, "name": "x"}')
    assert parsed == {"tool": "get_availability", "args": ["p0"], "raw": "p0"}


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


# --------------------------------------------------------------------------- #
# Empty-turn A/B: the locked arm (A) and the experimental retry arm (B).
# The default MUST stay arm A -- only the A/B script passes retry_on_empty=True.
# --------------------------------------------------------------------------- #
def test_arm_a_is_the_default_first_empty_turn_finalises() -> None:
    client = _ScriptedClient([
        _resp(answer="Action: list_people[]"),
        _resp(thinking="thinking with nothing after it", answer=""),
        _resp(answer='{"meetings": []}'),  # terminal emit
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert client.i == 3  # one tool step, one empty turn, then straight to the emit
    assert [c.role for c in result.calls] == ["react_step", "react_step", "finalize"]
    assert result.empty_turns == 1
    assert result.termination == "aborted"


def test_arm_b_retries_once_after_an_empty_turn() -> None:
    # Same script as arm A, except the retry is granted: the agent gets one more
    # decide step, uses it, and the run reaches a plan the locked arm never sees.
    client = _ScriptedClient([
        _resp(answer="Action: list_people[]"),
        _resp(thinking="thinking with nothing after it", answer=""),
        _resp(answer="Action: propose[p0@10, p1@60]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": [{"person_id": "p0", "start_time": 10}, '
                     '{"person_id": "p1", "start_time": 60}]}'),
    ])
    result = run_react(_instance(), client, cap=8000, retry_on_empty=True)  # type: ignore[arg-type]

    assert client.i == 5
    assert result.empty_turns == 1
    assert result.termination == "agent_finish"
    assert len(result.best_plan_so_far.meetings) == 2
    # The retry nudge is a plain Observation -- no validity, no budget, no oracle terms.
    nudges = [m for m in result.transcript if "no answer" in m["content"]]
    assert len(nudges) == 1
    assert "Action: finish" in nudges[0]["content"]


def test_arm_b_stops_on_two_consecutive_empty_turns() -> None:
    client = _ScriptedClient([
        _resp(thinking="nothing after this", answer=""),
        _resp(thinking="nothing after this either", answer=""),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000, retry_on_empty=True)  # type: ignore[arg-type]

    assert client.i == 3  # two empty decide steps, then the emit -- no third retry
    assert result.empty_turns == 2
    assert result.termination == "aborted"


def test_arm_b_streak_resets_after_a_useful_turn() -> None:
    # empty -> retry -> useful turn -> empty again: the second empty is a FIRST
    # consecutive empty, so it is retried too. Only back-to-back empties stop the run.
    client = _ScriptedClient([
        _resp(thinking="x", answer=""),
        _resp(answer="Action: list_people[]"),
        _resp(thinking="y", answer=""),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000, retry_on_empty=True)  # type: ignore[arg-type]

    assert client.i == 5
    assert result.empty_turns == 2
    assert result.termination == "agent_finish"


def test_termination_reason_distinguishes_budget_from_finish() -> None:
    finished = _ScriptedClient([
        _resp(answer="Action: finish"), _resp(answer='{"meetings": []}'),
    ])
    assert run_react(_instance(), finished, cap=8000).termination == "agent_finish"  # type: ignore[arg-type]

    # A cap that only affords the terminal emit: the guard routes before any decide step.
    starved = _ScriptedClient([_resp(answer='{"meetings": []}')])
    starved_result = run_react(_instance(), starved, cap=300)  # type: ignore[arg-type]
    assert starved_result.termination == "budget"
    assert starved_result.empty_turns == 0


# --------------------------------------------------------------------------- #
# Rejected-proposal taxonomy: one row per propose, written when it is handled.
# Diagnostics only -- the score and the agent's view must not change.
# --------------------------------------------------------------------------- #
def _rows(result) -> list[dict]:
    return result.proposals


def test_taxonomy_records_an_unparseable_proposal() -> None:
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10, p1@60+T2]"),   # symbolic offset, not a number
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    row = _rows(result)[0]
    assert row["schema_version"] == PROPOSAL_SCHEMA_VERSION
    assert row["condition"] == "c1_react" and row["role"] == "react_step"
    assert row["parsed"] is False and row["plan"] is None
    assert row["n_meetings"] == 0
    assert row["valid"] is False and row["reasons"] == ["malformed"]
    assert row["accepted_into_best_plan"] is False
    assert "p1@60+T2" in row["raw"]
    # The scored artefact is untouched by the diagnostics.
    assert result.best_plan_so_far.meetings == []


def test_taxonomy_records_a_travel_infeasible_proposal() -> None:
    # p0 is at A and p1 at B; p0 runs 10-40, A->B costs 20, so p1 cannot start before
    # 60. Starting it at 45 leaves the meetings disjoint but the travel impossible --
    # the failure mode that dominates the live 32B runs.
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10, p1@45]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    row = _rows(result)[0]
    assert row["parsed"] is True
    assert row["n_meetings"] == 2
    assert row["valid"] is False
    assert InvalidReason.TRAVEL_INFEASIBLE.value in row["reasons"]
    assert row["accepted_into_best_plan"] is False
    assert row["plan"] == [{"person_id": "p0", "start_time": 10},
                           {"person_id": "p1", "start_time": 45}]
    assert result.best_plan_so_far.meetings == []


def test_taxonomy_records_a_valid_but_not_longer_proposal() -> None:
    # First a valid 2-meeting plan is accepted; then a valid 1-meeting plan arrives.
    # It is valid, but not strictly longer, so the hidden gate refuses it.
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10, p1@60]"),
        _resp(answer="Action: propose[p0@10]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": [{"person_id": "p0", "start_time": 10}, '
                     '{"person_id": "p1", "start_time": 60}]}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    first, second = _rows(result)[0], _rows(result)[1]
    assert first["valid"] is True and first["accepted_into_best_plan"] is True
    assert second["valid"] is True and second["reasons"] == []
    assert second["n_meetings"] == 1
    assert second["accepted_into_best_plan"] is False  # valid, but not an improvement
    # best_plan_so_far still holds the longer plan.
    assert len(result.best_plan_so_far.meetings) == 2


def test_taxonomy_records_a_valid_accepted_proposal() -> None:
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@10, p1@60]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": [{"person_id": "p0", "start_time": 10}, '
                     '{"person_id": "p1", "start_time": 60}]}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    row = _rows(result)[0]
    assert row["valid"] is True
    assert row["reasons"] == []
    assert row["n_meetings"] == 2
    assert row["accepted_into_best_plan"] is True
    assert row["step"] == 1
    # Nothing about validity reached the agent: the Observation echoes only the count.
    obs = [m["content"] for m in result.transcript if m["content"].startswith("Observation")]
    assert any("Recorded a plan with 2 meeting(s)." in o for o in obs)
    assert not any("valid" in o.lower() or "infeasible" in o.lower() for o in obs)


def test_taxonomy_reasons_are_multi_label() -> None:
    # p0 runs 75-105 but its window closes at 100 -> WINDOW_VIOLATION; p1 then starts
    # at 110 while arrival is only possible at 125 (105 + 20 travel) -> TRAVEL_INFEASIBLE.
    # Both must be reported: collapsing to the first would bias the breakdown toward
    # whichever check the validator happens to run earlier.
    client = _ScriptedClient([
        _resp(answer="Action: propose[p0@75, p1@110]"),
        _resp(answer="Action: finish"),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    row = _rows(result)[0]
    assert row["parsed"] is True
    assert row["valid"] is False
    assert row["reasons"] == [InvalidReason.TRAVEL_INFEASIBLE.value,
                              InvalidReason.WINDOW_VIOLATION.value]
    assert row["accepted_into_best_plan"] is False


def test_finish_reason_is_carried_into_the_call_record() -> None:
    # The empty-post-think rule cannot tell a generation cut short by the grant from a
    # model that closed </think> and said nothing -- the token counts are identical.
    # finish_reason is the only thing that separates them, so it must survive the hop
    # from the client response into the run record.
    client = _ScriptedClient([
        LLMResponse(thinking="ran out of room", answer="", raw_text="<think>ran out",
                    input_tokens=50, thinking_tokens=10, answer_tokens=0,
                    finish_reason="length"),
        _resp(answer='{"meetings": []}'),
    ])
    result = run_react(_instance(), client, cap=8000)  # type: ignore[arg-type]

    assert [c.finish_reason for c in result.calls] == ["length", "stop"]
    # The rule itself is unchanged: an empty visible answer still aborts.
    assert result.termination == "aborted"
    assert result.empty_turns == 1
