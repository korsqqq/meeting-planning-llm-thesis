# src/agents/react_core.py
"""Shared ReAct loop core: the pieces C1 uses that the other conditions reuse.

Only what C1 already uses lives here -- the Thought/Action/Observation parsers, the
tool dispatch, the budget-guarded `agent` / `tool` decide+act steps, and the §2
finalisation node (thinking OFF, guided JSON, a SHORT serialisation of
`best_plan_so_far`, never the trajectory). Condition-specific nodes (C2's verify/revise,
C3's supervisor/worker/aggregator/critic) do NOT belong here -- this module grows only
when a piece is reused by >= 2 conditions.

The nodes are plain functions of `(state, ctx)`; each condition wires its own graph over
them (the wiring differs per condition, so it is not shared). The budget discipline
(THESIS_DECISIONS section 2) is identical across conditions and lives in `BudgetLedger`:
every reasoning call is capped at `max_new_tokens = max(0, remaining - reserve - input)`,
so `remaining >= reserve` always holds and the short finalisation emit always fits. The
ledger also carries an OPTIONAL per-call ceiling (`max_call_tokens`, off by default --
see `src/core/budget.py` for why it cannot be switched on as it stands).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Any, TypedDict

from src.agents.tools import get_availability, get_travel_time, list_people
from src.core import BudgetLedger, LLMClient
from src.oracle import validate
from src.schemas import AnswerContract, CallRecord, Instance, Meeting, TokenUsage

__all__ = [
    "ANSWER_GUIDED_SCHEMA",
    "FINALIZE_INSTRUCTION",
    "ReactState",
    "ReactResult",
    "LoopContext",
    "agent_step",
    "tool_step",
    "finalize_step",
    "route_after_agent",
    "initial_state",
    "termination_reason",
    "proposal_record",
]

# Minimal schema for the terminal emit (vLLM `guided_json`; the prompt guides Ollama).
ANSWER_GUIDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meetings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "string"},
                    "start_time": {"type": "integer"},
                },
                "required": ["person_id", "start_time"],
            },
        }
    },
    "required": ["meetings"],
}

FINALIZE_INSTRUCTION = (
    "Output the final plan as a JSON object and nothing else, exactly like:\n"
    '{"meetings": [{"person_id": "<id>", "start_time": <integer minute>}, ...]}\n'
    "List the meetings in visiting order. If no meeting is possible, output "
    '{"meetings": []}.'
)

_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]\w*)\s*\[(.*?)\]", re.DOTALL)
_FINISH_RE = re.compile(r"Action:\s*finish\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Pure helpers (parsing, tool dispatch, prompt building).
# --------------------------------------------------------------------------- #
def _parse_action(text: str) -> dict[str, Any] | None:
    """Extract one `Action: tool[args]` (or `Action: finish`) from the model text."""
    match = _ACTION_RE.search(text)
    if match:
        tool = match.group(1)
        arg_str = match.group(2).strip()
        # Strip surrounding quotes the model often adds: get_travel_time['start', 'loc_0'].
        args = [a.strip().strip("'\"") for a in arg_str.split(",")] if arg_str else []
        if tool == "finish":
            return {"tool": "finish", "args": [], "raw": arg_str}
        # `raw` is the argument text exactly as written, kept so a rejected proposal can
        # be reported as the agent phrased it (diagnostics only; nothing parses it).
        return {"tool": tool, "args": args, "raw": arg_str}
    if _FINISH_RE.search(text):
        return {"tool": "finish", "args": [], "raw": ""}
    return None


def _execute_tool(instance: Instance, action: dict[str, Any]) -> str:
    """Run a parsed tool call and return a text Observation (errors are text, not raises)."""
    tool, args = action["tool"], action["args"]
    try:
        if tool == "list_people":
            return "People: " + ", ".join(list_people(instance))
        if tool == "get_availability":
            if len(args) != 1:
                return "Error: get_availability expects 1 argument: person_id."
            return f"availability({args[0]}) = {get_availability(instance, args[0])}"
        if tool == "get_travel_time":
            if len(args) != 2:
                return "Error: get_travel_time expects 2 arguments: from_location, to_location."
            minutes = get_travel_time(instance, args[0], args[1])
            return f"travel_time({args[0]} -> {args[1]}) = {minutes} min"
        return (
            f"Error: unknown tool '{tool}'. Available: list_people, "
            "get_availability, get_travel_time."
        )
    except KeyError as exc:
        return f"Error: {exc}"


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def _parse_answer_plan(text: str) -> AnswerContract | None:
    """Parse the terminal emit into an AnswerContract, or None on malformed output."""
    try:
        data = json.loads(_extract_json(text))
        meetings = [
            Meeting(person_id=str(m["person_id"]), start_time=int(m["start_time"]))
            for m in data.get("meetings", [])
        ]
        return AnswerContract(meetings=meetings)
    except Exception:
        return None


def _initial_messages(instance: Instance) -> list[dict[str, str]]:
    if instance.waiting_allowed:
        waiting = "You may arrive early and wait for a window to open."
    else:
        waiting = "You cannot wait: a meeting must start exactly when you arrive."
    system = (
        "You are a meeting-planning agent. Plan a one-day route that meets as many "
        "people as possible.\n"
        f"You start at location '{instance.start_location}' at minute {instance.start_time}. "
        f"Every meeting lasts exactly {instance.meeting_duration} minutes and must fit "
        "inside the person's availability window. You must have time to travel between "
        f"consecutive meetings, and every meeting must end by minute {instance.end_of_day}. "
        f"{waiting}\n\n"
        "Work step by step. Each step write one 'Thought:' line, then exactly one of:\n"
        "  Action: list_people[]\n"
        "  Action: get_availability[person_id]\n"
        "  Action: get_travel_time[from_location, to_location]\n"
        "You will receive an 'Observation:' with the result. Record your plan (even a "
        "partial one) at any time with:\n"
        "  Action: propose[p0@10, p1@60]   (person_id @ meeting-start-minute)\n"
        "You may refine and propose again. When you are done, write 'Action: finish'."
    )
    user = "Plan the route now. Start by listing the people."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


PROPOSAL_SCHEMA_VERSION = "proposal/1.0"


def proposal_record(
    instance: Instance,
    plan: AnswerContract | None,
    *,
    step: int,
    role: str,
    condition: str,
    raw: str,
    best_len: int,
) -> dict[str, Any]:
    """One row of the rejected-proposal taxonomy (§5 secondary metric, §6).

    Written at the moment the propose is handled, where the parse result and the
    validator verdict are both already in hand -- never by re-parsing a stored
    transcript, which would depend on the transcript's formatting and could drift
    from what the harness actually did.

    Measured on what the agent PROPOSED, not on the scored plan: `best_plan_so_far`
    only ever accepts validator-approved plans and the scored `final_plan` equals it,
    so the scored artefact is valid or empty by construction and carries no error
    information. Every propose the hidden gate refused does.

    Harness-side only: the row is never shown to the agent, exactly like the gate
    itself, and nothing in the loop routes on it. `plan is None` means the arguments
    did not parse into meetings at all.

    Row shape (`PROPOSAL_SCHEMA_VERSION`, stable -- fields are added, never renamed):
      schema_version, condition, role, step, raw, parsed, plan, n_meetings, valid,
      reasons (multi-label, empty iff valid), accepted_into_best_plan.
    """
    parsed = plan is not None
    reasons = (
        ["malformed"] if not parsed
        else [str(getattr(r, "value", r)) for r in validate(instance, plan)]
    )
    n_meetings = len(plan.meetings) if parsed else 0
    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "condition": condition,
        "role": role,
        "step": step,
        "raw": raw,
        "parsed": parsed,
        "plan": (
            [{"person_id": m.person_id, "start_time": m.start_time} for m in plan.meetings]
            if parsed else None
        ),
        "n_meetings": n_meetings,
        "valid": parsed and not reasons,
        # Multi-label on purpose: one plan can break several hard constraints at once,
        # and collapsing them to the first would bias the breakdown toward whichever
        # check the validator happens to run first.
        "reasons": reasons,
        # Whether this proposal actually became best_plan_so_far: valid AND strictly
        # longer than what was held, the same gate the scoring path applies.
        "accepted_into_best_plan": parsed and not reasons and n_meetings > best_len,
    }


def _plan_from_args(args: list[str]) -> AnswerContract | None:
    """Parse a proposed plan written as `p0@10, p1@60` into an AnswerContract."""
    meetings: list[Meeting] = []
    for token in args:
        person, sep, start = token.partition("@")
        person, start = person.strip(), start.strip()
        if not sep or not person or not start.lstrip("-").isdigit():
            return None
        meetings.append(Meeting(person_id=person, start_time=int(start)))
    return AnswerContract(meetings=meetings)


# --------------------------------------------------------------------------- #
# Shared graph state, result, and per-run context.
# --------------------------------------------------------------------------- #
class ReactState(TypedDict):
    messages: Annotated[list[dict[str, str]], add]
    calls: Annotated[list[CallRecord], add]
    # Every propose the agent made, with the validator's verdict. Harness-side only:
    # this is the rejected-proposal taxonomy, never anything the agent can see.
    proposals: Annotated[list[dict[str, Any]], add]
    pending_tool: dict[str, Any] | None
    finalize: bool
    via_budget: bool
    # The loop ended abnormally: empty post-think content, or a second consecutive
    # format failure after the single permitted retry. Every condition must route an
    # aborted state straight to finalisation (THESIS_DECISIONS section 2) -- no
    # condition may spend further reasoning calls on it.
    aborted: bool
    final_plan: AnswerContract | None
    best_plan: AnswerContract
    step: int
    no_action_streak: int
    # Consecutive empty post-think turns, and the run total. Only the EXPERIMENTAL
    # `retry_on_empty` treatment reads the streak; `empty_turns` is diagnostic in both
    # arms, so the locked behaviour can be reported on the same footing.
    empty_streak: int
    empty_turns: int
    # True iff the terminal emit parsed but did NOT round-trip best_plan_so_far (the plan
    # was then discarded in favour of the structural best). Integrity diagnostic only.
    finalization_mismatch: bool


@dataclass(frozen=True)
class ReactResult:
    """Artifacts of one run (scoring happens separately, Layer 5)."""

    final_plan: AnswerContract | None
    best_plan_so_far: AnswerContract
    tokens: TokenUsage
    calls: list[CallRecord]
    transcript: list[dict[str, str]]
    n_steps: int
    finalization_mismatch: bool = False
    # Diagnostics, both arms of the empty-turn A/B (section 2). `empty_turns` counts
    # post-think turns that carried no content; `termination` says what ended the loop.
    empty_turns: int = 0
    termination: str = "unknown"
    # Rejected-proposal taxonomy: one row per propose, with the validator's reasons.
    proposals: list[dict[str, Any]] = field(default_factory=list)
    # Condition-specific, LOG-ONLY diagnostics, written straight into the result document
    # and read by nothing in any loop. Empty for every condition that has none. It exists
    # so a condition can record what its own structure produced -- C5's three trajectory
    # products and the winning attempt, C4's pool and evidence coverage -- without either
    # inventing a per-condition result type or teaching this core about any condition.
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopContext:
    """Per-run handles the nodes need (closed over by each condition's graph)."""

    instance: Instance
    client: LLMClient
    ledger: BudgetLedger
    max_steps: int
    # Condition label stamped onto every proposal row, so the taxonomy can be split by
    # condition without joining against the run record. Diagnostics only.
    condition: str = ""
    # EXPERIMENTAL, default OFF -- the locked section-2 behaviour is "first empty
    # post-think turn finalises immediately". True grants exactly one retry, the same
    # single retry a non-empty formatting slip already gets; a second CONSECUTIVE empty
    # turn finalises as before. Only `scripts/run_empty_retry_ab.py` sets this.
    retry_on_empty: bool = False


def initial_state(instance: Instance) -> ReactState:
    return {
        "messages": _initial_messages(instance),
        "calls": [],
        "proposals": [],
        "pending_tool": None,
        "finalize": False,
        "via_budget": False,
        "aborted": False,
        "final_plan": None,
        "best_plan": AnswerContract.empty(),
        "step": 0,
        "no_action_streak": 0,
        "empty_streak": 0,
        "empty_turns": 0,
        "finalization_mismatch": False,
    }


def termination_reason(state: ReactState, max_steps: int) -> str:
    """Why the loop stopped: 'aborted' | 'budget' | 'max_steps' | 'agent_finish'.

    Diagnostic only -- nothing routes on it. The order matters: an aborted loop may
    also happen to be out of budget, and the abort is the more specific statement.
    """
    if state.get("aborted"):
        return "aborted"
    if state.get("via_budget"):
        return "budget"
    if state.get("step", 0) >= max_steps:
        return "max_steps"
    return "agent_finish"


# --------------------------------------------------------------------------- #
# Shared nodes (plain functions of (state, ctx); each condition wires its graph).
# --------------------------------------------------------------------------- #
def agent_step(state: ReactState, ctx: LoopContext) -> dict[str, Any]:
    """Decide step (thinking ON): one model call, parse one Action."""
    client, ledger, instance = ctx.client, ctx.ledger, ctx.instance
    input_tokens = client.count_input(state["messages"], enable_thinking=True)
    if state["step"] >= ctx.max_steps or ledger.should_finalize(input_tokens):
        return {"finalize": True, "via_budget": ledger.should_finalize(input_tokens)}

    max_new = ledger.max_new_tokens(input_tokens, finalizing=False)
    t0 = time.perf_counter()
    resp = client.complete(state["messages"], max_tokens=max_new, enable_thinking=True)
    latency = time.perf_counter() - t0
    ledger.record_call(
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
    )
    call = CallRecord(
        role="react_step",
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
        latency_seconds=latency,
        finish_reason=resp.finish_reason,
        requested_max_tokens=resp.requested_max_tokens,
        effective_max_tokens=resp.effective_max_tokens,
        context_limited=resp.context_limited,
    )
    # Only the post-think answer is ever visible to the loop. raw_text may hold a
    # truncated <think> block (the grant was spent inside reasoning); it must never
    # be parsed for Actions nor re-fed into the conversation history (Qwen3 guidance).
    visible = resp.answer
    action = _parse_action(visible)
    updates: dict[str, Any] = {"calls": [call], "step": state["step"] + 1}

    if action is None:
        if not visible.strip():
            # Empty post-think content (nothing after </think>, or nothing at all).
            # LOCKED default: finalise now, while the most budget remains for the
            # terminal emit's own input -- retrying only re-encodes context and drains
            # the budget. `aborted` makes every condition route straight to finalisation.
            empty_streak = state.get("empty_streak", 0) + 1
            updates["empty_turns"] = state.get("empty_turns", 0) + 1
            if ctx.retry_on_empty and empty_streak < 2:
                # EXPERIMENTAL arm only: one retry, exactly as a non-empty formatting
                # slip gets. A second CONSECUTIVE empty turn falls through and aborts.
                updates["messages"] = [
                    {"role": "assistant", "content": visible},
                    {"role": "user", "content": "Observation: Your last turn contained "
                     "no answer. Write one line 'Action: <tool>[args]' or "
                     "'Action: finish'."},
                ]
                updates["empty_streak"] = empty_streak
            else:
                updates["messages"] = [{"role": "assistant", "content": visible}]
                updates["finalize"] = True
                updates["aborted"] = True
        else:
            streak = state.get("no_action_streak", 0) + 1
            if streak >= 2:  # one retry for a non-empty formatting slip, then stop
                updates["messages"] = [{"role": "assistant", "content": visible}]
                updates["finalize"] = True
                updates["aborted"] = True
            else:
                updates["messages"] = [
                    {"role": "assistant", "content": visible},
                    {"role": "user", "content": "Observation: No valid Action found. Write "
                     "one line 'Action: <tool>[args]' or 'Action: finish'."},
                ]
                updates["no_action_streak"] = streak
                updates["empty_streak"] = 0  # a non-empty turn breaks the empty chain
    elif action["tool"] == "propose":
        updates["no_action_streak"] = 0
        updates["empty_streak"] = 0
        plan = _plan_from_args(action["args"])
        record = proposal_record(
            instance, plan, step=state["step"] + 1, role="react_step",
            condition=ctx.condition, raw=action.get("raw", ""),
            best_len=len(state["best_plan"].meetings),
        )
        updates["proposals"] = [record]
        if plan is None:
            obs = "Could not parse the plan. Use 'Action: propose[p0@10, p1@60]'."
        else:
            # Keep the best VALID plan as best_plan_so_far. This is hidden harness
            # bookkeeping: the agent is told nothing about validity (no oracle/validator
            # leakage) -- the Observation only echoes the count it proposed.
            if record["accepted_into_best_plan"]:
                updates["best_plan"] = plan
            obs = f"Recorded a plan with {len(plan.meetings)} meeting(s)."
        updates["messages"] = [
            {"role": "assistant", "content": visible},
            {"role": "user", "content": f"Observation: {obs}"},
        ]
    elif action["tool"] == "finish":
        updates["no_action_streak"] = 0
        updates["empty_streak"] = 0
        if action["args"]:  # finish may carry an inline plan
            plan = _plan_from_args(action["args"])
            record = proposal_record(
                instance, plan, step=state["step"] + 1, role="finish",
                condition=ctx.condition, raw=action.get("raw", ""),
                best_len=len(state["best_plan"].meetings),
            )
            updates["proposals"] = [record]
            if record["accepted_into_best_plan"]:
                updates["best_plan"] = plan
        updates["messages"] = [{"role": "assistant", "content": visible}]
        updates["finalize"] = True
    else:
        updates["messages"] = [{"role": "assistant", "content": visible}]
        updates["no_action_streak"] = 0
        updates["empty_streak"] = 0
        updates["pending_tool"] = action
    return updates


def tool_step(state: ReactState, ctx: LoopContext) -> dict[str, Any]:
    """Run the parsed tool and append an Observation."""
    observation = _execute_tool(ctx.instance, state["pending_tool"])
    return {
        "messages": [{"role": "user", "content": f"Observation: {observation}"}],
        "pending_tool": None,
    }


def finalize_step(state: ReactState, ctx: LoopContext) -> dict[str, Any]:
    """Terminal serialiser (thinking OFF + guided JSON) of best_plan_so_far."""
    client, ledger = ctx.client, ctx.ledger
    # Stateless serialisation of best_plan_so_far: a SHORT prompt (the decided plan as
    # text + the schema), NOT the trajectory. This is what makes the 256 reserve
    # sufficient (THESIS_DECISIONS section 2). The per-call cap has kept remaining >=
    # reserve, so this short emit always fits.
    best = state.get("best_plan") or AnswerContract.empty()
    plan_text = ", ".join(f"{m.person_id}@{m.start_time}" for m in best.meetings) or "(none)"
    fmsgs = [
        {"role": "system", "content": "Serialise the given meeting plan as JSON. Do not change it."},
        {"role": "user", "content": f"Plan: {plan_text}\n{FINALIZE_INSTRUCTION}"},
    ]
    input_tokens = client.count_input(fmsgs, enable_thinking=False)
    max_new = ledger.max_new_tokens(input_tokens, finalizing=True)

    emitted: AnswerContract | None = None
    calls_update: list[CallRecord] = []
    if max_new > 0:
        t0 = time.perf_counter()
        resp = client.complete(
            fmsgs, max_tokens=max_new, enable_thinking=False,
            guided_json=ANSWER_GUIDED_SCHEMA,
        )
        latency = time.perf_counter() - t0
        ledger.record_call(
            input_tokens=resp.input_tokens,
            thinking_tokens=resp.thinking_tokens,
            answer_tokens=resp.answer_tokens,
        )
        calls_update = [CallRecord(
            role="finalize",
            input_tokens=resp.input_tokens,
            thinking_tokens=resp.thinking_tokens,
            answer_tokens=resp.answer_tokens,
            latency_seconds=latency,
            finish_reason=resp.finish_reason,
            requested_max_tokens=resp.requested_max_tokens,
            effective_max_tokens=resp.effective_max_tokens,
            context_limited=resp.context_limited,
        )]
        # thinking is OFF here, so the whole completion is the answer; raw_text is
        # never consulted (it could only ever add a stray <think> block).
        emitted = _parse_answer_plan(resp.answer)

    # Faithful serialisation, ENFORCED (not merely intended). The emit is accepted only when
    # it round-trips `best_plan_so_far` exactly; any other parseable JSON is discarded in
    # favour of the structural plan. Guided decoding constrains the SHAPE of the emit, not its
    # VALUES, so without this check a model that changed a start time, dropped or added a
    # person, or reordered the route would have its altered plan scored -- silently breaking
    # the section-2 invariant that finalise is a pure serialisation of best_plan_so_far.
    # The mismatch is not repaired and not retried; it is recorded so the pilot can report a
    # mismatch rate as an integrity check on the terminal emit (THESIS_DECISIONS section 7).
    mismatch = emitted is not None and emitted != best
    final = emitted if emitted is not None and not mismatch else best
    return {
        "calls": calls_update,
        "final_plan": final,
        "best_plan": best,
        "finalization_mismatch": mismatch,
    }


def route_after_agent(state: ReactState) -> str:
    if state.get("finalize"):
        return "finalize"
    if state.get("pending_tool"):
        return "tool"
    return "agent"
