# src/agents/single_agent/react_verify_revise.py
"""C2 -- ReAct + mandated verify/revise (condition 2), as a fixed LangGraph workflow.

C2 is C1 with a fixed reflection structure appended (THESIS_DECISIONS section 4):

    Draft -> Verify -> Revise -> Verify -> Revise -> Stop    (max 2 revisions)

It isolates the effect of *mandated* verification against the C1 baseline. The draft
phase is the C1 loop, reused unchanged from `src.agents.react_core` (agent <-> tool,
then the same terminal finalisation). Only two condition-specific nodes are added here
-- `verify` and `revise` -- plus the routing that runs them at most twice; nothing in
this file modifies the shared core, so C1's behaviour is provably untouched.

Design (locked):
  * One agent, one context. Verify/Revise are further turns in the SAME conversation,
    so they see the whole draft trajectory (the tool observations gathered there). This
    re-encoded context is a real cost and is counted in the shared budget like any input.
  * Verify/Revise are reflection only -- they do NOT call tools. The draft does the
    search; the reflection reasons over what the draft already gathered. This keeps the
    structure fixed and the contrast with C1 a pure "verification effect".
  * Fixed structure: the cycle always runs (V then R) up to MAX_REVISIONS; the only
    early exit is budget exhaustion. No prompt-dependent "looks good -> stop" branch
    (that would be a control-flow confound between conditions).
  * No oracle/validator leakage: the verify/revise prompts reference only the agent's
    own plan and the task constraints it was already given. The hidden best_plan_so_far
    bookkeeping (`is_valid`) stays harness-side, exactly as in the shared `agent_step`.

Budget discipline is the section-2 invariant, identical to every other condition: each
reasoning call (draft / verify / revise) is guarded by `should_finalize` before the call
and capped by `max_new_tokens` so the finalisation reserve is never touched; empty
post-think content routes straight to finalisation.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.react_core import (
    LoopContext,
    ReactResult,
    ReactState,
    _parse_action,
    _plan_from_args,
    agent_step,
    finalize_step,
    initial_state,
    tool_step,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger, LLMClient
from src.oracle import is_valid
from src.schemas import CallRecord, Instance

__all__ = ["ReactResult", "run_verify_revise", "MAX_REVISIONS"]

# Fixed by THESIS_DECISIONS section 4: Draft -> V -> R -> V -> R -> Stop.
MAX_REVISIONS = 2

_VERIFY_INSTRUCTION = (
    "Verification step. Review the plan you have proposed so far against the task "
    "constraints: every meeting must fit the person's availability window, you must "
    "have time to travel between consecutive meetings, and every meeting must end by "
    "the end of the day. List concretely any problem you find -- an infeasible "
    "meeting, a missing travel margin, or a person you could still add. Do not call "
    "any tools. If the plan is already the best you can justify, say so and why."
)

_REVISE_INSTRUCTION = (
    "Revision step. Based on your review, output your improved plan on a single line "
    "as 'Action: propose[p0@10, p1@60]' (person_id @ meeting-start-minute), in visiting "
    "order. Propose the FULL plan, not just the change. If no change is warranted, "
    "propose the same plan again. Do not call any tools."
)


# --------------------------------------------------------------------------- #
# C2 state: the shared loop state plus the revision counter and a stop flag.
# --------------------------------------------------------------------------- #
class ReactStateC2(ReactState):
    revision_count: int
    vr_done: bool  # the verify/revise phase has decided to stop -> route to finalize


def _initial_state(instance: Instance) -> ReactStateC2:
    state: dict[str, Any] = dict(initial_state(instance))
    state["revision_count"] = 0
    state["vr_done"] = False
    return state  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Condition-specific nodes (the section-2 budget guard, mirrored from agent_step).
# --------------------------------------------------------------------------- #
def _guarded_call(
    state: ReactStateC2, ctx: LoopContext, instruction: str, role: str
) -> tuple[str | None, dict[str, Any]]:
    """Run one reflection call under the shared budget guard.

    Returns `(visible, updates)`. `visible` is None when no call could be made (budget
    too small) or the model returned empty post-think content; in both cases `updates`
    already carries `vr_done=True` so the caller just routes to finalisation. Otherwise
    `updates` holds the appended turns and the booked CallRecord, and the caller parses
    `visible`.
    """
    client, ledger = ctx.client, ctx.ledger
    msgs = state["messages"] + [{"role": "user", "content": instruction}]
    input_tokens = client.count_input(msgs, enable_thinking=True)
    if ledger.should_finalize(input_tokens):
        return None, {"vr_done": True, "via_budget": True}

    max_new = ledger.max_new_tokens(input_tokens, finalizing=False)
    t0 = time.perf_counter()
    resp = client.complete(msgs, max_tokens=max_new, enable_thinking=True)
    latency = time.perf_counter() - t0
    ledger.record_call(
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
    )
    call = CallRecord(
        role=role,
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
        latency_seconds=latency,
    )
    # Post-think answer only: raw_text may hold a truncated <think> block and must
    # never be parsed nor re-fed into the conversation (same rule as agent_step).
    visible = resp.answer
    updates: dict[str, Any] = {
        "messages": [{"role": "user", "content": instruction},
                     {"role": "assistant", "content": visible}],
        "calls": [call],
    }
    if not visible.strip():
        # Empty post-think -> finalise immediately (a retry has the same budget).
        updates["vr_done"] = True
        return None, updates
    return visible, updates


def verify_step(state: ReactStateC2, ctx: LoopContext) -> dict[str, Any]:
    """Self-critique of the current plan (thinking ON). No tools, no validator."""
    _, updates = _guarded_call(state, ctx, _VERIFY_INSTRUCTION, role="verify")
    return updates


def revise_step(state: ReactStateC2, ctx: LoopContext) -> dict[str, Any]:
    """Re-propose an improved plan (thinking ON). Counts one revision."""
    visible, updates = _guarded_call(state, ctx, _REVISE_INSTRUCTION, role="revise")
    updates["revision_count"] = state["revision_count"] + 1
    if visible is None:
        return updates  # budget/empty already set vr_done

    action = _parse_action(visible)
    if action is not None and action["tool"] == "propose":
        plan = _plan_from_args(action["args"])
        # Same hidden bookkeeping as the draft: keep the best VALID plan; the agent is
        # told nothing about validity (no oracle/validator leakage).
        if (plan is not None and is_valid(ctx.instance, plan)
                and len(plan.meetings) > len(state["best_plan"].meetings)):
            updates["best_plan"] = plan
    elif action is not None and action["tool"] == "finish":
        updates["vr_done"] = True
    return updates


# --------------------------------------------------------------------------- #
# Routing (C2 wires its own; the shared core is condition-agnostic).
# --------------------------------------------------------------------------- #
def route_after_agent(state: ReactStateC2) -> str:
    """Draft routing: tool/loop as in C1, but a draft that finished NORMALLY enters
    verification. A draft that stopped on budget, or aborted (empty post-think
    content / a second format failure after the one permitted retry), goes straight
    to finalisation -- the section-2 rule is identical on every condition, so C2 may
    not convert an aborted draft into extra verify/revise turns (a hidden retry
    bonus C1 does not get)."""
    if state.get("finalize"):
        if state.get("via_budget") or state.get("aborted"):
            return "finalize"
        return "verify"
    if state.get("pending_tool"):
        return "tool"
    return "agent"


def route_after_verify(state: ReactStateC2) -> str:
    return "finalize" if state.get("vr_done") else "revise"


def route_after_revise(state: ReactStateC2) -> str:
    if state.get("vr_done") or state["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    return "verify"


def run_verify_revise(
    instance: Instance,
    client: LLMClient,
    *,
    cap: int,
    max_steps: int = 12,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
) -> ReactResult:
    """Run C2 on one instance under a token `cap`. Returns the run artifacts."""
    ledger = BudgetLedger(cap=cap, finalization_reserve=finalization_reserve)
    ctx = LoopContext(instance=instance, client=client, ledger=ledger, max_steps=max_steps)

    graph = StateGraph(ReactStateC2)
    graph.add_node("agent", lambda s: agent_step(s, ctx))
    graph.add_node("tool", lambda s: tool_step(s, ctx))
    graph.add_node("verify", lambda s: verify_step(s, ctx))
    graph.add_node("revise", lambda s: revise_step(s, ctx))
    graph.add_node("finalize", lambda s: finalize_step(s, ctx))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent,
        {"agent": "agent", "tool": "tool", "verify": "verify", "finalize": "finalize"},
    )
    graph.add_edge("tool", "agent")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"revise": "revise", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "revise", route_after_revise, {"verify": "verify", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    app = graph.compile()

    final_state = app.invoke(
        _initial_state(instance), config={"recursion_limit": max_steps * 2 + 20}
    )

    return ReactResult(
        final_plan=final_state["final_plan"],
        best_plan_so_far=final_state["best_plan"],
        tokens=ledger.to_token_usage(
            budget_exhausted=final_state.get("via_budget", False) or ledger.budget_exhausted
        ),
        calls=final_state["calls"],
        transcript=final_state["messages"],
        n_steps=final_state["step"],
        finalization_mismatch=final_state.get("finalization_mismatch", False),
    )
