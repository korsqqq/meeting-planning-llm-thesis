# src/agents/single_agent/react.py
"""C1 -- the single ReAct agent (condition 1), as a fixed LangGraph workflow.

The loop primitives live in `src.agents.react_core` (shared with the other conditions);
C1 just wires them: START -> agent -> {tool -> agent | agent | finalize}; finalize -> END.
One agent solves the whole task in one context (Yao et al., 2023); the budget guard and
the §2 finalisation discipline are inherited unchanged from the shared core.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.react_core import (
    LoopContext,
    ReactResult,
    ReactState,
    agent_step,
    finalize_step,
    initial_state,
    route_after_agent,
    termination_reason,
    tool_step,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger, LLMClient
from src.schemas import Instance

# Re-exported so existing tests can import the pure helpers from here unchanged.
from src.agents.react_core import (  # noqa: E402,F401  (re-export for tests)
    _execute_tool,
    _parse_action,
    _parse_answer_plan,
    _plan_from_args,
)

__all__ = ["ReactResult", "run_react"]


def run_react(
    instance: Instance,
    client: LLMClient,
    *,
    cap: int,
    max_steps: int = 12,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
    retry_on_empty: bool = False,
) -> ReactResult:
    """Run C1 on one instance under a token `cap`. Returns the run artifacts.

    `retry_on_empty` is the EXPERIMENTAL arm of the empty-turn A/B and defaults to the
    locked behaviour (False). See `LoopContext.retry_on_empty`.
    """
    ledger = BudgetLedger(cap=cap, finalization_reserve=finalization_reserve)
    ctx = LoopContext(
        instance=instance, client=client, ledger=ledger, max_steps=max_steps,
        retry_on_empty=retry_on_empty, condition="c1_react",
    )

    graph = StateGraph(ReactState)
    graph.add_node("agent", lambda s: agent_step(s, ctx))
    graph.add_node("tool", lambda s: tool_step(s, ctx))
    graph.add_node("finalize", lambda s: finalize_step(s, ctx))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent,
        {"agent": "agent", "tool": "tool", "finalize": "finalize"},
    )
    graph.add_edge("tool", "agent")
    graph.add_edge("finalize", END)
    app = graph.compile()

    final_state = app.invoke(
        initial_state(instance), config={"recursion_limit": max_steps * 2 + 10}
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
        empty_turns=final_state.get("empty_turns", 0),
        termination=termination_reason(final_state, max_steps),
        proposals=final_state.get("proposals", []),
    )
