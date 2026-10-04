# src/agents/multi_agent/worker.py
"""C3 worker: the byte-identical C1 loop over a sub-instance, under a quota view.

THESIS_DECISIONS section 4, C3-1/C3-5 (LOCKED). A worker is the shared react_core
agent<->tool loop -- same nodes, same system-prompt template (built by the shared
`initial_state`), same three tools bound to the sub-instance, same section-2 rules
including the hidden best-plan bookkeeping (checked against the sub-instance).
Workers run in ISOLATED conversations and never serialise: the product is the
worker's best_plan_so_far, and the instance-level finalise runs once at the end.

Budget (C3-5): each worker call is capped by BOTH the global section-2 guard and
the worker's quota -- `max_new = max(0, min(remaining - reserve, quota_remaining)
- input)` -- and every call is booked to both. `WorkerBudgetView` implements this
as a drop-in for the `BudgetLedger` interface the shared `agent_step` consumes, so
the core runs unchanged. Quotas never transfer between workers; quota exhaustion
is an internal allocation boundary, not global budget exhaustion (the view tracks
separately whether the GLOBAL bound was the one that fired).
"""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from src.agents.react_core import (
    LoopContext,
    ReactState,
    agent_step,
    initial_state,
    route_after_agent,
    tool_step,
)
from src.core import BudgetLedger, LLMClient
from src.schemas import Instance

from .state import WorkerOutcome

__all__ = ["WorkerBudgetView", "run_worker"]


@dataclass
class WorkerBudgetView:
    """Quota-constrained view over the single shared ledger (C3-5).

    Duck-types the `BudgetLedger` surface `agent_step` uses (`max_new_tokens`,
    `should_finalize`, `record_call`). The shared ledger stays the sole budget
    authority: every call is booked there with the same identity, and the view
    only ADDS the worker-quota bound on top of the global guard.
    """

    shared: BudgetLedger
    quota: int
    spent: int = 0
    # Set when the GLOBAL guard (not the quota) is what blocked the next call.
    global_cut: bool = False

    @property
    def quota_remaining(self) -> int:
        return self.quota - self.spent

    def max_new_tokens(self, input_tokens: int, *, finalizing: bool = False) -> int:
        if finalizing:
            # Workers never finalise (C3-1); the reserve belongs to the shared
            # instance-level finalisation node only.
            raise ValueError("a C3 worker cannot issue a finalising call")
        room_global = self.shared.max_new_tokens(input_tokens, finalizing=False)
        room_quota = max(0, self.quota_remaining - input_tokens)
        return min(room_global, room_quota)

    def should_finalize(self, next_input_tokens: int = 0) -> bool:
        """True when the next worker call no longer fits (quota or global).
        For the worker graph this means "end the sub-loop", not instance finalise."""
        blocked = self.max_new_tokens(next_input_tokens) <= self.shared.route_threshold
        if blocked and self.shared.should_finalize(next_input_tokens):
            self.global_cut = True
        return blocked

    def record_call(
        self, *, input_tokens: int, thinking_tokens: int, answer_tokens: int
    ) -> None:
        self.shared.record_call(
            input_tokens=input_tokens,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
        )
        self.spent += input_tokens + thinking_tokens + answer_tokens


def run_worker(
    sub_instance: Instance | None,
    client: LLMClient,
    shared_ledger: BudgetLedger,
    *,
    quota: int,
    max_steps: int,
    label: str,
) -> WorkerOutcome:
    """Run one worker sub-loop; return its outcome. Empty cluster -> skipped outright."""
    if sub_instance is None:
        return WorkerOutcome.skipped(label)

    view = WorkerBudgetView(shared=shared_ledger, quota=quota)
    # The view duck-types the ledger interface agent_step consumes.
    ctx = LoopContext(
        instance=sub_instance, client=client, ledger=view,  # type: ignore[arg-type]
        max_steps=max_steps, condition="c3_mas",
    )

    # The C1 wiring with one difference: "finalize" ends the SUB-LOOP (C3-6 --
    # the section-2 empty/budget rule scoped to the enclosing loop), it does not
    # run a serialisation node.
    graph = StateGraph(ReactState)
    graph.add_node("agent", lambda s: agent_step(s, ctx))
    graph.add_node("tool", lambda s: tool_step(s, ctx))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"agent": "agent", "tool": "tool", "finalize": END}
    )
    graph.add_edge("tool", "agent")
    app = graph.compile()

    final_state = app.invoke(
        initial_state(sub_instance), config={"recursion_limit": max_steps * 2 + 10}
    )

    # Relabel the CallRecords to the worker's role (C3-7): content unchanged.
    calls = [c.model_copy(update={"role": label}) for c in final_state["calls"]]
    return WorkerOutcome(
        label=label,
        sub_plan=final_state["best_plan"],
        calls=calls,
        transcript=final_state["messages"],
        n_steps=final_state["step"],
        aborted=final_state.get("aborted", False),
        # Relabelled like the calls, so a worker's rejected proposals stay attributable
        # to it once the two workers' rows are merged at the instance level.
        proposals=[{**p, "role": label} for p in final_state.get("proposals", [])],
        global_budget_cut=view.global_cut,
    )
