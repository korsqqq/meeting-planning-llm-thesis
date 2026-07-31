# src/agents/multi_agent/hierarchical.py
"""C3 -- the hierarchical MAS (condition 3), as a fixed LangGraph workflow.

    supervisor -> worker A -> worker B -> aggregator -> critic -> finalise

THESIS_DECISIONS section 4, C3-1..C3-8 (LOCKED; C3-3/C3-4 revised 2026-07-16
after design review). The supervisor and the aggregator are deterministic
zero-token nodes; the LLM agents are the two workers (the byte-identical C1 loop
over their sub-instances, under equal quotas) and the critic (one fresh-context
coordination call over the workers' candidate pool). Workers run SEQUENTIALLY
(A then B) in isolated conversations -- a determinism device only: the quotas
make the order budget-irrelevant (C3-5). The instance-level finalise is the
byte-identical react_core node (same reserve, thinking OFF, guided JSON,
serialising best_plan_so_far).

The aggregate node performs NO planning (C3-3): the better single worker
sub-plan (ties -> A) passes the shared hidden gate as the fallback; the draft is
the raw one-order union (times unchanged, possibly invalid, never scored); the
candidate pool is the set of worker-proposed person_ids -- the only people the
critic may coordinate, enforced by the gate's subset conjunct in critic_step.

Routing rules (C3-6): a worker that aborts or runs out of room ends only its own
sub-loop -- the sibling still runs; the aggregator always runs (zero tokens); the
critic is SKIPPED when the candidate pool is empty (coordinating "nothing" would
be a fresh solo attempt -- the hidden retry the section-2 park-flag forbids) or
when its input no longer fits the global guard.

Nothing in this file (or this package) modifies the shared core: C1/C2 behaviour
is provably untouched, and no oracle component is imported by the agent side --
the hidden `is_valid` gate is the same harness-side bookkeeping every condition
uses.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.react_core import LoopContext, ReactResult, finalize_step
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger, LLMClient
from src.oracle import is_valid
from src.schemas import Instance

from .aggregator import build_draft, candidate_ids
from .critic import critic_step
from .state import MasState, WorkerOutcome, initial_mas_state
from .supervisor import split_instance
from .worker import run_worker

__all__ = ["ReactResult", "run_hierarchical", "WORKER_SHARE_NUM", "WORKER_SHARE_DEN"]

# C3-5 (LOCKED mechanism; value [PILOT]): each worker's quota is
# floor(W * 3/8) of the working budget W = cap - reserve, held as an integer
# fraction so the arithmetic is exact. The two workers can jointly book at most
# 3/4 W, which structurally guarantees the critic >= W/4 of room.
WORKER_SHARE_NUM = 3
WORKER_SHARE_DEN = 8


def _transcript_block(label: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Concatenation marker + one agent's isolated conversation (log only --
    no model ever sees the marker)."""
    if not messages:
        return []
    return [{"role": "system", "content": f"[{label} conversation]"}, *messages]


def run_hierarchical(
    instance: Instance,
    client: LLMClient,
    *,
    cap: int,
    max_steps: int = 12,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
) -> ReactResult:
    """Run C3 on one instance under a token `cap`. Returns the run artifacts."""
    ledger = BudgetLedger(cap=cap, finalization_reserve=finalization_reserve)
    ctx = LoopContext(instance=instance, client=client, ledger=ledger, max_steps=max_steps)
    working = cap - finalization_reserve
    quota = working * WORKER_SHARE_NUM // WORKER_SHARE_DEN

    def supervisor_node(state: MasState) -> dict[str, Any]:
        sub_a, sub_b = split_instance(ctx.instance)
        return {"sub_a": sub_a, "sub_b": sub_b}

    def worker_node(label: str, sub_key: str, out_key: str):
        def _node(state: MasState) -> dict[str, Any]:
            outcome = run_worker(
                state[sub_key], client, ledger,
                quota=quota, max_steps=max_steps, label=label,
            )
            return {out_key: outcome, "calls": outcome.calls}
        return _node

    def aggregate_node(state: MasState) -> dict[str, Any]:
        oc_a: WorkerOutcome = state["outcome_a"]
        oc_b: WorkerOutcome = state["outcome_b"]
        # Fallback = the better single worker sub-plan (ties -> A): selection
        # between two finished artifacts, never construction (C3-3).
        fallback = (
            oc_b.sub_plan
            if len(oc_b.sub_plan.meetings) > len(oc_a.sub_plan.meetings)
            else oc_a.sub_plan
        )
        updates: dict[str, Any] = {
            "draft_plan": build_draft(oc_a.sub_plan, oc_b.sub_plan),
            "candidate_pool": sorted(candidate_ids(oc_a.sub_plan, oc_b.sub_plan)),
        }
        # The fallback enters the same hidden gate as any propose: full-instance
        # validator, adopted only if strictly longer than the current best
        # (initially empty). The draft never enters the gate.
        if is_valid(ctx.instance, fallback) and len(fallback.meetings) > len(
            state["best_plan"].meetings
        ):
            updates["best_plan"] = fallback
        if oc_a.global_budget_cut or oc_b.global_budget_cut:
            updates["via_budget"] = True
        return updates

    def route_after_aggregate(state: MasState) -> str:
        # C3-6: an empty candidate pool skips the critic (hidden-retry boundary).
        return "critic" if state["candidate_pool"] else "finalize"

    graph = StateGraph(MasState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker_a", worker_node("worker_a", "sub_a", "outcome_a"))
    graph.add_node("worker_b", worker_node("worker_b", "sub_b", "outcome_b"))
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("critic", lambda s: critic_step(s, ctx))
    graph.add_node("finalize", lambda s: finalize_step(s, ctx))
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "worker_a")
    graph.add_edge("worker_a", "worker_b")
    graph.add_edge("worker_b", "aggregate")
    graph.add_conditional_edges(
        "aggregate", route_after_aggregate, {"critic": "critic", "finalize": "finalize"}
    )
    graph.add_edge("critic", "finalize")
    graph.add_edge("finalize", END)
    app = graph.compile()

    final_state = app.invoke(initial_mas_state())

    oc_a: WorkerOutcome = final_state["outcome_a"]
    oc_b: WorkerOutcome = final_state["outcome_b"]
    transcript = (
        _transcript_block("worker_a", oc_a.transcript)
        + _transcript_block("worker_b", oc_b.transcript)
        + _transcript_block("critic", final_state["critic_messages"])
    )
    n_steps = oc_a.n_steps + oc_b.n_steps + (1 if final_state["critic_ran"] else 0)

    return ReactResult(
        final_plan=final_state["final_plan"],
        best_plan_so_far=final_state["best_plan"],
        tokens=ledger.to_token_usage(
            budget_exhausted=final_state["via_budget"] or ledger.budget_exhausted
        ),
        calls=final_state["calls"],
        transcript=transcript,
        n_steps=n_steps,
        finalization_mismatch=final_state.get("finalization_mismatch", False),
    )
