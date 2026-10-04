# src/agents/multi_agent/critic.py
"""C3 critic: one guarded fresh-context coordination call over the candidate pool.

THESIS_DECISIONS section 4, C3-4 (revised 2026-07-16 after design review). The
caller routes here only when the candidate pool is non-empty (C3-6: a critic over
an empty pool would be a fresh solo attempt -- the hidden retry the park-flag
forbids). Fresh context: the input is the pool-restricted critic prompt (task
rules + candidate people + restricted travel matrix + the raw draft), never the
worker transcripts and never the full instance.

Exactly ONE call, thinking ON, no tools, under the plain global section-2 guard
(no quota -- the critic's room is whatever the workers left, structurally at
least W/4 by C3-5). Failure semantics mirror C2's reflection turns: empty
post-think or an unparseable answer is a NO-OP (the fallback stands; no retry).

Hidden gate, three conjuncts, all harness-side (the critic is never told the
outcome): proposed ids SUBSET-OF pool AND full-instance validator passes AND
strictly longer than best_plan_so_far (the fallback). The subset conjunct makes
an out-of-pool proposal structurally worthless; the strictly-longer conjunct
means adoption happens exactly on a genuine coordination gain over the best
single worker.
"""

from __future__ import annotations

import time
from typing import Any

from src.agents.react_core import LoopContext, _parse_action, _plan_from_args
from src.agents.react_core import proposal_record
from src.schemas import CallRecord

from .prompts import critic_messages
from .state import MasState

__all__ = ["critic_step"]


def critic_step(state: MasState, ctx: LoopContext) -> dict[str, Any]:
    """One guarded critic call; every outcome leaves best_plan >= the fallback."""
    client, ledger = ctx.client, ctx.ledger
    pool = set(state["candidate_pool"])
    msgs = critic_messages(ctx.instance, state["draft_plan"], pool)
    input_tokens = client.count_input(msgs, enable_thinking=True)
    if ledger.should_finalize(input_tokens):
        # The pool dump does not fit the remaining room -> guard-skip straight to
        # finalise (pre-registered as the expected 2k-cap behaviour, C3-5).
        return {"critic_ran": False, "via_budget": True}

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
        role="critic",
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
        latency_seconds=latency,
        finish_reason=resp.finish_reason,
        requested_max_tokens=resp.requested_max_tokens,
        effective_max_tokens=resp.effective_max_tokens,
        context_limited=resp.context_limited,
    )
    # Post-think answer only; raw_text is never parsed nor re-fed (section 2).
    visible = resp.answer
    updates: dict[str, Any] = {
        "calls": [call],
        "critic_ran": True,
        "critic_messages": msgs + [{"role": "assistant", "content": visible}],
    }
    if not visible.strip():
        return updates  # empty post-think -> no-op; the graph proceeds to finalise

    action = _parse_action(visible)
    if action is not None and action["tool"] == "propose":
        plan = _plan_from_args(action["args"])
        # The hidden gate (C3-4): subset of the pool, full-instance validator,
        # strictly longer than the fallback. The critic learns nothing about it.
        record = proposal_record(
            ctx.instance, plan, step=0, role="critic",
            condition=ctx.condition, raw=action.get("raw", ""),
            best_len=len(state["best_plan"].meetings),
        )
        in_pool = plan is not None and {m.person_id for m in plan.meetings} <= pool
        # The subset conjunct is the critic's own gate, not a validator verdict, so it
        # is recorded separately -- an out-of-pool plan can be perfectly feasible.
        record["in_pool"] = in_pool
        record["accepted_into_best_plan"] = record["accepted_into_best_plan"] and in_pool
        updates["critic_proposal"] = record
        if record["accepted_into_best_plan"]:
            updates["best_plan"] = plan
    # Any other shape (unparseable, finish, tool text) is a no-op: C2's revise
    # has exactly this semantics -- a wasted turn, not a retried one.
    return updates
