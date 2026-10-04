# src/agents/single_agent/best_of_3.py
"""C5 -- Validated Best-of-3 single-agent sampling, as a fixed workflow.

    attempt 0 -> attempt 1 -> attempt 2 -> select (0 tokens) -> finalise (once)

THESIS_DECISIONS section 4, C5 (LOCKED 2026-07-26). Three independent C1 SEARCH
trajectories share the instance-level budget; a deterministic zero-token selector
compares their structured products; then the shared react_core finalisation runs ONCE.

**Not three full C1 runs at cap/3.** A full C1 run carries its own finalisation; here the
inner trajectories have none, so the reserve is charged once rather than three times.
Charging three reserves would hand C5 a structural handicap at the tight caps that has
nothing to do with sampling.

Budget (mirrors C3-5, not a new discipline). `W = cap - reserve`, and each attempt gets
`quota_i = floor(W / 3)`. Quotas are **non-transferable**: unused quota does not pass to
the next attempt, because that would make the result depend on attempt order -- the same
reason worker quotas do not transfer in C3. The integer remainder of `W // 3` (0-2 tokens)
is **left unused** and assigned to no attempt; handing it to the first or last attempt
would reintroduce a small order dependence for no benefit.

Sampling diversity, and the seed split. Every decoding parameter stays byte-identical to
the other conditions, so C5 gains no hidden decoding advantage. Only the **attempt seed**
varies, as the pre-specified sequence `BASE_SEED + attempt_index` = 42 / 43 / 44, applied
by a thin client adapter so `react_core` runs unchanged. The finalisation call is excluded
from that sequence: it goes through the unwrapped client and therefore keeps the single
fixed seed shared by C1-C5. Residual engine non-determinism is never relied on as the
source of diversity -- it is a confound, not a mechanism.

Selection (deterministic, zero tokens, oracle-free). Each attempt yields its own
`best_plan_so_far`, already filtered by the in-loop hidden validator gate. The selector
compares those three OBJECTS, not three finalised texts: discard what the full-instance
validator rejects, take the most meetings among the rest, break ties by lowest attempt
index, and fall back to attempt 0 if none is valid (the score is 0 either way). **The
solver optimum is never consulted**, and nothing about the selection is shown to the model.

Why selection must precede finalisation: finalising three attempts and then choosing among
their outputs would spend three terminal emits instead of one and would make the choice
depend on which emits happened to round-trip cleanly.

Nothing in this file modifies `react_core` or any other condition: C1/C2/C3 behaviour is
provably untouched, and no oracle component beyond the same harness-side `is_valid`
bookkeeping every condition already uses is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.react_core import (
    LoopContext,
    ReactResult,
    ReactState,
    agent_step,
    finalize_step,
    initial_state,
    route_after_agent,
    tool_step,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger, LLMClient
from src.core.llm_client import SEED as BASE_SEED
from src.oracle import is_valid
from src.schemas import AnswerContract, CallRecord, Instance

__all__ = [
    "N_ATTEMPTS",
    "BASE_SEED",
    "AttemptBudgetView",
    "AttemptOutcome",
    "attempt_seed",
    "attempt_quota",
    "select_best",
    "run_best_of_3",
]

# Fixed by section 4 C5: three trajectories, seeds BASE_SEED + index (42 / 43 / 44).
N_ATTEMPTS = 3
CONDITION = "c5_best_of_3"


def attempt_seed(index: int) -> int:
    """The pre-specified seed sequence, computed in ONE place and logged per call."""
    return BASE_SEED + index


def attempt_quota(cap: int, finalization_reserve: int) -> int:
    """`floor(W / 3)` with `W = cap - reserve`. The 0-2 token remainder stays unused."""
    return max(0, cap - finalization_reserve) // N_ATTEMPTS


# --------------------------------------------------------------------------- #
# Budget view: the same quota discipline C3's workers use, stated locally.
# --------------------------------------------------------------------------- #
@dataclass
class AttemptBudgetView:
    """Quota-constrained view over the single shared ledger (C5 budget rule).

    Duck-types the `BudgetLedger` surface `agent_step` consumes, so the shared core runs
    unchanged. The shared ledger stays the sole budget authority: every call is booked
    there, and the view only ADDS the attempt-quota bound on top of the global guard.

    Deliberately a local mirror of C3's `WorkerBudgetView` rather than an import across
    the single-agent / multi-agent boundary: C3 is frozen and must not acquire a
    dependency from a condition specified after it. A test asserts the two produce
    identical arithmetic, so the duplication is proven equivalent rather than assumed.
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
            # Attempts never finalise (C5): the reserve belongs to the single
            # instance-level finalisation node, charged once for the whole condition.
            raise ValueError("a C5 attempt cannot issue a finalising call")
        room_global = self.shared.max_new_tokens(input_tokens, finalizing=False)
        room_quota = max(0, self.quota_remaining - input_tokens)
        return min(room_global, room_quota)

    def should_finalize(self, next_input_tokens: int = 0) -> bool:
        """True when the next call no longer fits (quota or global). For an attempt this
        means "end this trajectory", not instance finalise."""
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


# --------------------------------------------------------------------------- #
# Per-attempt seeding, without touching react_core.
# --------------------------------------------------------------------------- #
@dataclass
class _SeededClient:
    """The run's client with one attempt's seed pinned on every completion.

    A thin adapter rather than a core change: `agent_step` calls `ctx.client.complete(...)`
    with no seed, so the seed is supplied here and nowhere else. The finalisation node
    receives the UNWRAPPED client, which is what keeps the terminal emit on the single
    fixed seed shared by C1-C5.
    """

    inner: Any
    seed: int

    def count_input(self, *args: Any, **kwargs: Any) -> int:
        return self.inner.count_input(*args, **kwargs)

    def complete(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("seed", self.seed)
        return self.inner.complete(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # usage_log and anything else the harness reads off the client.
        return getattr(self.inner, name)


@dataclass(frozen=True)
class AttemptOutcome:
    """One trajectory's product. Attempts never serialise (C5)."""

    index: int
    seed: int
    plan: AnswerContract                 # best VALID plan under the in-loop gate
    calls: list[CallRecord]
    transcript: list[dict[str, str]]
    proposals: list[dict[str, Any]]
    n_steps: int
    aborted: bool
    empty_turns: int
    global_budget_cut: bool


def _run_attempt(
    instance: Instance,
    client: LLMClient,
    shared_ledger: BudgetLedger,
    *,
    index: int,
    quota: int,
    max_steps: int,
    retry_on_empty: bool,
) -> AttemptOutcome:
    """One C1 search trajectory under its own quota, with no finalisation node."""
    seed = attempt_seed(index)
    view = AttemptBudgetView(shared=shared_ledger, quota=quota)
    ctx = LoopContext(
        instance=instance,
        client=_SeededClient(inner=client, seed=seed),  # type: ignore[arg-type]
        ledger=view,                                     # type: ignore[arg-type]
        max_steps=max_steps,
        retry_on_empty=retry_on_empty,
        condition=CONDITION,
    )

    # The C1 wiring with one difference: "finalize" ends THIS trajectory; the terminal
    # serialisation runs once, after selection.
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
        initial_state(instance), config={"recursion_limit": max_steps * 2 + 10}
    )

    role = f"bon_attempt_{index}"
    calls = [
        c.model_copy(update={"role": role, "attempt_index": index, "sampling_seed": seed})
        for c in final_state["calls"]
    ]
    return AttemptOutcome(
        index=index,
        seed=seed,
        plan=final_state["best_plan"],
        calls=calls,
        transcript=final_state["messages"],
        proposals=[{**p, "role": role, "attempt_index": index}
                   for p in final_state.get("proposals", [])],
        n_steps=final_state["step"],
        aborted=final_state.get("aborted", False),
        empty_turns=final_state.get("empty_turns", 0),
        global_budget_cut=view.global_cut,
    )


# --------------------------------------------------------------------------- #
# Selection: deterministic, zero tokens, oracle-free.
# --------------------------------------------------------------------------- #
def select_best(
    instance: Instance, outcomes: list[AttemptOutcome]
) -> tuple[int, AnswerContract]:
    """`(winning attempt index, plan)` by the fixed rule. Never consults the optimum.

    (1) discard candidates the full-instance validator rejects; (2) among the rest take
    the most meetings; (3) ties -> lowest attempt index; (4) none valid -> attempt 0, whose
    score is 0 either way. The comparison is over structured plans, not finalised texts.
    """
    valid = [o for o in outcomes if is_valid(instance, o.plan)]
    if not valid:
        return outcomes[0].index, outcomes[0].plan
    # max() is stable, so the earliest attempt already wins a tie; the key states it
    # explicitly rather than relying on that.
    best = min(valid, key=lambda o: (-len(o.plan.meetings), o.index))
    return best.index, best.plan


def _transcript_block(label: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Concatenation marker + one attempt's isolated conversation (log only -- no model
    ever sees the marker)."""
    if not messages:
        return []
    return [{"role": "system", "content": f"[{label} conversation]"}, *messages]


def run_best_of_3(
    instance: Instance,
    client: LLMClient,
    *,
    cap: int,
    max_steps: int = 12,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
    retry_on_empty: bool = False,
) -> ReactResult:
    """Run C5 on one instance under a token `cap`. Returns the run artifacts.

    `retry_on_empty` keeps the shared call shape with C1/C2/C3; it is the EXPERIMENTAL
    empty-turn arm and defaults to the locked behaviour.
    """
    ledger = BudgetLedger(cap=cap, finalization_reserve=finalization_reserve)
    quota = attempt_quota(cap, finalization_reserve)

    outcomes = [
        _run_attempt(
            instance, client, ledger,
            index=i, quota=quota, max_steps=max_steps, retry_on_empty=retry_on_empty,
        )
        for i in range(N_ATTEMPTS)
    ]

    winner_index, winner_plan = select_best(instance, outcomes)

    # ONE finalisation for the whole condition, through the shared react_core node and
    # the UNWRAPPED client, so the terminal emit keeps the single fixed seed.
    ctx = LoopContext(
        instance=instance, client=client, ledger=ledger, max_steps=max_steps,
        retry_on_empty=retry_on_empty, condition=CONDITION,
    )
    state: ReactState = initial_state(instance)  # type: ignore[assignment]
    state["best_plan"] = winner_plan
    final = finalize_step(state, ctx)

    calls = [c for o in outcomes for c in o.calls] + list(final["calls"])
    transcript = [
        m for o in outcomes
        for m in _transcript_block(f"bon_attempt_{o.index}", o.transcript)
    ]
    via_budget = any(o.global_budget_cut for o in outcomes)

    # LOG-ONLY. Preserved explicitly rather than reconstructed from the proposal rows:
    # reconstruction works only while "best_plan equals the last accepted proposal" holds,
    # which is true today but is an invariant of another module. Nothing here is read by
    # the loop, the selector or the ledger, and no model call was made to produce it.
    products = [
        {"attempt_index": o.index, "seed": o.seed,
         "plan": [{"person_id": m.person_id, "start_time": m.start_time}
                  for m in o.plan.meetings],
         "n_meetings": len(o.plan.meetings)}
        for o in outcomes
    ]
    distinct = {tuple((m.person_id, m.start_time) for m in o.plan.meetings)
                for o in outcomes}
    diagnostics = {
        "attempt_products": products,
        "winner_attempt_index": winner_index,
        "n_distinct_products": len(distinct),
    }
    return ReactResult(
        final_plan=final["final_plan"],
        best_plan_so_far=final["best_plan"],
        tokens=ledger.to_token_usage(
            budget_exhausted=via_budget or ledger.budget_exhausted
        ),
        calls=calls,
        transcript=transcript,
        n_steps=sum(o.n_steps for o in outcomes),
        finalization_mismatch=bool(final.get("finalization_mismatch", False)),
        empty_turns=sum(o.empty_turns for o in outcomes),
        termination=(
            "aborted" if all(o.aborted for o in outcomes)
            else "budget" if via_budget
            else "agent_finish"
        ),
        proposals=[p for o in outcomes for p in o.proposals],
        diagnostics=diagnostics,
    )
