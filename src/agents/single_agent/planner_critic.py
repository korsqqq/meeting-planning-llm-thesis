# src/agents/single_agent/planner_critic.py
"""C4 -- one full-instance planner + one fresh-context critic, no decomposition.

    planner (C1 loop, full instance) -> evidence/pool/draft (0 tokens) -> critic -> finalise

THESIS_DECISIONS section 4, C4 (design E, specified 2026-08-12). C4 sits between C2 and C3:
it keeps C2's single planner working on the whole task, and replaces C2's same-context
verify/revise with C3's fresh-context critic. Nothing is decomposed.

*Why the literal "C3 with one worker" is rejected.* With one worker the candidate pool is
exactly the people of that worker's best plan, which is also the fallback, so the gate's
strictly-longer conjunct is unsatisfiable by construction: no subset of the pool can hold
more people than the pool. C4 would be C1 plus a wasted call. In C3 the pool is the UNION
of two sub-plans while the fallback is the BETTER SINGLE one, and that gap is the room the
critic works in. C4 recovers a real gap differently -- from the planner's own rejected
proposals, which name people its best valid plan does not contain.

*The information rule, which is the whole point of design E.* A deterministic harness-side
evidence cache holds exactly what the planner's tool calls exposed: ids from `list_people`,
a person's location and window only after `get_availability`, one ordered entry per
`get_travel_time`. **A proposal discloses nothing.** Naming a person the planner never
queried must not hand the critic that person's facts, because C1 and C2 have to buy every
task fact through the tools. Travel pairs the critic would need but the planner never asked
for are shown as `unknown` rather than filled in: the critic then knows the limits of its
own knowledge, and no fact is invented.

*What is deliberately identical to C3*: the acceptance gate (subset of pool ∧ full-instance
validator ∧ strictly longer than the fallback), the draft as a raw union that may be
invalid and is never scored, the fallback as the search stage's best valid plan, the
failure semantics (every bad outcome is a no-op with no retry), and one finalisation.

*What differs from C3, and is recorded rather than hidden*: C3's critic receives the
coordination information required to reconnect decomposed subproblems -- cross-cluster
travel entries no worker could have acquired, because partitioning removed them from the
sub-matrices. C4's planner faced no such partition, so its critic is held to what the
planner actually gathered. The gap is measured by `evidence_travel_coverage` rather than
equalised away.

Nothing here modifies `react_core` or any other condition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    proposal_record,
    route_after_agent,
    tool_step,
)
from src.core import DEFAULT_FINALIZATION_RESERVE, BudgetLedger, LLMClient
from src.schemas import AnswerContract, CallRecord, Instance, Meeting

__all__ = [
    "PLANNER_SHARE_NUM",
    "PLANNER_SHARE_DEN",
    "Evidence",
    "PlannerBudgetView",
    "planner_quota",
    "collect_evidence",
    "candidate_pool",
    "build_draft",
    "critic_messages",
    "run_planner_critic",
]

CONDITION = "c4_planner_critic"
UNKNOWN = "unknown"

# The planner's quota is floor(W * 3/4) of the working budget W = cap - reserve, held as
# an integer fraction so the arithmetic is exact. It matches C3's maximum pre-critic search
# allowance -- two workers at 3/8 W each -- and leaves the critic the same structural
# guarantee of at least W/4.
PLANNER_SHARE_NUM = 3
PLANNER_SHARE_DEN = 4


def planner_quota(cap: int, finalization_reserve: int) -> int:
    return max(0, cap - finalization_reserve) * PLANNER_SHARE_NUM // PLANNER_SHARE_DEN


# --------------------------------------------------------------------------- #
# Budget view: the same quota discipline C3's worker and C5's attempt use.
# --------------------------------------------------------------------------- #
@dataclass
class PlannerBudgetView:
    """Quota-constrained view over the single shared ledger.

    Duck-types the `BudgetLedger` surface `agent_step` consumes, so the shared core runs
    unchanged. The shared ledger stays the sole budget authority: every call is booked
    there, and the view only ADDS the planner-quota bound on top of the global guard.

    A local mirror of C3's `WorkerBudgetView` and C5's `AttemptBudgetView` rather than an
    import: both of those belong to frozen conditions and must not acquire a dependency on
    a condition specified after them. A test asserts all three produce identical
    arithmetic, so the repetition is proven equivalent rather than assumed.
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
            # The planner never finalises: the reserve belongs to the single
            # instance-level finalisation node, charged once for the whole condition.
            raise ValueError("the C4 planner cannot issue a finalising call")
        room_global = self.shared.max_new_tokens(input_tokens, finalizing=False)
        room_quota = max(0, self.quota_remaining - input_tokens)
        return min(room_global, room_quota)

    def should_finalize(self, next_input_tokens: int = 0) -> bool:
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
# The evidence cache: exactly what the planner's tool calls exposed.
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """Task facts the planner actually acquired. Nothing else may reach the critic."""

    people_listed: set[str] = field(default_factory=set)
    # person_id -> {"location", "window_start", "window_end"}, only after get_availability
    windows: dict[str, dict[str, Any]] = field(default_factory=dict)
    # (from, to) -> minutes, one entry per successful get_travel_time
    travel: dict[tuple[str, str], int] = field(default_factory=dict)


def collect_evidence(instance: Instance, transcript: list[dict[str, str]]) -> Evidence:
    """Rebuild the evidence cache from the planner's finished trajectory.

    Reconstruction rather than instrumentation, so `react_core` keeps its semantics: the
    assistant turns are re-parsed with the SAME `_parse_action` the loop used, and a tool
    action counts as exposing a fact only when it would have succeeded -- an unknown
    person or location produced an error Observation and revealed nothing. `propose` and
    `finish` are skipped: a proposal discloses no task fact.
    """
    ev = Evidence()
    known_people = {p.person_id for p in instance.people}
    known_locs = set(instance.locations)
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        action = _parse_action(message.get("content", "") or "")
        if action is None:
            continue
        tool, args = action["tool"], action["args"]
        if tool == "list_people":
            ev.people_listed |= known_people
        elif tool == "get_availability" and len(args) == 1 and args[0] in known_people:
            person = instance.person(args[0])
            ev.windows[person.person_id] = {
                "location": person.location,
                "window_start": person.window_start,
                "window_end": person.window_end,
            }
        elif tool == "get_travel_time" and len(args) == 2:
            a, b = args
            if a in known_locs and b in known_locs:
                ev.travel[(a, b)] = instance.travel_time(a, b)
    return ev


# --------------------------------------------------------------------------- #
# Pool and draft: deterministic, zero tokens, no planning decision.
# --------------------------------------------------------------------------- #
def _proposed_plans(proposals: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The parsed plans, in proposal order. Unparseable rows carry no plan."""
    return [row["plan"] for row in proposals if row.get("parsed") and row.get("plan")]


def candidate_pool(proposals: list[dict[str, Any]], evidence: Evidence) -> set[str]:
    """Design E: proposed AND availability-exposed.

    Validity never filters membership. A pool built only from validator-approved
    proposals would encode the hidden verdict in the critic's input, which is the one
    thing the gate must never leak. Both conjuncts matter: the first keeps the pool
    comparable with C3's (people a search product named), the second stops a bare mention
    from disclosing facts the planner never bought.
    """
    proposed = {m["person_id"] for plan in _proposed_plans(proposals) for m in plan}
    return {pid for pid in proposed if pid in evidence.windows}


def build_draft(proposals: list[dict[str, Any]]) -> AnswerContract:
    """The planner's proposals unioned in ONE fixed order, unrepaired.

    Per person, the start time from the LATEST proposal naming them -- the planner's most
    recent opinion -- then ordered by (start_time, person_id). No re-timing, no drops, no
    feasibility judgement: every repair decision belongs to the critic's counted call. The
    draft may be invalid on the full instance and is never scored.
    """
    latest: dict[str, int] = {}
    for plan in _proposed_plans(proposals):
        for m in plan:
            latest[m["person_id"]] = int(m["start_time"])
    meetings = [Meeting(person_id=pid, start_time=t) for pid, t in latest.items()]
    return AnswerContract(meetings=sorted(meetings,
                                          key=lambda m: (m.start_time, m.person_id)))


# --------------------------------------------------------------------------- #
# The critic's prompt: evidence only, with unknowns named.
# --------------------------------------------------------------------------- #
def _task_rules(instance: Instance) -> str:
    """The same task rules the planner saw (mirrors the shared C1 template)."""
    if instance.waiting_allowed:
        waiting = "You may arrive early and wait for a window to open."
    else:
        waiting = "You cannot wait: a meeting must start exactly when you arrive."
    return (
        f"The route starts at location '{instance.start_location}' at minute "
        f"{instance.start_time}. Every meeting lasts exactly {instance.meeting_duration} "
        "minutes and must fit inside the person's availability window. There must be "
        "enough time to travel between consecutive meetings, and every meeting must end "
        f"by minute {instance.end_of_day}. {waiting}"
    )


def _format_plan(plan: AnswerContract) -> str:
    return ", ".join(f"{m.person_id}@{m.start_time}" for m in plan.meetings) or "(none)"


def serialise_evidence(
    instance: Instance, pool: set[str], evidence: Evidence
) -> tuple[str, int, int]:
    """The candidate block, and (known pairs, unknown pairs) for the diagnostics.

    Only cached facts appear. Every ordered pair over depot + pool locations is listed:
    the cached minutes when the planner asked for it, the literal `unknown` when it did
    not. Naming the gaps costs nothing and adds nothing, and it stops the critic from
    silently assuming a distance it was never given.
    """
    lines = ["Candidate people (person_id @ location, window start-end):"]
    for p in instance.people:                       # instance order, pool members only
        if p.person_id in pool:
            w = evidence.windows[p.person_id]
            lines.append(f"{p.person_id} @ {w['location']}, "
                         f"{w['window_start']}-{w['window_end']}")
    locs = [instance.start_location] + sorted(
        {evidence.windows[pid]["location"] for pid in pool}
        - {instance.start_location}
    )
    lines.append("Travel times in minutes (from: to=minutes; 'unknown' means it was "
                 "never looked up and you must not assume a value):")
    known = unknown = 0
    for a in locs:
        cells = []
        for b in locs:
            if a == b:
                cells.append(f"{b}=0")
                continue
            value = evidence.travel.get((a, b))
            if value is None:
                cells.append(f"{b}={UNKNOWN}")
                unknown += 1
            else:
                cells.append(f"{b}={value}")
                known += 1
        lines.append(f"from {a}: " + ", ".join(cells))
    return "\n".join(lines), known, unknown


def critic_messages(
    instance: Instance, draft: AnswerContract, pool: set[str], evidence: Evidence
) -> tuple[list[dict[str, str]], int, int]:
    """The critic's whole two-message conversation: one call, fresh context, no tools."""
    block, known, unknown = serialise_evidence(instance, pool, evidence)
    system = (
        "You are a meeting-plan reviewer. A planning agent proposed meetings with the "
        "candidate people listed below; its combined schedule is a rough draft that may "
        "break the rules. Produce the best full plan you can justify from these "
        "candidates.\n" + _task_rules(instance)
    )
    user = (
        "Task data.\n"
        f"{block}\n"
        f"Draft plan (person_id @ start-minute, in visiting order; it may break the "
        f"rules): {_format_plan(draft)}\n\n"
        "Check the draft step by step: every availability window, the travel time "
        "between consecutive meetings, and the end-of-day deadline. You may reorder "
        "meetings, change start minutes, or leave meetings out. Use ONLY people from the "
        "candidate list above -- do not add anyone else. Where a travel time is "
        "'unknown', you do not know it: prefer an order whose travel times are given. "
        "Then output your final plan on ONE line, exactly in this format:\n"
        "Action: propose[p0@10, p1@60]\n"
        "Propose the FULL plan in visiting order (person_id @ meeting-start-minute). "
        "Output exactly one such line and nothing after it."
    )
    return ([{"role": "system", "content": system},
             {"role": "user", "content": user}], known, unknown)


# --------------------------------------------------------------------------- #
# The condition.
# --------------------------------------------------------------------------- #
def _run_planner(
    instance: Instance,
    client: LLMClient,
    shared_ledger: BudgetLedger,
    *,
    quota: int,
    max_steps: int,
    retry_on_empty: bool,
) -> tuple[dict[str, Any], PlannerBudgetView]:
    """The C1 search loop over the FULL instance, under a quota, with no finalisation."""
    view = PlannerBudgetView(shared=shared_ledger, quota=quota)
    ctx = LoopContext(
        instance=instance, client=client, ledger=view,  # type: ignore[arg-type]
        max_steps=max_steps, retry_on_empty=retry_on_empty, condition=CONDITION,
    )
    graph = StateGraph(ReactState)
    graph.add_node("agent", lambda s: agent_step(s, ctx))
    graph.add_node("tool", lambda s: tool_step(s, ctx))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"agent": "agent", "tool": "tool", "finalize": END}
    )
    graph.add_edge("tool", "agent")
    app = graph.compile()
    state = app.invoke(
        initial_state(instance), config={"recursion_limit": max_steps * 2 + 10}
    )
    return state, view


def run_planner_critic(
    instance: Instance,
    client: LLMClient,
    *,
    cap: int,
    max_steps: int = 12,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
    retry_on_empty: bool = False,
) -> ReactResult:
    """Run C4 on one instance under a token `cap`. Returns the run artifacts."""
    ledger = BudgetLedger(cap=cap, finalization_reserve=finalization_reserve)
    quota = planner_quota(cap, finalization_reserve)

    planner_state, view = _run_planner(
        instance, client, ledger,
        quota=quota, max_steps=max_steps, retry_on_empty=retry_on_empty,
    )
    planner_calls = [c.model_copy(update={"role": "planner"})
                     for c in planner_state["calls"]]
    proposals = [{**p, "role": "planner"} for p in planner_state.get("proposals", [])]
    fallback: AnswerContract = planner_state["best_plan"]
    planner_spent = view.spent

    # Zero-token, non-planning construction.
    evidence = collect_evidence(instance, planner_state["messages"])
    pool = candidate_pool(proposals, evidence)
    draft = build_draft(proposals)

    critic_calls: list[CallRecord] = []
    critic_messages_log: list[dict[str, str]] = []
    critic_reached = False
    critic_improved = False
    known_pairs = unknown_pairs = 0
    best = fallback
    critic_room = max(0, ledger.remaining - finalization_reserve)

    # An empty pool skips the critic, exactly as in C3: coordinating "nothing" would be a
    # fresh solo attempt, which is the hidden retry the section-2 park flag forbids.
    if pool:
        msgs, known_pairs, unknown_pairs = critic_messages(
            instance, draft, pool, evidence)
        input_tokens = client.count_input(msgs, enable_thinking=True)
        if not ledger.should_finalize(input_tokens):
            critic_reached = True
            max_new = ledger.max_new_tokens(input_tokens, finalizing=False)
            t0 = time.perf_counter()
            resp = client.complete(msgs, max_tokens=max_new, enable_thinking=True)
            latency = time.perf_counter() - t0
            ledger.record_call(
                input_tokens=resp.input_tokens,
                thinking_tokens=resp.thinking_tokens,
                answer_tokens=resp.answer_tokens,
            )
            critic_calls = [CallRecord(
                role="critic",
                input_tokens=resp.input_tokens,
                thinking_tokens=resp.thinking_tokens,
                answer_tokens=resp.answer_tokens,
                latency_seconds=latency,
                finish_reason=resp.finish_reason,
                requested_max_tokens=resp.requested_max_tokens,
                effective_max_tokens=resp.effective_max_tokens,
                context_limited=resp.context_limited,
            )]
            visible = resp.answer          # post-think answer only, as everywhere
            critic_messages_log = msgs + [{"role": "assistant", "content": visible}]
            action = _parse_action(visible) if visible.strip() else None
            if action is not None and action["tool"] == "propose":
                plan = _plan_from_args(action["args"])
                record = proposal_record(
                    instance, plan, step=0, role="critic", condition=CONDITION,
                    raw=action.get("raw", ""), best_len=len(fallback.meetings),
                )
                in_pool = plan is not None and {
                    m.person_id for m in plan.meetings} <= pool
                # The subset conjunct is the critic's own gate, not a feasibility
                # verdict, so it is recorded separately: an out-of-pool plan can be
                # perfectly feasible and is still refused.
                record["in_pool"] = in_pool
                record["accepted_into_best_plan"] = (
                    record["accepted_into_best_plan"] and in_pool)
                proposals.append(record)
                if record["accepted_into_best_plan"]:
                    best = plan
                    critic_improved = True
            # Any other shape -- empty, unparseable, finish, tool text -- is a no-op with
            # no retry, and the fallback stands.

    # ONE finalisation, the shared react_core node, from best_plan_so_far.
    fctx = LoopContext(
        instance=instance, client=client, ledger=ledger, max_steps=max_steps,
        retry_on_empty=retry_on_empty, condition=CONDITION,
    )
    fstate: ReactState = initial_state(instance)  # type: ignore[assignment]
    fstate["best_plan"] = best
    final = finalize_step(fstate, fctx)

    pool_size, fallback_size = len(pool), len(fallback.meetings)
    diagnostics = {
        "pool_size": pool_size,
        "fallback_size": fallback_size,
        "headroom": pool_size - fallback_size,
        "critic_reached": critic_reached,
        "critic_improved": critic_improved,
        "evidence_people_covered": len(evidence.windows),
        "evidence_travel_coverage": (round(known_pairs / (known_pairs + unknown_pairs), 4)
                                     if (known_pairs + unknown_pairs) else None),
        "unknown_pairs_shown": unknown_pairs,
        "known_pairs_shown": known_pairs,
        "planner_spent": planner_spent,
        "planner_quota": quota,
        "critic_room": critic_room,
        "pool": sorted(pool),
    }

    transcript = list(planner_state["messages"])
    if critic_messages_log:
        transcript = (transcript
                      + [{"role": "system", "content": "[critic conversation]"}]
                      + critic_messages_log)

    return ReactResult(
        final_plan=final["final_plan"],
        best_plan_so_far=final["best_plan"],
        tokens=ledger.to_token_usage(
            budget_exhausted=view.global_cut or ledger.budget_exhausted
        ),
        calls=planner_calls + critic_calls + list(final["calls"]),
        transcript=transcript,
        n_steps=planner_state["step"] + (1 if critic_reached else 0),
        finalization_mismatch=bool(final.get("finalization_mismatch", False)),
        empty_turns=planner_state.get("empty_turns", 0),
        termination=(
            "aborted" if planner_state.get("aborted")
            else "budget" if view.global_cut
            else "agent_finish"
        ),
        proposals=proposals,
        diagnostics=diagnostics,
    )
