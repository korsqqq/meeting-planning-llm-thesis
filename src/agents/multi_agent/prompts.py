# src/agents/multi_agent/prompts.py
"""C3 prompt builders: the critic's pool-restricted prompt.

Only the CRITIC has prompts here. The workers deliberately have NONE: each worker
is the byte-identical C1 loop and takes its system/user template from the shared
`react_core.initial_state(sub_instance)` -- identical template for both workers by
construction (THESIS_DECISIONS section 4, C3-1).

The critic prompt (C3-4, revised 2026-07-16 after design review) carries the
CANDIDATE POOL, not the instance: only people appearing in at least one worker
sub-plan (their id, location, window), the travel matrix restricted to depot +
candidate locations, and the draft (marked as possibly rule-breaking). People no
worker discovered stay invisible -- C1/C2 must buy every task fact through the
three tools, so a full-instance dump would be privileged information. The pool is
the information the workers' tool calls already surfaced, plus exactly the
cross-candidate travel entries coordination needs; it is counted input like any
inter-agent message. No optimum, no validator verdicts, no oracle vocabulary
(the transcript sanity scan stays clean).
"""

from __future__ import annotations

from collections.abc import Iterable

from src.schemas import AnswerContract, Instance

__all__ = ["serialise_candidates", "format_plan", "critic_messages"]


def _pool_locations(instance: Instance, pool: set[str]) -> list[str]:
    """Depot + the candidate people's locations, in parent `locations` order."""
    keep = {instance.start_location} | {
        p.location for p in instance.people if p.person_id in pool
    }
    return [loc for loc in instance.locations if loc in keep]


def serialise_candidates(instance: Instance, pool_ids: Iterable[str]) -> str:
    """Deterministic dump of the candidate pool ONLY (C3-4 input, items i-ii):
    pool people in instance order; the travel matrix restricted to depot +
    candidate locations, in `locations` order."""
    pool = set(pool_ids)
    lines = ["Candidate people (person_id @ location, window start-end):"]
    for p in instance.people:  # instance order
        if p.person_id in pool:
            lines.append(f"{p.person_id} @ {p.location}, {p.window_start}-{p.window_end}")
    locs = _pool_locations(instance, pool)
    lines.append("Travel times in minutes (from: to=minutes):")
    for a in locs:
        row = ", ".join(f"{b}={instance.travel_times[a][b]}" for b in locs)
        lines.append(f"from {a}: {row}")
    return "\n".join(lines)


def format_plan(plan: AnswerContract) -> str:
    """A plan as `person@start` pairs in visiting order."""
    return ", ".join(f"{m.person_id}@{m.start_time}" for m in plan.meetings) or "(none)"


def _task_rules(instance: Instance) -> str:
    """The same task rules the workers see (mirrors the shared C1 template)."""
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


def critic_messages(
    instance: Instance, draft: AnswerContract, pool_ids: Iterable[str]
) -> list[dict[str, str]]:
    """The critic's full two-message conversation (C3-4: one call, fresh context,
    no tools, candidate pool only)."""
    system = (
        "You are a meeting-plan reviewer. Two planning agents proposed meetings with "
        "the candidate people listed below; their combined schedule is a rough draft "
        "that may break the rules. Coordinate the candidates into the best full plan "
        "you can justify.\n"
        + _task_rules(instance)
    )
    user = (
        "Task data.\n"
        f"{serialise_candidates(instance, pool_ids)}\n"
        f"Draft plan (person_id @ start-minute, in visiting order; it may break the "
        f"rules): {format_plan(draft)}\n\n"
        "Check the draft step by step: every availability window, the travel time "
        "between consecutive meetings, and the end-of-day deadline. You may reorder "
        "meetings, change start minutes, or leave meetings out. Use ONLY people from "
        "the candidate list above -- do not add anyone else. Then output your final "
        "plan on ONE line, exactly in this format:\n"
        "Action: propose[p0@10, p1@60]\n"
        "Propose the FULL plan in visiting order (person_id @ meeting-start-minute). "
        "Output exactly one such line and nothing after it."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
