# src/agents/multi_agent/aggregator.py
"""C3 aggregator: deterministic, zero-token, and NON-PLANNING.

THESIS_DECISIONS section 4, C3-3 (revised 2026-07-16 after design review). The
earlier simulate-and-drop merge (three candidate orders, earliest-feasible
re-timing, drop-on-infeasible, argmax) was rejected as a handcrafted planning
heuristic: uncounted central repair that C1/C2 never receive. This module now
contains NO planning decision -- no re-timing, no order search, no drops, no
feasibility judgement:

  * `build_draft` -- the union of the two sub-plans in ONE fixed order (sorted by
    worker-assigned start_time, ties by person_id), times UNCHANGED, nothing
    dropped. The draft may be INVALID on the full instance, deliberately: it is
    raw material for the critic, never a scoring candidate. Every repair decision
    (re-order, re-time, drop) belongs to the critic's LLM call, where it is paid
    for in counted tokens.
  * `candidate_ids` -- the pool of person_ids appearing in at least one sub-plan.
    The critic may coordinate exactly these candidates (C3-4); people no worker
    discovered stay invisible.

The fallback (the better single worker sub-plan, ties -> worker A) is selected in
the workflow node (hierarchical.py) and passes the shared hidden gate there --
selection between two finished artifacts, not construction.
"""

from __future__ import annotations

from src.schemas import AnswerContract, Meeting

__all__ = ["build_draft", "candidate_ids"]


def build_draft(plan_a: AnswerContract, plan_b: AnswerContract) -> AnswerContract:
    """Union of the two sub-plans on the shared timeline: ONE fixed order (by
    worker-assigned start_time, ties by person_id), times unchanged, nothing
    dropped or repaired. May be invalid on the full instance -- it is a draft."""
    meetings: list[Meeting] = sorted(
        list(plan_a.meetings) + list(plan_b.meetings),
        key=lambda m: (m.start_time, m.person_id),
    )
    return AnswerContract(meetings=meetings)


def candidate_ids(plan_a: AnswerContract, plan_b: AnswerContract) -> set[str]:
    """The candidate pool: every person_id appearing in at least one sub-plan."""
    return {m.person_id for m in plan_a.meetings} | {m.person_id for m in plan_b.meetings}
