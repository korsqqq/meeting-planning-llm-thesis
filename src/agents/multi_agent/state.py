# src/agents/multi_agent/state.py
"""C3 outer-graph state and the per-worker outcome record.

The outer LangGraph state (`MasState`) is deliberately flat and small: the worker
loops run as NESTED graphs inside their own node (each over the shared
`react_core.ReactState`), so the outer state carries only their products -- the
sub-instances, the two `WorkerOutcome`s, the merged plan, and the same
`best_plan` / `final_plan` / `calls` channels the shared `finalize_step` node
reads and writes (THESIS_DECISIONS section 4, C3-7: the instance-level
finalisation is the byte-identical react_core node).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, TypedDict

from src.schemas import AnswerContract, CallRecord, Instance

__all__ = ["WorkerOutcome", "MasState", "initial_mas_state"]


@dataclass(frozen=True)
class WorkerOutcome:
    """Everything one worker's sub-loop produced (C3-1: its product is its
    best_plan_so_far; workers never serialise)."""

    label: str                                  # "worker_a" | "worker_b"
    sub_plan: AnswerContract                    # best VALID sub-plan, possibly empty
    calls: list[CallRecord] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    n_steps: int = 0
    aborted: bool = False                       # empty post-think / second format failure
    # Rejected-proposal taxonomy rows from this worker's sub-loop, relabelled to it.
    proposals: list[dict] = field(default_factory=list)
    # The GLOBAL guard (not the worker's quota) ended the loop -- feeds the
    # instance-level budget_exhausted flag (C3-5: quota exhaustion is an internal
    # allocation boundary, not budget exhaustion).
    global_budget_cut: bool = False

    @classmethod
    def skipped(cls, label: str) -> "WorkerOutcome":
        """The empty-cluster case (C3-1): the worker is skipped outright --
        zero calls, zero tokens, empty sub-plan."""
        return cls(label=label, sub_plan=AnswerContract.empty())


class MasState(TypedDict):
    """Outer C3 graph state. `best_plan` / `final_plan` / `calls` are the same
    channels ReactState exposes, so `react_core.finalize_step` runs unchanged.
    `draft_plan` is the aggregator's raw union (possibly invalid, never scored);
    `candidate_pool` is the sorted list of person_ids from the worker sub-plans
    -- the only people the critic may coordinate (C3-3/C3-4)."""

    sub_a: Instance | None
    sub_b: Instance | None
    outcome_a: WorkerOutcome | None
    outcome_b: WorkerOutcome | None
    draft_plan: AnswerContract
    candidate_pool: list[str]
    best_plan: AnswerContract
    final_plan: AnswerContract | None
    calls: Annotated[list[CallRecord], add]
    critic_ran: bool
    critic_messages: list[dict[str, str]]
    # The critic's single proposal with the validator's verdict, or None if it never
    # proposed. Carries an extra `in_pool` flag: the subset conjunct is the critic's
    # own gate, not a feasibility verdict, so the two reasons for refusal stay apart.
    critic_proposal: dict | None
    via_budget: bool
    # Set by the shared finalize_step: the terminal emit parsed but did not round-trip
    # best_plan_so_far (integrity diagnostic; the structural plan is scored either way).
    finalization_mismatch: bool


def initial_mas_state() -> MasState:
    return {
        "sub_a": None,
        "sub_b": None,
        "outcome_a": None,
        "outcome_b": None,
        "draft_plan": AnswerContract.empty(),
        "candidate_pool": [],
        "best_plan": AnswerContract.empty(),
        "final_plan": None,
        "calls": [],
        "critic_ran": False,
        "critic_messages": [],
        "critic_proposal": None,
        "via_budget": False,
        "finalization_mismatch": False,
    }
