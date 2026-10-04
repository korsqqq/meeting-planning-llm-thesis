# src/schemas/common.py
"""Shared enumerations for the Layer-0 schemas.

These are the fixed vocabularies referenced by the instance, answer, and
run-result schemas. Keeping them in one place prevents string drift (e.g. one
file writing "c1" and another "C1_ReAct").
"""

from __future__ import annotations

from enum import StrEnum


class Level(StrEnum):
    """Complexity level. Assigned to an instance AFTER it is solved and the
    pilot has fixed the bin boundaries (see THESIS_DECISIONS.md section 3)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Condition(StrEnum):
    """The experimental conditions (THESIS_DECISIONS.md section 4, C1-C5)."""

    C1_REACT = "c1_react"
    C2_VERIFY_REVISE = "c2_verify_revise"
    C3_MAS = "c3_mas"
    C4_PLANNER_CRITIC = "c4_planner_critic"  # optional, cut first if scope is tight
    # Validated Best-of-3 single-agent sampling (LOCKED 2026-07-26). The secondary
    # strong single-agent baseline: three independent C1 search trajectories under one
    # shared budget, deterministic zero-token selection, then ONE finalisation.
    C5_BEST_OF_3 = "c5_best_of_3"


class InvalidReason(StrEnum):
    """Hard-constraint violation types. A plan may carry several at once, so the
    run-result stores a list (multi-label invalid breakdown)."""

    MALFORMED = "malformed"                  # could not be parsed into AnswerContract
    WINDOW_VIOLATION = "window_violation"    # meeting outside a person's availability
    TRAVEL_INFEASIBLE = "travel_infeasible"  # not enough time to travel between meetings
    DUPLICATE_MEETING = "duplicate_meeting"  # same person met more than once
    OVERLAPPING_MEETING = "overlapping_meeting"
    DEADLINE_EXCEEDED = "deadline_exceeded"  # a meeting ends after end-of-day
    UNKNOWN_PERSON = "unknown_person"        # references a person not in the instance
