# src/agents/single_agent/__init__.py
"""Single-agent conditions: C1 (ReAct), C2 (verify/revise), C4 (planner+critic)
and C5 (Best-of-3)."""

from __future__ import annotations

from .best_of_3 import N_ATTEMPTS, run_best_of_3
from .planner_critic import run_planner_critic
from .react import ReactResult, run_react
from .react_verify_revise import MAX_REVISIONS, run_verify_revise

__all__ = [
    "run_react",
    "ReactResult",
    "run_verify_revise",
    "MAX_REVISIONS",
    "run_best_of_3",
    "N_ATTEMPTS",
    "run_planner_critic",
]
