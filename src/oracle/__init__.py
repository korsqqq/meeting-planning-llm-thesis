# src/oracle/__init__.py
"""Layer-1 oracle: ground truth and hard-constraint validation.

Three independent components over the Layer-0 schemas:

  * solver        - CP-SAT optimum + the binding-conflict complexity metric.
  * brute_force   - exhaustive cross-check for n <= 6 (must agree with solver).
  * validator     - hidden hard-constraint checker for an agent's AnswerContract.

The solver is never exposed to the agents (THESIS_DECISIONS.md section 3); it is
used only for scoring and complexity measurement.
"""

from __future__ import annotations

from .brute_force import MAX_BRUTE_FORCE_N, BruteForceSolution, brute_force_solve
from .solver import (
    SOLVER_SEED,
    OracleSolution,
    conflict_graph,
    count_conflict_pairs,
    solve,
)
from .validator import is_valid, validate

__all__ = [
    "solve",
    "OracleSolution",
    "SOLVER_SEED",
    "conflict_graph",
    "count_conflict_pairs",
    "brute_force_solve",
    "BruteForceSolution",
    "MAX_BRUTE_FORCE_N",
    "validate",
    "is_valid",
]
