# src/data/complexity.py
"""Complexity annotation and pilot level-binning for generated instances.

THESIS_DECISIONS.md section 3: complexity is the count of binding pairwise conflict
pairs, MEASURED after the instance is solved, and the easy/medium/hard boundaries
are cut by quantiles over the measured distribution of a 200+ instance pilot -- not
set a priori.

This module connects the Layer-1 oracle to the Layer-0 Instance:

  * annotate_complexity   -> Instance with complexity_metric filled.
  * solve_and_annotate    -> also returns the OracleSolution (its optimum is needed
                             to drop optimum == 0 instances from the pilot).
  * level_boundaries      -> tertile cut points over a batch of metrics.
  * assign_level          -> Instance with level filled given those cut points.

Fields are filled functionally via model_copy (no mutation of the input).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from src.oracle import OracleSolution, count_conflict_pairs, solve
from src.schemas import Instance, Level

__all__ = [
    "annotate_complexity",
    "solve_and_annotate",
    "level_boundaries",
    "assign_level",
]


def annotate_complexity(instance: Instance) -> Instance:
    """Return a copy of `instance` with `complexity_metric` set (binding conflicts)."""
    metric = count_conflict_pairs(instance)
    return instance.model_copy(update={"complexity_metric": metric})


def solve_and_annotate(instance: Instance) -> tuple[Instance, OracleSolution]:
    """Solve for the optimum and annotate complexity in one pass.

    The optimum is returned (not stored on the Instance -- the schema has no such
    field) so the pilot can exclude optimum == 0 instances from the primary analysis.
    """
    solution = solve(instance)
    return annotate_complexity(instance), solution


def level_boundaries(metrics: Sequence[int]) -> tuple[int, int]:
    """Cut points (b1, b2) over the measured complexity distribution.

    Interpreted by `assign_level` as: metric <= b1 -> easy, <= b2 -> medium,
    else hard.

    The conflict distribution is strongly zero-inflated (roughly half of a broad
    pilot sweep has 0 binding conflicts), so a naive tertile over the whole sample
    collapses -- both cut points land on 0 and the medium bin vanishes. We therefore
    anchor easy on the 0-conflict mass (b1 = 0, consistent with the predefined easy-level anchor) and split the POSITIVE conflicts at their median into medium and hard
    (the quantile cut THESIS_DECISIONS.md section 3 prescribes, applied where it is
    not degenerate). Boundaries still come from the pilot data, not a priori.
    """
    if not metrics:
        raise ValueError("need at least one metric to compute boundaries")

    positives = sorted(m for m in metrics if m > 0)
    if not positives:
        return (0, 0)  # no conflicts anywhere -> everything is easy

    median_positive = int(statistics.median(positives))
    b2 = max(1, median_positive)  # ensure medium = [1, b2] is non-empty
    return (0, b2)


def assign_level(instance: Instance, boundaries: tuple[int, int]) -> Instance:
    """Return a copy of `instance` with `level` set from its complexity_metric."""
    if instance.complexity_metric is None:
        raise ValueError(
            f"instance {instance.instance_id!r} has no complexity_metric; "
            "call annotate_complexity first"
        )
    b1, b2 = boundaries
    metric = instance.complexity_metric
    if metric <= b1:
        level = Level.EASY
    elif metric <= b2:
        level = Level.MEDIUM
    else:
        level = Level.HARD
    return instance.model_copy(update={"level": level})
