# src/data/__init__.py
"""Layer-2 data generation: deterministic OPTW instances and complexity binning.

  * generator   - build reproducible Instances from (knobs, seed).
  * complexity  - annotate the binding-conflict metric and cut pilot level bins.
"""

from __future__ import annotations

from .complexity import (
    annotate_complexity,
    assign_level,
    level_boundaries,
    solve_and_annotate,
)
from .generator import (
    DEFAULT_END_OF_DAY,
    DEFAULT_MEETING_DURATION,
    DEFAULT_START_TIME,
    generate_dataset,
    generate_instance,
)

__all__ = [
    "generate_instance",
    "generate_dataset",
    "DEFAULT_START_TIME",
    "DEFAULT_END_OF_DAY",
    "DEFAULT_MEETING_DURATION",
    "annotate_complexity",
    "solve_and_annotate",
    "level_boundaries",
    "assign_level",
]
