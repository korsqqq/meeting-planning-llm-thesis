# src/evaluation/__init__.py
"""Evaluation layer.

  * scorer - plan + oracle optimum -> Score (correctness only, CPU-only).

Efficiency metrics (satisfaction / 1k tokens) live at aggregation, where Score and
TokenUsage meet (RunResult), not here.
"""

from __future__ import annotations

from .scorer import score_plan

__all__ = ["score_plan"]
