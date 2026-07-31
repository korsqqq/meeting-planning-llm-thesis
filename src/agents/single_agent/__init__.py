# src/agents/single_agent/__init__.py
"""Single-agent conditions: C1 (ReAct) and C2 (verify/revise)."""

from __future__ import annotations

from .react import ReactResult, run_react
from .react_verify_revise import MAX_REVISIONS, run_verify_revise

__all__ = ["run_react", "ReactResult", "run_verify_revise", "MAX_REVISIONS"]
