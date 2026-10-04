# src/agents/multi_agent/__init__.py
"""C3 -- hierarchical MAS: supervisor -> workers -> aggregator -> critic -> finalise.

Design locked in THESIS_DECISIONS section 4 (C3-1..C3-8). Entry point:
`run_hierarchical`, with the same call shape as the C1/C2 runners.
"""

from __future__ import annotations

from .hierarchical import (
    WORKER_SHARE_DEN,
    WORKER_SHARE_NUM,
    ReactResult,
    run_hierarchical,
)

__all__ = ["run_hierarchical", "ReactResult", "WORKER_SHARE_NUM", "WORKER_SHARE_DEN"]
