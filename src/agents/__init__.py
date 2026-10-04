# src/agents/__init__.py
"""Agent layer.

Layer 3 so far is `tools` -- the read-only Instance interface shared by every
condition (C1-C4). The single-agent and MAS implementations live in the
`single_agent` and `multi_agent` subpackages.
"""

from __future__ import annotations

from .tools import get_availability, get_travel_time, list_people

__all__ = ["list_people", "get_availability", "get_travel_time"]
