# src/agents/tools.py
"""Agent tools: the read-only interface every condition uses to inspect an Instance.

THESIS_DECISIONS.md section 4: all conditions (C1-C4) share exactly
these three tools -- list_people, get_availability, get_travel_time. They are pure
functions of an Instance: no global state, no parsing of a prompt string, no
condition-specific behaviour. Keeping them identical across conditions is an
experimental invariant -- a per-condition tool would confound the comparison, so
there is deliberately only one version of each.

The oracle is never reachable from here; these tools expose only the raw task an
agent is allowed to see (people, their windows and locations, travel times).
Returns are plain JSON-serialisable values so the harness can drop them into a
prompt or a tool-call response unchanged.
"""

from __future__ import annotations

from src.schemas import Instance

__all__ = ["list_people", "get_availability", "get_travel_time"]


def list_people(instance: Instance) -> list[str]:
    """Return the person_ids in the instance, in generation order."""
    return [p.person_id for p in instance.people]


def get_availability(instance: Instance, person_id: str) -> dict[str, str | int]:
    """Return one person's location and availability window.

    Keys mirror the schema field names exactly (`location`, `window_start`,
    `window_end`); `person_id` is echoed back for clarity. Raises KeyError if
    `person_id` is not in the instance.
    """
    person = instance.person(person_id)  # raises KeyError on an unknown id
    return {
        "person_id": person.person_id,
        "location": person.location,
        "window_start": person.window_start,
        "window_end": person.window_end,
    }


def get_travel_time(instance: Instance, from_loc: str, to_loc: str) -> int:
    """Return travel time in minutes from `from_loc` to `to_loc`.

    Raises KeyError if either location is absent from the instance's travel matrix.
    """
    return instance.travel_time(from_loc, to_loc)
