# tests/test_tools.py
"""Smoke tests for the shared agent tools (Layer 3).

One test per tool, plus the KeyError contract for the lookups. The tools are pure
functions of an Instance, so the fixture is a small hand-built instance with known
values.
"""

from __future__ import annotations

import pytest

from src.agents.tools import get_availability, get_travel_time, list_people
from src.schemas import GeneratorParams, Instance, Person, TravelStructure


def _instance() -> Instance:
    return Instance(
        instance_id="tools-smoke",
        seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=300,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=[
            Person(person_id="a", location="A", window_start=10, window_end=100),
            Person(person_id="b", location="B", window_start=20, window_end=200),
        ],
        travel_times={
            "S": {"S": 0, "A": 15, "B": 25},
            "A": {"S": 15, "A": 0, "B": 40},
            "B": {"S": 25, "A": 40, "B": 0},
        },
    )


def test_list_people() -> None:
    assert list_people(_instance()) == ["a", "b"]


def test_get_availability() -> None:
    assert get_availability(_instance(), "a") == {
        "person_id": "a",
        "location": "A",
        "window_start": 10,
        "window_end": 100,
    }
    with pytest.raises(KeyError):
        get_availability(_instance(), "unknown")


def test_get_travel_time() -> None:
    inst = _instance()
    assert get_travel_time(inst, "S", "B") == 25
    assert get_travel_time(inst, "A", "B") == 40
    assert get_travel_time(inst, "A", "A") == 0
    with pytest.raises(KeyError):
        get_travel_time(inst, "S", "Z")
