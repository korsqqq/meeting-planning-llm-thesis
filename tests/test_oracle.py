# tests/test_oracle.py
"""Layer-1 oracle tests.

The load-bearing test is `test_cpsat_matches_brute_force`: THESIS_DECISIONS.md
section 3 requires the CP-SAT solver and the exhaustive brute force to agree on
n <= 6. The rest pin down the validator's reason mapping and the conflict-graph
complexity metric. The Layer-2 generator does not exist yet, so instances are
built locally here.
"""

from __future__ import annotations

import random

import pytest

from src.oracle import (
    brute_force_solve,
    conflict_graph,
    count_conflict_pairs,
    is_valid,
    solve,
    validate,
)
from src.schemas import (
    AnswerContract,
    GeneratorParams,
    Instance,
    InvalidReason,
    Meeting,
    Person,
    TravelStructure,
)


# --------------------------------------------------------------------------- #
# Instance builders (local; the real generator is Layer 2 / TODO).
# --------------------------------------------------------------------------- #
def build_instance(
    *,
    people: list[Person],
    travel_times: dict[str, dict[str, int]],
    locations: list[str],
    start_location: str = "S",
    start_time: int = 0,
    end_of_day: int = 300,
    duration: int = 30,
    waiting_allowed: bool = True,
    seed: int = 0,
    instance_id: str = "t",
) -> Instance:
    return Instance(
        instance_id=instance_id,
        seed=seed,
        generator_params=GeneratorParams(
            n_people=max(len(people), 1),
            tightness=0.5,
            overlap=0.5,
            travel_structure=TravelStructure.RANDOM,
        ),
        start_location=start_location,
        start_time=start_time,
        end_of_day=end_of_day,
        meeting_duration=duration,
        waiting_allowed=waiting_allowed,
        tie_break="earliest_start",
        locations=locations,
        people=people,
        travel_times=travel_times,
    )


def random_instance(seed: int, n_people: int, *, waiting_allowed: bool = True) -> Instance:
    """A random but well-formed small instance for the agreement check."""
    rng = random.Random(seed)
    n_loc = 4
    locations = [f"L{k}" for k in range(n_loc)]
    travel = {
        a: {b: (0 if a == b else rng.randint(5, 50)) for b in locations}
        for a in locations
    }
    duration = rng.choice([20, 30, 40])
    end_of_day = 360
    people = []
    for i in range(n_people):
        loc = rng.choice(locations)
        ws = rng.randint(0, 250)
        we = ws + duration + rng.randint(0, 120)  # window can always hold one meeting
        people.append(
            Person(person_id=f"p{i}", location=loc, window_start=ws, window_end=we)
        )
    return build_instance(
        people=people,
        travel_times=travel,
        locations=locations,
        start_location=locations[0],
        start_time=0,
        end_of_day=end_of_day,
        duration=duration,
        waiting_allowed=waiting_allowed,
        seed=seed,
        instance_id=f"rand-{seed}",
    )


# A fixed 4-location instance reused by the validator tests.
_TRAVEL_4 = {
    "S": {"S": 0, "A": 10, "B": 10, "C": 10},
    "A": {"S": 10, "A": 0, "B": 20, "C": 30},
    "B": {"S": 10, "A": 20, "B": 0, "C": 30},
    "C": {"S": 10, "A": 30, "B": 30, "C": 0},
}


def fixed_instance(*, waiting_allowed: bool = True) -> Instance:
    people = [
        Person(person_id="a", location="A", window_start=0, window_end=100),
        Person(person_id="b", location="B", window_start=0, window_end=200),
        Person(person_id="c", location="C", window_start=0, window_end=400),
    ]
    return build_instance(
        people=people,
        travel_times=_TRAVEL_4,
        locations=["S", "A", "B", "C"],
        end_of_day=300,
        duration=30,
        waiting_allowed=waiting_allowed,
    )


# --------------------------------------------------------------------------- #
# CP-SAT vs brute force (the agreement THESIS_DECISIONS.md section 3 mandates).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("waiting", [True, False])
@pytest.mark.parametrize("seed", range(30))
def test_cpsat_matches_brute_force(seed: int, waiting: bool) -> None:
    n = 2 + (seed % 5)  # 2..6 people
    inst = random_instance(seed, n, waiting_allowed=waiting)

    bf = brute_force_solve(inst)
    sol = solve(inst)

    assert sol.optimum == bf.optimum, f"seed={seed} waiting={waiting}"
    assert sol.proven_optimal

    # Both canonical plans must hit the optimum and pass the independent validator.
    assert len(sol.plan.meetings) == sol.optimum
    assert len(bf.plan.meetings) == bf.optimum
    assert is_valid(inst, sol.plan)
    assert is_valid(inst, bf.plan)


def test_solve_empty_people_is_zero() -> None:
    inst = build_instance(people=[], travel_times={"S": {"S": 0}}, locations=["S"])
    sol = solve(inst)
    assert sol.optimum == 0
    assert sol.plan.meetings == []


# --------------------------------------------------------------------------- #
# Validator: one targeted case per InvalidReason.
# --------------------------------------------------------------------------- #
def test_valid_plan_has_no_reasons() -> None:
    inst = fixed_instance()
    # a at 10 (arrival from S), then b at 60 (a ends 40, +20 travel A->B).
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="b", start_time=60)])
    assert validate(inst, plan) == []
    assert is_valid(inst, plan)


def test_empty_plan_is_valid() -> None:
    inst = fixed_instance()
    assert validate(inst, AnswerContract.empty()) == []


def test_malformed_when_answer_is_none() -> None:
    inst = fixed_instance()
    assert validate(inst, None) == [InvalidReason.MALFORMED]


def test_window_violation() -> None:
    inst = fixed_instance()
    # a ends at 120 > window_end 100.
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=90)])
    assert validate(inst, plan) == [InvalidReason.WINDOW_VIOLATION]


def test_deadline_exceeded() -> None:
    inst = fixed_instance()
    # c ends at 320 > end_of_day 300 but inside its window [0, 400].
    plan = AnswerContract(meetings=[Meeting(person_id="c", start_time=290)])
    assert validate(inst, plan) == [InvalidReason.DEADLINE_EXCEEDED]


def test_travel_infeasible() -> None:
    inst = fixed_instance()
    # a ends 40, travel A->B is 20 -> b cannot start before 60; 50 is too early.
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="b", start_time=50)])
    assert validate(inst, plan) == [InvalidReason.TRAVEL_INFEASIBLE]


def test_overlapping_meeting() -> None:
    inst = fixed_instance()
    # b starts at 30 before a ends at 40.
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="b", start_time=30)])
    assert validate(inst, plan) == [InvalidReason.OVERLAPPING_MEETING]


def test_duplicate_meeting() -> None:
    inst = fixed_instance()
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="a", start_time=50)])
    assert validate(inst, plan) == [InvalidReason.DUPLICATE_MEETING]


def test_unknown_person() -> None:
    inst = fixed_instance()
    plan = AnswerContract(meetings=[Meeting(person_id="zzz", start_time=10)])
    assert validate(inst, plan) == [InvalidReason.UNKNOWN_PERSON]


def test_no_waiting_rejects_idle_time() -> None:
    inst = fixed_instance(waiting_allowed=False)
    # Arrival at A is 10; with waiting forbidden the start must equal 10.
    assert validate(inst, AnswerContract(meetings=[Meeting(person_id="a", start_time=10)])) == []
    assert validate(
        inst, AnswerContract(meetings=[Meeting(person_id="a", start_time=15)])
    ) == [InvalidReason.TRAVEL_INFEASIBLE]


def test_multi_label_reasons() -> None:
    inst = fixed_instance()
    # Unknown person AND a window violation on a known one -> two labels, sorted.
    plan = AnswerContract(meetings=[Meeting(person_id="zzz", start_time=10),
                                    Meeting(person_id="a", start_time=90)])
    assert validate(inst, plan) == sorted(
        [InvalidReason.UNKNOWN_PERSON, InvalidReason.WINDOW_VIOLATION],
        key=lambda r: r.value,
    )


# --------------------------------------------------------------------------- #
# Conflict-graph complexity metric.
# --------------------------------------------------------------------------- #
def test_conflict_pair_detected() -> None:
    # a@A and b@B each reachable alone, but A->B (and B->A) travel of 100 with
    # windows ending at 40 makes meeting both impossible.
    travel = {
        "S": {"S": 0, "A": 10, "B": 10},
        "A": {"S": 10, "A": 0, "B": 100},
        "B": {"S": 10, "A": 100, "B": 0},
    }
    people = [
        Person(person_id="a", location="A", window_start=0, window_end=40),
        Person(person_id="b", location="B", window_start=0, window_end=40),
    ]
    inst = build_instance(
        people=people, travel_times=travel, locations=["S", "A", "B"], duration=30
    )
    assert conflict_graph(inst) == {frozenset({"a", "b"})}
    assert count_conflict_pairs(inst) == 1


def test_no_conflicts_when_compatible() -> None:
    # Both at the same location, wide windows -> can meet both, no conflict.
    travel = {"S": {"S": 0, "A": 10}, "A": {"S": 10, "A": 0}}
    people = [
        Person(person_id="a", location="A", window_start=0, window_end=300),
        Person(person_id="b", location="A", window_start=0, window_end=300),
    ]
    inst = build_instance(
        people=people, travel_times=travel, locations=["S", "A"], duration=30
    )
    assert count_conflict_pairs(inst) == 0


def test_individually_infeasible_person_not_a_conflict() -> None:
    # x cannot be met even alone (window too short for a 30-min meeting); it must
    # not create a conflict edge with the feasible person a.
    travel = {"S": {"S": 0, "A": 10}, "A": {"S": 10, "A": 0}}
    people = [
        Person(person_id="a", location="A", window_start=0, window_end=300),
        Person(person_id="x", location="A", window_start=0, window_end=20),
    ]
    inst = build_instance(
        people=people, travel_times=travel, locations=["S", "A"], duration=30
    )
    assert count_conflict_pairs(inst) == 0
