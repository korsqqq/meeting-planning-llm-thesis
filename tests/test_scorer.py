# tests/test_scorer.py
"""Scorer tests (Layer 5). Correctness only -- no tokens involved.

Covers the satisfaction ratio, independent revalidation (invalid -> 0), the malformed
and empty-plan cases, the optimum == 0 degenerate rule, and the solver/validator
disagreement guard. One end-to-end check scores the oracle's own optimal plan.
"""

from __future__ import annotations

import pytest

from src.data import generate_instance
from src.evaluation import score_plan
from src.oracle import solve
from src.schemas import (
    AnswerContract,
    GeneratorParams,
    Instance,
    InvalidReason,
    Meeting,
    Person,
    TravelStructure,
)

_TRAVEL = {
    "S": {"S": 0, "A": 10, "B": 10},
    "A": {"S": 10, "A": 0, "B": 20},
    "B": {"S": 10, "A": 20, "B": 0},
}


def _instance(people: list[Person], *, end_of_day: int = 300) -> Instance:
    return Instance(
        instance_id="score-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=len(people), tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=end_of_day,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=people,
        travel_times=_TRAVEL,
    )


def _two_person() -> Instance:
    # Both reachable together -> optimum 2 (a@10..40, travel 20, b@60..90).
    return _instance([
        Person(person_id="a", location="A", window_start=0, window_end=100),
        Person(person_id="b", location="B", window_start=0, window_end=200),
    ])


def test_optimal_plan_scores_one() -> None:
    inst = _two_person()
    opt = solve(inst).optimum
    assert opt == 2
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="b", start_time=60)])
    s = score_plan(inst, plan, opt)
    assert s.valid and s.satisfaction == 1.0 and s.optimality
    assert s.n_valid_meetings == 2 and s.invalid_reasons == []


def test_suboptimal_valid_plan_partial_credit() -> None:
    inst = _two_person()
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10)])
    s = score_plan(inst, plan, solver_optimum=2)
    assert s.valid and s.satisfaction == 0.5 and not s.optimality
    assert s.n_valid_meetings == 1


def test_invalid_plan_scores_zero() -> None:
    inst = _two_person()
    # a ends at 120 > window_end 100.
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=90)])
    s = score_plan(inst, plan, solver_optimum=2)
    assert not s.valid and s.satisfaction == 0.0
    assert InvalidReason.WINDOW_VIOLATION in s.invalid_reasons


def test_malformed_plan_scores_zero() -> None:
    inst = _two_person()
    s = score_plan(inst, None, solver_optimum=2)
    assert not s.valid and s.satisfaction == 0.0
    assert s.invalid_reasons == [InvalidReason.MALFORMED]


def test_empty_plan_is_valid_but_zero_when_optimum_positive() -> None:
    inst = _two_person()
    s = score_plan(inst, AnswerContract.empty(), solver_optimum=2)
    assert s.valid and s.satisfaction == 0.0 and not s.optimality
    assert s.n_valid_meetings == 0


def test_degenerate_zero_optimum_scores_one() -> None:
    # x's window (20 min) cannot hold a 30-min meeting -> nothing feasible.
    inst = _instance([Person(person_id="x", location="A", window_start=0, window_end=20)])
    assert solve(inst).optimum == 0
    s = score_plan(inst, AnswerContract.empty(), solver_optimum=0)
    assert s.valid and s.satisfaction == 1.0 and s.optimality
    assert s.n_valid_meetings == 0


def test_achieved_above_optimum_raises() -> None:
    inst = _two_person()
    plan = AnswerContract(meetings=[Meeting(person_id="a", start_time=10),
                                    Meeting(person_id="b", start_time=60)])
    with pytest.raises(ValueError):
        score_plan(inst, plan, solver_optimum=1)  # valid 2-meeting plan vs claimed optimum 1


def test_end_to_end_oracle_plan_scores_one() -> None:
    inst = generate_instance(n_people=7, tightness=0.7, overlap=0.7,
                             travel_structure=TravelStructure.CLUSTERED, seed=3)
    sol = solve(inst)
    s = score_plan(inst, sol.plan, sol.optimum)
    assert s.valid and s.satisfaction == 1.0 and s.optimality
