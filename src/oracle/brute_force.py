# src/oracle/brute_force.py
"""Exhaustive cross-check of the CP-SAT oracle for small instances (n <= 6).

THESIS_DECISIONS.md section 3: "CP-SAT and brute-force must agree on n <= 6; a
mismatch means a bug in one of them -- stop and fix." This module is that second,
independent witness. It deliberately re-implements the route-feasibility rules
from scratch (it does NOT import them from solver.py) so the two agree only if the
shared task model -- not shared code -- is correct.

It enumerates every subset of people and every visiting order, and returns the
largest subset that admits a feasible schedule. Feasible for n <= 6 only; the
search is factorial and is meant purely as a correctness oracle for the CP-SAT
formulation, never for the main sweep.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, permutations

from src.schemas import AnswerContract, Instance, Meeting

__all__ = ["MAX_BRUTE_FORCE_N", "BruteForceSolution", "brute_force_solve"]

# Hard ceiling; above this the factorial enumeration is infeasible and CP-SAT is
# the only oracle (THESIS_DECISIONS.md section 3).
MAX_BRUTE_FORCE_N = 10


@dataclass(frozen=True)
class BruteForceSolution:
    """Result of the exhaustive search."""

    optimum: int                 # max number of people that can be met
    plan: AnswerContract         # a canonical optimal plan (first in enumeration order)


def _schedule(instance: Instance, order: Sequence[str]) -> list[int] | None:
    """Earliest feasible start time per stop, or None if `order` is infeasible.

    Independent re-implementation of the OPTW feasibility rules (see solver.py for
    the shared task model). Kept separate on purpose.
    """
    duration = instance.meeting_duration
    clock = instance.start_time
    here = instance.start_location
    starts: list[int] = []

    for pid in order:
        person = instance.person(pid)
        arrival = clock + instance.travel_time(here, person.location)

        if instance.waiting_allowed:
            begin = arrival if arrival >= person.window_start else person.window_start
        else:
            begin = arrival
            if begin < person.window_start:
                return None  # cannot wait for the window to open

        finish = begin + duration
        if finish > person.window_end:
            return None
        if finish > instance.end_of_day:
            return None

        starts.append(begin)
        clock = finish
        here = person.location

    return starts


def brute_force_solve(
    instance: Instance, max_n: int = MAX_BRUTE_FORCE_N
) -> BruteForceSolution:
    """Exhaustively find the optimum number of people that can be met.

    Enumeration is deterministic: subsets are taken over people sorted by id, and
    orderings in lexicographic permutation order, largest subset size first. The
    first feasible plan found at the optimum size is returned as the canonical plan.

    Raises ValueError if the instance has more than `max_n` people (the factorial
    search is then infeasible -- use the CP-SAT oracle instead).
    """
    person_ids = sorted(p.person_id for p in instance.people)
    n = len(person_ids)
    if n > max_n:
        raise ValueError(
            f"brute force supports at most {max_n} people, got {n}; use solve() instead"
        )

    for size in range(n, 0, -1):
        for subset in combinations(person_ids, size):
            for order in permutations(subset):
                sched = _schedule(instance, order)
                if sched is not None:
                    meetings = [
                        Meeting(person_id=pid, start_time=s)
                        for pid, s in zip(order, sched)
                    ]
                    return BruteForceSolution(
                        optimum=size, plan=AnswerContract(meetings=meetings)
                    )

    # No single meeting is feasible.
    return BruteForceSolution(optimum=0, plan=AnswerContract.empty())
