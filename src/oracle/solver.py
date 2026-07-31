# src/oracle/solver.py
"""CP-SAT oracle for the meeting-planning task (Orienteering Problem with Time
Windows, OPTW).

Ground truth only. Per THESIS_DECISIONS.md section 3 the solver is NEVER shown
to the agents; it serves two roles:

  * `solve`               -> the optimum (max number of people that can be met),
                             used as the denominator of the satisfaction rate.
  * `count_conflict_pairs`/`conflict_graph` -> the complexity metric (number of
                             binding pairwise conflict pairs), measured AFTER the
                             instance is built and written into Instance.complexity_metric.

Task model (single source of truth, mirrored independently by brute_force.py and
validator.py):

  * A tour starts at `start_location` at `start_time`.
  * Every person carries prize 1; the objective is to maximise the count of people
    met (an Orienteering Problem, not a TSP -- the route need not return to start
    and need not visit everyone).
  * A meeting for person p occupies [s, s + meeting_duration]; it is feasible iff
        window_start[p] <= s        and    s + meeting_duration <= window_end[p]
        s + meeting_duration <= end_of_day.
  * Travelling from the previous stop (the depot for the first meeting) to p takes
    travel_times[prev_loc][p.location] minutes; the next meeting cannot start
    before arrival.
  * `waiting_allowed` controls idle time. True  -> s >= arrival (the traveller may
    wait for the window to open). False -> s == arrival (no idling; the schedule is
    forced by the route).

Determinism: fixed solver seed, single search worker. The solver seed (0) is
deliberately distinct from the generator seed and the inference seed, and all
three are logged separately as part of the reproducibility protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from ortools.sat.python import cp_model

from src.schemas import AnswerContract, Instance, Meeting

__all__ = [
    "SOLVER_SEED",
    "OracleSolution",
    "solve",
    "conflict_graph",
    "count_conflict_pairs",
]

# Ground-truth seed. Distinct from generator seed and inference seed on purpose.
SOLVER_SEED = 0

# Wall-clock ceiling for the (small) ground-truth models. The oracle must return a
# PROVEN optimum; if a model fails to close within this budget we raise rather than
# silently log a non-optimal denominator. The proof of the visit-count UPPER bound
# is the bottleneck: it grows sharply past ~9-10 people on clustered, medium-window
# instances (few pairwise conflicts but a binding global time budget). At the
# recommended size (n <= 9, see generator.RECOMMENDED_MAX_N) the worst single-worker
# case is ~8s, so 60s leaves margin and still fails fast on an intractable one.
_MAX_SOLVE_SECONDS = 60.0

# Single search worker. A multi-worker portfolio is faster but returns DIFFERENT
# optimal solutions across runs (only the objective value is guaranteed stable),
# which breaks reproducibility of the canonical optimal plan. One worker is fully
# deterministic -- identical optimum AND identical plan across runs and machines --
# and the cuts in solve() keep it fast enough at the recommended instance size.
_NUM_WORKERS = 1


@dataclass(frozen=True)
class OracleSolution:
    """Result of solving one instance for ground truth."""

    optimum: int                 # max number of people that can be met
    plan: AnswerContract         # one optimal plan (earliest-start canonical schedule)
    status: str                  # CP-SAT status name ("OPTIMAL", "FEASIBLE", ...)
    proven_optimal: bool         # True iff CP-SAT proved optimality


def _earliest_schedule(
    instance: Instance, order: Sequence[str]
) -> list[int] | None:
    """Earliest feasible start time for each stop on a fixed visiting `order`.

    Returns the list of start times, or None if the route is infeasible. Earliest
    start is the dominant choice for time-window feasibility (it leaves the most
    slack for later stops), so this is an exact feasibility test for the ordered
    route, used both to build a canonical optimal plan and to test pairwise
    conflicts.
    """
    duration = instance.meeting_duration
    t = instance.start_time
    prev_loc = instance.start_location
    starts: list[int] = []
    for pid in order:
        p = instance.person(pid)
        arrival = t + instance.travel_time(prev_loc, p.location)
        if instance.waiting_allowed:
            s = max(arrival, p.window_start)
        else:
            # No idling: the start is forced to the arrival time.
            s = arrival
            if s < p.window_start:
                return None  # arrived before the window opens and cannot wait
        end = s + duration
        if end > p.window_end:
            return None      # window cannot contain the meeting
        if end > instance.end_of_day:
            return None      # past the hard deadline
        starts.append(s)
        t = end
        prev_loc = p.location
    return starts


def _is_metric(instance: Instance) -> bool:
    """True iff the travel matrix obeys the triangle inequality.

    Needed to know whether pairwise-conflict cuts are valid: under the triangle
    inequality, two people who cannot be met as a standalone pair cannot coexist in
    any larger tour either (intermediate stops only add travel), so visit[i] +
    visit[j] <= 1 is a sound cut. A non-metric matrix (TravelStructure.RANDOM) can
    break that, so the cuts are skipped there.
    """
    locs = instance.locations
    tt = instance.travel_time
    for a in locs:
        for c in locs:
            ac = tt(a, c)
            for b in locs:
                if ac > tt(a, b) + tt(b, c):
                    return False
    return True


def _pair_infeasible(instance: Instance, a: str, b: str) -> bool:
    """True iff neither ordering of the pair {a, b} admits a feasible schedule."""
    return (
        _earliest_schedule(instance, [a, b]) is None
        and _earliest_schedule(instance, [b, a]) is None
    )


def solve(instance: Instance) -> OracleSolution:
    """Solve one instance for the optimum number of people that can be met.

    The objective is the count of visited people (each prize 1). Returns the
    optimum, a canonical earliest-start optimal plan, and the proof status.
    """
    people = instance.people
    n = len(people)

    if n == 0:
        return OracleSolution(
            optimum=0,
            plan=AnswerContract.empty(),
            status="OPTIMAL",
            proven_optimal=True,
        )

    duration = instance.meeting_duration
    start_time = instance.start_time
    end_of_day = instance.end_of_day
    depot_loc = instance.start_location

    model = cp_model.CpModel()

    # Node 0 is the depot; nodes 1..n are the people (node i <-> people[i - 1]).
    visit: dict[int, cp_model.IntVar] = {}
    start: dict[int, cp_model.IntVar] = {}

    arcs: list[tuple[int, int, cp_model.IntVar]] = []

    # Depot self-loop: selected iff no person is visited (empty tour).
    no_visits = model.NewBoolVar("no_visits")
    arcs.append((0, 0, no_visits))

    for i, p in enumerate(people, start=1):
        vi = model.NewBoolVar(f"visit_{p.person_id}")
        visit[i] = vi
        # AddCircuit self-loop literal true => node skipped, so it is visit.Not().
        arcs.append((i, i, vi.Not()))

        # Earliest possible meeting start: going directly from the depot first is
        # the soonest j can ever begin (any other route only delays it). Valid
        # regardless of the triangle inequality, and it tightens the domain.
        lo = max(p.window_start, start_time + instance.travel_time(depot_loc, p.location))
        hi = min(p.window_end - duration, end_of_day - duration)
        if hi < lo:
            # The meeting cannot fit even in isolation; forbid visiting.
            model.Add(vi == 0)
            start[i] = model.NewIntVar(lo, lo, f"start_{p.person_id}")
        else:
            start[i] = model.NewIntVar(lo, hi, f"start_{p.person_id}")

    # Travel on the timed arcs (depot -> person and person -> person, but NOT the
    # untimed return-to-depot). Used for the redundant time-budget bound below.
    timed_arc_terms: list[tuple[int, cp_model.IntVar]] = []

    # Depot -> person and person -> depot arcs.
    for i, p in enumerate(people, start=1):
        a0i = model.NewBoolVar(f"arc_0_{i}")
        arcs.append((0, i, a0i))
        tt = instance.travel_time(depot_loc, p.location)
        if instance.waiting_allowed:
            model.Add(start[i] >= start_time + tt).OnlyEnforceIf(a0i)
        else:
            model.Add(start[i] == start_time + tt).OnlyEnforceIf(a0i)
        timed_arc_terms.append((tt, a0i))

        # Return to depot closes the circuit; no timing constraint on the return.
        ai0 = model.NewBoolVar(f"arc_{i}_0")
        arcs.append((i, 0, ai0))

    # Person -> person arcs.
    for i, p in enumerate(people, start=1):
        for j, q in enumerate(people, start=1):
            if i == j:
                continue
            aij = model.NewBoolVar(f"arc_{i}_{j}")
            arcs.append((i, j, aij))
            tt = instance.travel_time(p.location, q.location)
            if instance.waiting_allowed:
                model.Add(start[j] >= start[i] + duration + tt).OnlyEnforceIf(aij)
            else:
                model.Add(start[j] == start[i] + duration + tt).OnlyEnforceIf(aij)
            timed_arc_terms.append((tt, aij))

    model.AddCircuit(arcs)

    # Redundant global time-budget bound. Along any route the elapsed time is
    # sum(timed travel) + duration * (#visited) + waiting, and the last meeting ends
    # by end_of_day, so (waiting >= 0):
    #     sum(timed travel) + duration * (#visited) <= end_of_day - start_time.
    # Valid in both waiting modes; it bounds the visit count directly and lets CP-SAT
    # prove optimality fast on loose-window instances where pairwise conflicts are 0
    # but the global deadline is the binding constraint.
    day_span = end_of_day - start_time
    model.Add(
        sum(tt * lit for tt, lit in timed_arc_terms)
        + duration * sum(visit.values())
        <= day_span
    )

    # Knapsack cut on the visit count. Every visited person j has exactly one
    # incoming timed arc whose travel is at least min_in[j] (the cheapest possible
    # predecessor -> j hop), so
    #     sum_j min_in[j] * visit[j] + duration * sum(visit) <= day_span.
    # This is a pure 0/1 constraint on the visit variables, which sharpens the LP
    # bound on the objective directly (the global arc bound above needs the arc
    # variables to do the same work).
    min_in: dict[int, int] = {}
    for i, p in enumerate(people, start=1):
        loc = p.location
        candidates = [instance.travel_time(depot_loc, loc)]
        candidates += [
            instance.travel_time(q.location, loc)
            for k, q in enumerate(people, start=1)
            if k != i
        ]
        min_in[i] = min(candidates)
    model.Add(
        sum(min_in[i] * visit[i] for i in visit)
        + duration * sum(visit.values())
        <= day_span
    )

    # Pairwise-conflict cuts (valid only when the matrix is metric -- see
    # _is_metric). For each pair that cannot be met together in any order, forbid
    # visiting both. These encode the binding window/travel conflicts directly and
    # are what lets CP-SAT prove the count bound quickly on clustered instances.
    if _is_metric(instance):
        for i in range(1, n + 1):
            pid_i = people[i - 1].person_id
            for j in range(i + 1, n + 1):
                pid_j = people[j - 1].person_id
                if _pair_infeasible(instance, pid_i, pid_j):
                    model.Add(visit[i] + visit[j] <= 1)

    model.Maximize(sum(visit.values()))

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = SOLVER_SEED
    # One worker -> fully reproducible optimum and plan (see _NUM_WORKERS). The
    # redundant time bound, knapsack cut and conflict cuts above are what keep the
    # single-worker proof fast.
    solver.parameters.num_search_workers = _NUM_WORKERS
    solver.parameters.max_time_in_seconds = _MAX_SOLVE_SECONDS

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"oracle failed on instance {instance.instance_id!r}: status {status_name}"
        )
    if status != cp_model.OPTIMAL:
        # A non-proven optimum would corrupt the satisfaction denominator.
        raise RuntimeError(
            f"oracle did not prove optimality on instance {instance.instance_id!r} "
            f"within {_MAX_SOLVE_SECONDS}s (status {status_name})"
        )

    visited = [
        (i, solver.Value(start[i]))
        for i in visit
        if solver.Value(visit[i]) == 1
    ]
    # Meetings are strictly time-ordered (duration > 0), so sorting by start gives
    # the visiting order.
    visited.sort(key=lambda pair: pair[1])
    order = [people[i - 1].person_id for i, _ in visited]
    optimum = len(order)

    # Canonical plan: earliest-start schedule for the chosen order. Falls back to
    # the raw solver assignment only if the recomputation disagrees (it should not).
    sched = _earliest_schedule(instance, order)
    if sched is None:
        meetings = [
            Meeting(person_id=people[i - 1].person_id, start_time=solver.Value(start[i]))
            for i, _ in visited
        ]
    else:
        meetings = [
            Meeting(person_id=pid, start_time=s) for pid, s in zip(order, sched)
        ]

    return OracleSolution(
        optimum=optimum,
        plan=AnswerContract(meetings=meetings),
        status=status_name,
        proven_optimal=True,
    )


def conflict_graph(instance: Instance) -> set[frozenset[str]]:
    """The binding pairwise conflict graph (THESIS_DECISIONS.md section 3).

    An edge {a, b} exists iff both a and b are individually reachable (each can be
    met alone) yet no ordering of the pair is feasible -- the conflict is the thing
    that binds them. Pairs where one endpoint is individually infeasible are not
    counted; that is an isolated infeasibility, not a pairwise conflict.

    Feasibility uses the same earliest-start schedule as `solve`, so the metric is
    solver-verified.
    """
    feasible_alone = {
        p.person_id
        for p in instance.people
        if _earliest_schedule(instance, [p.person_id]) is not None
    }

    edges: set[frozenset[str]] = set()
    for a, b in combinations(sorted(feasible_alone), 2):
        both_possible = (
            _earliest_schedule(instance, [a, b]) is not None
            or _earliest_schedule(instance, [b, a]) is not None
        )
        if not both_possible:
            edges.add(frozenset((a, b)))
    return edges


def count_conflict_pairs(instance: Instance) -> int:
    """Number of binding pairwise conflict pairs -- the raw complexity metric."""
    return len(conflict_graph(instance))
