# tests/test_mas_split_merge.py
"""Offline unit tests for the deterministic C3 components (no LLM, no client).

Covers the locked C3-1/C3-3/C3-4 data-side algorithms and their invariants
(THESIS_DECISIONS section 4, C3-3/C3-4 revised 2026-07-16 after design review):
the split is deterministic and a hard partition; sub-instances are faithful
restrictions (sub-valid => full-valid); the aggregator performs NO planning --
the draft is the one-fixed-order union with worker times unchanged and nothing
dropped (it may be invalid); the candidate pool is the union of proposed ids;
the critic prompt is pool-restricted (no people or locations a worker did not
surface) and free of oracle vocabulary; and the agent-side package can never
reach the solver (AST import scan).
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.agents.multi_agent.aggregator import build_draft, candidate_ids
from src.agents.multi_agent.prompts import critic_messages, serialise_candidates
from src.agents.multi_agent.supervisor import make_sub_instance, split_instance, split_people
from src.harness.sanity import ORACLE_TERMS
from src.oracle import is_valid
from src.schemas import (
    AnswerContract,
    GeneratorParams,
    Instance,
    Meeting,
    Person,
    TravelStructure,
)


# --------------------------------------------------------------------------- #
# Instance builders.
# --------------------------------------------------------------------------- #
def _travel(locs: list[str], pairs: dict[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    t = {a: {b: 0 for b in locs} for a in locs}
    for (a, b), d in pairs.items():
        t[a][b] = d
        t[b][a] = d
    return t


def _mk(
    people: list[Person],
    locations: list[str],
    travel: dict[str, dict[str, int]],
    *,
    waiting: bool = True,
    end: int = 480,
) -> Instance:
    return Instance(
        instance_id="mas-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=len(people), tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.CLUSTERED,
        ),
        start_location="S",
        start_time=0,
        end_of_day=end,
        meeting_duration=30,
        waiting_allowed=waiting,
        tie_break="earliest_start",
        locations=locations,
        people=people,
        travel_times=travel,
    )


def _geo_instance(*, waiting: bool = True) -> Instance:
    """Two clean geographic clusters: {p0, p1} near the depot, {p2, p3} far."""
    locs = ["S", "A1", "A2", "B1", "B2"]
    travel = _travel(locs, {
        ("S", "A1"): 10, ("S", "A2"): 10, ("A1", "A2"): 10,
        ("S", "B1"): 60, ("S", "B2"): 60, ("B1", "B2"): 10,
        ("A1", "B1"): 60, ("A1", "B2"): 60, ("A2", "B1"): 60, ("A2", "B2"): 60,
    })
    people = [
        Person(person_id="p0", location="A1", window_start=0, window_end=200),
        Person(person_id="p1", location="A2", window_start=0, window_end=200),
        Person(person_id="p2", location="B1", window_start=200, window_end=480),
        Person(person_id="p3", location="B2", window_start=200, window_end=480),
    ]
    return _mk(people, locs, travel, waiting=waiting)


def _far_instance() -> Instance:
    """One person near A, two chained people near B; A <-> B is unreachable."""
    locs = ["S", "A1", "B1", "B2"]
    travel = _travel(locs, {
        ("S", "A1"): 10, ("S", "B1"): 10, ("S", "B2"): 20,
        ("A1", "B1"): 200, ("A1", "B2"): 200, ("B1", "B2"): 10,
    })
    people = [
        Person(person_id="p0", location="A1", window_start=0, window_end=100),
        Person(person_id="p2", location="B1", window_start=0, window_end=100),
        Person(person_id="p3", location="B2", window_start=0, window_end=100),
    ]
    return _mk(people, locs, travel)


# --------------------------------------------------------------------------- #
# C3-1: split.
# --------------------------------------------------------------------------- #
def test_split_deterministic() -> None:
    inst = _geo_instance()
    a1, b1 = split_people(inst)
    a2, b2 = split_people(inst)
    assert [p.person_id for p in a1] == [p.person_id for p in a2]
    assert [p.person_id for p in b1] == [p.person_id for p in b2]


def test_split_is_hard_partition_matching_geography() -> None:
    inst = _geo_instance()
    a, b = split_people(inst)
    a_ids = {p.person_id for p in a}
    b_ids = {p.person_id for p in b}
    assert a_ids == {"p0", "p1"}
    assert b_ids == {"p2", "p3"}
    assert a_ids & b_ids == set()
    assert a_ids | b_ids == {p.person_id for p in inst.people}


def test_split_all_ties_go_to_cluster_a_with_deterministic_seeds() -> None:
    # Everyone at one location: all distances 0, so the seeds are the first
    # id-sorted pair (p0, p1) and every non-seed ties to cluster A.
    locs = ["S", "L"]
    travel = _travel(locs, {("S", "L"): 10})
    people = [
        Person(person_id=f"p{i}", location="L", window_start=0, window_end=480)
        for i in range(4)
    ]
    inst = _mk(people, locs, travel)
    a, b = split_people(inst)
    assert [p.person_id for p in a] == ["p0", "p2", "p3"]
    assert [p.person_id for p in b] == ["p1"]


def test_split_single_person_gives_empty_cluster_b() -> None:
    locs = ["S", "A1"]
    travel = _travel(locs, {("S", "A1"): 10})
    people = [Person(person_id="p0", location="A1", window_start=0, window_end=480)]
    sub_a, sub_b = split_instance(_mk(people, locs, travel))
    assert sub_a is not None and [p.person_id for p in sub_a.people] == ["p0"]
    assert sub_b is None


def test_sub_instance_is_faithful_restriction() -> None:
    inst = _geo_instance()
    a, _ = split_people(inst)
    sub = make_sub_instance(inst, a, "A")
    assert sub is not None
    assert sub.instance_id == "mas-test::A"
    # Scalar fields copied verbatim; annotations stay None.
    for f in ("seed", "start_location", "start_time", "end_of_day",
              "meeting_duration", "waiting_allowed", "tie_break", "generator_params"):
        assert getattr(sub, f) == getattr(inst, f)
    assert sub.complexity_metric is None and sub.level is None
    # Locations: depot + cluster locations, parent order preserved.
    assert sub.locations == ["S", "A1", "A2"]
    # Travel matrix: exact restriction of the parent.
    for x in sub.locations:
        for y in sub.locations:
            assert sub.travel_times[x][y] == inst.travel_times[x][y]


def test_subplan_valid_in_sub_is_valid_in_full() -> None:
    inst = _geo_instance()
    sub_a, sub_b = split_instance(inst)
    assert sub_a is not None and sub_b is not None
    plan_a = AnswerContract(meetings=[
        Meeting(person_id="p0", start_time=10), Meeting(person_id="p1", start_time=50),
    ])
    plan_b = AnswerContract(meetings=[
        Meeting(person_id="p2", start_time=200), Meeting(person_id="p3", start_time=240),
    ])
    assert is_valid(sub_a, plan_a) and is_valid(inst, plan_a)
    assert is_valid(sub_b, plan_b) and is_valid(inst, plan_b)


# --------------------------------------------------------------------------- #
# C3-3: aggregator = draft + candidate pool, NO planning.
# --------------------------------------------------------------------------- #
def _plan(*pairs: tuple[str, int]) -> AnswerContract:
    return AnswerContract(
        meetings=[Meeting(person_id=p, start_time=t) for p, t in pairs]
    )


def test_draft_is_one_fixed_order_with_worker_times_unchanged() -> None:
    # Single order: worker-assigned start_time, ties by person_id. The times are
    # exactly the workers' own; nothing is re-timed, nothing is dropped.
    draft = build_draft(_plan(("p1", 50), ("p0", 10)), _plan(("p2", 200), ("p3", 240)))
    assert [(m.person_id, m.start_time) for m in draft.meetings] == [
        ("p0", 10), ("p1", 50), ("p2", 200), ("p3", 240)
    ]
    # Tie on start_time -> person_id order; both meetings are kept even though
    # they collide in time (the draft is allowed to break the rules).
    tied = build_draft(_plan(("p2", 10)), _plan(("p0", 10)))
    assert [(m.person_id, m.start_time) for m in tied.meetings] == [
        ("p0", 10), ("p2", 10)
    ]


def test_draft_may_be_invalid_and_is_never_repaired() -> None:
    # p0 and p2/p3 are mutually unreachable: the union cannot be a feasible
    # route, and the aggregator must NOT fix that -- no drop, no re-timing.
    inst = _far_instance()
    draft = build_draft(_plan(("p0", 10)), _plan(("p2", 10), ("p3", 50)))
    assert [(m.person_id, m.start_time) for m in draft.meetings] == [
        ("p0", 10), ("p2", 10), ("p3", 50)
    ]
    assert not is_valid(inst, draft)  # stays broken: repair belongs to the critic


def test_draft_deterministic_and_pure() -> None:
    a, b = _plan(("p0", 10)), _plan(("p2", 10), ("p3", 50))
    assert build_draft(a, b) == build_draft(a, b)


def test_candidate_pool_is_union_of_proposed_ids() -> None:
    assert candidate_ids(_plan(("p0", 10)), _plan(("p2", 10), ("p3", 50))) == {
        "p0", "p2", "p3"
    }
    assert candidate_ids(_plan(), _plan()) == set()
    assert candidate_ids(_plan(("p1", 5)), _plan()) == {"p1"}


# --------------------------------------------------------------------------- #
# C3-4: the critic prompt is pool-restricted.
# --------------------------------------------------------------------------- #
def test_critic_prompt_contains_only_pool_people_and_locations() -> None:
    inst = _geo_instance()
    draft = _plan(("p0", 10), ("p2", 200))
    msgs = critic_messages(inst, draft, {"p0", "p2"})
    text = " ".join(m["content"] for m in msgs)
    # Pool people are present with their data...
    assert "p0 @ A1, 0-200" in text
    assert "p2 @ B1, 200-480" in text
    # ...people no worker proposed are invisible: no data line, no location.
    # ("p1 @" is the data-line shape; the bare "p1@60" in the fixed format
    # example of the instruction is a template artifact, not task data.)
    assert "p1 @" not in text
    assert "p3" not in text
    assert "A2" not in text
    assert "B2" not in text
    # The travel matrix is restricted to depot + candidate locations.
    assert "from S: S=0, A1=10, B1=60" in text
    assert "from A1: S=10, A1=0, B1=60" in text
    assert "from A2" not in text and "from B2" not in text
    # The draft is shown verbatim.
    assert "p0@10, p2@200" in text


def test_critic_prompt_free_of_oracle_vocabulary() -> None:
    inst = _geo_instance()
    msgs = critic_messages(inst, _plan(("p0", 10), ("p2", 200)), {"p0", "p2"})
    low = " ".join(m["content"] for m in msgs).lower()
    for term in ORACLE_TERMS:
        assert term not in low, term


def test_candidate_dump_is_deterministic() -> None:
    inst = _geo_instance()
    assert serialise_candidates(inst, {"p0", "p2"}) == serialise_candidates(inst, {"p2", "p0"})


# --------------------------------------------------------------------------- #
# Boundary invariant: no solver on the agent side.
# --------------------------------------------------------------------------- #
def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            mods.append(mod)
            mods.extend(f"{mod}.{alias.name}" for alias in node.names)
    return mods


def test_multi_agent_package_never_imports_the_solver() -> None:
    import src.agents.multi_agent as pkg

    pkg_dir = Path(pkg.__file__).parent
    allowed_oracle = {"src.oracle", "src.oracle.is_valid"}  # hidden validator gate only
    for py in sorted(pkg_dir.glob("*.py")):
        mods = _imports_of(py)
        for m in mods:
            assert "solver" not in m, f"{py.name} imports {m}"
            assert "brute_force" not in m, f"{py.name} imports {m}"
            assert "ortools" not in m, f"{py.name} imports {m}"
            if m.startswith("src.oracle"):
                assert m in allowed_oracle, f"{py.name} imports {m}"
        # The deterministic data-side modules must not touch the oracle at all.
        if py.name in {"supervisor.py", "aggregator.py", "prompts.py", "state.py",
                       "worker.py"}:
            assert not any(m.startswith("src.oracle") for m in mods), py.name