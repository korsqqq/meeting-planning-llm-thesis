# tests/test_generator.py
"""Layer-2 generator and complexity-binning tests.

Two things matter most:
  * determinism -- same (knobs, seed) reproduce a byte-identical instance, and
    every emitted instance is schema-valid and solvable by the oracle;
  * the knobs actually move the binding-conflict count, so the pilot gets a real
    easy -> hard spread (test_knobs_drive_complexity).
"""

from __future__ import annotations

import statistics

import pytest

from src.data import (
    annotate_complexity,
    assign_level,
    generate_dataset,
    generate_instance,
    level_boundaries,
    solve_and_annotate,
)
from src.oracle import count_conflict_pairs, is_valid, solve
from src.schemas import Instance, Level, TravelStructure

_STRUCTURES = list(TravelStructure)


def _easy_kwargs(seed: int) -> dict:
    return dict(
        n_people=8, tightness=0.0, overlap=0.0,
        travel_structure=TravelStructure.UNIFORM, seed=seed,
    )


def _hard_kwargs(seed: int) -> dict:
    return dict(
        n_people=8, tightness=1.0, overlap=1.0,
        travel_structure=TravelStructure.CLUSTERED, seed=seed,
    )


# --------------------------------------------------------------------------- #
# Determinism and well-formedness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("structure", _STRUCTURES)
def test_deterministic(structure: TravelStructure) -> None:
    a = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                          travel_structure=structure, seed=11)
    b = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                          travel_structure=structure, seed=11)
    assert a.model_dump() == b.model_dump()


def test_different_seed_changes_instance() -> None:
    a = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                          travel_structure=TravelStructure.UNIFORM, seed=1)
    b = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                          travel_structure=TravelStructure.UNIFORM, seed=2)
    assert a.model_dump() != b.model_dump()


@pytest.mark.parametrize("structure", _STRUCTURES)
def test_emitted_instance_is_wellformed(structure: TravelStructure) -> None:
    inst = generate_instance(n_people=7, tightness=0.6, overlap=0.4,
                             travel_structure=structure, seed=3)
    # Construction already ran the schema validators; check the extra invariants.
    assert inst.complexity_metric is None and inst.level is None
    assert inst.start_location in inst.locations
    assert len(inst.people) == 7
    # Travel matrix is complete with a zero diagonal.
    for a in inst.locations:
        assert set(inst.travel_times[a]) == set(inst.locations)
        assert inst.travel_times[a][a] == 0
    # The oracle accepts it and the optimum is in range; its plan is valid.
    sol = solve(inst)
    assert 0 <= sol.optimum <= 7
    assert sol.proven_optimal
    assert is_valid(inst, sol.plan)


def test_metric_structures_symmetric_random_need_not_be() -> None:
    metric = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                               travel_structure=TravelStructure.UNIFORM, seed=5)
    for a in metric.locations:
        for b in metric.locations:
            assert metric.travel_times[a][b] == metric.travel_times[b][a]

    rnd = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                            travel_structure=TravelStructure.RANDOM, seed=5)
    asymmetric = any(
        rnd.travel_times[a][b] != rnd.travel_times[b][a]
        for a in rnd.locations
        for b in rnd.locations
    )
    assert asymmetric  # a random matrix is essentially never symmetric


# --------------------------------------------------------------------------- #
# The knobs must drive complexity (the whole point of the generator).
# --------------------------------------------------------------------------- #
def test_knobs_drive_complexity() -> None:
    seeds = range(20)
    easy = [count_conflict_pairs(generate_instance(**_easy_kwargs(s))) for s in seeds]
    hard = [count_conflict_pairs(generate_instance(**_hard_kwargs(s))) for s in seeds]

    # Wide, staggered, uniform -> essentially no conflicts.
    assert statistics.mean(easy) < 1.0
    # Tight, fully-overlapping, clustered -> many conflicts.
    assert statistics.mean(hard) > statistics.mean(easy)
    assert max(hard) > 0


# --------------------------------------------------------------------------- #
# Complexity annotation and pilot level-binning.
# --------------------------------------------------------------------------- #
def test_annotate_sets_metric_without_mutating_input() -> None:
    inst = generate_instance(n_people=6, tightness=0.8, overlap=0.8,
                             travel_structure=TravelStructure.CLUSTERED, seed=4)
    annotated = annotate_complexity(inst)
    assert inst.complexity_metric is None            # input untouched
    assert annotated.complexity_metric == count_conflict_pairs(inst)


def test_solve_and_annotate_returns_optimum() -> None:
    inst = generate_instance(n_people=6, tightness=0.5, overlap=0.5,
                             travel_structure=TravelStructure.LINE, seed=9)
    annotated, sol = solve_and_annotate(inst)
    assert annotated.complexity_metric is not None
    assert sol.proven_optimal
    assert 0 <= sol.optimum <= 6


def test_level_boundaries_and_assignment() -> None:
    metrics = [0, 0, 0, 1, 2, 3, 5, 8, 13, 21]
    b1, b2 = level_boundaries(metrics)
    assert b1 <= b2
    # Smallest metric is easy, largest is hard.
    easy_inst = _fake_annotated(0)
    hard_inst = _fake_annotated(max(metrics))
    assert assign_level(easy_inst, (b1, b2)).level is Level.EASY
    assert assign_level(hard_inst, (b1, b2)).level is Level.HARD


def test_level_boundaries_degenerate_all_zero() -> None:
    assert level_boundaries([0, 0, 0, 0]) == (0, 0)
    inst = _fake_annotated(0)
    assert assign_level(inst, (0, 0)).level is Level.EASY


def test_assign_level_requires_annotation() -> None:
    inst = generate_instance(n_people=4, tightness=0.5, overlap=0.5,
                             travel_structure=TravelStructure.UNIFORM, seed=0)
    with pytest.raises(ValueError):
        assign_level(inst, (0, 1))


def test_generate_dataset_grid_size_and_determinism() -> None:
    kw = dict(
        n_people_values=[5, 6],
        tightness_values=[0.0, 1.0],
        overlap_values=[0.5],
        structures=[TravelStructure.UNIFORM, TravelStructure.CLUSTERED],
        seeds=[0, 1, 2],
    )
    ds1 = generate_dataset(**kw)
    ds2 = generate_dataset(**kw)
    assert len(ds1) == 2 * 2 * 1 * 2 * 3  # 24
    assert [i.instance_id for i in ds1] == [i.instance_id for i in ds2]
    assert len({i.instance_id for i in ds1}) == len(ds1)  # ids unique


def _fake_annotated(metric: int) -> Instance:
    """A minimal annotated instance just to exercise assign_level."""
    inst = generate_instance(n_people=3, tightness=0.5, overlap=0.5,
                             travel_structure=TravelStructure.UNIFORM, seed=0)
    return inst.model_copy(update={"complexity_metric": metric})
