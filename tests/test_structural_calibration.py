# tests/test_structural_calibration.py
"""Offline tests for the structural-calibration collector. CPU only, no LLM, no endpoint.

The collector's job is to measure and not to decide, so the tests fall into three groups:
the graph quantities are exact and are checked against hand-computed cases; the guards that
keep a metric from reporting a number it never computed; and the seed-range gate that stops
the held-out main range being consumed by a typo.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.collect_structural_calibration import (
    CALIBRATION_SEED_MAX,
    CALIBRATION_SEED_MIN,
    CalibrationRangeError,
    PoolAuditError,
    Row,
    audit_pool,
    independence_number,
    main,
    measure,
    triangle_violations,
)
from src.data.generator import generate_instance
from src.schemas import TravelStructure


def inst(**kw):
    base = dict(n_people=8, tightness=1.0, overlap=1.0,
                travel_structure=TravelStructure.CLUSTERED, seed=CALIBRATION_SEED_MIN)
    base.update(kw)
    return generate_instance(**base)


# --------------------------------------------------------------------------- #
# Graph quantities: exact at n = 8, so they are checked exactly.
# --------------------------------------------------------------------------- #
def test_independence_number_on_an_empty_graph():
    alpha, count = independence_number(["a", "b", "c"], set())
    assert (alpha, count) == (3, 1)


def test_independence_number_on_a_complete_graph():
    v = ["a", "b", "c"]
    edges = {frozenset(("a", "b")), frozenset(("a", "c")), frozenset(("b", "c"))}
    alpha, count = independence_number(v, edges)
    assert (alpha, count) == (1, 3)


def test_independence_number_counts_every_maximum_set():
    # Path a-b-c: maximum independent sets are {a,c} only.
    edges = {frozenset(("a", "b")), frozenset(("b", "c"))}
    assert independence_number(["a", "b", "c"], edges) == (2, 1)
    # Two disjoint edges: {a,c}, {a,d}, {b,c}, {b,d} all have size 2.
    edges = {frozenset(("a", "b")), frozenset(("c", "d"))}
    assert independence_number(["a", "b", "c", "d"], edges) == (2, 4)


def test_independence_number_ignores_edges_outside_the_vertex_set():
    """Vertices are the reachable people; an edge touching anyone else must not count."""
    edges = {frozenset(("a", "b")), frozenset(("a", "zz"))}
    alpha, _ = independence_number(["a", "b"], edges)
    assert alpha == 1  # only a-b constrains


def test_a_star_graph_keeps_alpha_high():
    """The hub case the concentration diagnostic exists for: high density, trivial answer."""
    v = list("abcde")
    edges = {frozenset(("a", x)) for x in "bcde"}
    alpha, _ = independence_number(v, edges)
    assert alpha == 4  # drop the hub and everything else is compatible


# --------------------------------------------------------------------------- #
# Guards against a metric reporting a number it never computed.
# --------------------------------------------------------------------------- #
def test_triangle_check_actually_walks_the_nested_matrix():
    """travel_times is `from -> to -> minutes`; a tuple-keyed read silently checks nothing."""
    bad, checked = triangle_violations(inst())
    assert checked > 0
    assert 0 <= bad <= checked


def test_triangle_check_detects_a_real_violation():
    i = inst()
    locs = sorted(i.travel_times)
    a, b, c = locs[0], locs[1], locs[2]
    i.travel_times[a][b] = 1
    i.travel_times[b][c] = 1
    i.travel_times[a][c] = 500
    bad, checked = triangle_violations(i)
    assert checked > 0 and bad >= 1


def test_random_structure_violates_far_more_than_the_position_derived_ones():
    """The generator specifies RANDOM as non-metric; this holds it to that."""
    def rate(structure):
        i = inst(travel_structure=structure)
        bad, checked = triangle_violations(i)
        return bad / checked
    assert rate(TravelStructure.RANDOM) > rate(TravelStructure.CLUSTERED)


def test_measure_refuses_a_matrix_it_could_not_traverse(monkeypatch):
    monkeypatch.setattr("scripts.collect_structural_calibration.triangle_violations",
                        lambda _i: (0, 0))
    with pytest.raises(ValueError, match="examined no triples"):
        measure(inst())


# --------------------------------------------------------------------------- #
# The row itself.
# --------------------------------------------------------------------------- #
def test_measure_produces_a_consistent_row():
    row = measure(inst())
    assert isinstance(row, Row)
    assert row.n_people == 8
    assert row.conflicting_pairs == round(row.D * 28)
    assert row.higher_order_gap_H == row.alpha_reachable - row.oracle_optimum
    assert row.alpha_reachable <= row.individually_reachable_count
    assert row.oracle_optimum <= row.n_people
    assert row.largest_component <= row.individually_reachable_count


def test_both_denominators_are_recorded():
    """D over C(8,2) is the axis; D_reachable exists because the two can disagree."""
    row = measure(inst())
    assert row.D is not None
    assert row.D_reachable is None or row.D_reachable >= row.D


def test_edge_shares_are_undefined_rather_than_zero_without_edges():
    row = measure(inst(tightness=0.2, overlap=0.2))
    if row.conflicting_pairs == 0:
        assert row.edge_share_top1 is None and row.edge_share_top2 is None
        assert row.max_degree == 0


# --------------------------------------------------------------------------- #
# Seed range: the held-out set must not be consumable by a typo.
# --------------------------------------------------------------------------- #
def test_the_held_out_range_is_refused(tmp_path):
    with pytest.raises(CalibrationRangeError, match="reserved structural_calibration range"):
        main(["--seed-start", "100000", "--n-seeds", "1", "--out", str(tmp_path)])


def test_only_the_frozen_held_out_blocks_have_a_name_to_ask_for():
    """This test used to assert that NO range reached held-out, which was the correct
    guarantee while the range was still locked. The amendment of 2026-08-26 opened it, and
    the guarantee it replaces is narrower rather than absent: exactly the four frozen blocks
    are nameable, the unused reserve 140000-199999 is still not, and no invented name works.

    The point of the original tripwire survives -- reaching held-out still requires a name
    that only a recorded decision puts in the table, not a --seed-start.
    """
    from scripts.collect_structural_calibration import SEED_RANGES
    held_out = {k: v for k, v in SEED_RANGES.items() if v[1] >= 100000}
    assert held_out == {
    "held_out_n8": (100_000, 109_999),
    "held_out_n4": (110_000, 119_999),
    "held_out_n5": (120_000, 129_999),
    "held_out_n6": (130_000, 139_999),
}
    assert max(hi for _lo, hi in SEED_RANGES.values()) < 140000, (
        "140000-199999 is unused reserve and must stay unnameable")
    for invented in ("main_heldout", "held_out", "held_out_reserve"):
        with pytest.raises(SystemExit):
            main(["--range", invented, "--n-seeds", "1"])


def test_the_budget_dev_range_is_reachable_and_bounded(tmp_path):
    from scripts.collect_structural_calibration import SEED_RANGES
    assert SEED_RANGES["budget_dev"] == (30000, 39999)
    with pytest.raises(CalibrationRangeError, match="reserved budget_dev range"):
        main(["--range", "budget_dev", "--seed-start", "39998", "--n-seeds", "5",
              "--out", str(tmp_path)])


def test_running_off_the_end_of_the_range_is_refused(tmp_path):
    with pytest.raises(CalibrationRangeError):
        main(["--seed-start", str(CALIBRATION_SEED_MAX), "--n-seeds", "5",
              "--out", str(tmp_path)])


def test_a_small_pool_is_collected_and_decides_nothing(tmp_path):
    rc = main(["--n-seeds", "1", "--tightness", "0.2,1.0", "--overlap", "1.0",
               "--structures", "clustered", "--progress-every", "0", "--out", str(tmp_path)])
    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "candidates.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    # No band, level or admissibility verdict may appear anywhere in the output.
    header = set(rows[0])
    assert not {"level", "band", "D_band", "admitted", "excluded"} & header
    assert "D" in header and "higher_order_gap_H" in header
    assert (tmp_path / "pool_meta.json").exists()


def test_the_collector_does_not_reference_the_agent_or_client_stack():
    """A regression guard, not a proof: a transitive import could still reintroduce it."""
    src = Path("scripts/collect_structural_calibration.py").read_text(encoding="utf-8")
    for forbidden in ("src.agents", "llm_client", "LLMClient", "openai"):
        assert forbidden not in src


# --------------------------------------------------------------------------- #
# Completeness audit of the pool.
# --------------------------------------------------------------------------- #
def _row(**kw) -> Row:
    base = dict(
        instance_id="i", seed=CALIBRATION_SEED_MIN, n_people=8, tightness=1.0, overlap=1.0,
        travel_structure="clustered", individually_reachable_count=8, conflicting_pairs=4,
        D=0.142857, D_reachable=0.142857, oracle_optimum=5, optimum_over_n=0.625,
        proven_optimal=True, solver_status="OPTIMAL", alpha_reachable=6,
        higher_order_gap_H=1, n_max_independent_sets=2, max_degree=2,
        edge_share_top1=0.5, edge_share_top2=0.75, largest_component=3,
        triangle_violations=3, triangle_triples_checked=504, solve_seconds=0.2,
    )
    base.update(kw)
    return Row(**base)


def test_audit_accepts_a_complete_pool():
    rows = [_row(instance_id=f"i{i}", seed=CALIBRATION_SEED_MIN + i) for i in range(3)]
    out = audit_pool(rows, expected=3, n_people=8,
                     seeds=[CALIBRATION_SEED_MIN + i for i in range(3)])
    assert out["all_checks_passed"] is True
    assert out["triangle_triples_per_instance"] == 504


def test_audit_rejects_a_short_pool():
    rows = [_row(instance_id="i0")]
    with pytest.raises(PoolAuditError, match="expected 2 candidates"):
        audit_pool(rows, expected=2, n_people=8, seeds=[CALIBRATION_SEED_MIN])


def test_audit_rejects_duplicate_knob_keys():
    rows = [_row(instance_id="a"), _row(instance_id="b")]  # same seed/knobs
    with pytest.raises(PoolAuditError, match="duplicate"):
        audit_pool(rows, expected=2, n_people=8, seeds=[CALIBRATION_SEED_MIN])


def test_audit_rejects_an_unproven_optimum():
    """At n<=9 this must not happen; if it does, the tractability assumption broke."""
    rows = [_row(instance_id="a", proven_optimal=False, solver_status="FEASIBLE")]
    with pytest.raises(PoolAuditError, match="unproven optim"):
        audit_pool(rows, expected=1, n_people=8, seeds=[CALIBRATION_SEED_MIN])


def test_audit_rejects_a_zero_or_varying_triple_count():
    rows = [_row(instance_id="a", triangle_triples_checked=0)]
    with pytest.raises(PoolAuditError, match="never traversed"):
        audit_pool(rows, expected=1, n_people=8, seeds=[CALIBRATION_SEED_MIN])
    rows = [_row(instance_id="a"), _row(instance_id="b", seed=CALIBRATION_SEED_MIN + 1,
                                        triangle_triples_checked=210)]
    with pytest.raises(PoolAuditError, match="one positive constant"):
        audit_pool(rows, expected=2, n_people=8,
                   seeds=[CALIBRATION_SEED_MIN, CALIBRATION_SEED_MIN + 1])


def test_pool_meta_records_provenance_and_the_audit(tmp_path):
    rc = main(["--n-seeds", "1", "--tightness", "0.2,1.0", "--overlap", "1.0",
               "--structures", "clustered", "--progress-every", "0", "--out", str(tmp_path)])
    assert rc == 0
    meta = json.loads((tmp_path / "pool_meta.json").read_text(encoding="utf-8"))
    assert meta["n_expected"] == meta["n_completed"] == 2
    assert meta["seed_start"] == CALIBRATION_SEED_MIN
    assert meta["reserved_range"] == [CALIBRATION_SEED_MIN, CALIBRATION_SEED_MAX]
    assert meta["n_people"] == 8
    assert meta["audit"]["all_checks_passed"] is True
    assert "git_commit" in meta and "schema_version" in meta
