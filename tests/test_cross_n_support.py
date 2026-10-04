# tests/test_cross_n_support.py
"""Offline tests for the cross-size support analyser.

The analyser's whole job is to decide nothing, so most of what is worth testing is what it
refuses to do: read an agent outcome, read a pool that failed its own audit, or quietly
report a capacity larger than the smallest contributing size can supply.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyse_cross_n_support import (
    PoolError,
    capacity,
    ceiling_analysis,
    diversity_headroom,
    joint_capacity,
    load_pool,
    main,
    matched_variant,
    sanity_check_high_optimum,
    scaled_histogram,
)

FIELDS = [
    "instance_id", "seed", "n_people", "tightness", "overlap", "travel_structure",
    "individually_reachable_count", "conflicting_pairs", "D", "D_reachable",
    "oracle_optimum", "optimum_over_n", "proven_optimal", "solver_status",
    "alpha_reachable", "higher_order_gap_H", "n_max_independent_sets", "max_degree",
    "edge_share_top1", "edge_share_top2", "largest_component", "triangle_violations",
    "triangle_triples_checked", "solve_seconds",
]


def row(n, o, *, seed, k=0, t=0.8, overlap=0.5, structure="uniform", alpha=None,
        proven=True):
    alpha = o if alpha is None else alpha
    return {
        "instance_id": f"{structure}-n{n}-t{int(t*100)}-o{int(overlap*100)}-s{seed}",
        "seed": seed, "n_people": n, "tightness": t, "overlap": overlap,
        "travel_structure": structure, "individually_reachable_count": n,
        "conflicting_pairs": k, "D": round(k / (n * (n - 1) / 2), 6), "D_reachable": "",
        "oracle_optimum": o, "optimum_over_n": round(o / n, 6),
        "proven_optimal": proven, "solver_status": "OPTIMAL", "alpha_reachable": alpha,
        "higher_order_gap_H": alpha - o, "n_max_independent_sets": 1, "max_degree": 0,
        "edge_share_top1": "", "edge_share_top2": "", "largest_component": 1,
        "triangle_violations": 0, "triangle_triples_checked": 24, "solve_seconds": 0.01,
    }


def parsed(n, o, *, seed, k=0, t=0.8, overlap=0.5, structure="uniform", alpha=None):
    """A row in the shape `load_pool` hands to the analysis functions.

    `row` above is the on-disk CSV shape; the two are deliberately different, and passing
    one where the other belongs is the mistake this helper exists to prevent.
    """
    alpha = o if alpha is None else alpha
    return {
        "instance_id": f"{structure}-n{n}-s{seed}", "seed": seed, "n": n,
        "tightness": t, "overlap": overlap, "structure": structure,
        "k": k, "D": round(k / (n * (n - 1) / 2), 6), "O": o,
        "reachable": n, "alpha": alpha, "H": alpha - o, "proven": True,
    }


def write_pool(dirpath: Path, rows, *, audit_ok=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    with (dirpath / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (dirpath / "pool_meta.json").write_text(json.dumps({
        "schema_version": "structural_calibration/1.0", "git_commit": "abc",
        "reserved_range": [40000, 49999],
        "audit": {"all_checks_passed": audit_ok},
    }), encoding="utf-8")
    return dirpath


# --------------------------------------------------------------------------- #
# Loading refuses what must not be read.
# --------------------------------------------------------------------------- #
def test_a_pool_that_failed_its_audit_is_refused(tmp_path):
    p = write_pool(tmp_path / "p", [row(4, 3, seed=40000)], audit_ok=False)
    with pytest.raises(PoolError, match="completeness audit"):
        load_pool(p, 4)


def test_a_pool_of_the_wrong_size_is_refused(tmp_path):
    p = write_pool(tmp_path / "p", [row(4, 3, seed=40000)])
    with pytest.raises(PoolError, match="expected n_people=5"):
        load_pool(p, 5)


def test_an_unproven_optimum_is_refused(tmp_path):
    """The design matches on the optimum, so a number the solver could not prove cannot
    be matched on and must not be counted into a capacity."""
    p = write_pool(tmp_path / "p", [row(4, 3, seed=40000, proven=False)])
    with pytest.raises(PoolError, match="unproven"):
        load_pool(p, 4)


# --------------------------------------------------------------------------- #
# Capacity arithmetic.
# --------------------------------------------------------------------------- #
def test_joint_capacity_is_the_smallest_contributing_size():
    pools = {4: [parsed(4, 3, seed=i) for i in range(10)],
             8: [parsed(8, 3, seed=i) for i in range(3)]}
    cap = capacity(pools, (3,))
    assert cap == {4: {3: 10}, 8: {3: 3}}
    assert joint_capacity(cap, (3,)) == {3: 3}


def test_the_scaling_is_deterministic_including_the_ties():
    assert scaled_histogram(12, {3: 1, 4: 1}) == {3: 6, 4: 6}
    assert scaled_histogram(12, {3: 41, 4: 24}) == {3: 8, 4: 4}
    # A tie on the remainder is broken on the optimum value, never on dict order.
    a = scaled_histogram(5, {3: 1, 4: 1})
    b = scaled_histogram(5, {4: 1, 3: 1})
    assert a == b == {3: 3, 4: 2}


def test_a_variant_reports_which_size_binds():
    pools = {4: [parsed(4, 3, seed=i) for i in range(50)],
             8: [parsed(8, 3, seed=i) for i in range(20)]}
    v = matched_variant(pools, (3,), 12)
    assert v["joint_capacity"] == {"3": 20}
    assert v["binding_size_per_optimum"] == {"3": 8}
    assert v["sufficient_capacity"] is True


def test_insufficient_capacity_is_reported_not_worked_around():
    pools = {4: [parsed(4, 3, seed=i) for i in range(50)],
             8: [parsed(8, 3, seed=i) for i in range(4)]}
    v = matched_variant(pools, (3,), 12)
    assert v["sufficient_capacity"] is False


# --------------------------------------------------------------------------- #
# The diversity rules, as necessary conditions.
# --------------------------------------------------------------------------- #
def test_an_optimum_living_at_one_tightness_trips_the_sixty_percent_rule():
    """The failure the frozen n=8 low band actually hit: O = 3 existed only at one
    tightness value, which capped it below the scaled histogram."""
    rows = [parsed(4, 3, seed=i, t=0.8) for i in range(50)]
    d = diversity_headroom(rows, (3,), {3: 12}, 12)
    assert d["tightness_values_per_optimum"]["3"] == [0.8]
    assert d["tightness_ceiling_at_size"] == 7
    assert d["single_tightness_rule_violated"] is True


def test_two_tightness_values_clear_the_rule():
    rows = ([parsed(4, 3, seed=i, t=0.8) for i in range(25)]
            + [parsed(4, 3, seed=100 + i, t=0.6) for i in range(25)])
    d = diversity_headroom(rows, (3,), {3: 12}, 12)
    assert d["single_tightness_rule_violated"] is False
    assert d["at_least_two_tightness_possible"] is True


# --------------------------------------------------------------------------- #
# The ceiling.
# --------------------------------------------------------------------------- #
def test_the_ceiling_analysis_flags_optimum_equal_to_n():
    pools = {4: [parsed(4, 4, seed=i, alpha=4) for i in range(5)]}
    got = {(r["n"], r["optimum"]): r for r in ceiling_analysis(pools, (4,))}
    r = got[4, 4]
    assert r["meets_everyone"] is True
    assert r["optimum_over_n"] == 1.0
    assert r["arithmetic_H_ceiling"] == 0


def test_the_arithmetic_h_ceiling_falls_with_n_at_a_matched_optimum():
    pools = {n: [parsed(n, 3, seed=n * 1000 + i, alpha=3) for i in range(5)]
             for n in (4, 5, 6, 8)}
    ceilings = {r["n"]: r["arithmetic_H_ceiling"] for r in ceiling_analysis(pools, (3,))}
    assert ceilings == {4: 1, 5: 2, 6: 3, 8: 5}


def test_an_empty_cell_is_reported_rather_than_dropped():
    pools = {4: [parsed(4, 3, seed=1)]}
    got = {(r["n"], r["optimum"]): r for r in ceiling_analysis(pools, (3, 4))}
    assert got[4, 4]["count"] == 0


# --------------------------------------------------------------------------- #
# The sanity check must stay labelled as one.
# --------------------------------------------------------------------------- #
def test_the_high_optimum_option_is_labelled_not_a_candidate():
    rows = [parsed(8, 7, seed=i, k=0, alpha=7) for i in range(4)]
    sc = sanity_check_high_optimum(rows)
    assert "NOT A SELECTION CANDIDATE" in sc["status"]
    assert "confound" in sc["why_excluded"]
    assert sc["count"] == 4


# --------------------------------------------------------------------------- #
# Nothing about an agent may reach this file.
# --------------------------------------------------------------------------- #
def test_the_analyser_never_reads_an_agent_quantity():
    import tokenize
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyse_cross_n_support.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING)).lower()
    for banned in ("satisfaction", "tokens_total", "termination", "run_result",
                   "c1_react", "c3_mas", "final_plan", "transcript"):
        assert banned not in code, f"the support analyser reads {banned!r}"


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_both_artifacts_and_selects_nothing(tmp_path):
    p4 = write_pool(tmp_path / "n4", [row(4, 3, seed=40000 + i, t=0.8 if i % 2 else 0.6,
                                          structure="uniform" if i % 2 else "line")
                                      for i in range(30)])
    p8 = write_pool(tmp_path / "n8", [row(8, 3, seed=30000 + i, t=0.8 if i % 2 else 0.9,
                                          structure="uniform" if i % 2 else "random")
                                      for i in range(30)])
    out = tmp_path / "out"
    assert main(["--pool", f"4={p4}", "--pool", f"8={p8}", "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["decides"].startswith("nothing")
    assert "instances" not in doc and "selected" not in doc
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Selects nothing" in report
    assert "Nothing above is frozen" in report


def test_the_cli_needs_a_size_other_than_eight(tmp_path):
    p8 = write_pool(tmp_path / "n8", [row(8, 3, seed=30000)])
    with pytest.raises(SystemExit):
        main(["--pool", f"8={p8}", "--out", str(tmp_path / "o")])
