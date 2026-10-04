# tests/test_structural_analysis.py
"""Offline tests for the pre-specified structural-calibration analysis.

Two things matter here. The diagnostics must be arithmetically right on hand-built pools,
and the script must be unable to choose a band -- the whole point of committing it before
the full distribution was read is that it cannot be shaped by what the data allows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyse_structural_calibration import (
    EXPECTED_TRIPLES,
    MAX_PAIRS,
    N_PEOPLE,
    PoolMismatchError,
    load_pool,
    main,
    section_a,
    section_b,
    section_c,
    section_d,
    section_e,
    spearman,
)

FIELDS = [
    "instance_id", "seed", "n_people", "tightness", "overlap", "travel_structure",
    "individually_reachable_count", "conflicting_pairs", "D", "D_reachable",
    "oracle_optimum", "optimum_over_n", "proven_optimal", "solver_status",
    "alpha_reachable", "higher_order_gap_H", "n_max_independent_sets", "max_degree",
    "edge_share_top1", "edge_share_top2", "largest_component", "triangle_violations",
    "triangle_triples_checked", "solve_seconds",
]


def row(i: int, *, pairs: int, optimum: int, tightness: float = 1.0, overlap: float = 1.0,
        structure: str = "clustered", H: int = 0, violations: int = 3,
        reachable: int = N_PEOPLE, triples: int = EXPECTED_TRIPLES,
        proven: bool = True) -> dict:
    return {
        "instance_id": f"i{i}", "seed": 20000 + i, "n_people": N_PEOPLE,
        "tightness": tightness, "overlap": overlap, "travel_structure": structure,
        "individually_reachable_count": reachable, "conflicting_pairs": pairs,
        "D": round(pairs / MAX_PAIRS, 6), "D_reachable": round(pairs / MAX_PAIRS, 6),
        "oracle_optimum": optimum, "optimum_over_n": round(optimum / N_PEOPLE, 6),
        "proven_optimal": proven, "solver_status": "OPTIMAL",
        "alpha_reachable": optimum + H, "higher_order_gap_H": H,
        "n_max_independent_sets": 2, "max_degree": 3, "edge_share_top1": 0.5,
        "edge_share_top2": 0.75, "largest_component": 4,
        "triangle_violations": violations, "triangle_triples_checked": triples,
        "solve_seconds": 0.2,
    }


def write_pool(tmp_path: Path, rows: list[dict], **meta_over) -> Path:
    d = tmp_path / "pool"
    d.mkdir(exist_ok=True)
    with (d / "candidates.csv").open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    meta = {"schema_version": "structural_calibration/1.0", "git_commit": "abc123",
            "n_people": N_PEOPLE, "seed_start": 20000, "seed_end": 20000 + len(rows),
            "reserved_range": [20000, 29999], "tightness": [1.0], "overlap": [1.0],
            "structures": ["clustered"], "n_expected": len(rows),
            "n_completed": len(rows), "n_not_proven": 0, "wall_seconds": 1.0}
    meta.update(meta_over)
    (d / "pool_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Re-verification of the pool.
# --------------------------------------------------------------------------- #
def test_a_consistent_pool_loads(tmp_path):
    rows, meta = load_pool(write_pool(tmp_path, [row(i, pairs=i, optimum=5)
                                                 for i in range(4)]))
    assert len(rows) == 4 and meta["n_people"] == N_PEOPLE
    assert rows[0]["proven_optimal"] is True


def test_a_row_count_disagreeing_with_metadata_fails(tmp_path):
    d = write_pool(tmp_path, [row(0, pairs=1, optimum=5)], n_completed=99)
    with pytest.raises(PoolMismatchError, match="metadata says 99"):
        load_pool(d)


def test_an_unproven_optimum_fails(tmp_path):
    d = write_pool(tmp_path, [row(0, pairs=1, optimum=5, proven=False)])
    with pytest.raises(PoolMismatchError, match="unproven optimum"):
        load_pool(d)


def test_a_varying_triple_count_fails(tmp_path):
    d = write_pool(tmp_path, [row(0, pairs=1, optimum=5),
                              row(1, pairs=2, optimum=5, triples=210)])
    with pytest.raises(PoolMismatchError, match="triangle triple count"):
        load_pool(d)


# --------------------------------------------------------------------------- #
# A: axis existence.
# --------------------------------------------------------------------------- #
def test_definitional_split_is_exact():
    rows = ([row(0, pairs=0, optimum=8)] * 3 + [row(1, pairs=5, optimum=5)] * 2
            + [row(2, pairs=MAX_PAIRS, optimum=1)])
    s = section_a(rows)["definitional_split"]
    assert (s["D_zero"], s["D_between"], s["D_one"]) == (3, 2, 1)


def test_cumulative_share_reaches_one():
    rows = [row(i, pairs=i % 5, optimum=5) for i in range(20)]
    ecdf = section_a(rows)["counts_per_pair_count"]
    assert ecdf[-1]["cumulative_share"] == 1.0
    assert len(ecdf) == MAX_PAIRS + 1  # every possible D value is listed, even empty ones


# --------------------------------------------------------------------------- #
# B: the central question.
# --------------------------------------------------------------------------- #
def test_per_optimum_shows_the_D_range_available_at_each_optimum():
    rows = [row(0, pairs=1, optimum=6), row(1, pairs=10, optimum=6),
            row(2, pairs=20, optimum=2)]
    b = section_b(rows)
    assert b["per_optimum"]["6"]["n"] == 2
    assert b["per_optimum"]["6"]["pair_count_range"] == [1, 10]
    assert b["per_optimum"]["2"]["distinct_pair_counts"] == 1


def test_negative_correlation_between_density_and_optimum_is_reported():
    """The mechanism the optimum-matching rule exists for; the sign must survive."""
    rows = [row(i, pairs=i, optimum=8 - i // 4) for i in range(24)]
    assert section_b(rows)["spearman_D_vs_optimum"] < 0


# --------------------------------------------------------------------------- #
# C: generator confounding.
# --------------------------------------------------------------------------- #
def test_dominant_tightness_share_is_computed_per_pair_count():
    rows = [row(0, pairs=7, optimum=5, tightness=1.0),
            row(1, pairs=7, optimum=5, tightness=1.0),
            row(2, pairs=7, optimum=5, tightness=0.8)]
    c = section_c(rows)["per_pair_count"]["7"]
    assert c["distinct_tightness"] == 2
    assert c["dominant_tightness"] == 1.0
    assert c["dominant_tightness_share"] == pytest.approx(2 / 3, abs=1e-4)


def test_structure_diversity_is_counted():
    rows = [row(0, pairs=3, optimum=5, structure="uniform"),
            row(1, pairs=3, optimum=5, structure="line")]
    assert section_c(rows)["per_pair_count"]["3"]["distinct_structures"] == 2


# --------------------------------------------------------------------------- #
# D and E.
# --------------------------------------------------------------------------- #
def test_graph_section_counts_instances_below_full_reachability():
    rows = [row(0, pairs=2, optimum=5), row(1, pairs=2, optimum=5, reachable=5)]
    assert section_d(rows)["reachable_below_n"] == 1


def test_negative_H_is_reported_and_not_clipped():
    rows = [row(0, pairs=2, optimum=5, H=-1, structure="random"),
            row(1, pairs=2, optimum=5, H=0, structure="random")]
    e = section_e(rows)
    assert e["H_sign_overall"]["negative"] == 1
    assert e["per_structure"]["random"]["H"]["min"] == -1
    assert e["negative_H_instances"] == ["i0"]


def test_violation_rate_uses_the_examined_triple_count():
    rows = [row(0, pairs=1, optimum=5, structure="random", violations=EXPECTED_TRIPLES)]
    assert section_e(rows)["per_structure"]["random"]["violation_rate_mean"] == 1.0


def test_spearman_is_undefined_without_variation():
    assert spearman([1, 2, 3], [4, 4, 4]) is None


# --------------------------------------------------------------------------- #
# The script must not be able to choose a band.
# --------------------------------------------------------------------------- #
def test_the_script_contains_no_band_definition():
    src = Path("scripts/analyse_structural_calibration.py").read_text(encoding="utf-8")
    lowered = src.lower()
    for forbidden in ("low =", "medium =", "high =", "low_d", "medium_d", "high_d",
                      "band_edges", "best_bands", "choose_band", "select_band"):
        assert forbidden not in lowered, f"the analyser must not define bands: {forbidden!r}"


def test_the_output_declares_that_it_decides_nothing(tmp_path):
    rows = [row(i, pairs=i, optimum=max(1, 8 - i // 3), tightness=1.0 if i % 2 else 0.8,
                structure="clustered" if i % 3 else "uniform") for i in range(30)]
    out = tmp_path / "out"
    assert main(["--pool", str(write_pool(tmp_path, rows)), "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["decides_bands"] is False
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "No band is chosen here" in text
    assert "It names no band and applies no threshold" in text
    for forbidden in ("Low band", "Medium band", "High band"):
        assert forbidden not in text


def test_artifacts_are_byte_reproducible(tmp_path):
    rows = [row(i, pairs=i % 9, optimum=max(1, 8 - i // 5)) for i in range(25)]
    pool = write_pool(tmp_path, rows)
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        main(["--pool", str(pool), "--out", str(out)])
    for name in ("summary.json", "report.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
        assert b"\r\n" not in (a / name).read_bytes()
