# tests/test_band_admissibility.py
"""Offline tests for the single-design admissibility checker.

This checker evaluates one proposed design and is superseded for search purposes by
`scripts/search_band_design.py`. It is kept because it produced the recorded rejection of
the `O = 4, H = 0` proposal, and that rejection is evidence the later decision rests on.
The tests hold it to the property that matters: it evaluates, it does not repair.
"""

from __future__ import annotations

import json

import pytest

from scripts.check_band_admissibility import (
    MAX_DOMINANT_TIGHTNESS_SHARE,
    MIN_CANDIDATES,
    BandSpecError,
    common_allocation,
    evaluate,
    main,
    parse_band,
)
from tests.test_structural_analysis import row, write_pool


def pool(spec) -> list[dict]:
    rows, i = [], 0
    for pairs, optimum, tight, struct, count, *rest in spec:
        H = rest[0] if rest else 0
        for _ in range(count):
            rows.append(row(i, pairs=pairs, optimum=optimum, tightness=tight,
                            structure=struct, H=H))
            i += 1
    return rows


# --------------------------------------------------------------------------- #
# Band specifications.
# --------------------------------------------------------------------------- #
def test_a_band_spec_is_inclusive_on_both_ends():
    name, rng = parse_band("Low=1-4")
    assert name == "Low" and list(rng) == [1, 2, 3, 4]


@pytest.mark.parametrize("spec", ["Low1-4", "Low=4", "Low=5-3", "Low=1-99"])
def test_a_malformed_or_impossible_band_is_refused(spec):
    with pytest.raises((BandSpecError, ValueError)):
        parse_band(spec)


def test_overlapping_bands_are_refused():
    rows = pool([(k, 4, 1.0, "uniform", 60) for k in range(1, 12)])
    with pytest.raises(BandSpecError, match="both contain"):
        evaluate(rows, optimum=4, gap=0,
                 bands=[("A", range(1, 6)), ("B", range(4, 9))])


# --------------------------------------------------------------------------- #
# The criteria.
# --------------------------------------------------------------------------- #
def test_a_thin_band_fails_the_candidate_count():
    spec = [(1, 4, 1.0, "uniform", 5), (1, 4, 0.8, "line", 5)]
    spec += [(k, 4, t, s, 40) for k in (7, 13) for t, s in ((1.0, "uniform"), (0.8, "line"))]
    e = evaluate(pool(spec), optimum=4, gap=0,
                 bands=[("Low", range(1, 2)), ("Mid", range(7, 8)), ("High", range(13, 14))])
    assert e["bands"]["Low"]["n"] < MIN_CANDIDATES
    assert e["bands"]["Low"]["checks"]["candidates_at_least_50"] is False
    assert e["admissible"] is False


def test_a_band_dominated_by_one_tightness_fails():
    spec = [(1, 4, 1.0, "uniform", 90), (1, 4, 0.8, "line", 10)]
    e = evaluate(pool(spec), optimum=4, gap=0, bands=[("Low", range(1, 2))])
    b = e["bands"]["Low"]
    assert b["dominant_tightness_share"] > MAX_DOMINANT_TIGHTNESS_SHARE
    assert b["checks"]["dominant_tightness_at_most_60pct"] is False


def test_a_band_with_one_travel_structure_fails():
    spec = [(1, 4, t, "uniform", 40) for t in (0.8, 1.0)]
    e = evaluate(pool(spec), optimum=4, gap=0, bands=[("Low", range(1, 2))])
    assert e["bands"]["Low"]["checks"]["travel_structures_at_least_2"] is False


def test_the_stratum_filters_on_both_optimum_and_higher_order_gap():
    spec = [(1, 4, 1.0, "uniform", 30, 0), (1, 4, 1.0, "uniform", 30, 1),
            (1, 3, 1.0, "uniform", 30, 0)]
    e = evaluate(pool(spec), optimum=4, gap=0, bands=[("Low", range(1, 2))])
    assert e["stratum"]["n_in_stratum"] == 30      # only O=4 and H=0


def test_buffer_values_between_bands_are_reported():
    spec = [(k, 4, t, s, 30) for k in range(1, 15)
            for t, s in ((1.0, "uniform"), (0.8, "line"))]
    e = evaluate(pool(spec), optimum=4, gap=0,
                 bands=[("Low", range(1, 5)), ("Mid", range(7, 11)), ("High", range(13, 15))])
    assert e["buffer_pair_counts"] == [5, 6, 11, 12]


# --------------------------------------------------------------------------- #
# Identical composition across bands.
# --------------------------------------------------------------------------- #
def test_common_allocation_is_limited_by_the_scarcest_band():
    bands = {
        "A": pool([(1, 4, 1.0, "uniform", 40), (1, 4, 0.8, "line", 40)]),
        "B": pool([(7, 4, 1.0, "uniform", 5), (7, 4, 0.8, "line", 5)]),
    }
    a = common_allocation(bands, target=50)
    assert a["capacity"] == 10          # the scarce band caps every shared stratum
    assert a["feasible"] is False and a["shortfall"] == 40


def test_common_allocation_needs_strata_present_in_every_band():
    bands = {
        "A": pool([(1, 4, 1.0, "uniform", 60)]),
        "B": pool([(7, 4, 0.8, "line", 60)]),      # disjoint strata
    }
    a = common_allocation(bands, target=50)
    assert a["shared_strata"] == 0 and a["capacity"] == 0


# --------------------------------------------------------------------------- #
# It evaluates; it never repairs.
# --------------------------------------------------------------------------- #
def test_a_failing_design_returns_a_verdict_and_no_alternative(tmp_path):
    spec = [(1, 4, 1.0, "uniform", 9)]
    spec += [(k, 4, t, s, 40) for k in (7, 13) for t, s in ((1.0, "uniform"), (0.8, "line"))]
    out = tmp_path / "out"
    rc = main(["--pool", str(write_pool(tmp_path, pool(spec))), "--out", str(out),
               "--optimum", "4", "--higher-order-gap", "0",
               "--band", "Low=1-4", "--band", "Mid=7-10", "--band", "High=13-16"])
    assert rc == 1                                     # non-zero on a failed design
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["evaluation"]["admissible"] is False
    assert "proposed_bands" in doc                     # the input is echoed, not replaced
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "NOT ADMISSIBLE" in text
    assert "does not suggest a repair" in text
    assert "recorded fallback" in text
