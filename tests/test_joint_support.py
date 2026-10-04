# tests/test_joint_support.py
"""Offline tests for the post-specified joint-support check.

Three things are tested: the cell arithmetic, that the check still cannot select a band,
and that its post-specified status is stated in every artifact it writes. The last one
matters because the check was written after the pool was read, and a reader must be able to
tell it apart from the analysis that was fixed beforehand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_joint_support import joint_support, main
from tests.test_structural_analysis import row, write_pool


def test_cells_are_keyed_by_optimum_and_higher_order_gap():
    rows = [row(0, pairs=2, optimum=4, H=0), row(1, pairs=9, optimum=4, H=0),
            row(2, pairs=3, optimum=4, H=1)]
    j = joint_support(rows)
    assert set(j["per_cell"]) == {"O=4,H=0", "O=4,H=1"}
    assert j["per_cell"]["O=4,H=0"]["n"] == 2


def test_a_cell_reports_the_density_span_available_inside_it():
    """The whole point: how far D can move while O and H stay fixed."""
    rows = [row(i, pairs=p, optimum=3, H=1) for i, p in enumerate([2, 5, 9, 14])]
    c = joint_support(rows)["per_cell"]["O=3,H=1"]
    assert (c["pair_min"], c["pair_max"], c["pair_span"]) == (2, 14, 12)
    assert c["distinct_pair_counts"] == 4
    assert c["D_min"] == pytest.approx(2 / 28, abs=1e-4)
    assert c["D_max"] == pytest.approx(14 / 28, abs=1e-4)


def test_a_cell_reports_whether_the_tightness_confound_survives_conditioning():
    rows = [row(i, pairs=p, optimum=3, H=0, tightness=t)
            for i, (p, t) in enumerate([(1, 0.8), (5, 0.8), (10, 1.0), (20, 1.0)])]
    c = joint_support(rows)["per_cell"]["O=3,H=0"]
    assert c["distinct_tightness"] == 2
    assert c["dominant_tightness_share"] == 0.5
    # Density still tracks the knob inside the cell. The ceiling is below 1 because
    # tightness has two ties here, which caps the achievable rank correlation.
    assert c["spearman_D_vs_tightness"] > 0.85


def test_h_distribution_is_reported_per_optimum():
    rows = [row(0, pairs=1, optimum=5, H=0), row(1, pairs=1, optimum=5, H=2),
            row(2, pairs=1, optimum=3, H=1)]
    dist = joint_support(rows)["H_distribution_given_optimum"]
    assert dist["5"] == {0: 1, 2: 1}
    assert dist["3"] == {1: 1}


def test_negative_h_cells_are_kept():
    """The two random instances with H < 0 must not vanish from the support tables."""
    rows = [row(0, pairs=4, optimum=3, H=-1, structure="random"),
            row(1, pairs=4, optimum=3, H=0)]
    assert "O=3,H=-1" in joint_support(rows)["per_cell"]


# --------------------------------------------------------------------------- #
# It must not select, and it must announce what it is.
# --------------------------------------------------------------------------- #
def test_the_script_selects_no_band_and_ranks_no_cell():
    src = Path("scripts/check_joint_support.py").read_text(encoding="utf-8").lower()
    for forbidden in ("low =", "medium =", "high =", "low_d", "medium_d", "high_d",
                      "best_cell", "best_bands", "select_band", "choose_band"):
        assert forbidden not in src, f"the check must not select: {forbidden!r}"


def test_every_artifact_states_that_the_check_is_post_specified(tmp_path):
    rows = [row(i, pairs=i % 12, optimum=3 + i % 3, H=i % 2) for i in range(40)]
    out = tmp_path / "out"
    assert main(["--pool", str(write_pool(tmp_path, rows)), "--out", str(out)]) == 0

    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["selects_bands"] is False
    assert "post-specified" in doc["status"]

    text = (out / "report.md").read_text(encoding="utf-8")
    assert "post-specified" in text
    assert "was not part of the analysis committed before" in text
    assert "No band is selected here" in text
    # H must be named a matching variable, never part of the complexity definition.
    assert "does **not** enter the definition of complexity" in text
    assert "D + H" in text  # stated only to say no such composite is formed


def test_the_three_recorded_outcomes_are_named_in_the_report(tmp_path):
    rows = [row(i, pairs=i % 9, optimum=4, H=i % 2) for i in range(20)]
    out = tmp_path / "out"
    main(["--pool", str(write_pool(tmp_path, rows)), "--out", str(out)])
    text = (out / "report.md").read_text(encoding="utf-8")
    for outcome in ("common support is adequate", "cannot be balanced",
                    "no common support"):
        assert outcome in text


def test_artifacts_are_byte_reproducible(tmp_path):
    rows = [row(i, pairs=i % 15, optimum=2 + i % 4, H=i % 3) for i in range(45)]
    pool = write_pool(tmp_path, rows)
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        main(["--pool", str(pool), "--out", str(out)])
    for name in ("summary.json", "report.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
        assert b"\r\n" not in (a / name).read_bytes()
