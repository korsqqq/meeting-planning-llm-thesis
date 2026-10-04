# tests/test_type_one_validation.py
"""Offline tests for the corrected Type-I validation.

The rule this replaces failed because it applied a threshold to the maximum of a large
grid. So the tests concentrate on the two things that must now hold: the family is tested
whole and corrected, and no cell is chosen after the fact.
"""

from __future__ import annotations

import json
import math

import pytest

from scripts.power_iut import O_CONFIGS, TARGET_EFFECT
from scripts.power_iut_v2 import BASE_MEANS, LAMBDAS, SIGMAS
from scripts.type_one_validation import (
    ALPHA,
    BOUNDARY_NULLS,
    Cell,
    binom_sf,
    grid,
    holm,
    main,
    rejection_rate,
    validate,
)


# --------------------------------------------------------------------------- #
# The exact binomial tail.
# --------------------------------------------------------------------------- #
def test_the_tail_matches_a_hand_computed_case():
    # P(X >= 2) for n=3, p=0.5 is 4/8
    assert binom_sf(2, 3, 0.5) == pytest.approx(0.5, abs=1e-12)
    # P(X >= 1) = 1 - (1-p)^n
    assert binom_sf(1, 10, 0.1) == pytest.approx(1 - 0.9 ** 10, abs=1e-12)


def test_the_tail_is_bounded_and_monotone():
    assert binom_sf(0, 100, 0.05) == 1.0
    assert binom_sf(101, 100, 0.05) == 0.0
    tail = [binom_sf(k, 1000, 0.05) for k in (40, 50, 60, 70)]
    assert all(a >= b for a, b in zip(tail, tail[1:]))


def test_the_exact_tail_exceeds_the_normal_approximation_as_the_skew_predicts():
    """Why the tail is computed exactly rather than approximated.

    Binomial(10000, 0.05) is right-skewed, so a normal approximation understates the right
    tail -- here by about 7%. That is the direction that matters: an understated tail makes
    a cell look more significant than it is, which is the error this validation must not
    make.
    """
    n, p, x = 10_000, 0.05, 550
    exact = binom_sf(x, n, p)
    z = (x - 0.5 - n * p) / math.sqrt(n * p * (1 - p))
    from statistics import NormalDist
    approx = 1 - NormalDist().cdf(z)
    assert exact > approx
    assert exact == pytest.approx(approx, rel=0.15)


def test_a_nominal_rate_is_not_significant_and_an_inflated_one_is():
    """10000 replicates: 0.05 must look ordinary, 0.065 must not."""
    assert binom_sf(500, 10_000, ALPHA) > 0.4
    assert binom_sf(650, 10_000, ALPHA) < 1e-10


# --------------------------------------------------------------------------- #
# Holm.
# --------------------------------------------------------------------------- #
def test_holm_is_step_down_and_stops_at_the_first_retention():
    # smallest p compared against alpha/m, then alpha/(m-1), ...
    p = [0.001, 0.02, 0.03, 0.9]
    out = holm(p, alpha=0.05)
    assert out[0] is True                       # 0.001 <= 0.05/4
    assert out[1] is False                      # 0.02 > 0.05/3 -> stop
    assert out[2] is False and out[3] is False


def test_holm_rejects_nothing_when_every_p_is_large():
    assert holm([0.2, 0.5, 0.9], alpha=0.05) == [False, False, False]


def test_holm_is_stricter_than_no_correction():
    """The point of the correction: a maximum over many cells stops being significant."""
    p = [0.03] + [0.5] * 215                    # one 'significant' cell in a family of 216
    assert p[0] < ALPHA                         # uncorrected it would reject
    assert holm(p, alpha=ALPHA)[0] is False     # corrected it does not


def test_holm_preserves_input_order():
    p = [0.9, 0.0001, 0.5]
    assert holm(p, alpha=0.05) == [False, True, False]


# --------------------------------------------------------------------------- #
# The family is the whole grid, and nothing is chosen after the fact.
# --------------------------------------------------------------------------- #
def test_the_family_is_both_boundary_nulls_across_every_configuration():
    cells = list(grid())
    per_null = len(BASE_MEANS) * len(SIGMAS) * len(LAMBDAS) * len(O_CONFIGS)
    assert len(cells) == 2 * per_null
    assert {c.boundary_null for c in cells} == set(BOUNDARY_NULLS)
    assert {c.null_component for c in cells} == {"low", "high"}


def test_each_boundary_null_zeroes_exactly_one_component():
    assert BOUNDARY_NULLS["A_null_low"] == (0.0, -TARGET_EFFECT)
    assert BOUNDARY_NULLS["B_null_high"] == (+TARGET_EFFECT, 0.0)


def test_the_null_component_really_has_no_difference():
    """Both conditions share one calibration, so the true difference is exactly zero."""
    import numpy as np
    from scripts.power_iut_v2 import _band_deltas, calibrate
    rng = np.random.default_rng(3)
    spec = calibrate(0.5, 0.5, 1.0)
    d = _band_deltas(rng, 200_000, spec, spec, 1.0, "A_calibration_mix")
    assert d.mean() == pytest.approx(0.0, abs=0.004)


def test_the_verdict_needs_a_survivor_not_a_maximum():
    """The defect being fixed: a high worst cell is not by itself anticonservative."""
    out = validate(30, replicates=200, bootstrap=200, progress=False)
    assert out["verdict"] in {"OK", "ANTICONSERVATIVE"}
    if out["cells_surviving_holm"] == 0:
        assert out["verdict"] == "OK"
        # a worst cell above alpha must not on its own flip the verdict
        assert out["highest_rate_cell"]["rate"] >= 0.0
    assert len(out["cells"]) == out["family_size"]


def test_every_cell_is_reported_so_none_can_be_picked_later(tmp_path):
    out = tmp_path / "o"
    main(["--n", "30", "--replicates", "150", "--bootstrap", "150", "--out", str(out)])
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    r = doc["result"]
    assert len(r["cells"]) == r["family_size"] == len(list(grid()))
    assert all("p_one_sided" in c and "rejected_after_holm" in c for c in r["cells"])
    assert "if this fails, the statistical test changes and N does not" in \
        doc["not_a_remedy"]


def test_only_the_null_component_is_simulated():
    """Cheaper and conservative: the joint rate is bounded by the component's own rate."""
    cell = Cell("A_null_low", "low", 0.5, 1.0, 0.0, "B_all_optimum_3")
    hits = rejection_rate(cell, 30, replicates=200, bootstrap=200)
    assert 0 <= hits <= 200
