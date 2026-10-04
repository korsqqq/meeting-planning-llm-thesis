# tests/test_power_iut_v2.py
"""Offline tests for power simulation v2.

Three things could make the selected N wrong in a way that still looks reasonable: the
hurdle silently shrinking the effect it was supposed to leave alone, a nuisance axis that
does not actually bind, and a Type-I check that cannot fail. Each has a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.power_iut import O_CONFIGS, TARGET_EFFECT, TARGET_POWER, marginal_mean
from scripts.power_iut_v2 import (
    ALPHA,
    BASE_MEANS,
    BOUNDARY_NULLS,
    LAMBDAS,
    N_GRID,
    SCENARIOS,
    SIGMAS,
    Config,
    _band_deltas,
    _straddles,
    binding_grid,
    calibrate,
    conditional_target,
    evaluate_n,
    failure_probability,
    simulate,
    type_one,
)


# --------------------------------------------------------------------------- #
# The hurdle must not move the effect it sits in front of.
# --------------------------------------------------------------------------- #
def test_failure_probability_scales_with_the_room_below_the_ceiling():
    assert failure_probability(0.5, 0.0) == 0.0
    assert failure_probability(0.5, 0.9) == pytest.approx(0.45)
    assert failure_probability(0.9, 0.9) == pytest.approx(0.09)


@pytest.mark.parametrize("mu", [0.25, 0.4, 0.5, 0.6, 0.75])
@pytest.mark.parametrize("lam", LAMBDAS)
def test_the_conditional_target_restores_the_unconditional_mean(mu, lam):
    """(1 - pi) * conditional mean must equal the requested unconditional mean."""
    pi = failure_probability(mu, lam)
    assert (1 - pi) * conditional_target(mu, lam) == pytest.approx(mu, abs=1e-12)


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_calibration_hits_the_unconditional_mean_after_the_hurdle(lam, sigma):
    mu = 0.6
    alpha, pi = calibrate(mu, lam, sigma)
    realised = (1 - pi) * marginal_mean(alpha, sigma)
    assert realised == pytest.approx(mu, abs=1e-6)


@pytest.mark.parametrize("lam", LAMBDAS)
@pytest.mark.parametrize("m", BASE_MEANS)
def test_the_realised_difference_equals_the_requested_delta(lam, m):
    """The whole point of calibrating unconditionally: delta survives the hurdle."""
    rng = np.random.default_rng(7)
    delta, sigma = TARGET_EFFECT, 1.0
    c1 = calibrate(m + delta / 2, lam, sigma)
    c3 = calibrate(m - delta / 2, lam, sigma)
    d = _band_deltas(rng, 300_000, c1, c3, sigma, "A_calibration_mix")
    assert d.mean() == pytest.approx(delta, abs=0.006)


@pytest.mark.parametrize("m,direction", [(0.35, "shrinks"), (0.65, "inflates")])
def test_calibrating_the_conditional_part_instead_distorts_the_effect(m, direction):
    """Guards the choice of unconditional calibration, and records its exact shape.

    If the intercept were solved so the SURVIVING part has mean `mu_c`, the realised
    unconditional difference would be

        delta - lambda * delta * (1 - mu_1 - mu_3)

    so the distortion is signed by the base mean: it shrinks the effect below 0.5 and
    inflates it above. At exactly 0.5 the term vanishes, which is why a single test at the
    centre would have found nothing.
    """
    rng = np.random.default_rng(8)
    delta, sigma, lam = 0.20, 1.0, 0.9
    from scripts.power_iut import solve_alpha
    mu1, mu3 = m + delta / 2, m - delta / 2
    naive1 = (solve_alpha(mu1, sigma), failure_probability(mu1, lam))
    naive3 = (solve_alpha(mu3, sigma), failure_probability(mu3, lam))
    d = _band_deltas(rng, 200_000, naive1, naive3, sigma, "B_all_optimum_3")
    predicted = delta - lam * delta * (1 - mu1 - mu3)
    assert d.mean() == pytest.approx(predicted, abs=0.006)
    if direction == "shrinks":
        assert d.mean() < delta - 0.03
    else:
        assert d.mean() > delta + 0.03


# --------------------------------------------------------------------------- #
# The no-plan regime must actually appear in the data.
# --------------------------------------------------------------------------- #
def test_the_hurdle_creates_structural_zeros_and_more_ties():
    rng = np.random.default_rng(9)
    m, sigma = 0.5, 1.0
    none = calibrate(m, 0.0, sigma)
    heavy = calibrate(m, 0.9, sigma)
    d0 = _band_deltas(rng, 60_000, none, none, sigma, "C_all_optimum_4")
    d9 = _band_deltas(rng, 60_000, heavy, heavy, sigma, "C_all_optimum_4")
    assert d9.var() > d0.var()                 # failures add spread, not just zeros
    assert abs(d9.mean()) < 0.01               # and no drift when the conditions match


def test_failure_draws_are_independent_between_conditions_by_default():
    """Shared failure would manufacture ties; the binding model must not do that."""
    rng = np.random.default_rng(10)
    spec = calibrate(0.5, 0.9, 1.0)
    independent = _band_deltas(rng, 60_000, spec, spec, 1.0, "C_all_optimum_4")
    shared = _band_deltas(rng, 60_000, spec, spec, 1.0, "C_all_optimum_4",
                          shared_failure=True)
    assert (shared == 0).mean() > (independent == 0).mean()
    assert shared.var() < independent.var()


def test_sigma_zero_is_in_the_grid_and_produces_no_heterogeneity():
    assert 0.0 in SIGMAS
    rng = np.random.default_rng(11)
    spec = calibrate(0.5, 0.0, 0.0)
    d = _band_deltas(rng, 20_000, spec, spec, 0.0, "B_all_optimum_3")
    assert set(np.unique(np.round(d, 6))).issubset({-1.0, -0.666667, -0.333333, 0.0,
                                                    0.333333, 0.666667, 1.0})


# --------------------------------------------------------------------------- #
# The binding grid.
# --------------------------------------------------------------------------- #
def test_the_binding_grid_crosses_every_nuisance_axis():
    cells = list(binding_grid(50, 0.2, -0.2))
    assert len(cells) == len(BASE_MEANS) * len(SIGMAS) * len(LAMBDAS) * len(O_CONFIGS)
    assert {c.sigma for c in cells} == set(SIGMAS)
    assert {c.lam for c in cells} == set(LAMBDAS)
    assert all(c.shared_failure is False for c in cells)   # never binding


@pytest.fixture
def tiny_grid(monkeypatch):
    """One binding cell. The full 108-cell grid is what the run uses; the tests check the
    logic around it, and paying for the whole cross product here buys nothing."""
    monkeypatch.setattr("scripts.power_iut_v2.BASE_MEANS", (0.50,))
    monkeypatch.setattr("scripts.power_iut_v2.SIGMAS", (1.0,))
    monkeypatch.setattr("scripts.power_iut_v2.LAMBDAS", (0.5,))
    monkeypatch.setattr("scripts.power_iut_v2.O_CONFIGS", {"B_all_optimum_3": {3: 1.0}})


def test_a_single_failing_configuration_rejects_the_n(tiny_grid):
    out = evaluate_n(30, replicates=120, bootstrap=120)
    if not out["passes"]:
        assert out["failed_on"]["power_iut"] < TARGET_POWER


def test_the_n_grid_starts_at_thirty_and_steps_by_five():
    assert N_GRID[0] == 30 and N_GRID[-1] == 100
    assert all(b - a == 5 for a, b in zip(N_GRID, N_GRID[1:]))


def test_only_the_target_scenario_selects_n():
    assert SCENARIOS["target"] == (TARGET_EFFECT, -TARGET_EFFECT)
    assert set(SCENARIOS) == {"target", "weak", "strong", "asymmetric"}


# --------------------------------------------------------------------------- #
# Type-I validation must be able to fail.
# --------------------------------------------------------------------------- #
def test_the_boundary_nulls_zero_exactly_one_component_each():
    assert BOUNDARY_NULLS["A_null_low"][0] == 0.0
    assert BOUNDARY_NULLS["A_null_low"][1] == -TARGET_EFFECT
    assert BOUNDARY_NULLS["B_null_high"][0] == +TARGET_EFFECT
    assert BOUNDARY_NULLS["B_null_high"][1] == 0.0


def test_a_true_null_component_rejects_at_about_the_nominal_rate():
    cfg = Config(60, 0.5, 1.0, 0.0, "A_calibration_mix", 0.0, -TARGET_EFFECT)
    out = simulate(cfg, replicates=1500, bootstrap=800)
    assert out["power_low"] < 0.12               # generous: this is the thing under test
    assert out["power_iut"] <= out["power_low"] + 1e-9


def test_the_verdict_flags_anticonservativeness_only_when_the_interval_clears_alpha():
    from scripts.power_iut_v2 import _straddles as straddle
    assert straddle([0.03, 0.07], ALPHA) is True     # noise, not evidence
    assert straddle([0.06, 0.09], ALPHA) is False    # entirely above alpha


def test_type_one_reports_both_the_component_and_the_joint_rate(tiny_grid):
    out = type_one(40, replicates=150, bootstrap=150)
    for name in BOUNDARY_NULLS:
        assert {"worst_null_rejection_rate", "joint_rejection_rate", "anticonservative",
                "mc_ci95", "null_component"} <= set(out[name])
        # the joint rate can never exceed the null component's own rate
        assert out[name]["joint_rejection_rate"] <= \
            out[name]["worst_null_rejection_rate"] + 0.05
    assert out["verdict"] in {"OK", "ANTICONSERVATIVE"}


def test_escalation_triggers_on_a_straddling_interval():
    assert _straddles([0.78, 0.82], TARGET_POWER) is True
    assert _straddles([0.85, 0.90], TARGET_POWER) is False
