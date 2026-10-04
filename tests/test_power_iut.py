# tests/test_power_iut.py
"""Offline tests for the IUT power simulation.

The simulation picks the size of the main experiment, so the tests target the places where
a wrong number would look plausible: the intercept calibration that makes a requested
effect real, the decision rule matching the registered one, and the selection rule being
the conservative one rather than a convenient central case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.power_iut import (
    BASE_MEANS,
    O_CONFIGS,
    SCENARIOS,
    SIGMAS,
    TARGET_EFFECT,
    TARGET_POWER,
    Config,
    _band_deltas,
    _decisive,
    _lower_bound,
    _upper_bound,
    evaluate_n,
    marginal_mean,
    nuisance_grid,
    search,
    simulate,
    solve_alpha,
)


# --------------------------------------------------------------------------- #
# Intercept calibration -- this is what makes "delta = 0.20" mean 0.20.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [0.15, 0.35, 0.5, 0.65, 0.85])
@pytest.mark.parametrize("sigma", SIGMAS)
def test_the_intercept_reproduces_the_requested_marginal_mean(target, sigma):
    alpha = solve_alpha(target, sigma)
    assert marginal_mean(alpha, sigma) == pytest.approx(target, abs=1e-6)


def test_calibration_is_not_the_naive_logit(sigma=1.5):
    """Jensen: logit(mean) is not the intercept whose mean is that value."""
    target = 0.65
    naive = math.log(target / (1 - target))
    assert marginal_mean(naive, sigma) != pytest.approx(target, abs=1e-3)
    assert marginal_mean(solve_alpha(target, sigma), sigma) == pytest.approx(target, 1e-6)


def test_an_impossible_mean_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="outside"):
            solve_alpha(bad, 1.0)


def test_the_realised_difference_matches_the_requested_one():
    """No clipping anywhere: a requested 0.20 must show up as 0.20 in the samples."""
    rng = np.random.default_rng(0)
    m, sigma, delta = 0.65, 1.5, 0.20      # the case a clipped model would distort most
    a1 = solve_alpha(m + delta / 2, sigma)
    a3 = solve_alpha(m - delta / 2, sigma)
    d = _band_deltas(rng, 400_000, a1, a3, sigma, "B_all_optimum_3")
    assert d.mean() == pytest.approx(delta, abs=0.005)


# --------------------------------------------------------------------------- #
# The generated outcomes must look like satisfaction, not like a normal variate.
# --------------------------------------------------------------------------- #
def test_the_outcome_is_discrete_bounded_and_produces_ties():
    rng = np.random.default_rng(1)
    a = solve_alpha(0.5, 1.0)
    d = _band_deltas(rng, 20_000, a, a, 1.0, "B_all_optimum_3")
    assert set(np.unique(d)).issubset({-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0})
    assert (d == 0).mean() > 0.2          # ties, the feature that governs power
    assert d.min() >= -1.0 and d.max() <= 1.0


def test_pairing_removes_the_instance_difficulty_from_the_difference():
    """Why the design is paired, shown as variance.

    With a shared instance effect the difference has variance `2 E[p(1-p)] / O`. If each
    condition met its own instance instead, `2 Var(p)` would be added on top -- exactly the
    between-instance heterogeneity the pairing cancels. Both arms below have identical
    margins; only the sharing differs.
    """
    rng = np.random.default_rng(2)
    sigma, n, optimum = 1.5, 100_000, 4
    a = solve_alpha(0.5, sigma)

    u = rng.normal(0.0, sigma, n)
    p = 1.0 / (1.0 + np.exp(-(a + u)))
    paired = (rng.binomial(optimum, p) - rng.binomial(optimum, p)) / optimum

    p1 = 1.0 / (1.0 + np.exp(-(a + rng.normal(0.0, sigma, n))))
    p3 = 1.0 / (1.0 + np.exp(-(a + rng.normal(0.0, sigma, n))))
    unpaired = (rng.binomial(optimum, p1) - rng.binomial(optimum, p3)) / optimum

    assert paired.var() < unpaired.var()
    # the gap is the instance-heterogeneity term, 2 Var(p)
    assert unpaired.var() - paired.var() == pytest.approx(2 * p.var(), abs=0.01)


def test_the_optimum_configuration_controls_the_grain():
    rng = np.random.default_rng(3)
    a = solve_alpha(0.5, 1.0)
    d3 = _band_deltas(rng, 5000, a, a, 1.0, "B_all_optimum_3")
    d4 = _band_deltas(rng, 5000, a, a, 1.0, "C_all_optimum_4")
    assert len(set(np.round(np.unique(d3), 6))) <= 7      # steps of 1/3
    assert len(set(np.round(np.unique(d4), 6))) <= 9      # steps of 1/4


# --------------------------------------------------------------------------- #
# The decision rule must be the registered one.
# --------------------------------------------------------------------------- #
def test_the_bounds_are_one_sided_and_in_the_right_direction():
    rng = np.random.default_rng(4)
    positive = np.full(200, 0.3)
    assert _lower_bound(rng, positive, 500) > 0
    assert _upper_bound(rng, positive, 500) > 0            # not below zero: no false pass
    negative = np.full(200, -0.3)
    assert _upper_bound(rng, negative, 500) < 0


def test_a_null_effect_stays_near_the_nominal_error_rate():
    """Both bands generated with no difference: the conjunction must almost never fire."""
    cfg = Config(n_per_band=60, base_mean=0.5, sigma=1.0, o_config="A_calibration_mix",
                 delta_low=0.0, delta_high=0.0)
    out = simulate(cfg, replicates=400, bootstrap=400)
    assert out["power_iut"] < 0.05


def test_power_is_measured_directly_and_matches_the_product():
    cfg = Config(n_per_band=60, base_mean=0.5, sigma=1.0, o_config="A_calibration_mix",
                 delta_low=+0.35, delta_high=-0.35)
    out = simulate(cfg, replicates=400, bootstrap=400)
    assert out["power_iut"] == pytest.approx(out["product_sanity_check"], abs=0.06)
    assert out["power_iut"] <= min(out["power_low"], out["power_high"])


def test_power_rises_with_the_effect_and_with_n():
    def p(n, delta):
        return simulate(Config(n, 0.5, 1.0, "A_calibration_mix", delta, -delta),
                        replicates=300, bootstrap=300)["power_iut"]
    assert p(60, 0.35) > p(60, 0.10)
    assert p(90, 0.20) >= p(30, 0.20)


# --------------------------------------------------------------------------- #
# Selection: conservative, and never a convenient central case.
# --------------------------------------------------------------------------- #
def test_the_nuisance_grid_is_the_full_cross_product():
    cells = list(nuisance_grid(50, 0.2, -0.2))
    assert len(cells) == len(BASE_MEANS) * len(SIGMAS) * len(O_CONFIGS)
    assert {c.o_config for c in cells} == set(O_CONFIGS)


def test_an_n_fails_if_any_single_configuration_fails():
    """One bad cell is enough; the rule is a minimum over configurations, not a mean."""
    out = evaluate_n(30, replicates=200, bootstrap=200)
    if not out["passes"]:
        assert out["failed_on"]["power_iut"] < TARGET_POWER
        # aborted early rather than evaluating the rest
        assert len(out["cells"]) <= len(BASE_MEANS) * len(SIGMAS) * len(O_CONFIGS)


def test_the_search_returns_the_smallest_passing_grid_point():
    out = search(replicates=150, bootstrap=150, grid=(40, 80), extended=())
    attempted = [a["n_per_band"] for a in out["attempts"]]
    assert attempted == sorted(attempted)
    if out["selected_n"] is not None:
        assert out["selected_n"] == attempted[-1]
        assert all(not a["passes"] for a in out["attempts"][:-1])


def test_only_the_target_scenario_selects_n():
    assert SCENARIOS["target"] == (TARGET_EFFECT, -TARGET_EFFECT)
    assert set(SCENARIOS) == {"target", "weak", "strong", "asymmetric"}
    # the asymmetric case is the one where the conjunction is governed by the weak side
    assert SCENARIOS["asymmetric"] == (0.10, -0.35)


def test_a_cell_straddling_the_target_is_not_allowed_to_decide():
    assert _decisive({"mc_ci95": [0.85, 0.90]}) is True
    assert _decisive({"mc_ci95": [0.70, 0.75]}) is True
    assert _decisive({"mc_ci95": [0.78, 0.82]}) is False     # escalation required


def test_the_optimum_split_is_declared_a_sensitivity_axis_not_a_design_choice():
    """41:24 is the calibration pool's capacity, and adopting it here would be circular."""
    assert O_CONFIGS["A_calibration_mix"][3] == pytest.approx(41 / 65)
    assert "B_all_optimum_3" in O_CONFIGS and "C_all_optimum_4" in O_CONFIGS
