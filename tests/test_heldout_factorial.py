# tests/test_heldout_factorial.py
"""Deterministic unit tests of the factorial-analysis helpers. No held-out data is read."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyse_heldout_factorial import (
    ARCHS, AnalysisRefused, bootstrap_mean_ci, describe_values, holm, labelperm_p, omnibus_family,
    orthonormal_contrasts, outcome_array, permute_codes, rng_for, signflip_p, task_vectors,
    within_stat,
)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_contrasts_are_orthonormal_and_sum_to_zero(k: int) -> None:
    c = orthonormal_contrasts(k)
    assert np.allclose(c @ c.T, np.eye(k - 1))
    assert np.allclose(c.sum(axis=1), 0.0)


def test_statistic_does_not_depend_on_contrast_basis() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 5, 4))
    z = task_vectors(y, "A")
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert within_stat(z) == pytest.approx(within_stat(z @ q.T))


def test_architecture_statistic_is_variance_of_marginal_means() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=(20, 5, 4))
    m = y.mean(axis=(0, 2))
    assert within_stat(task_vectors(y, "A")) == pytest.approx(((m - m.mean()) ** 2).sum())


def test_interaction_vector_is_zero_for_additive_data() -> None:
    task = np.arange(10.0)[:, None, None]
    arch = np.array([0.0, 1, 2, 3, 4])[None, :, None]
    cap = np.array([0.0, 5, 7, 9])[None, None, :]
    assert np.allclose(task_vectors(task + arch + cap, "AxB"), 0.0)


def test_holm_matches_hand_computation() -> None:
    adj = [a for a, _ in holm([0.01, 0.04, 0.03, 0.20])]
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_signflip_p_bounds() -> None:
    rng = rng_for(1, "t")
    assert signflip_p(np.zeros((40, 3)), 999, rng)[1] == 1.0
    strong = np.ones((40, 1))
    assert signflip_p(strong, 999, rng)[1] == pytest.approx(1 / 1000, abs=2e-3)


def test_label_permutation_keeps_counts_and_detects_shift() -> None:
    codes = np.repeat(np.arange(3), 10)
    perm = permute_codes(codes, 50, np.random.default_rng(2))
    assert all((np.bincount(row, minlength=3) == 10).all() for row in perm)
    z = np.concatenate([np.zeros(20), np.full(10, 5.0)])[:, None]
    labels = np.repeat(["Low", "Medium", "High"], 10)
    _, p = labelperm_p(z, labels, 999, np.random.default_rng(3))
    assert p < 0.01


def test_rng_depends_on_key_not_order() -> None:
    a = rng_for(7, "x").random(3)
    rng_for(7, "y").random(3)
    assert np.array_equal(a, rng_for(7, "x").random(3))


def test_bootstrap_is_reproducible_and_brackets_mean() -> None:
    x = np.linspace(0, 1, 60)
    strata = np.repeat(["a", "b", "c"], 20)
    lo1, hi1 = bootstrap_mean_ci(x, strata, 2000, rng_for(5, "k"))
    lo2, hi2 = bootstrap_mean_ci(x, strata, 2000, rng_for(5, "k"))
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= x.mean() <= hi1


def test_describe_values_fields() -> None:
    d = describe_values([0.0, 0.5, 1.0, 1.0])
    assert d["n"] == 4 and d["median"] == 0.75 and d["iqr"] == pytest.approx(d["q3"] - d["q1"])


def test_only_the_four_defined_omnibus_families_exist() -> None:
    assert omnibus_family("A", "satisfaction", False) == ("blockA_satisfaction_primary", "primary")
    assert omnibus_family("A", "satisfaction", True) == ("blockA_satisfaction_no16k_sensitivity", "sensitivity")
    assert omnibus_family("A", "valid_nonempty", False) == ("blockA_valid_nonempty_secondary", "secondary")
    assert omnibus_family("B", "satisfaction", False) == ("blockB_satisfaction", "block B")
    for undefined in (("A", "valid_nonempty", True), ("B", "satisfaction", True), ("B", "valid_nonempty", False)):
        with pytest.raises(AnalysisRefused):
            omnibus_family(*undefined)


def test_fit_mixed_succeeds_on_synthetic_block_a_design() -> None:
    """Regression: lbfgs raised LinAlgError on this design; the fallback chain must fit it."""
    smf = pytest.importorskip("statsmodels.formula.api")
    from scripts.analyse_heldout_factorial import CAPS, _cap_label, fit_mixed
    from scripts.validate_factorial_synthetic import simulate

    y, levels = simulate("null_additive", np.random.default_rng(100))
    frame = pd.DataFrame([
        {"instance_id": f"t{t}", "arch": a, "budget": _cap_label(c), "complexity": levels[t],
         "satisfaction": float(y[t, ai, ci])}
        for t in range(y.shape[0]) for ai, a in enumerate(ARCHS) for ci, c in enumerate(CAPS)])
    model = smf.mixedlm("satisfaction ~ C(arch, Sum) * C(budget, Sum) * C(complexity, Sum)",
                        frame, groups=frame["instance_id"])
    res, log, _ = fit_mixed(model)
    assert res.converged and log["optimizer_used"] == "bfgs"
    assert log["max_abs_fixed_effect_difference"] < 1e-3


def _frame(drop_last: bool = False, duplicate: bool = False) -> pd.DataFrame:
    rows = [{"block": "A", "instance_id": f"t{t}", "arch": a, "cap": c, "satisfaction": 0.5,
             "complexity": "Low", "n_people": 8}
            for t in range(2) for a in ARCHS for c in (16000, 32000, 64000, 128000)]
    if drop_last:
        rows = rows[:-1]
    if duplicate:
        rows.append(rows[0])
    return pd.DataFrame(rows)


def test_outcome_array_shape() -> None:
    tasks, y, strata = outcome_array(_frame(), "A", "satisfaction")
    assert y.shape == (2, 5, 4) and list(strata) == ["Low", "Low"] and tasks == ["t0", "t1"]


def test_outcome_array_refuses_missing_and_duplicate_cells() -> None:
    with pytest.raises(AnalysisRefused):
        outcome_array(_frame(drop_last=True), "A", "satisfaction")
    with pytest.raises(AnalysisRefused):
        outcome_array(_frame(duplicate=True), "A", "satisfaction")
