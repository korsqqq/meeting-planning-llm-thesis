# scripts/power_iut_v2.py
"""Power simulation v2 for the pre-registered intersection-union test. CPU only.

    python -m scripts.power_iut_v2 --out results/analysis/power_iut_v2

Written and committed before it was run. It replaces `scripts/power_iut.py`, whose result
(N = 35) is kept and marked superseded in `results/analysis/power_iut/SUPERSEDED.md`.

Three things changed, and nothing else:

**A structural no-plan regime.** The earlier model drew achieved meetings from a binomial
and had no way to represent a run that produces no plan at all. In the pilot that was the
dominant source of zeros -- at cap 64 000 only 0.467 of C1 runs held any valid plan. A
hurdle now sits in front of the outcome:

    pi_c   = lambda * (1 - mu_c)                  failure intensity, lambda in {0, 0.5, 0.9}
    G_ci   ~ Bernoulli(1 - pi_c)                  drawn INDEPENDENTLY for C1 and C3
    S_ci   = 0                                     if G_ci = 0
    S_ci   = Binomial(O_i, logistic(alpha_c + u_i)) / O_i    otherwise

`u_i` remains shared between the conditions, so instance difficulty is still paired; only
the failure draw is independent. A shared-failure variant is available as a **non-binding**
sensitivity and cannot lower the selected N, because correlated failures produce more ties
and would flatter the design.

**The target effect stays unconditional.** `alpha_c` is solved so that `E[S_c]` *after* the
hurdle equals `mu_c = m +/- delta/2`, hence `E[S_C1] - E[S_C3] = delta` as specified. The
conditional mean it must hit is `mu_c / (1 - pi_c)`, which stays inside (0, 1] for every
admissible `mu_c` and `lambda`. Calibrating the conditional part instead would silently
shrink the realised effect as `lambda` grows.

**`sigma = 0` is in the grid.** The earlier binding case was the smallest sigma, and that is
systematic: larger heterogeneity pushes the success probability toward 0 and 1 where the
binomial variance is smaller and the paired difference is less noisy. The worst case is
therefore no heterogeneity at all, which the earlier grid excluded.

**Type-I validation is mandatory and can stop the design.** The percentile bootstrap has
never been checked against a boundary null on data this discrete. Two boundary nulls are
simulated, each with one component's difference set to zero. If the null component's
rejection rate is anticonservative against alpha = 0.05 with Monte-Carlo uncertainty taken
into account, the run reports NO-GO. Raising N is not a remedy and is not attempted: an
anticonservative interval procedure stays anticonservative, and the inferential procedure
would have to be revised before the main experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.power_iut import (  # noqa: E402  -- the calibration machinery is unchanged
    BOOTSTRAP_B,
    BOOTSTRAP_CONFIRM,
    O_CONFIGS,
    OUTER_R,
    OUTER_R_ESCALATED,
    TARGET_EFFECT,
    TARGET_POWER,
    _draw_optima,
    _lower_bound,
    _upper_bound,
    marginal_mean,
    solve_alpha,
)

SCHEMA_VERSION = "power_iut/2.0"
SEED = 20260810

BASE_MEANS = (0.35, 0.50, 0.65)
SIGMAS = (0.0, 0.5, 1.0, 1.5)          # sigma = 0 included: it is the worst case
LAMBDAS = (0.0, 0.5, 0.9)              # structural no-plan intensity
N_GRID = tuple(range(30, 101, 5))
N_GRID_EXTENDED = (125, 150, 175, 200)

ALPHA = 0.05
SCENARIOS: dict[str, tuple[float, float]] = {
    "target": (+TARGET_EFFECT, -TARGET_EFFECT),
    "weak": (+0.10, -0.10),
    "strong": (+0.35, -0.35),
    "asymmetric": (+0.10, -0.35),
}
BOUNDARY_NULLS: dict[str, tuple[float, float]] = {
    "A_null_low": (0.0, -TARGET_EFFECT),
    "B_null_high": (+TARGET_EFFECT, 0.0),
}


def failure_probability(mu: float, lam: float) -> float:
    """pi = lambda * (1 - mu): more failure where the ceiling leaves more room."""
    return lam * (1.0 - mu)


def conditional_target(mu: float, lam: float) -> float:
    """The mean the surviving part must hit so the UNCONDITIONAL mean is `mu`."""
    pi = failure_probability(mu, lam)
    if pi >= 1.0:
        raise ValueError(f"failure probability {pi} leaves no mass")
    target = mu / (1.0 - pi)
    if not 0.0 < target < 1.0:
        raise ValueError(
            f"unconditional mean {mu} is unreachable at lambda {lam}: the surviving part "
            f"would need mean {target}")
    return target


def calibrate(mu: float, lam: float, sigma: float) -> tuple[float, float]:
    """(alpha of the conditional logistic, failure probability) for an unconditional mu."""
    return solve_alpha(conditional_target(mu, lam), sigma), failure_probability(mu, lam)


@dataclass(frozen=True)
class Config:
    n_per_band: int
    base_mean: float
    sigma: float
    lam: float
    o_config: str
    delta_low: float
    delta_high: float
    shared_failure: bool = False


def _band_deltas(rng: np.random.Generator, n: int, spec1: tuple[float, float],
                 spec3: tuple[float, float], sigma: float, o_config: str, *,
                 shared_failure: bool = False) -> np.ndarray:
    """Paired S_C1 - S_C3 for one band, with the hurdle in front of the outcome."""
    (a1, pi1), (a3, pi3) = spec1, spec3
    optima = _draw_optima(rng, n, o_config)
    u = rng.normal(0.0, sigma, size=n) if sigma > 0 else np.zeros(n)
    p1 = 1.0 / (1.0 + np.exp(-(a1 + u)))
    p3 = 1.0 / (1.0 + np.exp(-(a3 + u)))
    s1 = rng.binomial(optima, p1) / optima
    s3 = rng.binomial(optima, p3) / optima
    if shared_failure:
        # Non-binding sensitivity only: one draw decides both, which produces more ties.
        v = rng.random(n)
        s1 = np.where(v < pi1, 0.0, s1)
        s3 = np.where(v < pi3, 0.0, s3)
    else:
        s1 = np.where(rng.random(n) < pi1, 0.0, s1)
        s3 = np.where(rng.random(n) < pi3, 0.0, s3)
    return s1 - s3


def simulate(cfg: Config, *, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B,
             seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(
        abs(hash((cfg.n_per_band, cfg.base_mean, cfg.sigma, cfg.lam, cfg.o_config,
                  cfg.delta_low, cfg.delta_high, cfg.shared_failure, seed))) % (2 ** 63))
    m = cfg.base_mean
    low_c1 = calibrate(m + cfg.delta_low / 2, cfg.lam, cfg.sigma)
    low_c3 = calibrate(m - cfg.delta_low / 2, cfg.lam, cfg.sigma)
    high_c1 = calibrate(m + cfg.delta_high / 2, cfg.lam, cfg.sigma)
    high_c3 = calibrate(m - cfg.delta_high / 2, cfg.lam, cfg.sigma)

    low_hits = high_hits = both = 0
    for _ in range(replicates):
        d_low = _band_deltas(rng, cfg.n_per_band, low_c1, low_c3, cfg.sigma, cfg.o_config,
                             shared_failure=cfg.shared_failure)
        d_high = _band_deltas(rng, cfg.n_per_band, high_c1, high_c3, cfg.sigma,
                              cfg.o_config, shared_failure=cfg.shared_failure)
        lo = _lower_bound(rng, d_low, bootstrap) > 0.0
        hi = _upper_bound(rng, d_high, bootstrap) < 0.0
        low_hits += lo
        high_hits += hi
        both += lo and hi

    power = both / replicates
    se = math.sqrt(max(power * (1 - power), 1e-12) / replicates)
    p_low, p_high = low_hits / replicates, high_hits / replicates

    def ci(p: float) -> list[float]:
        s = math.sqrt(max(p * (1 - p), 1e-12) / replicates)
        return [round(max(0.0, p - 1.96 * s), 4), round(min(1.0, p + 1.96 * s), 4)]

    return {
        **asdict(cfg),
        "replicates": replicates, "bootstrap": bootstrap,
        "power_iut": round(power, 4), "power_low": round(p_low, 4),
        "power_high": round(p_high, 4),
        "product_sanity_check": round(p_low * p_high, 4),
        "mc_se": round(se, 5), "mc_ci95": ci(power),
        "mc_ci95_low": ci(p_low), "mc_ci95_high": ci(p_high),
    }


def binding_grid(n: int, delta_low: float, delta_high: float) -> Iterator[Config]:
    """Every binding nuisance configuration: base mean x sigma x lambda x optimum."""
    for m, sigma, lam, o in product(BASE_MEANS, SIGMAS, LAMBDAS, sorted(O_CONFIGS)):
        yield Config(n, m, sigma, lam, o, delta_low, delta_high)


def _straddles(ci: Sequence[float], boundary: float) -> bool:
    return ci[0] <= boundary <= ci[1]


def evaluate_n(n: int, *, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B,
               seed: int = SEED) -> dict[str, Any]:
    dl, dh = SCENARIOS["target"]
    cells: list[dict[str, Any]] = []
    for cfg in binding_grid(n, dl, dh):
        cell = simulate(cfg, replicates=replicates, bootstrap=bootstrap, seed=seed)
        if _straddles(cell["mc_ci95"], TARGET_POWER):
            cell = simulate(cfg, replicates=OUTER_R_ESCALATED, bootstrap=bootstrap,
                            seed=seed)
            cell["escalated"] = True
        cells.append(cell)
        if cell["power_iut"] < TARGET_POWER:
            return {"n_per_band": n, "passes": False, "cells": cells,
                    "failed_on": {k: cell[k] for k in
                                  ("base_mean", "sigma", "lam", "o_config", "power_iut",
                                   "mc_ci95")}}
    worst = min(cells, key=lambda c: c["power_iut"])
    return {"n_per_band": n, "passes": True, "cells": cells, "n_cells": len(cells),
            "worst_cell": {k: worst[k] for k in
                           ("base_mean", "sigma", "lam", "o_config", "power_iut",
                            "mc_ci95", "replicates")}}


def type_one(n: int, *, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B,
             seed: int = SEED) -> dict[str, Any]:
    """Boundary nulls. The sharp diagnostic is the NULL component's rejection rate.

    Under an intersection-union test the joint rate is bounded by the null component's own
    rate, so the joint figure alone can hide an anticonservative interval behind a
    component that rarely fires. Both are reported and the component rate decides.
    """
    out: dict[str, Any] = {}
    for name, (dl, dh) in BOUNDARY_NULLS.items():
        null_side = "low" if dl == 0.0 else "high"
        worst: dict[str, Any] | None = None
        for cfg in binding_grid(n, dl, dh):
            cell = simulate(cfg, replicates=replicates, bootstrap=bootstrap, seed=seed)
            rate = cell[f"power_{null_side}"]
            if _straddles(cell[f"mc_ci95_{null_side}"], ALPHA):
                cell = simulate(cfg, replicates=OUTER_R_ESCALATED, bootstrap=bootstrap,
                                seed=seed)
                cell["escalated"] = True
                rate = cell[f"power_{null_side}"]
            if worst is None or rate > worst[f"power_{null_side}"]:
                worst = cell
        assert worst is not None
        rate = worst[f"power_{null_side}"]
        ci = worst[f"mc_ci95_{null_side}"]
        out[name] = {
            "null_component": null_side,
            "worst_null_rejection_rate": rate,
            "mc_ci95": ci,
            "joint_rejection_rate": worst["power_iut"],
            "config": {k: worst[k] for k in ("base_mean", "sigma", "lam", "o_config")},
            # Anticonservative only when the interval lies entirely above alpha; a rate
            # above alpha whose interval still covers it is Monte-Carlo noise.
            "anticonservative": ci[0] > ALPHA,
        }
    out["verdict"] = ("OK" if not any(v["anticonservative"] for v in out.values()
                                      if isinstance(v, dict)) else "ANTICONSERVATIVE")
    return out


def search(*, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B, seed: int = SEED,
           grid: Sequence[int] = N_GRID, extended: Sequence[int] = N_GRID_EXTENDED,
           ) -> dict[str, Any]:
    tried = []
    for n in list(grid) + list(extended):
        r = evaluate_n(n, replicates=replicates, bootstrap=bootstrap, seed=seed)
        tried.append({k: r[k] for k in r if k != "cells"})
        if r["passes"]:
            return {"selected_n": n, "attempts": tried, "selected_detail": r}
    return {"selected_n": None, "attempts": tried, "selected_detail": None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/analysis/power_iut_v2"))
    ap.add_argument("--replicates", type=int, default=OUTER_R)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--grid", default=",".join(str(n) for n in N_GRID))
    args = ap.parse_args(argv)

    grid = tuple(int(x) for x in args.grid.split(",") if x)
    found = search(replicates=args.replicates, bootstrap=args.bootstrap, grid=grid)
    n = found["selected_n"]

    scenarios: dict[str, Any] = {}
    type_one_result: dict[str, Any] = {}
    confirmation: dict[str, Any] = {}
    if n:
        for name, (dl, dh) in SCENARIOS.items():
            cells = [simulate(c, replicates=args.replicates, bootstrap=args.bootstrap)
                     for c in binding_grid(n, dl, dh)]
            worst = min(cells, key=lambda c: c["power_iut"])
            scenarios[name] = {
                "delta_low": dl, "delta_high": dh,
                "worst_power": worst["power_iut"],
                "median_power": round(float(np.median([c["power_iut"] for c in cells])), 4),
                "selects_n": name == "target",
            }
        type_one_result = type_one(n, replicates=args.replicates, bootstrap=args.bootstrap)
        conf = evaluate_n(n, replicates=args.replicates, bootstrap=BOOTSTRAP_CONFIRM)
        confirmation = {"bootstrap": BOOTSTRAP_CONFIRM, "passes": conf["passes"],
                        "worst": conf.get("worst_cell") or conf.get("failed_on")}
        # Non-binding: correlated failures make the design look better, so this may not
        # lower N and is reported only.
        shared = [simulate(Config(n, c.base_mean, c.sigma, c.lam, c.o_config,
                                  c.delta_low, c.delta_high, shared_failure=True),
                           replicates=args.replicates, bootstrap=args.bootstrap)
                  for c in binding_grid(n, *SCENARIOS["target"])]
        scenarios["shared_failure_sensitivity"] = {
            "binding": False,
            "worst_power": min(c["power_iut"] for c in shared),
            "note": "correlated failures cannot lower the selected N",
        }

    go = bool(n) and type_one_result.get("verdict") == "OK" and confirmation.get("passes")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": "results/analysis/power_iut (N=35)",
        "binding_axes": {"base_mean": list(BASE_MEANS), "sigma_logit": list(SIGMAS),
                         "lambda_no_plan": list(LAMBDAS),
                         "o_config": {k: v for k, v in O_CONFIGS.items()}},
        "selection_rule": "smallest N reaching target power in EVERY binding configuration",
        "monte_carlo": {"replicates": args.replicates,
                        "escalated": OUTER_R_ESCALATED, "seed": SEED,
                        "escalation": "any cell whose 95% interval straddles the decision "
                                      "boundary is re-run automatically"},
        "search": found,
        "confirmation_at_registered_bootstrap": confirmation,
        "type_one_validation": type_one_result,
        "scenarios_at_selected_n": scenarios,
        "verdict": "GO" if go else "NO-GO",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    cells_per_n = len(BASE_MEANS) * len(SIGMAS) * len(LAMBDAS) * len(O_CONFIGS)
    print(f"binding cells per N: {cells_per_n}")
    for a in found["attempts"]:
        d = a.get("worst_cell") or a.get("failed_on") or {}
        print(f"  N={a['n_per_band']:>4}  {'PASS' if a['passes'] else 'fail'}  "
              f"power {d.get('power_iut')}  m={d.get('base_mean')} s={d.get('sigma')} "
              f"lam={d.get('lam')} {d.get('o_config')}")
    print(f"SELECTED N: {n}")
    for name, s in scenarios.items():
        print(f"  {name:<28} worst {s['worst_power']}")
    for name, t in type_one_result.items():
        if isinstance(t, dict):
            print(f"  type-I {name:<12} null={t['null_component']} "
                  f"rate {t['worst_null_rejection_rate']} CI {t['mc_ci95']} "
                  f"{'ANTICONSERVATIVE' if t['anticonservative'] else 'ok'}")
    print(f"VERDICT: {doc['verdict']}")
    print(f"written: {args.out}/summary.json")
    return 0 if go else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
