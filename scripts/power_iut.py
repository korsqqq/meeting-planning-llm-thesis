# scripts/power_iut.py
"""Power simulation for the pre-registered intersection-union test. CPU only.

    python -m scripts.power_iut --out results/analysis/power_iut

Answers one question: **how many instances per band are needed so that the primary test
has at least 80% power against the target effect?** The 50 carried over from the earlier
design is deliberately not assumed; the primary inference is now a conjunction, whose power
is governed by the weaker component, so the number has to be established against that rule.

Written and committed before it was run.

The decision rule simulated here is the one registered in THESIS_DECISIONS section 5, not
an approximation of it: paired bootstrap over instances, percentile method, and a crossover
declared only when the 95% one-sided LOWER bound on the Low-band difference is above zero
AND the 95% one-sided UPPER bound on the High-band difference is below zero.

Power is measured **directly** as the share of simulations in which both component
decisions pass. The product of the marginal rates is reported beside it as an independence
sanity check -- the two bands are disjoint samples, so the two should agree -- rather than
being used in place of the measurement.

### Outcome model

Satisfaction is a bounded ratio with mass at 0 and 1, and the paired difference has a large
point mass at zero: in the pilot, between a third and all of the pairs were ties. A normal
model would miss exactly the feature that governs power. Instead the achieved meetings are
modelled, and the discreteness, the bounds and the ties follow:

    u_i  ~ Normal(0, sigma)                     shared by both conditions in instance i
    p_ci = logistic(alpha_c + u_i)
    a_ci ~ Binomial(O_i, p_ci)
    S_ci = a_ci / O_i

`u_i` is shared, which is what makes the design paired: instances differ in difficulty and
both architectures feel the same instance. `alpha_c` is solved numerically so that the
marginal mean of `p_c`, integrated over `u`, equals the requested value. A clipped
probability scale was considered and rejected: clipping at 0 and 1 changes the realised
marginal difference, so a run labelled "delta = 0.20" would not have that difference.

### What selects N, and what only gets reported

The **target effect** is a difference of 0.20 in oracle-normalised satisfaction -- twenty
percentage points -- with the sign reversed between bands. It is the minimum difference of
substantive interest and the effect the experiment is sized for; it is not "medium".

N is chosen as the smallest value on the grid whose joint power reaches 0.80 **in every
nuisance configuration**, not in one convenient central case. The weak, strong and
asymmetric scenarios are reported and select nothing. The asymmetric one matters most for
reading the result: when one side of the crossover is weaker, the conjunction is governed
by that side.

The optimum distribution is a **sensitivity axis, not a frozen design assumption**. The
41:24 split is the calibration pool's maximum common capacity, not an adopted histogram;
adopting it here and then deriving the histogram from N would be circular.
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

SCHEMA_VERSION = "power_iut/1.0"

TARGET_EFFECT = 0.20
TARGET_POWER = 0.80
N_GRID = tuple(range(30, 101, 5))
N_GRID_EXTENDED = (125, 150, 175, 200)
BASE_MEANS = (0.35, 0.50, 0.65)
SIGMAS = (0.5, 1.0, 1.5)            # logit scale: one sigma multiplies the odds by e^sigma
BOOTSTRAP_B = 2000                  # inside the simulation; the registered analysis uses
BOOTSTRAP_CONFIRM = 10000           # this many, so the selected N is confirmed with it
OUTER_R = 2000
OUTER_R_ESCALATED = 10000
SEED = 20260809

# Optimum sensitivity. A is the calibration pool's common capacity, B and C are the coarse
# extremes. None of them is an adopted design histogram.
O_CONFIGS: dict[str, dict[int, float]] = {
    "A_calibration_mix": {3: 41 / 65, 4: 24 / 65},
    "B_all_optimum_3": {3: 1.0},
    "C_all_optimum_4": {4: 1.0},
}

SCENARIOS: dict[str, tuple[float, float]] = {
    "target": (+TARGET_EFFECT, -TARGET_EFFECT),
    "weak": (+0.10, -0.10),
    "strong": (+0.35, -0.35),
    "asymmetric": (+0.10, -0.35),
}

_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(64)


def marginal_mean(alpha: float, sigma: float) -> float:
    """E_u[ logistic(alpha + u) ] for u ~ Normal(0, sigma), by Gauss-Hermite quadrature."""
    x = alpha + sigma * _GH_NODES
    return float(np.sum(_GH_WEIGHTS / np.sqrt(2 * math.pi)
                        * (1.0 / (1.0 + np.exp(-x)))))


def solve_alpha(target: float, sigma: float, *, tol: float = 1e-10) -> float:
    """The intercept whose marginal mean is `target`. Bisection; the map is monotone."""
    if not 0.0 < target < 1.0:
        raise ValueError(f"marginal mean {target} outside (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if marginal_mean(mid, sigma) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


@dataclass(frozen=True)
class Config:
    n_per_band: int
    base_mean: float
    sigma: float
    o_config: str
    delta_low: float
    delta_high: float


def _draw_optima(rng: np.random.Generator, n: int, o_config: str) -> np.ndarray:
    weights = O_CONFIGS[o_config]
    values = np.array(sorted(weights), dtype=np.int64)
    probs = np.array([weights[v] for v in values], dtype=float)
    return rng.choice(values, size=n, p=probs / probs.sum())


def _band_deltas(rng: np.random.Generator, n: int, alpha_c1: float, alpha_c3: float,
                 sigma: float, o_config: str) -> np.ndarray:
    """Paired satisfaction differences S_C1 - S_C3 for one band, one simulated sample."""
    optima = _draw_optima(rng, n, o_config)
    u = rng.normal(0.0, sigma, size=n)
    p1 = 1.0 / (1.0 + np.exp(-(alpha_c1 + u)))
    p3 = 1.0 / (1.0 + np.exp(-(alpha_c3 + u)))
    a1 = rng.binomial(optima, p1)
    a3 = rng.binomial(optima, p3)
    return (a1 - a3) / optima


def _lower_bound(rng: np.random.Generator, deltas: np.ndarray, b: int) -> float:
    """95% one-sided lower percentile bound on the mean, resampling instances."""
    idx = rng.integers(0, deltas.size, size=(b, deltas.size))
    return float(np.percentile(deltas[idx].mean(axis=1), 5.0))


def _upper_bound(rng: np.random.Generator, deltas: np.ndarray, b: int) -> float:
    idx = rng.integers(0, deltas.size, size=(b, deltas.size))
    return float(np.percentile(deltas[idx].mean(axis=1), 95.0))


def simulate(cfg: Config, *, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B,
             seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(
        abs(hash((cfg.n_per_band, cfg.base_mean, cfg.sigma, cfg.o_config,
                  cfg.delta_low, cfg.delta_high, seed))) % (2 ** 63))
    m = cfg.base_mean
    # Low: C1 ahead by delta_low. High: C3 ahead by |delta_high|. The base level is held
    # constant so only the architectural advantage moves.
    a_low_c1 = solve_alpha(m + cfg.delta_low / 2, cfg.sigma)
    a_low_c3 = solve_alpha(m - cfg.delta_low / 2, cfg.sigma)
    a_high_c1 = solve_alpha(m + cfg.delta_high / 2, cfg.sigma)
    a_high_c3 = solve_alpha(m - cfg.delta_high / 2, cfg.sigma)

    low_hits = high_hits = both_hits = 0
    for _ in range(replicates):
        d_low = _band_deltas(rng, cfg.n_per_band, a_low_c1, a_low_c3, cfg.sigma,
                             cfg.o_config)
        d_high = _band_deltas(rng, cfg.n_per_band, a_high_c1, a_high_c3, cfg.sigma,
                              cfg.o_config)
        low_ok = _lower_bound(rng, d_low, bootstrap) > 0.0
        high_ok = _upper_bound(rng, d_high, bootstrap) < 0.0
        low_hits += low_ok
        high_hits += high_ok
        both_hits += low_ok and high_ok

    p_low = low_hits / replicates
    p_high = high_hits / replicates
    power = both_hits / replicates
    se = math.sqrt(max(power * (1 - power), 1e-12) / replicates)
    return {
        **asdict(cfg),
        "replicates": replicates,
        "bootstrap": bootstrap,
        "power_iut": round(power, 4),
        "power_low": round(p_low, 4),
        "power_high": round(p_high, 4),
        "product_sanity_check": round(p_low * p_high, 4),
        "mc_se": round(se, 5),
        "mc_ci95": [round(max(0.0, power - 1.96 * se), 4),
                    round(min(1.0, power + 1.96 * se), 4)],
    }


def nuisance_grid(n: int, delta_low: float, delta_high: float) -> Iterator[Config]:
    for m, sigma, o in product(BASE_MEANS, SIGMAS, sorted(O_CONFIGS)):
        yield Config(n_per_band=n, base_mean=m, sigma=sigma, o_config=o,
                     delta_low=delta_low, delta_high=delta_high)


def _decisive(cell: dict[str, Any], *, target: float = TARGET_POWER) -> bool:
    """A cell decides only when its Monte-Carlo interval does not straddle the target."""
    lo, hi = cell["mc_ci95"]
    return not (lo <= target <= hi)


def evaluate_n(n: int, *, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B,
               seed: int = SEED) -> dict[str, Any]:
    """Every nuisance configuration at the target effect. Aborts on the first failure.

    Escalation is automatic and leaves no room for judgement: a cell whose interval
    contains the target is re-run with more replicates before it is allowed to decide.
    """
    dl, dh = SCENARIOS["target"]
    cells: list[dict[str, Any]] = []
    for cfg in nuisance_grid(n, dl, dh):
        cell = simulate(cfg, replicates=replicates, bootstrap=bootstrap, seed=seed)
        if not _decisive(cell):
            cell = simulate(cfg, replicates=OUTER_R_ESCALATED, bootstrap=bootstrap,
                            seed=seed)
            cell["escalated"] = True
        cells.append(cell)
        if cell["power_iut"] < TARGET_POWER:
            return {"n_per_band": n, "passes": False, "cells": cells,
                    "failed_on": {k: cell[k] for k in
                                  ("base_mean", "sigma", "o_config", "power_iut")}}
    worst = min(cells, key=lambda c: c["power_iut"])
    return {"n_per_band": n, "passes": True, "cells": cells,
            "worst_cell": {k: worst[k] for k in
                           ("base_mean", "sigma", "o_config", "power_iut")}}


def search(*, replicates: int = OUTER_R, bootstrap: int = BOOTSTRAP_B, seed: int = SEED,
           grid: Sequence[int] = N_GRID, extended: Sequence[int] = N_GRID_EXTENDED,
           ) -> dict[str, Any]:
    tried: list[dict[str, Any]] = []
    for n in list(grid) + list(extended):
        result = evaluate_n(n, replicates=replicates, bootstrap=bootstrap, seed=seed)
        tried.append({k: result[k] for k in result if k != "cells"})
        if result["passes"]:
            return {"selected_n": n, "attempts": tried, "selected_detail": result}
    return {"selected_n": None, "attempts": tried, "selected_detail": None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/analysis/power_iut"))
    ap.add_argument("--replicates", type=int, default=OUTER_R)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--grid", default=",".join(str(n) for n in N_GRID))
    args = ap.parse_args(argv)

    grid = tuple(int(x) for x in args.grid.split(",") if x)
    found = search(replicates=args.replicates, bootstrap=args.bootstrap, grid=grid)

    scenarios: dict[str, Any] = {}
    if found["selected_n"]:
        n = found["selected_n"]
        for name, (dl, dh) in SCENARIOS.items():
            cells = [simulate(cfg, replicates=args.replicates, bootstrap=args.bootstrap)
                     for cfg in nuisance_grid(n, dl, dh)]
            worst = min(cells, key=lambda c: c["power_iut"])
            scenarios[name] = {
                "delta_low": dl, "delta_high": dh,
                "worst_power": worst["power_iut"],
                "median_power": round(float(np.median([c["power_iut"] for c in cells])), 4),
                "selects_n": name == "target",
            }
        # The registered analysis uses more bootstrap resamples than the search; the chosen
        # N is confirmed under exactly that setting before it is reported.
        confirm = evaluate_n(n, replicates=args.replicates, bootstrap=BOOTSTRAP_CONFIRM)
        found["confirmation_at_registered_bootstrap"] = {
            "bootstrap": BOOTSTRAP_CONFIRM, "passes": confirm["passes"],
            "worst": confirm.get("worst_cell") or confirm.get("failed_on"),
        }

    doc = {
        "schema_version": SCHEMA_VERSION,
        "selects_n_on": {"effect": "target", "delta": TARGET_EFFECT,
                         "power": TARGET_POWER,
                         "rule": "smallest N reaching the target power in EVERY nuisance "
                                 "configuration"},
        "nuisance_axes": {"base_mean": list(BASE_MEANS), "sigma_logit": list(SIGMAS),
                          "o_config": {k: v for k, v in O_CONFIGS.items()}},
        "o_distribution_status": "sensitivity axis, not an adopted design histogram",
        "monte_carlo": {"replicates": args.replicates,
                        "escalated_replicates": OUTER_R_ESCALATED,
                        "escalation_rule": "any cell whose 95% interval contains the "
                                           "target power is re-run automatically",
                        "seed": SEED},
        "bootstrap": {"unit": "instance", "paired": True, "method": "percentile",
                      "resamples_in_search": args.bootstrap,
                      "resamples_registered": BOOTSTRAP_CONFIRM,
                      "low_rule": "5th percentile of resampled mean > 0",
                      "high_rule": "95th percentile of resampled mean < 0"},
        "search": found,
        "scenarios_at_selected_n": scenarios,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    print(f"target effect {TARGET_EFFECT}, target power {TARGET_POWER}, "
          f"{len(BASE_MEANS) * len(SIGMAS) * len(O_CONFIGS)} nuisance cells per N")
    for a in found["attempts"]:
        mark = "PASS" if a["passes"] else "fail"
        detail = a.get("worst_cell") or a.get("failed_on") or {}
        print(f"  N={a['n_per_band']:>4}  {mark}  worst power "
              f"{detail.get('power_iut')}  at m={detail.get('base_mean')} "
              f"sigma={detail.get('sigma')} {detail.get('o_config')}")
    print(f"SELECTED N: {found['selected_n']}")
    for name, s in scenarios.items():
        print(f"  {name:<11} delta {s['delta_low']:+.2f}/{s['delta_high']:+.2f}  "
              f"worst {s['worst_power']:.3f}  median {s['median_power']:.3f}"
              + ("   <- selects N" if s["selects_n"] else ""))
    print(f"written: {args.out}/summary.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
