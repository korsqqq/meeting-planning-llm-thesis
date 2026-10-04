# scripts/type_one_validation.py
"""Corrected Type-I validation of the bootstrap decision rule. CPU only.

    python -m scripts.type_one_validation --n 95

Replaces the check inside `scripts/power_iut_v2.py`, whose verdict is recorded and set
aside in `736f9c4`. That rule declared the bootstrap anticonservative from the **worst** of
108 nuisance cells, which reads a maximum over 108 draws as a single estimate: at 2000
replicates one cell has a standard error of 0.0049 under a true rate of 0.05, so the
maximum is expected near 0.063 when nothing at all is wrong. A threshold applied to the
maximum of a large grid is crossed by construction.

What this does instead:

* **every** configuration is tested, for both boundary nulls, and none is selected after
  the fact. Re-running "the two worst" at high precision was the obvious next move and is
  exactly the same selection effect one level down;
* each cell is a one-sided test of `H0: rejection rate <= 0.05` with an **exact binomial
  tail**, not an interval comparison;
* the family of 216 tests is corrected by **Holm**, which controls the family-wise error
  rate without assuming independence between cells;
* 10 000 replicates per cell, fixed seeds.

The procedure is called anticonservative only if some cell survives Holm at the family
level. If it does, the answer is to change the statistical test. **Raising `N` is not a
remedy and is not offered here**: an anticonservative interval procedure stays
anticonservative at any sample size.

Only the null component is simulated. Under an intersection-union test the joint rejection
rate is bounded above by the null component's own rate, so the component rate is both the
sharp diagnostic and the conservative one; simulating the other band would double the cost
and could not change a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.power_iut import O_CONFIGS, TARGET_EFFECT, _lower_bound, _upper_bound  # noqa: E402
from scripts.power_iut_v2 import (  # noqa: E402
    ALPHA,
    BASE_MEANS,
    LAMBDAS,
    SIGMAS,
    _band_deltas,
    calibrate,
)

SCHEMA_VERSION = "type_one_validation/1.0"
SEED = 20260810
REPLICATES = 10_000
BOOTSTRAP = 2_000
DEFAULT_N = 95

# One component of each boundary null is exactly on the null; the other is at the target
# effect, matching the situation the main analysis would actually be in.
BOUNDARY_NULLS: dict[str, tuple[float, float]] = {
    "A_null_low": (0.0, -TARGET_EFFECT),
    "B_null_high": (+TARGET_EFFECT, 0.0),
}


def binom_sf(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p), summed exactly in log space.

    An exact tail rather than a normal approximation: the decision sits in the far tail
    after Holm, where the approximation is least trustworthy and cheapest to avoid.
    """
    if x <= 0:
        return 1.0
    if x > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms: list[float] = []
    peak = -math.inf
    for k in range(x, n + 1):
        t = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
             + k * log_p + (n - k) * log_q)
        peak = max(peak, t)
        terms.append(t)
        # Past the mode the terms fall monotonically; stop once they cannot contribute.
        if t < peak - 60 and k > n * p:
            break
    return float(min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms)))


def holm(pvalues: Sequence[float], alpha: float = ALPHA) -> list[bool]:
    """Holm step-down. Returns, per input position, whether that test is rejected."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - rank):
            rejected[idx] = True
        else:
            break                      # step-down: everything larger is retained too
    return rejected


@dataclass(frozen=True)
class Cell:
    boundary_null: str
    null_component: str
    base_mean: float
    sigma: float
    lam: float
    o_config: str


def grid() -> Iterator[Cell]:
    for name, (dl, dh) in BOUNDARY_NULLS.items():
        side = "low" if dl == 0.0 else "high"
        for m, sigma, lam, o in product(BASE_MEANS, SIGMAS, LAMBDAS, sorted(O_CONFIGS)):
            yield Cell(name, side, m, sigma, lam, o)


def rejection_rate(cell: Cell, n: int, *, replicates: int = REPLICATES,
                   bootstrap: int = BOOTSTRAP, seed: int = SEED) -> int:
    """How often the decision rule fires on a component whose true difference is zero."""
    rng = np.random.default_rng(
        abs(hash((cell.boundary_null, cell.base_mean, cell.sigma, cell.lam,
                  cell.o_config, n, seed))) % (2 ** 63))
    m = cell.base_mean
    # The null component has both conditions at the same marginal mean, so its true
    # difference is exactly zero; nothing else about the generator changes.
    spec = calibrate(m, cell.lam, cell.sigma)
    hits = 0
    for _ in range(replicates):
        d = _band_deltas(rng, n, spec, spec, cell.sigma, cell.o_config)
        if cell.null_component == "low":
            hits += _lower_bound(rng, d, bootstrap) > 0.0
        else:
            hits += _upper_bound(rng, d, bootstrap) < 0.0
    return hits


def validate(n: int, *, replicates: int = REPLICATES, bootstrap: int = BOOTSTRAP,
             seed: int = SEED, progress: bool = True) -> dict[str, Any]:
    cells = list(grid())
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for i, cell in enumerate(cells, 1):
        hits = rejection_rate(cell, n, replicates=replicates, bootstrap=bootstrap,
                              seed=seed)
        rate = hits / replicates
        rows.append({
            "boundary_null": cell.boundary_null, "null_component": cell.null_component,
            "base_mean": cell.base_mean, "sigma": cell.sigma, "lam": cell.lam,
            "o_config": cell.o_config, "rejections": hits, "replicates": replicates,
            "rate": round(rate, 5),
            "p_one_sided": binom_sf(hits, replicates, ALPHA),
        })
        if progress:
            done = time.perf_counter() - started
            eta = done / i * (len(cells) - i)
            print(f"  [{i:>3}/{len(cells)}] {cell.boundary_null} m={cell.base_mean} "
                  f"s={cell.sigma} lam={cell.lam} {cell.o_config[:3]}  "
                  f"rate {rate:.4f}  p {rows[-1]['p_one_sided']:.4g}  "
                  f"eta {eta / 60:.1f} min", flush=True)

    flags = holm([r["p_one_sided"] for r in rows])
    for r, f in zip(rows, flags):
        r["rejected_after_holm"] = f
    surviving = [r for r in rows if r["rejected_after_holm"]]
    worst = max(rows, key=lambda r: r["rate"])
    smallest_p = min(rows, key=lambda r: r["p_one_sided"])

    return {
        "n_per_band": n,
        "family_size": len(rows),
        "alpha": ALPHA,
        "correction": "Holm step-down over the whole family",
        "cells_surviving_holm": len(surviving),
        "surviving": surviving,
        "highest_rate_cell": {k: worst[k] for k in
                              ("boundary_null", "base_mean", "sigma", "lam", "o_config",
                               "rate", "p_one_sided", "rejected_after_holm")},
        "smallest_p_cell": {k: smallest_p[k] for k in
                            ("boundary_null", "base_mean", "sigma", "lam", "o_config",
                             "rate", "p_one_sided", "rejected_after_holm")},
        "mean_rate": round(float(np.mean([r["rate"] for r in rows])), 5),
        "verdict": "OK" if not surviving else "ANTICONSERVATIVE",
        "cells": rows,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/type_one_validation"))
    args = ap.parse_args(argv)

    print(f"Type-I validation at N={args.n}: {len(list(grid()))} cells, "
          f"{args.replicates} replicates each, Holm over the whole family")
    result = validate(args.n, replicates=args.replicates, bootstrap=args.bootstrap)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": "the worst-cell rule inside scripts/power_iut_v2.py",
        "decision_rule": "one-sided exact binomial test of H0: rate <= 0.05 per cell, "
                         "Holm-corrected over the family; anticonservative iff any cell "
                         "survives",
        "not_a_remedy": "if this fails, the statistical test changes and N does not",
        "bootstrap": {"unit": "instance", "method": "percentile",
                      "resamples": args.bootstrap,
                      "low_rule": "5th percentile of resampled mean > 0",
                      "high_rule": "95th percentile of resampled mean < 0"},
        "seed": SEED,
        "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")

    print()
    print(f"family size            : {result['family_size']}")
    print(f"mean rejection rate    : {result['mean_rate']}")
    print(f"highest-rate cell      : {result['highest_rate_cell']}")
    print(f"smallest p cell        : {result['smallest_p_cell']}")
    print(f"surviving Holm         : {result['cells_surviving_holm']}")
    print(f"VERDICT                : {result['verdict']}")
    print(f"written: {args.out}/summary.json")
    return 0 if result["verdict"] == "OK" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
