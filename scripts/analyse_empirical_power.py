# scripts/analyse_empirical_power.py
"""Empirical calibration and power, from real development runs only. CPU, no LLM.

    python scripts/analyse_empirical_power.py --out results/analysis/empirical_power

WHAT THIS IS. The registered route to `N`. The amendment of 2026-08-10 stopped
sizing the experiment from the synthetic intersection-union model -- `N = 35` and
`N = 95` from two runs of it differ by a factor of three, and all of that
difference comes from assumptions nobody measured. The replacement recorded there
is this: take the spread, the zero rate and the tie rate from real development
runs, and freeze `N` from those before the held-out range opens.

WHAT IT READS. Only frozen development run documents: the C1 budget-dev sweep and
the C2/C3/C4/C5 architecture diagnostic, both on the same 60 `n = 8` instances at
cap 64000, paired per instance, 20 per band. No held-out seed is touched -- none
exists.

WHY A SIGN-FLIP NULL. `satisfaction = achieved / O` with `O` in {3, 4}, so a
paired difference can only take a handful of values and a large share of pairs are
exact ties. That is the structure the corrected Type-I validation warned about on
synthetic data. Randomly flipping the signs of the OBSERVED differences is the
exact null of symmetry about zero for a paired design, and it preserves the real
magnitudes, the real tie mass and the real zero mass instead of inventing them.
The Type-I rate measured under it is therefore a statement about this experiment's
own data rather than about a model of it.

TESTS COMPARED
  bootstrap  percentile bootstrap CI on the mean paired difference (the rule whose
             calibration is in question)
  sign       exact two-sided binomial sign test on non-tied pairs
  perm       sign-flip permutation test on the mean

Deterministic: one fixed seed, reported in the output.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

SEED = 20260826
ALPHA = 0.05

DEV_SOURCES = (
    "results/dev_runs/budget_dev/*.json",
    "results/dev_runs/budget_dev_arch_diag/*.json",
)
CAP = 64000
CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
BANDS = ("easy", "medium", "hard")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_paired(root: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(instance_id, band) -> {condition: satisfaction} for the fully paired cell."""
    table: dict[tuple[str, str], dict[str, float]] = {}
    for pattern in DEV_SOURCES:
        for path in glob.glob(str(root / pattern)):
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            rr = doc["run_result"]
            if rr.get("cap") != CAP:
                continue
            key = (rr["instance_id"], rr["level"])
            table.setdefault(key, {})[rr["condition"]] = (rr.get("score") or {})["satisfaction"]
    complete = {k: v for k, v in table.items() if all(c in v for c in CONDITIONS)}
    if not complete:
        raise SystemExit("no fully paired development instances found -- refusing to guess")
    return complete


def differences(paired, a: str, b: str, band: str | None) -> list[float]:
    keys = sorted(k for k in paired if band is None or k[1] == band)
    return [paired[k][a] - paired[k][b] for k in keys]


# --------------------------------------------------------------------------- #
# Tests. Each returns True when it rejects H0 at ALPHA, two-sided.
# --------------------------------------------------------------------------- #
def _mean(xs) -> float:
    return sum(xs) / len(xs)


def test_bootstrap(diffs: Sequence[float], rng: random.Random, b: int = 300) -> bool:
    """Percentile bootstrap CI on the mean excludes zero."""
    n = len(diffs)
    choices = rng.choices
    means = sorted(_mean(choices(diffs, k=n)) for _ in range(b))
    lo = means[int((ALPHA / 2) * b)]
    hi = means[min(int((1 - ALPHA / 2) * b), b - 1)]
    return not (lo <= 0.0 <= hi)


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0 = 0.5, by summing the tail no larger."""
    from math import comb
    if n == 0:
        return 1.0
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12))


def test_sign(diffs: Sequence[float], rng: random.Random) -> bool:
    """Exact binomial sign test; exact ties are discarded, which is the standard rule."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return _binom_two_sided(pos, pos + neg) < ALPHA


def test_perm(diffs: Sequence[float], rng: random.Random, b: int = 300) -> bool:
    """Sign-flip permutation test on the mean."""
    obs = abs(_mean(diffs))
    rand = rng.random
    hits = 0
    for _ in range(b):
        if abs(_mean([d if rand() < 0.5 else -d for d in diffs])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (b + 1) < ALPHA


TESTS: dict[str, Callable[..., bool]] = {
    "bootstrap": test_bootstrap,
    "sign": test_sign,
    "perm": test_perm,
}


# --------------------------------------------------------------------------- #
# Calibration and power, both by resampling the real differences
# --------------------------------------------------------------------------- #
def type_one(diffs: Sequence[float], n: int, rng: random.Random, reps: int) -> dict[str, float]:
    """Rejection rate under the sign-flip null: magnitudes, ties and zeros kept."""
    # perm is omitted on purpose: under a sign-flip null a sign-flip permutation
    # test is exact by construction, so measuring it would report its own definition.
    measured = {k: v for k, v in TESTS.items() if k != "perm"}
    out = {name: 0 for name in measured}
    for _ in range(reps):
        sample = rng.choices(diffs, k=n)
        null = [d if rng.random() < 0.5 else -d for d in sample]
        for name, fn in measured.items():
            if fn(null, rng):
                out[name] += 1
    return {name: c / reps for name, c in out.items()}


def power(diffs: Sequence[float], n: int, rng: random.Random, reps: int) -> dict[str, float]:
    """Rejection rate when resampling the observed differences as they are."""
    out = {name: 0 for name in TESTS}
    for _ in range(reps):
        sample = rng.choices(diffs, k=n)
        for name, fn in TESTS.items():
            if fn(sample, rng):
                out[name] += 1
    return {name: c / reps for name, c in out.items()}


def describe(diffs: Sequence[float]) -> dict:
    n = len(diffs)
    return {
        "n": n,
        "mean": _mean(diffs),
        "median": statistics.median(diffs),
        "sd": statistics.pstdev(diffs) if n > 1 else 0.0,
        "ties": sum(1 for d in diffs if d == 0),
        "tie_rate": sum(1 for d in diffs if d == 0) / n,
        "distinct_values": len(set(diffs)),
        "value_counts": {str(k): v for k, v in sorted(Counter(diffs).items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/empirical_power"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--sizes", default="20,30,50,75")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rng = random.Random(SEED)
    paired = load_paired(args.root)

    contrasts = [("c1_react", c) for c in CONDITIONS if c != "c1_react"]
    scopes: list[tuple[str, str | None]] = [("pooled", None)] + [(b, b) for b in BANDS]

    report: dict = {
        "schema_version": "empirical_power/1.0",
        "standing": (
            "Calibration and power estimated by resampling REAL development paired "
            "differences. Development data only -- no held-out seed is read. This is "
            "the registered replacement for the synthetic power model (amendment "
            "2026-08-10); it binds N and the choice of test, and reports no outcome."
        ),
        "seed": SEED,
        "alpha": ALPHA,
        "cap": CAP,
        "reps": args.reps,
        "paired_instances": len(paired),
        "contrasts": {},
    }

    print(f"paired development instances at cap {CAP}: {len(paired)}  (seed {SEED})")
    for a, b in contrasts:
        label = f"{a}-vs-{b}"
        report["contrasts"][label] = {}
        print(f"\n=== {label}")
        use = scopes if b == "c3_mas" else scopes[:1]
        for scope_name, band in use:
            diffs = differences(paired, a, b, band)
            desc = describe(diffs)
            entry = {"observed": desc, "type_one": {}, "power": {}}
            t1 = type_one(diffs, len(diffs), rng, args.reps)
            entry["type_one"][str(len(diffs))] = t1
            use_sizes = sizes if b == "c3_mas" else sizes[::2]
            for n in use_sizes:
                entry["power"][str(n)] = power(diffs, n, rng, args.reps)
            report["contrasts"][label][scope_name] = entry

            print(f"  {scope_name:<7} n={desc['n']:<3} mean={desc['mean']:+.3f} "
                  f"ties={desc['tie_rate']:.0%} distinct={desc['distinct_values']}")
            print(f"    type-I @n={desc['n']:<3} " +
                  "  ".join(f"{k}={v:.3f}" for k, v in t1.items()))
            for n in use_sizes:
                p = entry["power"][str(n)]
                print(f"    power  @n={n:<3} " +
                      "  ".join(f"{k}={v:.3f}" for k, v in p.items()))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
