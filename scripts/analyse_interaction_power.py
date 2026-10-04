# scripts/analyse_interaction_power.py
"""Crossover and interaction power, under the plan frozen 2026-08-27. CPU, no LLM.

    uv run --with numpy python scripts/analyse_interaction_power.py

WHY THE OVERLAY. numpy is not a project dependency and must not become one: the
HPC provenance record hashes `uv.lock`, and that hash has to match between the
calibration jobs already run and the held-out run still to come. `uv run --with`
layers numpy on ephemerally and leaves the lock untouched. Checked, not assumed.

WHAT THIS IS. The registered computation behind
`results/analysis/interaction_power/PRE_ANALYSIS_PLAN.md` and its amendment of
2026-08-27. It sizes the held-out experiment for the CROSSOVER claim. The
companion `analyse_empirical_power.py` sized it for the per-band MAIN effect;
that is a different and easier question, and its N does not transfer.

    delta_b = satisfaction(C3) - satisfaction(C1)      within band b

Note the sign: this is the NEGATION of the expose convention `S_C1 - S_C3`.
Every number here uses the delta above.

WHY AN ASSUMED ALTERNATIVE. In the development runs delta is positive in all
three bands -- C3 ahead everywhere, no crossover anywhere. Resampling those
differences would return crossover power near zero by construction and would say
nothing about whether the design can detect a crossover that exists. So the noise
structure (magnitudes, tie mass, zero mass, the coarse achieved/O lattice) comes
from the data and the SIGN PATTERN is assumed:

    high band : observed differences, unchanged      (positive)
    low band  : observed differences, negated        (positive -> negative)

Sign reversal keeps every value on the lattice. A constant shift would not, and
would quietly destroy the tie structure this analysis exists to respect.
Effect-size sensitivity is by thinning: with probability 1-s a drawn difference
is replaced by 0. That shrinks the mean by exactly s, keeps the lattice, and
raises the tie rate -- the conservative direction.

MEDIUM BAND. Descriptive only, per amendment A2. It is given no crossover
orientation, enters no Holm family, and does not determine N. Reporting it under
an assumed sign would invent the very thing the band is meant to describe.

TESTS. Amendment A1 fixes one estimand -- the mean paired difference -- so there
is one primary test, the permutation test on the mean. The sign test targets
P(delta > 0), a different functional, and appears here as sensitivity only.

  primary   perm_low   one-sided sign-flip permutation, H0: delta_low  >= 0
            perm_high  one-sided sign-flip permutation, H0: delta_high <= 0
            IUT        rejects only if BOTH reject; size <= alpha by construction
            Delta      two-sided LABEL permutation on delta_high - delta_low
  sensitivity  the IUT rebuilt from the exact binomial sign test

ASSUMPTIONS, stated rather than assumed away (amendment A3):
  * The sign-flip reference distribution is exact for the null that each paired
    difference is SYMMETRIC about zero. Symmetry implies mean zero; mean zero
    does not imply symmetry. This is not a distribution-free exact test of the
    mean, and skewness is reported per band so the assumption is auditable.
  * For the one-sided composite nulls the boundary symmetric case is used as the
    reference, which is the standard construction and is valid under the usual
    stochastic-ordering monotonicity.
  * The label permutation for Delta is exact under exchangeability of the two
    bands' differences, i.e. H0: F_low = F_high. That is a genuinely
    distribution-free statement, and a different null from the sign-flip one.

UNCERTAINTY. Each band supplies 20 real differences. Every power figure is a
plug-in estimate; resampling 20 points up to 65 redistributes information, it
does not create any. So power is reported as a nested-resampling band: draw a
pseudo-population of 20 by resampling the observed 20, estimate power from it,
repeat, and report the median with the 10th-90th percentile spread. The decision
rule reads the 10th percentile, never the point estimate.

HELD-OUT. Reads frozen development documents only. Every instance seed is
asserted below the held-out floor; the script aborts if one is not.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from math import comb
from pathlib import Path

import numpy as np

SEED = 20260827
ALPHA = 0.05
HELD_OUT_SEED_FLOOR = 100_000

DEV_SOURCES = (
    "results/dev_runs/budget_dev/*.json",
    "results/dev_runs/budget_dev_arch_diag/*.json",
)
CAP = 64000
C_LOW, C_HIGH = "c1_react", "c3_mas"
# Band label in the run documents -> role in the D (structural density) ladder.
LOW_BAND, MID_BAND, HIGH_BAND = "easy", "medium", "hard"

SCALES = (1.0, 0.75, 0.5)
SIZES = (50, 60, 65)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_deltas(root: Path) -> dict[str, np.ndarray]:
    """band -> array of delta = satisfaction(C3) - satisfaction(C1), one per instance."""
    table: dict[tuple[str, str], dict[str, float]] = {}
    for pattern in DEV_SOURCES:
        for path in glob.glob(str(root / pattern)):
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            rr = doc["run_result"]
            if rr.get("cap") != CAP:
                continue
            instance_id = rr["instance_id"]
            match = re.search(r"-s(\d+)$", instance_id)
            if match and int(match.group(1)) >= HELD_OUT_SEED_FLOOR:
                raise SystemExit(
                    f"REFUSING TO RUN: {instance_id} carries a seed at or above the "
                    f"held-out floor {HELD_OUT_SEED_FLOOR}. This analysis is registered "
                    "as development-only."
                )
            table.setdefault((instance_id, rr["level"]), {})[rr["condition"]] = (
                rr.get("score") or {})["satisfaction"]

    out: dict[str, list[float]] = {}
    for (_, band), by_condition in sorted(table.items()):
        if C_LOW in by_condition and C_HIGH in by_condition:
            out.setdefault(band, []).append(by_condition[C_HIGH] - by_condition[C_LOW])
    missing = [b for b in (LOW_BAND, MID_BAND, HIGH_BAND) if b not in out]
    if missing:
        raise SystemExit(f"no paired development data for band(s): {missing}")
    return {b: np.asarray(v, dtype=float) for b, v in out.items()}


def describe(d: np.ndarray) -> dict:
    """Location, spread, tie mass and skewness. Skewness is the A3 diagnostic."""
    n = int(d.size)
    m = float(d.mean())
    centred = d - m
    m2 = float((centred ** 2).mean())
    m3 = float((centred ** 3).mean())
    return {
        "n": n,
        "mean": m,
        "sd": float(np.sqrt(m2)),
        "tie_rate": float((d == 0).mean()),
        "distinct_values": int(np.unique(d).size),
        "skewness": (m3 / m2 ** 1.5) if m2 > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Tests. Vectorised over R independent replicates at once.
# --------------------------------------------------------------------------- #
def signflip_p(samples: np.ndarray, side: str, b: int, rng: np.random.Generator) -> np.ndarray:
    """One-sided sign-flip permutation p-values on the mean. samples: (R, N) -> (R,)."""
    r, n = samples.shape
    obs = samples.mean(1)
    signs = rng.integers(0, 2, size=(r, b, n), dtype=np.int8).astype(np.float64) * 2.0 - 1.0
    perm = (samples[:, None, :] * signs).mean(2)
    if side == "lower":                       # reject for small means
        hits = (perm <= obs[:, None] + 1e-12).sum(1)
    else:                                     # reject for large means
        hits = (perm >= obs[:, None] - 1e-12).sum(1)
    return (hits + 1.0) / (b + 1.0)


def label_perm_p(low: np.ndarray, high: np.ndarray, b: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Two-sided label-permutation p for Delta = mean(high) - mean(low). -> (R,)."""
    r, n = low.shape
    pooled = np.concatenate([low, high], axis=1)          # (R, 2N)
    total = pooled.sum(1)
    obs = np.abs(high.mean(1) - low.mean(1))
    keys = rng.random((r, b, 2 * n))
    idx = np.argpartition(keys, n, axis=2)[:, :, :n]      # a uniform N-subset per (r, b)
    rows = np.arange(r)[:, None, None]
    sel = pooled[rows, idx].sum(2)                        # (R, B)
    stat = np.abs((2.0 * sel - total[:, None]) / n)
    hits = (stat >= obs[:, None] - 1e-12).sum(1)
    return (hits + 1.0) / (b + 1.0)


def _binom_tail(k: int, n: int, side: str) -> float:
    if n == 0:
        return 1.0
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    if side == "lower":
        return sum(pmf[: k + 1])
    if side == "upper":
        return sum(pmf[k:])
    return min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12))


def sign_p(samples: np.ndarray, side: str) -> np.ndarray:
    """SENSITIVITY ONLY. Exact binomial sign test; a different estimand (A1)."""
    pos = (samples > 0).sum(1)
    neg = (samples < 0).sum(1)
    cache: dict[tuple[int, int], float] = {}
    out = np.empty(samples.shape[0])
    for i, (p_, n_) in enumerate(zip(pos.tolist(), neg.tolist())):
        key = (p_, p_ + n_)
        if key not in cache:
            cache[key] = _binom_tail(p_, p_ + n_, side)
        out[i] = cache[key]
    return out


def holm2(p_a: np.ndarray, p_b: np.ndarray, alpha: float = ALPHA):
    """Holm over exactly two hypotheses. Returns (reject_a, reject_b)."""
    smaller = np.minimum(p_a, p_b)
    step1 = smaller <= alpha / 2.0            # if the smaller fails, neither is rejected
    return step1 & (p_a <= alpha), step1 & (p_b <= alpha)


# --------------------------------------------------------------------------- #
# One replicate batch: draw, test, decide
# --------------------------------------------------------------------------- #
def draw(pop: np.ndarray, r: int, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Resample the pseudo-population, then thin to the requested effect scale."""
    sample = rng.choice(pop, size=(r, n), replace=True)
    if scale < 1.0:
        sample = np.where(rng.random((r, n)) < scale, sample, 0.0)
    return sample


def evaluate(pop_low, pop_high, n, scale, r, b, rng) -> dict[str, float]:
    """Rejection rates over r replicates for the primary family and the sensitivity set."""
    low = draw(pop_low, r, n, scale, rng)
    high = draw(pop_high, r, n, scale, rng)

    p_low = signflip_p(low, "lower", b, rng)
    p_high = signflip_p(high, "upper", b, rng)
    p_iut = np.maximum(p_low, p_high)
    p_delta = label_perm_p(low, high, b, rng)

    iut_holm, delta_holm = holm2(p_iut, p_delta)

    s_iut = np.maximum(sign_p(low, "lower"), sign_p(high, "upper"))
    sens_iut_holm = holm2(s_iut, p_delta)[0]

    return {
        "iut_raw": float((p_iut <= ALPHA).mean()),
        "iut_holm": float(iut_holm.mean()),
        "delta_raw": float((p_delta <= ALPHA).mean()),
        "delta_holm": float(delta_holm.mean()),
        "both_holm": float((iut_holm & delta_holm).mean()),
        "sens_iut_raw": float((s_iut <= ALPHA).mean()),
        "sens_iut_holm": float(sens_iut_holm.mean()),
    }


def nested(pop_low, pop_high, n, scale, outer, r, b, rng) -> dict[str, dict[str, float]]:
    """Nested resampling: pseudo-population -> power, repeated. Reports median and 10-90."""
    collected: dict[str, list[float]] = {}
    for _ in range(outer):
        pl = rng.choice(pop_low, size=pop_low.size, replace=True)
        ph = rng.choice(pop_high, size=pop_high.size, replace=True)
        for key, value in evaluate(pl, ph, n, scale, r, b, rng).items():
            collected.setdefault(key, []).append(value)
    return {
        key: {
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }
        for key, v in collected.items()
    }


def type_one(pop_low, pop_high, n, outer, r, b, rng) -> dict[str, dict[str, float]]:
    """Level under the least-favourable nulls.

    IUT   : both bands symmetric about zero -- the boundary of both one-sided nulls.
    Delta : both samples drawn from one population -- exchangeability holds exactly.
    """
    pooled = np.concatenate([np.abs(pop_low), np.abs(pop_high)])
    collected: dict[str, list[float]] = {}
    for _ in range(outer):
        mags = rng.choice(pooled, size=pooled.size, replace=True)
        null_pop = mags * rng.choice([-1.0, 1.0], size=mags.size)
        for key, value in evaluate(null_pop, null_pop, n, 1.0, r, b, rng).items():
            collected.setdefault(key, []).append(value)
    return {
        key: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
        for key, v in collected.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Crossover / interaction power, dev data only.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/interaction_power"))
    ap.add_argument("--outer", type=int, default=60, help="nested-resampling draws")
    ap.add_argument("--reps", type=int, default=200, help="replicates per pseudo-population")
    ap.add_argument("--perms", type=int, default=199, help="permutations per test")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    deltas = load_deltas(args.root)
    obs_low, obs_mid, obs_high = deltas[LOW_BAND], deltas[MID_BAND], deltas[HIGH_BAND]

    # The assumed crossover alternative. The high band keeps its observed sign; the
    # low band is NEGATED, which is the whole construction -- without this the
    # alternative still has C3 ahead in both bands, the one-sided low test cannot
    # reject by construction, and IUT power is identically zero.
    alt_low, alt_high = -obs_low, obs_high

    report: dict = {
        "schema_version": "interaction_power/1.0",
        "plan": ("results/analysis/interaction_power/PRE_ANALYSIS_PLAN.md "
                 "(frozen 2026-08-27, amended 2026-08-27)"),
        "standing": (
            "Development data only; no held-out seed is read. Power for the CROSSOVER "
            "claim under an ASSUMED sign-reversal alternative -- the development data "
            "supplies noise structure, not direction. Primary estimand: the mean paired "
            "difference. Sign test is sensitivity on a different functional. The medium "
            "band is descriptive and determines nothing."
        ),
        "seed": SEED,
        "alpha": ALPHA,
        "cap": CAP,
        "resampling": {"outer": args.outer, "reps": args.reps, "perms": args.perms},
        "observed": {
            "low": describe(obs_low),
            "medium": describe(obs_mid),
            "high": describe(obs_high),
        },
        "type_one": {},
        "power": {},
    }

    print(f"paired development deltas at cap {CAP}  (seed {SEED})")
    print("  delta = satisfaction(C3) - satisfaction(C1); positive = hierarchy ahead\n")
    for name, d in (("low", obs_low), ("medium (descriptive)", obs_mid), ("high", obs_high)):
        d_ = describe(d)
        print(f"  {name:<21} n={d_['n']:<3} mean={d_['mean']:+.3f} sd={d_['sd']:.3f} "
              f"ties={d_['tie_rate']:.0%} skew={d_['skewness']:+.2f}")

    print(f"\ntype-I under the least-favourable nulls "
          f"(outer={args.outer}, reps={args.reps}, perms={args.perms})")
    for n in SIZES:
        t1 = type_one(alt_low, alt_high, n, args.outer, args.reps, args.perms, rng)
        report["type_one"][str(n)] = t1
        print(f"  N={n:<3} IUT raw={t1['iut_raw']['median']:.4f} "
              f"holm={t1['iut_holm']['median']:.4f}   "
              f"Delta raw={t1['delta_raw']['median']:.4f} "
              f"holm={t1['delta_holm']['median']:.4f}")

    print("\npower under the assumed crossover alternative "
          "(median [p10, p90] over nested resampling)")
    for scale in SCALES:
        report["power"][str(scale)] = {}
        print(f"\n  effect scale {scale:g}")
        for n in SIZES:
            res = nested(alt_low, alt_high, n, scale, args.outer, args.reps, args.perms, rng)
            report["power"][str(scale)][str(n)] = res
            i, d, si = res["iut_holm"], res["delta_holm"], res["sens_iut_holm"]
            print(f"    N={n:<3} IUT {i['median']:.3f} [{i['p10']:.3f}, {i['p90']:.3f}]"
                  f"  (raw {res['iut_raw']['median']:.3f})   "
                  f"Delta {d['median']:.3f} [{d['p10']:.3f}, {d['p90']:.3f}]"
                  f"  (raw {res['delta_raw']['median']:.3f})   "
                  f"sens-IUT {si['median']:.3f} [{si['p10']:.3f}, {si['p90']:.3f}]")

    # Registered decision rule: smallest N with p10 >= 0.80 on BOTH primary tests,
    # at effect scale 1.0, after Holm. The medium band plays no part (amendment A2).
    chosen = None
    for n in SIZES:
        cell = report["power"]["1.0"][str(n)]
        if cell["iut_holm"]["p10"] >= 0.80 and cell["delta_holm"]["p10"] >= 0.80:
            chosen = n
            break
    report["decision"] = {
        "rule": ("smallest N in {50,60,65} with 10th-percentile power >= 0.80 for BOTH "
                 "the IUT and Delta, at effect scale 1.0, after Holm across the two "
                 "primary tests"),
        "chosen_N": chosen,
        "satisfied": chosen is not None,
    }
    print("\ndecision rule (10th percentile, scale 1.0, after Holm, both primary tests)")
    print(f"  chosen N: {chosen if chosen else 'NONE of 50/60/65 satisfies the rule'}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
