# scripts/validate_factorial_synthetic.py
"""Synthetic validation of the permutation tests in analyse_heldout_factorial. No held-out data.

    uv run --with numpy --with pandas python -m scripts.validate_factorial_synthetic
    uv run --with numpy --with pandas python -m scripts.validate_factorial_synthetic --reps 500 --perms 1999

Run this BEFORE any held-out omnibus or pairwise step. It answers three questions:

1. Under a true null, is the rejection rate of each test close to 0.05?
2. Is an injected effect detected?
3. Is the repeated-measures structure kept? (Checked by construction: every test resamples
   whole task vectors; `permute_codes` is also checked to keep the level counts.)

Scenarios, all shaped like Block A (150 tasks, 3 levels x 50, 5 architectures x 4 caps):

  null_discrete   Satisfaction on the real discrete scale (meetings / optimum, many zeros
                  and ties). Architectures are exchangeable within task x cap and tasks do
                  not depend on their level. True nulls: Architecture, Complexity, and every
                  interaction. Budget has a real effect.
  null_additive   Continuous, additive task + architecture + budget effects. True nulls:
                  Complexity and every interaction. Architecture and Budget are real effects.
  effects         null_additive plus an Architecture x Budget interaction (C3 gains at the
                  two largest caps) and an Architecture x Budget x Complexity interaction
                  (a further gain at the largest cap in High only). Checks detection.

For the pairwise family, null_discrete also gives the per-test rejection rate and the
family-wise error rate after Holm over 40 tests.

A rejection rate is flagged when it lies more than 2.5 Monte-Carlo standard errors above
0.05. If a flag appears, do not change the test silently: report it first.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.analyse_heldout_factorial import (
    ALPHA, ARCHS, CAPS, LEVELS, OUT_DIR, TERMS_BLOCK_A, holm, labelperm_p, permute_codes,
    signflip_p, task_vectors,
)

N_PER_LEVEL = 50


def simulate(scenario: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`(Y[task, architecture, cap], level per task)` for one synthetic dataset."""
    n = N_PER_LEVEL * len(LEVELS)
    levels = np.repeat(np.array(LEVELS), N_PER_LEVEL)
    ka, kb = len(ARCHS), len(CAPS)
    if scenario == "null_discrete":
        optimum = rng.choice([3, 4], size=n, p=[0.64, 0.36])
        ability = rng.uniform(0.3, 1.0, size=n)
        reach = np.array([0.02, 0.25, 0.55, 0.75])  # P(non-empty plan) by cap
        y = np.zeros((n, ka, kb))
        for t in range(n):
            for b in range(kb):
                nonempty = rng.random(ka) < reach[b] * ability[t]
                meetings = rng.binomial(optimum[t], min(1.0, 0.4 + 0.5 * ability[t]), size=ka)
                y[t, :, b] = np.where(nonempty, np.maximum(meetings, 1), 0) / optimum[t]
        return y, levels
    task = rng.normal(0.4, 0.15, size=(n, 1, 1))
    arch = np.array([0.0, 0.03, 0.08, 0.05, 0.02])[None, :, None]
    budget = np.array([0.0, 0.1, 0.25, 0.35])[None, None, :]
    y = task + arch + budget + rng.normal(0.0, 0.2, size=(n, ka, kb))
    if scenario == "effects":
        y[:, 2, 2:] += 0.08
        y[levels == "High", 2, 3] += 0.10
    elif scenario != "null_additive":
        raise ValueError(scenario)
    return y, levels


TRUE_NULL = {
    "null_discrete": {"Architecture", "Complexity", "Architecture x Budget", "Architecture x Complexity",
                      "Budget x Complexity", "Architecture x Budget x Complexity"},
    "null_additive": {"Complexity", "Architecture x Budget", "Architecture x Complexity",
                      "Budget x Complexity", "Architecture x Budget x Complexity"},
    # The injected three-way gain lives in High only, so it also shifts the Complexity,
    # A x C and B x C means: no term is a true null in this scenario.
    "effects": set(),
}


def omnibus_pvalues(y: np.ndarray, levels: np.ndarray, perms: int, rng: np.random.Generator) -> dict[str, float]:
    out = {}
    for name, vec, kind in TERMS_BLOCK_A:
        z = task_vectors(y, vec)
        out[name] = (signflip_p(z, perms, rng) if kind == "within" else labelperm_p(z, levels, perms, rng))[1]
    return out


def pairwise_pvalues(y: np.ndarray, perms: int, rng: np.random.Generator) -> list[float]:
    ps = []
    for b in range(len(CAPS)):
        for i in range(len(ARCHS)):
            for j in range(i + 1, len(ARCHS)):
                ps.append(signflip_p((y[:, i, b] - y[:, j, b])[:, None], perms, rng)[1])
    return ps


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--perms", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "synthetic_validation")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # Structure check: label permutations keep the level counts of whole tasks.
    codes = np.repeat(np.arange(3), N_PER_LEVEL)
    perm = permute_codes(codes, 200, rng)
    assert all((np.bincount(row, minlength=3) == N_PER_LEVEL).all() for row in perm)

    report: dict[str, object] = {"reps": args.reps, "perms": args.perms, "seed": args.seed,
                                 "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "scenarios": {}}
    se = float(np.sqrt(ALPHA * (1 - ALPHA) / args.reps))  # plain float: json cannot encode numpy scalars
    flags = []
    for scenario in ("null_discrete", "null_additive", "effects"):
        rejections = {name: 0 for name, _, _ in TERMS_BLOCK_A}
        pair_reject, fwer = np.zeros(40), 0
        for _ in range(args.reps):
            y, levels = simulate(scenario, rng)
            for name, p in omnibus_pvalues(y, levels, args.perms, rng).items():
                rejections[name] += p <= ALPHA
            if scenario == "null_discrete":
                ps = pairwise_pvalues(y, args.perms, rng)
                pair_reject += np.array(ps) <= ALPHA
                fwer += any(reject for _, reject in holm(ps))
        rows = {}
        for name, count in rejections.items():
            rate = count / args.reps
            is_null = name in TRUE_NULL[scenario]
            flag = bool(is_null and rate > ALPHA + 2.5 * se)
            rows[name] = {"true_null": is_null, "rejection_rate": rate, "mc_se_at_0.05": se, "flag": flag}
            if flag:
                flags.append(f"{scenario}: {name} rejects {rate:.3f}")
            print(f"{scenario:14s} {name:38s} null={str(is_null):5s} rate={rate:.3f}{'  FLAG' if flag else ''}")
        entry: dict[str, object] = {"terms": rows}
        if scenario == "null_discrete":
            entry["pairwise_rejection_rate_mean"] = float(pair_reject.mean() / args.reps)
            entry["pairwise_rejection_rate_max"] = float(pair_reject.max() / args.reps)
            entry["pairwise_fwer_after_holm"] = fwer / args.reps
            print(f"pairwise null: mean rate {entry['pairwise_rejection_rate_mean']:.3f}, "
                  f"max {entry['pairwise_rejection_rate_max']:.3f}, FWER after Holm {entry['pairwise_fwer_after_holm']:.3f}")
            if entry["pairwise_fwer_after_holm"] > ALPHA + 2.5 * se:  # type: ignore[operator]
                flags.append("pairwise FWER after Holm above 0.05")
        report["scenarios"][scenario] = entry  # type: ignore[index]
    report["flags"] = flags
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synthetic_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FLAGS:" if flags else "no flags", *flags, sep="\n  ")
    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
