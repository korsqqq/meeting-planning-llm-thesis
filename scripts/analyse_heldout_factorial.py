# scripts/analyse_heldout_factorial.py
"""Factorial analysis of the held-out runs across all budgets. CPU only, no LLM.

Every step is run by hand, one at a time, and its output is inspected before the next:

    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial audit
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial describe
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial distributions
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial omnibus --block A
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial omnibus --block A --exclude-16k
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial omnibus --block A --outcome valid_nonempty
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial omnibus --block B
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial pairwise
    uv run --with numpy --with pandas --with statsmodels python -m scripts.analyse_heldout_factorial mixed --block A
    uv run --with numpy --with pandas --with statsmodels python -m scripts.analyse_heldout_factorial mixed --block A --exclude-16k
    uv run --with numpy --with pandas --with statsmodels python -m scripts.analyse_heldout_factorial mixed --block B
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial cost
    uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial optimality
    uv run --with numpy --with pandas --with matplotlib==3.11.1 python -m scripts.analyse_heldout_factorial figures

matplotlib is pinned to 3.11.1, the version of the existing thesis figures: on the author's
Windows machine the application-control policy blocks a compiled DLL of the 3.11.2 wheel.

Before any held-out step, validate the permutation machinery on synthetic data with
`scripts/validate_factorial_synthetic.py`.

STANDING (THESIS_DECISIONS.md section 5, amendment 2026-09-14). The design is
Architecture x Budget x Complexity with caps 16k/32k/64k/128k, and the primary analysis
covers all of them. The specific procedures below were fixed after all runs were
complete and after the descriptive results and the registered cap-64000 analysis were
available. They are not pre-registered.

DATA. `results/exports/heldout/runs.csv`, 3960 rows. Nothing under `results/logs/`,
`results/exports/` or `results/analysis/heldout*/` is modified. Outputs go to
`results/analysis/heldout_factorial/<step>/`, each with a provenance JSON.

UNIT OF RESAMPLING. The task (instance). Block A: 150 tasks, each with a complete
5 architectures x 4 caps vector; complexity level (Low/Medium/High, 50 each) varies
between tasks. Block B: 48 tasks (n = 4, 5, 6; 16 each); size is described, never tested.

TESTS.
  Within-task terms (Architecture, Budget, Architecture x Budget)
      For each task, the orthonormal contrast vector z_i of the term is computed from its
      own 20 outcomes. Statistic Q = ||mean_i z_i||^2. Null distribution: flip the sign of
      each task's whole vector (task-level sign-flip). Q does not depend on the choice of
      orthonormal contrast basis.
  Terms with Complexity (Complexity, A x C, B x C, A x B x C)
      Same per-task vectors (the task mean for the Complexity main effect). Statistic
      T = sum_g n_g ||mean_g z - mean z||^2. Null distribution: permute the complexity labels
      between whole tasks; the observations of one task are never separated.
  Pairwise (Block A, pooled over complexity levels)
      Paired difference per task d_i = S_X - S_Y at one cap; two-sided task-level
      sign-flip on |mean d|. 10 pairs x 4 caps = 40 tests.
  p = (number of permuted statistics >= observed + 1) / (permutations + 1), 99 999
  permutations. The seed of each test is derived from one fixed seed and the test's
  name, so results do not depend on the order in which steps run.

MULTIPLICITY (fixed 2026-09-14). Holm within each of five families, never across them:
  blockA_satisfaction_primary              7 terms, all caps       -- the factorial analysis
  blockA_satisfaction_no16k_sensitivity    7 terms, 32k/64k/128k   -- sensitivity only; never
                                                                      replaces the primary result
  blockA_valid_nonempty_secondary          7 terms, all caps       -- secondary outcome
  blockB_satisfaction                      3 terms (A, B, A x B)   -- size is descriptive
  blockA_satisfaction_pairwise             10 pairs x 4 caps = 40  -- satisfaction only
Any other omnibus or pairwise combination is refused. The mixed model is a supporting
check, outside every Holm family.

EFFECT SIZES. rms_effect = square root of the mean squared contrast per degree of freedom
(for a main effect: the standard deviation of the level means), in satisfaction units.
It is a magnitude summary, not explained variance. Pairwise: mean paired difference,
counts better / equal / worse, paired Cohen's d (d_z) as secondary.

INTERVALS. Task bootstrap of the mean, 10 000 resamples, percentile 95%; stratified by
complexity level (Block A pooled cells) or by size (Block B pooled cells).

SIGN CONVENTION. Delta_{X-Y} = S_X - S_Y. Positive means the first-named architecture
scored higher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import warnings
import zlib
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO / "results/exports/heldout/runs.csv"
OUT_DIR = REPO / "results/analysis/heldout_factorial"
MANIFESTS = tuple(REPO / "results/manifests" / name for name in (
    "expected_runs__bands_held_out_n8.json",
    "expected_runs__heldout_n4.json",
    "expected_runs__heldout_n5.json",
    "expected_runs__heldout_n6.json",
))

SCHEMA_VERSION = "heldout_factorial/1.0"
PERMUTATIONS = 99_999
BOOTSTRAP_RESAMPLES = 10_000
PERM_SEED = 20260914
BOOT_SEED = 20260915
ALPHA = 0.05

ARCH_OF = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
           "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
CONDITION_OF = {v: k for k, v in ARCH_OF.items()}
ARCHS = ("C1", "C2", "C3", "C4", "C5")
ARCH_NAME = {"C1": "single ReAct", "C2": "single verify-revise",
             "C3": "hierarchical multi-agent system",
             "C4": "planner with fresh-context critic", "C5": "best-of-3"}
CAPS = (16000, 32000, 64000, 128000)
LEVEL_OF = {"low": "Low", "medium": "Medium", "high": "High"}
LEVELS = ("Low", "Medium", "High")
SIZES = (4, 5, 6)
TERMINATIONS = ("agent_finish", "aborted", "budget")

# (name, per-task vector, kind). kind "within": sign-flip; "between": label permutation.
TERMS_BLOCK_A: tuple[tuple[str, str, str], ...] = (
    ("Architecture", "A", "within"),
    ("Budget", "B", "within"),
    ("Complexity", "mean", "between"),
    ("Architecture x Budget", "AxB", "within"),
    ("Architecture x Complexity", "A", "between"),
    ("Budget x Complexity", "B", "between"),
    ("Architecture x Budget x Complexity", "AxB", "between"),
)
TERMS_BLOCK_B: tuple[tuple[str, str, str], ...] = TERMS_BLOCK_A[:2] + TERMS_BLOCK_A[3:4]

# The four omnibus Holm families, fixed 2026-09-14. (block, outcome, exclude_16k) ->
# (family name, role). Any other combination is refused: it would be an undefined family.
OMNIBUS_FAMILIES: dict[tuple[str, str, bool], tuple[str, str]] = {
    ("A", "satisfaction", False): ("blockA_satisfaction_primary", "primary"),
    ("A", "satisfaction", True): ("blockA_satisfaction_no16k_sensitivity", "sensitivity"),
    ("A", "valid_nonempty", False): ("blockA_valid_nonempty_secondary", "secondary"),
    ("B", "satisfaction", False): ("blockB_satisfaction", "block B"),
}
PAIRWISE_FAMILY = "blockA_satisfaction_pairwise"


def omnibus_family(block: str, outcome: str, exclude_16k: bool) -> tuple[str, str]:
    """Name and role of the omnibus Holm family; refuses combinations that are not defined."""
    try:
        return OMNIBUS_FAMILIES[(block, outcome, exclude_16k)]
    except KeyError:
        raise AnalysisRefused(
            f"no Holm family is defined for block={block}, outcome={outcome}, exclude_16k={exclude_16k}; "
            f"defined: {sorted(name for name, _ in OMNIBUS_FAMILIES.values())}") from None


class AnalysisRefused(RuntimeError):
    """The inputs do not have the structure the analysis requires. Never worked around."""


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #
def load_runs(path: Path = RUNS_CSV) -> pd.DataFrame:
    """Read runs.csv and derive the analysis columns. Raises on unknown labels."""
    df = pd.read_csv(path, dtype={"band": str}, keep_default_na=False)
    details = df["details_json"].map(json.loads)
    df["arch"] = df["condition"].map(ARCH_OF)
    if df["arch"].isna().any():
        raise AnalysisRefused(f"unknown conditions: {sorted(set(df.loc[df.arch.isna(), 'condition']))}")
    df["complexity"] = df["band"].map(LEVEL_OF)
    if (df.loc[df.block == "A", "complexity"].isna()).any():
        raise AnalysisRefused("Block A row without a Low/Medium/High label")
    df["latency_seconds"] = details.map(lambda d: float(d["latency_seconds"]))
    df["cap_utilisation"] = details.map(lambda d: float(d["cap_utilisation"]))
    df["optimal"] = details.map(lambda d: int(bool(d["score"]["optimality"])))
    df["valid_nonempty"] = ((df["valid"] == 1) & (df["n_meetings"] > 0)).astype(int)
    df["empty_plan"] = (df["n_meetings"] == 0).astype(int)
    df["gap_meetings"] = np.where(df["valid"] == 1, df["optimum"] - df["n_meetings"], df["optimum"])
    return df.drop(columns=["details_json"])


def caps_for(exclude_16k: bool) -> tuple[int, ...]:
    """The cap ladder of a primary analysis or of the sensitivity analysis without 16k."""
    return CAPS[1:] if exclude_16k else CAPS


def outcome_array(df: pd.DataFrame, block: str, outcome: str,
                  caps: Sequence[int] = CAPS) -> tuple[list[str], np.ndarray, np.ndarray]:
    """`(task ids, Y[task, architecture, cap], stratum per task)`.

    The stratum is the complexity level in Block A and the size n in Block B. Refuses a
    duplicate or missing (task, architecture, cap) cell rather than dropping the task.
    """
    sub = df[(df["block"] == block) & (df["cap"].isin(caps))]
    if sub.duplicated(["instance_id", "arch", "cap"]).any():
        raise AnalysisRefused(f"duplicate (task, architecture, cap) rows in Block {block}")
    tasks = sorted(sub["instance_id"].unique())
    t_idx = {t: i for i, t in enumerate(tasks)}
    y = np.full((len(tasks), len(ARCHS), len(caps)), np.nan)
    y[sub["instance_id"].map(t_idx).to_numpy(),
      sub["arch"].map({a: i for i, a in enumerate(ARCHS)}).to_numpy(),
      sub["cap"].map({c: i for i, c in enumerate(caps)}).to_numpy()] = sub[outcome].to_numpy(float)
    if np.isnan(y).any():
        raise AnalysisRefused(f"Block {block}: {int(np.isnan(y).sum())} missing cells for {outcome}")
    per_task = sub.drop_duplicates("instance_id").set_index("instance_id")
    col = "complexity" if block == "A" else "n_people"
    strata = per_task.loc[tasks, col].to_numpy()
    return tasks, y, strata


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def rng_for(seed: int, key: str) -> np.random.Generator:
    """A generator that depends only on the fixed seed and the test's name."""
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def orthonormal_contrasts(k: int) -> np.ndarray:
    """(k-1) x k normalised Helmert contrasts: orthonormal rows, each orthogonal to ones."""
    c = np.zeros((k - 1, k))
    for r in range(1, k):
        c[r - 1, :r] = 1.0
        c[r - 1, r] = -float(r)
        c[r - 1] /= np.linalg.norm(c[r - 1])
    return c


def task_vectors(y: np.ndarray, term: str) -> np.ndarray:
    """Per-task contrast vectors of one term from Y[task, architecture, cap]."""
    n, ka, kb = y.shape
    ca, cb = orthonormal_contrasts(ka), orthonormal_contrasts(kb)
    if term == "A":
        return y.mean(axis=2) @ ca.T
    if term == "B":
        return y.mean(axis=1) @ cb.T
    if term == "AxB":
        return np.einsum("pa,nab,qb->npq", ca, y, cb).reshape(n, -1)
    if term == "mean":
        return y.mean(axis=(1, 2))[:, None]
    raise ValueError(term)


def within_stat(z: np.ndarray) -> float:
    """Q = squared norm of the mean contrast vector."""
    return float((z.mean(axis=0) ** 2).sum())


def between_stat(z: np.ndarray, codes: np.ndarray, n_levels: int) -> float:
    """T = sum over levels of n_g times the squared distance of the level mean to the grand mean."""
    zbar = z.mean(axis=0)
    total = 0.0
    for g in range(n_levels):
        mask = codes == g
        total += mask.sum() * float(((z[mask].mean(axis=0) - zbar) ** 2).sum())
    return total


def _tolerance(obs: float) -> float:
    return 1e-12 * max(1.0, abs(obs))


def signflip_p(z: np.ndarray, perms: int, rng: np.random.Generator,
               chunk: int = 5000) -> tuple[float, float]:
    """Task-level sign-flip test of E[z] = 0. Returns `(observed Q, p)`."""
    z = np.asarray(z, dtype=float)
    n = z.shape[0]
    obs = within_stat(z)
    tol = _tolerance(obs)
    count, done = 0, 0
    while done < perms:
        b = min(chunk, perms - done)
        signs = rng.integers(0, 2, size=(b, n)) * 2.0 - 1.0
        stats = ((signs @ z / n) ** 2).sum(axis=1)
        count += int((stats >= obs - tol).sum())
        done += b
    return obs, (count + 1.0) / (perms + 1.0)


def permute_codes(codes: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    """`b` random permutations of the task labels; each row keeps the level counts."""
    order = np.argsort(rng.random((b, codes.size)), axis=1)
    return codes[order]


def labelperm_p(z: np.ndarray, labels: Sequence[Any], perms: int, rng: np.random.Generator,
                chunk: int = 2000) -> tuple[float, float]:
    """Permutation test that the per-task vectors have the same mean in every level.

    Whole tasks are reassigned to levels; the rows of `z` are never split.
    Returns `(observed T, p)`.
    """
    z = np.asarray(z, dtype=float)
    levels, codes = np.unique(np.asarray(labels), return_inverse=True)
    k = len(levels)
    counts = np.bincount(codes, minlength=k).astype(float)
    zbar = z.mean(axis=0)
    obs = between_stat(z, codes, k)
    tol = _tolerance(obs)
    count, done = 0, 0
    while done < perms:
        b = min(chunk, perms - done)
        perm = permute_codes(codes, b, rng)
        stats = np.zeros(b)
        for g in range(k):
            means = (perm == g).astype(float) @ z / counts[g]
            stats += counts[g] * ((means - zbar) ** 2).sum(axis=1)
        count += int((stats >= obs - tol).sum())
        done += b
    return obs, (count + 1.0) / (perms + 1.0)


def holm(p_values: Sequence[float], alpha: float = ALPHA) -> list[tuple[float, bool]]:
    """Holm step-down adjusted p-values and decisions, in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return [(adjusted[i], adjusted[i] <= alpha) for i in range(m)]


def bootstrap_mean_ci(x: Sequence[float], strata: Sequence[Any] | None, resamples: int,
                      rng: np.random.Generator) -> tuple[float, float]:
    """Percentile 95% interval of the mean; tasks resampled within each stratum."""
    x = np.asarray(x, dtype=float)
    groups = [x] if strata is None else [x[np.asarray(strata) == s] for s in pd.unique(np.asarray(strata))]
    total = np.zeros(resamples)
    for xs in groups:
        idx = rng.integers(0, xs.size, size=(resamples, xs.size))
        total += xs[idx].sum(axis=1)
    lo, hi = np.percentile(total / x.size, [2.5, 97.5])
    return float(lo), float(hi)


def describe_values(x: Sequence[float]) -> dict[str, float]:
    """N, mean, median, SD (ddof=1), variance, min, max, quartiles (linear), IQR."""
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(med), "sd": sd,
            "variance": sd ** 2, "min": float(x.min()), "max": float(x.max()),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


# --------------------------------------------------------------------------- #
# Provenance and output.
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_provenance(out: Path, step: str, args: argparse.Namespace, inputs: Sequence[Path]) -> None:
    """Versions, seeds, configuration and input hashes next to the step's outputs."""
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("statsmodels", "scipy", "matplotlib"):
        if mod in sys.modules:
            versions[mod] = getattr(sys.modules[mod], "__version__", "unknown")
    record = {
        "schema_version": SCHEMA_VERSION, "step": step,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "scripts", "src")),
        "versions": versions,
        "configuration": {k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(args).items() if k != "func"},
        "inputs": {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): _sha256(p)
                   for p in inputs},
        "standing": "procedures specified after all runs were complete; not pre-registered "
                    "(THESIS_DECISIONS.md section 5, amendment 2026-09-14)",
    }
    (out / f"provenance_{step}{_suffix(args)}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def _suffix(args: argparse.Namespace) -> str:
    parts = []
    if getattr(args, "block", None):
        parts.append(f"block{args.block}")
    if getattr(args, "outcome", None) and args.outcome != "satisfaction":
        parts.append(args.outcome)
    if getattr(args, "exclude_16k", False):
        parts.append("no16k")
    return ("__" + "__".join(parts)) if parts else ""


def step_dir(args: argparse.Namespace, step: str) -> Path:
    out = args.out / step
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cap_label(cap: int) -> str:
    return f"{cap // 1000}k"


# --------------------------------------------------------------------------- #
# Steps.
# --------------------------------------------------------------------------- #
def step_audit(args: argparse.Namespace) -> None:
    """Expected vs present runs per cell, termination outcomes, structural checks."""
    df = load_runs(args.runs)
    expected: list[dict[str, Any]] = []
    for path in MANIFESTS:
        expected.extend(json.loads(path.read_text(encoding="utf-8"))["runs"])
    exp_ids = [r["run_id"] for r in expected]
    present = set(df["run_id"])
    meta = df.drop_duplicates("instance_id").set_index("instance_id")[["block", "complexity", "n_people"]]
    exp = pd.DataFrame(expected)
    exp["arch"] = exp["condition"].map(ARCH_OF)
    exp = exp.join(meta, on="instance_id")

    keys = ["block", "arch", "cap", "complexity", "n_people"]
    exp_counts = exp.groupby(keys, dropna=False).size().rename("expected")
    grp = df.groupby(keys, dropna=False)
    cells = pd.concat([
        exp_counts,
        grp.size().rename("present"),
        *[grp["termination"].apply(lambda s, t=t: int((s == t).sum())).rename(t) for t in TERMINATIONS],
        grp.apply(lambda g: int(((g.termination == "aborted") & (g.n_meetings > 0)).sum()),
                  include_groups=False).rename("aborted_saved_plan_retained"),
        grp["empty_plan"].sum().rename("empty_plan"),
        grp["valid"].mean().rename("valid_rate"),
        grp["satisfaction"].apply(lambda s: int(s.isna().sum())).rename("missing_satisfaction"),
    ], axis=1).reset_index()
    cells["expected"] = cells["expected"].fillna(0).astype(int)
    cells["present"] = cells["present"].fillna(0).astype(int)
    cells["complete"] = cells["expected"] == cells["present"]

    runs_per_task = df.groupby("instance_id").size()
    checks = {
        "n_rows": int(len(df)), "n_tasks": int(df["instance_id"].nunique()),
        "n_expected_runs": len(exp_ids), "n_duplicate_expected_ids": len(exp_ids) - len(set(exp_ids)),
        "n_duplicate_run_ids": int(df["run_id"].duplicated().sum()),
        "missing_runs": sorted(set(exp_ids) - present),
        "unexpected_runs": sorted(present - set(exp_ids)),
        "tasks_without_20_runs": runs_per_task[runs_per_task != 20].to_dict(),
        "all_cells_complete": bool(cells["complete"].all()),
        "valid_rate_all_ones": bool((df["valid"] == 1).all()),
        "termination_totals": df["termination"].value_counts().to_dict(),
        "aborted_saved_plan_retained_total": int(((df.termination == "aborted") & (df.n_meetings > 0)).sum()),
        "aborted_with_positive_satisfaction": int(((df.termination == "aborted") & (df.satisfaction > 0)).sum()),
    }
    out = step_dir(args, "audit")
    cells.to_csv(out / "audit_cells.csv", index=False)
    (out / "audit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    write_provenance(out, "audit", args, [args.runs, *MANIFESTS])
    print(json.dumps({k: v for k, v in checks.items() if not isinstance(v, (list, dict)) or not v}, indent=2))


def _describe_rows(df: pd.DataFrame, outcome: str, boot: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(block: str, scope: str, x: np.ndarray, strata: np.ndarray | None, **labels: Any) -> None:
        key = f"describe|{outcome}|{block}|{scope}|" + "|".join(f"{k}={v}" for k, v in labels.items())
        lo, hi = bootstrap_mean_ci(x, strata, boot, rng_for(seed, key))
        rows.append({"block": block, "scope": scope, "outcome": outcome,
                     "architecture": labels.get("architecture", ""), "cap": labels.get("cap", ""),
                     "complexity": labels.get("complexity", ""), "size": labels.get("size", ""),
                     **describe_values(x), "ci_low": lo, "ci_high": hi})

    _, ya, sa = outcome_array(df, "A", outcome)
    for ai, arch in enumerate(ARCHS):
        for ci, cap in enumerate(CAPS):
            add("A", "architecture_cap", ya[:, ai, ci], sa, architecture=arch, cap=cap)
            for level in LEVELS:
                m = sa == level
                add("A", "architecture_cap_complexity", ya[m, ai, ci], None,
                    architecture=arch, cap=cap, complexity=level)
        add("A", "marginal_architecture", ya[:, ai, :].mean(axis=1), sa, architecture=arch)
    for ci, cap in enumerate(CAPS):
        add("A", "marginal_cap", ya[:, :, ci].mean(axis=1), sa, cap=cap)
    for level in LEVELS:
        m = sa == level
        add("A", "marginal_complexity", ya[m].mean(axis=(1, 2)), None, complexity=level)

    _, yb, sb = outcome_array(df, "B", outcome)
    for ai, arch in enumerate(ARCHS):
        for ci, cap in enumerate(CAPS):
            add("B", "architecture_cap", yb[:, ai, ci], sb, architecture=arch, cap=cap)
            for size in SIZES:
                m = sb == size
                add("B", "architecture_cap_size", yb[m, ai, ci], None, architecture=arch, cap=cap, size=size)
    return rows


def step_describe(args: argparse.Namespace) -> None:
    """Descriptive statistics and bootstrap intervals of the mean for every cell."""
    df = load_runs(args.runs)
    rows = []
    for outcome in ("satisfaction", "valid_nonempty", "optimal"):
        rows.extend(_describe_rows(df, outcome, args.boot, args.boot_seed))
    out = step_dir(args, "describe")
    pd.DataFrame(rows).to_csv(out / "descriptives.csv", index=False)
    write_provenance(out, "describe", args, [args.runs])
    print(f"wrote {len(rows)} rows to {out / 'descriptives.csv'}")


def _value_label(v: float) -> str:
    return str(Fraction(v).limit_denominator(12))


def step_distributions(args: argparse.Namespace) -> None:
    """Share of runs at each discrete satisfaction value, per cell."""
    df = load_runs(args.runs).copy()
    df["value"] = df["satisfaction"].round(6)
    rows = []
    specs = (("A", "architecture_cap", ["arch", "cap"]),
             ("A", "architecture_cap_complexity", ["arch", "cap", "complexity"]),
             ("B", "architecture_cap", ["arch", "cap"]),
             ("B", "architecture_cap_size", ["arch", "cap", "n_people"]))
    for block, scope, keys in specs:
        sub = df[df["block"] == block]
        counts = sub.groupby(keys + ["value"]).size().rename("count").reset_index()
        totals = sub.groupby(keys).size().rename("n").reset_index()
        counts = counts.merge(totals, on=keys)
        counts["share"] = counts["count"] / counts["n"]
        counts["value_label"] = counts["value"].map(_value_label)
        counts.insert(0, "scope", scope)
        counts.insert(0, "block", block)
        rows.append(counts.rename(columns={"arch": "architecture", "n_people": "size"}))
    out = step_dir(args, "distributions")
    pd.concat(rows, ignore_index=True).to_csv(out / "satisfaction_value_distribution.csv", index=False)
    write_provenance(out, "distributions", args, [args.runs])
    print(f"wrote {out / 'satisfaction_value_distribution.csv'}")


def omnibus_rows(df: pd.DataFrame, block: str, outcome: str, exclude_16k: bool,
                 perms: int, seed: int) -> list[dict[str, Any]]:
    """All factorial terms of one block as one named Holm family."""
    family, role = omnibus_family(block, outcome, exclude_16k)
    caps = caps_for(exclude_16k)
    _, y, strata = outcome_array(df, block, outcome, caps)
    n = y.shape[0]
    terms = TERMS_BLOCK_A if block == "A" else TERMS_BLOCK_B
    rows = []
    for name, vec, kind in terms:
        z = task_vectors(y, vec)
        d = z.shape[1]
        key = f"omnibus|{block}|{outcome}|{','.join(map(str, caps))}|{name}"
        if kind == "within":
            stat, p = signflip_p(z, perms, rng_for(seed, key))
            df_term, rms = d, float(np.sqrt(stat / d))
            test = "task-level sign-flip"
        else:
            k = len(pd.unique(strata))
            stat, p = labelperm_p(z, strata, perms, rng_for(seed, key))
            df_term = d * (k - 1)
            rms = float(np.sqrt(stat * k / (n * df_term)))
            test = "complexity-label permutation between tasks"
        rows.append({"block": block, "outcome": outcome, "caps": "/".join(_cap_label(c) for c in caps),
                     "term": name, "test": test, "df": df_term, "statistic": stat,
                     "rms_effect": rms, "n_tasks": n, "permutations": perms, "p_raw": p})
    for row, (p_adj, reject) in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"], row["reject_holm_0.05"] = p_adj, reject
        row["holm_family"], row["role"] = family, role
    return rows


def step_omnibus(args: argparse.Namespace) -> None:
    """Factorial terms, task-level permutation tests, Holm within one of the four named families."""
    family, role = omnibus_family(args.block, args.outcome, args.exclude_16k)
    df = load_runs(args.runs)
    rows = omnibus_rows(df, args.block, args.outcome, args.exclude_16k, args.perms, args.perm_seed)
    out = step_dir(args, "omnibus")
    pd.DataFrame(rows).to_csv(out / f"omnibus__{family}.csv", index=False)
    write_provenance(out, "omnibus", args, [args.runs])
    print(f"Holm family: {family} ({role})")
    print(pd.DataFrame(rows)[["term", "df", "rms_effect", "p_raw", "p_holm"]].to_string(index=False))


def pairwise_rows(df: pd.DataFrame, caps: Sequence[int], perms: int, perm_seed: int,
                  boot: int, boot_seed: int) -> list[dict[str, Any]]:
    """All 10 architecture pairs at each cap, Block A pooled over complexity; one Holm family."""
    _, y, strata = outcome_array(df, "A", "satisfaction", caps)
    rows = []
    for ci, cap in enumerate(caps):
        for i in range(len(ARCHS)):
            for j in range(i + 1, len(ARCHS)):
                d = y[:, i, ci] - y[:, j, ci]
                key = f"pairwise|A|{_cap_label(cap)}|{ARCHS[i]}-{ARCHS[j]}"
                _, p = signflip_p(d[:, None], perms, rng_for(perm_seed, key))
                lo, hi = bootstrap_mean_ci(d, strata, boot, rng_for(boot_seed, key))
                sd = float(d.std(ddof=1))
                rows.append({
                    "block": "A", "cap": cap, "pair": f"{ARCHS[i]}-{ARCHS[j]}",
                    "first": ARCHS[i], "second": ARCHS[j], "n_tasks": int(d.size),
                    "delta_mean": float(d.mean()), "ci_low": lo, "ci_high": hi,
                    "first_better": int((d > 0).sum()), "equal": int((d == 0).sum()),
                    "second_better": int((d < 0).sum()), "sd_diff": sd,
                    "cohens_dz": float(d.mean() / sd) if sd > 0 else float("nan"),
                    "permutations": perms, "p_raw": p,
                })
    for row, (p_adj, reject) in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"], row["reject_holm_0.05"] = p_adj, reject
        row["holm_family"] = PAIRWISE_FAMILY
    return rows


def step_pairwise(args: argparse.Namespace) -> None:
    """Satisfaction, Block A: 10 pairs x 4 caps = 40 tests, one Holm family. No other pairwise family."""
    df = load_runs(args.runs)
    rows = pairwise_rows(df, CAPS, args.perms, args.perm_seed, args.boot, args.boot_seed)
    if len(rows) != 40:
        raise AnalysisRefused(f"the pairwise family must hold 40 tests, got {len(rows)}")
    out = step_dir(args, "pairwise")
    pd.DataFrame(rows).to_csv(out / "pairwise.csv", index=False)
    write_provenance(out, "pairwise", args, [args.runs])
    print(pd.DataFrame(rows)[["cap", "pair", "delta_mean", "ci_low", "ci_high", "p_raw", "p_holm"]]
          .to_string(index=False))


MIXED_OPTIMIZERS = ("bfgs", "powell", "nm")


def fit_mixed(model: Any) -> tuple[Any, dict[str, Any], list[Any]]:
    """REML fit of one MixedLM, same model whichever optimizer succeeds.

    `lbfgs` is not used: in statsmodels 0.15 it raises `LinAlgError: Singular matrix` on the
    Block A design (full-rank fixed-effect matrix, 60 columns) even on synthetic data, while
    bfgs, powell and nm converge to identical estimates. The first optimizer in
    MIXED_OPTIMIZERS that returns is used; a second successful optimizer is fitted as a
    cross-check and the largest absolute difference in fixed effects is recorded.
    """
    attempts: list[dict[str, Any]] = []
    results: list[tuple[str, Any]] = []
    caught_all: list[Any] = []
    for method in MIXED_OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = model.fit(reml=True, method=[method])
            except (np.linalg.LinAlgError, ValueError) as exc:
                attempts.append({"method": method, "status": f"{type(exc).__name__}: {exc}"})
                continue
        attempts.append({"method": method, "status": "ok", "converged": bool(res.converged)})
        caught_all.extend(caught)
        results.append((method, res))
        if len(results) == 2:
            break
    if not results:
        raise AnalysisRefused(f"mixed model could not be fitted by any optimizer: {attempts}")
    used, res = results[0]
    log: dict[str, Any] = {"optimizer_used": used, "attempts": attempts}
    if len(results) == 2:
        other = results[1][1]
        log["cross_check_optimizer"] = results[1][0]
        log["max_abs_fixed_effect_difference"] = float((res.fe_params - other.fe_params).abs().max())
    return res, log, caught_all


def step_mixed(args: argparse.Namespace) -> None:
    """Supporting linear mixed model with a random intercept per task. Outside every Holm family."""
    import statsmodels
    import statsmodels.formula.api as smf
    from scipy import stats

    df = load_runs(args.runs)
    caps = caps_for(args.exclude_16k)
    sub = df[(df["block"] == args.block) & df["cap"].isin(caps)].copy()
    sub["budget"] = sub["cap"].map(_cap_label)
    rhs = "C(arch, Sum) * C(budget, Sum)" + (" * C(complexity, Sum)" if args.block == "A" else "")
    model = smf.mixedlm(f"satisfaction ~ {rhs}", sub, groups=sub["instance_id"])
    res, fit_log, caught = fit_mixed(model)
    fe = res.fe_params
    cov = res.cov_params().loc[fe.index, fe.index]
    rename = {"C(arch, Sum)": "Architecture", "C(budget, Sum)": "Budget", "C(complexity, Sum)": "Complexity"}
    # Group coefficient columns by term from their names ("C(arch, Sum)[S.C1]:C(budget, Sum)[S.128k]"),
    # which works whichever formula backend statsmodels uses.
    by_term: dict[str, list[str]] = {}
    for col in fe.index:
        if col == "Intercept":
            continue
        name = ":".join(re.sub(r"\[.*?\]$", "", part) for part in col.split(":"))
        by_term.setdefault(name, []).append(col)
    term_rows = []
    for name, cols in by_term.items():
        b = fe[cols].to_numpy()
        v = cov.loc[cols, cols].to_numpy()
        wald = float(b @ np.linalg.pinv(v) @ b)
        dof = int(np.linalg.matrix_rank(v))
        term_rows.append({"block": args.block, "caps": "/".join(_cap_label(c) for c in caps),
                          "term": " x ".join(rename.get(part, part) for part in name.split(":")),
                          "wald_chi2": wald, "df": dof, "p_wald": float(stats.chi2.sf(wald, dof))})
    out = step_dir(args, "mixed")
    sfx = _suffix(args)
    pd.DataFrame(term_rows).to_csv(out / f"mixed_terms{sfx}.csv", index=False)
    pd.DataFrame({"estimate": fe, "se": res.bse_fe}).to_csv(out / f"mixed_fixed_effects{sfx}.csv")
    (out / f"mixed_fit_log{sfx}.json").write_text(json.dumps(fit_log, indent=2), encoding="utf-8")
    (out / f"mixed_summary{sfx}.txt").write_text(
        res.summary().as_text() + "\n\nconverged: " + str(res.converged)
        + "\nfit: " + json.dumps(fit_log)
        + "\nwarnings:\n" + "\n".join(str(w.message) for w in caught)
        + f"\n\nstatsmodels {statsmodels.__version__}. Supporting check only: residuals of a bounded,"
          " discrete outcome are not normal; p-values here are not part of any Holm family.\n",
        encoding="utf-8")
    write_provenance(out, "mixed", args, [args.runs])
    print(pd.DataFrame(term_rows).to_string(index=False))
    print(f"converged={res.converged}; {len(caught)} warning(s); fit={json.dumps(fit_log)}")


def _cost_cells(df: pd.DataFrame) -> pd.DataFrame:
    specs = (("A", ["arch", "cap"]), ("A", ["arch", "cap", "complexity"]),
             ("B", ["arch", "cap"]), ("B", ["arch", "cap", "n_people"]))
    frames = []
    for block, keys in specs:
        g = df[df["block"] == block].groupby(keys)
        t = pd.DataFrame({
            "n": g.size(),
            "satisfaction_mean": g["satisfaction"].mean(),
            "tokens_used_mean": g["tokens_total"].mean(), "tokens_used_median": g["tokens_total"].median(),
            "tokens_used_sd": g["tokens_total"].std(ddof=1),
            "cap_utilisation_mean": g["cap_utilisation"].mean(),
            "calls_mean": g["n_calls"].mean(), "calls_median": g["n_calls"].median(),
            "model_call_wall_time_s_mean": g["latency_seconds"].mean(),
            "model_call_wall_time_s_median": g["latency_seconds"].median(),
            "model_call_wall_time_s_p25": g["latency_seconds"].quantile(0.25),
            "model_call_wall_time_s_p75": g["latency_seconds"].quantile(0.75),
        }).reset_index()
        t["satisfaction_per_1k_tokens"] = t["satisfaction_mean"] / t["tokens_used_mean"] * 1000.0
        t["satisfaction_per_call"] = t["satisfaction_mean"] / t["calls_mean"]
        t.insert(0, "scope", "_".join({"arch": "architecture", "n_people": "size"}.get(k, k) for k in keys))
        t.insert(0, "block", block)
        frames.append(t.rename(columns={"arch": "architecture", "n_people": "size"}))
    return pd.concat(frames, ignore_index=True)


def step_cost(args: argparse.Namespace) -> None:
    """Tokens actually used, calls, model-call wall time, and quality per token and per call."""
    df = load_runs(args.runs)
    cells = _cost_cells(df)
    out = step_dir(args, "cost")
    cells.to_csv(out / "cost.csv", index=False)
    write_provenance(out, "cost", args, [args.runs])
    print(f"wrote {out / 'cost.csv'} ({len(cells)} rows). Wall time = sum of model-call wall time per run.")


def step_optimality(args: argparse.Namespace) -> None:
    """Optimality, valid non-empty solution rate, empty plans and the gap to the optimum in meetings."""
    df = load_runs(args.runs)
    specs = (("A", ["arch", "cap"]), ("A", ["arch", "cap", "complexity"]),
             ("B", ["arch", "cap"]), ("B", ["arch", "cap", "n_people"]))
    frames = []
    for block, keys in specs:
        g = df[df["block"] == block].groupby(keys)
        t = pd.DataFrame({
            "n": g.size(), "satisfaction_mean": g["satisfaction"].mean(),
            "optimality_rate": g["optimal"].mean(), "valid_rate": g["valid"].mean(),
            "valid_nonempty_solution_rate": g["valid_nonempty"].mean(),
            "empty_plan_rate": g["empty_plan"].mean(),
            "gap_meetings_mean": g["gap_meetings"].mean(), "gap_meetings_median": g["gap_meetings"].median(),
        }).reset_index()
        t.insert(0, "scope", "_".join({"arch": "architecture", "n_people": "size"}.get(k, k) for k in keys))
        t.insert(0, "block", block)
        frames.append(t.rename(columns={"arch": "architecture", "n_people": "size"}))
    out = step_dir(args, "optimality")
    pd.concat(frames, ignore_index=True).to_csv(out / "optimality.csv", index=False)
    write_provenance(out, "optimality", args, [args.runs])
    print(f"wrote {out / 'optimality.csv'}")


def step_figures(args: argparse.Namespace) -> None:
    """Figures from the CSVs of the earlier steps. Fits and tests nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.plot_heldout_figures import COLOR, MARKER

    color = {a: COLOR[CONDITION_OF[a]] for a in ARCHS}
    marker = {a: MARKER[CONDITION_OF[a]] for a in ARCHS}
    desc_path = args.out / "describe/descriptives.csv"
    dist_path = args.out / "distributions/satisfaction_value_distribution.csv"
    pair_path = args.out / "pairwise/pairwise.csv"
    cost_path = args.out / "cost/cost.csv"
    for p in (desc_path, dist_path, pair_path, cost_path):
        if not p.exists():
            raise AnalysisRefused(f"run the step that writes {p} first")
    desc = pd.read_csv(desc_path, keep_default_na=False)
    desc = desc[desc["outcome"] == "satisfaction"]
    out = step_dir(args, "figures")
    xs = np.arange(len(CAPS))

    def save(fig: Any, name: str) -> None:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def lines(ax: Any, cell: pd.DataFrame) -> None:
        for k, arch in enumerate(ARCHS):
            c = cell[cell["architecture"] == arch].assign(cap=lambda t: t["cap"].astype(int)).sort_values("cap")
            off = (k - 2) * 0.06
            ax.errorbar(xs + off, c["mean"], yerr=[c["mean"] - c["ci_low"], c["ci_high"] - c["mean"]],
                        color=color[arch], marker=marker[arch], ms=4.5, lw=1.5, capsize=2, label=arch)
        ax.set_xticks(xs, [_cap_label(c) for c in CAPS])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#DDDDDD")

    # Main figure: satisfaction vs token budget cap, one panel per complexity level.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, level in zip(axes, LEVELS):
        lines(ax, desc[(desc["block"] == "A") & (desc["scope"] == "architecture_cap_complexity")
                       & (desc["complexity"] == level)])
        ax.set_title(f"{level} complexity", fontsize=10)
        ax.set_xlabel("token budget cap")
    axes[0].set_ylabel("mean satisfaction")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "main_satisfaction_vs_cap_by_complexity")

    # Block B: one panel per size.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, size in zip(axes, SIZES):
        lines(ax, desc[(desc["block"] == "B") & (desc["scope"] == "architecture_cap_size")
                       & (desc["size"].astype(str) == str(size))])
        ax.set_title(f"n = {size}", fontsize=10)
        ax.set_xlabel("token budget cap")
    axes[0].set_ylabel("mean satisfaction")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "blockB_satisfaction_vs_cap_by_size")

    # Cost-quality: satisfaction vs tokens actually used, one panel per cap (Block A pooled).
    cost = pd.read_csv(cost_path, keep_default_na=False)
    cost = cost[(cost["block"] == "A") & (cost["scope"] == "architecture_cap")]
    pooled = desc[(desc["block"] == "A") & (desc["scope"] == "architecture_cap")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, cap in zip(axes, CAPS):
        for arch in ARCHS:
            c = cost[(cost["architecture"] == arch) & (cost["cap"].astype(int) == cap)].iloc[0]
            s = pooled[(pooled["architecture"] == arch) & (pooled["cap"].astype(int) == cap)].iloc[0]
            ax.errorbar(c["tokens_used_mean"] / 1000, s["mean"],
                        yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                        color=color[arch], marker=marker[arch], ms=6, capsize=2, label=arch)
        ax.set_title(f"cap {_cap_label(cap)}", fontsize=10)
        ax.set_xlabel("tokens actually used (thousands)")
        ax.grid(color="#EEEEEE")
    axes[0].set_ylabel("mean satisfaction")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "cost_quality_satisfaction_vs_tokens")

    # Discrete-value distributions, Block A: rows = complexity, columns = cap, bars = architecture.
    dist = pd.read_csv(dist_path, keep_default_na=False)
    values = sorted(dist["value"].astype(float).unique())
    cmap = plt.get_cmap("Blues")
    vcol = {v: ("#E0E0E0" if v == 0 else cmap(0.25 + 0.75 * v)) for v in values}

    def stacked(ax: Any, cell: pd.DataFrame) -> None:
        bottom = np.zeros(len(ARCHS))
        for v in values:
            share = np.array([cell[(cell["architecture"] == a) & (cell["value"].astype(float) == v)]["share"].sum()
                              for a in ARCHS])
            ax.bar(np.arange(len(ARCHS)), share, bottom=bottom, color=vcol[v], edgecolor="white", lw=0.4,
                   label=_value_label(v))
            bottom += share
        ax.set_xticks(np.arange(len(ARCHS)), ARCHS, fontsize=8)
        ax.set_ylim(0, 1)

    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharey=True)
    a_cells = dist[(dist["block"] == "A") & (dist["scope"] == "architecture_cap_complexity")]
    for r, level in enumerate(LEVELS):
        for c, cap in enumerate(CAPS):
            stacked(axes[r, c], a_cells[(a_cells["complexity"] == level) & (a_cells["cap"].astype(int) == cap)])
            if r == 0:
                axes[r, c].set_title(f"cap {_cap_label(cap)}", fontsize=10)
        axes[r, 0].set_ylabel(f"{level}\nshare of runs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Anchored just right of the axes (default right edge 0.9), not at the figure edge, so the
    # tight bounding box does not keep a wide empty strip between the panels and the legend.
    fig.legend(handles[::-1], labels[::-1], title="satisfaction", frameon=False, fontsize=8,
               loc="center left", bbox_to_anchor=(0.905, 0.5))
    save(fig, "distribution_satisfaction_values_blockA")

    # 32k follow-up: distribution per architecture and paired differences C2 - X with CI.
    pair = pd.read_csv(pair_path)
    pair = pair[pair["cap"] == 32000]
    # wspace keeps the right panel's "C2 − X" tick labels clear of the left panel; the value
    # legend sits above the left panel instead of covering the tops of the C4 and C5 bars.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 3.9),
                                     gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.45})
    stacked(ax_l, dist[(dist["block"] == "A") & (dist["scope"] == "architecture_cap")
                       & (dist["cap"].astype(int) == 32000)])
    ax_l.set_ylabel("share of runs (Block A, cap 32k)")
    h, lab = ax_l.get_legend_handles_labels()
    ax_l.legend(h[::-1], lab[::-1], title="satisfaction", frameon=False, fontsize=7, ncol=len(values),
                loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
    others = [a for a in ARCHS if a != "C2"]
    for k, other in enumerate(others):
        row = pair[(pair["first"].isin(["C2", other])) & (pair["second"].isin(["C2", other]))].iloc[0]
        sign = 1.0 if row["first"] == "C2" else -1.0
        est = sign * row["delta_mean"]
        lo, hi = sorted((sign * row["ci_low"], sign * row["ci_high"]))
        ax_r.errorbar(est, k, xerr=[[est - lo], [hi - est]], fmt="o", color=color[other], capsize=3)
        # The inferential result is the Holm-adjusted p of the 40-test family, not the CI.
        p_holm = float(row["p_holm"])
        p_text = "$p_{Holm}$ < 0.001" if p_holm < 0.001 else f"$p_{{Holm}}$ = {p_holm:.3f}"
        ax_r.text(1.02, k, p_text, transform=ax_r.get_yaxis_transform(), va="center", ha="left",
                  fontsize=9, color="#333333")
    ax_r.axvline(0.0, color="#333333", lw=1.0)
    ax_r.set_yticks(range(len(others)), [f"C2 − {o}" for o in others])
    ax_r.invert_yaxis()
    ax_r.set_xlabel("mean paired difference in satisfaction (95% bootstrap CI)")
    save(fig, "followup_32k_c2")
    write_provenance(out, "figures", args, [desc_path, dist_path, pair_path, cost_path])
    print(f"wrote figures to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=RUNS_CSV)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    sub = parser.add_subparsers(dest="step", required=True)

    def add(name: str, func: Any, *, perms: bool = False, boot: bool = False,
            block: bool = False, outcome: bool = False, no16k: bool = False) -> None:
        p = sub.add_parser(name, help=func.__doc__.split("\n")[0])
        p.set_defaults(func=func)
        if perms:
            p.add_argument("--perms", type=int, default=PERMUTATIONS)
            p.add_argument("--perm-seed", type=int, default=PERM_SEED)
        if boot:
            p.add_argument("--boot", type=int, default=BOOTSTRAP_RESAMPLES)
            p.add_argument("--boot-seed", type=int, default=BOOT_SEED)
        if block:
            p.add_argument("--block", choices=("A", "B"), default="A")
        if outcome:
            p.add_argument("--outcome", choices=("satisfaction", "valid_nonempty"), default="satisfaction")
        if no16k:
            p.add_argument("--exclude-16k", action="store_true")

    add("audit", step_audit)
    add("describe", step_describe, boot=True)
    add("distributions", step_distributions)
    add("omnibus", step_omnibus, perms=True, block=True, outcome=True, no16k=True)
    add("pairwise", step_pairwise, perms=True, boot=True)
    add("mixed", step_mixed, block=True, no16k=True)
    add("cost", step_cost)
    add("optimality", step_optimality)
    add("figures", step_figures)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except AnalysisRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
