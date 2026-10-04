# scripts/analyse_heldout.py
"""The held-out confirmatory analysis, under the plan frozen 2026-08-27. CPU, no LLM.

    .venv-harness/bin/python -m scripts.analyse_heldout \
        --runs results/logs/heldout \
        --expected results/manifests/expected_runs__bands_held_out_n8.json \
        --expected results/manifests/expected_runs__heldout_n4.json \
        --expected results/manifests/expected_runs__heldout_n5.json \
        --expected results/manifests/expected_runs__heldout_n6.json \
        --subset results/manifests/subset__bands_held_out_n8__278ce0f2e8fe.json \
        --subset results/manifests/subset__heldout_n4__54f3f48d6232.json \
        --subset results/manifests/subset__heldout_n5__4b91dee07c34.json \
        --subset results/manifests/subset__heldout_n6__44418ababc22.json \
        --pool results/calibration/heldout_n8/candidates.csv \
        --out results/analysis/heldout

ENVIRONMENT. This runs in `.venv-harness` as it stands: numpy is already pinned in
`requirements/harness.lock.txt` (2.5.1, transitively), so no package is added and
the lock hash the HPC provenance record depends on is untouched. Nothing here
needs a GPU or a served model. In an environment without numpy, layer it
ephemerally -- `uv run --with numpy python -m scripts.analyse_heldout ...` -- which
is what `scripts/analyse_interaction_power.py` documents; that module is also where
the permutation machinery below is imported from rather than rewritten.

WHAT THIS IS. The registered analysis of `results/analysis/interaction_power/
PRE_ANALYSIS_PLAN.md` (frozen 2026-08-27) with amendments 1 (A1-A4) and 2 (B1-B3),
executed on the held-out runs. **This script introduces no hypothesis, no metric, no
test and no diagnostic that is not in that plan.** Where the plan is silent the
script reports rather than decides.

SIGN CONVENTION (plan section 1, generalised by amendment B2). Every contrast is
written

    delta = satisfaction(later) - satisfaction(earlier)

where `earlier` and `later` are fixed per contrast. For the five confirmatory
contrasts the orientation is the REGISTERED one and is listed explicitly here and
in `CONFIRMATORY` below, verbatim from amendment B2:

    C1->C2   delta = sat(C2) - sat(C1)
    C2->C4   delta = sat(C4) - sat(C2)
    C4->C3   delta = sat(C3) - sat(C4)
    C1->C5   delta = sat(C5) - sat(C1)
    C5->C3   delta = sat(C3) - sat(C5)

The registered expectation, declared uniformly for all five, is `delta_low < 0`
and `delta_high > 0`.

THE CONFIRMATORY FAMILY (amendment B2). Those five contrasts, each carrying exactly
two pre-declared tests at the primary cap 64000:

  crossover     intersection-union: delta_low < 0 AND delta_high > 0, one-sided
                sign-flip permutation on the mean for each component,
                p_IUT = max(p_low, p_high)
  interaction   Delta = delta_high - delta_low, two-sided label permutation

Ten p-values, ONE Holm correction across all ten. That is the whole confirmatory
analysis. The other three caps, the other five pairwise contrasts, the medium band
and Block B carry effect estimates and intervals and no confirmatory claim.

TESTS AND THEIR STANDING (amendment A1).
  primary       sign-flip permutation on the mean paired difference -- sole test of
                the primary estimand
  sensitivity   exact binomial sign test; a DIFFERENT functional, P(delta > 0). It
                may not promote, demote or replace a primary result
  descriptive   percentile bootstrap interval. Excluded as primary on the evidence
                of `results/analysis/empirical_power/`: it exceeded nominal 0.05 in
                six of seven cells, worst where ties were heaviest

PRE-DECLARED DIAGNOSTICS, and only these. Per-band skewness of the paired
differences (amendment A3, because the sign-flip null is symmetry about zero and
not merely a zero mean), and the `H` sensitivity of plan section 7 -- the primary
IUT and Delta repeated within `H = 0` and `H >= 1`, as sensitivity that cannot
promote or demote the primary result.

WHAT THIS SCRIPT REFUSES TO DO. It will not analyse a sweep whose completeness
audit does not pass, it will not analyse runs whose seeds lie outside the held-out
range, and it will not silently drop a pre-declared diagnostic: a missing `H` pool
is an error, not an omission.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Permutation / sign-flip machinery and the descriptive interval come from the
# modules that already carry them. Nothing statistical is re-implemented here
# except the two pieces the plan needs at a size those modules do not cover:
# Holm over ten hypotheses (`holm2` covers exactly two) and a label permutation
# for unequal group sizes (`label_perm_p` requires equal ones), which is used only
# by the H-stratum sensitivity.
from scripts.analyse_formal_pilot import paired_bootstrap_ci  # noqa: E402
from scripts.analyse_interaction_power import (  # noqa: E402
    ALPHA,
    SEED,
    describe,
    label_perm_p,
    sign_p,
    signflip_p,
)
from scripts.audit_expected_runs import audit, load_expected  # noqa: E402

SCHEMA_VERSION = "heldout_analysis/1.0"
PLAN = ("results/analysis/interaction_power/PRE_ANALYSIS_PLAN.md "
        "(frozen 2026-08-27; amendment 2026-08-27 A1-A4; amendment 2 2026-08-27 B1-B3)")

HELD_OUT_SEED_FLOOR = 100_000
PRIMARY_CAP = 64000
CAPS = (16000, 32000, 64000, 128000)

# A TECHNICAL ORIENTATION CONVENTION, and nothing more. Its only job is to give each
# of the ten unordered pairs one fixed direction, so that a contrast keeps a stable
# sign wherever it appears and the descriptive matrix never reports the same pair
# twice with opposite signs.
#
# It is NOT a hierarchy of the architectures, NOT an ordering by amount of structure
# and NOT a ranking. The plan registers no such ordering, and no reading of any
# result may rest on a condition's position in this tuple.
#
# The five confirmatory orientations are NOT derived from it. They are declared
# explicitly in `CONFIRMATORY` below, verbatim from amendment B2. This tuple is
# chosen to agree with those five so that one convention covers the whole matrix;
# `test_the_orientation_convention_agrees_with_the_registered_five` fails if it ever
# stops agreeing, and in that case the registered five win and this tuple changes.
ORIENTATION_ORDER = ("c1_react", "c2_verify_revise", "c5_best_of_3",
                     "c4_planner_critic", "c3_mas")
ORIENTATION_INDEX = {c: i for i, c in enumerate(ORIENTATION_ORDER)}
CONDITIONS = tuple(sorted(ORIENTATION_ORDER))

# Amendment B2, verbatim in the order the plan tabulates them. REGISTERED: these five
# orientations are the plan's, not a consequence of ORIENTATION_ORDER.
CONFIRMATORY: tuple[tuple[str, str, str], ...] = (
    ("C1->C2", "c1_react", "c2_verify_revise"),
    ("C2->C4", "c2_verify_revise", "c4_planner_critic"),
    ("C4->C3", "c4_planner_critic", "c3_mas"),
    ("C1->C5", "c1_react", "c5_best_of_3"),
    ("C5->C3", "c5_best_of_3", "c3_mas"),
)

# Band labels in the run documents -> the D (structural density) ladder of the plan.
# The manifest calls these a "compatibility alias for the bands".
LEVEL_TO_BAND = {"easy": "low", "medium": "medium", "hard": "high"}
LOW_BAND, MID_BAND, HIGH_BAND = "low", "medium", "high"

PERMUTATIONS = 99999
BOOTSTRAP_RESAMPLES = 10000

NULL_CROSSOVER_WORDING = (
    "Failure to establish crossover statistically is not evidence that no crossover "
    "exists."
)


class AnalysisRefused(RuntimeError):
    """The inputs do not support the registered analysis. Never worked around."""


# --------------------------------------------------------------------------- #
# Statistics that the imported modules do not cover at the size the plan needs.
# --------------------------------------------------------------------------- #
def holm(p_values: Sequence[float], alpha: float = ALPHA) -> list[dict[str, Any]]:
    """Holm-Bonferroni over m hypotheses. Reduces to `holm2` when m == 2.

    Returned in input order: the adjusted p-value (step-down maximum, capped at 1)
    and the rejection decision. Reporting the adjusted value rather than only the
    decision is what makes the cost of the correction visible.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return [{"p_raw": float(p_values[i]), "p_holm": float(adjusted[i]),
             "reject": bool(adjusted[i] <= alpha)} for i in range(m)]


def label_perm_unequal(low: Sequence[float], high: Sequence[float], b: int,
                       rng: np.random.Generator) -> float:
    """Two-sided label permutation for `Delta` when the groups differ in size.

    `label_perm_p` is the primary path and requires equal group sizes, which the
    frozen 50/50 bands satisfy. The H strata do not: splitting each band on
    `H = 0` against `H >= 1` leaves whatever sizes the pool happens to contain.
    This is the same construction -- pool the two groups, redraw the labels,
    compare |mean difference| -- generalised to unequal sizes, and it is used ONLY
    for that sensitivity.
    """
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    n_lo, n_hi = lo.size, hi.size
    if n_lo == 0 or n_hi == 0:
        return float("nan")
    pooled = np.concatenate([lo, hi])
    obs = abs(float(hi.mean()) - float(lo.mean()))
    keys = rng.random((b, n_lo + n_hi))
    idx = np.argsort(keys, axis=1)
    drawn = pooled[idx]
    stat = np.abs(drawn[:, n_lo:].mean(1) - drawn[:, :n_lo].mean(1))
    return float((int((stat >= obs - 1e-12).sum()) + 1.0) / (b + 1.0))


def perm_one_sided(deltas: Sequence[float], side: str, b: int,
                   rng: np.random.Generator) -> float:
    """One-sided sign-flip permutation p on the mean, for a single sample."""
    arr = np.asarray(deltas, dtype=float)[None, :]
    return float(signflip_p(arr, side, b, rng)[0])


def sign_one_sided(deltas: Sequence[float], side: str) -> float:
    """SENSITIVITY ONLY (amendment A1): exact binomial sign test, other functional."""
    arr = np.asarray(deltas, dtype=float)[None, :]
    return float(sign_p(arr, side)[0])


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #
def load_instances(subsets: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """instance_id -> its frozen structural facts, and which instances may be read.

    Block A is the `n = 8` density design and is the only block with D bands. Block B
    (`n = 4/5/6`) is descriptive by construction: `n` is a generator parameter and
    never a complexity level (amendment B2), so its `level` label is deliberately NOT
    carried into a band.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in subsets:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for level, rows in doc["instances"].items():
            for e in rows:
                iid = e["instance_id"]
                if iid in out:
                    raise AnalysisRefused(
                        f"{iid} appears in more than one subset manifest; the analysis "
                        "would then weight it twice")
                n_people = e["cell"]["n_people"]
                block = "A" if n_people == 8 else "B"
                out[iid] = {
                    "instance_id": iid,
                    "seed": e["seed"],
                    "n_people": n_people,
                    "block": block,
                    "level": level,
                    "band": LEVEL_TO_BAND[level] if block == "A" else None,
                    "optimum": e["optimum"],
                    "complexity_metric": e["complexity_metric"],
                    "tightness": e["cell"]["tightness"],
                    "overlap": e["cell"]["overlap"],
                    "travel_structure": e["cell"]["travel_structure"],
                    "subset": path.name,
                }
    if not out:
        raise AnalysisRefused("no instances in the subset manifests")
    return out


def load_higher_order_gap(pools: Sequence[Path]) -> dict[str, int]:
    """instance_id -> higher_order_gap_H, from the frozen structural pools.

    Structural, computed by the oracle before any agent ran. It is read here because
    plan section 7 pre-declares a sensitivity stratified on it.
    """
    out: dict[str, int] = {}
    for path in pools:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out[row["instance_id"]] = int(row["higher_order_gap_H"])
    return out


def load_records(runs_dir: Path, instances: dict[str, dict[str, Any]]
                 ) -> list[dict[str, Any]]:
    """One row per run document, joined to its frozen structural facts."""
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr = doc["run_result"]
        info = instances.get(rr["instance_id"])
        if info is None or rr["condition"] not in CONDITIONS:
            continue
        score = rr["score"]
        meta = doc.get("pilot_sweep") or {}
        rows.append({
            "run_id": rr["run_id"],
            "instance_id": rr["instance_id"],
            "condition": rr["condition"],
            "cap": rr["cap"],
            "model": rr["model"],
            "block": info["block"],
            "band": info["band"],
            "n_people": info["n_people"],
            "optimum": info["optimum"],
            "satisfaction": score["satisfaction"],
            "tokens_total": (rr.get("tokens") or {}).get("total"),
            "git_commit": meta.get("git_commit"),
            "binning_hash": meta.get("binning_hash"),
        })
    return rows


# --------------------------------------------------------------------------- #
# The gate. Everything here runs BEFORE a single mean is computed.
# --------------------------------------------------------------------------- #
def check_inputs(rows: Sequence[dict[str, Any]], instances: dict[str, dict[str, Any]],
                 gaps: dict[str, int]) -> dict[str, Any]:
    """Compatibility of the inputs with the registered design, or refuse.

    The completeness audit proves every expected run is on disk. This proves the set
    on disk is the design the plan was written for: held-out seeds, one model, the
    full condition x cap matrix once per instance, and the pre-declared `H`
    stratification actually available for Block A.
    """
    stray_seeds = sorted(i["instance_id"] for i in instances.values()
                         if i["seed"] < HELD_OUT_SEED_FLOOR)
    if stray_seeds:
        raise AnalysisRefused(
            f"{len(stray_seeds)} instance(s) carry a seed below the held-out floor "
            f"{HELD_OUT_SEED_FLOOR}, e.g. {stray_seeds[:3]}. This analysis is "
            "registered as the held-out confirmatory analysis and may not read "
            "development seeds.")

    models = sorted({r["model"] for r in rows})
    if len(models) != 1:
        raise AnalysisRefused(f"runs come from more than one model: {models}")

    cells = Counter((r["instance_id"], r["condition"], r["cap"]) for r in rows)
    duplicated = [k for k, n in cells.items() if n > 1]
    if duplicated:
        raise AnalysisRefused(
            f"{len(duplicated)} (instance, condition, cap) cell(s) appear more than "
            f"once, e.g. {duplicated[:3]}")

    expected = len(instances) * len(CONDITIONS) * len(CAPS)
    if len(rows) != expected:
        raise AnalysisRefused(
            f"{len(rows)} run records for {len(instances)} instances x "
            f"{len(CONDITIONS)} conditions x {len(CAPS)} caps; expected {expected}")

    caps_seen = sorted({r["cap"] for r in rows})
    if tuple(caps_seen) != CAPS:
        raise AnalysisRefused(f"caps on disk {caps_seen}, registered {list(CAPS)}")

    block_a = [i for i in instances.values() if i["block"] == "A"]
    missing_h = sorted(i["instance_id"] for i in block_a
                       if i["instance_id"] not in gaps)
    if missing_h:
        raise AnalysisRefused(
            f"{len(missing_h)} Block A instance(s) have no higher_order_gap_H in the "
            f"pools supplied, e.g. {missing_h[:3]}. Plan section 7 pre-declares an H "
            "sensitivity; a missing pool is an error, not an omission.")

    band_counts = Counter(i["band"] for i in block_a)
    if set(band_counts) != {LOW_BAND, MID_BAND, HIGH_BAND}:
        raise AnalysisRefused(f"Block A bands present: {sorted(band_counts)}; "
                              f"expected {[LOW_BAND, MID_BAND, HIGH_BAND]}")

    # Plan section on the satisfaction rate: optimum == 0 is excluded from the primary
    # analysis and reported separately. Held-out is matched at O in {3, 4}, so this is
    # a guard that reports rather than a routine step.
    zero_optimum = sorted(i["instance_id"] for i in instances.values()
                          if i["optimum"] == 0)

    return {
        "model": models[0],
        "n_instances": len(instances),
        "n_runs": len(rows),
        "caps": list(CAPS),
        "block_a_band_counts": {b: band_counts[b] for b in
                                (LOW_BAND, MID_BAND, HIGH_BAND)},
        "block_b_size_counts": dict(sorted(Counter(
            i["n_people"] for i in instances.values() if i["block"] == "B").items())),
        "excluded_optimum_zero": zero_optimum,
        "git_commits": sorted({r["git_commit"] for r in rows if r["git_commit"]}),
        "binning_hashes": sorted({r["binning_hash"] for r in rows
                                  if r["binning_hash"]}),
    }


# --------------------------------------------------------------------------- #
# Paired differences.
# --------------------------------------------------------------------------- #
def orient(cond_a: str, cond_b: str) -> tuple[str, str]:
    """Return the pair in its conventional direction, so its sign is stable.

    Bookkeeping only (see ORIENTATION_ORDER), never a statement about the
    architectures. The five confirmatory contrasts carry their registered
    orientation from `CONFIRMATORY`, not from this function."""
    return ((cond_a, cond_b)
            if ORIENTATION_INDEX[cond_a] < ORIENTATION_INDEX[cond_b]
            else (cond_b, cond_a))


def contrast_label(earlier: str, later: str) -> str:
    short = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
             "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
    return f"{short[earlier]}->{short[later]}"


def paired_deltas(rows: Iterable[dict[str, Any]], earlier: str, later: str, *,
                  cap: int, keep=None) -> list[tuple[str, float]]:
    """delta = satisfaction(later) - satisfaction(earlier), paired within instance.

    The key is the instance at ONE cap. Pooling caps under an instance key would let a
    later record overwrite an earlier one and still look like a complete pairing.
    """
    by_instance: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["cap"] != cap or r["condition"] not in (earlier, later):
            continue
        if keep is not None and not keep(r):
            continue
        by_instance[r["instance_id"]][r["condition"]] = r["satisfaction"]
    out: list[tuple[str, float]] = []
    for iid in sorted(by_instance):
        pair = by_instance[iid]
        if earlier in pair and later in pair:
            out.append((iid, pair[later] - pair[earlier]))
    return out


def summarise_deltas(deltas: Sequence[float], *,
                     resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Effect estimate, descriptive interval, counts, and the A3 skewness diagnostic."""
    if not deltas:
        return {"n_pairs": 0, "mean_delta": None}
    arr = np.asarray(deltas, dtype=float)
    ci = paired_bootstrap_ci(list(deltas), resamples=resamples, seed=SEED)
    described = describe(arr)
    return {
        "n_pairs": int(arr.size),
        "mean_delta": round(float(arr.mean()), 4),
        "sd": round(described["sd"], 4),
        "skewness": round(described["skewness"], 4),
        "tie_rate": round(described["tie_rate"], 4),
        "later_ahead": int((arr > 0).sum()),
        "earlier_ahead": int((arr < 0).sum()),
        "tied": int((arr == 0).sum()),
        "bootstrap_ci_descriptive": {"low": ci["ci_low"], "high": ci["ci_high"],
                                     "resamples": ci["resamples"]},
    }


# --------------------------------------------------------------------------- #
# The confirmatory family. Primary cap only, ten p-values, one Holm.
# --------------------------------------------------------------------------- #
def confirmatory_contrast(rows: Sequence[dict[str, Any]], earlier: str, later: str, *,
                          perms: int, rng: np.random.Generator,
                          boot: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """The two pre-declared tests for one contrast at the primary cap.

    crossover     IUT over the two one-sided sign-flip components
                  H0_low : delta_low  >= 0   rejected for a SMALL mean
                  H0_high: delta_high <= 0   rejected for a LARGE mean
    interaction   Delta = delta_high - delta_low, two-sided label permutation
    """
    block_a = [r for r in rows if r["block"] == "A"]
    low = [d for _, d in paired_deltas(block_a, earlier, later, cap=PRIMARY_CAP,
                                       keep=lambda r: r["band"] == LOW_BAND)]
    mid = [d for _, d in paired_deltas(block_a, earlier, later, cap=PRIMARY_CAP,
                                       keep=lambda r: r["band"] == MID_BAND)]
    high = [d for _, d in paired_deltas(block_a, earlier, later, cap=PRIMARY_CAP,
                                        keep=lambda r: r["band"] == HIGH_BAND)]
    if not low or not high:
        raise AnalysisRefused(
            f"{contrast_label(earlier, later)}: empty low or high band at the primary "
            f"cap {PRIMARY_CAP}; the confirmatory family cannot be computed")
    if len(low) != len(high):
        raise AnalysisRefused(
            f"{contrast_label(earlier, later)}: bands are {len(low)} and {len(high)} "
            "pairs. The registered label permutation for Delta is exact under "
            "exchangeability of two equally sized bands; unequal bands are a design "
            "change, not an analysis choice.")

    p_low = perm_one_sided(low, "lower", perms, rng)
    p_high = perm_one_sided(high, "upper", perms, rng)
    p_iut = max(p_low, p_high)
    p_delta = float(label_perm_p(np.asarray(low)[None, :], np.asarray(high)[None, :],
                                 perms, rng)[0])

    return {
        "contrast": contrast_label(earlier, later),
        "earlier": earlier,
        "later": later,
        "cap": PRIMARY_CAP,
        "bands": {"low": summarise_deltas(low, resamples=boot),
                  "medium": summarise_deltas(mid, resamples=boot),
                  "high": summarise_deltas(high, resamples=boot)},
        "crossover": {
            "p_low_one_sided": round(p_low, 5),
            "p_high_one_sided": round(p_high, 5),
            "p_iut": round(p_iut, 5),
            "components_directionally_consistent": bool(
                np.mean(low) < 0 < np.mean(high)),
        },
        "interaction": {
            "delta_statistic": round(float(np.mean(high) - np.mean(low)), 4),
            "p_two_sided": round(p_delta, 5),
        },
        "sensitivity_sign_test": {
            "standing": ("SENSITIVITY ONLY -- a different functional, P(delta > 0). "
                         "It may not promote, demote or replace a primary result."),
            "p_low_one_sided": round(sign_one_sided(low, "lower"), 5),
            "p_high_one_sided": round(sign_one_sided(high, "upper"), 5),
            "p_iut": round(max(sign_one_sided(low, "lower"),
                               sign_one_sided(high, "upper")), 5),
        },
        "_p_iut": p_iut,
        "_p_delta": p_delta,
    }


def confirmatory_family(rows: Sequence[dict[str, Any]], *, perms: int,
                        rng: np.random.Generator,
                        boot: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Five contrasts x two tests = ten p-values, Holm across all ten (amendment B2)."""
    results = [confirmatory_contrast(rows, earlier, later, perms=perms, rng=rng,
                                     boot=boot)
               for _, earlier, later in CONFIRMATORY]

    labels: list[str] = []
    p_values: list[float] = []
    for res in results:
        labels.append(f"{res['contrast']}|crossover_iut")
        p_values.append(res["_p_iut"])
        labels.append(f"{res['contrast']}|interaction")
        p_values.append(res["_p_delta"])

    decided = holm(p_values)
    by_label = dict(zip(labels, decided))
    for res in results:
        res["crossover"]["holm"] = by_label[f"{res['contrast']}|crossover_iut"]
        res["interaction"]["holm"] = by_label[f"{res['contrast']}|interaction"]
        # A crossover is established only if the IUT rejects; the direction is part of
        # the null, so a rejection with the components pointing the other way cannot
        # occur -- the flag is carried so the report can state it rather than imply it.
        res["crossover"]["established"] = bool(
            res["crossover"]["holm"]["reject"]
            and res["crossover"]["components_directionally_consistent"])
        del res["_p_iut"], res["_p_delta"]

    return {
        "standing": ("Confirmatory. Five architecture-motivated contrasts, each with "
                     "an interaction test and a crossover IUT, at the primary cap "
                     f"{PRIMARY_CAP}. Holm across all ten p-values, one family."),
        "alpha": ALPHA,
        "permutations": perms,
        "n_hypotheses": len(p_values),
        "family": [{"hypothesis": lab, **dec} for lab, dec in zip(labels, decided)],
        "contrasts": results,
        "any_crossover_established": any(r["crossover"]["established"]
                                         for r in results),
        "null_crossover_wording": NULL_CROSSOVER_WORDING,
    }


# --------------------------------------------------------------------------- #
# Pre-declared sensitivity: H strata (plan section 7).
# --------------------------------------------------------------------------- #
def h_sensitivity(rows: Sequence[dict[str, Any]], gaps: dict[str, int], *, perms: int,
                  rng: np.random.Generator,
                  boot: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Repeat the primary IUT and Delta within `H = 0` and `H >= 1`.

    Sensitivity only. It cannot promote or demote the primary result, and the strata
    are whatever sizes the frozen pool contains -- which is why Delta here uses the
    unequal-size label permutation rather than the primary path.
    """
    block_a = [r for r in rows if r["block"] == "A"]
    strata = {
        "H_eq_0": lambda r: gaps.get(r["instance_id"], -1) == 0,
        "H_ge_1": lambda r: gaps.get(r["instance_id"], -1) >= 1,
    }
    out: dict[str, Any] = {
        "standing": ("Pre-declared sensitivity (plan section 7). Declared as "
                     "sensitivity only: it cannot promote or demote the primary "
                     "result. Delta uses the unequal-size label permutation because "
                     "the strata are not balanced by design."),
        "strata": {},
    }
    for name, keep in strata.items():
        per_contrast = []
        for _, earlier, later in CONFIRMATORY:
            low = [d for _, d in paired_deltas(
                block_a, earlier, later, cap=PRIMARY_CAP,
                keep=lambda r, k=keep: r["band"] == LOW_BAND and k(r))]
            high = [d for _, d in paired_deltas(
                block_a, earlier, later, cap=PRIMARY_CAP,
                keep=lambda r, k=keep: r["band"] == HIGH_BAND and k(r))]
            entry: dict[str, Any] = {
                "contrast": contrast_label(earlier, later),
                "low": summarise_deltas(low, resamples=boot),
                "high": summarise_deltas(high, resamples=boot),
            }
            if low and high:
                p_low = perm_one_sided(low, "lower", perms, rng)
                p_high = perm_one_sided(high, "upper", perms, rng)
                entry["p_iut"] = round(max(p_low, p_high), 5)
                entry["p_interaction"] = round(
                    label_perm_unequal(low, high, perms, rng), 5)
            else:
                entry["p_iut"] = None
                entry["p_interaction"] = None
                entry["note"] = "a stratum is empty in one band; no test is computed"
            per_contrast.append(entry)
        out["strata"][name] = per_contrast
    return out


# --------------------------------------------------------------------------- #
# Descriptive: the complete comparison matrix, every cap, every pair.
# --------------------------------------------------------------------------- #
def all_pairs() -> list[tuple[str, str]]:
    """The ten unordered pairs, each given one fixed direction by the convention."""
    pairs: list[tuple[str, str]] = []
    for i, earlier in enumerate(ORIENTATION_ORDER):
        for later in ORIENTATION_ORDER[i + 1:]:
            pairs.append((earlier, later))
    return pairs


def comparison_matrix(rows: Sequence[dict[str, Any]], *,
                      boot: int = BOOTSTRAP_RESAMPLES) -> list[dict[str, Any]]:
    """Effect estimates and intervals for every pair, band/size and cap. Descriptive.

    Nothing here carries a confirmatory claim: the ten tests of the confirmatory
    family live at the primary cap in `confirmatory_family` and nowhere else.
    """
    confirmatory_pairs = {(e, l) for _, e, l in CONFIRMATORY}
    out: list[dict[str, Any]] = []
    for earlier, later in all_pairs():
        for cap in CAPS:
            for band in (LOW_BAND, MID_BAND, HIGH_BAND):
                deltas = [d for _, d in paired_deltas(
                    [r for r in rows if r["block"] == "A"], earlier, later, cap=cap,
                    keep=lambda r, b=band: r["band"] == b)]
                out.append({
                    "block": "A", "stratum": f"band={band}", "cap": cap,
                    "contrast": contrast_label(earlier, later),
                    "in_confirmatory_family": (earlier, later) in confirmatory_pairs,
                    "is_primary_cap": cap == PRIMARY_CAP,
                    **summarise_deltas(deltas, resamples=boot),
                })
            for n_people in sorted({r["n_people"] for r in rows
                                    if r["block"] == "B"}):
                deltas = [d for _, d in paired_deltas(
                    [r for r in rows if r["block"] == "B"], earlier, later, cap=cap,
                    keep=lambda r, n=n_people: r["n_people"] == n)]
                out.append({
                    "block": "B", "stratum": f"n={n_people}", "cap": cap,
                    "contrast": contrast_label(earlier, later),
                    "in_confirmatory_family": False,
                    "is_primary_cap": cap == PRIMARY_CAP,
                    **summarise_deltas(deltas, resamples=boot),
                })
    return out


def condition_means(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean satisfaction per condition x stratum x cap. Description, not a contrast."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        stratum = (f"band={r['band']}" if r["block"] == "A"
                   else f"n={r['n_people']}")
        acc[f"{r['block']}|{stratum}|{r['cap']}|{r['condition']}"].append(
            r["satisfaction"])
    return {k: {"n": len(v), "mean_satisfaction": round(sum(v) / len(v), 4)}
            for k, v in sorted(acc.items())}


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def build(rows: Sequence[dict[str, Any]], instances: dict[str, dict[str, Any]],
          gaps: dict[str, int], audit_report: dict[str, Any], integrity: dict[str, Any],
          *, perms: int, boot: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": PLAN,
        "standing": (
            "Held-out confirmatory analysis. The confirmatory family is five "
            "architecture-motivated contrasts at the primary cap 64000, each with an "
            "interaction test and a crossover IUT, Holm across all ten p-values. "
            "Everything else in this document -- the other three caps, the other five "
            "pairwise contrasts, the medium band, Block B, the sign test and the "
            "bootstrap intervals -- is descriptive or sensitivity and carries no "
            "confirmatory claim."),
        "sign_convention": (
            "delta = satisfaction(later) - satisfaction(earlier). The five "
            "confirmatory orientations are the REGISTERED ones (amendment B2) and "
            "are listed under `confirmatory_orientations`. The direction of the "
            "remaining five pairs follows a technical convention whose only purpose "
            "is to give each pair one stable sign in the descriptive matrix; it "
            "encodes no hierarchy and no ranking of the architectures. Registered "
            "expectation for the five: delta_low < 0 and delta_high > 0."),
        "confirmatory_orientations": {
            label: f"delta = sat({later}) - sat({earlier})"
            for label, earlier, later in CONFIRMATORY},
        "orientation_convention": {
            "order": list(ORIENTATION_ORDER),
            "standing": ("Bookkeeping only: it fixes the direction of each unordered "
                         "pair so a contrast keeps one sign throughout. Not a "
                         "hierarchy, not a ranking, and never a basis for reading a "
                         "result."),
        },
        "seed": SEED,
        "alpha": ALPHA,
        "primary_cap": PRIMARY_CAP,
        "secondary_caps": [c for c in CAPS if c != PRIMARY_CAP],
        "resampling": {"permutations": perms,
                       "bootstrap_resamples": boot,
                       "bootstrap_standing": (
                           "descriptive interval only; excluded as primary on the "
                           "evidence of results/analysis/empirical_power/")},
        "completeness_audit": {
            "n_expected": audit_report["n_expected"],
            "n_verified": audit_report["n_verified"],
            "all_checks_passed": audit_report["all_checks_passed"],
        },
        "integrity": integrity,
        "condition_means": condition_means(rows),
        "confirmatory": confirmatory_family(rows, perms=perms, rng=rng, boot=boot),
        "h_sensitivity": h_sensitivity(rows, gaps, perms=perms, rng=rng, boot=boot),
        "comparison_matrix": comparison_matrix(rows, boot=boot),
        "medium_band_standing": (
            "Descriptive (amendment A2). It receives an effect estimate and an "
            "interval, no confirmatory test, and no membership of the Holm family."),
        "block_b_standing": (
            "Descriptive (amendment B2). `n` is a generator parameter and never a "
            "complexity level, so Block B is not a D band and forms no confirmatory "
            "contrast."),
    }


def render(summary: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    conf = summary["confirmatory"]

    add("# Held-out experiment — confirmatory analysis")
    add("")
    add(f"Plan: `{summary['plan']}`.")
    add("")
    add(summary["standing"])
    add("")
    add(f"Sign: {summary['sign_convention']}")
    add("")
    add("Registered orientations of the confirmatory five (amendment B2):")
    add("")
    for label, expression in summary["confirmatory_orientations"].items():
        add(f"- `{label}` — {expression}")
    add("")
    add(f"Every other pair is oriented by a bookkeeping convention "
        f"(`{' < '.join(summary['orientation_convention']['order'])}`) that exists "
        "only to keep one sign per pair. "
        f"{summary['orientation_convention']['standing']}")
    add("")
    integrity = summary["integrity"]
    add(f"{integrity['n_runs']} runs over {integrity['n_instances']} instances, "
        f"model `{integrity['model']}`. Completeness audit: "
        f"{summary['completeness_audit']['n_verified']}/"
        f"{summary['completeness_audit']['n_expected']} verified, passed = "
        f"{summary['completeness_audit']['all_checks_passed']}.")
    add("")

    add(f"## The confirmatory family — primary cap {summary['primary_cap']}")
    add("")
    add(f"Ten p-values, one Holm correction, α = {summary['alpha']}, "
        f"{conf['permutations']} permutations.")
    add("")
    add("| contrast | δ low | δ medium | δ high | p IUT | p IUT (Holm) | Δ | "
        "p interaction | p interaction (Holm) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for res in conf["contrasts"]:
        bands = res["bands"]
        add(f"| {res['contrast']} | {bands['low']['mean_delta']:+.3f} | "
            f"{bands['medium']['mean_delta']:+.3f} | {bands['high']['mean_delta']:+.3f} | "
            f"{res['crossover']['p_iut']:.4f} | "
            f"{res['crossover']['holm']['p_holm']:.4f} | "
            f"{res['interaction']['delta_statistic']:+.3f} | "
            f"{res['interaction']['p_two_sided']:.4f} | "
            f"{res['interaction']['holm']['p_holm']:.4f} |")
    add("")
    add("*The medium column is descriptive and enters no test and no Holm family.*")
    add("")

    add("### Crossover")
    add("")
    add("| contrast | δ low < 0 | δ high > 0 | IUT rejects after Holm | crossover |")
    add("|---|---|---|---|---|")
    for res in conf["contrasts"]:
        b = res["bands"]
        add(f"| {res['contrast']} | {b['low']['mean_delta'] < 0} | "
            f"{b['high']['mean_delta'] > 0} | "
            f"{res['crossover']['holm']['reject']} | "
            f"{'ESTABLISHED' if res['crossover']['established'] else 'not established'} |")
    add("")
    if not conf["any_crossover_established"]:
        add(f"**No crossover is established.** {conf['null_crossover_wording']} A "
            "non-rejection is consistent both with no crossover and with a real "
            "crossover this design could not resolve, and this report does not choose "
            "between those readings.")
        add("")

    add("### Per-band estimates, with the pre-declared skewness diagnostic")
    add("")
    add("| contrast | band | n | mean δ | 95% bootstrap CI (descriptive) | ties | skew |")
    add("|---|---|---|---|---|---|---|")
    for res in conf["contrasts"]:
        for band in ("low", "medium", "high"):
            s = res["bands"][band]
            ci = s["bootstrap_ci_descriptive"]
            add(f"| {res['contrast']} | {band} | {s['n_pairs']} | "
                f"{s['mean_delta']:+.3f} | [{ci['low']:+.3f}, {ci['high']:+.3f}] | "
                f"{s['tie_rate']:.0%} | {s['skewness']:+.2f} |")
    add("")
    add("The sign-flip permutation is exact under symmetry of the paired differences "
        "about zero, not merely a zero mean. Skewness is reported per band so that "
        "assumption is auditable rather than assumed away.")
    add("")

    add("### Sign test — sensitivity only")
    add("")
    add("A different functional, `P(δ > 0)`. It may not promote, demote or replace a "
        "primary result.")
    add("")
    add("| contrast | p IUT (sign test) | p IUT (primary permutation) |")
    add("|---|---|---|")
    for res in conf["contrasts"]:
        add(f"| {res['contrast']} | {res['sensitivity_sign_test']['p_iut']:.4f} | "
            f"{res['crossover']['p_iut']:.4f} |")
    add("")

    add("## H sensitivity")
    add("")
    add(summary["h_sensitivity"]["standing"])
    add("")
    add("| stratum | contrast | n low | n high | mean δ low | mean δ high | p IUT | "
        "p interaction |")
    add("|---|---|---|---|---|---|---|---|")
    for name, entries in summary["h_sensitivity"]["strata"].items():
        for e in entries:
            lo, hi = e["low"], e["high"]
            p_iut = "—" if e["p_iut"] is None else f"{e['p_iut']:.4f}"
            p_int = "—" if e["p_interaction"] is None else f"{e['p_interaction']:.4f}"
            lo_m = "—" if lo["mean_delta"] is None else f"{lo['mean_delta']:+.3f}"
            hi_m = "—" if hi["mean_delta"] is None else f"{hi['mean_delta']:+.3f}"
            add(f"| {name} | {e['contrast']} | {lo['n_pairs']} | {hi['n_pairs']} | "
                f"{lo_m} | {hi_m} | {p_iut} | {p_int} |")
    add("")

    add("## Complete comparison matrix — descriptive")
    add("")
    add("All ten pairwise contrasts, every band and size, every cap. The five given an "
        "architectural reading are marked; the rest complete the matrix, in the "
        "bookkeeping direction described above and with no hierarchy implied. No cell "
        "here carries a confirmatory claim, and the three non-primary caps are "
        "pre-declared secondary.")
    add("")
    add("The full matrix is written to `contrasts.csv`. The primary-cap Block A rows "
        "are reproduced here.")
    add("")
    add("| contrast | band | mean δ | 95% CI | n | family |")
    add("|---|---|---|---|---|---|")
    for row in summary["comparison_matrix"]:
        if row["block"] != "A" or row["cap"] != summary["primary_cap"]:
            continue
        ci = row["bootstrap_ci_descriptive"]
        add(f"| {row['contrast']} | {row['stratum'].split('=')[1]} | "
            f"{row['mean_delta']:+.3f} | [{ci['low']:+.3f}, {ci['high']:+.3f}] | "
            f"{row['n_pairs']} | "
            f"{'confirmatory' if row['in_confirmatory_family'] else 'descriptive'} |")
    add("")

    add("## Standing of the parts")
    add("")
    add(f"- **Medium band.** {summary['medium_band_standing']}")
    add(f"- **Block B (`n = 4/5/6`).** {summary['block_b_standing']}")
    add("- **Secondary caps.** 16000, 32000 and 128000 are run and reported, never "
        "pooled with the primary cap, and carry no confirmatory test.")
    add("- **Bootstrap.** Descriptive interval only.")
    add("")
    add("---")
    add("")
    add(f"*{NULL_CROSSOVER_WORDING} This applies to each of the five contrasts "
        "separately, and nothing here licenses the reverse claim.*")
    return "\n".join(L) + "\n"


def write_matrix_csv(path: Path, matrix: Sequence[dict[str, Any]]) -> None:
    fields = ["block", "stratum", "cap", "contrast", "in_confirmatory_family",
              "is_primary_cap", "n_pairs", "mean_delta", "sd", "skewness", "tie_rate",
              "later_ahead", "earlier_ahead", "tied", "ci_low", "ci_high"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in matrix:
            ci = row.get("bootstrap_ci_descriptive") or {}
            writer.writerow({**{k: row.get(k) for k in fields if k in row},
                             "ci_low": ci.get("low"), "ci_high": ci.get("high")})


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--expected", action="append", type=Path, required=True)
    ap.add_argument("--subset", action="append", type=Path, required=True)
    ap.add_argument("--pool", action="append", type=Path, required=True,
                    help="structural pool CSV carrying higher_order_gap_H (plan §7)")
    ap.add_argument("--out", type=Path, default=Path("results/analysis/heldout"))
    ap.add_argument("--perms", type=int, default=PERMUTATIONS)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES,
                    help="resamples for the DESCRIPTIVE interval; not a test")
    args = ap.parse_args(argv)

    # The completeness audit is a precondition, not a report section: an analysis over
    # a partial sweep would still produce readable means.
    audit_report = audit(load_expected(args.expected), args.runs)
    if not audit_report["all_checks_passed"]:
        raise AnalysisRefused(
            f"completeness audit failed: {audit_report['n_missing']} missing, "
            f"{audit_report['n_incompatible']} incompatible, "
            f"{audit_report['n_unexpected_files']} unexpected. Analysis may not begin.")

    instances = load_instances(args.subset)
    gaps = load_higher_order_gap(args.pool)
    rows = load_records(args.runs, instances)
    integrity = check_inputs(rows, instances, gaps)

    summary = build(rows, instances, gaps, audit_report, integrity, perms=args.perms,
                    boot=args.bootstrap)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(render(summary), encoding="utf-8")
    write_matrix_csv(args.out / "contrasts.csv", summary["comparison_matrix"])

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    conf = summary["confirmatory"]
    print(f"audit passed: {integrity['n_runs']} runs over "
          f"{integrity['n_instances']} instances\n")
    print(f"confirmatory family at cap {PRIMARY_CAP}: "
          f"{conf['n_hypotheses']} p-values, one Holm correction\n")
    print(f"{'contrast':<10} {'d_low':>8} {'d_high':>8} {'p_IUT':>9} "
          f"{'p_IUT_holm':>11} {'p_inter':>9} {'p_inter_holm':>13}")
    for res in conf["contrasts"]:
        add_low = res["bands"]["low"]["mean_delta"]
        add_high = res["bands"]["high"]["mean_delta"]
        print(f"{res['contrast']:<10} {add_low:>+8.3f} {add_high:>+8.3f} "
              f"{res['crossover']['p_iut']:>9.4f} "
              f"{res['crossover']['holm']['p_holm']:>11.4f} "
              f"{res['interaction']['p_two_sided']:>9.4f} "
              f"{res['interaction']['holm']['p_holm']:>13.4f}")
    print()
    if conf["any_crossover_established"]:
        established = [r["contrast"] for r in conf["contrasts"]
                       if r["crossover"]["established"]]
        print(f"crossover ESTABLISHED for: {', '.join(established)}")
    else:
        print("no crossover established. " + NULL_CROSSOVER_WORDING)
    print(f"\nwritten: {args.out}/summary.json, report.md, contrasts.csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
