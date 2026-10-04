# scripts/build_lowcx_manifest.py
"""Selector for the lower-complexity extension: 12 instances at O = 3 per size. CPU, no LLM.

    python -m scripts.build_lowcx_manifest \
        --pool 4=results/calibration/lowcx_n4 \
        --pool 5=results/calibration/lowcx_n5 \
        --pool 6=results/calibration/lowcx_n6 \
        --out results/manifests

WHAT THIS IS, AND WHAT IT IS NOT. It cuts 12 instances at each of `n = 4, 5, 6`, all at
oracle optimum 3, under the diversity rules the `n = 8` design has used since the
2026-08-09 amendment. It is a **lower-complexity extension**. It does not replace, re-cut
or re-open the frozen `n = 8` design, and it never selects a new `n = 8` subsample: the
frozen 60-instance set stands as it is.

WHY `O = 3` AND ONLY `O = 3`. `satisfaction = achieved / O`, so an unmatched optimum makes
the metric's granularity differ between the groups being compared, on top of whatever
structural difference was intended. Fixing `O = 3` at every size makes the denominator
identical rather than merely comparable -- satisfaction takes the same four values
{0, 1/3, 2/3, 1} everywhere. The measured alternative, admitting `O = 4`, was rejected on
the pool: at `n = 4` the cell `O = 4` is `O = n`, and 5123 of its 5126 candidates have zero
conflicting pairs and zero higher-order gap, so half of an `O in {3,4}` sample at `n = 4`
would have sat in a cell that is degenerate on three dimensions at once.

WHAT SMALLER `n` IS BEING USED FOR, STATED PRECISELY. At a fixed optimum the higher-order
gap obeys `H = alpha_reachable - O <= n - O`, so a smaller size mechanically bounds the
interaction the pairwise density metric cannot see. That is the whole reason the sizes
differ here. **`n` remains a generator parameter and does not become a complexity metric**,
and the design does not read "n = 4 is Low, n = 6 is High". `n` and the attainable `H` are
tied together by that inequality, so the separate causal effect of either is not identified
by this design; the amendment records that as a limitation rather than working around it.

LEVEL LABELS. The frozen sweep runner requires every entry's level to be re-derivable from
its conflict count through the binning boundaries, so a level here is a statement about
**conflict density and never about size**. The boundaries are the frozen `n = 8` cut-points
expressed as densities and read back at each size: `easy` is `D <= 1/28` and `medium` is
`D <= 7/28`, which reproduces `b1_easy_max = 1` and `b2_medium_max = 7` exactly at `n = 8`.
A dense `n = 5` instance can therefore land above a sparse `n = 6` one, which is the correct
behaviour and not a defect.

NO AGENT QUANTITY REACHES THIS FILE. Selection uses the generator, CP-SAT, the conflict
graph and a deterministic order. Choosing instances after seeing how C1 or C3 did on them
would tune the axis to model behaviour; section 3 states that rule as absolute and a
regression test enforces it here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ortools.sat.python import cp_model  # noqa: E402

from scripts.build_pilot_manifest import content_hash  # noqa: E402
from scripts.collect_structural_calibration import SEED_RANGES  # noqa: E402

BINNING_SCHEMA_VERSION = "pilot_manifest/1.0"
SUBSET_SCHEMA_VERSION = "pilot_subset/1.0"
SELECTOR_VERSION = "lowcx_manifest/1.0"

SIZES: tuple[int, ...] = (4, 5, 6)
FIXED_OPTIMUM = 3
PER_SIZE = 12

# The frozen n = 8 boundaries, as densities. Read back at any size they give that size's
# conflict-count cut-points; at n = 8 they return 1 and 7, which is where they came from.
REFERENCE_N = 8
REFERENCE_EASY_MAX_PAIRS = 1
REFERENCE_MEDIUM_MAX_PAIRS = 7

# Diversity, unchanged from the n = 8 design (THESIS_DECISIONS section 3, 2026-08-09).
MIN_TIGHTNESS_VALUES = 2
MIN_TRAVEL_STRUCTURES = 2
MAX_TIGHTNESS_SHARE_NUM, MAX_TIGHTNESS_SHARE_DEN = 6, 10

# Travel structure is fixed exactly, identically at every size, and this is stronger than
# the `>= 2 structures` rule it replaces -- an equal split satisfies that rule trivially.
#
# WHY AN EXACT HISTOGRAM RATHER THAN A SHARE CEILING. A first selection under the n = 8
# rules returned 11 clustered and 1 line at every size. The cause is structural and was
# found before any agent ran: one instance per seed, a canonical order that sorts the
# structure name alphabetically, `clustered` first in that order, and an `O = 3` candidate
# of that structure available for all 100 seeds -- so the cheapest choice for every seed
# was the same topology. At n = 8 the band restriction removed most seeds' in-band
# clustered candidate and the effect never appeared. A ceiling of 60% would still permit
# 7 of one topology against 5 of another, which does not fix what went wrong.
#
# WHY 3/3/3/3 RATHER THAN THE FROZEN n = 8 COMPOSITION. The frozen set's `O = 3` entries
# do not supply one template to copy: by band they are clustered 5/3/9, line 2/7/1, random
# 4/2/2 and uniform 1/0/0, so `uniform` is absent from two bands of three and the pooled
# shape 17/10/8/1 is itself dominated by one topology. With no natural composition to
# continue, the equal split is used -- and being identical across sizes, it guarantees that
# changing `n` does not change travel topology.
_STRUCTURES: tuple[str, ...] = ("clustered", "line", "random", "uniform")


def structure_histogram(per_size: int) -> dict[str, int]:
    """The equal split across travel structures, for any size of selection.

    The rule above is "the equal split, identical at every size". At a given `per_size`
    that fixes the counts with no freedom left, so this derives them rather than carrying
    a hard-coded table that silently belongs to one `per_size`.
    """
    if per_size % len(_STRUCTURES):
        raise ValueError(f"per_size {per_size} is not divisible by {len(_STRUCTURES)} "
                         "travel structures, so there is no equal split")
    return {s: per_size // len(_STRUCTURES) for s in _STRUCTURES}


REQUIRED_STRUCTURE_HISTOGRAM: dict[str, int] = structure_histogram(PER_SIZE)

_SOLVER_SEED = 0
_SOLVER_SECONDS = 120.0

# One reserved seed block per (family, size). The held-out blocks are the table frozen in
# the THESIS_DECISIONS amendment of 2026-08-26; they are a different partition of a
# different locked range, not a re-cut of the calibration pools.
SEED_RANGE_FAMILIES: dict[str, dict[int, str]] = {
    "lower_complexity": {4: "lower_complexity_n4", 5: "lower_complexity_n5",
                         6: "lower_complexity_n6"},
    "held_out": {4: "held_out_n4", 5: "held_out_n5", 6: "held_out_n6"},
}
LABEL_FOR_FAMILY: dict[str, str] = {"lower_complexity": "lowcx", "held_out": "heldout"}
DEFAULT_FAMILY = "lower_complexity"
SEED_RANGE_FOR_SIZE = SEED_RANGE_FAMILIES[DEFAULT_FAMILY]


class SelectionError(RuntimeError):
    """The pool cannot supply the design. Never silently worked around."""


class SeedRangeError(RuntimeError):
    """A selected seed lies outside the range reserved for its size."""


# --------------------------------------------------------------------------- #
# Boundaries.
# --------------------------------------------------------------------------- #
def max_pairs(n: int) -> int:
    return n * (n - 1) // 2


def boundaries_for(n: int) -> dict[str, int]:
    """The frozen density cut-points, expressed in conflict counts at size `n`.

    Floor rather than round: a boundary must never admit a density the frozen design
    excluded, and floor is the direction that cannot.
    """
    ref = max_pairs(REFERENCE_N)
    scale = max_pairs(n)
    return {
        "b1_easy_max": REFERENCE_EASY_MAX_PAIRS * scale // ref,
        "b2_medium_max": REFERENCE_MEDIUM_MAX_PAIRS * scale // ref,
    }


def level_for(pairs: int, boundaries: dict[str, int]) -> str:
    if pairs <= boundaries["b1_easy_max"]:
        return "easy"
    if pairs <= boundaries["b2_medium_max"]:
        return "medium"
    return "hard"


# --------------------------------------------------------------------------- #
# The pool.
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool(pool_dir: Path, n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads((pool_dir / "pool_meta.json").read_text(encoding="utf-8"))
    if not meta.get("audit", {}).get("all_checks_passed"):
        raise SelectionError(f"{pool_dir}: pool failed its completeness audit; a sample "
                             "must not be cut from it")
    rows: list[dict[str, Any]] = []
    with (pool_dir / "candidates.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "instance_id": r["instance_id"],
                "seed": int(r["seed"]),
                "n_people": int(r["n_people"]),
                "tightness": float(r["tightness"]),
                "overlap": float(r["overlap"]),
                "travel_structure": r["travel_structure"],
                "conflicting_pairs": int(r["conflicting_pairs"]),
                "oracle_optimum": int(r["oracle_optimum"]),
                "proven_optimal": r["proven_optimal"] == "True",
                "alpha_reachable": int(r["alpha_reachable"]),
                "higher_order_gap_H": int(r["higher_order_gap_H"]),
                "individually_reachable_count": int(r["individually_reachable_count"]),
            })
    bad = {r["n_people"] for r in rows} - {n}
    if bad:
        raise SelectionError(f"{pool_dir}: expected n_people={n}, found {sorted(bad)}")
    meta["_dir"] = str(pool_dir)
    meta["_candidates_sha256"] = _sha256(pool_dir / "candidates.csv")
    return rows, meta


def canonical_key(row: dict[str, Any]) -> tuple:
    """Selection order: the collector's own walk. Nothing about difficulty enters it."""
    return (row["seed"], row["travel_structure"], row["tightness"], row["overlap"])


def eligible(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidates at the fixed optimum, in canonical order.

    An unproven optimum is excluded: the design is matched on the optimum, so a number the
    solver could not prove cannot be matched on. The collector audits that this set is
    empty, so this is a guard rather than a routine filter.
    """
    out = [r for r in rows
           if r["oracle_optimum"] == FIXED_OPTIMUM and r["proven_optimal"]]
    out.sort(key=canonical_key)
    return out


# --------------------------------------------------------------------------- #
# The selection.
# --------------------------------------------------------------------------- #
def select(pools: dict[int, list[dict[str, Any]]], per_size: int = PER_SIZE
           ) -> dict[int, list[dict[str, Any]]]:
    """Earliest admissible selection under the canonical order.

    Decision variables are per candidate instance rather than per structural cell: one
    instance per seed and the rank tie-break are properties of instances and cannot be
    written over counts.
    """
    model = cp_model.CpModel()
    x: dict[tuple[int, int], Any] = {}
    for n, rows in pools.items():
        for i in range(len(rows)):
            x[n, i] = model.NewBoolVar(f"x_{n}_{i}")

    for n, rows in pools.items():
        if len(rows) < per_size:
            raise SelectionError(
                f"n = {n}: only {len(rows)} candidates at O = {FIXED_OPTIMUM}, "
                f"{per_size} required")
        model.Add(sum(x[n, i] for i in range(len(rows))) == per_size)

        used_t = []
        for t in sorted({r["tightness"] for r in rows}):
            xt = sum(x[n, i] for i, r in enumerate(rows) if r["tightness"] == t)
            model.Add(MAX_TIGHTNESS_SHARE_DEN * xt <= MAX_TIGHTNESS_SHARE_NUM * per_size)
            u = model.NewBoolVar(f"u_{n}_{t}")
            model.Add(u <= xt)
            used_t.append(u)
        model.Add(sum(used_t) >= MIN_TIGHTNESS_VALUES)

        # The exact travel-structure histogram, identical at every size. Structures outside
        # it are excluded rather than left free, or the required counts could be met while
        # the selection still carried something else on top.
        wanted = structure_histogram(per_size)
        for s, want in sorted(wanted.items()):
            model.Add(sum(x[n, i] for i, r in enumerate(rows)
                          if r["travel_structure"] == s) == want)
        model.Add(sum(x[n, i] for i, r in enumerate(rows)
                      if r["travel_structure"] not in wanted) == 0)

    # One instance per seed across the whole selection. The three reserved ranges are
    # disjoint so this binds within a size in practice, but it is written across the
    # selection because that is the rule the n = 8 design states.
    by_seed: dict[int, list[Any]] = {}
    for n, rows in pools.items():
        for i, r in enumerate(rows):
            by_seed.setdefault(r["seed"], []).append(x[n, i])
    for vars_ in by_seed.values():
        if len(vars_) > 1:
            model.Add(sum(vars_) <= 1)

    model.Minimize(sum(i * x[n, i] for n, rows in pools.items() for i in range(len(rows))))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1        # determinism, as everywhere else
    solver.parameters.random_seed = _SOLVER_SEED
    solver.parameters.max_time_in_seconds = _SOLVER_SECONDS
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise SelectionError(
            "no admissible selection under the frozen diversity rules. The recorded "
            "response is to report it, never to relax a rule or to widen the optimum.")
    return {n: [r for i, r in enumerate(rows) if solver.Value(x[n, i])]
            for n, rows in sorted(pools.items())}


# --------------------------------------------------------------------------- #
# Manifests.
# --------------------------------------------------------------------------- #
def _entry(row: dict[str, Any], boundaries: dict[str, int]) -> dict[str, Any]:
    return {
        "instance_id": row["instance_id"],
        "seed": row["seed"],
        "level": level_for(row["conflicting_pairs"], boundaries),
        "optimum": row["oracle_optimum"],
        "complexity_metric": row["conflicting_pairs"],
        "cell": {
            "n_people": row["n_people"],
            "tightness": row["tightness"],
            "overlap": row["overlap"],
            "travel_structure": row["travel_structure"],
        },
    }


def build_manifests(n: int, chosen: Sequence[dict[str, Any]], pool_meta: dict[str, Any],
                    provenance: dict[str, Any], per_size: int = PER_SIZE,
                    family: str = DEFAULT_FAMILY) -> tuple[dict, dict]:
    range_name = SEED_RANGE_FAMILIES[family][n]
    lo, hi = SEED_RANGES[range_name]
    stray = sorted({r["seed"] for r in chosen if not lo <= r["seed"] <= hi})
    if stray:
        raise SeedRangeError(f"n = {n}: {len(stray)} seed(s) outside the reserved "
                             f"{range_name} range {lo}-{hi}: {stray[:8]}")

    bounds = boundaries_for(n)
    entries = [_entry(r, bounds) for r in chosen]
    by_level: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for e in entries:
        by_level[e["level"]].append(e)

    binning: dict[str, Any] = {
        "schema_version": BINNING_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "grid_name": f"lowcx_n{n}",
        "purpose": ("lower-complexity extension: 12 instances at a fixed oracle optimum. "
                    "Level labels describe conflict density and never task size; n is a "
                    "generator parameter and is not a complexity metric"),
        "seed_range_name": range_name,
        "seed_range": {"start": lo, "end": hi},
        "n_people": n,
        "fixed_optimum": FIXED_OPTIMUM,
        "n_per_size": per_size,
        "n_accepted": len(entries),
        "boundaries": bounds,
        "boundary_derivation": {
            "rule": ("the frozen n = 8 cut-points as densities, floored at this size: "
                     "easy is D <= 1/28, medium is D <= 7/28"),
            "reference_n": REFERENCE_N,
            "reference_pairs": [REFERENCE_EASY_MAX_PAIRS, REFERENCE_MEDIUM_MAX_PAIRS],
            "max_pairs_here": max_pairs(n),
        },
        "level_counts": {lv: len(v) for lv, v in by_level.items()},
        "pool": {
            "dir": pool_meta.get("_dir"),
            "git_commit": pool_meta.get("git_commit"),
            "seed_start": pool_meta.get("seed_start"),
            "seed_end": pool_meta.get("seed_end"),
            "n_candidates": pool_meta.get("n_completed"),
            "candidates_sha256": pool_meta.get("_candidates_sha256"),
        },
        "selection": provenance,
    }
    binning["content_hash"] = content_hash(binning)
    binning["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    subset: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "manifest_hash": binning["content_hash"],
        "manifest_schema_version": binning["schema_version"],
        "grid_name": binning["grid_name"],
        "seed_range_name": range_name,
        "n_people": n,
        "fixed_optimum": FIXED_OPTIMUM,
        "per_level_requested": None,        # levels are an outcome here, not a target
        "counts": {lv: len(v) for lv, v in by_level.items()},
        "representation": {
            "n_people": sorted({e["cell"]["n_people"] for e in entries}),
            "travel_structure": {s: c for s, c in sorted(Counter(
                e["cell"]["travel_structure"] for e in entries).items())},
            "tightness": {str(t): c for t, c in sorted(Counter(
                e["cell"]["tightness"] for e in entries).items())},
            "overlap": {str(o): c for o, c in sorted(Counter(
                e["cell"]["overlap"] for e in entries).items())},
            "optimum_counts": {str(k): c for k, c in sorted(Counter(
                e["optimum"] for e in entries).items())},
            "conflict_pairs": {str(k): c for k, c in sorted(Counter(
                e["complexity_metric"] for e in entries).items())},
            "higher_order_gap_H": {str(k): c for k, c in sorted(Counter(
                r["higher_order_gap_H"] for r in chosen).items())},
        },
        "instances": by_level,
    }
    subset["content_hash"] = content_hash(subset)
    subset["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return binning, subset


# --------------------------------------------------------------------------- #
# Verification, from the emitted documents alone.
# --------------------------------------------------------------------------- #
def verify(binning: dict[str, Any], subset: dict[str, Any], per_size: int = PER_SIZE,
           family: str = DEFAULT_FAMILY) -> list[str]:
    """Re-derive every requirement from the output rather than trusting the search.

    A constraint that was mis-stated in the CP-SAT model would be satisfied by its own
    solution and still be wrong, so this reads the documents.
    """
    problems: list[str] = []
    n = binning["n_people"]
    range_name = SEED_RANGE_FAMILIES[family].get(n)
    if range_name is None:
        problems.append(f"n = {n} has no reserved seed range")
        return problems
    lo, hi = SEED_RANGES[range_name]
    entries = [e for rows in subset["instances"].values() for e in rows]

    if len(entries) != per_size:
        problems.append(f"n = {n}: {len(entries)} instances, expected {per_size}")

    seeds = Counter(e["seed"] for e in entries)
    repeated = sorted(s for s, c in seeds.items() if c > 1)
    if repeated:
        problems.append(f"seed(s) used more than once: {repeated[:8]}")
    for e in entries:
        if not lo <= e["seed"] <= hi:
            problems.append(f"{e['instance_id']}: seed {e['seed']} outside {lo}-{hi}")
        if e["optimum"] != FIXED_OPTIMUM:
            problems.append(f"{e['instance_id']}: optimum {e['optimum']}, "
                            f"expected {FIXED_OPTIMUM}")
        if e["cell"]["n_people"] != n:
            problems.append(f"{e['instance_id']}: n_people {e['cell']['n_people']}")

    tight = Counter(e["cell"]["tightness"] for e in entries)
    if len(tight) < MIN_TIGHTNESS_VALUES:
        problems.append(f"n = {n}: {len(tight)} tightness value(s), "
                        f"at least {MIN_TIGHTNESS_VALUES} required")
    for t, c in tight.items():
        if MAX_TIGHTNESS_SHARE_DEN * c > MAX_TIGHTNESS_SHARE_NUM * len(entries):
            problems.append(f"n = {n}: tightness {t} is {c}/{len(entries)}, above 60%")
    structures = Counter(e["cell"]["travel_structure"] for e in entries)
    required = structure_histogram(per_size)
    if dict(structures) != required:
        problems.append(f"n = {n}: travel structures {dict(sorted(structures.items()))}, "
                        f"required {required}")
    # The rule this replaces, checked as well: the exact histogram is meant to be strictly
    # stronger, so a change to it that weakened the older guarantee must still be caught.
    if len(structures) < MIN_TRAVEL_STRUCTURES:
        problems.append(f"n = {n}: fewer than {MIN_TRAVEL_STRUCTURES} travel structures")

    bounds = binning["boundaries"]
    if bounds != boundaries_for(n):
        problems.append(f"n = {n}: boundaries {bounds} are not the derived "
                        f"{boundaries_for(n)}")
    for level, rows in subset["instances"].items():
        for e in rows:
            derived = level_for(e["complexity_metric"], bounds)
            if derived != level or derived != e["level"]:
                problems.append(
                    f"{e['instance_id']}: {e['complexity_metric']} conflict pairs give "
                    f"level {derived!r}, but it is filed under {level!r} as {e['level']!r}")

    if subset["manifest_hash"] != binning["content_hash"]:
        problems.append("subset does not point at this binning document")
    for doc, name in ((binning, "binning"), (subset, "subset")):
        stored = doc["content_hash"]
        recomputed = content_hash({k: v for k, v in doc.items()
                                   if k not in ("content_hash", "timestamp_utc")})
        if stored != recomputed:
            problems.append(f"{name}: content hash does not reproduce")
    return problems


def audit_report(n: int, chosen: Sequence[dict[str, Any]], subset: dict[str, Any]
                 ) -> dict[str, Any]:
    """Everything the amendment has to show, computed from the selection."""
    bounds = boundaries_for(n)
    return {
        "n_people": n,
        "n_instances": len(chosen),
        "content_hash": subset["content_hash"],
        "instance_ids": [r["instance_id"] for r in chosen],
        "seeds": sorted(r["seed"] for r in chosen),
        "optimum": {str(k): c for k, c in sorted(Counter(
            r["oracle_optimum"] for r in chosen).items())},
        "conflict_pairs": {str(k): c for k, c in sorted(Counter(
            r["conflicting_pairs"] for r in chosen).items())},
        "density_D": sorted({round(r["conflicting_pairs"] / max_pairs(n), 4)
                             for r in chosen}),
        "higher_order_gap_H": {str(k): c for k, c in sorted(Counter(
            r["higher_order_gap_H"] for r in chosen).items())},
        "alpha_reachable": {str(k): c for k, c in sorted(Counter(
            r["alpha_reachable"] for r in chosen).items())},
        "individually_reachable": {str(k): c for k, c in sorted(Counter(
            r["individually_reachable_count"] for r in chosen).items())},
        "tightness": {str(t): c for t, c in sorted(Counter(
            r["tightness"] for r in chosen).items())},
        "overlap": {str(o): c for o, c in sorted(Counter(
            r["overlap"] for r in chosen).items())},
        "travel_structure": {s: c for s, c in sorted(Counter(
            r["travel_structure"] for r in chosen).items())},
        "levels": {lv: len(v) for lv, v in subset["instances"].items()},
        "boundaries": bounds,
        "max_pairs": max_pairs(n),
    }


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", required=True, metavar="N=DIR")
    ap.add_argument("--out", type=Path, default=Path("results/manifests"))
    ap.add_argument("--per-size", type=int, default=PER_SIZE)
    ap.add_argument("--seed-range-family", choices=sorted(SEED_RANGE_FAMILIES),
                    default=DEFAULT_FAMILY,
                    help="which reserved seed-block partition the pools come from")
    args = ap.parse_args(argv)

    pools_raw: dict[int, list[dict[str, Any]]] = {}
    metas: dict[int, dict[str, Any]] = {}
    for spec in args.pool:
        if "=" not in spec:
            ap.error(f"--pool expects N=DIR, got {spec!r}")
        n_str, path = spec.split("=", 1)
        n = int(n_str)
        if n not in SEED_RANGE_FAMILIES[args.seed_range_family]:
            ap.error(f"n = {n} has no reserved seed range in "
                     f"{args.seed_range_family}; sizes are "
                     f"{sorted(SEED_RANGE_FAMILIES[args.seed_range_family])}")
        pools_raw[n], metas[n] = load_pool(Path(path), n)

    pools = {n: eligible(rows) for n, rows in pools_raw.items()}
    for n, rows in sorted(pools.items()):
        print(f"n = {n}: {len(pools_raw[n])} candidates, "
              f"{len(rows)} at O = {FIXED_OPTIMUM}")

    chosen = select(pools, args.per_size)

    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for n, rows in sorted(chosen.items()):
        provenance = {
            "rule": (f"earliest admissible selection in canonical order (seed, travel "
                     f"structure, tightness, overlap) at O = {FIXED_OPTIMUM}"),
            "fixed_optimum": FIXED_OPTIMUM,
            "diversity": {
                "min_tightness_values": MIN_TIGHTNESS_VALUES,
                "max_tightness_share": f"{MAX_TIGHTNESS_SHARE_NUM}/{MAX_TIGHTNESS_SHARE_DEN}",
                "travel_structure_histogram": structure_histogram(args.per_size),
                "travel_structure_rule": (
                    "exact and identical at every size, so a change in n is not "
                    "accompanied by a change in travel topology; supersedes the weaker "
                    "'at least 2 structures' rule, which an equal split satisfies"),
                "one_instance_per_seed": True,
            },
            "n_eligible": len(pools[n]),
            "solver": {"workers": 1, "random_seed": _SOLVER_SEED,
                       "objective": "minimise total canonical rank"},
            "no_agent_information_used": True,
        }
        binning, subset = build_manifests(n, rows, metas[n], provenance,
                                          args.per_size, args.seed_range_family)
        problems = verify(binning, subset, args.per_size, args.seed_range_family)
        if problems:
            raise SelectionError("the emitted manifests fail their own verification:\n  - "
                                 + "\n  - ".join(problems))
        # Each document is named after its OWN content hash, as the n = 8 pair is. The
        # sweep runner finds the binning document by searching for the hash the subset
        # points at, so naming the binning file after the subset's hash leaves it
        # undiscoverable and every command needs --binning-manifest by hand.
        label = LABEL_FOR_FAMILY[args.seed_range_family]
        b_path = args.out / f"binning__{label}_n{n}__{binning['content_hash'][:12]}.json"
        s_path = args.out / f"subset__{label}_n{n}__{subset['content_hash'][:12]}.json"
        for p in (b_path, s_path):
            if p.exists():
                raise SelectionError(f"refusing to overwrite {p}")
        write_atomic(b_path, json.dumps(binning, indent=2, ensure_ascii=False) + "\n")
        write_atomic(s_path, json.dumps(subset, indent=2, ensure_ascii=False) + "\n")
        reports[str(n)] = audit_report(n, rows, subset)
        reports[str(n)]["binning_path"] = str(b_path)
        reports[str(n)]["subset_path"] = str(s_path)
        print(f"  written: {s_path.name} (verified)")

    audit_path = args.out / f"audit__{LABEL_FOR_FAMILY[args.seed_range_family]}.json"
    write_atomic(audit_path, json.dumps({
        "schema_version": "lowcx_audit/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixed_optimum": FIXED_OPTIMUM,
        "per_size": args.per_size,
        "sizes": sorted(reports),
        "no_agent_information_used": True,
        "per_size_report": reports,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"written: {audit_path}")
    print(f"selected {sum(r['n_instances'] for r in reports.values())} instances, "
          f"all at O = {FIXED_OPTIMUM}, every manifest verified from its own documents")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
