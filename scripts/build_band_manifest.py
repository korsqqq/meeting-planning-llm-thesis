# scripts/build_band_manifest.py
"""Deterministic instance selector for the frozen density bands. CPU only, no LLM.

    python -m scripts.build_band_manifest \
        --pool results/calibration/budget_dev \
        --per-band 20 --range budget_dev --out results/manifests

Takes a structural candidate pool and cuts from it a sample that satisfies, jointly and on
the selected instances themselves, every requirement frozen in THESIS_DECISIONS section 3:

  * `n_people = 8`;
  * the three frozen bands -- conflict pairs 0-1 (Low), 7 (Medium), 13-16 (High);
  * an **identical empirical oracle-optimum histogram** across the three bands;
  * at least two tightness values per band, none above 60% of the band;
  * at least two travel structures per band.

Nothing here is chosen by judgement. The optimum histogram is fixed before the solve, the
selection is the earliest admissible one under a canonical order, and the solver is used
only as feasibility machinery over structural facts. **No agent, plan, score or token count
enters this file**, which is what keeps the complexity axis from being tuned to observed
model behaviour.

WHAT IS NEW HERE, AND WHAT WAS ALREADY FROZEN.

Frozen earlier (2026-08-09 amendment): the bands, `n = 8`, the identical-`O` requirement and
the diversity requirements. Fixed here, before any development run existed:

  * *the shape of the shared `O` histogram*. The frozen design's own histogram -- 41 at
    `O = 3` and 24 at `O = 4` out of 65 -- is scaled to the requested band size by
    largest-remainder rounding. At 20 per band that is 13 and 7. Keeping the composition of
    the frozen design is what lets a rate measured on this sample be read as a rate on the
    main sample; choosing a fresh histogram per sample would quietly change the difficulty
    of the set between calibration and confirmation. If that histogram is not attainable the
    script walks a fixed ladder of alternatives ordered by distance from it, and records
    which one was used and why -- rather than silently returning whatever fits.
  * *one instance per seed across the whole manifest*. The primary analysis resamples
    instances as independent units, and two instances built from one seed share a random
    stream. This costs nothing here and removes a dependence the bootstrap would otherwise
    have to assume away.
  * *the canonical order* -- seed, then travel structure, then tightness, then overlap --
    which matches the collector's own walk. Among all admissible selections the solver
    returns the one minimising total rank in that order, so the result is a fact about the
    pool rather than a solver artefact.

SEED RANGES. Only the named ranges reserved in section 3 are reachable, and the held-out
main range is not one of them. Opening held-out is a decision recorded in the document, not
a command-line argument. Every emitted manifest states its range and the selector re-checks
each chosen seed against it.

OUTPUT. A `binning__*` / `subset__*` pair in the schema `scripts/run_pilot_sweep.py`
already verifies, so the frozen sweep runner executes this selection without modification
and re-derives every instance from its seed before spending a token. The three bands map
onto the runner's level labels exactly: with boundaries `easy <= 1` and `medium <= 7`, the
selected conflict counts 0-1 / 7 / 13-16 fall into easy / medium / hard with no candidate
near a boundary. The labels are a compatibility alias; the reported names are Low, Medium
and High pairwise conflict density.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ortools.sat.python import cp_model  # noqa: E402

from scripts.build_pilot_manifest import content_hash  # noqa: E402
from scripts.collect_structural_calibration import SEED_RANGES  # noqa: E402

BINNING_SCHEMA_VERSION = "pilot_manifest/1.0"
SUBSET_SCHEMA_VERSION = "pilot_subset/1.0"
SELECTOR_VERSION = "band_manifest/1.0"

N_PEOPLE = 8
MAX_PAIRS = N_PEOPLE * (N_PEOPLE - 1) // 2          # 28

# The frozen bands (THESIS_DECISIONS section 3, amendment of 2026-08-09), in the order
# Low, Medium, High, with the runner's level label each maps onto.
BANDS: tuple[tuple[str, str, int, int], ...] = (
    ("low", "easy", 0, 1),
    ("medium", "medium", 7, 7),
    ("high", "hard", 13, 16),
)
# Chosen so that `level_for_metric` reproduces the band membership exactly on the selected
# sample. Both boundaries sit in the frozen gaps, not inside a band.
LEVEL_BOUNDARIES = {"b1_easy_max": 1, "b2_medium_max": 7}

# The frozen design's own optimum composition, scaled to whatever band size is requested.
REFERENCE_OPTIMUM_HISTOGRAM: dict[int, int] = {3: 41, 4: 24}

MIN_TIGHTNESS_VALUES = 2
MIN_TRAVEL_STRUCTURES = 2
MAX_TIGHTNESS_SHARE_NUM, MAX_TIGHTNESS_SHARE_DEN = 6, 10    # 60%, as exact integers
_SOLVER_SEED = 0
_SOLVER_SECONDS = 120.0


class SelectionError(RuntimeError):
    """The pool cannot supply the frozen design. Never silently worked around."""


class SeedRangeError(RuntimeError):
    """A candidate seed lies outside the requested reserved range."""


# --------------------------------------------------------------------------- #
# The pool.
# --------------------------------------------------------------------------- #
def load_pool(pool_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a structural calibration pool written by the collector."""
    meta = json.loads((pool_dir / "pool_meta.json").read_text(encoding="utf-8"))
    with (pool_dir / "candidates.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    typed: list[dict[str, Any]] = []
    for r in rows:
        typed.append({
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
        })
    return typed, meta


def canonical_key(row: dict[str, Any]) -> tuple:
    """Selection order: the collector's own walk. Nothing about difficulty enters it."""
    return (row["seed"], row["travel_structure"], row["tightness"], row["overlap"])


def band_of(pairs: int) -> str | None:
    for name, _level, lo, hi in BANDS:
        if lo <= pairs <= hi:
            return name
    return None


def eligible(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Candidates per band, in canonical order.

    An unproven optimum is excluded: the whole design is matched on the optimum, so a
    number the solver could not prove cannot be matched on. The collector audits that this
    set is empty, so the filter is a guard rather than a routine step.
    """
    out: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in BANDS}
    for r in rows:
        if r["n_people"] != N_PEOPLE or not r["proven_optimal"]:
            continue
        name = band_of(r["conflicting_pairs"])
        if name is not None:
            out[name].append(r)
    for name in out:
        out[name].sort(key=canonical_key)
    return out


# --------------------------------------------------------------------------- #
# The shared optimum histogram, fixed before the solve.
# --------------------------------------------------------------------------- #
def scaled_histogram(total: int,
                     reference: dict[int, int] = REFERENCE_OPTIMUM_HISTOGRAM
                     ) -> dict[int, int]:
    """The reference composition at a new size, by largest-remainder rounding.

    Deterministic including the ties: remainders are compared first, then the optimum
    value, so no ordering of a dictionary can change the answer.
    """
    ref_total = sum(reference.values())
    exact = {o: total * n / ref_total for o, n in reference.items()}
    counts = {o: int(v) for o, v in exact.items()}
    short = total - sum(counts.values())
    ranked = sorted(exact, key=lambda o: (-(exact[o] - counts[o]), o))
    for o in ranked[:short]:
        counts[o] += 1
    return {o: c for o, c in sorted(counts.items()) if c}


def histogram_ladder(total: int, optima: Sequence[int]) -> Iterator[dict[int, int]]:
    """Every histogram over `optima` summing to `total`, nearest the reference first.

    The first entry is `scaled_histogram`. The rest exist so an unattainable composition
    produces a recorded, ordered retreat instead of an arbitrary one; distance is L1 from
    the reference and ties are broken lexicographically.
    """
    target = scaled_histogram(total)
    seen: set[tuple] = set()

    def distance(h: dict[int, int]) -> int:
        keys = set(h) | set(target)
        return sum(abs(h.get(k, 0) - target.get(k, 0)) for k in keys)

    def compositions(remaining: int, rest: Sequence[int]) -> Iterator[tuple[int, ...]]:
        # Enumerated by recursion rather than a cross product: at 20 per band over a
        # handful of optima the cross product is tens of millions of tuples, almost all
        # discarded for the wrong sum.
        if len(rest) == 1:
            yield (remaining,)
            return
        for c in range(remaining + 1):
            for tail in compositions(remaining - c, rest[1:]):
                yield (c, *tail)

    candidates: list[dict[int, int]] = []
    for combo in compositions(total, list(optima)):
        h = {o: c for o, c in zip(optima, combo) if c}
        key = tuple(sorted(h.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(h)
    candidates.sort(key=lambda h: (distance(h), tuple(sorted(h.items()))))
    yield from candidates


# --------------------------------------------------------------------------- #
# The selection.
# --------------------------------------------------------------------------- #
def select(pools: dict[str, list[dict[str, Any]]], per_band: int,
           histogram: dict[int, int]) -> dict[str, list[dict[str, Any]]] | None:
    """Earliest admissible selection under the canonical order, or None if impossible.

    Decision variables are per candidate instance rather than per structural cell, because
    two of the requirements -- one instance per seed, and a deterministic tie-break by rank
    -- are properties of instances and cannot be expressed over counts.
    """
    model = cp_model.CpModel()
    x: dict[tuple[str, int], Any] = {}
    for name, rows in pools.items():
        for i, _r in enumerate(rows):
            x[name, i] = model.NewBoolVar(f"x_{name}_{i}")

    for name, rows in pools.items():
        model.Add(sum(x[name, i] for i in range(len(rows))) == per_band)

        for o, want in histogram.items():
            model.Add(sum(x[name, i] for i, r in enumerate(rows)
                          if r["oracle_optimum"] == o) == want)
        # Nothing outside the shared histogram may enter, or the histograms would match on
        # the listed optima while differing overall.
        model.Add(sum(x[name, i] for i, r in enumerate(rows)
                      if r["oracle_optimum"] not in histogram) == 0)

        used_t = []
        for t in sorted({r["tightness"] for r in rows}):
            xt = sum(x[name, i] for i, r in enumerate(rows) if r["tightness"] == t)
            model.Add(MAX_TIGHTNESS_SHARE_DEN * xt
                      <= MAX_TIGHTNESS_SHARE_NUM * per_band)
            u = model.NewBoolVar(f"u_{name}_{t}")
            model.Add(u <= xt)
            used_t.append(u)
        model.Add(sum(used_t) >= MIN_TIGHTNESS_VALUES)

        used_s = []
        for s in sorted({r["travel_structure"] for r in rows}):
            xs = sum(x[name, i] for i, r in enumerate(rows) if r["travel_structure"] == s)
            v = model.NewBoolVar(f"v_{name}_{s}")
            model.Add(v <= xs)
            used_s.append(v)
        model.Add(sum(used_s) >= MIN_TRAVEL_STRUCTURES)

    # One instance per seed across the whole manifest, bands included.
    by_seed: dict[int, list[Any]] = {}
    for name, rows in pools.items():
        for i, r in enumerate(rows):
            by_seed.setdefault(r["seed"], []).append(x[name, i])
    for vars_ in by_seed.values():
        if len(vars_) > 1:
            model.Add(sum(vars_) <= 1)

    # Among admissible selections, the earliest in canonical order.
    model.Minimize(sum(i * x[name, i]
                       for name, rows in pools.items() for i in range(len(rows))))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1        # determinism, as everywhere else here
    solver.parameters.random_seed = _SOLVER_SEED
    solver.parameters.max_time_in_seconds = _SOLVER_SECONDS
    status = solver.Solve(model)
    # OPTIMAL only. Accepting FEASIBLE made the selection depend on wall-clock:
    # the objective is "earliest in canonical order", so a run that hit the time
    # limit would return SOME admissible selection rather than THE earliest one,
    # and a loaded node would then produce a different manifest from an idle one
    # with identical code, pool and seed. That is a discretionary choice made by
    # the machine's load, which is exactly what this selector exists to exclude.
    # A timeout is now a loud failure, not a quietly different answer.
    if status == cp_model.FEASIBLE:
        raise SelectionError(
            f"selection hit the {_SOLVER_SECONDS}s limit and is only FEASIBLE, not "
            "OPTIMAL: the result would not be the earliest admissible selection. "
            "Raise _SOLVER_SECONDS and re-run; never accept this silently."
        )
    if status != cp_model.OPTIMAL:
        return None
    return {name: [r for i, r in enumerate(rows) if solver.Value(x[name, i])]
            for name, rows in pools.items()}


def choose(pools: dict[str, list[dict[str, Any]]], per_band: int
           ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Walk the histogram ladder and take the first attainable composition."""
    optima = sorted({r["oracle_optimum"] for rows in pools.values() for r in rows})
    attempts: list[dict[str, Any]] = []
    for h in histogram_ladder(per_band, optima):
        chosen = select(pools, per_band, h)
        attempts.append({"histogram": {str(k): v for k, v in h.items()},
                         "attainable": chosen is not None})
        if chosen is not None:
            return chosen, {
                "optimum_histogram": {str(k): v for k, v in h.items()},
                "reference_histogram": {str(k): v for k, v
                                        in REFERENCE_OPTIMUM_HISTOGRAM.items()},
                "scaled_reference": {str(k): v for k, v
                                     in scaled_histogram(per_band).items()},
                "used_the_scaled_reference": h == scaled_histogram(per_band),
                "attempts": attempts,
            }
        if len(attempts) >= 40:
            break
    raise SelectionError(
        f"no attainable optimum histogram at {per_band} per band after "
        f"{len(attempts)} candidates. The pool cannot supply the frozen design; the "
        "recorded response is the fallback in THESIS_DECISIONS section 3, never a change "
        "to the bands or to the matching variable."
    )


# --------------------------------------------------------------------------- #
# Manifests.
# --------------------------------------------------------------------------- #
def _entry(row: dict[str, Any], level: str) -> dict[str, Any]:
    return {
        "instance_id": row["instance_id"],
        "seed": row["seed"],
        "level": level,
        "optimum": row["oracle_optimum"],
        "complexity_metric": row["conflicting_pairs"],
        "cell": {
            "n_people": row["n_people"],
            "tightness": row["tightness"],
            "overlap": row["overlap"],
            "travel_structure": row["travel_structure"],
        },
    }


def build_manifests(chosen: dict[str, list[dict[str, Any]]], *, per_band: int,
                    seed_range: str, pool_meta: dict[str, Any],
                    provenance: dict[str, Any]) -> tuple[dict, dict]:
    lo, hi = SEED_RANGES[seed_range]
    stray = sorted({r["seed"] for rows in chosen.values() for r in rows
                    if not lo <= r["seed"] <= hi})
    if stray:
        raise SeedRangeError(
            f"{len(stray)} selected seed(s) outside the reserved {seed_range} range "
            f"{lo}-{hi}: {stray[:8]}")

    by_level = {level: [_entry(r, level) for r in chosen[band]]
                for band, level, _lo, _hi in BANDS}

    binning: dict[str, Any] = {
        "schema_version": BINNING_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "grid_name": f"bands_{seed_range}",
        "purpose": ("frozen density bands (THESIS_DECISIONS section 3, amendment "
                    "2026-08-09); level labels are a compatibility alias for the bands"),
        "seed_range_name": seed_range,
        "seed_range": {"start": lo, "end": hi},
        "n_people": N_PEOPLE,
        "bands": [{"band": b, "level": lv, "pairs": [lo_k, hi_k],
                   "D": [round(lo_k / MAX_PAIRS, 4), round(hi_k / MAX_PAIRS, 4)]}
                  for b, lv, lo_k, hi_k in BANDS],
        "boundaries": dict(LEVEL_BOUNDARIES),
        "n_per_band": per_band,
        "n_accepted": sum(len(v) for v in by_level.values()),
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
        "seed_range_name": seed_range,
        "per_level_requested": per_band,
        "counts": {lv: len(v) for lv, v in by_level.items()},
        "short_levels": [lv for lv, v in by_level.items() if len(v) < per_band],
        "representation": {
            lv: {
                "n_people": sorted({e["cell"]["n_people"] for e in v}),
                "travel_structure": sorted({e["cell"]["travel_structure"] for e in v}),
                "tightness": sorted({e["cell"]["tightness"] for e in v}),
                "optimum_counts": {str(k): c for k, c in
                                   sorted(Counter(e["optimum"] for e in v).items())},
                "conflict_pairs": sorted({e["complexity_metric"] for e in v}),
            }
            for lv, v in by_level.items()
        },
        "instances": by_level,
    }
    subset["content_hash"] = content_hash(subset)
    subset["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return binning, subset


def verify(binning: dict[str, Any], subset: dict[str, Any], per_band: int,
           seed_range: str) -> list[str]:
    """Re-derive every frozen requirement from the emitted documents alone.

    Deliberately independent of the solver: a constraint that was mis-stated in the model
    would be satisfied by the solution and still be wrong, so the check reads the output
    rather than trusting the search that produced it.
    """
    problems: list[str] = []
    lo, hi = SEED_RANGES[seed_range]
    levels = {lv for _b, lv, _l, _h in BANDS}
    seen_seeds: Counter = Counter()

    if set(subset["instances"]) != levels:
        problems.append(f"levels {sorted(subset['instances'])}, expected {sorted(levels)}")

    for band, level, k_lo, k_hi in BANDS:
        rows = subset["instances"].get(level, [])
        if len(rows) != per_band:
            problems.append(f"{band}: {len(rows)} instances, expected {per_band}")
        for e in rows:
            seen_seeds[e["seed"]] += 1
            if not lo <= e["seed"] <= hi:
                problems.append(f"{e['instance_id']}: seed {e['seed']} outside {lo}-{hi}")
            if not k_lo <= e["complexity_metric"] <= k_hi:
                problems.append(f"{e['instance_id']}: {e['complexity_metric']} conflict "
                                f"pairs outside the {band} band {k_lo}-{k_hi}")
            if e["cell"]["n_people"] != N_PEOPLE:
                problems.append(f"{e['instance_id']}: n_people {e['cell']['n_people']}")

        tight = Counter(e["cell"]["tightness"] for e in rows)
        if len(tight) < MIN_TIGHTNESS_VALUES:
            problems.append(f"{band}: {len(tight)} tightness value(s)")
        for t, c in tight.items():
            if MAX_TIGHTNESS_SHARE_DEN * c > MAX_TIGHTNESS_SHARE_NUM * len(rows):
                problems.append(f"{band}: tightness {t} is {c}/{len(rows)}, above 60%")
        if len({e["cell"]["travel_structure"] for e in rows}) < MIN_TRAVEL_STRUCTURES:
            problems.append(f"{band}: fewer than {MIN_TRAVEL_STRUCTURES} travel structures")

    histograms = {lv: tuple(sorted(Counter(e["optimum"] for e in rows).items()))
                  for lv, rows in subset["instances"].items()}
    if len(set(histograms.values())) > 1:
        problems.append(f"optimum histograms differ across bands: {histograms}")

    repeated = {s: c for s, c in seen_seeds.items() if c > 1}
    if repeated:
        problems.append(f"{len(repeated)} seed(s) used more than once: "
                        f"{sorted(repeated)[:8]}")

    b = binning["boundaries"]
    for _band, level, k_lo, k_hi in BANDS:
        derived = ("easy" if k_hi <= b["b1_easy_max"]
                   else "medium" if k_hi <= b["b2_medium_max"] else "hard")
        low_end = ("easy" if k_lo <= b["b1_easy_max"]
                   else "medium" if k_lo <= b["b2_medium_max"] else "hard")
        if derived != level or low_end != level:
            problems.append(f"band {k_lo}-{k_hi} does not map onto level {level!r} under "
                            f"boundaries {b}")

    if content_hash(binning) != binning["content_hash"]:
        problems.append("binning content hash does not reproduce")
    if content_hash(subset) != subset["content_hash"]:
        problems.append("subset content hash does not reproduce")
    if subset["manifest_hash"] != binning["content_hash"]:
        problems.append("subset does not point at the emitted binning manifest")
    return problems


def write_report(binning: dict, subset: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    add(f"# Band manifest — {binning['seed_range_name']}, "
        f"{binning['n_per_band']} per band\n")
    add("Selected from a structural pool by fixed rule. No agent, plan, score or token "
        "count entered the selection.\n")
    add(f"* subset hash `{subset['content_hash']}`")
    add(f"* binning hash `{binning['content_hash']}`")
    add(f"* pool `{binning['pool']['dir']}` at commit `{binning['pool']['git_commit']}`, "
        f"{binning['pool']['n_candidates']} candidates, seeds "
        f"{binning['pool']['seed_start']}–{binning['pool']['seed_end']}")
    sel = binning["selection"]
    add(f"* shared optimum histogram `{sel['optimum_histogram']}` "
        f"(scaled reference `{sel['scaled_reference']}`, "
        f"used the scaled reference: {sel['used_the_scaled_reference']})\n")

    add("\n## Composition\n")
    add("| band | level | conflict pairs | D | n | optimum | tightness | travel structures |")
    add("|---|---|---|---|---|---|---|---|")
    for spec in binning["bands"]:
        lv = spec["level"]
        rep = subset["representation"][lv]
        add(f"| {spec['band']} | {lv} | {spec['pairs'][0]}–{spec['pairs'][1]} | "
            f"{spec['D'][0]:.3f}–{spec['D'][1]:.3f} | {subset['counts'][lv]} | "
            f"{rep['optimum_counts']} | {rep['tightness']} | "
            f"{', '.join(rep['travel_structure'])} |")

    add("\n## Instances\n")
    add("| band | instance_id | seed | k | D | O | tightness | overlap | travel |")
    add("|---|---|---|---|---|---|---|---|---|")
    for spec in binning["bands"]:
        for e in subset["instances"][spec["level"]]:
            c = e["cell"]
            add(f"| {spec['band']} | `{e['instance_id']}` | {e['seed']} | "
                f"{e['complexity_metric']} | "
                f"{e['complexity_metric'] / MAX_PAIRS:.3f} | {e['optimum']} | "
                f"{c['tightness']} | {c['overlap']} | {c['travel_structure']} |")

    seeds = sorted(e["seed"] for v in subset["instances"].values() for e in v)
    add("\n## Seeds\n")
    add(f"{len(seeds)} instances, {len(set(seeds))} distinct seeds, range "
        f"{min(seeds)}–{max(seeds)}, reserved range "
        f"{binning['seed_range']['start']}–{binning['seed_range']['end']}.\n")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def _write(payload: dict[str, Any], out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}__{payload['content_hash'][:12]}.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite an existing manifest: {path}")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--per-band", type=int, default=20)
    ap.add_argument("--range", dest="seed_range", choices=sorted(SEED_RANGES),
                    default="budget_dev",
                    help="reserved seed range; held-out main is not offered")
    ap.add_argument("--out", type=Path, default=Path("results/manifests"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    import hashlib
    rows, meta = load_pool(args.pool)
    meta["_dir"] = args.pool.as_posix()
    meta["_candidates_sha256"] = hashlib.sha256(
        (args.pool / "candidates.csv").read_bytes()).hexdigest()

    lo, hi = SEED_RANGES[args.seed_range]
    outside = sorted({r["seed"] for r in rows if not lo <= r["seed"] <= hi})
    if outside:
        raise SeedRangeError(
            f"the pool contains {len(outside)} seed(s) outside the requested "
            f"{args.seed_range} range {lo}-{hi}: {outside[:8]}")

    pools = eligible(rows)
    print(f"pool {args.pool}: {len(rows)} candidates, "
          + ", ".join(f"{b}={len(pools[b])}" for b, *_ in BANDS))

    chosen, provenance = choose(pools, args.per_band)
    binning, subset = build_manifests(chosen, per_band=args.per_band,
                                      seed_range=args.seed_range, pool_meta=meta,
                                      provenance=provenance)

    problems = verify(binning, subset, args.per_band, args.seed_range)
    for spec in binning["bands"]:
        rep = subset["representation"][spec["level"]]
        print(f"  {spec['band']:<7} k {spec['pairs'][0]}-{spec['pairs'][1]:<3} "
              f"n {subset['counts'][spec['level']]:<3} O {rep['optimum_counts']} "
              f"tightness {rep['tightness']} travel {rep['travel_structure']}")
    print(f"shared optimum histogram: {provenance['optimum_histogram']} "
          f"(scaled reference {provenance['scaled_reference']})")

    if problems:
        print("\nVERIFICATION FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("verification passed: bands, identical optimum histogram, tightness and travel "
          "diversity, one instance per seed, seed range")

    if args.dry_run:
        print(f"dry run: nothing written (subset would be {subset['content_hash'][:12]})")
        return 0

    binning_path = _write(binning, args.out, f"binning__bands_{args.seed_range}")
    subset_path = _write(subset, args.out, f"subset__bands_{args.seed_range}")
    report = args.report or (args.out / f"report__bands_{args.seed_range}.md")
    write_report(binning, subset, report)
    print(f"written: {binning_path}")
    print(f"written: {subset_path}")
    print(f"written: {report}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
