# scripts/build_pilot_manifest.py
"""Pilot instance selection, in two stages. CPU only -- no GPU, no LLM, no network.

    # stage 1: scan the seed range, accept the first N qualifying per cell, bin levels
    python -m scripts.build_pilot_manifest binning --grid dev  --seeds 0:100
    python -m scripts.build_pilot_manifest binning --grid pilot --seeds 10000:20000

    # stage 2: pick the LLM subset out of an existing manifest
    python -m scripts.build_pilot_manifest subset --manifest results/manifests/<file>.json

STAGE 1 -- binning manifest. Walks each generator-parameter cell independently, seeds in
ASCENDING order, and accepts the first N instances that qualify. Qualifying means all of:
the instance generates, CP-SAT PROVES the optimum within its time limit, and the optimum
is > 0 (an instance nobody can be met on carries no signal). Every seed considered is
recorded -- accepted or rejected, with the reason, the solver status, the optimum and the
complexity metric -- so the scan is auditable and the rejection rates are measurable per
cell rather than inferred. Level boundaries are cut ONCE over the whole accepted pool
(THESIS_DECISIONS.md section 3: easy = 0 binding conflicts, the positive conflicts split
at their median), never per cell.

STAGE 2 -- LLM pilot subset. The binning pool is a CPU artefact; running every instance
through the conditions would be wasteful. Stage 2 picks a fixed number per level,
deterministically: strata (n_people x travel structure) are visited round-robin so the
subset keeps the grid's variety instead of filling up from whichever cell happens to sort
first, and within a stratum instances are taken in ascending seed order. No manual choice
at any point.

DETERMINISM. Both documents carry a content hash over a canonical payload that EXCLUDES
wall-clock timings -- rerunning the same stage on the same range must reproduce the same
hash. Timings are recorded outside the hashed payload, where they inform the cost
estimate without making the artefact irreproducible.

WHICH SEED RANGE MAY BE INSPECTED, AND WHEN.
Reading a scan of the PILOT range is already the start of the pilot: once those numbers
have been seen, adjusting the grid or the selection rule in response to them would make
the selection chosen rather than pre-registered. The procedure is therefore:

  1. Preflight on the DEVELOPMENT range (0-9999) with the full grid. Look at it freely;
     that range never enters the pilot or the main experiment.
  2. Freeze the grid and the rules on the basis of that preflight alone.
  3. Run the pilot range (10000-19999) ONCE, straight to a written manifest, without a
     prior look. Any rerun exists only to confirm the same content hash.

TECHNICAL FAILURE CRITERIA (fixed in advance, the only grounds for changing anything
after the preflight):
  * a cell does not reach N accepted before its seed range is exhausted
    (`seed_range_exhausted`);
  * the solver fails to prove the optimum for the majority of seeds considered in a cell
    (`not_proven_rate` > 0.5) -- the cell is then beyond the oracle, not merely hard;
  * a write error, including the refusal to overwrite an existing manifest;
  * a content hash that does not reproduce on a rerun.

A high `optimum_zero_rate` is NOT one of them. It is recorded and reported, but an
instance nobody can be met on is a property of the parameter combination, not a defect:
once pilot generation has begun it is not grounds for touching the grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import generate_instance                       # noqa: E402
from src.data.complexity import level_boundaries             # noqa: E402
from src.schemas import Instance, Level, TravelStructure     # noqa: E402

MANIFEST_SCHEMA_VERSION = "pilot_manifest/1.0"
SUBSET_SCHEMA_VERSION = "pilot_subset/1.0"

# The locked pilot grid (THESIS_DECISIONS.md section 3). `line` is left out as
# qualitatively close to `clustered`; `random` breaks the triangle inequality, which
# would undermine reading the metric as geometric conflict -- both stay available for
# robustness work outside the pilot.
PILOT_GRID: dict[str, list] = {
    "n_people": [4, 6, 8],
    "tightness": [0.2, 0.7, 0.85, 1.0],
    "overlap": [0.2, 0.8],
    "travel_structure": [TravelStructure.UNIFORM, TravelStructure.CLUSTERED],
}

# A deliberately tiny grid for dry runs on the development seed range (0-9999), which
# never enters the pilot or the main experiment.
DEV_GRID: dict[str, list] = {
    "n_people": [4],
    "tightness": [0.2, 1.0],
    "overlap": [0.2],
    "travel_structure": [TravelStructure.UNIFORM],
}

GRIDS = {"pilot": PILOT_GRID, "dev": DEV_GRID}


@dataclass(frozen=True, order=True)
class Cell:
    """One generator-parameter combination. Ordered so cell iteration is stable."""

    n_people: int
    tightness: float
    overlap: float
    travel_structure: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_people": self.n_people,
            "tightness": self.tightness,
            "overlap": self.overlap,
            "travel_structure": self.travel_structure,
        }

    @property
    def key(self) -> str:
        return (f"n{self.n_people}-t{int(round(self.tightness * 100))}"
                f"-o{int(round(self.overlap * 100))}-{self.travel_structure}")


def grid_cells(grid: dict[str, list]) -> list[Cell]:
    """All cells of a grid in a fixed order (sorted, so it cannot drift)."""
    cells = [
        Cell(n, t, o, str(getattr(tr, "value", tr)))
        for n in grid["n_people"]
        for t in grid["tightness"]
        for o in grid["overlap"]
        for tr in grid["travel_structure"]
    ]
    return sorted(cells)


# --------------------------------------------------------------------------- #
# Evaluation of one candidate seed. Injectable so the selection logic is testable
# without CP-SAT.
# --------------------------------------------------------------------------- #
def evaluate_seed(cell: Cell, seed: int) -> dict[str, Any]:
    """Generate, solve and measure one candidate. Never raises on a bad instance."""
    from src.data.complexity import solve_and_annotate

    t0 = time.perf_counter()
    instance: Instance = generate_instance(
        n_people=cell.n_people,
        tightness=cell.tightness,
        overlap=cell.overlap,
        travel_structure=TravelStructure(cell.travel_structure),
        seed=seed,
    )
    annotated, solution = solve_and_annotate(instance)
    elapsed = time.perf_counter() - t0
    return {
        "instance_id": annotated.instance_id,
        "status": solution.status,
        "proven_optimal": solution.proven_optimal,
        "optimum": solution.optimum,
        "complexity_metric": annotated.complexity_metric,
        "solve_seconds": round(elapsed, 3),
    }


def rejection_reason(result: dict[str, Any]) -> str | None:
    """Why this candidate cannot enter the pool, or None if it qualifies.

    Order matters for the audit trail: an unproven optimum is a solver limitation, an
    optimum of 0 is a property of the instance, and they are counted separately.
    """
    if not result.get("proven_optimal"):
        return "not_proven"
    if result.get("optimum", 0) <= 0:
        return "optimum_zero"
    return None


def scan_cell(
    cell: Cell,
    seeds: Iterable[int],
    n_accept: int,
    evaluate: Callable[[Cell, int], dict[str, Any]] = evaluate_seed,
) -> dict[str, Any]:
    """First `n_accept` qualifying seeds of one cell, in ascending order.

    Every seed touched is recorded. A rejected seed is consumed like any other -- it is
    not silently retried or swapped, which is what makes the selection pre-registered
    rather than chosen.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    considered = 0
    exhausted = True

    for seed in seeds:
        considered += 1
        result = evaluate(cell, seed)
        row = {"seed": seed, **result}
        reason = rejection_reason(result)
        if reason is None:
            accepted.append(row)
            if len(accepted) == n_accept:
                exhausted = False
                break
        else:
            rejected.append({**row, "reason": reason})

    n_not_proven = sum(r["reason"] == "not_proven" for r in rejected)
    n_optimum_zero = sum(r["reason"] == "optimum_zero" for r in rejected)
    return {
        "cell": cell.as_dict(),
        "cell_key": cell.key,
        "accepted": accepted,
        "rejected": rejected,
        "n_considered": considered,
        "complete": len(accepted) == n_accept,
        "seed_range_exhausted": exhausted and len(accepted) < n_accept,
        "rates": {
            # Denominator is every seed considered in this cell, so the rates say how
            # expensive the cell is to fill, not just how many failed.
            "not_proven_rate": round(n_not_proven / considered, 4) if considered else None,
            "optimum_zero_rate": round(n_optimum_zero / considered, 4) if considered else None,
            "n_not_proven": n_not_proven,
            "n_optimum_zero": n_optimum_zero,
        },
    }


# --------------------------------------------------------------------------- #
# Hashing: a canonical payload WITHOUT timings, so a rerun reproduces the hash.
# --------------------------------------------------------------------------- #
_VOLATILE_KEYS = {"solve_seconds", "timestamp_utc", "wall_seconds", "content_hash"}


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in sorted(obj.items()) if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def content_hash(payload: dict[str, Any]) -> str:
    """SHA-256 over the reproducible core of a document."""
    canonical = json.dumps(_strip_volatile(payload), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Stage 1.
# --------------------------------------------------------------------------- #
def build_binning_manifest(
    grid: dict[str, list],
    grid_name: str,
    seed_start: int,
    seed_end: int,
    n_per_cell: int,
    evaluate: Callable[[Cell, int], dict[str, Any]] = evaluate_seed,
) -> dict[str, Any]:
    """Scan every cell, then bin the whole accepted pool into levels once."""
    cells = grid_cells(grid)
    t0 = time.perf_counter()
    scans = [scan_cell(c, range(seed_start, seed_end), n_per_cell, evaluate) for c in cells]
    wall = time.perf_counter() - t0

    pool = [{**row, "cell": s["cell"], "cell_key": s["cell_key"]}
            for s in scans for row in s["accepted"]]
    metrics = [row["complexity_metric"] for row in pool]
    b1, b2 = level_boundaries(metrics) if metrics else (0, 0)

    for row in pool:
        m = row["complexity_metric"]
        row["level"] = (Level.EASY if m <= b1 else Level.MEDIUM if m <= b2 else Level.HARD).value

    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "grid_name": grid_name,
        "grid": {
            "n_people": grid["n_people"],
            "tightness": grid["tightness"],
            "overlap": grid["overlap"],
            "travel_structure": [str(getattr(t, "value", t)) for t in grid["travel_structure"]],
        },
        "seed_range": {"start": seed_start, "end": seed_end},
        "n_per_cell": n_per_cell,
        "n_cells": len(cells),
        "n_accepted": len(pool),
        "boundaries": {"b1_easy_max": b1, "b2_medium_max": b2},
        "level_counts": {
            lvl.value: sum(row["level"] == lvl.value for row in pool) for lvl in Level
        },
        "cells": scans,
        "pool": pool,
        "incomplete_cells": [s["cell_key"] for s in scans if not s["complete"]],
    }
    payload["content_hash"] = content_hash(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_seconds"] = round(wall, 1)
    return payload


# --------------------------------------------------------------------------- #
# Stage 2.
# --------------------------------------------------------------------------- #
def select_pilot_subset(manifest: dict[str, Any], per_level: int = 10) -> dict[str, Any]:
    """Pick `per_level` instances per level, deterministically and with variety.

    Strata are (n_people, travel_structure). Visiting them round-robin stops the subset
    from filling up out of whichever cell sorts first, which a plain ascending-seed cut
    over the level would do -- the grid's variety would then be lost exactly where the
    LLM runs happen. Within a stratum, ascending seed; ties broken by the full cell key
    so the order cannot depend on dict iteration.
    """
    chosen: dict[str, list[dict[str, Any]]] = {}
    for level in (Level.EASY, Level.MEDIUM, Level.HARD):
        rows = [r for r in manifest["pool"] if r["level"] == level.value]
        strata: dict[tuple, list[dict[str, Any]]] = {}
        for r in rows:
            key = (r["cell"]["n_people"], r["cell"]["travel_structure"])
            strata.setdefault(key, []).append(r)
        for key in strata:
            strata[key].sort(key=lambda r: (r["seed"], r["cell_key"]))

        picked: list[dict[str, Any]] = []
        cursor = {k: 0 for k in strata}
        while len(picked) < per_level:
            progressed = False
            for key in sorted(strata):
                if len(picked) == per_level:
                    break
                i = cursor[key]
                if i < len(strata[key]):
                    picked.append(strata[key][i])
                    cursor[key] = i + 1
                    progressed = True
            if not progressed:
                break  # the level cannot supply per_level instances; reported below
        chosen[level.value] = picked

    payload: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "manifest_hash": manifest["content_hash"],
        "manifest_schema_version": manifest["schema_version"],
        "grid_name": manifest["grid_name"],
        "per_level_requested": per_level,
        "counts": {lvl: len(rows) for lvl, rows in chosen.items()},
        "short_levels": [lvl for lvl, rows in chosen.items() if len(rows) < per_level],
        "representation": {
            lvl: {
                "n_people": sorted({r["cell"]["n_people"] for r in rows}),
                "travel_structure": sorted({r["cell"]["travel_structure"] for r in rows}),
            }
            for lvl, rows in chosen.items()
        },
        "instances": {
            lvl: [
                {
                    "instance_id": r["instance_id"],
                    "seed": r["seed"],
                    "level": r["level"],
                    "optimum": r["optimum"],
                    "complexity_metric": r["complexity_metric"],
                    "cell": r["cell"],
                }
                for r in rows
            ]
            for lvl, rows in chosen.items()
        },
    }
    payload["content_hash"] = content_hash(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return payload


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _parse_seeds(text: str) -> tuple[int, int]:
    start, _, end = text.partition(":")
    return int(start), int(end)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("binning", help="stage 1: scan seeds, accept, bin levels")
    b.add_argument("--grid", choices=sorted(GRIDS), default="dev")
    b.add_argument("--seeds", default="0:100", help="half-open range START:END")
    b.add_argument("--n-per-cell", type=int, default=5)
    b.add_argument("--out", type=Path, default=Path("results/manifests"))
    b.add_argument("--dry-run", action="store_true",
                   help="print the summary and the hash, write nothing")

    s = sub.add_parser("subset", help="stage 2: pick the LLM subset from a manifest")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--per-level", type=int, default=10)
    s.add_argument("--out", type=Path, default=Path("results/manifests"))
    s.add_argument("--dry-run", action="store_true")
    return parser


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
    args = build_parser().parse_args(argv)

    if args.stage == "binning":
        start, end = _parse_seeds(args.seeds)
        payload = build_binning_manifest(
            GRIDS[args.grid], args.grid, start, end, args.n_per_cell
        )
        print(f"grid {args.grid}: {payload['n_cells']} cells x {args.n_per_cell} "
              f"= {payload['n_accepted']} accepted, seeds {start}:{end}")
        print(f"boundaries: easy <= {payload['boundaries']['b1_easy_max']}, "
              f"medium <= {payload['boundaries']['b2_medium_max']}")
        print(f"levels: {payload['level_counts']}")
        for scan in payload["cells"]:
            r = scan["rates"]
            print(f"  {scan['cell_key']:<28} considered {scan['n_considered']:>4} "
                  f"| not_proven {r['n_not_proven']:>3} ({r['not_proven_rate']}) "
                  f"| optimum_zero {r['n_optimum_zero']:>3} ({r['optimum_zero_rate']})"
                  + ("" if scan["complete"] else "  <-- INCOMPLETE"))
        # The technical failure criteria, evaluated rather than left to the reader.
        starved = [s["cell_key"] for s in payload["cells"] if s["seed_range_exhausted"]]
        unprovable = [s["cell_key"] for s in payload["cells"]
                      if (s["rates"]["not_proven_rate"] or 0) > 0.5]
        if starved:
            print(f"FAIL seed range exhausted before N: {starved}")
        if unprovable:
            print(f"FAIL solver could not prove the majority in: {unprovable}")
        if payload["incomplete_cells"] and not starved:
            print(f"INCOMPLETE cells: {payload['incomplete_cells']}")
        if not starved and not unprovable:
            print("technical failure criteria: none triggered")
        print(f"wall {payload['wall_seconds']}s | content_hash {payload['content_hash']}")
        if args.dry_run:
            print("dry run: nothing written")
            return 0
        print(f"written: {_write(payload, args.out, f'binning__{args.grid}')}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = select_pilot_subset(manifest, args.per_level)
    print(f"subset from manifest {payload['manifest_hash'][:12]}: {payload['counts']}")
    for lvl, rep in payload["representation"].items():
        print(f"  {lvl:<7} n_people {rep['n_people']} travel {rep['travel_structure']}")
    if payload["short_levels"]:
        print(f"SHORT levels (fewer than requested): {payload['short_levels']}")
    print(f"content_hash {payload['content_hash']}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    stem = "subset__" + str(payload["grid_name"])
    print(f"written: {_write(payload, args.out, stem)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
