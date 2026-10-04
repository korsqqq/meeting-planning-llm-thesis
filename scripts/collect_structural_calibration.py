# scripts/collect_structural_calibration.py
"""Phase 1 of the revised complexity axis: structural calibration collector. CPU only.

    python -m scripts.collect_structural_calibration --out results/calibration/structural

This walks a candidate pool at a fixed `n_people` over the generator's knobs and records,
for every instance, the structural facts the amendment will need. It **collects and does
not decide**: no band boundary, no admissibility threshold and no exclusion rule is applied
or computed here. Deciding what counts as Low, Medium or High while the pool is still being
built would make the boundary a function of whatever the walk happened to reach first.
Distributions are read afterwards by a separate analysis step, and only then are the bands
frozen in a dated amendment to THESIS_DECISIONS section 3.

**No LLM is involved at any point.** The whole pipeline here is the generator, CP-SAT, the
conflict graph and plain combinatorics over subsets. Selecting instances by running an
agent on them would tune the complexity axis to observed model behaviour and make the
design circular; section 3 states that rule as absolute. A regression test prevents this
file from directly importing or referencing the agent and client stack -- that is a guard
against the dependency reappearing, not a proof of absence, since a transitive import could
in principle reintroduce it. What actually establishes it is the import graph and the way
the collector is run.

Seed range: one of the **named** ranges reserved in section 3, defaulting to structural
calibration (20000-29999). Every named range is disjoint from the pilot's consumed range,
and the held-out main range is not among them at all, so no `--seed-start` -- stray or
deliberate -- can consume a held-out instance through this script.

At `n = 8` every graph quantity below is exhaustive over 2**8 = 256 subsets, so nothing
here is approximated or sampled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.generator import generate_instance  # noqa: E402
from src.oracle.solver import conflict_graph, solve  # noqa: E402
from src.schemas import Instance, TravelStructure  # noqa: E402

# The seed ranges reserved in THESIS_DECISIONS section 3 that this collector may walk.
SEED_RANGES: dict[str, tuple[int, int]] = {
    "structural_calibration": (20000, 29999),
    "budget_dev": (30000, 39999),
    # Lower-complexity calibration (2026-08-16). One reserved range per `n`, never shared:
    # a pool is identified by its seeds alone in every downstream artifact, so two sizes
    # drawing on one range would make `instance_id` collisions possible and would stop the
    # seed from identifying which pool a candidate came from.
    "lower_complexity_n4": (40000, 49999),
    "lower_complexity_n5": (50000, 59999),
    "lower_complexity_n6": (60000, 69999),
    # Held-out main. This entry was deliberately absent until the decision existed: the
    # note here used to say that opening 100000-199999 needs a decision recorded in the
    # document rather than a command-line argument. That decision is the amendment of
    # 2026-08-26, "HELD-OUT DATASET DESIGN FROZEN", which partitions the locked range into
    # one reserved block per pool and fixes the sample sizes and the O histogram before any
    # pool exists. The blocks below are that table, copied verbatim; 140000-199999 stays
    # unused reserve and is deliberately still not offered here.
    "held_out_n8": (100_000, 109_999),
    "held_out_n4": (110_000, 119_999),
    "held_out_n5": (120_000, 129_999),
    "held_out_n6": (130_000, 139_999),
}
DEFAULT_RANGE = "structural_calibration"

CALIBRATION_SEED_MIN, CALIBRATION_SEED_MAX = SEED_RANGES[DEFAULT_RANGE]

DEFAULT_N_PEOPLE = 8
# The knob grid. Deliberately wide: the pilot showed conflicts appear only in the upper
# tightness region, so a narrow grid would hand back a pool with no high-density instances
# and the appearance of an impossible axis.
DEFAULT_TIGHTNESS = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
DEFAULT_OVERLAP = (0.2, 0.5, 0.8, 1.0)
DEFAULT_STRUCTURES = tuple(t.value for t in TravelStructure)

CSV_SCHEMA_VERSION = "structural_calibration/1.0"


class CalibrationRangeError(RuntimeError):
    """A requested seed lies outside the reserved calibration range."""


# --------------------------------------------------------------------------- #
# Graph quantities. Exhaustive at n = 8; none of this is sampled.
# --------------------------------------------------------------------------- #
def independence_number(vertices: Sequence[str], edges: set[frozenset[str]]) -> tuple[int, int]:
    """(alpha, number of maximum independent sets) by exhaustive subset enumeration.

    `vertices` must already be restricted to the individually reachable people. An
    unreachable person has no conflict edges, so leaving them in would let them join every
    independent set and inflate alpha -- which would then make `alpha - optimum` measure
    unreachability rather than higher-order interaction.
    """
    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    masks = [0] * n
    for e in edges:
        a, b = tuple(e)
        if a in idx and b in idx:
            masks[idx[a]] |= 1 << idx[b]
            masks[idx[b]] |= 1 << idx[a]

    best, count = 0, 0
    for subset in range(1 << n):
        ok = True
        s = subset
        while s:
            low = s & -s
            i = low.bit_length() - 1
            if masks[i] & subset:
                ok = False
                break
            s ^= low
        if not ok:
            continue
        size = bin(subset).count("1")
        if size > best:
            best, count = size, 1
        elif size == best:
            count += 1
    return best, count


def triangle_violations(instance: Instance) -> tuple[int, int]:
    """(violating ordered triples, triples checked) for travel(a,c) > travel(a,b)+travel(b,c).

    `RANDOM` is specified as an arbitrary asymmetric matrix with no triangle inequality, so
    violations there are expected; the count exists to confirm the implementation matches
    that specification and to explain any negative higher-order gap.

    `travel_times` is a nested mapping `from -> to -> minutes`, not a flat tuple-keyed dict.
    Indexing it the wrong way returns None for every lookup and the function then reports
    zero violations without having compared anything, so the number of triples actually
    examined is returned alongside the count and the caller refuses a zero denominator.
    """
    t = instance.travel_times
    locs = sorted(t)
    bad = total = 0
    for a in locs:
        row_a = t.get(a) or {}
        for b in locs:
            if b == a:
                continue
            ab = row_a.get(b)
            row_b = t.get(b) or {}
            for c in locs:
                if c in (a, b):
                    continue
                ac, bc = row_a.get(c), row_b.get(c)
                if ab is None or ac is None or bc is None:
                    continue
                total += 1
                if ac > ab + bc:
                    bad += 1
    return bad, total


# --------------------------------------------------------------------------- #
# One candidate.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Row:
    instance_id: str
    seed: int
    n_people: int
    tightness: float
    overlap: float
    travel_structure: str
    individually_reachable_count: int
    conflicting_pairs: int
    D: float                      # over C(n_people, 2), as defined in section 3
    D_reachable: float | None     # over C(reachable, 2); None when reachable < 2
    oracle_optimum: int
    optimum_over_n: float
    proven_optimal: bool
    solver_status: str
    alpha_reachable: int
    higher_order_gap_H: int
    n_max_independent_sets: int
    max_degree: int
    edge_share_top1: float | None
    edge_share_top2: float | None
    largest_component: int
    triangle_violations: int
    triangle_triples_checked: int
    solve_seconds: float


def _components(vertices: Sequence[str], edges: set[frozenset[str]]) -> int:
    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for e in edges:
        a, b = tuple(e)
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    largest = 0
    for v in vertices:
        if v in seen:
            continue
        stack, size = [v], 0
        seen.add(v)
        while stack:
            u = stack.pop()
            size += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        largest = max(largest, size)
    return largest


def measure(instance: Instance) -> Row:
    t0 = time.perf_counter()
    solution = solve(instance)
    solve_seconds = time.perf_counter() - t0

    edges = conflict_graph(instance)
    endpoints = {p for e in edges for p in e}
    # `conflict_graph` builds edges only over individually reachable people, but a
    # reachable person with no conflicts leaves no trace in the edge set, so reachability
    # is recomputed rather than inferred from the endpoints.
    from src.oracle.solver import _earliest_schedule  # noqa: PLC0415  (internal, CPU-only)
    reachable = sorted(
        p.person_id for p in instance.people
        if _earliest_schedule(instance, [p.person_id]) is not None
    )
    assert endpoints <= set(reachable), "conflict edge outside the reachable set"

    alpha, n_alpha_sets = independence_number(reachable, edges)
    degrees = {v: 0 for v in reachable}
    for e in edges:
        for p in e:
            degrees[p] += 1
    ordered = sorted(degrees.values(), reverse=True) or [0]
    n_edges = len(edges)
    bad, checked = triangle_violations(instance)
    if checked == 0:
        # A zero here means the travel matrix was walked incorrectly, not that the matrix
        # is metric. Reporting "no violations" off an empty denominator is exactly the
        # failure this project refuses elsewhere.
        raise ValueError(
            f"{instance.instance_id}: the triangle-inequality check examined no triples; "
            "the travel matrix was not traversed correctly"
        )
    gp = instance.generator_params
    n = gp.n_people
    n_reach = len(reachable)

    return Row(
        instance_id=instance.instance_id,
        seed=instance.seed,
        n_people=n,
        tightness=gp.tightness,
        overlap=gp.overlap,
        travel_structure=gp.travel_structure.value,
        individually_reachable_count=n_reach,
        conflicting_pairs=n_edges,
        D=round(n_edges / (n * (n - 1) / 2), 6),
        D_reachable=(round(n_edges / (n_reach * (n_reach - 1) / 2), 6)
                     if n_reach >= 2 else None),
        oracle_optimum=solution.optimum,
        optimum_over_n=round(solution.optimum / n, 6),
        proven_optimal=solution.proven_optimal,
        solver_status=solution.status,
        alpha_reachable=alpha,
        higher_order_gap_H=alpha - solution.optimum,
        n_max_independent_sets=n_alpha_sets,
        max_degree=ordered[0],
        edge_share_top1=round(ordered[0] / n_edges, 6) if n_edges else None,
        edge_share_top2=(round(sum(ordered[:2]) / n_edges, 6) if n_edges else None),
        largest_component=_components(reachable, edges),
        triangle_violations=bad,
        triangle_triples_checked=checked,
        solve_seconds=round(solve_seconds, 4),
    )


# --------------------------------------------------------------------------- #
# The walk.
# --------------------------------------------------------------------------- #
def candidates(
    *, n_people: int, seeds: Sequence[int], tightness: Sequence[float],
    overlap: Sequence[float], structures: Sequence[str],
) -> Iterator[tuple[int, float, float, str]]:
    """Seed-major so a truncated run still covers the whole knob grid evenly."""
    for seed in seeds:
        for structure in structures:
            for t in tightness:
                for o in overlap:
                    yield (seed, t, o, structure)


class PoolAuditError(RuntimeError):
    """The collected pool is not the pool that was asked for."""


def _git_commit() -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def audit_pool(rows: Sequence[Row], *, expected: int, n_people: int,
               seeds: Sequence[int]) -> dict[str, Any]:
    """Prove the pool is complete and uniform before any distribution is read from it.

    Same principle as the formal pilot's completeness audit: a pool that is silently short,
    duplicated, or mixed would still yield a perfectly readable distribution and the bands
    cut from it would mean nothing. This runs AFTER the CSV is written, so a failure costs
    the audit and not the hour of CPU.
    """
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    check(len(rows) == expected, f"expected {expected} candidates, collected {len(rows)}")
    ids = {r.instance_id for r in rows}
    check(len(ids) == len(rows), f"{len(rows) - len(ids)} duplicate instance_id(s)")
    keys = {(r.seed, r.tightness, r.overlap, r.travel_structure) for r in rows}
    check(len(keys) == len(rows),
          f"{len(rows) - len(keys)} duplicate (seed, tightness, overlap, structure) key(s)")

    bad_n = {r.n_people for r in rows} - {n_people}
    check(not bad_n, f"n_people is not uniform: {sorted(bad_n)}")
    lo, hi = min(seeds), max(seeds)
    out_of_range = [r.seed for r in rows if not lo <= r.seed <= hi]
    check(not out_of_range, f"{len(out_of_range)} row(s) outside the requested seed range")

    unproven = [r.instance_id for r in rows if not r.proven_optimal]
    check(not unproven,
          f"{len(unproven)} unproven optimum/optima -- the n<=9 tractability assumption "
          f"did not hold; report, never silently drop: {unproven[:5]}")
    statuses = {r.solver_status for r in rows}
    check(statuses <= {"OPTIMAL"}, f"solver statuses beyond OPTIMAL: {sorted(statuses)}")

    triples = {r.triangle_triples_checked for r in rows}
    check(len(triples) == 1 and 0 not in triples,
          f"triangle-inequality check examined {sorted(triples)} triples; at fixed n it "
          "must be one positive constant, and zero means the matrix was never traversed")

    forbidden = {"level", "band", "D_band", "admitted", "excluded"}
    present = forbidden & {f.name for f in Row.__dataclass_fields__.values()}
    check(not present, f"the collector must decide nothing, but emits {sorted(present)}")

    if problems:
        raise PoolAuditError("structural pool failed its completeness audit:\n  - "
                             + "\n  - ".join(problems))
    return {
        "n_candidates": len(rows),
        "n_expected": expected,
        "distinct_instance_ids": len(ids),
        "distinct_knob_keys": len(keys),
        "seed_range_observed": [lo, hi],
        "unproven_optima": 0,
        "triangle_triples_per_instance": triples.pop(),
        "all_checks_passed": True,
    }


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/calibration/structural"))
    ap.add_argument("--n-people", type=int, default=DEFAULT_N_PEOPLE)
    ap.add_argument("--range", dest="seed_range", choices=sorted(SEED_RANGES),
                    default=DEFAULT_RANGE,
                    help="which reserved seed range to walk")
    ap.add_argument("--seed-start", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--tightness", default=",".join(str(x) for x in DEFAULT_TIGHTNESS))
    ap.add_argument("--overlap", default=",".join(str(x) for x in DEFAULT_OVERLAP))
    ap.add_argument("--structures", default=",".join(DEFAULT_STRUCTURES))
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args(argv)

    lo, hi = SEED_RANGES[args.seed_range]
    seed_start = lo if args.seed_start is None else args.seed_start
    seeds = list(range(seed_start, seed_start + args.n_seeds))
    if seeds[0] < lo or seeds[-1] > hi:
        raise CalibrationRangeError(
            f"seeds {seeds[0]}-{seeds[-1]} leave the reserved {args.seed_range} range "
            f"{lo}-{hi} (THESIS_DECISIONS section 3). Held-out main is not reachable from "
            "this script at all: opening it is a decision recorded in the document, not a "
            "command-line argument."
        )

    tightness = [float(x) for x in args.tightness.split(",") if x]
    overlap = [float(x) for x in args.overlap.split(",") if x]
    structures = [x for x in args.structures.split(",") if x]

    grid = list(candidates(n_people=args.n_people, seeds=seeds, tightness=tightness,
                           overlap=overlap, structures=structures))
    print(f"candidates: {len(grid)} "
          f"({len(seeds)} seeds x {len(structures)} structures x {len(tightness)} tightness "
          f"x {len(overlap)} overlap), n_people={args.n_people}")

    rows: list[Row] = []
    t_start = time.perf_counter()
    for i, (seed, t, o, structure) in enumerate(grid, 1):
        inst = generate_instance(n_people=args.n_people, tightness=t, overlap=o,
                                 travel_structure=TravelStructure(structure), seed=seed)
        rows.append(measure(inst))
        if args.progress_every and i % args.progress_every == 0:
            rate = i / (time.perf_counter() - t_start)
            print(f"  [{i}/{len(grid)}] {rate:.1f}/s "
                  f"eta {int((len(grid) - i) / max(rate, 1e-9)) // 60} min", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in Row.__dataclass_fields__.values()]
    buf = []
    import io
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(asdict(r))
    write_atomic(args.out / "candidates.csv", sio.getvalue())

    meta: dict[str, Any] = {
        "schema_version": CSV_SCHEMA_VERSION,
        "purpose": "structural calibration pool; collects, decides nothing",
        "git_commit": _git_commit(),
        "n_people": args.n_people,
        "seed_range_name": args.seed_range,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "reserved_range": [lo, hi],
        "tightness": tightness,
        "overlap": overlap,
        "structures": structures,
        "n_expected": len(grid),
        "n_completed": len(rows),
        "n_not_proven": sum(1 for r in rows if not r.proven_optimal),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    print(f"written: {args.out}/candidates.csv ({len(rows)} rows)")

    try:
        meta["audit"] = audit_pool(rows, expected=len(grid), n_people=args.n_people,
                                   seeds=seeds)
        audit_ok = True
    except PoolAuditError as exc:
        meta["audit"] = {"all_checks_passed": False, "error": str(exc)}
        audit_ok = False

    write_atomic(args.out / "pool_meta.json",
                 json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"written: {args.out}/pool_meta.json")

    if not audit_ok:
        print()
        print(meta["audit"]["error"])
        print("the CSV is kept; fix the cause and rerun rather than reading these "
              "distributions")
        return 1

    print(f"audit passed — {len(rows)} candidates, "
          f"{meta['audit']['triangle_triples_per_instance']} triples checked per instance, "
          "0 unproven optima")
    print("bands are NOT decided here: read the distributions with a separate step, then "
          "freeze them in a dated amendment before touching the held-out range")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
