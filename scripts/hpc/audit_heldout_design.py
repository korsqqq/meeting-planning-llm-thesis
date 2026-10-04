# scripts/hpc/audit_heldout_design.py
"""Machine-checkable audit of the frozen held-out design, before any GPU time.

    python scripts/hpc/audit_heldout_design.py --num-shards 24

Reads the four frozen held-out manifests and checks, mechanically, that they are
the design THESIS_DECISIONS froze on 2026-08-26 and that the shard split over them
is complete and disjoint. Runs no experiment and contacts no endpoint.

Every check is a statement the design already makes. Nothing here decides anything:
if a count is wrong the right response is to stop, not to adjust a rule.

WHY AN AGGREGATE AUDIT. `verify_shard_plan.py` proves the split for ONE manifest.
The campaign runs four, sharded independently by the same `index % num_shards`, so
the union is what has to be complete and disjoint -- and a per-manifest proof
cannot see a collision between manifests, nor a shard that inherits the expensive
end of every one of them at once.

Exits 0 when the design is sound, 9 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.collect_structural_calibration import SEED_RANGES  # noqa: E402
from scripts.run_pilot_sweep import canonical_items, select_items  # noqa: E402

CAPS = (16000, 32000, 64000, 128000)
CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")

# The frozen design, restated here so the audit fails loudly if a manifest drifts.
BLOCK_A_PER_BAND = 50
BLOCK_A_O_HISTOGRAM = {3: 32, 4: 18}
BLOCK_B_PER_SIZE = 16
BLOCK_B_FIXED_OPTIMUM = 3
EXPECTED_INSTANCES = 3 * BLOCK_A_PER_BAND + 3 * BLOCK_B_PER_SIZE
EXPECTED_RUNS = EXPECTED_INSTANCES * len(CAPS) * len(CONDITIONS)
HELD_OUT_FLOOR = 100_000

EXIT_FAIL = 9


def entries(subset: dict) -> list[dict]:
    return [e for rows in subset["instances"].values() for e in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", type=Path, default=Path("results/manifests"))
    ap.add_argument("--num-shards", type=int, default=24)
    args = ap.parse_args()

    problems: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")
        if not ok:
            problems.append(message)

    # ----------------------------------------------------------------- load
    wanted = {
        "held_out_n8": ("bands_held_out_n8", 8),
        "held_out_n4": ("heldout_n4", 4),
        "held_out_n5": ("heldout_n5", 5),
        "held_out_n6": ("heldout_n6", 6),
    }
    subsets: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for range_name, (stem, _n) in wanted.items():
        found = sorted(args.manifests.glob(f"subset__{stem}__*.json"))
        if len(found) != 1:
            print(f"  FAIL  expected exactly one subset__{stem}__*.json, found {len(found)}")
            return EXIT_FAIL
        paths[range_name] = found[0]
        subsets[range_name] = json.loads(found[0].read_text(encoding="utf-8"))
        print(f"  loaded {found[0].name}")

    print("\n--- instances")
    all_ids: list[str] = []
    all_seeds: list[int] = []
    for range_name, subset in subsets.items():
        rows = entries(subset)
        all_ids += [e["instance_id"] for e in rows]
        all_seeds += [e["seed"] for e in rows]
        lo, hi = SEED_RANGES[range_name]
        stray = [e["seed"] for e in rows if not lo <= e["seed"] <= hi]
        check(not stray, f"{range_name}: every seed inside {lo}-{hi} ({len(stray)} stray)")

    check(len(all_ids) == EXPECTED_INSTANCES,
          f"{len(all_ids)} instances, expected {EXPECTED_INSTANCES}")
    check(len(set(all_ids)) == len(all_ids),
          f"instance ids unique ({len(all_ids) - len(set(all_ids))} duplicates)")
    check(len(set(all_seeds)) == len(all_seeds),
          f"seeds unique across all four manifests "
          f"({len(all_seeds) - len(set(all_seeds))} duplicates)")
    check(all(s >= HELD_OUT_FLOOR for s in all_seeds),
          f"no development seed below {HELD_OUT_FLOOR} "
          f"({sum(1 for s in all_seeds if s < HELD_OUT_FLOOR)} found)")

    print("\n--- block A (n = 8): band sizes and the frozen O histogram")
    a_rows = subsets["held_out_n8"]["instances"]
    for band in ("easy", "medium", "hard"):
        rows = a_rows.get(band, [])
        check(len(rows) == BLOCK_A_PER_BAND,
              f"band {band}: {len(rows)} instances, expected {BLOCK_A_PER_BAND}")
        hist = Counter(e["optimum"] for e in rows)
        check(dict(hist) == BLOCK_A_O_HISTOGRAM,
              f"band {band}: O histogram {dict(sorted(hist.items()))}, "
              f"expected {BLOCK_A_O_HISTOGRAM}")
        check(all(e["cell"]["n_people"] == 8 for e in rows), f"band {band}: every n_people = 8")

    print("\n--- block B (n = 4, 5, 6): size and the fixed optimum")
    for range_name, (_stem, n) in wanted.items():
        if n == 8:
            continue
        rows = entries(subsets[range_name])
        check(len(rows) == BLOCK_B_PER_SIZE,
              f"n = {n}: {len(rows)} instances, expected {BLOCK_B_PER_SIZE}")
        opts = {e["optimum"] for e in rows}
        check(opts == {BLOCK_B_FIXED_OPTIMUM},
              f"n = {n}: optima {sorted(opts)}, expected only {BLOCK_B_FIXED_OPTIMUM}")
        check(all(e["cell"]["n_people"] == n for e in rows), f"n = {n}: every n_people = {n}")

    print("\n--- expected runs")
    expected: set[tuple[str, int, str]] = set()
    duplicates = 0
    for subset in subsets.values():
        for item in canonical_items(subset, CAPS):
            for cond in CONDITIONS:
                key = (item.instance_id, item.cap, cond)
                if key in expected:
                    duplicates += 1
                expected.add(key)
    check(duplicates == 0, f"no duplicate (instance, cap, condition) cell ({duplicates} found)")
    check(len(expected) == EXPECTED_RUNS, f"{len(expected)} expected runs, wanted {EXPECTED_RUNS}")
    missing = {(i, c, k) for i in set(all_ids) for c in CAPS for k in CONDITIONS} - expected
    check(not missing, f"no missing (instance, cap, condition) cell ({len(missing)} found)")

    print(f"\n--- shard plan over all four manifests, num_shards = {args.num_shards}")
    seen: dict[tuple[str, int], int] = {}
    per_shard_runs: Counter[int] = Counter()
    per_shard_caps: dict[int, Counter[int]] = defaultdict(Counter)
    collisions = 0
    for subset in subsets.values():
        items = canonical_items(subset, CAPS)
        for shard in range(args.num_shards):
            for item in select_items(items, shard_index=shard, num_shards=args.num_shards):
                key = (item.instance_id, item.cap)
                if key in seen:
                    collisions += 1
                seen[key] = shard
                per_shard_runs[shard] += len(CONDITIONS)
                per_shard_caps[shard][item.cap] += 1

    check(collisions == 0, f"every (instance, cap) lands in exactly one shard "
                           f"({collisions} collisions)")
    check(len(seen) * len(CONDITIONS) == EXPECTED_RUNS,
          f"shards cover {len(seen) * len(CONDITIONS)} runs, wanted {EXPECTED_RUNS}")
    check(len(per_shard_runs) == args.num_shards,
          f"all {args.num_shards} shards non-empty ({len(per_shard_runs)} used)")

    lo_runs, hi_runs = min(per_shard_runs.values()), max(per_shard_runs.values())
    ideal = EXPECTED_RUNS / args.num_shards
    check(hi_runs - lo_runs <= len(CONDITIONS) * len(subsets),
          f"runs per shard {lo_runs}-{hi_runs} (ideal {ideal:.1f}), spread within "
          f"one item per manifest")
    for cap in CAPS:
        counts = [per_shard_caps[s][cap] for s in range(args.num_shards)]
        check(max(counts) - min(counts) <= len(subsets),
              f"cap {cap}: {min(counts)}-{max(counts)} instances per shard, "
              f"no shard inherits the expensive end")

    print()
    if problems:
        print(f"AUDIT FAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return EXIT_FAIL
    print(f"AUDIT PASSED -- {EXPECTED_INSTANCES} instances, {EXPECTED_RUNS} runs, "
          f"{args.num_shards} shards, {lo_runs}-{hi_runs} runs per shard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
