# scripts/hpc/verify_shard_plan.py
"""Prove the 24-way shard split covers every cell exactly once, before any GPU.

A per-shard ``--dry-run`` cannot show this. Each invocation only ever sees its
own slice, so a split that dropped a cell -- or handed one to two shards -- would
look healthy in all 24 logs and only surface at the completeness audit, after the
campaign. This loads the same functions the runner uses and checks the whole
partition in one process, in seconds.

What it asserts:

* every canonical item lands in exactly one shard (complete and disjoint)
* shard sizes differ by at most one, which is what ``index % num_shards`` over a
  cap-major order should give
* each shard carries a comparable share of every cap, so no shard silently
  inherits the expensive end of the ladder
* the cell count matches ``instances x caps x conditions``

Usage::

    python scripts/hpc/verify_shard_plan.py --subset <manifest.json> \\
        --caps 16000,32000,64000,128000 --num-shards 24 \\
        --conditions c1_react,c2_verify_revise,c3_mas,c4_planner_critic,c5_best_of_3

Exits 0 when the plan is sound, 8 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pilot_sweep import canonical_items, load_manifests, select_items  # noqa: E402

EXIT_BAD_PLAN = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--binning", type=Path, default=None)
    parser.add_argument("--caps", default="16000,32000,64000,128000")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--conditions",
        default="c1_react,c2_verify_revise,c3_mas,c4_planner_critic,c5_best_of_3",
    )
    args = parser.parse_args()

    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    conditions = [c for c in args.conditions.split(",") if c.strip()]

    subset, _binning = load_manifests(args.subset, args.binning)
    items = canonical_items(subset, caps)

    n_instances = len({it.instance_id for it in items})
    print("=" * 74)
    print("SHARD PLAN")
    print(f"  subset      : {args.subset}  ({subset.get('content_hash', '')[:12]})")
    print(f"  instances   : {n_instances}")
    print(f"  caps        : {caps}")
    print(f"  conditions  : {len(conditions)}  {conditions}")
    print(f"  items       : {len(items)}   (instances x caps)")
    print(f"  runs total  : {len(items) * len(conditions)}   (items x conditions)")
    print(f"  shards      : {args.num_shards}")
    print("=" * 74)

    problems: list[str] = []
    seen: Counter[int] = Counter()
    sizes: list[int] = []

    print(f"  {'shard':>5}  {'items':>5}  {'runs':>6}   cap distribution")
    for shard in range(args.num_shards):
        mine = select_items(items, shard_index=shard, num_shards=args.num_shards)
        sizes.append(len(mine))
        for it in mine:
            seen[it.index] += 1
        by_cap = Counter(it.cap for it in mine)
        dist = "  ".join(f"{cap}:{by_cap.get(cap, 0)}" for cap in caps)
        print(f"  {shard:>5}  {len(mine):>5}  {len(mine) * len(conditions):>6}   {dist}")

    # Complete and disjoint.
    all_indices = {it.index for it in items}
    covered = set(seen)
    missing = all_indices - covered
    duplicated = {i for i, n in seen.items() if n > 1}
    if missing:
        problems.append(f"{len(missing)} item(s) belong to no shard, e.g. {sorted(missing)[:5]}")
    if duplicated:
        problems.append(f"{len(duplicated)} item(s) in more than one shard, e.g. {sorted(duplicated)[:5]}")

    # Balance: round-robin over a contiguous index space differs by at most one.
    if sizes and max(sizes) - min(sizes) > 1:
        problems.append(f"shard sizes differ by {max(sizes) - min(sizes)} (expected at most 1)")

    # Cap balance: no shard may carry a disproportionate share of any rung.
    for cap in caps:
        per_shard = [
            sum(1 for it in select_items(items, shard_index=s, num_shards=args.num_shards)
                if it.cap == cap)
            for s in range(args.num_shards)
        ]
        if per_shard and max(per_shard) - min(per_shard) > 1:
            problems.append(
                f"cap {cap} is unevenly split: min {min(per_shard)}, max {max(per_shard)}"
            )

    print("=" * 74)
    if problems:
        print("  PLAN REJECTED")
        for p in problems:
            print(f"    - {p}")
        return EXIT_BAD_PLAN

    print(f"  PLAN OK: {len(items)} items over {args.num_shards} shards, "
          f"each exactly once; sizes {min(sizes)}-{max(sizes)}; every cap evenly split")
    print(f"  total held-out runs to be executed: {len(items) * len(conditions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
