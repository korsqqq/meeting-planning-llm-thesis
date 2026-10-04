# scripts/hpc/merge_pool_blocks.py
"""Join structural-calibration pool blocks collected over disjoint seed ranges.

    python scripts/hpc/merge_pool_blocks.py \\
        --block results/calibration/heldout_n8_b0 \\
        --block results/calibration/heldout_n8_b1 \\
        --out results/calibration/heldout_n8

WHY THIS EXISTS. The collector is a single process and prices out at roughly five
candidates a second at `n = 8`. Block A needs a pool of 400 seeds, which is over
two hours in one process and about thirty-five minutes in four. The seed blocks
are disjoint by construction, the collector is deterministic per candidate, and
the selector reads only `candidates.csv`, so collecting in parallel and joining
here produces the same pool as one long run.

WHAT IT DOES NOT DO. It does not filter, deduplicate by structure, re-solve or
reorder anything except into the canonical order the collector itself walks. A
row is copied verbatim. If two blocks overlap on a seed, that is a collection
mistake and this refuses rather than picking a winner.

PROVENANCE. Each block's own `pool_meta.json` is kept verbatim under `blocks`, so
the git commit and candidate hash of every block survive the join. The merged meta
is marked as merged; nothing pretends this came from a single process. The selector
recomputes the candidate hash from the joined CSV itself, so its own provenance
record stays authoritative either way.

Exits 0 on success, 7 on any inconsistency between blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

EXIT_INCONSISTENT = 7


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", action="append", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    metas: list[dict] = []
    seen_seeds: dict[int, str] = {}

    for block in args.block:
        meta = json.loads((block / "pool_meta.json").read_text(encoding="utf-8"))
        with (block / "candidates.csv").open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if header is None:
                header = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != header:
                print(f"!!! {block}: column set differs from the first block")
                return EXIT_INCONSISTENT
            block_rows = list(reader)
        for r in block_rows:
            seed = int(r["seed"])
            owner = seen_seeds.get(seed)
            if owner is not None and owner != str(block):
                print(f"!!! seed {seed} appears in both {owner} and {block}; "
                      "the blocks are not disjoint")
                return EXIT_INCONSISTENT
            seen_seeds[seed] = str(block)
        n_people = {r["n_people"] for r in block_rows}
        if len(n_people) != 1:
            print(f"!!! {block}: mixed n_people {sorted(n_people)}")
            return EXIT_INCONSISTENT
        rows += block_rows
        meta["_block_dir"] = block.as_posix()
        metas.append(meta)
        print(f"  {block.name}: {len(block_rows)} candidates, "
              f"seeds {min(int(r['seed']) for r in block_rows)}-"
              f"{max(int(r['seed']) for r in block_rows)}")

    sizes = {m.get("n_people") for m in metas}
    if len(sizes) != 1:
        print(f"!!! blocks disagree on n_people: {sizes}")
        return EXIT_INCONSISTENT
    ranges = {m.get("seed_range_name") for m in metas}
    if len(ranges) != 1:
        print(f"!!! blocks disagree on seed_range_name: {ranges}")
        return EXIT_INCONSISTENT

    # The collector's own walk: seed, then travel structure, then tightness, then
    # overlap. Joining blocks in submission order would leave the file in an order
    # no single run would ever produce.
    rows.sort(key=lambda r: (int(r["seed"]), r["travel_structure"],
                             float(r["tightness"]), float(r["overlap"])))

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header or [])
        writer.writeheader()
        writer.writerows(rows)

    seeds = sorted(seen_seeds)
    merged = {
        "schema_version": metas[0].get("schema_version"),
        "merged": True,
        "merged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_people": metas[0].get("n_people"),
        "seed_range_name": metas[0].get("seed_range_name"),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "n_seeds": len(seeds),
        "n_completed": len(rows),
        "tightness": metas[0].get("tightness"),
        "overlap": metas[0].get("overlap"),
        "structures": metas[0].get("structures"),
        "candidates_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "blocks": metas,
    }
    (args.out / "pool_meta.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nmerged {len(args.block)} blocks -> {csv_path}")
    print(f"  {len(rows)} candidates, {len(seeds)} distinct seeds "
          f"({seeds[0]}-{seeds[-1]}), no overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
