# scripts/hpc/guard_seed_range.py
"""Refuse to proceed unless a manifest sits entirely in the seed range declared.

Filename checks are not a safeguard: a held-out manifest called ``test.json``
would pass one. This reads the manifest and inspects the instances themselves.

The partition is locked in ``THESIS_DECISIONS.md`` (LOCKED 2026-08-04)::

    0-9999        development, live smoke, ad-hoc calibration
    10000-19999   pilot: complexity binning, cap calibration, C3-8 slice
    100000-199999 held-out main experiment

The guard works in BOTH directions, which is what matters now that a held-out
array exists. ``--expect development`` catches held-out data reaching a
development job. ``--expect held_out`` catches the opposite and more dangerous
mistake: the held-out array pointed at development instances, which would run a
whole campaign of the wrong cells and look entirely healthy while doing it.

Usage::

    python scripts/hpc/guard_seed_range.py <manifest.json> \\
        [--expect development|held_out] [--expect-content-hash <prefix>]

Exits 0 when every seed is inside the declared range, 5 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HELD_OUT_SEED_FLOOR = 100_000

SEED_RANGES: dict[str, tuple[int, int]] = {
    "development": (0, HELD_OUT_SEED_FLOOR - 1),
    "held_out": (HELD_OUT_SEED_FLOOR, 199_999),
}

EXIT_VIOLATION = 5


def collect_instances(manifest: dict) -> list[dict] | None:
    """Return every instance record, whatever shape ``instances`` has.

    The pilot manifest keys instances by level; other manifests use a flat
    list. Both are accepted. Any other shape returns ``None`` so the caller
    fails loudly -- guessing here could produce an empty list, and an empty
    list would sail through the seed check.
    """
    raw = manifest.get("instances")
    if isinstance(raw, dict):
        out: list[dict] = []
        for level_instances in raw.values():
            out.extend(level_instances)
        return out
    if isinstance(raw, list):
        return raw
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expect", choices=sorted(SEED_RANGES), default="development",
                        help="which locked seed range every instance must fall in")
    parser.add_argument(
        "--expect-content-hash",
        default=None,
        help="require the manifest's content_hash to start with this prefix",
    )
    args = parser.parse_args()

    low, high = SEED_RANGES[args.expect]

    if not args.manifest.is_file():
        print(f"GUARD FAILED: manifest not found: {args.manifest}")
        return EXIT_VIOLATION

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instances = collect_instances(manifest)

    if instances is None:
        print(
            "GUARD FAILED: manifest has no usable 'instances' field "
            f"(got {type(manifest.get('instances')).__name__}); refusing to guess"
        )
        return EXIT_VIOLATION
    if not instances:
        print("GUARD FAILED: manifest contains zero instances")
        return EXIT_VIOLATION

    seeds = [i["seed"] for i in instances if "seed" in i]
    if len(seeds) != len(instances):
        print(
            f"GUARD FAILED: {len(instances) - len(seeds)} instance(s) carry no seed; "
            "cannot prove which range they belong to"
        )
        return EXIT_VIOLATION

    content_hash = manifest.get("content_hash", "")
    offenders = sorted(
        (i["instance_id"], i["seed"])
        for i in instances
        if not low <= i["seed"] <= high
    )

    print(f"  manifest      : {args.manifest}")
    print(f"  content_hash  : {content_hash}")
    print(f"  instances     : {len(instances)}")
    print(f"  seed range    : {min(seeds)} .. {max(seeds)}")
    print(f"  expected      : {args.expect}  ({low} .. {high})")

    if offenders:
        print(f"GUARD FAILED: {len(offenders)} instance(s) outside the {args.expect} range:")
        for instance_id, seed in offenders[:10]:
            print(f"    {instance_id}  seed={seed}")
        return EXIT_VIOLATION

    if args.expect_content_hash and not content_hash.startswith(args.expect_content_hash):
        print(
            f"GUARD FAILED: content_hash does not start with the expected "
            f"{args.expect_content_hash!r}"
        )
        return EXIT_VIOLATION

    print(f"  GUARD PASSED  : every instance is in the locked {args.expect} range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
