# scripts/build_probe_subset.py
"""Cut the frozen 12-instance budget probe subset out of the budget-dev selection. CPU only.

    python -m scripts.build_probe_subset \
        --parent results/manifests/subset__bands_budget_dev__911e4864d6fb.json \
        --out results/manifests

This **selects, it does not generate**. Every entry is copied verbatim from the parent
manifest: same instance ids, same seeds, same band labels, same recorded optimum and
complexity metric. Nothing is regenerated, re-solved or altered, so the probe runs on
literally the same task objects the 64 000 arm already ran on -- which is what makes the
comparison paired within instance.

*The selection rule, fixed before any 128 000 outcome exists.* Per band, the first two
instances with oracle optimum 3 and the first two with optimum 4, in the parent manifest's
own canonical order. It reads only position and structural facts.

*Why the optimum is balanced 2:2 rather than left to fall where it may.* Satisfaction is
`achieved / O`, so one meeting is worth 0.333 at `O = 3` and 0.25 at `O = 4`. A band that
came out all `O = 3` next to one that came out mostly `O = 4` would measure the same
improvement on different scales, and the per-band rows of a paired table would not be
comparable. Matching on the optimum is also the discipline the complexity bands themselves
were frozen under (§3), so the probe keeps it rather than inventing a looser rule.

**Forbidden inputs, and the reason.** The rule may not read satisfaction, feasibility,
termination, token usage or any other agent outcome from the completed 64 000 runs.
Selecting the probe set by how C1 or C3 already did on it would make the 64 000 arm of the
paired comparison a chosen baseline rather than a measured one. A test asserts this file
never mentions such a quantity.

The emitted subset keeps the parent's `manifest_hash`, so the sweep runner locates the same
binning manifest and re-derives every band label from the same frozen boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_pilot_manifest import content_hash  # noqa: E402

SUBSET_SCHEMA_VERSION = "pilot_subset/1.0"
SELECTOR_VERSION = "probe_subset/1.0"

LEVELS = ("easy", "medium", "hard")
BAND_OF_LEVEL = {"easy": "low", "medium": "medium", "hard": "high"}
# Two per optimum value, per band: 4 instances a band, 12 in total.
OPTIMA = (3, 4)
PER_OPTIMUM = 2
DEV_SEED_RANGE = (30000, 39999)


class ProbeSelectionError(RuntimeError):
    """The parent manifest cannot supply the probe set as specified."""


def select(parent: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The rule, applied to the parent's own order. Copies entries, never builds them."""
    out: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        rows = parent["instances"][level]
        chosen: list[dict[str, Any]] = []
        for optimum in OPTIMA:
            matching = [e for e in rows if e["optimum"] == optimum]
            if len(matching) < PER_OPTIMUM:
                raise ProbeSelectionError(
                    f"{level}: only {len(matching)} instance(s) with optimum {optimum}, "
                    f"need {PER_OPTIMUM}. The parent cannot supply a 2:2 split and the "
                    "rule is not relaxed to make it fit.")
            chosen.extend(matching[:PER_OPTIMUM])
        out[level] = chosen
    return out


def build(parent: dict[str, Any], cap: int) -> dict[str, Any]:
    chosen = select(parent)
    payload: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "purpose": ("frozen 12-instance subset for the exploratory 128 000 budget probe; "
                    "entries copied verbatim from the parent, nothing regenerated"),
        # Kept so the sweep runner finds the SAME binning manifest and re-derives every
        # band label from the same frozen boundaries.
        "manifest_hash": parent["manifest_hash"],
        "manifest_schema_version": parent["manifest_schema_version"],
        "grid_name": parent["grid_name"],
        "seed_range_name": parent.get("seed_range_name"),
        "parent_subset_hash": parent["content_hash"],
        "probe_cap": cap,
        "selection_rule": (
            f"per band, the first {PER_OPTIMUM} instances with oracle optimum 3 and the "
            f"first {PER_OPTIMUM} with optimum 4, in the parent manifest's canonical "
            "order; position and structural facts only"),
        "forbidden_inputs": [
            "satisfaction", "feasibility", "termination", "token usage",
            "any other agent outcome from the completed 64 000 runs",
        ],
        "per_level_requested": len(OPTIMA) * PER_OPTIMUM,
        "counts": {lv: len(rows) for lv, rows in chosen.items()},
        "short_levels": [lv for lv, rows in chosen.items()
                         if len(rows) < len(OPTIMA) * PER_OPTIMUM],
        "representation": {
            lv: {
                "n_people": sorted({e["cell"]["n_people"] for e in rows}),
                "travel_structure": sorted({e["cell"]["travel_structure"] for e in rows}),
                "tightness": sorted({e["cell"]["tightness"] for e in rows}),
                "optimum_counts": {str(k): v for k, v in
                                   sorted(Counter(e["optimum"] for e in rows).items())},
                "conflict_pairs": sorted({e["complexity_metric"] for e in rows}),
            }
            for lv, rows in chosen.items()
        },
        "instances": chosen,
    }
    payload["content_hash"] = content_hash(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return payload


def audit(subset: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Every invariant the probe was specified under, checked on the emitted document."""
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    rows = [e for lv in LEVELS for e in subset["instances"][lv]]
    ids = [e["instance_id"] for e in rows]
    seeds = [e["seed"] for e in rows]
    expected = len(LEVELS) * len(OPTIMA) * PER_OPTIMUM

    check(len(rows) == expected, f"{len(rows)} instances, expected {expected}")
    check(len(set(ids)) == len(ids), f"{len(ids) - len(set(ids))} duplicate instance id(s)")
    check(len(set(seeds)) == len(seeds), "a seed is used more than once")
    for lv in LEVELS:
        n = len(subset["instances"][lv])
        check(n == len(OPTIMA) * PER_OPTIMUM,
              f"{lv}: {n} instances, expected {len(OPTIMA) * PER_OPTIMUM}")
        hist = Counter(e["optimum"] for e in subset["instances"][lv])
        check(all(hist.get(o, 0) == PER_OPTIMUM for o in OPTIMA),
              f"{lv}: optimum histogram {dict(hist)}, expected "
              f"{{{OPTIMA[0]}: {PER_OPTIMUM}, {OPTIMA[1]}: {PER_OPTIMUM}}}")

    lo, hi = DEV_SEED_RANGE
    stray = sorted(s for s in seeds if not lo <= s <= hi)
    check(not stray, f"{len(stray)} seed(s) outside the development range {lo}-{hi}: "
                     f"{stray[:8]}")
    check(not any(s >= 100000 for s in seeds), "a held-out seed reached the probe subset")

    # Every entry must be byte-identical to the parent's: selection, never construction.
    parent_by_id = {e["instance_id"]: e for lv in LEVELS for e in parent["instances"][lv]}
    for e in rows:
        original = parent_by_id.get(e["instance_id"])
        if original is None:
            problems.append(f"{e['instance_id']} is not in the parent manifest")
        elif original != e:
            problems.append(f"{e['instance_id']} differs from the parent entry")

    check(subset["manifest_hash"] == parent["manifest_hash"],
          "the probe subset points at a different binning manifest than the parent")
    check(content_hash(subset) == subset["content_hash"],
          "content hash does not reproduce")

    if problems:
        raise ProbeSelectionError("probe subset failed its audit:\n  - "
                                  + "\n  - ".join(problems))
    return {
        "n_instances": len(rows),
        "per_level": {lv: len(subset["instances"][lv]) for lv in LEVELS},
        "per_band": {BAND_OF_LEVEL[lv]: len(subset["instances"][lv]) for lv in LEVELS},
        "distinct_instance_ids": len(set(ids)),
        "distinct_seeds": len(set(seeds)),
        "seed_range_observed": [min(seeds), max(seeds)],
        "all_seeds_in_development_range": True,
        "held_out_seeds_present": False,
        "entries_identical_to_parent": True,
        "parent_subset_hash": parent["content_hash"],
        "binning_manifest_hash": subset["manifest_hash"],
        "all_checks_passed": True,
    }


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
    ap.add_argument("--parent", type=Path,
                    default=Path("results/manifests/"
                                 "subset__bands_budget_dev__911e4864d6fb.json"))
    ap.add_argument("--cap", type=int, default=128000)
    ap.add_argument("--out", type=Path, default=Path("results/manifests"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    if content_hash(parent) != parent["content_hash"]:
        raise SystemExit(f"{args.parent} has been modified: its content hash does not "
                         "reproduce, so nothing may be cut from it")

    subset = build(parent, args.cap)
    report = audit(subset, parent)

    print(f"parent  : {args.parent.name}  ({parent['content_hash'][:12]})")
    print(f"rule    : {subset['selection_rule']}")
    for lv in LEVELS:
        rep = subset["representation"][lv]
        print(f"  {BAND_OF_LEVEL[lv]:<7} n {subset['counts'][lv]}  "
              f"O {rep['optimum_counts']}  k {rep['conflict_pairs']}  "
              f"tightness {rep['tightness']}  travel {rep['travel_structure']}")
        for e in subset["instances"][lv]:
            print(f"      {e['instance_id']:<32} seed {e['seed']}  "
                  f"k {e['complexity_metric']:>2}  O {e['optimum']}")
    print(f"audit   : {report['n_instances']} instances, "
          f"{report['distinct_seeds']} distinct seeds, "
          f"range {report['seed_range_observed'][0]}-{report['seed_range_observed'][1]}, "
          "entries identical to the parent, no held-out seed")
    print(f"hash    : {subset['content_hash']}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    print(f"written : {_write(subset, args.out, 'subset__budget_probe_128k')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
