# scripts/check_band_admissibility.py
"""Admissibility of a PROPOSED band design against the frozen calibration pool. CPU only.

    python -m scripts.check_band_admissibility \
        --optimum 4 --higher-order-gap 0 \
        --band "Low=1-4" --band "Medium=7-10" --band "High=13-16"

The bands are **inputs**, not outputs. This script evaluates a design somebody proposed
against the criteria written in THESIS_DECISIONS section 3; it does not search for bands,
rank alternatives, or suggest a repair when one fails. That separation is the point: a
script that could widen a failing band until it passed would be choosing the design from
its own outcome, and the failure path is a recorded fallback rather than an adjustment.

Criteria, all fixed before this ran:

  1. at least 50 calibration candidates in each band  (evidence of support, not the sample)
  2. at least two distinct tightness values in each band
  3. the most frequent tightness accounts for at most 60% of each band
  4. at least two distinct travel structures in each band
  5. a common stratified allocation exists: 50 instances per band can be drawn so that the
     joint (tightness, travel structure) distribution is identical across the three bands

Criterion 5 is the one the others do not cover. Bands can each pass the diversity rules
separately and still differ from one another in composition, which would reintroduce the
confound between bands rather than inside them. The check asks whether the strata common
to all three bands hold enough candidates for an identical allocation.

**The calibration pool is support evidence, not the sample.** The held-out instances are
generated later from the main seed range; these counts show that the design is structurally
populated, and the actual manifest is built separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyse_structural_calibration import MAX_PAIRS, load_pool  # noqa: E402

SCHEMA_VERSION = "band_admissibility/1.0"

MIN_CANDIDATES = 50
MIN_TIGHTNESS_VALUES = 2
MAX_DOMINANT_TIGHTNESS_SHARE = 0.60
MIN_TRAVEL_STRUCTURES = 2
TARGET_PER_BAND = 50


class BandSpecError(ValueError):
    """A band specification could not be parsed, or the bands overlap."""


def parse_band(spec: str) -> tuple[str, range]:
    """`"Low=1-4"` -> `("Low", range(1, 5))`, inclusive on both ends."""
    if "=" not in spec:
        raise BandSpecError(f"band {spec!r} must look like Name=lo-hi")
    name, rng = spec.split("=", 1)
    if "-" not in rng:
        raise BandSpecError(f"band {spec!r} must give an inclusive range lo-hi")
    lo_s, hi_s = rng.split("-", 1)
    lo, hi = int(lo_s), int(hi_s)
    if not 0 <= lo <= hi <= MAX_PAIRS:
        raise BandSpecError(f"band {spec!r} outside 0..{MAX_PAIRS}")
    return name.strip(), range(lo, hi + 1)


def common_allocation(bands: dict[str, list[dict[str, Any]]],
                      target: int = TARGET_PER_BAND) -> dict[str, Any]:
    """Can `target` instances per band be drawn with an identical knob distribution?

    Strata are (tightness, travel structure). A stratum is usable only if every band has at
    least one candidate in it; the number drawable from it, identically in all bands, is the
    minimum across bands. If those minima sum to `target` or more, an identical allocation
    exists.
    """
    per_band: dict[str, Counter] = {
        name: Counter((r["tightness"], r["travel_structure"]) for r in rows)
        for name, rows in bands.items()
    }
    shared = set.intersection(*(set(c) for c in per_band.values())) if per_band else set()
    minima = {s: min(c[s] for c in per_band.values()) for s in shared}
    capacity = sum(minima.values())

    # A deterministic allocation: fill the shared strata in a fixed order until the target
    # is reached, so the same pool always yields the same plan.
    allocation: dict[str, int] = {}
    remaining = target
    for s in sorted(minima, key=lambda k: (-minima[k], str(k))):
        if remaining <= 0:
            break
        take = min(minima[s], remaining)
        allocation[f"{s[0]}|{s[1]}"] = take
        remaining -= take

    return {
        "target_per_band": target,
        "shared_strata": len(shared),
        "capacity": capacity,
        "feasible": capacity >= target,
        "shortfall": max(0, target - capacity),
        "per_band_stratum_counts": {
            name: {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
            for name, c in per_band.items()
        },
        "deterministic_allocation": allocation if capacity >= target else {},
    }


def evaluate(rows: Sequence[dict[str, Any]], *, optimum: int, gap: int,
             bands: Sequence[tuple[str, range]]) -> dict[str, Any]:
    stratum = [r for r in rows
               if r["oracle_optimum"] == optimum and r["higher_order_gap_H"] == gap]

    seen: dict[int, str] = {}
    for name, rng in bands:
        for k in rng:
            if k in seen:
                raise BandSpecError(
                    f"bands {seen[k]!r} and {name!r} both contain {k} conflicting pairs")
            seen[k] = name

    grouped = {name: [r for r in stratum if r["conflicting_pairs"] in rng]
               for name, rng in bands}

    per_band: dict[str, Any] = {}
    for (name, rng) in bands:
        g = grouped[name]
        tight = Counter(r["tightness"] for r in g)
        structures = Counter(r["travel_structure"] for r in g)
        dominant_share = (max(tight.values()) / len(g)) if g else None
        checks = {
            "candidates_at_least_50": len(g) >= MIN_CANDIDATES,
            "tightness_values_at_least_2": len(tight) >= MIN_TIGHTNESS_VALUES,
            "dominant_tightness_at_most_60pct": (
                dominant_share is not None and dominant_share <= MAX_DOMINANT_TIGHTNESS_SHARE),
            "travel_structures_at_least_2": len(structures) >= MIN_TRAVEL_STRUCTURES,
        }
        per_band[name] = {
            "pairs": [rng.start, rng.stop - 1],
            "D": [round(rng.start / MAX_PAIRS, 4), round((rng.stop - 1) / MAX_PAIRS, 4)],
            "n": len(g),
            "distinct_pair_counts": len({r["conflicting_pairs"] for r in g}),
            "tightness_counts": {str(k): v for k, v in sorted(tight.items())},
            "dominant_tightness_share": round(dominant_share, 4) if dominant_share else None,
            "structure_counts": dict(sorted(structures.items())),
            "overlap_counts": {str(k): v for k, v in
                               sorted(Counter(r["overlap"] for r in g).items())},
            "checks": checks,
            "passed": all(checks.values()),
        }

    alloc = common_allocation(grouped)
    buffers = sorted(set(range(min(b[1].start for b in bands),
                              max(b[1].stop for b in bands)))
                     - set(seen))
    return {
        "stratum": {"n_people": 8, "oracle_optimum": optimum, "higher_order_gap_H": gap,
                    "n_in_stratum": len(stratum)},
        "bands": per_band,
        "buffer_pair_counts": buffers,
        "common_allocation": alloc,
        "all_bands_passed": all(b["passed"] for b in per_band.values()),
        "admissible": all(b["passed"] for b in per_band.values()) and alloc["feasible"],
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    e = doc["evaluation"]
    s = e["stratum"]

    add("# Band admissibility — proposed design against the frozen calibration pool\n")
    add("The bands below were **proposed, not derived here**. This report applies the "
        "criteria recorded in THESIS_DECISIONS §3 and reports pass or fail. It does not "
        "search for bands and does not suggest a repair: a failing design triggers the "
        "recorded fallback rather than an adjustment, because widening a band until it "
        "passes would choose the design from its own outcome.\n")
    add(f"Structural stratum: `n_people = {s['n_people']}`, oracle optimum "
        f"`O = {s['oracle_optimum']}`, higher-order gap `H = {s['higher_order_gap_H']}` — "
        f"{s['n_in_stratum']} calibration candidates.\n")
    add(f"Buffer pair counts, deliberately unused: {e['buffer_pair_counts'] or 'none'}.\n")

    add("\n## Per band\n")
    add("| band | pairs | D | n | ≥50 | distinct tightness | ≥2 | dominant share | ≤60% | structures | ≥2 | verdict |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, b in e["bands"].items():
        c = b["checks"]
        tick = lambda ok: "pass" if ok else "**FAIL**"  # noqa: E731
        add(f"| {name} | {b['pairs'][0]}–{b['pairs'][1]} | "
            f"{b['D'][0]:.3f}–{b['D'][1]:.3f} | {b['n']} | {tick(c['candidates_at_least_50'])} | "
            f"{len(b['tightness_counts'])} | {tick(c['tightness_values_at_least_2'])} | "
            f"{b['dominant_tightness_share']:.3f} | "
            f"{tick(c['dominant_tightness_at_most_60pct'])} | "
            f"{len(b['structure_counts'])} | {tick(c['travel_structures_at_least_2'])} | "
            f"{'PASS' if b['passed'] else '**FAIL**'} |")

    add("\n### Composition\n")
    for name, b in e["bands"].items():
        add(f"**{name}** — tightness {b['tightness_counts']}, structures "
            f"{b['structure_counts']}, overlap {b['overlap_counts']}\n")

    a = e["common_allocation"]
    add("\n## Identical composition across bands\n")
    add("Each band passing the diversity rules separately does not stop the bands differing "
        "from one another, which would move the confound between bands instead of inside "
        "them. This asks whether the strata common to all three hold enough candidates to "
        "draw the same joint (tightness, travel structure) distribution in each.\n")
    add(f"- shared strata: **{a['shared_strata']}**")
    add(f"- capacity for an identical allocation: **{a['capacity']}** per band")
    add(f"- target: {a['target_per_band']} per band")
    add(f"- feasible: **{'yes' if a['feasible'] else 'no'}**"
        + (f", short by {a['shortfall']}" if a["shortfall"] else "") + "\n")
    if a["deterministic_allocation"]:
        add("| stratum (tightness \\| structure) | instances per band |")
        add("|---|---|")
        for k, v in a["deterministic_allocation"].items():
            add(f"| {k} | {v} |")

    add(f"\n## Verdict\n")
    add(f"**{'ADMISSIBLE' if e['admissible'] else 'NOT ADMISSIBLE'}** under the "
        "pre-registered criteria.\n")
    if not e["admissible"]:
        add("The recorded response is the fallback in §3, not a revision of these bands. "
            "Moving a boundary until the check passes would make the design a function of "
            "the check.\n")
    add("The calibration pool is support evidence, not the sample. The held-out instances "
        "are generated from the main seed range and their manifest is built separately.\n")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("results/calibration/structural"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/band_admissibility"))
    ap.add_argument("--optimum", type=int, required=True)
    ap.add_argument("--higher-order-gap", type=int, required=True)
    ap.add_argument("--band", action="append", required=True,
                    help="Name=lo-hi in conflicting pairs, inclusive; repeat per band")
    args = ap.parse_args(argv)

    rows, meta = load_pool(args.pool)
    bands = [parse_band(b) for b in args.band]
    evaluation = evaluate(rows, optimum=args.optimum, gap=args.higher_order_gap, bands=bands)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "evaluates a proposed design; selects nothing",
        "pool_dir": args.pool.as_posix(),
        "pool_commit": meta.get("git_commit"),
        "proposed_bands": {name: [r.start, r.stop - 1] for name, r in bands},
        "criteria": {
            "min_candidates": MIN_CANDIDATES,
            "min_tightness_values": MIN_TIGHTNESS_VALUES,
            "max_dominant_tightness_share": MAX_DOMINANT_TIGHTNESS_SHARE,
            "min_travel_structures": MIN_TRAVEL_STRUCTURES,
            "target_per_band": TARGET_PER_BAND,
        },
        "evaluation": evaluation,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    print(f"stratum O={args.optimum}, H={args.higher_order_gap}: "
          f"{evaluation['stratum']['n_in_stratum']} candidates")
    for name, b in evaluation["bands"].items():
        flags = " ".join(k for k, ok in b["checks"].items() if not ok) or "all criteria met"
        print(f"  {name:<8} pairs {b['pairs'][0]:>2}-{b['pairs'][1]:<2} n={b['n']:<4} "
              f"dominant tightness {b['dominant_tightness_share']:.2f} -> "
              f"{'PASS' if b['passed'] else 'FAIL: ' + flags}")
    a = evaluation["common_allocation"]
    print(f"  identical composition across bands: capacity {a['capacity']}/"
          f"{a['target_per_band']} -> {'feasible' if a['feasible'] else 'NOT feasible'}")
    print(f"VERDICT: {'ADMISSIBLE' if evaluation['admissible'] else 'NOT ADMISSIBLE'}")
    print(f"written: {args.out}/summary.json, report.md")
    return 0 if evaluation["admissible"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
