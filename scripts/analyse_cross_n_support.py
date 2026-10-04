# scripts/analyse_cross_n_support.py
"""Cross-size structural support for the lower-complexity calibration. CPU only, read-only.

    python -m scripts.analyse_cross_n_support \
        --pool 4=results/calibration/lowcx_n4 \
        --pool 5=results/calibration/lowcx_n5 \
        --pool 6=results/calibration/lowcx_n6 \
        --pool 8=results/calibration/budget_dev \
        --out results/analysis/cross_n_support

**This decides nothing.** It selects no instance, fixes no optimum histogram, cuts no band
and writes no manifest. It reads structural pools and prints what support exists, so that
the optimum distribution for the 12-per-size sample can be chosen from measured capacity
rather than from intuition. The selection itself lives in a separate script that is written
and committed after this output has been read.

WHY THE OPTIMUM IS A MATCHING VARIABLE AND NOT A FREE ONE. `satisfaction = achieved / O`.
If one part of the design sits mostly at `O = 2` and another at `O = 4` then the two differ
in the granularity of the metric and in how much reward a single meeting is worth, on top of
whatever structural difference was intended. The pilot failed exactly here -- its optimum
fell with the nominal level, so the denominator shrank along the axis it was supposed to
index. Section 3 therefore matches `O` across the compared groups. Adding `n` as a second
axis does not remove that requirement; it adds a second direction in which it has to hold.

WHAT THE SANITY-CHECK SECTION IS FOR. A lower-complexity design could also be built at
`n = 8` by admitting high-optimum instances (`k <= 1, O >= 6`), which the frozen bands
exclude because `O >= 6` does not exist in the high band. That option is reported here as a
read-only comparison so the cost of not taking it is visible. It is **not** a candidate for
the selection: it reintroduces the optimum confound this design exists to remove.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "cross_n_support/1.0"

# Section 3's diversity rules, restated here only to test attainability. Nothing is enforced.
MIN_TIGHTNESS_VALUES = 2
MIN_TRAVEL_STRUCTURES = 2
MAX_TIGHTNESS_SHARE_NUM, MAX_TIGHTNESS_SHARE_DEN = 6, 10

# The frozen n=8 subset, for the comparison row. Read from the manifest, never retyped.
FROZEN_SUBSET = Path("results/manifests/subset__bands_budget_dev__911e4864d6fb.json")


class PoolError(RuntimeError):
    """A pool is missing, malformed, or not the size it claims."""


# --------------------------------------------------------------------------- #
# Loading.
# --------------------------------------------------------------------------- #
def load_pool(pool_dir: Path, expect_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path, csv_path = pool_dir / "pool_meta.json", pool_dir / "candidates.csv"
    if not meta_path.exists() or not csv_path.exists():
        raise PoolError(f"{pool_dir}: expected pool_meta.json and candidates.csv")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("audit", {}).get("all_checks_passed"):
        raise PoolError(f"{pool_dir}: the pool did not pass its own completeness audit; "
                        "its distributions must not be read")
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "instance_id": r["instance_id"],
                "seed": int(r["seed"]),
                "n": int(r["n_people"]),
                "tightness": float(r["tightness"]),
                "overlap": float(r["overlap"]),
                "structure": r["travel_structure"],
                "k": int(r["conflicting_pairs"]),
                "D": float(r["D"]),
                "O": int(r["oracle_optimum"]),
                "reachable": int(r["individually_reachable_count"]),
                "alpha": int(r["alpha_reachable"]),
                "H": int(r["higher_order_gap_H"]),
                "proven": r["proven_optimal"] == "True",
            })
    bad = {r["n"] for r in rows} - {expect_n}
    if bad:
        raise PoolError(f"{pool_dir}: expected n_people={expect_n}, found {sorted(bad)}")
    unproven = [r["instance_id"] for r in rows if not r["proven"]]
    if unproven:
        raise PoolError(f"{pool_dir}: {len(unproven)} unproven optima; matching on an "
                        f"unproven optimum is not possible: {unproven[:3]}")
    return rows, meta


def max_pairs(n: int) -> int:
    return n * (n - 1) // 2


# --------------------------------------------------------------------------- #
# Capacity under a candidate optimum support.
# --------------------------------------------------------------------------- #
def capacity(pools: dict[int, list[dict[str, Any]]], support: Sequence[int]
             ) -> dict[int, dict[int, int]]:
    """`capacity[n][O]` -- candidates available at each size for each optimum in support."""
    return {n: {o: sum(1 for r in rows if r["O"] == o) for o in support}
            for n, rows in sorted(pools.items())}


def joint_capacity(cap: dict[int, dict[int, int]], support: Sequence[int]) -> dict[int, int]:
    """The binding size for each optimum: a matched design cannot exceed the smallest pool."""
    return {o: min(cap[n][o] for n in cap) for o in support}


def scaled_histogram(total: int, reference: dict[int, int]) -> dict[int, int]:
    """Largest-remainder scaling, identical in behaviour to the frozen band selector.

    Ties are broken on the remainder and then on the optimum value, so dictionary order
    cannot change the answer.
    """
    ref_total = sum(reference.values())
    if ref_total <= 0:
        raise ValueError("empty reference histogram")
    exact = {o: total * n / ref_total for o, n in reference.items()}
    counts = {o: int(v) for o, v in exact.items()}
    short = total - sum(counts.values())
    ranked = sorted(exact, key=lambda o: (-(exact[o] - counts[o]), o))
    for o in ranked[:short]:
        counts[o] += 1
    return {o: c for o, c in sorted(counts.items()) if c}


def diversity_headroom(rows: Sequence[dict[str, Any]], support: Sequence[int],
                       want: dict[int, int], per_size: int) -> dict[str, Any]:
    """Necessary conditions for the diversity rules at this size. Not a feasibility proof.

    A full answer needs the selector's CP-SAT model, because the rules interact with
    one-instance-per-seed. What is computable here is whether each rule could be met at all:
    if the optimum a histogram needs exists at only one tightness value and that value must
    then carry more than the 60% ceiling, no selection can succeed and there is no point
    running the solver.
    """
    eligible = [r for r in rows if r["O"] in support]
    per_o_tightness = {o: Counter(r["tightness"] for r in eligible if r["O"] == o)
                       for o in support}
    ceiling = MAX_TIGHTNESS_SHARE_NUM * per_size // MAX_TIGHTNESS_SHARE_DEN

    # Worst case per optimum: if an optimum lives at a single tightness value, that value
    # must carry all `want[o]` of them.
    forced: dict[int, int] = {}
    for o, n_want in want.items():
        counts = per_o_tightness.get(o, Counter())
        forced[o] = n_want if len(counts) == 1 else 0
    max_forced = max(forced.values(), default=0)

    return {
        "tightness_values_per_optimum": {str(o): sorted(per_o_tightness[o]) for o in support},
        "structures_available": sorted({r["structure"] for r in eligible}),
        "distinct_seeds_available": len({r["seed"] for r in eligible}),
        "tightness_ceiling_at_size": ceiling,
        "forced_single_tightness_load": {str(o): v for o, v in forced.items() if v},
        "single_tightness_rule_violated": max_forced > ceiling,
        "at_least_two_tightness_possible":
            len({r["tightness"] for r in eligible}) >= MIN_TIGHTNESS_VALUES,
        "at_least_two_structures_possible":
            len({r["structure"] for r in eligible}) >= MIN_TRAVEL_STRUCTURES,
    }


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def per_size_summary(pools: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for n, rows in sorted(pools.items()):
        ks = [r["k"] for r in rows]
        out.append({
            "n": n,
            "candidates": len(rows),
            "max_possible_pairs": max_pairs(n),
            "optimum_histogram": dict(sorted(Counter(r["O"] for r in rows).items())),
            "k_min": min(ks), "k_max": max(ks),
            "D_min": round(min(ks) / max_pairs(n), 4),
            "D_max": round(max(ks) / max_pairs(n), 4),
            "share_conflict_free": round(sum(1 for k in ks if k == 0) / len(rows), 4),
            "structures": dict(sorted(Counter(r["structure"] for r in rows).items())),
            "tightness": dict(sorted(Counter(r["tightness"] for r in rows).items())),
            "H_histogram": dict(sorted(Counter(r["H"] for r in rows).items())),
            "seeds": [min(r["seed"] for r in rows), max(r["seed"] for r in rows)],
        })
    return out


def cross_tab(pools: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for n, rows in sorted(pools.items()):
        by_o: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_o[r["O"]].append(r)
        for o in sorted(by_o):
            grp = by_o[o]
            ks = sorted(r["k"] for r in grp)
            hs = sorted(r["H"] for r in grp)
            out.append({
                "n": n, "optimum": o, "count": len(grp),
                "k_min": ks[0], "k_median": ks[len(ks) // 2], "k_max": ks[-1],
                "D_min": round(ks[0] / max_pairs(n), 4),
                "D_median": round(ks[len(ks) // 2] / max_pairs(n), 4),
                "D_max": round(ks[-1] / max_pairs(n), 4),
                "H_min": hs[0], "H_median": hs[len(hs) // 2], "H_max": hs[-1],
                "H_mean": round(statistics.fmean(hs), 3),
                "optimum_over_n": round(o / n, 3),
                "tightness_values": sorted({r["tightness"] for r in grp}),
                "structures": sorted({r["structure"] for r in grp}),
                "distinct_seeds": len({r["seed"] for r in grp}),
            })
    return out


def matched_variant(pools: dict[int, list[dict[str, Any]]], support: Sequence[int],
                    per_size: int) -> dict[str, Any]:
    """What a matched design over `support` would look like. Proposes nothing."""
    cap = capacity(pools, support)
    joint = joint_capacity(cap, support)
    total_joint = sum(joint.values())
    hist = scaled_histogram(per_size, joint) if total_joint else {}
    return {
        "support": list(support),
        "per_size_capacity": {str(n): {str(o): cap[n][o] for o in support} for n in cap},
        "joint_capacity": {str(o): joint[o] for o in support},
        "joint_capacity_total": total_joint,
        "binding_size_per_optimum": {
            str(o): min(cap, key=lambda n: cap[n][o]) for o in support} if cap else {},
        "scaled_to_per_size": {str(o): c for o, c in hist.items()},
        "sufficient_capacity": all(joint[o] >= hist.get(o, 0) for o in support),
        "diversity": {str(n): diversity_headroom(rows, support, hist, per_size)
                      for n, rows in sorted(pools.items())},
    }


def ceiling_analysis(pools: dict[int, list[dict[str, Any]]],
                     support: Sequence[int]) -> list[dict[str, Any]]:
    """How much of the task a matched optimum represents at each size.

    `O / n` is the fraction of people the best possible plan meets. As it approaches 1 the
    metric loses room below the top: a condition that finds the whole plan and a condition
    that finds all but one are separated by a single meeting, and every condition that
    solves the instance scores exactly 1.0. That is a resolution problem, not a difficulty
    one, and it is why this is reported per (n, O) rather than per n.
    """
    out = []
    for n, rows in sorted(pools.items()):
        for o in support:
            grp = [r for r in rows if r["O"] == o]
            if not grp:
                out.append({"n": n, "optimum": o, "count": 0})
                continue
            out.append({
                "n": n, "optimum": o, "count": len(grp),
                "optimum_over_n": round(o / n, 3),
                "meets_everyone": o == n,
                "satisfaction_step": round(1.0 / o, 4),
                "arithmetic_H_ceiling": n - o,
                "observed_H_max": max(r["H"] for r in grp),
                "observed_H_mean": round(statistics.fmean([r["H"] for r in grp]), 3),
                "mean_unreachable": round(
                    statistics.fmean([n - r["reachable"] for r in grp]), 3),
            })
    return out


def sanity_check_high_optimum(pool_n8: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The rejected `n = 8, k <= 1, O >= 6` option, reported so its cost is visible.

    Read-only and explicitly excluded from selection: admitting it would put the compared
    groups at different optima, which is the confound the matching rule exists to remove.
    """
    sub = [r for r in pool_n8 if r["k"] <= 1 and r["O"] >= 6]
    matched = [r for r in pool_n8 if r["k"] <= 1 and r["O"] in (3, 4)]
    return {
        "status": "READ-ONLY SANITY CHECK -- NOT A SELECTION CANDIDATE",
        "why_excluded": ("O >= 6 does not exist in the frozen high band, so admitting it "
                         "puts the compared groups at different optima and reintroduces "
                         "the confound the matching rule removes"),
        "filter": "n = 8, conflict pairs <= 1, O >= 6",
        "count": len(sub),
        "optimum_histogram": dict(sorted(Counter(r["O"] for r in sub).items())),
        "tightness": dict(sorted(Counter(r["tightness"] for r in sub).items())),
        "H_histogram": dict(sorted(Counter(r["H"] for r in sub).items())),
        "H_mean": round(statistics.fmean([r["H"] for r in sub]), 3) if sub else None,
        "mean_unreachable": round(
            statistics.fmean([8 - r["reachable"] for r in sub]), 3) if sub else None,
        "for_contrast_the_matched_low_band": {
            "filter": "n = 8, conflict pairs <= 1, O in {3, 4}",
            "count": len(matched),
            "H_histogram": dict(sorted(Counter(r["H"] for r in matched).items())),
            "H_mean": round(statistics.fmean([r["H"] for r in matched]), 3) if matched else None,
        },
    }


def frozen_subset_row(path: Path) -> dict[str, Any] | None:
    """The frozen n=8 development subset, read from its manifest for the comparison row."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for rows in doc["instances"].values() for e in rows]
    return {
        "source": str(path),
        "content_hash": doc.get("content_hash", "")[:12],
        "n_instances": len(entries),
        "optimum_histogram": dict(sorted(Counter(e["optimum"] for e in entries).items())),
        "per_band_optimum_histogram": {
            band: dict(sorted(Counter(e["optimum"] for e in rows).items()))
            for band, rows in doc["instances"].items()},
        "k_range": [min(e["complexity_metric"] for e in entries),
                    max(e["complexity_metric"] for e in entries)],
        "tightness": dict(sorted(Counter(e["cell"]["tightness"] for e in entries).items())),
        "structures": dict(sorted(Counter(e["cell"]["travel_structure"]
                                          for e in entries).items())),
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def render(summary: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    add("# Cross-size structural support — lower-complexity calibration")
    add("")
    add("**Read-only. Selects nothing, fixes no optimum histogram, cuts no band.** The "
        "numbers below exist so that the optimum distribution for the 12-per-size sample "
        "is chosen from measured capacity rather than from intuition.")
    add("")

    add("## Pools")
    add("")
    add("| n | candidates | seeds | C(n,2) | k range | D range | conflict-free | audit |")
    add("|---|---|---|---|---|---|---|---|")
    for s in summary["per_size"]:
        add(f"| {s['n']} | {s['candidates']} | {s['seeds'][0]}–{s['seeds'][1]} | "
            f"{s['max_possible_pairs']} | {s['k_min']}–{s['k_max']} | "
            f"{s['D_min']:.3f}–{s['D_max']:.3f} | {s['share_conflict_free']:.1%} | passed |")
    add("")

    add("## Optimum distribution per size")
    add("")
    optima = sorted({o for s in summary["per_size"] for o in map(int, s["optimum_histogram"])})
    add("| n | " + " | ".join(f"O={o}" for o in optima) + " |")
    add("|---|" + "---|" * len(optima))
    for s in summary["per_size"]:
        h = {int(k): v for k, v in s["optimum_histogram"].items()}
        add(f"| {s['n']} | " + " | ".join(str(h.get(o, "—")) for o in optima) + " |")
    add("")

    add("## Cross-tab: size × optimum")
    add("")
    add("| n | O | count | O/n | k min | k med | k max | D med | H min | H med | H max | "
        "tightness values | structures |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["cross_tab"]:
        add(f"| {r['n']} | {r['optimum']} | {r['count']} | {r['optimum_over_n']} | "
            f"{r['k_min']} | {r['k_median']} | {r['k_max']} | {r['D_median']:.3f} | "
            f"{r['H_min']} | {r['H_median']} | {r['H_max']} | "
            f"{len(r['tightness_values'])} | {len(r['structures'])} |")
    add("")

    for key, title in (("variant_O34", "Matched variant — common support O ∈ {3, 4}"),
                       ("variant_O3", "Matched variant — common support O = {3} only")):
        v = summary[key]
        add(f"## {title}")
        add("")
        add("| n | " + " | ".join(f"capacity O={o}" for o in v["support"]) + " |")
        add("|---|" + "---|" * len(v["support"]))
        for n, caps in v["per_size_capacity"].items():
            add(f"| {n} | " + " | ".join(str(caps[str(o)]) for o in v["support"]) + " |")
        add("")
        add(f"* joint capacity {v['joint_capacity']}, binding size per optimum "
            f"{v['binding_size_per_optimum']}")
        add(f"* scaled to {summary['per_size_target']} per size: "
            f"{v['scaled_to_per_size']} — capacity sufficient: {v['sufficient_capacity']}")
        blocked = [n for n, d in v["diversity"].items()
                   if d["single_tightness_rule_violated"]
                   or not d["at_least_two_tightness_possible"]
                   or not d["at_least_two_structures_possible"]]
        add(f"* diversity rules cannot be met at: {blocked or 'no size — all clear'}")
        add("")

    add("## Ceiling analysis")
    add("")
    add("| n | O | O/n | meets everyone | satisfaction step | H ceiling (n−O) | H max seen "
        "| H mean | mean unreachable |")
    add("|---|---|---|---|---|---|---|---|---|")
    for r in summary["ceiling"]:
        if not r.get("count"):
            add(f"| {r['n']} | {r['optimum']} | — | — | — | — | — | — | — |")
            continue
        add(f"| {r['n']} | {r['optimum']} | {r['optimum_over_n']} | "
            f"{'yes' if r['meets_everyone'] else 'no'} | {r['satisfaction_step']:.3f} | "
            f"{r['arithmetic_H_ceiling']} | {r['observed_H_max']} | {r['observed_H_mean']} | "
            f"{r['mean_unreachable']} |")
    add("")

    sc = summary["sanity_check"]
    add("## Sanity check — the rejected high-optimum option at n = 8")
    add("")
    add(f"**{sc['status']}**")
    add("")
    add(f"{sc['why_excluded']}.")
    add("")
    add(f"* `{sc['filter']}` → {sc['count']} candidates, optimum histogram "
        f"{sc['optimum_histogram']}, mean H {sc['H_mean']}, mean unreachable "
        f"{sc['mean_unreachable']}")
    m = sc["for_contrast_the_matched_low_band"]
    add(f"* for contrast `{m['filter']}` → {m['count']} candidates, H histogram "
        f"{m['H_histogram']}, mean H {m['H_mean']}")
    add("")

    fs = summary.get("frozen_n8_subset")
    if fs:
        add("## The frozen n = 8 development subset, for comparison")
        add("")
        add(f"* `{fs['source']}` (`{fs['content_hash']}`), {fs['n_instances']} instances")
        add(f"* optimum histogram {fs['optimum_histogram']}, identical per band: "
            f"{fs['per_band_optimum_histogram']}")
        add(f"* k range {fs['k_range']}, tightness {fs['tightness']}")
        add("")

    add("---")
    add("")
    add("*Nothing above is frozen. The optimum distribution for the 36 instances is a "
        "separate dated decision, taken after this output has been read and recorded in "
        "THESIS_DECISIONS before any instance is selected.*")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", required=True, metavar="N=DIR",
                    help="size and pool directory, e.g. 4=results/calibration/lowcx_n4")
    ap.add_argument("--per-size", type=int, default=12,
                    help="sample size the capacity is judged against; scales nothing else")
    ap.add_argument("--out", type=Path, default=Path("results/analysis/cross_n_support"))
    args = ap.parse_args(argv)

    pools: dict[int, list[dict[str, Any]]] = {}
    metas: dict[int, dict[str, Any]] = {}
    for spec in args.pool:
        if "=" not in spec:
            ap.error(f"--pool expects N=DIR, got {spec!r}")
        n_str, path = spec.split("=", 1)
        n = int(n_str)
        pools[n], metas[n] = load_pool(Path(path), n)

    small = {n: rows for n, rows in pools.items() if n != 8}
    if not small:
        ap.error("no size other than 8 was supplied; there is nothing to compare")

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decides": "nothing — read-only support tables",
        "per_size_target": args.per_size,
        "pools": {str(n): {"dir": spec.split("=", 1)[1], "commit": metas[n].get("git_commit"),
                           "seed_range": metas[n].get("reserved_range"),
                           "n_candidates": len(pools[n])}
                  for n, spec in zip(sorted(pools), sorted(args.pool, key=lambda s: int(s.split("=")[0])))},
        "per_size": per_size_summary(pools),
        "cross_tab": cross_tab(pools),
        "variant_O34": matched_variant(pools, (3, 4), args.per_size),
        "variant_O3": matched_variant(pools, (3,), args.per_size),
        "ceiling": ceiling_analysis(pools, (2, 3, 4)),
    }
    if 8 in pools:
        summary["sanity_check"] = sanity_check_high_optimum(pools[8])
        summary["frozen_n8_subset"] = frozen_subset_row(_REPO_ROOT / FROZEN_SUBSET)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")

    # The files are UTF-8 regardless; only the console needs persuading. A Windows terminal
    # defaults to a legacy code page here and would otherwise abort the run on the first
    # multiplication sign -- after both artifacts are already on disk, which is the
    # confusing kind of failure.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - stream without reconfigure
        pass
    print(report)
    print(f"written: {args.out}/summary.json, report.md")
    print("nothing is selected or frozen here")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
