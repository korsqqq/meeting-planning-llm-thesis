# scripts/analyse_structural_calibration.py
"""Pre-specified diagnostics over the structural calibration pool. CPU only, read-only.

    python -m scripts.analyse_structural_calibration \
        --pool results/calibration/structural \
        --out results/analysis/structural_calibration

**This script does not choose the complexity bands.** It contains no Low/Medium/High
constant, no threshold, and no search for the triple of `D` regions that would be best
separated, best matched on the optimum, or best balanced across generator knobs. Such a
search is an optimisation over the design space, and running it would make the bands a
function of whatever the data happens to allow — which is the failure this whole revision
exists to avoid. The script shows the structural space; the bands are decided against the
admissibility rules already written in THESIS_DECISIONS section 3 and are then fixed in a
dated amendment.

Written and committed **before the complete pool was read**. The 288-instance smoke sample
used while building the collector had already shown the ranges of `D`, `O`, `H` and the
triangle violations, so the honest claim is narrower than "blind": *the full-pool analysis
procedure was fixed before inspection of the complete 9 600-instance distribution.*

One consequence of fixing `n = 8` is that no binning decision is needed anywhere here.
`D = conflicting_pairs / 28` takes exactly 29 possible values, so every table below is cut
on the exact pair count and nothing is grouped by a width somebody had to choose. The only
coarser split reported is `D = 0` / `0 < D < 1` / `D = 1`, which is definitional rather than
selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

N_PEOPLE = 8
MAX_PAIRS = N_PEOPLE * (N_PEOPLE - 1) // 2          # 28
EXPECTED_TRIPLES = 504                               # 9 locations incl. start: 9*8*7
STRUCTURES = ("uniform", "clustered", "line", "random")
SCHEMA_VERSION = "structural_calibration_analysis/1.0"

_INT = ("seed", "n_people", "individually_reachable_count", "conflicting_pairs",
        "oracle_optimum", "alpha_reachable", "higher_order_gap_H",
        "n_max_independent_sets", "max_degree", "largest_component",
        "triangle_violations", "triangle_triples_checked")
_FLOAT = ("tightness", "overlap", "D", "optimum_over_n", "solve_seconds")
_OPT_FLOAT = ("D_reachable", "edge_share_top1", "edge_share_top2")


class PoolMismatchError(RuntimeError):
    """The CSV is not the pool its metadata describes."""


# --------------------------------------------------------------------------- #
# Small statistics, kept local so the script stands alone.
# --------------------------------------------------------------------------- #
def _q(values: Sequence[float]) -> dict[str, float | None]:
    v = sorted(values)
    if not v:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}

    def pick(p: float) -> float:
        i = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
        return round(float(v[i]), 4)

    return {"n": len(v), "min": round(float(v[0]), 4), "q1": pick(0.25),
            "median": pick(0.5), "q3": pick(0.75), "max": round(float(v[-1]), 4)}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def _mean(v: Sequence[float]) -> float | None:
    return round(statistics.fmean(v), 4) if v else None


def _share(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


# --------------------------------------------------------------------------- #
# Load and re-verify.
# --------------------------------------------------------------------------- #
def load_pool(pool_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads((pool_dir / "pool_meta.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with (pool_dir / "candidates.csv").open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            r: dict[str, Any] = dict(raw)
            for k in _INT:
                r[k] = int(raw[k])
            for k in _FLOAT:
                r[k] = float(raw[k])
            for k in _OPT_FLOAT:
                r[k] = float(raw[k]) if raw[k] not in ("", "None") else None
            r["proven_optimal"] = raw["proven_optimal"] == "True"
            rows.append(r)

    problems = []
    if len(rows) != meta["n_completed"]:
        problems.append(f"csv has {len(rows)} rows, metadata says {meta['n_completed']}")
    if {r["n_people"] for r in rows} != {N_PEOPLE}:
        problems.append(f"n_people is not uniformly {N_PEOPLE}")
    if {r["triangle_triples_checked"] for r in rows} != {EXPECTED_TRIPLES}:
        problems.append("triangle triple count is not the single expected constant")
    if any(not r["proven_optimal"] for r in rows):
        problems.append("the pool contains an unproven optimum")
    if not (meta["reserved_range"][0] <= min(r["seed"] for r in rows)
            and max(r["seed"] for r in rows) <= meta["reserved_range"][1]):
        problems.append("a seed lies outside the reserved calibration range")
    if problems:
        raise PoolMismatchError("pool failed re-verification:\n  - " + "\n  - ".join(problems))
    return rows, meta


# --------------------------------------------------------------------------- #
# A. Does the axis exist at all?
# --------------------------------------------------------------------------- #
def section_a(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    pairs = [r["conflicting_pairs"] for r in rows]
    by_pairs = Counter(pairs)
    ecdf = []
    cum = 0
    for p in range(MAX_PAIRS + 1):
        cum += by_pairs.get(p, 0)
        ecdf.append({"pairs": p, "D": round(p / MAX_PAIRS, 4),
                     "count": by_pairs.get(p, 0), "cumulative_share": round(cum / n, 4)})
    return {
        "n": n,
        "D_quantiles": _q([r["D"] for r in rows]),
        "definitional_split": {
            "D_zero": by_pairs.get(0, 0),
            "D_between": sum(v for k, v in by_pairs.items() if 0 < k < MAX_PAIRS),
            "D_one": by_pairs.get(MAX_PAIRS, 0),
        },
        "counts_per_pair_count": ecdf,
        "D_by_structure": {s: _q([r["D"] for r in rows if r["travel_structure"] == s])
                           for s in STRUCTURES},
        "D_by_tightness": {str(t): _q([r["D"] for r in rows if r["tightness"] == t])
                           for t in sorted({r["tightness"] for r in rows})},
        "D_by_overlap": {str(o): _q([r["D"] for r in rows if r["overlap"] == o])
                         for o in sorted({r["overlap"] for r in rows})},
    }


# --------------------------------------------------------------------------- #
# B. Can different D coexist with comparable optimum? The central question.
# --------------------------------------------------------------------------- #
def section_b(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_o: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_o[r["oracle_optimum"]].append(r)
    per_optimum = {
        str(o): {
            "n": len(g),
            "D": _q([r["D"] for r in g]),
            "distinct_pair_counts": len({r["conflicting_pairs"] for r in g}),
            "pair_count_range": [min(r["conflicting_pairs"] for r in g),
                                 max(r["conflicting_pairs"] for r in g)],
        }
        for o, g in sorted(by_o.items())
    }
    by_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pairs[r["conflicting_pairs"]].append(r)
    per_pair_count = {
        str(p): {
            "n": len(g),
            "optimum": _q([r["oracle_optimum"] for r in g]),
            "optimum_counts": dict(sorted(Counter(r["oracle_optimum"] for r in g).items())),
            "optimum_over_n_mean": _mean([r["optimum_over_n"] for r in g]),
        }
        for p, g in sorted(by_pairs.items())
    }
    return {
        "spearman_D_vs_optimum": spearman([r["D"] for r in rows],
                                          [r["oracle_optimum"] for r in rows]),
        "spearman_D_vs_optimum_over_n": spearman([r["D"] for r in rows],
                                                 [r["optimum_over_n"] for r in rows]),
        "per_optimum": per_optimum,
        "per_pair_count": per_pair_count,
    }


# --------------------------------------------------------------------------- #
# C. Is a region of D effectively one generator knob?
# --------------------------------------------------------------------------- #
def section_c(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pairs[r["conflicting_pairs"]].append(r)
    out = {}
    for p, g in sorted(by_pairs.items()):
        tight = Counter(r["tightness"] for r in g)
        top_t, top_n = tight.most_common(1)[0]
        out[str(p)] = {
            "n": len(g),
            "tightness_counts": {str(k): v for k, v in sorted(tight.items())},
            "distinct_tightness": len(tight),
            "dominant_tightness": top_t,
            "dominant_tightness_share": _share(top_n, len(g)),
            "overlap_counts": {str(k): v for k, v in
                               sorted(Counter(r["overlap"] for r in g).items())},
            "structure_counts": dict(sorted(Counter(r["travel_structure"] for r in g).items())),
            "distinct_structures": len({r["travel_structure"] for r in g}),
        }
    return {
        "spearman_D_vs_tightness": spearman([r["D"] for r in rows],
                                            [r["tightness"] for r in rows]),
        "spearman_D_vs_overlap": spearman([r["D"] for r in rows],
                                          [r["overlap"] for r in rows]),
        "per_pair_count": out,
    }


# --------------------------------------------------------------------------- #
# D. Does a high D mean real structure, or one person conflicting with everyone?
# --------------------------------------------------------------------------- #
_GRAPH_FIELDS = ("alpha_reachable", "higher_order_gap_H", "max_degree",
                 "edge_share_top1", "edge_share_top2", "largest_component",
                 "n_max_independent_sets", "individually_reachable_count")


def section_d(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall = {f: _q([r[f] for r in rows if r[f] is not None]) for f in _GRAPH_FIELDS}
    by_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pairs[r["conflicting_pairs"]].append(r)
    per_pair_count = {
        str(p): {
            "n": len(g),
            **{f: _mean([r[f] for r in g if r[f] is not None]) for f in _GRAPH_FIELDS},
        }
        for p, g in sorted(by_pairs.items())
    }
    corr = {}
    for f in _GRAPH_FIELDS:
        usable = [(r["D"], r[f]) for r in rows if r[f] is not None]
        corr[f] = spearman([a for a, _ in usable], [b for _, b in usable])
    return {
        "overall": overall,
        "spearman_D_vs": corr,
        "per_pair_count": per_pair_count,
        "reachable_below_n": sum(1 for r in rows
                                 if r["individually_reachable_count"] < N_PEOPLE),
    }


# --------------------------------------------------------------------------- #
# E. Triangle inequality and the sign of the higher-order gap.
# --------------------------------------------------------------------------- #
def section_e(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_structure = {}
    for s in STRUCTURES:
        g = [r for r in rows if r["travel_structure"] == s]
        v = [r["triangle_violations"] for r in g]
        per_structure[s] = {
            "n": len(g),
            "violations": _q(v),
            "violation_rate_mean": _mean([x / EXPECTED_TRIPLES for x in v]),
            "instances_with_any_violation": sum(1 for x in v if x > 0),
            "H_sign": {
                "negative": sum(1 for r in g if r["higher_order_gap_H"] < 0),
                "zero": sum(1 for r in g if r["higher_order_gap_H"] == 0),
                "positive": sum(1 for r in g if r["higher_order_gap_H"] > 0),
            },
            "H": _q([r["higher_order_gap_H"] for r in g]),
        }
    return {
        "triples_checked_per_instance": EXPECTED_TRIPLES,
        "per_structure": per_structure,
        "H_sign_overall": {
            "negative": sum(1 for r in rows if r["higher_order_gap_H"] < 0),
            "zero": sum(1 for r in rows if r["higher_order_gap_H"] == 0),
            "positive": sum(1 for r in rows if r["higher_order_gap_H"] > 0),
        },
        "negative_H_instances": sorted(
            r["instance_id"] for r in rows if r["higher_order_gap_H"] < 0)[:20],
    }


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _f(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, b, c, d, e = (doc["A_axis"], doc["B_optimum"], doc["C_generator"],
                     doc["D_graph"], doc["E_triangle"])

    add("# Structural calibration — pre-specified diagnostics\n")
    add("Read-only over the frozen candidate pool. **No band is chosen here.** This report "
        "contains no Low/Medium/High constant and no search for the best-separated or "
        "best-matched triple of `D` regions; the bands are decided against the "
        "admissibility rules in THESIS_DECISIONS §3 and fixed in a dated amendment.\n")
    add(f"Pool: {a['n']} candidates at `n_people = {N_PEOPLE}`, seeds "
        f"{doc['meta']['seed_start']}–{doc['meta']['seed_end']}, collector commit "
        f"`{str(doc['meta']['git_commit'])[:8]}`. Since `n` is fixed, `D` takes exactly "
        f"{MAX_PAIRS + 1} values and every table is cut on the exact conflict-pair count — "
        "no bin width was chosen by anyone.\n")

    add("\n## A. Does the axis exist\n")
    add(f"`D` quartiles: {_f(a['D_quantiles']['min'])} / {_f(a['D_quantiles']['q1'])} / "
        f"{_f(a['D_quantiles']['median'])} / {_f(a['D_quantiles']['q3'])} / "
        f"{_f(a['D_quantiles']['max'])}\n")
    s = a["definitional_split"]
    add(f"- `D = 0`: **{s['D_zero']}** candidates")
    add(f"- `0 < D < 1`: **{s['D_between']}**")
    add(f"- `D = 1`: **{s['D_one']}**\n")
    add("| pairs | D | count | cumulative share |")
    add("|---|---|---|---|")
    for row in a["counts_per_pair_count"]:
        if row["count"]:
            add(f"| {row['pairs']} | {row['D']:.3f} | {row['count']} | "
                f"{row['cumulative_share']:.3f} |")
    for label, key in (("travel structure", "D_by_structure"),
                       ("tightness", "D_by_tightness"), ("overlap", "D_by_overlap")):
        add(f"\n**`D` by {label}**\n")
        add("| value | n | min | q1 | median | q3 | max |")
        add("|---|---|---|---|---|---|---|")
        for k, q in a[key].items():
            add(f"| {k} | {q['n']} | {_f(q['min'])} | {_f(q['q1'])} | {_f(q['median'])} | "
                f"{_f(q['q3'])} | {_f(q['max'])} |")

    add("\n## B. Conflict density against the oracle optimum\n")
    add("The central question: can well-separated `D` coexist with a comparable optimum?\n")
    add(f"Spearman(`D`, optimum) = **{_f(b['spearman_D_vs_optimum'])}**; "
        f"Spearman(`D`, optimum/n) = {_f(b['spearman_D_vs_optimum_over_n'])}\n")
    add("**Which `D` are available at each optimum**\n")
    add("| optimum | n | D min | D median | D max | distinct pair counts | pair range |")
    add("|---|---|---|---|---|---|---|")
    for o, v in b["per_optimum"].items():
        add(f"| {o} | {v['n']} | {_f(v['D']['min'])} | {_f(v['D']['median'])} | "
            f"{_f(v['D']['max'])} | {v['distinct_pair_counts']} | {v['pair_count_range']} |")
    add("\n**Which optima occur at each pair count**\n")
    add("| pairs | n | optimum min | median | max | mean optimum/n |")
    add("|---|---|---|---|---|---|")
    for p, v in b["per_pair_count"].items():
        add(f"| {p} | {v['n']} | {_f(v['optimum']['min'])} | {_f(v['optimum']['median'])} | "
            f"{_f(v['optimum']['max'])} | {_f(v['optimum_over_n_mean'])} |")

    add("\n## C. Is a region of `D` effectively one generator knob\n")
    add(f"Spearman(`D`, tightness) = **{_f(c['spearman_D_vs_tightness'])}**; "
        f"Spearman(`D`, overlap) = {_f(c['spearman_D_vs_overlap'])}\n")
    add("| pairs | n | distinct tightness | dominant tightness | its share | distinct structures |")
    add("|---|---|---|---|---|---|")
    for p, v in c["per_pair_count"].items():
        add(f"| {p} | {v['n']} | {v['distinct_tightness']} | {v['dominant_tightness']} | "
            f"{_f(v['dominant_tightness_share'])} | {v['distinct_structures']} |")
    add("\nThe §3 diversity rule asks for at least two tightness values, at least two "
        "travel structures, and no single tightness above 60 % — applied to the final "
        "bands, not to individual pair counts. This table is the input to that check.\n")

    add("\n## D. Graph structure behind a high `D`\n")
    add("| quantity | min | q1 | median | q3 | max | Spearman with D |")
    add("|---|---|---|---|---|---|---|")
    for f in _GRAPH_FIELDS:
        q = d["overall"][f]
        add(f"| `{f}` | {_f(q['min'])} | {_f(q['q1'])} | {_f(q['median'])} | {_f(q['q3'])} | "
            f"{_f(q['max'])} | {_f(d['spearman_D_vs'][f])} |")
    add(f"\nCandidates with fewer than {N_PEOPLE} individually reachable people: "
        f"**{d['reachable_below_n']}**. These cannot reach the top of the `D` range, since "
        "the conflict graph is built over the reachable set only.\n")
    add("| pairs | n | alpha | H | max degree | edge share top1 | max ind. sets |")
    add("|---|---|---|---|---|---|---|")
    for p, v in d["per_pair_count"].items():
        add(f"| {p} | {v['n']} | {_f(v['alpha_reachable'])} | {_f(v['higher_order_gap_H'])} | "
            f"{_f(v['max_degree'])} | {_f(v['edge_share_top1'])} | "
            f"{_f(v['n_max_independent_sets'])} |")
    add("\nA high `D` together with a high `alpha_reachable` and a high `edge_share_top1` "
        "is the hub case: many conflicts sitting on one person, and a trivial answer.\n")

    add("\n## E. Triangle inequality and the sign of `H`\n")
    add(f"Ordered triples examined per instance: {e['triples_checked_per_instance']}.\n")
    add("| structure | n | violations median | max | mean rate | instances with any | H<0 | H=0 | H>0 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for s_name, v in e["per_structure"].items():
        h = v["H_sign"]
        add(f"| {s_name} | {v['n']} | {_f(v['violations']['median'])} | "
            f"{_f(v['violations']['max'])} | {_f(v['violation_rate_mean'])} | "
            f"{v['instances_with_any_violation']} | {h['negative']} | {h['zero']} | "
            f"{h['positive']} |")
    ov = e["H_sign_overall"]
    add(f"\nOverall: `H < 0` in **{ov['negative']}** candidates, `H = 0` in {ov['zero']}, "
        f"`H > 0` in {ov['positive']}. Negative values are reported, never clipped: they "
        "record that feasibility is not monotone under deletion.\n")

    add("\n## What this report deliberately does not do\n")
    add("- It names no band and applies no threshold.")
    add("- It does not search for the triple of `D` regions that would look best on "
        "separation, optimum matching or knob balance. That search would select the design "
        "from its own outcome.")
    add("- It draws no conclusion about whether Plan A is admissible. That decision is made "
        "against the rules already written in §3 and recorded in a dated amendment.\n")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("results/calibration/structural"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/structural_calibration"))
    args = ap.parse_args(argv)

    rows, meta = load_pool(args.pool)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "pool_dir": args.pool.as_posix(),
        "decides_bands": False,
        "meta": meta,
        "A_axis": section_a(rows),
        "B_optimum": section_b(rows),
        "C_generator": section_c(rows),
        "D_graph": section_d(rows),
        "E_triangle": section_e(rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    a, b = doc["A_axis"], doc["B_optimum"]
    print(f"pool re-verified: {a['n']} candidates, n={N_PEOPLE}, all optima proven")
    print(f"D: zero={a['definitional_split']['D_zero']} "
          f"between={a['definitional_split']['D_between']} "
          f"one={a['definitional_split']['D_one']}")
    print(f"Spearman(D, optimum) = {b['spearman_D_vs_optimum']}")
    print(f"written: {args.out}/summary.json, report.md")
    print("no band was chosen; decide against the §3 rules and record a dated amendment")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
