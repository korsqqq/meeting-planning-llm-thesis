# scripts/search_band_design.py
"""Exhaustive admissibility search over three-band designs. CPU only, no LLM quantity.

    python -m scripts.search_band_design --pool results/calibration/structural \
        --out results/analysis/band_search

This answers one question and no other: **does a triple of separated `D` regions exist for
which the previously registered requirements can all be met at once?** The earlier check
rejected one specific proposal (`O = 4`, `H = 0`, bands 1-4 / 7-10 / 13-16). That proposal
was stricter than anything registered, so its failure does not settle whether the
registered design is reachable. This search settles it.

The answer is produced by enumeration and a fixed tie-break, not by judgement. If nothing
is admissible the script returns NO_FEASIBLE_DESIGN, and no boundary, matching variable or
threshold may be changed afterwards in response.

Status of each ingredient, kept separate so the audit trail does not claim more foresight
than it had:

*Previously registered substantive requirements* (THESIS_DECISIONS section 3, before the
calibration pool existed): separated `D` levels; at least 50 instances per band; at least
two tightness values and two travel structures per band; no single tightness above 60%;
comparable `O` distribution across bands.

*Prospectively fixed operationalisation* (written now, before this search ran, with the
structural pool already seen but no new LLM outcome seen): "comparable `O` distribution"
means an **identical empirical `O` histogram** in the three selected samples; the
requirements must hold **jointly on the selected sample** rather than separately on the raw
bands; bands are contiguous intervals of the conflict count with at least one unused value
between neighbours; the tie-break below.

*Post-specified non-binding diagnostics*: `H`, `overlap`, `alpha_reachable`, conflict
concentration, triangle statistics, raw-band composition. **None of these may cause a
failure, enter the objective, or break a tie.** They are reported for the design the rule
selects, and an unattractive value in them is a residual limitation to be carried into the
analysis, never a reason to search again.

Why the requirements must be checked jointly. Each band can satisfy every rule on its raw
contents and the design still be impossible: the candidates that make the `O` histograms
match may all sit at one tightness value, so the selected sample would violate diversity
even though the band does not. The admission test is therefore the existence of a
selection, decided by CP-SAT over the calibration counts. The solver appears here only as
combinatorial feasibility machinery over structural counts; no agent, plan or score enters
it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ortools.sat.python import cp_model  # noqa: E402

from scripts.analyse_structural_calibration import MAX_PAIRS, load_pool  # noqa: E402

SCHEMA_VERSION = "band_design_search/1.0"

TARGET_PER_BAND = 50
MIN_TIGHTNESS_VALUES = 2
MIN_TRAVEL_STRUCTURES = 2
MAX_TIGHTNESS_SHARE_NUM, MAX_TIGHTNESS_SHARE_DEN = 6, 10   # 60%, as exact integers
MIN_GAP = 1                                                 # unused k values between bands
_SOLVER_SEED = 0

# Diagnostics that are reported but may never bind, enter the objective, or break a tie.
NON_BINDING = ("overlap", "higher_order_gap_H", "alpha_reachable", "edge_share_top1",
               "triangle_violations")


def band_triples(max_k: int = MAX_PAIRS, min_gap: int = MIN_GAP
                 ) -> Iterator[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Every ordered triple of contiguous, non-overlapping intervals with a real gap.

    Width-1 bands are allowed on purpose. If the cleanest three regions turn out to be
    three single conflict counts, that is a controlled three-level manipulation, not a
    defect; imposing a minimum width would be one more number with nothing behind it.
    """
    for a1 in range(max_k + 1):
        for b1 in range(a1, max_k + 1):
            for a2 in range(b1 + min_gap + 1, max_k + 1):
                for b2 in range(a2, max_k + 1):
                    for a3 in range(b2 + min_gap + 1, max_k + 1):
                        for b3 in range(a3, max_k + 1):
                            yield ((a1, b1), (a2, b2), (a3, b3))


def separation(bands: Sequence[tuple[int, int]]) -> int:
    """g = min number of unused conflict counts between adjacent bands."""
    return min(bands[i + 1][0] - bands[i][1] - 1 for i in range(len(bands) - 1))


class Counts:
    """Per-conflict-count cell counts, with prefix sums so an interval costs one pass."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.optima = sorted({r["oracle_optimum"] for r in rows})
        self.tightness = sorted({r["tightness"] for r in rows})
        self.structures = sorted({r["travel_structure"] for r in rows})
        self.cells = list(product(self.optima, self.tightness, self.structures))
        by_k: dict[int, Counter] = defaultdict(Counter)
        for r in rows:
            by_k[r["conflicting_pairs"]][
                (r["oracle_optimum"], r["tightness"], r["travel_structure"])] += 1
        # prefix[k][cell] = number of candidates with conflict count < k
        self.prefix: list[dict[tuple, int]] = [dict.fromkeys(self.cells, 0)]
        for k in range(MAX_PAIRS + 1):
            prev = self.prefix[-1]
            self.prefix.append({c: prev[c] + by_k[k][c] for c in self.cells})

        # Every interval is aggregated once. The enumeration visits a few hundred thousand
        # triples, so rebuilding a cell dictionary per visit dominates everything else;
        # there are only 435 intervals, and the per-optimum vector and total are the two
        # things the pruning needs.
        self._cache: dict[tuple[int, int], dict[tuple, int]] = {}
        self._optima_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._total_cache: dict[tuple[int, int], int] = {}
        for lo in range(MAX_PAIRS + 1):
            for hi in range(lo, MAX_PAIRS + 1):
                a, b = self.prefix[lo], self.prefix[hi + 1]
                avail = {c: b[c] - a[c] for c in self.cells}
                self._cache[(lo, hi)] = avail
                per_o: dict[int, int] = dict.fromkeys(self.optima, 0)
                for (o, _t, _s), v in avail.items():
                    per_o[o] += v
                self._optima_cache[(lo, hi)] = tuple(per_o[o] for o in self.optima)
                self._total_cache[(lo, hi)] = sum(avail.values())

    def interval(self, lo: int, hi: int) -> dict[tuple, int]:
        return self._cache[(lo, hi)]

    def optima_vector(self, lo: int, hi: int) -> tuple[int, ...]:
        return self._optima_cache[(lo, hi)]

    def total(self, lo: int, hi: int) -> int:
        return self._total_cache[(lo, hi)]

    def by_optimum(self, avail: dict[tuple, int]) -> dict[int, int]:
        out: dict[int, int] = dict.fromkeys(self.optima, 0)
        for (o, _t, _s), v in avail.items():
            out[o] += v
        return out


def optimum_capacity(counts: Counts, avails: Sequence[dict[tuple, int]]) -> int:
    """Upper bound on a matched sample, ignoring diversity: sum_o min_b n[b][o].

    A cheap necessary condition, used to prune before the solver is called. It cannot
    certify a design, because it ignores whether the matching candidates carry enough
    tightness and structure variety.
    """
    per_band = [counts.by_optimum(a) for a in avails]
    return sum(min(p[o] for p in per_band) for o in counts.optima)


def joint_allocation(counts: Counts, avails: Sequence[dict[tuple, int]], *,
                     minimum: int | None = None, maximise: bool = False
                     ) -> tuple[bool, int, dict[str, Any]]:
    """Does a selection exist meeting every binding requirement at once?

    Decision variables are how many candidates to take from each (optimum, tightness,
    structure) cell of each band. The `O` histogram is a shared vector, so it is identical
    across bands by construction rather than by approximation.
    """
    model = cp_model.CpModel()
    n_b = len(avails)
    x = {}
    for b, avail in enumerate(avails):
        for cell, cap in avail.items():
            x[b, cell] = model.NewIntVar(0, cap, f"x{b}_{cell}")

    ub = {o: min(counts.by_optimum(a)[o] for a in avails) for o in counts.optima}
    c = {o: model.NewIntVar(0, ub[o], f"c{o}") for o in counts.optima}
    for b in range(n_b):
        for o in counts.optima:
            model.Add(sum(x[b, (o, t, s)]
                          for t in counts.tightness for s in counts.structures) == c[o])

    total = sum(ub.values())
    m = model.NewIntVar(0, total, "m")
    model.Add(m == sum(c.values()))

    for b in range(n_b):
        used_t = []
        for t in counts.tightness:
            xt = sum(x[b, (o, t, s)] for o in counts.optima for s in counts.structures)
            # no tightness value above 60% of the selected band sample
            model.Add(MAX_TIGHTNESS_SHARE_DEN * xt <= MAX_TIGHTNESS_SHARE_NUM * m)
            u = model.NewBoolVar(f"u{b}_{t}")
            model.Add(u <= xt)
            used_t.append(u)
        model.Add(sum(used_t) >= MIN_TIGHTNESS_VALUES)

        used_s = []
        for s in counts.structures:
            xs = sum(x[b, (o, t, s)] for o in counts.optima for t in counts.tightness)
            v = model.NewBoolVar(f"v{b}_{s}")
            model.Add(v <= xs)
            used_s.append(v)
        model.Add(sum(used_s) >= MIN_TRAVEL_STRUCTURES)

    if minimum is not None:
        model.Add(m >= minimum)
    if maximise:
        model.Maximize(m)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1      # determinism, as everywhere else here
    solver.parameters.random_seed = _SOLVER_SEED
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not ok:
        return False, 0, {}
    value = solver.Value(m)
    plan = {
        "optimum_histogram": {str(o): solver.Value(c[o]) for o in counts.optima
                              if solver.Value(c[o])},
        "per_band": [
            {f"{o}|{t}|{s}": solver.Value(x[b, (o, t, s)])
             for (o, t, s) in counts.cells if solver.Value(x[b, (o, t, s)])}
            for b in range(n_b)
        ],
    }
    return True, value, plan


def _survives_prefilter(counts: Counts, bands, target: int) -> bool:
    """Necessary conditions implied by the binding criteria, checked without the solver.

    None of these is a new requirement. Each must hold for any valid selection, so a triple
    they reject cannot be admissible: the bands must be large enough; the optimum histograms
    must have `target` in common; and within the optima that can be matched, each band must
    still be able to supply `target` without any tightness exceeding the 60% cap, and must
    carry two tightness values and two travel structures.
    """
    if any(counts.total(lo, hi) < target for lo, hi in bands):
        return False
    vectors = [counts.optima_vector(lo, hi) for lo, hi in bands]
    cap = [min(v[i] for v in vectors) for i in range(len(counts.optima))]
    if sum(cap) < target:
        return False
    live = {counts.optima[i] for i in range(len(counts.optima)) if cap[i] > 0}
    ceiling = (MAX_TIGHTNESS_SHARE_NUM * target) // MAX_TIGHTNESS_SHARE_DEN
    for lo, hi in bands:
        per_t: dict[float, int] = dict.fromkeys(counts.tightness, 0)
        per_s: dict[str, int] = dict.fromkeys(counts.structures, 0)
        for (o, t, s), n in counts.interval(lo, hi).items():
            if n and o in live:
                per_t[t] += n
                per_s[s] += n
        if sum(min(v, ceiling) for v in per_t.values()) < target:
            return False
        if sum(1 for v in per_t.values() if v) < MIN_TIGHTNESS_VALUES:
            return False
        if sum(1 for v in per_s.values() if v) < MIN_TRAVEL_STRUCTURES:
            return False
    return True


def search(rows: Sequence[dict[str, Any]], *, target: int = TARGET_PER_BAND,
           max_k: int = MAX_PAIRS) -> dict[str, Any]:
    counts = Counts(rows)
    considered = pruned = solved = 0
    admissible: list[dict[str, Any]] = []

    # Enumerated by DESCENDING separation, because the tie-break maximises it first: the
    # first separation with any admissible triple is the winning one, and triples with a
    # smaller gap could never be selected. The choice is therefore identical to visiting
    # every triple, and only the count of admissible triples at unreachable separations is
    # left unmeasured. A verdict of NO_FEASIBLE_DESIGN still enumerates everything, since
    # it can only be reached by exhausting every separation.
    by_gap: dict[int, list] = defaultdict(list)
    for bands in band_triples(max_k=max_k):
        considered += 1
        by_gap[separation(bands)].append(bands)

    selected_gap = None
    for gap in sorted(by_gap, reverse=True):
        for bands in by_gap[gap]:
            if not _survives_prefilter(counts, bands, target):
                pruned += 1
                continue
            solved += 1
            avails = [counts.interval(lo, hi) for lo, hi in bands]
            ok, _m, _plan = joint_allocation(counts, avails, minimum=target)
            if ok:
                admissible.append({"bands": [list(b) for b in bands],
                                   "separation": gap})
        if admissible:
            selected_gap = gap
            break

    result: dict[str, Any] = {
        "triples_considered": considered,
        "triples_pruned_before_solver": pruned,
        "triples_sent_to_solver": solved,
        "n_admissible_at_selected_separation": len(admissible),
        "enumeration": ("descending separation, stopped at the first admissible level; "
                        "triples with a smaller gap cannot win the tie-break"),
        "target_per_band": target,
    }
    if not admissible:
        result["verdict"] = "NO_FEASIBLE_DESIGN"
        result["n_admissible"] = 0
        result["selected"] = None
        return result

    # Tie-break, fixed before the search ran and applied without discretion:
    #   1. the largest minimum gap between adjacent bands   (the loop above)
    #   2. the largest joint matched capacity
    #   3. the lexicographically smallest boundary tuple
    best_gap = selected_gap
    finalists = list(admissible)
    for a in finalists:
        avails = [counts.interval(lo, hi) for lo, hi in a["bands"]]
        _ok, cap, plan = joint_allocation(counts, avails, minimum=target, maximise=True)
        a["joint_capacity"] = cap
        a["allocation"] = plan
    best_cap = max(a["joint_capacity"] for a in finalists)
    finalists = [a for a in finalists if a["joint_capacity"] == best_cap]
    finalists.sort(key=lambda a: tuple(v for b in a["bands"] for v in b))

    result["verdict"] = "FEASIBLE"
    result["tie_break"] = {"max_separation": best_gap, "max_joint_capacity": best_cap,
                           "finalists_after_capacity": len(finalists)}
    result["selected"] = finalists[0]
    return result


def diagnose(rows: Sequence[dict[str, Any]], bands: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Non-binding description of the selected design. Never feeds the decision."""
    out = {}
    for i, (lo, hi) in enumerate(bands):
        g = [r for r in rows if lo <= r["conflicting_pairs"] <= hi]
        out[f"band{i + 1}"] = {
            "pairs": [lo, hi],
            "D": [round(lo / MAX_PAIRS, 4), round(hi / MAX_PAIRS, 4)],
            "n_raw": len(g),
            "overlap_counts": {str(k): v for k, v in
                               sorted(Counter(r["overlap"] for r in g).items())},
            "H_counts": dict(sorted(Counter(r["higher_order_gap_H"] for r in g).items())),
            "alpha_mean": round(sum(r["alpha_reachable"] for r in g) / len(g), 3) if g else None,
            "edge_share_top1_mean": (
                round(sum(r["edge_share_top1"] for r in g if r["edge_share_top1"] is not None)
                      / max(1, sum(1 for r in g if r["edge_share_top1"] is not None)), 4)),
            "triangle_violations_mean": round(
                sum(r["triangle_violations"] for r in g) / len(g), 2) if g else None,
        }
    return out


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    s = doc["search"]

    add("# Three-band design search — exhaustive, outcome-blind\n")
    add("This search answers whether the **previously registered** design is reachable. The "
        "earlier rejection concerned one proposal that was stricter than anything "
        "registered, so it could not settle the question.\n")
    add("**Status of the ingredients.** Previously registered: separated `D` levels, at "
        "least 50 per band, at least two tightness values and two travel structures per "
        "band, no tightness above 60%, comparable `O` distribution. Prospectively fixed "
        "before this search ran, with the structural pool already seen and no new LLM "
        "outcome seen: \"comparable\" means an identical `O` histogram, the requirements "
        "hold jointly on the selected sample, bands are contiguous with at least one unused "
        "value between them, and the tie-break below. Post-specified and non-binding: `H`, "
        "`overlap`, `alpha_reachable`, conflict concentration, triangle statistics — none of "
        "these can fail a design, enter the objective, or break a tie.\n")
    add(f"Triples considered: **{s['triples_considered']}**; pruned before the solver: "
        f"{s['triples_pruned_before_solver']}; decided by CP-SAT: "
        f"{s['triples_sent_to_solver']}; admissible at the deciding separation: "
        f"**{s['n_admissible_at_selected_separation']}**.\n")
    add(f"Enumeration: {s['enumeration']}.\n")

    if s["verdict"] == "NO_FEASIBLE_DESIGN":
        add("\n## Verdict: NO_FEASIBLE_DESIGN\n")
        add("No triple of separated bands admits a selection of "
            f"{s['target_per_band']} instances per band with an identical `O` histogram and "
            "the registered diversity requirements met simultaneously.\n")
        add("No boundary, matching variable or threshold is changed in response. The "
            "recorded fallback applies.\n")
    else:
        sel = s["selected"]
        add("\n## Verdict: FEASIBLE\n")
        add(f"Selected by the fixed tie-break — largest minimum gap "
            f"({s['tie_break']['max_separation']}), then largest joint capacity "
            f"({s['tie_break']['max_joint_capacity']}), then lexicographic order.\n")
        add("| band | pairs | D | joint capacity |")
        add("|---|---|---|---|")
        for i, (lo, hi) in enumerate(sel["bands"], 1):
            add(f"| {i} | {lo}–{hi} | {lo / MAX_PAIRS:.3f}–{hi / MAX_PAIRS:.3f} | "
                f"{sel['joint_capacity']} |")
        add(f"\nIdentical `O` histogram across the three bands: "
            f"`{sel['allocation']['optimum_histogram']}`\n")
        add("\n## Non-binding diagnostics for the selected design\n")
        add("Reported because they describe the design, not because they justified it. An "
            "unattractive value here is a residual limitation carried into the analysis, "
            "never a reason to search again.\n")
        add("| band | pairs | raw n | mean alpha | mean edge share top1 | H counts | overlap counts |")
        add("|---|---|---|---|---|---|---|")
        for name, d in doc["diagnostics"].items():
            add(f"| {name} | {d['pairs'][0]}–{d['pairs'][1]} | {d['n_raw']} | "
                f"{d['alpha_mean']} | {d['edge_share_top1_mean']} | {d['H_counts']} | "
                f"{d['overlap_counts']} |")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=Path("results/calibration/structural"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/band_search"))
    ap.add_argument("--target", type=int, default=TARGET_PER_BAND)
    args = ap.parse_args(argv)

    rows, meta = load_pool(args.pool)
    result = search(rows, target=args.target)
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pool_dir": args.pool.as_posix(),
        "pool_commit": meta.get("git_commit"),
        "non_binding_diagnostics": list(NON_BINDING),
        "search": result,
    }
    if result["selected"]:
        doc["diagnostics"] = diagnose(rows, result["selected"]["bands"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    print(f"considered {result['triples_considered']} triples, "
          f"{result['triples_sent_to_solver']} reached the solver, "
          f"{result['n_admissible_at_selected_separation']} admissible at the "
          "deciding separation")
    print(f"VERDICT: {result['verdict']}")
    if result["selected"]:
        sel = result["selected"]
        print(f"  bands {sel['bands']}  gap {sel['separation']}  "
              f"joint capacity {sel['joint_capacity']}")
        print(f"  identical O histogram {sel['allocation']['optimum_histogram']}")
    print(f"written: {args.out}/summary.json, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
