# scripts/build_results_data.py
"""Neutral data extraction for the held-out experiment. CPU, no LLM, no new tests.

    uv run --with matplotlib --with numpy python -m scripts.build_results_data

WHAT THIS IS. A factual result package. It reads only committed artefacts and writes
`results/analysis/heldout/RESULTS_DATA.md` plus neutral figures under
`results/analysis/heldout/figures/data/`. It performs NO hypothesis test, fits nothing,
and states no interpretation. Every number is either quoted from a committed artefact or
a direct descriptive statistic (mean/median/SD/percentile/rate/count) of the committed
per-run export. The frozen analysis files are read, never written.

INPUTS (committed):
  results/exports/heldout/runs.csv            (per run; latency_seconds & optimality live
                                               in the details_json column)
  results/exports/heldout/aggregates.csv      (committed per-cell satisfaction stats)
  results/analysis/heldout/contrasts.csv      (frozen pairwise deltas + bootstrap CIs)
  results/analysis/heldout/summary.json       (frozen confirmatory family + H sensitivity)
  results/analysis/heldout/completeness_audit.json
  results/calibration/heldout_n8/candidates.csv (higher_order_gap_H per instance)

CONVENTIONS. Satisfaction cell statistics (mean/median/SD/min/max) are quoted verbatim
from `aggregates.csv`. All statistics computed here from `runs.csv` (tokens, calls,
latency, optimality, rates, satisfaction-per-1k) use the sample SD (ddof=1); this is
stated in the document. Percentiles are linear-interpolation (numpy default).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
RUNS = _REPO / "results/exports/heldout/runs.csv"
AGG = _REPO / "results/exports/heldout/aggregates.csv"
CONTRASTS = _REPO / "results/analysis/heldout/contrasts.csv"
SUMMARY = _REPO / "results/analysis/heldout/summary.json"
AUDIT = _REPO / "results/analysis/heldout/completeness_audit.json"
POOL = _REPO / "results/calibration/heldout_n8/candidates.csv"
# The descriptive package is separate from the frozen confirmatory analysis
# (THESIS_DECISIONS section 5, amendment 2026-09-02). Figures I and J are the
# exception: they are specific to cap 64000 and to the C1-C3 pair, so they stay
# with the frozen analysis as its appendix rather than in the descriptive set.
OUT_MD = _REPO / "results/analysis/heldout_full_matrix/RESULTS_DATA.md"
FIG_DIR = _REPO / "results/analysis/heldout_full_matrix/figures"
APPENDIX_FIG_DIR = _REPO / "results/analysis/heldout/figures"
APPENDIX_STEMS = ("I_", "J_")

CONDS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
SHORT = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
         "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
DEF = {
    "c1_react": "single ReAct agent (reason-act-observe loop, no mandated verification stage)",
    "c2_verify_revise": "single ReAct agent with fixed draft, verify and revise steps",
    "c3_mas": "hierarchical multi-agent system (supervisor, workers, aggregator, critic)",
    "c4_planner_critic": "single planner with a separate fresh-context critic (no decomposition)",
    "c5_best_of_3": "best-of-3: three independent attempts with selection",
}
CAPS = (16000, 32000, 64000, 128000)
CAPK = {16000: "16k", 32000: "32k", 64000: "64k", 128000: "128k"}
BANDS = ("low", "medium", "high")
SMALL_N = (4, 5, 6)
COLOR = {"c1_react": "#4D4D4D", "c2_verify_revise": "#0072B2", "c3_mas": "#D55E00",
         "c4_planner_critic": "#009E73", "c5_best_of_3": "#CC79A7"}
MARKER = {"c1_react": "o", "c2_verify_revise": "s", "c3_mas": "D",
          "c4_planner_critic": "^", "c5_best_of_3": "v"}
LEGLAB = {"c1_react": "C1 ReAct", "c2_verify_revise": "C2 verify/revise",
          "c3_mas": "C3 hierarchical MAS", "c4_planner_critic": "C4 planner+critic",
          "c5_best_of_3": "C5 best-of-3"}


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_runs() -> list[dict[str, Any]]:
    out = []
    for r in csv.DictReader(RUNS.open(encoding="utf-8")):
        d = json.loads(r["details_json"])
        out.append({
            "condition": r["condition"], "cap": int(r["cap"]), "block": r["block"],
            "band": r["band"], "n_people": int(r["n_people"]),
            "satisfaction": float(r["satisfaction"]), "valid": int(r["valid"]),
            "n_meetings": int(r["n_meetings"]), "tokens": int(r["tokens_total"]),
            "calls": int(r["n_calls"]), "termination": r["termination"],
            "optimality": bool((d.get("score") or {}).get("optimality")),
            "latency": float(d["latency_seconds"]), "cap_util": float(d["cap_utilisation"]),
        })
    return out


def load_aggregates() -> dict[tuple[str, int, str], dict[str, str]]:
    out = {}
    for r in csv.DictReader(AGG.open(encoding="utf-8")):
        if r["grouping"] == "condition x cap x stratum":
            out[(r["condition"], int(r["cap"]), r["stratum"])] = r
    return out


def load_contrasts() -> list[dict[str, str]]:
    return list(csv.DictReader(CONTRASTS.open(encoding="utf-8")))


def load_H() -> dict[str, int]:
    out = {}
    for r in csv.DictReader(POOL.open(encoding="utf-8")):
        out[r["instance_id"]] = int(r["higher_order_gap_H"])
    return out


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def stats(vals: Sequence[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75))}


def md(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    h = "| " + " | ".join(str(x) for x in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    return "\n".join([h, sep, body])


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #
def build_document(runs, agg, contrasts, summary, audit, H) -> str:
    L: list[str] = []
    add = L.append
    A = [r for r in runs if r["block"] == "A"]
    B = [r for r in runs if r["block"] == "B"]

    def selA(cond, cap, band):
        return [r for r in A if r["condition"] == cond and r["cap"] == cap
                and r["band"] == band]

    def poolA(cond, cap):
        return [r for r in A if r["condition"] == cond and r["cap"] == cap]

    def selB(cond, cap, n):
        return [r for r in B if r["condition"] == cond and r["cap"] == cap
                and r["n_people"] == n]

    # ---- header ----
    add("# Held-out experiment — data extraction (RESULTS_DATA)")
    add("")
    add("Factual reference. Tables and figure pointers only; no interpretation, no "
        "discussion, no conclusions. Generated by `scripts/build_results_data.py` from "
        "committed artefacts. Satisfaction cell statistics are quoted from "
        "`results/exports/heldout/aggregates.csv`; statistics computed here from "
        "`results/exports/heldout/runs.csv` use the sample SD (ddof=1). Frozen files "
        "(`summary.json`, `report.md`, `contrasts.csv`, manifests, analyser) are read, "
        "never modified.")
    add("")

    # ---- 1. completeness ----
    add("## 1. Experiment completeness")
    add("")
    add(md(["field", "value"], [
        ["total instances", audit["n_verified"] // 20],
        ["total runs", audit["n_verified"]],
        ["architectures", "5 (C1, C2, C3, C4, C5)"],
        ["budgets", "4 (16000, 32000, 64000, 128000)"],
        ["Block A sizes", "n = 8 (150 instances: 50 low, 50 medium, 50 high)"],
        ["Block B sizes", "n = 4, 5, 6 (16 instances each; 48 total)"],
        ["model", ", ".join(audit["models"])],
        ["n_expected", audit["n_expected"]],
        ["n_verified", audit["n_verified"]],
        ["n_missing", audit["n_missing"]],
        ["n_incompatible", audit["n_incompatible"]],
        ["n_unexpected_files", audit["n_unexpected_files"]],
        ["all_checks_passed", audit["all_checks_passed"]],
    ]))
    add("")
    add("Verified runs by source manifest: " + ", ".join(
        f"{k} = {v}" for k, v in audit["verified_by_source_manifest"].items()) + ".")
    add("Every `condition@cap` cell holds "
        + str(sorted(set(audit["verified_by_condition_and_cap"].values()))[0])
        + " verified runs (all 20 cells).")
    add("")

    # ---- 2. architectures ----
    add("## 2. Architectures")
    add("")
    add(md(["id", "definition"], [[SHORT[c], DEF[c]] for c in CONDS]))
    add("")

    # ---- 3. satisfaction full Block A matrix ----
    add("## 3. Satisfaction — full Block A matrix (n = 8)")
    add("")
    add("Per cell: mean (SD), quoted from `aggregates.csv`. n = 50 for every cell. "
        "Per-condition mean confidence intervals are not present in the committed "
        "artefacts; difference intervals are in §6 and §7.")
    add("")
    for cap in CAPS:
        rows = []
        for band in BANDS:
            cells = []
            for c in CONDS:
                a = agg[(c, cap, f"band={band}")]
                cells.append(f"{float(a['mean_satisfaction']):.3f} ({float(a['sd_satisfaction']):.3f})")
            rows.append([band] + cells)
        add(f"**cap {CAPK[cap]}** — mean (SD)")
        add("")
        add(md(["band"] + [SHORT[c] for c in CONDS], rows))
        add("")

    # ---- 4. satisfaction by budget ----
    add("## 4. Satisfaction by budget (one table per band)")
    add("")
    add("Mean satisfaction (SD), from `aggregates.csv`. n = 50 per cell.")
    add("")
    for band in BANDS:
        rows = []
        for c in CONDS:
            cells = []
            for cap in CAPS:
                a = agg[(c, cap, f"band={band}")]
                cells.append(f"{float(a['mean_satisfaction']):.3f} ({float(a['sd_satisfaction']):.3f})")
            rows.append([SHORT[c]] + cells)
        add(f"**{band}**")
        add("")
        add(md(["arch"] + [CAPK[c] for c in CAPS], rows))
        add("")

    # ---- 5. satisfaction by complexity ----
    add("## 5. Satisfaction by complexity (one table per budget)")
    add("")
    add("Mean satisfaction (SD), from `aggregates.csv`. n = 50 per cell.")
    add("")
    for cap in CAPS:
        rows = []
        for c in CONDS:
            cells = []
            for band in BANDS:
                a = agg[(c, cap, f"band={band}")]
                cells.append(f"{float(a['mean_satisfaction']):.3f} ({float(a['sd_satisfaction']):.3f})")
            rows.append([SHORT[c]] + cells)
        add(f"**cap {CAPK[cap]}**")
        add("")
        add(md(["arch"] + list(BANDS), rows))
        add("")

    # ---- 6. pairwise contrasts ----
    add("## 6. Pairwise contrasts (from `contrasts.csv`)")
    add("")
    add("Sign convention (frozen): for a pair labelled `X->Y`, "
        "`delta = satisfaction(Y) - satisfaction(X)`. The ten unordered pairs are "
        "oriented by the analyser's bookkeeping order "
        "`c1_react < c2_verify_revise < c5_best_of_3 < c4_planner_critic < c3_mas`; this "
        "orientation is not a ranking. Columns are copied from `contrasts.csv`: "
        "mean_delta, n_pairs, sd, skewness, tie_rate, later_ahead, earlier_ahead, tied, "
        "ci_low, ci_high (descriptive bootstrap interval). Significance is not asserted "
        "here; the only inferential results are the frozen family in §8.")
    add("")
    for block, strata in (("A", [f"band={b}" for b in BANDS]),
                          ("B", [f"n={n}" for n in SMALL_N])):
        for cap in CAPS:
            rows = []
            for r in contrasts:
                if r["block"] == block and int(r["cap"]) == cap:
                    rows.append([r["contrast"], r["stratum"], r["n_pairs"],
                                 f"{float(r['mean_delta']):+.3f}", f"{float(r['sd']):.3f}",
                                 f"{float(r['skewness']):+.2f}", f"{float(r['tie_rate']):.2f}",
                                 r["later_ahead"], r["earlier_ahead"], r["tied"],
                                 f"{float(r['ci_low']):+.3f}", f"{float(r['ci_high']):+.3f}"])
            add(f"**Block {block} — cap {CAPK[cap]}**")
            add("")
            add(md(["pair", "stratum", "n", "mean_delta", "sd", "skew", "tie_rate",
                    "later_ahead", "earlier_ahead", "tied", "ci_low", "ci_high"], rows))
            add("")

    # ---- 7. C1-C3 expose-sign table ----
    add("## 7. C1–C3 with the exposé sign convention")
    add("")
    add("Derived by arithmetic sign conversion of the committed `C1->C3` rows: "
        "`Delta = satisfaction(C1) - satisfaction(C3) = -(mean_delta of C1->C3)`, with the "
        "descriptive bootstrap interval re-signed `[-ci_high, -ci_low]`. Block A, n = 50 "
        "per cell.")
    add("")
    c13 = {(int(r["cap"]), r["stratum"].split("=")[1]): r for r in contrasts
           if r["block"] == "A" and r["contrast"] == "C1->C3"}
    rows = []
    for cap in CAPS:
        cells = []
        for band in BANDS:
            r = c13[(cap, band)]
            m = -float(r["mean_delta"]); lo = -float(r["ci_high"]); hi = -float(r["ci_low"])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        rows.append([CAPK[cap]] + cells)
    add(md(["cap", "Delta low [CI]", "Delta medium [CI]", "Delta high [CI]"], rows))
    add("")

    # ---- 8. formal frozen analysis ----
    add("## 8. Formal frozen analysis (reproduced from `summary.json`)")
    add("")
    conf = summary["confirmatory"]
    add(f"Primary cap {summary['primary_cap']}; alpha {summary['alpha']}; "
        f"{conf['permutations']} permutations; Holm across "
        f"{conf['n_hypotheses']} p-values. Analyser sign convention: "
        "`delta = satisfaction(later) - satisfaction(earlier)`; crossover components are "
        "`delta_low` (one-sided lower) and `delta_high` (one-sided upper), "
        "`p_iut = max(p_low, p_high)`. Values are copied from the frozen output.")
    add("")
    rows = []
    for res in conf["contrasts"]:
        b = res["bands"]; cr = res["crossover"]; it = res["interaction"]
        rows.append([
            res["contrast"],
            f"{b['low']['mean_delta']:+.3f}", f"{b['medium']['mean_delta']:+.3f}",
            f"{b['high']['mean_delta']:+.3f}", f"{it['delta_statistic']:+.3f}",
            f"{it['p_two_sided']:.4f}", f"{it['holm']['p_holm']:.4f}",
            f"{cr['p_iut']:.4f}", f"{cr['holm']['p_holm']:.4f}",
            str(cr['holm']['reject']), str(cr['established'])])
    add(md(["contrast", "d_low", "d_med", "d_high", "interaction_stat",
            "p_int_raw", "p_int_holm", "p_iut_raw", "p_iut_holm",
            "iut_reject", "crossover_established"], rows))
    add("")
    add("Per-band ties and skewness, and the descriptive bootstrap intervals, for the "
        "five registered contrasts at the primary cap:")
    add("")
    rows = []
    for res in conf["contrasts"]:
        for band in ("low", "medium", "high"):
            s = res["bands"][band]; ci = s["bootstrap_ci_descriptive"]
            rows.append([res["contrast"], band, s["n_pairs"], f"{s['mean_delta']:+.3f}",
                         f"{s['sd']:.3f}", f"{s['tie_rate']:.2f}", f"{s['skewness']:+.2f}",
                         f"{ci['low']:+.3f}", f"{ci['high']:+.3f}"])
    add(md(["contrast", "band", "n", "mean_delta", "sd", "tie_rate", "skew",
            "ci_low", "ci_high"], rows))
    add("")
    add(f"`any_crossover_established`: {conf['any_crossover_established']}. Sign-test "
        "sensitivity p-values (different functional `P(delta>0)`, from the frozen output):")
    add("")
    rows = [[res["contrast"], f"{res['sensitivity_sign_test']['p_iut']:.4f}",
             f"{res['crossover']['p_iut']:.4f}"] for res in conf["contrasts"]]
    add(md(["contrast", "sign_test_p_iut", "permutation_p_iut"], rows))
    add("")

    # ---- helper for per (cond,cap) numeric tables ----
    def per_cond_cap_table(metric, fmt, pool=poolA, extra_pct=False):
        rows = []
        for c in CONDS:
            for cap in CAPS:
                g = pool(c, cap)
                vals = [metric(r) for r in g]
                s = stats(vals)
                cells = [SHORT[c], CAPK[cap], s["n"], fmt(s["mean"]), fmt(s["median"]),
                         fmt(s["sd"]), fmt(s["min"]), fmt(s["max"])]
                if extra_pct:
                    cells.append(f"{100*np.mean([r['cap_util'] for r in g]):.1f}%")
                rows.append(cells)
        return rows

    # ---- 9. tokens ----
    add("## 9. Token use (Block A, computed from `runs.csv`)")
    add("")
    add("Per architecture × budget over the 150 Block A runs. `%cap` is the mean of "
        "`cap_utilisation = tokens_total / cap`.")
    add("")
    add(md(["arch", "cap", "n", "mean", "median", "sd", "min", "max", "%cap"],
           per_cond_cap_table(lambda r: r["tokens"], lambda x: f"{x:.0f}", extra_pct=True)))
    add("")
    add("Block A token means split by band (mean tokens):")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            cells = [SHORT[c], CAPK[cap]]
            for band in BANDS:
                cells.append(f"{np.mean([r['tokens'] for r in selA(c,cap,band)]):.0f}")
            rows.append(cells)
    add(md(["arch", "cap", "low", "medium", "high"], rows))
    add("")

    # ---- 10. calls ----
    add("## 10. Model calls (Block A, computed from `runs.csv`)")
    add("")
    add(md(["arch", "cap", "n", "mean", "median", "sd", "min", "max"],
           per_cond_cap_table(lambda r: r["calls"], lambda x: f"{x:.2f}")))
    add("")

    # ---- 11. latency ----
    add("## 11. Latency — `RunResult.latency_seconds` (Block A, from the `details_json` "
        "column of `runs.csv`)")
    add("")
    add("Seconds. This is the per-run `latency_seconds` field recorded by the harness "
        "(sum of model-call wall time for the run), not any Slurm/queue/startup time.")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            s = stats([r["latency"] for r in poolA(c, cap)])
            rows.append([SHORT[c], CAPK[cap], s["n"], f"{s['mean']:.1f}", f"{s['median']:.1f}",
                         f"{s['sd']:.1f}", f"{s['p25']:.1f}", f"{s['p75']:.1f}",
                         f"{s['min']:.1f}", f"{s['max']:.1f}"])
    add(md(["arch", "cap", "n", "mean", "median", "sd", "p25", "p75", "min", "max"], rows))
    add("")
    add("Block A latency means split by band (mean seconds):")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            cells = [SHORT[c], CAPK[cap]]
            for band in BANDS:
                cells.append(f"{np.mean([r['latency'] for r in selA(c,cap,band)]):.1f}")
            rows.append(cells)
    add(md(["arch", "cap", "low", "medium", "high"], rows))
    add("")

    # ---- 12. satisfaction per 1000 tokens ----
    add("## 12. Satisfaction per 1000 tokens (Block A, derived)")
    add("")
    add("Descriptive derived metric: `sat_per_1k = mean_satisfaction / mean_tokens * 1000` "
        "per architecture × budget over the 150 Block A runs.")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            g = poolA(c, cap)
            ms = np.mean([r["satisfaction"] for r in g]); mt = np.mean([r["tokens"] for r in g])
            rows.append([SHORT[c], CAPK[cap], f"{ms:.3f}", f"{mt:.0f}", f"{ms/mt*1000:.4f}"])
    add(md(["arch", "cap", "mean_satisfaction", "mean_tokens", "sat_per_1k"], rows))
    add("")

    # ---- 13. validity / non-empty ----
    add("## 13. Validity and non-empty (Block A, computed from `runs.csv`)")
    add("")
    add("Definitions. **valid**: the final plan violates no hard constraint. "
        "**valid-non-empty**: valid and containing at least one scheduled meeting. "
        "**empty-plan rate**: share of runs whose final plan has zero meetings. "
        "Implementation note: an empty plan violates no hard constraint, so it is valid; "
        "therefore `valid rate` and `valid-non-empty rate` are different metrics.")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            g = poolA(c, cap); n = len(g)
            vr = np.mean([r["valid"] for r in g])
            vne = np.mean([1.0 if (r["valid"] == 1 and r["n_meetings"] > 0) else 0.0 for r in g])
            emp = np.mean([1.0 if r["n_meetings"] == 0 else 0.0 for r in g])
            rows.append([SHORT[c], CAPK[cap], n, f"{vr:.3f}", f"{vne:.3f}", f"{emp:.3f}"])
    add(md(["arch", "cap", "n", "valid_rate", "valid_nonempty_rate", "empty_plan_rate"], rows))
    add("")

    # ---- 14. optimality ----
    add("## 14. Optimality (Block A, computed from `runs.csv`)")
    add("")
    add("`optimality` is the frozen per-run `score.optimality` field (the plan is valid and "
        "reaches the solver optimum). Rate, count optimal, count non-optimal over 150 runs.")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            g = poolA(c, cap); n = len(g); k = sum(1 for r in g if r["optimality"])
            rows.append([SHORT[c], CAPK[cap], n, f"{k/n:.3f}", k, n - k])
    add(md(["arch", "cap", "n", "optimality_rate", "n_optimal", "n_non_optimal"], rows))
    add("")
    add("Optimality rate split by band:")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            cells = [SHORT[c], CAPK[cap]]
            for band in BANDS:
                g = selA(c, cap, band)
                cells.append(f"{np.mean([1.0 if r['optimality'] else 0.0 for r in g]):.3f}")
            rows.append(cells)
    add(md(["arch", "cap", "low", "medium", "high"], rows))
    add("")

    # ---- 15. termination ----
    add("## 15. Termination (Block A, counts from `runs.csv`)")
    add("")
    cats = sorted({r["termination"] for r in A})
    add("Recorded categories: " + ", ".join(f"`{x}`" for x in cats) + ".")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            g = poolA(c, cap); cc = Counter(r["termination"] for r in g)
            rows.append([SHORT[c], CAPK[cap], len(g)] + [cc.get(x, 0) for x in cats])
    add(md(["arch", "cap", "n"] + [f"`{x}`" for x in cats], rows))
    add("")

    # ---- 16. Block B ----
    add("## 16. Block B (n = 4, 5, 6 — kept separate; `n` is a generator parameter, not a "
        "complexity band)")
    add("")
    add("Per architecture × size × budget over the 16 Block B runs per cell. All "
        "seven reported metrics: satisfaction, valid-non-empty rate, optimality "
        "rate, tokens, calls, latency (seconds) and efficiency. Efficiency is "
        "`mean_satisfaction / mean_tokens * 1000`, the same ratio-of-means "
        "definition used for Block A in section 12.")
    add("")
    add("**The size is not collapsed.** Block B exists to show what changes when the "
        "task gets smaller, so `n = 4`, `n = 5` and `n = 6` keep separate values for "
        "every metric, condition and budget. The pooled summary at the end of this "
        "section is an additional view and never a substitute: pooling would hide any "
        "dependence on `n`, which is the one thing this block is here to show.")
    add("")
    for n in SMALL_N:
        rows = []
        for c in CONDS:
            for cap in CAPS:
                g = selB(c, cap, n)
                ms = np.mean([r["satisfaction"] for r in g])
                mt = np.mean([r["tokens"] for r in g])
                mc = np.mean([r["calls"] for r in g])
                ml = np.mean([r["latency"] for r in g])
                vne = np.mean([1.0 if (r["valid"] == 1 and r["n_meetings"] > 0) else 0.0 for r in g])
                opt = np.mean([1.0 if r["optimality"] else 0.0 for r in g])
                eff = ms / mt * 1000.0 if mt else float("nan")
                rows.append([SHORT[c], CAPK[cap], len(g), f"{ms:.3f}", f"{mt:.0f}",
                             f"{mc:.2f}", f"{ml:.1f}", f"{vne:.3f}", f"{opt:.3f}",
                             f"{eff:.4f}"])
        add(f"**n = {n}**")
        add("")
        add(md(["arch", "cap", "n_runs", "mean_sat", "mean_tokens", "mean_calls",
                "mean_latency_s", "valid_nonempty", "optimality", "sat_per_1k"],
               rows))
        add("")

    add("### Pooled over n = 4, 5, 6 — additional summary only")
    add("")
    add("48 runs per cell. Reported because a single overview is convenient; it does "
        "**not** replace the three tables above, and any statement about the effect of "
        "task size must be read from those.")
    add("")
    rows = []
    for c in CONDS:
        for cap in CAPS:
            g = [r for r in runs if r["block"] == "B" and r["condition"] == c
                 and r["cap"] == cap]
            ms = np.mean([r["satisfaction"] for r in g])
            mt = np.mean([r["tokens"] for r in g])
            vne = np.mean([1.0 if (r["valid"] == 1 and r["n_meetings"] > 0) else 0.0
                           for r in g])
            opt = np.mean([1.0 if r["optimality"] else 0.0 for r in g])
            mc = np.mean([r["calls"] for r in g])
            ml = np.median([r["latency"] for r in g])
            rows.append([SHORT[c], CAPK[cap], len(g), f"{ms:.3f}", f"{mt:.0f}",
                         f"{mc:.2f}", f"{ml:.1f}",
                         f"{vne:.3f}", f"{opt:.3f}", f"{ms / mt * 1000:.4f}"])
    add(md(["arch", "cap", "n_runs", "mean_sat", "mean_tokens", "mean_calls",
            "median_latency_s", "valid_nonempty", "optimality", "sat_per_1k"], rows))
    add("")

    # ---- 17. H data ----
    add("## 17. Higher-order gap H — coverage (Block A instances)")
    add("")
    add("`higher_order_gap_H = alpha_reachable - oracle_optimum` per instance, from the "
        "frozen structural pool `candidates.csv`. Counts of the 150 Block A instances by "
        "band and H stratum, including values outside the pre-declared `H = 0` / `H >= 1` "
        "strata.")
    add("")
    band_of = {}
    for r in A:
        band_of[r["condition"] + "|" + str(r["cap"])] = None  # placeholder
    # map instance -> band from runs (block A)
    inst_band = {}
    for r in csv.DictReader(RUNS.open(encoding="utf-8")):
        if r["block"] == "A":
            inst_band[r["instance_id"]] = r["band"]
    cov = defaultdict(Counter)
    anomalies = []
    for iid, band in inst_band.items():
        h = H.get(iid)
        if h is None:
            cov[band]["missing"] += 1
        elif h == 0:
            cov[band]["H=0"] += 1
        elif h >= 1:
            cov[band]["H>=1"] += 1
        else:
            cov[band]["H<0"] += 1
            anomalies.append((iid, band, h))
    rows = []
    for band in BANDS:
        c = cov[band]
        rows.append([band, c.get("H=0", 0), c.get("H>=1", 0), c.get("H<0", 0),
                     c.get("H=0", 0) + c.get("H>=1", 0) + c.get("H<0", 0)])
    add(md(["band", "H=0", "H>=1", "H<0", "total"], rows))
    add("")
    hdist = Counter(H[i] for i in inst_band if i in H)
    add("Full H distribution over the 150 Block A instances: "
        + ", ".join(f"H={k}: {hdist[k]}" for k in sorted(hdist)) + ".")
    if anomalies:
        add("")
        add("Instances with H < 0 (outside both strata): "
            + "; ".join(f"`{i}` (band {b}, H={h})" for i, b, h in anomalies) + ".")
    add("")
    hs = summary.get("h_sensitivity", {})
    if hs:
        add("From the frozen `summary.json` H sensitivity (paired counts per stratum, "
            "first registered contrast shown as the representative n): "
            + "; ".join(
                f"{name}: n_low={entries[0]['low']['n_pairs']}, "
                f"n_high={entries[0]['high']['n_pairs']}"
                for name, entries in hs.get("strata", {}).items()) + ".")
        add("")

    # ---- 18. figures ----
    add("## 18. Figures")
    add("")
    add("Neutral figures under `figures/` (vector PDF canonical, PNG preview, no "
        "embedded title). Captions describing what is plotted are in "
        "`figures/README.md`. Figures I and J are specific to cap 64000 and to the "
        "C1-C3 pair; they are not part of the descriptive set and live with the "
        "frozen confirmatory analysis, in "
        "`../heldout/figures/` (amendment 2026-09-02).")
    add("")
    add(md(["figure", "file", "what is plotted"], [
        ["F3", "F3_satisfaction_vs_budget_bands", "satisfaction vs budget cap; 3 panels (low/medium/high); lines C1-C5; bootstrap error bars"],
        ["B", "B_satisfaction_by_complexity", "mean satisfaction vs band; 4 panels (16k/32k/64k/128k); lines C1-C5"],
        ["C", "C_tokens_by_arch_budget", "mean tokens per run vs budget cap; lines C1-C5"],
        ["D", "D_calls_by_arch_budget", "mean model calls vs budget cap; lines C1-C5"],
        ["E", "E_latency_by_arch_budget", "median latency_seconds vs budget cap (p25-p75 whiskers); lines C1-C5"],
        ["F4", "F4_efficiency_satisfaction_vs_tokens", "satisfaction vs tokens actually spent; 4 panels per cap; bootstrap bars"],
        ["H", "H_optimality_by_arch_budget", "optimality rate vs budget cap; lines C1-C5"],
        ["L", "L_efficiency_by_budget", "satisfaction per 1000 tokens vs budget cap; 3 panels (low/medium/high); lines C1-C5"],
        ["F5", "F5_validity_and_termination_diagnostics", "valid-non-empty rate and non-voluntary-stop share vs budget cap"],
        ["F6", "F6_block_b_satisfaction_vs_budget", "Block B satisfaction vs budget cap, one line per architecture, by size"],
        ["M", "M_block_b_metrics_by_size", "Block B by size: six metrics besides satisfaction, 6 rows x n = 4/5/6 columns"],
        ["K", "K_pairwise_matrix_heatmap", "supplementary: all ten pairs x four budgets, one panel per band; cell = mean paired delta"],
    ]))
    add("")
    add("Appendix to the frozen confirmatory analysis, in `../heldout/figures/`:")
    add("")
    add(md(["figure", "file", "what is plotted"], [
        ["I", "I_frozen_forest_64k", "five registered contrasts at 64k (delta_low, delta_high, frozen bootstrap CI)"],
        ["J", "J_c1_minus_c3_delta", "Delta = S_C1 - S_C3 vs band; one line per cap; zero line; no verbal annotations"],
        ["F1", "F1_satisfaction_by_condition_band_64k", "satisfaction by condition and band at cap 64000"],
        ["F2", "F2_registered_contrasts_forest_64k", "the five registered contrasts at cap 64000"],
        ["F7", "F7_c1_vs_c3_crossover_delta", "C1 vs C3 delta across bands"],
    ]))
    add("")

    # ---- 20. coverage checklist ----
    add("## 20. Data coverage checklist")
    add("")
    for label in ["satisfaction", "all five architectures", "all four budgets",
                  "Low/Medium/High", "tokens", "calls", "latency",
                  "satisfaction/1k tokens", "valid", "valid-non-empty", "optimality",
                  "termination", "pairwise contrasts", "frozen formal p-values",
                  "Block B", "H strata", "figures"]:
        add(f"- [x] {label}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "legend.frameon": True,
        "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc", "legend.fontsize": 8,
    })


def save(fig, stem):
    out_dir = APPENDIX_FIG_DIR if stem.startswith(APPENDIX_STEMS) else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata=meta)
    plt.close(fig)


def _capaxis(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(list(CAPS))
    ax.set_xticklabels([CAPK[c] for c in CAPS]); ax.set_xlim(14000, 150000)
    ax.set_xlabel("token budget cap")


def line_vs_cap(mean_of, ylabel, stem, ylim=None, legend_ax0=True):
    """One line per architecture over the four caps (mean over Block A)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.array(CAPS, dtype=float)
    for c in CONDS:
        ax.plot(x, [mean_of(c, cap) for cap in CAPS], color=COLOR[c], marker=MARKER[c],
                ms=5, linewidth=1.7, label=LEGLAB[c])
    _capaxis(ax); ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    save(fig, stem)


def make_figures(runs, contrasts, summary):
    A = [r for r in runs if r["block"] == "A"]

    def poolA(c, cap):
        return [r for r in A if r["condition"] == c and r["cap"] == cap]

    def selA(c, cap, band):
        return [r for r in A if r["condition"] == c and r["cap"] == cap and r["band"] == band]

    def mean_sat(c, cap, band):
        return float(np.mean([r["satisfaction"] for r in selA(c, cap, band)]))

    # A was here: it duplicated F3, which plots the same surface with the
    # descriptive bootstrap intervals as well, so F3 is the canonical figure.

    # B: satisfaction vs complexity, 4 cap panels
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.2), sharey=True)
    bx = np.arange(len(BANDS))
    for ax, cap in zip(axes, CAPS):
        for c in CONDS:
            ax.plot(bx, [mean_sat(c, cap, b) for b in BANDS], color=COLOR[c],
                    marker=MARKER[c], ms=4.5, linewidth=1.6, label=LEGLAB[c])
        ax.set_xticks(bx); ax.set_xticklabels(BANDS); ax.set_ylim(0, 1.0)
        ax.set_title(f"cap {CAPK[cap]}"); ax.set_xlabel("complexity band")
    axes[0].set_ylabel("mean satisfaction"); axes[0].legend(loc="upper left", fontsize=7)
    save(fig, "B_satisfaction_by_complexity")

    # C tokens, D calls, G vne, H optimality
    line_vs_cap(lambda c, cap: float(np.mean([r["tokens"] for r in poolA(c, cap)])),
                "mean tokens per run", "C_tokens_by_arch_budget")
    line_vs_cap(lambda c, cap: float(np.mean([r["calls"] for r in poolA(c, cap)])),
                "mean model calls", "D_calls_by_arch_budget")
    # G was here: it duplicated the left panel of F5, which also carries the
    # non-voluntary-stop share, so F5 is the canonical figure.
    line_vs_cap(lambda c, cap: float(np.mean(
        [1.0 if r["optimality"] else 0.0 for r in poolA(c, cap)])),
        "optimality rate", "H_optimality_by_arch_budget", ylim=(0, 1.0))

    # E latency: median with p25-p75 whiskers
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.array(CAPS, float)
    for c in CONDS:
        med, lo, hi = [], [], []
        for cap in CAPS:
            v = np.array([r["latency"] for r in poolA(c, cap)])
            m = float(np.median(v)); med.append(m)
            lo.append(m - float(np.percentile(v, 25))); hi.append(float(np.percentile(v, 75)) - m)
        ax.errorbar(x, med, yerr=[lo, hi], color=COLOR[c], marker=MARKER[c], ms=5,
                    linewidth=1.7, elinewidth=0.8, capsize=2, label=LEGLAB[c])
    _capaxis(ax); ax.set_ylabel("latency_seconds (median, p25–p75)"); ax.legend(loc="best")
    save(fig, "E_latency_by_arch_budget")

    # F was here: it duplicated F4, which plots the same points with the
    # descriptive bootstrap intervals as well, so F4 is the canonical figure.

    # I frozen forest at 64k
    conf = summary["confirmatory"]; order = [r["contrast"] for r in conf["contrasts"]]
    dd = {}
    for r in contrasts:
        if r["block"] == "A" and int(r["cap"]) == 64000 and r["in_confirmatory_family"] == "True":
            dd[(r["contrast"], r["stratum"].split("=")[1])] = r
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    c_low, c_high = "#0072B2", "#D55E00"
    y_of = {c: len(order) - 1 - i for i, c in enumerate(order)}
    for c in order:
        for band, dy, col, mk in (("low", 0.17, c_low, "o"), ("high", -0.17, c_high, "s")):
            r = dd[(c, band)]; m = float(r["mean_delta"])
            ax.errorbar(m, y_of[c] + dy,
                        xerr=[[m - float(r["ci_low"])], [float(r["ci_high"]) - m]],
                        fmt=mk, color=col, ecolor=col, ms=5.5, elinewidth=1.4, capsize=3)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_yticks([y_of[c] for c in order]); ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"delta = satisfaction(later) - satisfaction(earlier), cap 64k")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=c_low, linestyle="", ms=6,
                              label="delta_low"),
                       Line2D([0], [0], marker="s", color=c_high, linestyle="", ms=6,
                              label="delta_high")], loc="center left",
              bbox_to_anchor=(0.015, 0.6))
    save(fig, "I_frozen_forest_64k")

    # J C1 - C3 delta vs band, one line per cap, no verbal annotations
    c13 = {(int(r["cap"]), r["stratum"].split("=")[1]): r for r in contrasts
           if r["block"] == "A" and r["contrast"] == "C1->C3"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bx = np.arange(len(BANDS))
    cap_color = {16000: "#BDD7E7", 32000: "#6BAED6", 64000: "#3182BD", 128000: "#08519C"}
    cap_marker = {16000: "o", 32000: "s", 64000: "^", 128000: "D"}
    dodge = {16000: -0.06, 32000: -0.02, 64000: 0.02, 128000: 0.06}
    for cap in CAPS:
        means, lo, hi = [], [], []
        for b in BANDS:
            r = c13[(cap, b)]; m = -float(r["mean_delta"])
            means.append(m); lo.append(m - (-float(r["ci_high"]))); hi.append((-float(r["ci_low"])) - m)
        ax.errorbar(bx + dodge[cap], means, yerr=[lo, hi], color=cap_color[cap],
                    marker=cap_marker[cap], ms=6, linewidth=1.8, elinewidth=1.0, capsize=2.5,
                    label=f"cap {CAPK[cap]}")
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(bx); ax.set_xticklabels(BANDS)
    ax.set_xlabel("complexity band (conflict-graph density)")
    ax.set_ylabel(r"$\Delta = S_{C1} - S_{C3}$"); ax.set_xlim(-0.4, len(BANDS) - 0.6)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              title="equal token budget", frameon=False)
    save(fig, "J_c1_minus_c3_delta")

    # ---------------------------------------------------------------- K ----
    # Supplementary. All ten pairs at once is dense by construction; it exists so the
    # complete matrix is visible in one place, not because it is easy to read. The
    # readable views are A, B and L, which show five architectures over four budgets.
    pair_rows = {}
    for r in contrasts:
        if r["block"] == "A" and r["n_pairs"]:
            pair_rows[(r["contrast"], int(r["cap"]),
                       r["stratum"].split("=")[1])] = float(r["mean_delta"])
    pairs = sorted({k[0] for k in pair_rows})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
    grids = [np.array([[pair_rows[(p, cap, band)] for cap in CAPS] for p in pairs])
             for band in BANDS]
    vmax = float(np.max([np.abs(g).max() for g in grids]))
    for ax, band, grid in zip(axes, BANDS, grids):
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CAPS))); ax.set_xticklabels([CAPK[c] for c in CAPS])
        ax.set_title(f"{band} complexity"); ax.set_xlabel("token budget cap")
        ax.grid(False)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 * vmax else "#222222")
    axes[0].set_yticks(range(len(pairs))); axes[0].set_yticklabels(pairs, fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="mean paired delta = satisfaction(later) - satisfaction(earlier)")
    save(fig, "K_pairwise_matrix_heatmap")

    # ---------------------------------------------------------------- L ----
    # Efficiency, same ratio-of-means definition as section 12, resolved by band.
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    for ax, band in zip(axes, BANDS):
        for c in CONDS:
            ys = []
            for cap in CAPS:
                g = selA(c, cap, band)
                ms = float(np.mean([r["satisfaction"] for r in g]))
                mt = float(np.mean([r["tokens"] for r in g]))
                ys.append(ms / mt * 1000.0 if mt else np.nan)
            ax.plot(np.array(CAPS, float), ys, color=COLOR[c], marker=MARKER[c],
                    ms=4.5, linewidth=1.6, label=LEGLAB[c])
        _capaxis(ax); ax.set_title(f"{band} complexity")
    axes[0].set_ylabel("satisfaction per 1000 tokens")
    # Figure-level legend, below the panels and outside every plotting area. In the low
    # panel C3 and C4 run through the upper half, so an in-axes legend hid the two
    # highest curves; moving it between corners only moves the problem to another line.
    fig.legend(handles=[Line2D([0], [0], color=COLOR[c], marker=MARKER[c],
                               linestyle="-", ms=5, label=LEGLAB[c]) for c in CONDS],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.10),
               frameon=False, fontsize=8)
    save(fig, "L_efficiency_by_budget")

    # ---------------------------------------------------------------- M ----
    # Block B by SIZE, not pooled. The block exists to show what changes when the task
    # gets smaller, so n is a panel column and never a thing to average over. Rows are
    # the six metrics besides satisfaction (satisfaction by size is F6), so Block B is
    # reported on the same seven metrics as Block A without collapsing the size axis.
    B = [r for r in runs if r["block"] == "B"]

    def selBn(c, cap, n):
        return [r for r in B if r["condition"] == c and r["cap"] == cap
                and r["n_people"] == n]

    panels = [
        ("valid-non-empty",
         lambda g: float(np.mean([1.0 if (r["valid"] == 1 and r["n_meetings"] > 0)
                                  else 0.0 for r in g])), (0, 1.05)),
        ("optimality rate",
         lambda g: float(np.mean([1.0 if r["optimality"] else 0.0 for r in g])), (0, 1.05)),
        ("mean tokens/run",
         lambda g: float(np.mean([r["tokens"] for r in g])), None),
        ("mean model calls",
         lambda g: float(np.mean([r["calls"] for r in g])), None),
        ("median latency (s)",
         lambda g: float(np.median([r["latency"] for r in g])), None),
        ("satisfaction / 1k tokens",
         lambda g: (float(np.mean([r["satisfaction"] for r in g]))
                    / float(np.mean([r["tokens"] for r in g])) * 1000.0), None),
    ]
    fig, axes = plt.subplots(len(panels), len(SMALL_N), figsize=(9.6, 13.2),
                             sharex=True)
    for i, (ylabel, fn, ylim) in enumerate(panels):
        row = axes[i]
        # One y scale across the row so the three sizes are comparable by eye; that is
        # the comparison the figure exists for.
        vals = [[fn(selBn(c, cap, n)) for cap in CAPS] for n in SMALL_N for c in CONDS]
        lo = min(min(v) for v in vals); hi = max(max(v) for v in vals)
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        for j, n in enumerate(SMALL_N):
            ax = row[j]
            for c in CONDS:
                ax.plot(np.array(CAPS, float), [fn(selBn(c, cap, n)) for cap in CAPS],
                        color=COLOR[c], marker=MARKER[c], ms=4.0, linewidth=1.4,
                        label=LEGLAB[c])
            ax.set_ylim(*(ylim if ylim else (lo - pad, hi + pad)))
            if i == 0:
                ax.set_title(f"n = {n}")
            if i == len(panels) - 1:
                _capaxis(ax)
            else:
                ax.set_xscale("log", base=2); ax.set_xticks(list(CAPS))
                ax.set_xticklabels([]); ax.set_xlim(14000, 150000)
            if j == 0:
                ax.set_ylabel(ylabel)
    axes[0][0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "M_block_b_metrics_by_size")


def validate(runs, agg) -> None:
    """Assert the computed per-cell aggregates reproduce the committed aggregates.csv."""
    A = [r for r in runs if r["block"] == "A"]
    for c in CONDS:
        for cap in CAPS:
            for band in BANDS:
                g = [r for r in A if r["condition"] == c and r["cap"] == cap and r["band"] == band]
                a = agg[(c, cap, f"band={band}")]
                # aggregates.csv stores rounded views (rates/satisfaction to 4 dp, tokens
                # to integer); compare to that stored precision.
                assert len(g) == int(a["n_runs"]) == 50, (c, cap, band, len(g))
                assert abs(np.mean([r["satisfaction"] for r in g]) - float(a["mean_satisfaction"])) < 1e-3
                assert abs(np.mean([r["valid"] for r in g]) - float(a["valid_rate"])) < 1e-3
                assert abs(np.mean([1.0 if (r["valid"] == 1 and r["n_meetings"] > 0) else 0.0 for r in g])
                           - float(a["nonempty_validated_rate"])) < 1e-3
                assert abs(np.mean([r["tokens"] for r in g]) - float(a["mean_tokens"])) < 1.0
    print("validation: computed per-cell aggregates match aggregates.csv")


def main() -> int:
    runs = load_runs()
    agg = load_aggregates()
    contrasts = load_contrasts()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    H = load_H()
    validate(runs, agg)
    OUT_MD.write_text(build_document(runs, agg, contrasts, summary, audit, H),
                      encoding="utf-8")
    set_style()
    make_figures(runs, contrasts, summary)
    print(f"wrote {OUT_MD.relative_to(_REPO)}")
    print(f"  figures B, C, D, E, H, K, L, M -> {FIG_DIR.relative_to(_REPO)}")
    print(f"  figures I, J (frozen appendix) -> {APPENDIX_FIG_DIR.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
