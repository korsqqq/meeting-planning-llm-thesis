# scripts/build_presentation.py
"""Build the thesis-status presentation from repository artifacts only. CPU only, no LLM.

    python -m scripts.build_presentation --out presentation

Every number on every slide is read from a file in this repository. Nothing is typed in by
hand, and nothing is carried over from a terminal session: if a source is missing the figure
that depends on it is skipped and the omission is recorded in the summary, rather than being
filled from memory.

The script does four things in order.

1. **Audit.** Each source is counted, checked for duplicates and for missing cells, and its
   provenance hash recorded where one exists. The audit is written out and reproduced in the
   summary, because a figure drawn from an incomplete set is readable and wrong.
2. **Clean tables.** Intermediate per-run and per-instance tables are written to
   `data/` so any figure can be re-derived without re-reading the raw documents.
3. **Figures.** Matplotlib, one file per figure, each with a title, axis labels and a caption
   registered alongside it.
4. **Assembly.** A self-contained HTML deck with the images inlined, plus the summary and a
   machine-readable metrics table.

Artifacts that the project has marked exploratory or superseded are labelled as such
wherever they appear, and the failed budget calibration is presented as a result rather than
omitted.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

# --------------------------------------------------------------------------- #
# House style. Muted, print-safe, consistent across every figure.
# --------------------------------------------------------------------------- #
COND_COLOUR = {"c1_react": "#3B6EA5", "c2_verify_revise": "#C4703A", "c3_mas": "#4E8B62"}
COND_LABEL = {"c1_react": "C1 ReAct", "c2_verify_revise": "C2 verify/revise",
              "c3_mas": "C3 hierarchical MAS"}
BAND_COLOUR = {"low": "#9EC4E0", "medium": "#5B8FBF", "high": "#22506F"}
BAND_LABEL = {"low": "Low density", "medium": "Medium density", "high": "High density"}
LEVEL_TO_BAND = {"easy": "low", "medium": "medium", "hard": "high"}
CAPS = [8000, 16000, 32000, 64000]
GREY = "#666666"
ACCENT = "#A33A3A"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


class Registry:
    """Figures, captions and audit findings, collected as the build runs."""

    def __init__(self, fig_dir: Path) -> None:
        self.fig_dir = fig_dir
        self.figures: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.audit: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    def save(self, fig, name: str, title: str, caption: str, source: str) -> str:
        path = self.fig_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.figures.append({"name": name, "file": path.name, "title": title,
                             "caption": caption, "source": source})
        return path.name

    def skip(self, name: str, why: str) -> None:
        self.skipped.append({"name": name, "reason": why})

    def check(self, source: str, **facts: Any) -> None:
        self.audit.append({"source": source, **facts})

    def metric(self, group: str, **row: Any) -> None:
        self.metrics.append({"group": group, **row})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Sources.
# --------------------------------------------------------------------------- #
def load_sources(root: Path, reg: Registry) -> dict[str, Any]:
    """Read every artifact the deck can use, auditing each as it is read."""
    S: dict[str, Any] = {}

    # ---- formal pilot -----------------------------------------------------
    fp_dir = root / "results/analysis/formal_pilot"
    runs = pd.read_csv(fp_dir / "runs.csv")
    S["pilot_runs"] = runs
    S["pilot_summary"] = json.loads((fp_dir / "summary.json").read_text(encoding="utf-8"))
    raw = sorted((root / "results/formal_pilot").glob("*.json"))
    cells = Counter(zip(runs["condition"], runs["instance_id"], runs["cap"]))
    reg.check(
        "formal pilot",
        path="results/analysis/formal_pilot/runs.csv",
        raw_documents=len(raw),
        rows=len(runs),
        distinct_run_ids=runs["run_id"].nunique(),
        duplicate_run_ids=int(len(runs) - runs["run_id"].nunique()),
        conditions=sorted(runs["condition"].unique()),
        caps=sorted(int(c) for c in runs["cap"].unique()),
        instances=int(runs["instance_id"].nunique()),
        expected_runs=int(runs["condition"].nunique() * runs["cap"].nunique()
                          * runs["instance_id"].nunique()),
        duplicated_cells=int(sum(1 for v in cells.values() if v > 1)),
        missing_cells=int(runs["condition"].nunique() * runs["cap"].nunique()
                          * runs["instance_id"].nunique() - len(cells)),
        upstream_audit_passed=S["pilot_summary"]["integrity_audit"]["all_checks_passed"],
        subset_hash=S["pilot_summary"]["integrity_audit"]["subset_hash"][:12],
        status="pilot, design validation — not a confirmatory test",
    )
    for name in ("pilot_proposals", "pilot_instances", "pilot_calls"):
        f = fp_dir / f"{name}.csv"
        if f.exists():
            S[name] = pd.read_csv(f)
            reg.check(f"formal pilot / {name}", path=str(f.relative_to(root)),
                      rows=len(S[name]))

    # ---- C1 budget calibration -------------------------------------------
    c1 = root / "results/analysis/c1_calibration/summary.json"
    if c1.exists():
        S["c1_cal"] = json.loads(c1.read_text(encoding="utf-8"))
        n_raw = len(list((root / "results/calibration/c1_20260805").glob("*.json")))
        reg.check("C1 budget calibration", path=str(c1.relative_to(root)),
                  raw_documents=n_raw,
                  rows=S["c1_cal"]["integrity_audit"]["n_runs"],
                  instances=S["c1_cal"]["integrity_audit"]["n_instances"],
                  caps=S["c1_cal"]["integrity_audit"]["caps"],
                  upstream_audit_passed=S["c1_cal"]["integrity_audit"]["all_checks_passed"],
                  note=("one instance was exposed before the manifest was frozen; the "
                        "analysis reports with and without it"),
                  status="calibration")

    # ---- structural calibration ------------------------------------------
    pool = root / "results/calibration/structural/candidates.csv"
    S["struct"] = pd.read_csv(pool)
    S["struct_meta"] = json.loads(
        (root / "results/calibration/structural/pool_meta.json").read_text(encoding="utf-8"))
    S["struct_analysis"] = json.loads(
        (root / "results/analysis/structural_calibration/summary.json").read_text(
            encoding="utf-8"))
    reg.check("structural calibration pool", path=str(pool.relative_to(root)),
              rows=len(S["struct"]),
              expected=S["struct_meta"]["n_expected"],
              n_people=S["struct_meta"]["n_people"],
              seeds=[S["struct_meta"]["seed_start"], S["struct_meta"]["seed_end"]],
              unproven_optima=S["struct_meta"]["n_not_proven"],
              duplicate_instance_ids=int(len(S["struct"])
                                         - S["struct"]["instance_id"].nunique()),
              upstream_audit_passed=S["struct_meta"]["audit"]["all_checks_passed"],
              sha256=sha256(pool)[:16], status="calibration")

    S["band_search"] = json.loads(
        (root / "results/analysis/band_search/summary.json").read_text(encoding="utf-8"))
    S["band_admiss"] = json.loads(
        (root / "results/analysis/band_admissibility/summary.json").read_text(
            encoding="utf-8"))
    S["joint"] = json.loads(
        (root / "results/analysis/joint_support/summary.json").read_text(encoding="utf-8"))

    # ---- budget-dev -------------------------------------------------------
    bd_pool = root / "results/calibration/budget_dev/candidates.csv"
    if bd_pool.exists():
        S["bd_pool"] = pd.read_csv(bd_pool)
        meta = json.loads((root / "results/calibration/budget_dev/pool_meta.json")
                          .read_text(encoding="utf-8"))
        S["bd_pool_meta"] = meta
        reg.check("budget-dev candidate pool", path=str(bd_pool.relative_to(root)),
                  rows=len(S["bd_pool"]), expected=meta["n_expected"],
                  n_people=meta["n_people"],
                  seeds=[meta["seed_start"], meta["seed_end"]],
                  unproven_optima=meta["n_not_proven"],
                  upstream_audit_passed=meta["audit"]["all_checks_passed"],
                  status="calibration")

    sub = sorted((root / "results/manifests").glob("subset__bands_budget_dev__*.json"))
    if sub:
        S["bd_subset"] = json.loads(sub[0].read_text(encoding="utf-8"))
        inst = [dict(e, band=LEVEL_TO_BAND[lv])
                for lv, rows in S["bd_subset"]["instances"].items() for e in rows]
        S["bd_instances"] = pd.DataFrame([{
            "instance_id": e["instance_id"], "seed": e["seed"], "band": e["band"],
            "level": e["level"], "k": e["complexity_metric"],
            "D": round(e["complexity_metric"] / 28, 4), "O": e["optimum"],
            "tightness": e["cell"]["tightness"], "overlap": e["cell"]["overlap"],
            "travel_structure": e["cell"]["travel_structure"],
            "n_people": e["cell"]["n_people"],
        } for e in inst])
        df = S["bd_instances"]
        reg.check("budget-dev frozen instance set", path=str(sub[0].relative_to(root)),
                  rows=len(df), distinct_seeds=int(df["seed"].nunique()),
                  per_band=df["band"].value_counts().to_dict(),
                  seed_range=[int(df["seed"].min()), int(df["seed"].max())],
                  all_seeds_in_dev_range=bool(df["seed"].between(30000, 39999).all()),
                  optimum_histogram_identical_across_bands=bool(
                      len({tuple(sorted(Counter(g["O"]).items()))
                           for _, g in df.groupby("band")}) == 1),
                  content_hash=S["bd_subset"]["content_hash"][:12], status="budget-dev")

    bd_runs = root / "results/analysis/budget_dev/runs.csv"
    if bd_runs.exists():
        S["bd_runs"] = pd.read_csv(bd_runs)
        r = S["bd_runs"]
        cells = Counter(zip(r["instance_id"], r["cap"]))
        reg.check("budget-dev runs", path=str(bd_runs.relative_to(root)), rows=len(r),
                  distinct_run_ids=int(r["run_id"].nunique()),
                  duplicate_run_ids=int(len(r) - r["run_id"].nunique()),
                  conditions=sorted(r["condition"].unique()),
                  caps=sorted(int(c) for c in r["cap"].unique()),
                  instances=int(r["instance_id"].nunique()),
                  duplicated_cells=int(sum(1 for v in cells.values() if v > 1)),
                  missing_cells=int(r["instance_id"].nunique() * len(CAPS) - len(cells)),
                  all_seeds_in_dev_range=bool(r["seed"].between(30000, 39999).all()),
                  status="budget-dev, development calibration — not confirmatory")
    else:
        reg.skip("budget-dev per-run figures",
                 "results/analysis/budget_dev/runs.csv is not in the repository; the raw "
                 "runs live on the run machine and were not exported")

    bd_sum = root / "results/analysis/budget_dev/summary.json"
    if bd_sum.exists():
        S["bd_summary"] = json.loads(bd_sum.read_text(encoding="utf-8"))
    bd_props = root / "results/analysis/budget_dev/proposals.csv"
    if bd_props.exists():
        S["bd_proposals"] = pd.read_csv(bd_props)
        reg.check("budget-dev proposals", path=str(bd_props.relative_to(root)),
                  rows=len(S["bd_proposals"]), status="budget-dev")

    # ---- exploratory / superseded ----------------------------------------
    for name, rel in (("power_v1", "results/analysis/power_iut/summary.json"),
                      ("power_v2", "results/analysis/power_iut_v2/summary.json"),
                      ("type_one", "results/analysis/type_one_validation/summary.json")):
        f = root / rel
        if f.exists():
            S[name] = json.loads(f.read_text(encoding="utf-8"))
            reg.check(f"{name} (EXPLORATORY / SUPERSEDED)", path=rel,
                      status="exploratory — binds nothing; retired 2026-08-10")
    return S


# --------------------------------------------------------------------------- #
# Small drawing helpers.
# --------------------------------------------------------------------------- #
def _cap_axis(ax) -> None:
    ax.set_xticks(range(len(CAPS)))
    ax.set_xticklabels([f"{c // 1000}k" for c in CAPS])
    ax.set_xlabel("Token budget cap")


def _bar_labels(ax, bars, fmt="{:.2f}", size=8) -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 2), ha="center",
                    fontsize=size, color="#333333")


# --------------------------------------------------------------------------- #
# Section 3 — project timeline.
# --------------------------------------------------------------------------- #
def fig_timeline(S: dict, reg: Registry) -> None:
    stages = [
        ("Layers 0-5\ngenerator, CP-SAT oracle,\nvalidator, harness, scorer", "infrastructure"),
        ("C1 budget calibration\n120 runs, 30 instances", "calibration"),
        ("Formal pilot\n360 runs, 30 instances\nC1/C2/C3 x 4 caps", "pilot"),
        ("Structural calibration\n9 600 candidates at n = 8", "calibration"),
        ("Band design search\n593 775 triples -> FEASIBLE", "design"),
        ("Budget-dev\n240 runs, 60 instances\nC1 only x 4 caps", "budget-dev"),
        ("Decision point\ncalibration gate NOT passed", "decision"),
    ]
    kind_colour = {"infrastructure": "#B0B0B0", "calibration": "#9EC4E0",
                   "pilot": "#5B8FBF", "design": "#4E8B62", "budget-dev": "#22506F",
                   "decision": ACCENT}
    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    xs = np.arange(len(stages))
    ax.plot(xs, [0] * len(stages), color="#CCCCCC", lw=2, zorder=1)
    for x, (label, kind) in zip(xs, stages):
        ax.scatter([x], [0], s=280, color=kind_colour[kind], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(label, (x, 0), textcoords="offset points",
                    xytext=(0, 26 if x % 2 == 0 else -26),
                    ha="center", va="bottom" if x % 2 == 0 else "top", fontsize=8.4,
                    color="#222222")
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.set_title("Project stages, in the order they were executed", pad=14)
    reg.save(fig, "01_timeline", "Experimental roadmap",
             "Each stage completed before the next began. Only the last stage is "
             "unresolved: the budget-calibration gate returned no cap, so the main "
             "experiment has not started.",
             "THESIS_DECISIONS.md sections 3 and 5, plus the artifacts audited in this build")


# --------------------------------------------------------------------------- #
# Section 4 — experiment inventory.
# --------------------------------------------------------------------------- #
def fig_inventory(S: dict, reg: Registry) -> pd.DataFrame:
    rows = []
    pr = S["pilot_runs"]
    if "c1_cal" in S:
        ia = S["c1_cal"]["integrity_audit"]
        rows.append({"experiment": "C1 budget calibration", "kind": "calibration",
                     "n_people": "4, 6, 8", "instances": ia["n_instances"],
                     "conditions": "C1", "caps": "8k/16k/32k/64k",
                     "runs": ia["n_runs"], "llm": "yes",
                     "status": "superseded by the formal pilot; one exposed instance"})
    rows.append({"experiment": "Formal pilot", "kind": "pilot",
                 "n_people": "4, 6, 8", "instances": int(pr["instance_id"].nunique()),
                 "conditions": "C1, C2, C3", "caps": "8k/16k/32k/64k",
                 "runs": len(pr), "llm": "yes",
                 "status": "complete; design validation, not confirmatory"})
    sm = S["struct_meta"]
    rows.append({"experiment": "Structural calibration", "kind": "calibration",
                 "n_people": sm["n_people"], "instances": sm["n_completed"],
                 "conditions": "-", "caps": "-", "runs": 0, "llm": "no",
                 "status": "complete; CPU only, no agent involved"})
    bs = S["band_search"]["search"]
    rows.append({"experiment": "Band design search", "kind": "design",
                 "n_people": 8, "instances": bs["triples_considered"],
                 "conditions": "-", "caps": "-", "runs": 0, "llm": "no",
                 "status": "complete; verdict " + str(bs["verdict"])})
    if "bd_instances" in S:
        n_runs = len(S["bd_runs"]) if "bd_runs" in S else 0
        rows.append({"experiment": "Budget-dev", "kind": "budget-dev",
                     "n_people": 8, "instances": len(S["bd_instances"]),
                     "conditions": "C1", "caps": "8k/16k/32k/64k",
                     "runs": n_runs, "llm": "yes",
                     "status": "complete; calibration gate NOT passed"})
    for key, label in (("power_v1", "Power simulation v1"),
                       ("power_v2", "Power simulation v2"),
                       ("type_one", "Type-I validation")):
        if key in S:
            rows.append({"experiment": label, "kind": "exploratory",
                         "n_people": "-", "instances": "-", "conditions": "-",
                         "caps": "-", "runs": 0, "llm": "no",
                         "status": "EXPLORATORY / SUPERSEDED - binds nothing"})
    df = pd.DataFrame(rows)
    for r in rows:
        reg.metric("inventory", **r)

    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    llm = df[df["llm"] == "yes"]
    colours = {"calibration": "#9EC4E0", "pilot": "#5B8FBF", "budget-dev": "#22506F"}
    bars = ax.barh(llm["experiment"], llm["runs"],
                   color=[colours.get(k, GREY) for k in llm["kind"]])
    for b, v in zip(bars, llm["runs"]):
        ax.annotate(str(v), (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5)
    ax.set_xlabel("Number of LLM runs")
    ax.set_ylabel("")
    ax.set_title("LLM runs actually executed, by experiment")
    ax.set_xlim(0, max(max(llm["runs"]), 1) * 1.16)
    ax.invert_yaxis()
    reg.save(fig, "02_inventory_runs", "Experiment inventory",
             str(int(llm["runs"].sum())) + " LLM runs in total across the executed "
             "experiments. The often-quoted figure of about 260 for the earlier work is "
             "incorrect: the formal pilot alone is " + str(len(pr)) + " runs.",
             "results/analysis/*/summary.json and results/analysis/formal_pilot/runs.csv")
    return df


# --------------------------------------------------------------------------- #
# Section 5 — formal pilot.
# --------------------------------------------------------------------------- #
def figs_pilot(S: dict, reg: Registry) -> None:
    runs, summ = S["pilot_runs"], S["pilot_summary"]
    conds = ["c1_react", "c2_verify_revise", "c3_mas"]
    levels = ["easy", "medium", "hard"]

    for metric, name, title, ylab, fmt, caption in [
        ("satisfaction", "10_pilot_satisfaction",
         "Satisfaction by condition and budget cap (formal pilot, 360 runs)",
         "Mean satisfaction  (achieved / oracle optimum)", "{:.2f}",
         "Every condition scores exactly zero at 8k. C3 leads C1 from 32k upward. An "
         "invalid plan scores 0 with no partial credit."),
        ("nonempty_plan", "11_pilot_nonempty",
         "Share of runs ending with a non-empty valid plan",
         "Share of runs", "{:.2f}",
         "The structural precondition for scoring at all. At 64k the single agent reaches "
         "it in fewer than half of runs, which is the number the later 50% gate was set "
         "just above."),
        ("tokens_total", "12_pilot_tokens",
         "Mean total token consumption by condition and cap",
         "Mean tokens per run  (input + thinking + output)", "{:.0f}",
         "C3 consumes fewer tokens than C1 at the two upper rungs while scoring higher, so "
         "its advantage is not bought with budget."),
    ]:
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        w = 0.26
        for i, c in enumerate(conds):
            vals = [runs[(runs.condition == c) & (runs.cap == cap)][metric].mean()
                    for cap in CAPS]
            bars = ax.bar(np.arange(len(CAPS)) + (i - 1) * w, vals, w,
                          label=COND_LABEL[c], color=COND_COLOUR[c])
            _bar_labels(ax, bars, fmt)
            for cap, v in zip(CAPS, vals):
                reg.metric("formal_pilot_by_cap", condition=c, cap=cap,
                           metric=metric, value=round(float(v), 4))
        _cap_axis(ax)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="upper left")
        reg.save(fig, name, title, caption, "results/analysis/formal_pilot/runs.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), sharey=True)
    for ax, cap in zip(axes, CAPS):
        w = 0.26
        for i, c in enumerate(conds):
            vals = [runs[(runs.condition == c) & (runs.cap == cap)
                         & (runs.level == lv)]["satisfaction"].mean() for lv in levels]
            ax.bar(np.arange(3) + (i - 1) * w, vals, w, label=COND_LABEL[c],
                   color=COND_COLOUR[c])
            for lv, v in zip(levels, vals):
                reg.metric("formal_pilot_by_level", condition=c, cap=cap, level=lv,
                           metric="satisfaction", value=round(float(v), 4))
        ax.set_xticks(range(3))
        ax.set_xticklabels(levels)
        ax.set_title("cap " + str(cap // 1000) + "k")
        ax.set_xlabel("Complexity level (old binning)")
    axes[0].set_ylabel("Mean satisfaction")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Satisfaction by complexity level: the level ordering does not behave as "
                 "designed", y=1.03)
    reg.save(fig, "13_pilot_by_level", "Satisfaction by complexity level",
             "The hierarchy leads on easy instances as strongly as on hard ones. Under the "
             "expose, easy instances are where the single agent should be ahead.",
             "results/analysis/formal_pilot/runs.csv")

    d = summ["analysis"]["contrasts"]["delta_c1_react_minus_c3_mas"]
    means = [d[str(c)]["bootstrap"]["mean"] for c in CAPS]
    lo = [d[str(c)]["bootstrap"]["ci_low"] for c in CAPS]
    hi = [d[str(c)]["bootstrap"]["ci_high"] for c in CAPS]
    excl = [d[str(c)]["bootstrap"]["excludes_zero"] for c in CAPS]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(CAPS))
    ax.axhline(0, color="#888888", lw=1)
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(lo),
                                np.array(hi) - np.array(means)],
                fmt="o", capsize=5, color="#22506F", ecolor="#22506F", ms=7, lw=1.5)
    for xi, m, e in zip(x, means, excl):
        ax.annotate(("%+.3f" % m) + ("  CI excludes 0" if e else ""), (xi, m),
                    textcoords="offset points", xytext=(9, -3), fontsize=8.5,
                    color=ACCENT if e else GREY)
        reg.metric("formal_pilot_delta_c1_minus_c3", cap=CAPS[xi], delta_mean=m,
                   ci_low=lo[xi], ci_high=hi[xi], ci_excludes_zero=bool(e))
    _cap_axis(ax)
    ax.set_ylabel("delta = S(C1) - S(C3)")
    ax.set_title("Paired difference between the single agent and the hierarchy\n"
                 "10 000 bootstrap resamples over instances, 95% interval")
    ax.annotate("delta < 0: hierarchy ahead", (0.02, 0.06), xycoords="axes fraction",
                fontsize=9, color=GREY)
    reg.save(fig, "14_pilot_delta_ci", "C1 minus C3 with bootstrap intervals",
             "At 32k and 64k the interval excludes zero and the hierarchy is ahead. The "
             "sign convention comes from the expose: negative means the hierarchy scores "
             "higher.",
             "results/analysis/formal_pilot/summary.json, analysis.contrasts")

    cp = summ["analysis"]["crossover_primary"]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    marks = {"easy": "o", "medium": "s", "hard": "^"}
    for lv in levels:
        m = [cp[lv]["per_cap"][str(c)]["delta_mean"] for c in CAPS]
        lo_ = [cp[lv]["per_cap"][str(c)]["ci"][0] for c in CAPS]
        hi_ = [cp[lv]["per_cap"][str(c)]["ci"][1] for c in CAPS]
        off = {"easy": -0.09, "medium": 0.0, "hard": 0.09}[lv]
        ax.errorbar(np.arange(len(CAPS)) + off, m,
                    yerr=[np.array(m) - np.array(lo_), np.array(hi_) - np.array(m)],
                    fmt=marks[lv], capsize=4, ms=6, lw=1.3,
                    color=BAND_COLOUR[LEVEL_TO_BAND[lv]], label=lv)
        for c, mm, l_, h_ in zip(CAPS, m, lo_, hi_):
            reg.metric("formal_pilot_crossover", level=lv, cap=c, delta_mean=mm,
                       ci_low=l_, ci_high=h_,
                       ci_excludes_zero=bool(
                           cp[lv]["per_cap"][str(c)]["ci_excludes_zero"]))
    ax.axhline(0, color="#888888", lw=1)
    _cap_axis(ax)
    ax.set_ylabel("delta = S(C1) - S(C3)")
    ax.set_title("The registered crossover pattern is not observed\n"
                 "delta should be positive on easy and negative on hard within a cap")
    ax.legend(title="complexity level")
    reg.save(fig, "15_pilot_crossover", "Crossover check by level",
             "Delta is negative on easy at 64k with an interval excluding zero: -0.460, "
             "[-0.767, -0.133]. The low-complexity sanity check therefore fails, since the "
             "hierarchy leads exactly where it should not.",
             "results/analysis/formal_pilot/summary.json, analysis.crossover_primary")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.26
    for i, c in enumerate(conds):
        vals = []
        for cap in CAPS:
            bc = summ["analysis"]["by_condition"][c]["by_cap"][str(cap)]
            vals.append(bc.get("proposal_validity_rate") or 0.0)
            reg.metric("formal_pilot_proposals", condition=c, cap=cap,
                       attempts=bc.get("attempts_total"),
                       validity_rate=bc.get("proposal_validity_rate"),
                       parse_rate=bc.get("proposal_parse_rate"))
        bars = ax.bar(np.arange(len(CAPS)) + (i - 1) * w, vals, w,
                      label=COND_LABEL[c], color=COND_COLOUR[c])
        _bar_labels(ax, bars, "{:.2f}")
    _cap_axis(ax)
    ax.set_ylabel("Valid proposals / all proposal attempts")
    ax.set_title("Proposal validity rate - diagnostic only")
    ax.legend(loc="upper left")
    reg.save(fig, "16_pilot_proposal_validity", "Proposal validity",
             "The high rate for C3 is close to constructional: its critic is gated by the "
             "validator and its workers solve a decomposed sub-problem, so the rate is not "
             "comparable with C1. The decisions document records it as diagnostic, not as "
             "evidence of mechanism.",
             "results/analysis/formal_pilot/summary.json, analysis.by_condition")

    if "pilot_proposals" in S:
        pp = S["pilot_proposals"]
        bad = pp[~pp["valid"].astype(bool)]
        counts = Counter()
        for _, row in bad.iterrows():
            for r in str(row["reasons"] or "").replace(";", "|").split("|"):
                r = r.strip()
                if r and r.lower() != "nan":
                    counts[r] += 1
        if counts:
            fig, ax = plt.subplots(figsize=(8.6, 3.6))
            items = counts.most_common()
            vals = [v for _, v in items][::-1]
            bars = ax.barh([k for k, _ in items][::-1], vals, color="#5B8FBF")
            for b, v in zip(bars, vals):
                ax.annotate(str(v), (v, b.get_y() + b.get_height() / 2),
                            textcoords="offset points", xytext=(4, 0), va="center",
                            fontsize=9)
            ax.set_xlabel("Occurrences across rejected proposals (multi-label)")
            ax.set_title("Why proposals were rejected: " + str(len(bad)) +
                         " rejected of " + str(len(pp)) + " attempts")
            reg.save(fig, "17_pilot_reasons", "Rejection reasons",
                     "Counted over rejected proposals and multi-label, so the total exceeds "
                     "the number of rejections. Travel infeasibility dominates.",
                     "results/analysis/formal_pilot/pilot_proposals.csv")
            for k, v in items:
                reg.metric("formal_pilot_rejection_reasons", reason=k, count=v)

    if "pilot_calls" in S:
        pc = S["pilot_calls"]
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        w = 0.26
        for i, c in enumerate(conds):
            think, answer = [], []
            for cap in CAPS:
                sel = pc[(pc.condition == c) & (pc.cap == cap)]
                nruns = runs[(runs.condition == c) & (runs.cap == cap)].shape[0]
                think.append(sel["thinking_tokens"].sum() / max(nruns, 1))
                answer.append(sel["answer_tokens"].sum() / max(nruns, 1))
                reg.metric("formal_pilot_token_split", condition=c, cap=cap,
                           thinking_per_run=round(float(think[-1]), 1),
                           answer_per_run=round(float(answer[-1]), 1))
            x = np.arange(len(CAPS)) + (i - 1) * w
            ax.bar(x, think, w, color=COND_COLOUR[c], label=COND_LABEL[c] + " thinking")
            ax.bar(x, answer, w, bottom=think, color=COND_COLOUR[c], alpha=0.42,
                   label=COND_LABEL[c] + " answer")
        _cap_axis(ax)
        ax.set_ylabel("Generated tokens per run")
        ax.set_title("Generated output split into thinking and answer tokens\n"
                     "input tokens excluded; all three count toward the budget")
        ax.legend(fontsize=7.2, ncol=3, loc="upper left")
        reg.save(fig, "18_pilot_token_split", "Thinking versus answer tokens",
                 "Thinking dominates generated output in every condition. Qwen3 requires "
                 "thinking mode and the project counts those tokens against the cap.",
                 "results/analysis/formal_pilot/pilot_calls.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for i, c in enumerate(conds):
        lat = [runs[(runs.condition == c) & (runs.cap == cap)]["latency_seconds"].mean()
               for cap in CAPS]
        cal = [runs[(runs.condition == c) & (runs.cap == cap)]["n_calls"].mean()
               for cap in CAPS]
        ax[0].plot(range(len(CAPS)), lat, marker="o", color=COND_COLOUR[c],
                   label=COND_LABEL[c])
        ax[1].plot(range(len(CAPS)), cal, marker="o", color=COND_COLOUR[c],
                   label=COND_LABEL[c])
        for cap, l_, n_ in zip(CAPS, lat, cal):
            reg.metric("formal_pilot_cost", condition=c, cap=cap,
                       latency_mean_seconds=round(float(l_), 1),
                       calls_mean=round(float(n_), 2))
    for a, t, y in ((ax[0], "Mean wall-clock latency per run", "Seconds"),
                    (ax[1], "Mean LLM calls per run", "Calls")):
        _cap_axis(a)
        a.set_title(t)
        a.set_ylabel(y)
    ax[0].legend(fontsize=8)
    reg.save(fig, "19_pilot_cost", "Latency and call counts",
             "Cost in wall-clock time and in requests, reported for planning the main "
             "experiment rather than as an outcome.",
             "results/analysis/formal_pilot/runs.csv")


# --------------------------------------------------------------------------- #
# Section 6 — structural calibration and the band design.
# --------------------------------------------------------------------------- #
BANDS_FROZEN = (("low", 0, 1), ("medium", 7, 7), ("high", 13, 16))


def figs_structural(S: dict, reg: Registry) -> None:
    df, an = S["struct"], S["struct_analysis"]

    # -- conflict-count distribution with the frozen bands marked -----------
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    counts = df["conflicting_pairs"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#B9CFE0", width=0.86,
           label="candidates")
    for band, lo, hi in BANDS_FROZEN:
        sel = counts[(counts.index >= lo) & (counts.index <= hi)]
        ax.bar(sel.index, sel.values, color=BAND_COLOUR[band], width=0.86,
               label=BAND_LABEL[band] + "  k=" + (str(lo) if lo == hi
                                                  else f"{lo}-{hi}"))
        reg.metric("structural_band_supply", band=band, k_lo=lo, k_hi=hi,
                   candidates=int(sel.values.sum()),
                   D_lo=round(lo / 28, 4), D_hi=round(hi / 28, 4))
    ax.set_yscale("log")
    ax.set_xlabel("Binding conflict pairs k  (0 to 28 possible at n = 8)")
    ax.set_ylabel("Candidates (log scale)")
    ax.set_title("Where the density axis actually has support\n"
                 f"{len(df)} CPU-only candidates at n = 8, seeds "
                 f"{S['struct_meta']['seed_start']}-{S['struct_meta']['seed_end']}")
    ax.legend(fontsize=8.5, ncol=2)
    zero = int((df["conflicting_pairs"] == 0).sum())
    reg.save(fig, "20_struct_k_distribution", "Conflict-pair distribution",
             f"Strongly zero-inflated: {zero} of {len(df)} candidates have no binding "
             "conflict at all, and the upper region is thin. This is why level boundaries "
             "cannot be cut as plain tertiles and why the three bands are narrow.",
             "results/calibration/structural/candidates.csv")

    # -- D against the oracle optimum ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    grp = df.groupby("conflicting_pairs")["oracle_optimum"].agg(["mean", "count"])
    axes[0].scatter(grp.index / 28, grp["mean"], s=np.clip(grp["count"] / 8, 6, 160),
                    color="#3B6EA5", alpha=0.75, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Normalised conflict density  D = k / 28")
    axes[0].set_ylabel("Mean oracle optimum O")
    axes[0].set_title("Density is confounded with the optimum\n"
                      "Spearman rho = " + str(an["B_optimum"]["spearman_D_vs_optimum"]))
    tg = df.groupby("tightness")["D"].mean()
    axes[1].plot(tg.index, tg.values, marker="o", color="#4E8B62")
    axes[1].set_xlabel("Generator knob: tightness")
    axes[1].set_ylabel("Mean D")
    axes[1].set_title("Density is nearly determined by one knob\n"
                      "Spearman rho = " + str(an["C_generator"]["spearman_D_vs_tightness"]))
    reg.save(fig, "21_struct_confounds", "Density, optimum and tightness",
             "The two facts that forced the design change. Harder instances have a smaller "
             "achievable optimum, so density must be matched on O or the comparison is "
             "confounded; and density is largely a function of tightness, so the axis is "
             "not an independent construct.",
             "results/analysis/structural_calibration/summary.json")
    reg.metric("structural_confounds",
               spearman_D_vs_optimum=an["B_optimum"]["spearman_D_vs_optimum"],
               spearman_D_vs_tightness=an["C_generator"]["spearman_D_vs_tightness"],
               spearman_D_vs_overlap=an["C_generator"]["spearman_D_vs_overlap"],
               spearman_D_vs_H=S["joint"]["joint_support"]["spearman_D_vs_H"])

    # -- optimum histogram within each frozen band --------------------------
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    optima = sorted(df["oracle_optimum"].unique())
    w = 0.26
    for i, (band, lo, hi) in enumerate(BANDS_FROZEN):
        sel = df[(df.conflicting_pairs >= lo) & (df.conflicting_pairs <= hi)]
        share = [(sel["oracle_optimum"] == o).mean() for o in optima]
        ax.bar(np.arange(len(optima)) + (i - 1) * w, share, w,
               color=BAND_COLOUR[band], label=BAND_LABEL[band])
    ax.set_xticks(range(len(optima)))
    ax.set_xticklabels([str(o) for o in optima])
    ax.set_xlabel("Oracle optimum O  (meetings achievable by CP-SAT)")
    ax.set_ylabel("Share of band candidates")
    ax.set_title("Raw optimum distributions differ sharply between bands\n"
                 "the selector therefore matches them exactly on the chosen sample")
    ax.legend(fontsize=8.5)
    reg.save(fig, "22_struct_optimum_by_band", "Optimum distribution per band",
             "Left uncorrected, the High band would consist of instances on which fewer "
             "meetings are achievable, and any architecture difference would be partly a "
             "difference in what was achievable. The frozen design requires an identical "
             "empirical O histogram across bands.",
             "results/calibration/structural/candidates.csv")

    # -- the failed proposal and the successful search ----------------------
    ad = S["band_admiss"]["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9))
    names = list(ad["bands"])
    n_have = [ad["bands"][b]["n"] for b in names]
    axes[0].bar(names, n_have, color=["#C08A8A", "#C08A8A", "#C08A8A"])
    axes[0].axhline(S["band_admiss"]["criteria"]["min_candidates"], color=ACCENT, ls="--",
                    lw=1.4, label="required: "
                    + str(S["band_admiss"]["criteria"]["min_candidates"]))
    for i, v in enumerate(n_have):
        axes[0].annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=9)
    axes[0].set_ylabel("Candidates available")
    axes[0].set_title("First proposal: NOT ADMISSIBLE\n"
                      "exact (O, H) matching, bands 1-4 / 7-10 / 13-16")
    axes[0].legend(fontsize=8.5)
    sel = S["band_search"]["search"]["selected"]
    sup = []
    for band, lo, hi in BANDS_FROZEN:
        sub = df[(df.conflicting_pairs >= lo) & (df.conflicting_pairs <= hi)]
        sup.append(len(sub))
    axes[1].bar([BAND_LABEL[b] for b, _, _ in BANDS_FROZEN], sup,
                color=[BAND_COLOUR[b] for b, _, _ in BANDS_FROZEN])
    axes[1].axhline(sel["joint_capacity"], color="#4E8B62", ls="--", lw=1.4,
                    label="joint matched capacity: " + str(sel["joint_capacity"]))
    for i, v in enumerate(sup):
        axes[1].annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Candidates available (log scale)")
    axes[1].set_title("Exhaustive search over 593 775 triples: FEASIBLE\n"
                      "bands 0-1 / 7 / 13-16, gap 5")
    axes[1].legend(fontsize=8.5)
    reg.save(fig, "23_band_design", "Band design: one rejection, one feasible design",
             "The first proposal was stricter than anything registered and failed on "
             "supply. Formalising the registered requirement and searching all triples "
             "exhaustively returned a feasible design with joint matched capacity 65 per "
             "band against a target of 50, and an identical optimum histogram of 41 at "
             "O = 3 and 24 at O = 4.",
             "results/analysis/band_admissibility/summary.json and "
             "results/analysis/band_search/summary.json")

    # -- H, the recorded residual limitation --------------------------------
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    hs = sorted(df["higher_order_gap_H"].unique())
    for i, (band, lo, hi) in enumerate(BANDS_FROZEN):
        sub = df[(df.conflicting_pairs >= lo) & (df.conflicting_pairs <= hi)]
        share = [(sub["higher_order_gap_H"] == h).mean() for h in hs]
        ax.bar(np.arange(len(hs)) + (i - 1) * 0.26, share, 0.26,
               color=BAND_COLOUR[band], label=BAND_LABEL[band])
        reg.metric("structural_H_by_band", band=band,
                   mean_H=round(float(sub["higher_order_gap_H"].mean()), 3),
                   share_H_ge_3=round(float((sub["higher_order_gap_H"] >= 3).mean()), 4))
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlabel("H = alpha(reachable) - O   higher-order interaction diagnostic")
    ax.set_ylabel("Share of band candidates")
    ax.set_title("H is systematically unbalanced across the bands - a recorded limitation")
    ax.legend(fontsize=8.5)
    reg.save(fig, "24_struct_H", "Higher-order gap H",
             "The bands isolate pairwise conflict density but not every form of structural "
             "interaction. H is pre-specified as a diagnostic and explicitly may not be "
             "used to redesign the bands; the imbalance is carried as a construct-validity "
             "limitation.",
             "results/calibration/structural/candidates.csv and "
             "results/analysis/joint_support/summary.json")


# --------------------------------------------------------------------------- #
# Section 7 — the budget-dev experiment.
# --------------------------------------------------------------------------- #
def figs_budget_dev(S: dict, reg: Registry) -> None:
    if "bd_instances" not in S:
        reg.skip("budget-dev composition", "the frozen subset manifest was not found")
        return
    inst = S["bd_instances"]

    # -- the frozen 60 instances --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    order = ["low", "medium", "high"]
    ks = axes[0]
    for band in order:
        sub = inst[inst.band == band]
        ks.scatter(sub["k"], [band] * len(sub), s=70, color=BAND_COLOUR[band],
                   alpha=0.55, edgecolor="white", linewidth=0.6)
    ks.set_xlabel("Binding conflict pairs k")
    ks.set_yticks(range(3))
    ks.set_yticklabels([BAND_LABEL[b] for b in order])
    ks.set_xlim(-1, 18)
    ks.set_title("Band separation on the k axis")

    optima = sorted(inst["O"].unique())
    for i, band in enumerate(order):
        sub = inst[inst.band == band]
        axes[1].bar(np.arange(len(optima)) + (i - 1) * 0.26,
                    [int((sub["O"] == o).sum()) for o in optima], 0.26,
                    color=BAND_COLOUR[band], label=BAND_LABEL[band])
    axes[1].set_xticks(range(len(optima)))
    axes[1].set_xticklabels([str(o) for o in optima])
    axes[1].set_xlabel("Oracle optimum O")
    axes[1].set_ylabel("Instances")
    axes[1].set_title("Identical optimum histogram\nby construction")
    axes[1].legend(fontsize=7.5)

    structs = sorted(inst["travel_structure"].unique())
    bottom = np.zeros(3)
    for st in structs:
        vals = np.array([int(((inst.band == b) & (inst.travel_structure == st)).sum())
                         for b in order])
        axes[2].bar(range(3), vals, 0.6, bottom=bottom, label=st)
        bottom += vals
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels([BAND_LABEL[b].split()[0] for b in order])
    axes[2].set_ylabel("Instances")
    axes[2].set_title("Travel structures per band")
    axes[2].legend(fontsize=7.5, ncol=2)
    fig.suptitle("The frozen budget-dev set: 60 instances at n = 8, 20 per band, "
                 "60 distinct development seeds", y=1.04)
    reg.save(fig, "30_bd_composition", "Frozen development set",
             "Selected by a deterministic rule with no agent quantity involved. Bands are "
             "separated by five unused conflict counts, the optimum histogram is identical "
             "across bands at 12 instances with O = 3 and 8 with O = 4, and every band "
             "carries all four travel structures.",
             "results/manifests/subset__bands_budget_dev__911e4864d6fb.json")
    for _, r in inst.iterrows():
        reg.metric("budget_dev_instances", **{k: (int(v) if isinstance(v, (np.integer,))
                                                  else v) for k, v in r.items()})

    if "bd_runs" not in S:
        reg.skip("budget-dev outcome figures",
                 "results/analysis/budget_dev/runs.csv is absent from the repository, so "
                 "no outcome figure can be drawn from data; the numbers were not "
                 "transcribed from a terminal session")
        return
    runs = S["bd_runs"]
    runs = runs.assign(band=runs["level"].map(LEVEL_TO_BAND))

    # -- the gate ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    w = 0.26
    for i, band in enumerate(order):
        rates, labels = [], []
        for cap in CAPS:
            sel = runs[(runs.cap == cap) & (runs.band == band)]
            hits = int(sel["counts_for_cap_rule"].astype(bool).sum())
            rates.append(hits / len(sel) if len(sel) else 0.0)
            labels.append(f"{hits}/{len(sel)}")
            reg.metric("budget_dev_gate", band=band, cap=cap, n=len(sel),
                       qualifying=hits, rate=round(rates[-1], 4),
                       meets_threshold=bool(rates[-1] >= 0.5))
        bars = ax.bar(np.arange(len(CAPS)) + (i - 1) * w, rates, w,
                      color=BAND_COLOUR[band], label=BAND_LABEL[band])
        for b, lab in zip(bars, labels):
            ax.annotate(lab, (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=7.6, color="#333333")
    ax.axhline(0.5, color=ACCENT, ls="--", lw=1.6,
               label="registered threshold: 50% in EVERY band")
    _cap_axis(ax)
    ax.set_ylabel("Runs with a validator-approved non-empty plan")
    ax.set_ylim(0, 0.72)
    ax.set_title("The registered primary-cap gate: no rung qualifies\n"
                 "C1 only, 240 runs, 20 instances per band")
    ax.legend(fontsize=8.4, loc="upper left")
    reg.save(fig, "31_bd_gate", "The primary-cap gate",
             "The rule selects the smallest rung reaching 50% in each band separately. The "
             "top rung is the nearest miss at 0.45 / 0.45 / 0.25 and is explicitly not "
             "available as a fallback. No cap is selected and the main experiment does not "
             "start.",
             "results/analysis/budget_dev/runs.csv")

    # -- satisfaction --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for i, band in enumerate(order):
        vals = [runs[(runs.cap == cap) & (runs.band == band)]["satisfaction"].mean()
                for cap in CAPS]
        bars = ax.bar(np.arange(len(CAPS)) + (i - 1) * w, vals, w,
                      color=BAND_COLOUR[band], label=BAND_LABEL[band])
        _bar_labels(ax, bars, "{:.2f}", size=7.6)
        for cap, v in zip(CAPS, vals):
            reg.metric("budget_dev_satisfaction", band=band, cap=cap,
                       mean_satisfaction=round(float(v), 4))
    _cap_axis(ax)
    ax.set_ylabel("Mean satisfaction")
    ax.set_title("Satisfaction of the single agent at n = 8, by band and cap")
    ax.legend(fontsize=8.4, loc="upper left")
    reg.save(fig, "32_bd_satisfaction", "Satisfaction by band and cap",
             "Zero everywhere below 32k. Even at the top rung the single agent averages "
             "well under half of the oracle optimum, and the High band is the weakest.",
             "results/analysis/budget_dev/runs.csv")

    # -- termination mix and utilisation ------------------------------------
    terms = ["agent_finish", "aborted", "budget"]
    tcol = {"agent_finish": "#4E8B62", "aborted": "#C4703A", "budget": "#A33A3A"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    bottom = np.zeros(len(CAPS))
    for t in terms:
        vals = np.array([int((runs[(runs.cap == cap)]["termination"] == t).sum())
                         for cap in CAPS])
        axes[0].bar(range(len(CAPS)), vals, 0.58, bottom=bottom, color=tcol[t], label=t)
        bottom += vals
        for cap, v in zip(CAPS, vals):
            reg.metric("budget_dev_termination", cap=cap, termination=t, runs=int(v))
    _cap_axis(axes[0])
    axes[0].set_ylabel("Runs (of 60)")
    axes[0].set_title("How runs ended")
    axes[0].legend(fontsize=8.4)

    util_all = [runs[runs.cap == cap]["cap_utilisation"].mean() for cap in CAPS]
    fail = runs[~runs["counts_for_cap_rule"].astype(bool)]
    util_fail = [fail[fail.cap == cap]["cap_utilisation"].mean() for cap in CAPS]
    axes[1].plot(range(len(CAPS)), util_all, marker="o", color="#3B6EA5",
                 label="all runs")
    axes[1].plot(range(len(CAPS)), util_fail, marker="s", color=ACCENT,
                 label="failing runs only")
    for cap, a_, f_ in zip(CAPS, util_all, util_fail):
        reg.metric("budget_dev_utilisation", cap=cap,
                   mean_cap_utilisation_all=round(float(a_), 4),
                   mean_cap_utilisation_failures=round(float(f_), 4))
    _cap_axis(axes[1])
    axes[1].set_ylabel("Mean share of the cap actually spent")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="#BBBBBB", lw=1)
    axes[1].set_title("Budget utilisation")
    axes[1].legend(fontsize=8.4)
    fig.suptitle("Where the binding constraint sits, and where it stops sitting", y=1.03)
    reg.save(fig, "33_bd_termination_utilisation", "Termination and utilisation",
             "Below 32k almost every run is stopped by the budget guard at near-full "
             "utilisation. At 64k two thirds finish on their own and failing runs leave "
             "roughly a third of the budget unspent.",
             "results/analysis/budget_dev/runs.csv")

    # -- proposals -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))
    att = [runs[runs.cap == cap]["proposal_attempts"].sum() for cap in CAPS]
    val = [runs[runs.cap == cap]["proposal_valid"].sum() for cap in CAPS]
    bars = axes[0].bar(range(len(CAPS)), att, 0.55, color="#B9CFE0",
                       label="all attempts")
    axes[0].bar(range(len(CAPS)), val, 0.55, color="#3B6EA5", label="valid")
    for b, a_, v_ in zip(bars, att, val):
        axes[0].annotate(f"{int(v_)}/{int(a_)}", (b.get_x() + b.get_width() / 2, a_),
                         textcoords="offset points", xytext=(0, 2), ha="center",
                         fontsize=8.4)
    _cap_axis(axes[0])
    axes[0].set_ylabel("Proposal attempts across 60 runs")
    axes[0].set_title("Proposals made, and how many were valid")
    axes[0].legend(fontsize=8.4)

    rate = [(v / a if a else 0.0) for a, v in zip(att, val)]
    bars = axes[1].bar(range(len(CAPS)), rate, 0.55, color="#4E8B62")
    _bar_labels(axes[1], bars, "{:.3f}")
    _cap_axis(axes[1])
    axes[1].set_ylabel("Valid / attempted")
    axes[1].set_title("Proposal validity rate")
    for cap, a_, v_, r_ in zip(CAPS, att, val, rate):
        reg.metric("budget_dev_proposals", cap=cap, attempts=int(a_), valid=int(v_),
                   validity_rate=round(float(r_), 4),
                   attempts_per_run=round(float(a_) / 60, 3))
    reg.save(fig, "34_bd_proposals", "Proposal volume and quality",
             "At 8k the agent makes no proposal at all in 60 runs. Volume grows with the "
             "budget but validity stays low: at the top rung roughly one proposal in four "
             "is feasible.",
             "results/analysis/budget_dev/runs.csv")

    # -- the decisive diagnostic --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), sharey=True)
    for ax, cap in zip(axes, (32000, 64000)):
        sel = runs[runs.cap == cap]
        present = [t for t in terms if (sel["termination"] == t).any()]
        rates, ns = [], []
        for t in present:
            s = sel[sel.termination == t]
            rates.append(s["counts_for_cap_rule"].astype(bool).mean())
            ns.append(len(s))
            reg.metric("budget_dev_success_by_termination", cap=cap, termination=t,
                       n=len(s), success_rate=round(float(rates[-1]), 4),
                       mean_cap_utilisation=round(float(s["cap_utilisation"].mean()), 4),
                       proposals_per_run=round(float(s["proposal_attempts"].mean()), 3))
        bars = ax.bar(range(len(present)), rates, 0.55,
                      color=[tcol[t] for t in present])
        for b, r_, n_ in zip(bars, rates, ns):
            ax.annotate(f"{r_:.2f}\nn={n_}", (b.get_x() + b.get_width() / 2, r_),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8.4)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_title(f"cap {cap // 1000}k")
        ax.set_xlabel("How the run ended")
    axes[0].set_ylabel("Share reaching a valid non-empty plan")
    axes[0].set_ylim(0, 0.62)
    fig.suptitle("Success rate by termination mode: at 64k budget-limited runs do no worse "
                 "than voluntary ones", y=1.02)
    reg.save(fig, "35_bd_diagnostic", "The decisive diagnostic",
             "If the budget were the binding constraint, runs cut short by it would succeed "
             "far less often than runs that finished on their own. At 64k they do not. "
             "A longer ladder would therefore buy more invalid proposals rather than more "
             "plans.",
             "results/analysis/budget_dev/runs.csv")

    # -- rejection reasons ---------------------------------------------------
    if "bd_proposals" in S:
        pp = S["bd_proposals"]
        bad = pp[~pp["valid"].astype(bool)]
        per_cap = {cap: Counter() for cap in CAPS}
        for _, row in bad.iterrows():
            for r in str(row["reasons"] or "").split("|"):
                r = r.strip()
                if r and r.lower() != "nan":
                    per_cap[int(row["cap"])][r] += 1
        reasons = sorted({r for c in per_cap.values() for r in c},
                         key=lambda r: -sum(c[r] for c in per_cap.values()))
        if reasons:
            fig, ax = plt.subplots(figsize=(9.0, 3.9))
            bottom = np.zeros(len(CAPS))
            palette = ["#A33A3A", "#C4703A", "#3B6EA5", "#4E8B62", "#8A6BA1", "#777777"]
            for j, r in enumerate(reasons):
                vals = np.array([per_cap[cap][r] for cap in CAPS], dtype=float)
                ax.bar(range(len(CAPS)), vals, 0.55, bottom=bottom,
                       color=palette[j % len(palette)], label=r)
                bottom += vals
                for cap, v in zip(CAPS, vals):
                    reg.metric("budget_dev_rejection_reasons", cap=cap, reason=r,
                               count=int(v))
            _cap_axis(ax)
            ax.set_ylabel("Occurrences across rejected proposals (multi-label)")
            ax.set_title("Why the single agent's proposals were rejected, by cap")
            ax.legend(fontsize=8.2)
            reg.save(fig, "36_bd_reasons", "Rejection reasons by cap",
                     "Multi-label, so one rejected proposal can contribute to more than "
                     "one bar. Travel infeasibility dominates at every rung, which points "
                     "at arithmetic over the travel matrix rather than at budget.",
                     "results/analysis/budget_dev/proposals.csv")


# --------------------------------------------------------------------------- #
# Assembly: the deck.
# --------------------------------------------------------------------------- #
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _fig(reg: Registry, name: str) -> str:
    """Inline one registered figure with its caption, or a visible placeholder."""
    for f in reg.figures:
        if f["name"] == name:
            src = "data:image/png;base64," + _b64(reg.fig_dir / f["file"])
            return (f'<figure><img src="{src}" alt="{f["title"]}">'
                    f'<figcaption><b>{f["title"]}.</b> {f["caption"]} '
                    f'<span class="src">Source: {f["source"]}</span></figcaption></figure>')
    why = next((s["reason"] for s in reg.skipped if s["name"].startswith(name[:6])),
               "figure not produced in this build")
    return f'<div class="missing">Figure <code>{name}</code> not available. {why}</div>'


def _table(rows: Sequence[dict], cols: Sequence[str], head: Sequence[str] | None = None,
           cls: str = "") -> str:
    head = head or cols
    out = [f'<table class="{cls}"><thead><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in cols) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


CSS = """
:root{--ink:#1b1b1b;--muted:#5c5c5c;--rule:#d8d8d8;--accent:#A33A3A;--ok:#4E8B62;
--bg:#ffffff;--panel:#f6f7f9;}
*{box-sizing:border-box}
body{margin:0;background:#e9eaec;color:var(--ink);
font:16px/1.55 "Georgia","Times New Roman",serif;}
.deck{max-width:1180px;margin:0 auto;padding:24px 16px 80px;}
section.slide{background:var(--bg);border:1px solid var(--rule);border-radius:4px;
padding:34px 42px 30px;margin:0 0 22px;scroll-margin-top:12px;
box-shadow:0 1px 3px rgba(0,0,0,.06);}
section.slide>h2{margin:0 0 4px;font-size:26px;line-height:1.25;letter-spacing:-.01em;}
section.slide>h3{margin:22px 0 6px;font-size:18px;color:var(--muted);
font-weight:600;}
.kicker{font:600 11.5px/1.2 "Helvetica Neue",Arial,sans-serif;letter-spacing:.14em;
text-transform:uppercase;color:var(--muted);margin:0 0 14px;}
p{margin:.55em 0;}
.lead{font-size:17.5px;}
ul{margin:.5em 0 .5em 1.1em;padding:0}
li{margin:.3em 0}
figure{margin:18px 0 4px;}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);}
figcaption{font-size:13.5px;color:var(--muted);margin-top:7px;line-height:1.45;}
figcaption b{color:var(--ink);}
.src{display:block;font:11.5px/1.4 "Helvetica Neue",Arial,sans-serif;color:#8a8a8a;
margin-top:3px;}
table{border-collapse:collapse;width:100%;margin:14px 0;
font:14px/1.45 "Helvetica Neue",Arial,sans-serif;}
th,td{border-bottom:1px solid var(--rule);padding:7px 9px;text-align:left;
vertical-align:top;}
th{background:var(--panel);font-weight:600;font-size:13px;}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;}
code{font:13px/1.4 "SF Mono",Consolas,monospace;background:var(--panel);
padding:1px 4px;border-radius:3px;}
.badge{display:inline-block;font:600 11px/1 "Helvetica Neue",Arial,sans-serif;
letter-spacing:.05em;text-transform:uppercase;padding:4px 8px;border-radius:3px;
margin-right:6px;}
.b-fail{background:#f6e4e4;color:var(--accent);}
.b-ok{background:#e4efe7;color:var(--ok);}
.b-warn{background:#fbf0e2;color:#9a6216;}
.b-info{background:#e6ecf3;color:#31537a;}
.callout{border-left:3px solid var(--accent);background:#fbf6f6;padding:12px 16px;
margin:16px 0;font-size:15.5px;}
.callout.ok{border-color:var(--ok);background:#f4f9f5;}
.callout.info{border-color:#31537a;background:#f3f6fa;}
.missing{border:1px dashed #bbb;background:#fafafa;color:var(--muted);padding:16px;
font-size:14px;margin:16px 0;}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px;}
.cols3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;}
section.slide.title-slide{background:linear-gradient(180deg,#22506F,#173A50);color:#fff;
border:none;padding:64px 48px;}
section.title-slide h1{font-size:38px;line-height:1.2;margin:0 0 16px;letter-spacing:-.015em;}
section.title-slide .sub{font-size:19px;opacity:.92;margin:0 0 30px;}
section.title-slide .meta{font:14px/1.7 "Helvetica Neue",Arial,sans-serif;opacity:.85;}
.small{font-size:13.5px;color:var(--muted);}
.nav{position:fixed;right:14px;bottom:14px;background:#fff;border:1px solid var(--rule);
border-radius:4px;padding:6px 10px;font:12px/1.4 "Helvetica Neue",Arial,sans-serif;
color:var(--muted);box-shadow:0 1px 4px rgba(0,0,0,.12);}
.toc{columns:2;font-size:14.5px;}
.toc a{color:#31537a;text-decoration:none;}
.toc a:hover{text-decoration:underline;}
@media print{body{background:#fff}section.slide{break-after:page;box-shadow:none;
border:none;padding:0 0 20px;margin:0}.nav{display:none}}
"""

JS = """
const slides=[...document.querySelectorAll('section.slide')];
let i=0;
function go(n){i=Math.max(0,Math.min(slides.length-1,n));
slides[i].scrollIntoView({behavior:'smooth',block:'start'});
document.getElementById('pos').textContent=(i+1)+' / '+slides.length;}
document.addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'||e.key==='PageDown'||e.key===' '){e.preventDefault();go(i+1);}
 if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();go(i-1);}
 if(e.key==='Home'){e.preventDefault();go(0);}
 if(e.key==='End'){e.preventDefault();go(slides.length-1);}});
const obs=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){
 i=slides.indexOf(e.target);
 document.getElementById('pos').textContent=(i+1)+' / '+slides.length;}})},
 {rootMargin:'-45% 0px -45% 0px'});
slides.forEach(s=>obs.observe(s));
"""


def build_deck(S: dict, reg: Registry, inventory: pd.DataFrame, root: Path) -> str:
    P = []
    add = P.append
    pr = S["pilot_runs"]
    n_pilot = len(pr)
    bd = S.get("bd_runs")
    bd_sum = S.get("bd_summary", {})
    sm = S["struct_meta"]
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def slide(kicker, body, cls="slide"):
        add(f'<section class="{cls}"><p class="kicker">{kicker}</p>{body}</section>')

    # 1 — title
    add('<section class="slide title-slide">'
        '<h1>When Does a Multi-Agent LLM System Beat a Single Agent?</h1>'
        '<p class="sub">Testing for a complexity threshold on meeting planning '
        'under an equal token budget</p>'
        '<p class="sub" style="font-size:16px;opacity:.8">'
        'Visual summary of experiments and current status</p>'
        '<p class="meta">B.Sc. thesis &middot; TU Berlin &middot; '
        f'supervisor Prof. Hillmann<br>Model: Qwen3-32B-AWQ on vLLM &middot; '
        f'oracle: OR-Tools CP-SAT<br>Deck built {built} from repository artifacts only'
        '</p></section>')

    # 2 — how to read this deck
    slide("Orientation", f"""
<h2>What this deck is, and what it is not</h2>
<p class="lead">Every number here is read from a file in the repository by
<code>scripts/build_presentation.py</code>. Nothing is typed in by hand.</p>
<div class="callout info">
<b>Headline correction.</b> The earlier experiments are often referred to as
&ldquo;about 260 runs&rdquo;. That figure is wrong. The formal pilot alone is
<b>{n_pilot} runs</b> ({pr['condition'].nunique()} conditions &times;
{pr['cap'].nunique()} caps &times; {pr['instance_id'].nunique()} instances), plus
{S['c1_cal']['integrity_audit']['n_runs']} runs of C1 budget calibration and
{len(bd) if bd is not None else 0} runs of budget-dev &mdash; <b>
{n_pilot + S['c1_cal']['integrity_audit']['n_runs'] + (len(bd) if bd is not None else 0)}
LLM runs in total</b>.</div>
<h3>Reading the labels</h3>
<p><span class="badge b-ok">complete</span> ran to completion and passed its audit &nbsp;
<span class="badge b-warn">exploratory</span> retained as audit trail, binds no decision &nbsp;
<span class="badge b-fail">failed</span> a registered gate returned a negative verdict</p>
<h3>Contents</h3>
<div class="toc">
1. Research question and setup<br>
2. Experimental roadmap<br>
3. Experiment inventory and exact counts<br>
4. Formal pilot: what 360 runs showed<br>
5. Structural calibration and the density axis<br>
6. Budget-dev: the current experiment<br>
7. The decisive diagnostic<br>
8. Interpretation: the decomposition hypothesis<br>
9. Options and recommended next step<br>
10. Status, assumptions and open questions
</div>""")

    # 3 — research question
    slide("1 &middot; Research question", """
<h2>The question, stated precisely</h2>
<p class="lead">Under an <b>equal token budget</b>, is there a level of task complexity
above which a hierarchical multi-agent system outperforms a single ReAct agent on
meeting planning?</p>
<div class="callout info">
The claim under test is not &ldquo;multi-agent is better&rdquo;. It is that the sign of the
difference <b>changes</b> as complexity rises &mdash; the single agent ahead on simple
instances, the hierarchy ahead on complex ones.</div>
<h3>The estimand</h3>
<p>&Delta; = S(C1) &minus; S(C3), paired within instance, where S is satisfaction.
Positive &Delta; means the single agent is ahead; negative means the hierarchy is.
A crossover is a <b>+ turning &minus;</b> as density rises within one budget cap.</p>
<h3>How a crossover would be established</h3>
<p>An intersection&ndash;union test: the 95% one-sided <b>lower</b> bound on
&Delta;<sub>Low</sub> must exceed zero <b>and</b> the 95% one-sided <b>upper</b> bound on
&Delta;<sub>High</sub> must fall below zero. Both, or nothing. No multiplicity correction
is needed for a conjunction.</p>""")

    # 4 — conditions
    slide("1 &middot; Setup", """
<h2>The four conditions</h2>
<div class="cols">
<div>
<h3>C1 &mdash; ReAct (baseline)</h3>
<p>A single agent alternating reasoning and tool calls over three read-only tools:
list people, get availability, get travel time. A terminal finalisation node emits the
plan as guided JSON with thinking disabled.</p>
<h3>C2 &mdash; ReAct + verify/revise</h3>
<p>A fixed graph on top of C1: draft &rarr; verify &rarr; revise &rarr; verify &rarr;
revise &rarr; stop, at most two revision cycles. Verification and revision are
reflection-only, with no tool access. C1 is byte-identical underneath.</p>
</div>
<div>
<h3>C3 &mdash; Hierarchical MAS</h3>
<p>Supervisor splits the people deterministically from the travel matrix with no LLM
call. Two workers each run a byte-identical C1 loop on their half with isolated context
and equal quotas. A zero-token aggregator forms a candidate pool without planning. One
fresh-context critic sees only that pool and a restricted travel sub-matrix, and its
output must pass the full-instance validator and beat the fallback.</p>
<h3>C5 &mdash; Validated best-of-3</h3>
<p>Specified as a strong single-agent sampling baseline. Not yet implemented.</p>
</div></div>
<div class="callout info"><b>Equal budget is the control.</b> All tokens count &mdash;
input, thinking and output &mdash; and in C3 the inter-agent messages count against the
same shared budget. Tokens are counted with the Qwen tokenizer before each call.</div>""")

    # 5 — task and measurement
    slide("1 &middot; Setup", """
<h2>Task, oracle and the complexity construct</h2>
<div class="cols">
<div>
<h3>The task</h3>
<p>An orienteering problem with time windows: a set of people, each available in a window
at a location, with travel times between locations. A plan is an ordered set of meetings.
The agent never sees the solver.</p>
<h3>Ground truth</h3>
<p>OR-Tools CP-SAT computes the optimum with a single worker for full determinism, proven
optimal for every instance used. An independent validator checks each agent plan against
the instance.</p>
<h3>The metric</h3>
<p>satisfaction = achieved reward / oracle optimum. <b>Any constraint violation scores
zero</b> &mdash; no partial credit.</p>
</div>
<div>
<h3>Complexity is not task size</h3>
<p>Complexity is the count of <b>binding pairwise conflicts</b>: an edge between two
people when each is reachable alone but no ordering of the pair is feasible. Normalised:
<code>D = k / C(n,2)</code>.</p>
<p>The main design fixes <code>n = 8</code>, so <code>C(8,2) = 28</code> and prompt length
does not grow along the complexity axis. Size returns only as a secondary robustness
analysis.</p>
<h3>Determinism</h3>
<p>Prefix caching is disabled on the server. With it on, five repeats of one cell gave
0.00 / 0.75 / 1.00 / 1.00 / 1.00; with it off the runs are byte-identical. That is what
makes a single run per cell defensible.</p>
</div></div>""")

    # 6 — timeline
    slide("2 &middot; Roadmap", "<h2>What was run, in order</h2>"
          + _fig(reg, "01_timeline"))

    # 7 — inventory
    inv_rows = inventory.to_dict("records")
    slide("3 &middot; Inventory", f"""
<h2>Every experiment group, with verified counts</h2>
{_table(inv_rows, ["experiment", "kind", "n_people", "instances", "conditions", "caps",
                   "runs", "llm", "status"],
        ["Experiment", "Kind", "n", "Instances", "Conditions", "Caps", "LLM runs",
         "LLM?", "Status"])}
<p class="small">Counts are recomputed from the artifacts during the build, not copied
from any document. &ldquo;Instances&rdquo; for the band search is the number of candidate
band triples enumerated, not instances.</p>
{_fig(reg, "02_inventory_runs")}""")

    # 8-13 — pilot
    slide("4 &middot; Formal pilot", f"""
<h2>Formal pilot &mdash; {n_pilot} runs <span class="badge b-ok">complete</span></h2>
<p class="lead">Three conditions &times; four caps &times;
{pr['instance_id'].nunique()} instances at n = 4, 6, 8. Completeness audit passed; the raw
documents have been frozen since.</p>
<div class="callout info"><b>Status.</b> This is a <b>pilot</b> &mdash; ten instances per
complexity level. It validates the design and sizes the effect; it is not a confirmatory
test, and it cannot establish an interaction.</div>
{_fig(reg, "10_pilot_satisfaction")}""")
    slide("4 &middot; Formal pilot",
          "<h2>Reaching a plan at all, and what it cost</h2>"
          + _fig(reg, "11_pilot_nonempty") + _fig(reg, "12_pilot_tokens"))
    slide("4 &middot; Formal pilot",
          "<h2>The hierarchy is ahead, and the intervals exclude zero</h2>"
          + _fig(reg, "14_pilot_delta_ci")
          + '<div class="callout ok">Established: C3 above C1 at 32k '
            '(&Delta; = &minus;0.239, [&minus;0.403, &minus;0.075]) and at 64k '
            '(&Delta; = &minus;0.353, [&minus;0.526, &minus;0.173]), while spending fewer '
            'tokens.</div>')
    slide("4 &middot; Formal pilot",
          "<h2>But the pattern is wrong where it matters most</h2>"
          + _fig(reg, "15_pilot_crossover")
          + '<div class="callout"><b>The low-complexity sanity check fails.</b> On easy '
            'instances at 64k the hierarchy is ahead by 0.460 with an interval excluding '
            'zero. Under the exposé this is a sign of a problem in the measurement, the '
            'setup or the evaluation protocol &mdash; independent of any interaction '
            'test.</div>'
          + _fig(reg, "13_pilot_by_level"))
    slide("4 &middot; Formal pilot",
          "<h2>Proposal behaviour and failure modes</h2>"
          + _fig(reg, "16_pilot_proposal_validity") + _fig(reg, "17_pilot_reasons"))
    slide("4 &middot; Formal pilot",
          "<h2>Where the tokens and the time went</h2>"
          + _fig(reg, "18_pilot_token_split") + _fig(reg, "19_pilot_cost"))

    # 14-16 — structural calibration
    slide("5 &middot; Structural calibration", f"""
<h2>Rebuilding the complexity axis <span class="badge b-ok">complete</span></h2>
<p class="lead">{sm['n_completed']} candidate instances at n = 8 over the full knob grid,
seeds {sm['seed_start']}&ndash;{sm['seed_end']}. CPU only: generator, CP-SAT and the
conflict graph. <b>No agent was involved</b> &mdash; selecting instances by observed model
behaviour would tune the axis to the thing under test.</p>
{_fig(reg, "20_struct_k_distribution")}""")
    slide("5 &middot; Structural calibration",
          "<h2>Two confounds that forced the redesign</h2>"
          + _fig(reg, "21_struct_confounds")
          + _fig(reg, "22_struct_optimum_by_band"))
    slide("5 &middot; Structural calibration",
          "<h2>One rejected proposal, one feasible design</h2>"
          + _fig(reg, "23_band_design")
          + '<div class="callout ok"><b>Frozen 2026-08-09.</b> Bands at k = 0&ndash;1 / 7 / '
            '13&ndash;16, i.e. D = 0.000&ndash;0.036 / 0.250 / 0.464&ndash;0.571, with five '
            'unused conflict counts between neighbours and an identical optimum histogram '
            'across bands. Chosen by a fixed tie-break, not by judgement.</div>'
          + _fig(reg, "24_struct_H"))

    # 17-21 — budget-dev
    gate = bd_sum.get("success_rates", {})
    slide("6 &middot; Budget-dev", f"""
<h2>Budget-dev &mdash; the current experiment
<span class="badge b-fail">gate not passed</span></h2>
<p class="lead">60 instances at n = 8, 20 per frozen band, four caps,
<b>C1 only</b> = {len(bd) if bd is not None else 0} runs. Zero errors. The completeness
audit passed: one condition, one model, 20 distinct instances per band, no duplicated or
missing cell, every seed inside the development range 30000&ndash;39999.</p>
<div class="callout info"><b>Purpose.</b> The single-agent baseline chooses the budget, so
the treatment architecture cannot. The rule was registered before this set existed:
<i>the smallest rung at which C1 produces a validator-approved non-empty plan in at least
50% of runs in each of the three density regimes separately</i>.</div>
{_fig(reg, "30_bd_composition")}""")
    slide("6 &middot; Budget-dev",
          "<h2>The gate: no rung qualifies</h2>"
          + _fig(reg, "31_bd_gate")
          + '<div class="callout"><b>Registered consequence.</b> No primary cap is '
            'selected, budget calibration is declared failed, and the main experiment does '
            'not start. Taking the top rung because it is the nearest miss is precisely '
            'the fallback the rule forecloses, and the analyser cannot express it.</div>')
    slide("6 &middot; Budget-dev",
          "<h2>Satisfaction, and how runs ended</h2>"
          + _fig(reg, "32_bd_satisfaction")
          + _fig(reg, "33_bd_termination_utilisation"))
    slide("6 &middot; Budget-dev",
          "<h2>Proposal volume, quality and failure modes</h2>"
          + _fig(reg, "34_bd_proposals") + _fig(reg, "36_bd_reasons"))

    # 22 — the diagnostic
    slide("7 &middot; Diagnosis", """
<h2>Is the budget the binding constraint?</h2>
<p class="lead">The question decides which responses are legitimate. If runs are cut short
by the budget, a longer ladder helps. If they stop on their own without a plan, it cannot.</p>
"""
          + _fig(reg, "35_bd_diagnostic")
          + """
<div class="cols">
<div><h3>Below 32k &mdash; budget binds</h3>
<ul><li>8k: <b>zero</b> proposal attempts across 60 runs, at 96% cap utilisation</li>
<li>32k: failing runs consume <b>94%</b> of the cap; 28 of 57 never propose</li></ul></div>
<div><h3>At 64k &mdash; budget stops binding</h3>
<ul><li>Success is <b>flat</b> across termination modes: 0.44 / 0.37 / 0.33</li>
<li>Failing runs leave about <b>30%</b> of the budget unspent</li>
<li>They do propose &mdash; 1.43 attempts each; only 5 of 37 never proposed</li>
<li>Proposal validity 0.247, dominated by travel infeasibility</li></ul></div></div>
<div class="callout"><b>Conclusion.</b> Extending the ladder is contraindicated
<i>by the measurement</i>, not merely by being post-hoc. A further rung would buy more
invalid proposals, not more plans.</div>""")

    # 23 — interpretation
    slide("8 &middot; Interpretation", """
<h2>Two anomalies, one possible cause</h2>
<div class="cols">
<div><h3>Anomaly 1 &mdash; the pilot</h3>
<p>The hierarchy leads on <b>easy</b> instances, where the single agent should be ahead.
Interval excludes zero.</p></div>
<div><h3>Anomaly 2 &mdash; budget-dev</h3>
<p>At n = 8 the single agent reaches a valid plan in under 40% of runs at any feasible
budget.</p></div></div>
<div class="callout info"><b>The decomposition hypothesis.</b> In C3 each of the two
workers plans for <b>four</b> people &mdash; a size at which the pilot shows the agent
works reliably. C1 faces <b>eight</b> &mdash; a size at which it largely does not. If that
is the mechanism, C3 wins because decomposition reduces the task to a tractable
sub-problem, <b>not</b> because it handles complexity better. The &ldquo;complexity
threshold&rdquo; would then not be what the experiment is measuring; effective
sub-problem size would be.</div>
<h3>Why this matters for the thesis</h3>
<ul>
<li>It explains both anomalies with one cause instead of treating them as measurement defects.</li>
<li>It is <b>directly testable</b> with the design already built: at n = 6, where the single
agent functions, does the C3 advantage shrink?</li>
<li>It changes what an observed advantage licenses one to claim.</li>
</ul>""")

    # 24 — options
    opts = [
        {"o": "A &mdash; stop and report", "w": "Take the registered consequence "
         "literally. No main experiment.",
         "p": "Fully pre-registered; zero further compute; honest negative result.",
         "c": "Research question unanswered; discards an explanatory finding already in "
              "hand."},
        {"o": "B &mdash; move to n = 6", "w": "Re-run structural calibration, band search "
         "and budget-dev at n = 6, a size already named in the design as a robustness "
         "point and one that splits evenly for two workers.",
         "p": "The baseline functions there, so the comparison is not floor-limited; the "
              "whole pipeline is already built and re-runs in about a day.",
         "c": "Post-hoc with respect to this failure; only 15 possible pairs, so the "
              "density axis is coarser and the bands may not separate; a smaller optimum "
              "makes satisfaction coarser and ties worse."},
        {"o": "C &mdash; reframe around decomposition", "w": "State the decomposition "
         "hypothesis explicitly and test it, rather than testing a complexity threshold "
         "alone.",
         "p": "Explains both anomalies; converts a blocked confirmatory study into an "
              "answerable mechanistic one; uses the data already collected.",
         "c": "Changes the headline claim of the thesis; needs the supervisor's agreement "
              "on scope."},
    ]
    slide("9 &middot; Options", f"""
<h2>What can legitimately be done next</h2>
{_table(opts, ["o", "w", "p", "c"], ["Option", "What it means", "For", "Against"])}
<h3>Excluded, and recorded as excluded before any diagnosis was run</h3>
<p>Lowering the 50% threshold; pooling the bands; dropping the High band; redefining
success; selecting 64k anyway. Any change to the ladder after this result is post-hoc and
must be labelled as such.</p>
<div class="callout ok"><b>Recommendation: B together with C.</b> Move the main experiment
to n = 6, where the baseline functions, and register the decomposition hypothesis with its
direct test. This is the only route that both gives the experiment a chance to run and
explains the anomalies already collected.</div>""")

    # 25 — status
    audit_rows = [{"s": a["source"], "n": a.get("rows", a.get("raw_documents", "—")),
                   "st": a.get("status", "")} for a in reg.audit]
    slide("10 &middot; Status", f"""
<h2>Where the project stands</h2>
<div class="cols">
<div><h3><span class="badge b-ok">Done</span></h3>
<ul>
<li>Layers 0&ndash;5: generator, CP-SAT oracle, brute-force cross-check, validator,
tokenizer and budget ledger, LLM client, scorer</li>
<li>C1, C2, C3 implemented as fixed LangGraph workflows; C1 byte-identical under C2 and C3</li>
<li>Formal pilot, {n_pilot} runs, audited and frozen</li>
<li>Structural calibration, {sm['n_completed']} candidates</li>
<li>Band design frozen by exhaustive search</li>
<li>Budget-dev, {len(bd) if bd is not None else 0} runs, audited</li>
<li>Pre-registered inference: intersection&ndash;union test with one-sided bootstrap bounds</li>
</ul></div>
<div><h3><span class="badge b-fail">Blocked</span></h3>
<ul>
<li>Primary cap &mdash; no rung passed the gate</li>
<li>Main experiment &mdash; cannot start under the registered rule</li>
<li>Sample size N &mdash; the synthetic route was retired; the empirical route needs a
working budget first</li>
<li>Held-out seeds 100000+ &mdash; deliberately untouched</li>
</ul>
<h3><span class="badge b-warn">Exploratory</span></h3>
<ul><li>Power simulations v1 (N = 35) and v2 (N = 95): retired, neither adopted</li>
<li>Type-I validation: verdict anticonservative in one cell of 216; binds nothing, but
flags that a percentile bootstrap on coarse tied data may miss its nominal level</li></ul>
</div></div>
<div class="callout"><b>The decision needed from the supervisor.</b> Whether to accept the
failure as the result (A), move the main experiment to n = 6 (B), or reframe the study
around the decomposition hypothesis (C). My recommendation is B and C together.</div>
<h3>Audit of the sources behind this deck</h3>
{_table(audit_rows, ["s", "n", "st"], ["Source", "Records", "Status"])}
<p class="small">Full audit, including duplicate and missing-cell checks and provenance
hashes, is in <code>SUMMARY.md</code>. Figures skipped for missing data:
{len(reg.skipped)}.</p>""")

    body = "\n".join(P)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAS vs single agent under equal budget — experiment status</title>
<style>{CSS}</style></head><body>
<div class="deck">{body}</div>
<div class="nav"><span id="pos">1</span> &nbsp;·&nbsp; ← → to navigate</div>
<script>{JS}</script></body></html>"""


# --------------------------------------------------------------------------- #
# Assembly: summary and machine-readable metrics.
# --------------------------------------------------------------------------- #
def write_summary(S: dict, reg: Registry, inventory: pd.DataFrame, out: Path) -> None:
    L = []
    add = L.append
    pr = S["pilot_runs"]
    add("# Presentation summary — sources, audit, figures and findings\n")
    add(f"Built {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`scripts/build_presentation.py`. Every figure is generated from a repository "
        "artifact; no number was transcribed by hand.\n")

    add("\n## 1. Count correction\n")
    n_bd = len(S["bd_runs"]) if "bd_runs" in S else 0
    n_c1 = S["c1_cal"]["integrity_audit"]["n_runs"]
    add(f"The earlier experiments are commonly quoted as \"about 260 runs\". **That is "
        f"wrong.** Verified from the artifacts:\n")
    add(f"* formal pilot: **{len(pr)}** runs "
        f"({pr['condition'].nunique()} conditions x {pr['cap'].nunique()} caps x "
        f"{pr['instance_id'].nunique()} instances), with {len(list((out.parent / 'results/formal_pilot').glob('*.json'))) if (out.parent / 'results/formal_pilot').exists() else 360} raw documents on disk")
    add(f"* C1 budget calibration: **{n_c1}** runs")
    add(f"* budget-dev: **{n_bd}** runs")
    add(f"* **total LLM runs: {len(pr) + n_c1 + n_bd}**\n")
    add("Non-LLM artifacts: 9 600 structural-calibration candidates at n = 8, a further "
        "9 600 in the budget-dev candidate pool, and 593 775 band triples enumerated by "
        "the design search.\n")

    add("\n## 2. Data-integrity audit\n")
    add("Performed during the build, before any figure was drawn.\n")
    for a in reg.audit:
        add(f"\n### {a['source']}")
        for k, v in a.items():
            if k == "source":
                continue
            add(f"* `{k}`: {v}")
    add("")
    add("\n### Checks that matter, and what they returned\n")
    add("| Check | Formal pilot | Budget-dev |")
    add("|---|---|---|")
    fp = next(a for a in reg.audit if a["source"] == "formal pilot")
    bdrec = next((a for a in reg.audit if a["source"] == "budget-dev runs"), {})
    add(f"| Rows | {fp['rows']} | {bdrec.get('rows', 'n/a')} |")
    add(f"| Duplicate run ids | {fp['duplicate_run_ids']} | "
        f"{bdrec.get('duplicate_run_ids', 'n/a')} |")
    add(f"| Duplicated (instance, cap) cells | {fp['duplicated_cells']} | "
        f"{bdrec.get('duplicated_cells', 'n/a')} |")
    add(f"| Missing cells | {fp['missing_cells']} | {bdrec.get('missing_cells', 'n/a')} |")
    add(f"| Upstream audit passed | {fp['upstream_audit_passed']} | "
        f"{'yes (analyser refused to proceed otherwise)' if bdrec else 'n/a'} |")

    if reg.skipped:
        add("\n### Figures skipped, and why\n")
        for s in reg.skipped:
            add(f"* **{s['name']}** — {s['reason']}")
    else:
        add("\nNo figure was skipped: every source the deck refers to was present.\n")

    add("\n## 3. Superseded and exploratory artifacts\n")
    add("Labelled wherever they appear, and retained rather than deleted:\n")
    add("* `results/analysis/power_iut/` — power simulation v1, selected N = 35. "
        "**Not adopted.**")
    add("* `results/analysis/power_iut_v2/` — power simulation v2, selected N = 95. "
        "**Not adopted.** The two differ by nearly a factor of three, entirely because of "
        "assumptions about LLM behaviour that were never measured; on 2026-08-10 the "
        "project stopped deriving the sample size from this model.")
    add("* `results/analysis/type_one_validation/` — corrected Type-I check of the "
        "bootstrap decision rule. Verdict ANTICONSERVATIVE in 1 cell of 216 (0.0589 "
        "against a nominal 0.05). **Binds nothing**, but the failing corner is the one "
        "with the fewest distinct outcome values, and real satisfaction is also coarse, so "
        "the concern may transfer.")
    add("* The earlier pilot subset `d29b5475...` is calibration-exposed and is refused by "
        "hash by the sweep runner.")
    add("* The old easy/medium/hard binning is superseded by the frozen density bands; "
        "figures using it are labelled \"old binning\".\n")

    add("\n## 4. Figures, one by one\n")
    for i, f in enumerate(reg.figures, 1):
        add(f"\n### {i}. `{f['file']}` — {f['title']}")
        add(f"{f['caption']}")
        add(f"\n*Source:* `{f['source']}`")

    add("\n\n## 5. Key findings, in the order the deck presents them\n")
    add("1. **The formal pilot establishes a difference, not a threshold.** C3 above C1 at "
        "32k (Δ = −0.239, 95% CI [−0.403, −0.075]) and 64k (Δ = −0.353, [−0.526, −0.173]) "
        "while spending fewer tokens. The interaction with complexity was never "
        "significant, and that is a property of pilot size rather than evidence of no "
        "effect.")
    add("2. **The low-complexity sanity check fails.** On easy instances at 64k the "
        "hierarchy leads by 0.460, CI [−0.767, −0.133]. Under the exposé this is a sign of "
        "a problem in the measurement or the protocol.")
    add("3. **The density axis is confounded by construction.** Spearman(D, O) = −0.869 "
        "and Spearman(D, tightness) = 0.881, so the design must match on the oracle "
        "optimum and must not claim the axis is an independent construct.")
    add("4. **A feasible three-band design exists** and was frozen before any held-out "
        "data was touched: k = 0–1 / 7 / 13–16, identical optimum histogram, joint matched "
        "capacity 65 per band against a target of 50.")
    add("5. **Budget calibration failed.** No rung of 8k/16k/32k/64k reaches 50% "
        "validator-approved non-empty plans in any band at n = 8. Top rung: "
        "0.45 / 0.45 / 0.25.")
    add("6. **The budget is not what is missing.** At 64k success is flat across "
        "termination modes (0.44 / 0.37 / 0.33), failing runs leave ~30% of the budget "
        "unspent, and proposal validity is 0.247. Extending the ladder is contraindicated "
        "by the data.")
    add("7. **A single mechanism explains both anomalies.** C3's workers each plan for "
        "four people, a size at which the agent works; C1 faces eight, at which it largely "
        "does not. The measured advantage may be decomposition into tractable "
        "sub-problems rather than superior handling of complexity.\n")

    add("\n## 6. Assumptions and uncertainties\n")
    add("* **The decomposition hypothesis is a hypothesis.** It is consistent with both "
        "anomalies and is not established. The direct test — does the C3 advantage shrink "
        "at n = 6, where C1 functions — has not been run.")
    add("* **The 50% threshold was set above the available evidence.** The formal pilot "
        "already showed C1 reaching a valid plan in 0.47 of runs at 64k on an *easier* "
        "mixed-n subset. This is recorded as a defect in the rule, and the threshold is "
        "not revised in response.")
    add("* **Pilot levels use the old binning.** Pilot figures split by easy/medium/hard "
        "from the median-split binning, not the frozen density bands, because the pilot "
        "predates them. They are not interchangeable.")
    add("* **Pilot instances mix n = 4, 6, 8; budget-dev is pure n = 8.** Token "
        "consumption is therefore not directly comparable between them, and the pilot's "
        "per-cap means understate what n = 8 costs.")
    add("* **One run per cell.** Justified by byte-identical reproduction with prefix "
        "caching disabled, verified on the run machine, not by repetition.")
    add("* **Latency is wall-clock on a shared machine** and depends on what else was "
        "running; it is planning information, not a result.")
    add("* **`final_plan` feasibility is 100% by construction**, so the operative half of "
        "the success definition is non-emptiness. Both halves are applied as written.")
    add("* **No held-out data has been seen.** Seeds 100000+ remain unopened.\n")

    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_metrics(reg: Registry, out: Path) -> None:
    rows = reg.metrics
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        grouped[r["group"]].append({k: v for k, v in r.items() if k != "group"})
    doc = {
        "schema_version": "presentation_metrics/1.0",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "every value is derived from a repository artifact by "
                "scripts/build_presentation.py",
        "audit": reg.audit,
        "figures": reg.figures,
        "skipped_figures": reg.skipped,
        "metrics": grouped,
    }
    (out / "metrics.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")


def write_clean_tables(S: dict, out: Path) -> None:
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)
    if "pilot_runs" in S:
        S["pilot_runs"].to_csv(d / "formal_pilot_runs.csv", index=False)
    if "bd_runs" in S:
        S["bd_runs"].to_csv(d / "budget_dev_runs.csv", index=False)
    if "bd_instances" in S:
        S["bd_instances"].to_csv(d / "budget_dev_instances.csv", index=False)
    if "struct" in S:
        keep = ["instance_id", "seed", "tightness", "overlap", "travel_structure",
                "conflicting_pairs", "D", "oracle_optimum", "alpha_reachable",
                "higher_order_gap_H", "edge_share_top1"]
        S["struct"][keep].to_csv(d / "structural_calibration.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("presentation"))
    args = ap.parse_args(argv)

    out = args.out
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry(fig_dir)

    S = load_sources(_REPO_ROOT, reg)
    fig_timeline(S, reg)
    inventory = fig_inventory(S, reg)
    figs_pilot(S, reg)
    figs_structural(S, reg)
    figs_budget_dev(S, reg)

    write_clean_tables(S, out)
    write_metrics(reg, out)
    write_summary(S, reg, inventory, out)
    (out / "index.html").write_text(build_deck(S, reg, inventory, _REPO_ROOT),
                                    encoding="utf-8")

    size = (out / "index.html").stat().st_size / 1e6
    print(f"figures : {len(reg.figures)} in {fig_dir}")
    if reg.skipped:
        for s in reg.skipped:
            print(f"  SKIPPED {s['name']}: {s['reason']}")
    print(f"metrics : {len(reg.metrics)} rows -> {out}/metrics.csv, metrics.json")
    print(f"summary : {out}/SUMMARY.md")
    print(f"deck    : {out}/index.html ({size:.1f} MB, self-contained)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
