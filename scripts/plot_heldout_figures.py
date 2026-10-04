# scripts/plot_heldout_figures.py
"""Thesis figures for the held-out experiment. CPU, no LLM, no new statistics.

    uv run --with matplotlib --with numpy python -m scripts.plot_heldout_figures

WHAT THIS IS. Reproducible rendering of the held-out results into figures for the
thesis. Every number plotted is read from a committed CSV; this script computes no
p-value, fits nothing and makes no decision. The only derived quantities are
DESCRIPTIVE percentile-bootstrap intervals on per-condition means (F1, F3, F4, F6),
by the same construction the frozen analyser uses (resample instances with
replacement, 10000 times, seed 20260827). F2 replots the frozen paired-difference
intervals from `contrasts.csv` unchanged.

STYLE. Thesis style: no embedded figure titles and no on-figure method notes -- both
belong in the Overleaf caption. The figures keep axis labels, panel labels, legends,
the F2 zero reference line and only the annotations needed to read the sign of a
difference. Suggested captions (method + standing) are in `figures/README.md`.

INPUTS (all committed):
  results/exports/heldout/runs.csv         one row per run (means recomputed here)
  results/analysis/heldout/contrasts.csv   frozen paired deltas + bootstrap CIs (F2)

OUTPUTS (results/analysis/heldout/figures/, PDF canonical, PNG preview):
  F1_satisfaction_by_condition_band_64k    bars: satisfaction x architecture x band, cap 64k
  F2_registered_contrasts_forest_64k       forest of the five registered contrasts
  F3_satisfaction_vs_budget_bands          lines: satisfaction vs cap, one panel per band
  F4_efficiency_satisfaction_vs_tokens     scatter: satisfaction vs tokens, one panel per cap
  F5_validity_and_termination_diagnostics  lines: valid-non-empty and non-voluntary stop
  F6_block_b_satisfaction_vs_budget        lines: satisfaction vs cap, one panel per small n

STANDING. F1, F2 are the descriptive frame around the confirmatory family at the
primary cap 64000; F3-F6 are descriptive, and the budget pattern they show is
exploratory and was not registered. Nothing here is a confirmatory claim. See
`results/analysis/heldout/INTERPRETATION.md`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = _REPO_ROOT / "results/exports/heldout/runs.csv"
CONTRASTS_CSV = _REPO_ROOT / "results/analysis/heldout/contrasts.csv"
# Two destinations, because the two figure families answer different questions.
# F3-F6 describe all four budgets and all five architectures and belong to the
# full-matrix package; F1, F2 and F7 are specific to cap 64000 or to the C1-C3
# pair and stay with the frozen confirmatory analysis as its appendix, so that
# neither that cap nor that pair appears to hold a special place in the
# descriptive results (THESIS_DECISIONS section 5, amendment 2026-09-02).
MAIN_DIR = _REPO_ROOT / "results/analysis/heldout_full_matrix/figures"
APPENDIX_DIR = _REPO_ROOT / "results/analysis/heldout/figures"
APPENDIX_STEMS = ("F1_", "F2_", "F7_")

PRIMARY_CAP = 64000
CAPS = (16000, 32000, 64000, 128000)
CAP_LABEL = {16000: "16k", 32000: "32k", 64000: "64k", 128000: "128k"}
BANDS = ("low", "medium", "high")
SMALL_N = (4, 5, 6)
BOOTSTRAP_RESAMPLES = 10000
SEED = 20260827  # same seed the frozen analyser uses for descriptive intervals

# Condition reading order, short labels, legend labels. Order is a fixed reading
# order, NOT a ranking (that caveat is the plan's, kept here too).
CONDS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
SHORT = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
         "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
LEGEND = {"c1_react": "C1  ReAct", "c2_verify_revise": "C2  verify/revise",
          "c3_mas": "C3  hierarchical MAS", "c4_planner_critic": "C4  planner+critic",
          "c5_best_of_3": "C5  best-of-3"}
# Okabe-Ito, colour-blind safe. C1 neutral grey (baseline), C3 vermillion (MAS).
COLOR = {"c1_react": "#4D4D4D", "c2_verify_revise": "#0072B2", "c3_mas": "#D55E00",
         "c4_planner_critic": "#009E73", "c5_best_of_3": "#CC79A7"}
MARKER = {"c1_react": "o", "c2_verify_revise": "s", "c3_mas": "D",
          "c4_planner_critic": "^", "c5_best_of_3": "v"}

# Five registered contrasts, verbatim from amendment B2 (for F2).
CONFIRMATORY = ("C1->C2", "C2->C4", "C4->C3", "C1->C5", "C5->C3")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_runs() -> list[dict[str, str]]:
    with RUNS_CSV.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def satisfaction_block_a(rows: list[dict[str, str]]
                         ) -> dict[tuple[str, int, str], np.ndarray]:
    """(condition, cap, band) -> Block A satisfaction values, one per instance."""
    acc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["block"] == "A":
            acc[(r["condition"], int(r["cap"]), r["band"])].append(
                float(r["satisfaction"]))
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def satisfaction_block_b(rows: list[dict[str, str]]
                         ) -> dict[tuple[str, int, int], np.ndarray]:
    """(condition, cap, n_people) -> Block B satisfaction values, one per instance."""
    acc: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for r in rows:
        if r["block"] == "B":
            acc[(r["condition"], int(r["cap"]), int(r["n_people"]))].append(
                float(r["satisfaction"]))
    return {k: np.asarray(v, dtype=float) for k, v in acc.items()}


def block_a_cell_stats(rows: list[dict[str, str]]
                       ) -> dict[tuple[str, int], dict[str, float]]:
    """(condition, cap) over Block A (150 runs): mean satisfaction, tokens, valid-non-
    empty rate, and the share of runs that did NOT end in a voluntary agent_finish."""
    acc: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["block"] == "A":
            acc[(r["condition"], int(r["cap"]))].append(r)
    out: dict[tuple[str, int], dict[str, float]] = {}
    rng = np.random.default_rng(SEED)
    for key, group in acc.items():
        sat = np.asarray([float(r["satisfaction"]) for r in group])
        _, lo, hi = boot_mean_ci(sat, rng)
        out[key] = {
            "mean_sat": float(sat.mean()), "sat_lo": lo, "sat_hi": hi,
            "mean_tokens": float(np.mean([int(r["tokens_total"]) for r in group])),
            "vne": float(np.mean([1.0 if (r["valid"] == "1" and int(r["n_meetings"]) > 0)
                                  else 0.0 for r in group])),
            "nonvoluntary": float(np.mean([0.0 if r["termination"] == "agent_finish"
                                           else 1.0 for r in group])),
        }
    return out


def boot_mean_ci(values: np.ndarray, rng: np.random.Generator, *,
                 resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05
                 ) -> tuple[float, float, float]:
    """Descriptive percentile bootstrap of the mean. Returns (mean, ci_low, ci_high)."""
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    return (mean, float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def load_confirmatory_deltas() -> dict[tuple[str, str], dict[str, float]]:
    """(contrast, band) -> {mean, lo, hi} at the primary cap, from frozen contrasts.csv."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with CONTRASTS_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["block"] == "A" and int(r["cap"]) == PRIMARY_CAP
                    and r["in_confirmatory_family"] == "True"):
                band = r["stratum"].split("=", 1)[1]
                out[(r["contrast"], band)] = {
                    "mean": float(r["mean_delta"]),
                    "lo": float(r["ci_low"]), "hi": float(r["ci_high"])}
    return out


def load_c1c3_expose_delta() -> dict[tuple[int, str], dict[str, float]]:
    """(cap, band) -> {mean, lo, hi} for the EXPOSE contrast Delta = S_C1 - S_C3.

    The frozen `contrasts.csv` stores C1->C3 in the analyser's later-minus-earlier
    convention (S_C3 - S_C1). The expose defines Delta = S_SA - S_MAS = S_C1 - S_C3, so
    every value is the exact negation and the interval bounds swap. No new statistic:
    the descriptive bootstrap interval is the frozen one, re-signed.
    """
    out: dict[tuple[int, str], dict[str, float]] = {}
    with CONTRASTS_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["block"] == "A" and r["contrast"] == "C1->C3":
                band = r["stratum"].split("=", 1)[1]
                out[(int(r["cap"]), band)] = {
                    "mean": -float(r["mean_delta"]),
                    "lo": -float(r["ci_high"]), "hi": -float(r["ci_low"])}
    return out


# --------------------------------------------------------------------------- #
# Style / IO
# --------------------------------------------------------------------------- #
def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,          # embed TrueType so Overleaf/reviewers keep the text
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 10,        # used only for PANEL labels, never a figure title
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 8.5,
    })


def save(fig: plt.Figure, stem: str) -> None:
    out_dir = APPENDIX_DIR if stem.startswith(APPENDIX_STEMS) else MAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # Omit the embedded creation date so the PDF bytes are deterministic and the
        # figures byte-compare across machines like the other analysis artifacts.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata=meta)
    plt.close(fig)


def condition_legend_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=COLOR[c], marker=MARKER[c], linestyle="-",
                   ms=5, label=LEGEND[c]) for c in CONDS]


# --------------------------------------------------------------------------- #
# F1 — satisfaction by condition x band at 64k, with descriptive CIs
# --------------------------------------------------------------------------- #
def figure_f1(sat_a: dict[tuple[str, int, str], np.ndarray]) -> None:
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    band_x = np.arange(len(BANDS))
    width = 0.16
    for i, cond in enumerate(CONDS):
        means, lo_err, hi_err = [], [], []
        for band in BANDS:
            vals = sat_a[(cond, PRIMARY_CAP, band)]
            assert vals.size == 50, (cond, band, vals.size)
            m, lo, hi = boot_mean_ci(vals, rng)
            means.append(m); lo_err.append(m - lo); hi_err.append(hi - m)
        xs = band_x + (i - (len(CONDS) - 1) / 2) * width
        ax.bar(xs, means, width, color=COLOR[cond], edgecolor="black",
               linewidth=0.4, label=LEGEND[cond])
        ax.errorbar(xs, means, yerr=[lo_err, hi_err], fmt="none", ecolor="#222222",
                    elinewidth=0.8, capsize=2.0)
    ax.set_xticks(band_x)
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_xlabel("complexity band (conflict-graph density)")
    ax.set_ylabel("mean satisfaction")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=2, loc="upper right")
    save(fig, "F1_satisfaction_by_condition_band_64k")


# --------------------------------------------------------------------------- #
# F2 — forest of the five registered contrasts, delta_low and delta_high at 64k
# --------------------------------------------------------------------------- #
def figure_f2(deltas: dict[tuple[str, str], dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    off = 0.17
    c_low, c_high = "#0072B2", "#D55E00"
    y_of = {c: len(CONFIRMATORY) - 1 - i for i, c in enumerate(CONFIRMATORY)}
    for c in CONFIRMATORY:
        y = y_of[c]
        for band, dy, colour, mark in (("low", +off, c_low, "o"),
                                       ("high", -off, c_high, "s")):
            d = deltas[(c, band)]
            ax.errorbar(d["mean"], y + dy,
                        xerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                        fmt=mark, color=colour, ecolor=colour, ms=5.5,
                        elinewidth=1.4, capsize=3.0, zorder=3)
    ax.axvline(0.0, color="#333333", linewidth=1.0, zorder=1)  # zero reference line
    ax.set_yticks([y_of[c] for c in CONFIRMATORY])
    ax.set_yticklabels(CONFIRMATORY)
    ax.set_ylim(-0.6, len(CONFIRMATORY) - 0.4)
    ax.set_xlabel(r"paired mean $\delta$ = satisfaction(later) $-$ satisfaction(earlier),  cap 64k")
    ax.set_xlim(-0.75, 0.85)
    handles = [Line2D([0], [0], marker="o", color=c_low, linestyle="", ms=6,
                      label=r"$\delta_{\mathrm{low}}$  (registered: $<0$)"),
               Line2D([0], [0], marker="s", color=c_high, linestyle="", ms=6,
                      label=r"$\delta_{\mathrm{high}}$  (registered: $>0$)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.015, 0.60))
    # Direction cue: reading the sign of a difference needs it, and the axis label alone
    # is easy to misread. Kept minimal.
    ax.text(-0.74, len(CONFIRMATORY) - 0.35, "← earlier ahead", fontsize=8,
            color="#888888", va="center")
    ax.text(0.84, len(CONFIRMATORY) - 0.35, "later ahead →", fontsize=8,
            color="#888888", va="center", ha="right")
    save(fig, "F2_registered_contrasts_forest_64k")


# --------------------------------------------------------------------------- #
# F3 — satisfaction vs budget cap, one panel per band
# --------------------------------------------------------------------------- #
def _lines_vs_cap(ax, cap_to_vals, rng) -> None:
    """Plot one line per condition over the caps, with bootstrap error bars."""
    x = np.array(CAPS, dtype=float)
    for cond in CONDS:
        means, lo_err, hi_err = [], [], []
        for cap in CAPS:
            m, lo, hi = boot_mean_ci(cap_to_vals[(cond, cap)], rng)
            means.append(m); lo_err.append(m - lo); hi_err.append(hi - m)
        ax.errorbar(x, means, yerr=[lo_err, hi_err], color=COLOR[cond],
                    marker=MARKER[cond], ms=4.5, linewidth=1.6, elinewidth=0.7,
                    capsize=1.8, label=LEGEND[cond])
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(CAPS))
    ax.set_xticklabels([CAP_LABEL[c] for c in CAPS])
    ax.set_xlim(14000, 150000)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("token budget cap")


def figure_f3(sat_a: dict[tuple[str, int, str], np.ndarray]) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    for ax, band in zip(axes, BANDS):
        _lines_vs_cap(ax, {(c, cap): sat_a[(c, cap, band)]
                           for c in CONDS for cap in CAPS}, rng)
        ax.set_title(f"{band} complexity")          # panel label
    axes[0].set_ylabel("mean satisfaction")
    axes[0].legend(loc="upper left", fontsize=8, borderpad=0.5)
    save(fig, "F3_satisfaction_vs_budget_bands")


# --------------------------------------------------------------------------- #
# F4 — efficiency: satisfaction vs tokens actually spent, one panel per cap
# --------------------------------------------------------------------------- #
def figure_f4(stats: dict[tuple[str, int], dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, len(CAPS), figsize=(10.0, 2.9))
    for ax, cap in zip(axes, CAPS):
        for cond in CONDS:
            s = stats[(cond, cap)]
            ax.errorbar(s["mean_tokens"] / 1000.0, s["mean_sat"],
                        yerr=[[s["mean_sat"] - s["sat_lo"]],
                              [s["sat_hi"] - s["mean_sat"]]],
                        marker=MARKER[cond], color=COLOR[cond], ms=6,
                        elinewidth=0.7, capsize=1.8, label=LEGEND[cond])
        ax.set_title(f"cap {CAP_LABEL[cap]}")        # panel label
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("tokens/run (k)")
        ax.margins(x=0.18)
    axes[0].set_ylabel("mean satisfaction")
    axes[0].legend(loc="upper left", fontsize=7.5, borderpad=0.4)
    save(fig, "F4_efficiency_satisfaction_vs_tokens")


# --------------------------------------------------------------------------- #
# F5 — validity and termination diagnostics (Block A, n=8)
# --------------------------------------------------------------------------- #
def figure_f5(stats: dict[tuple[str, int], dict[str, float]]) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
    x = np.array(CAPS, dtype=float)
    for cond in CONDS:
        ax_a.plot(x, [stats[(cond, cap)]["vne"] for cap in CAPS], color=COLOR[cond],
                  marker=MARKER[cond], ms=4.5, linewidth=1.6, label=LEGEND[cond])
        ax_b.plot(x, [stats[(cond, cap)]["nonvoluntary"] for cap in CAPS],
                  color=COLOR[cond], marker=MARKER[cond], ms=4.5, linewidth=1.6)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(CAPS)); ax.set_xticklabels([CAP_LABEL[c] for c in CAPS])
        ax.set_xlim(14000, 150000); ax.set_ylim(0, 1.02)
        ax.set_xlabel("token budget cap")
    ax_a.set_ylabel("share of runs")
    ax_a.set_title("valid non-empty plan")                  # panel label
    ax_b.set_title("stopped by budget or abort")            # panel label
    ax_a.legend(loc="upper left", fontsize=7.5, borderpad=0.4)
    save(fig, "F5_validity_and_termination_diagnostics")


# --------------------------------------------------------------------------- #
# F6 — Block B (small n) satisfaction vs budget, one panel per size (appendix)
# --------------------------------------------------------------------------- #
def figure_f6(sat_b: dict[tuple[str, int, int], np.ndarray]) -> None:
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(SMALL_N), figsize=(9.6, 3.4), sharey=True)
    for ax, n in zip(axes, SMALL_N):
        _lines_vs_cap(ax, {(c, cap): sat_b[(c, cap, n)]
                           for c in CONDS for cap in CAPS}, rng)
        ax.set_title(f"n = {n}")                     # panel label
    axes[0].set_ylabel("mean satisfaction")
    # Figure-level legend, below the panels and outside every plotting area. At n = 4
    # every corner of the first panel is occupied -- C3 and C4 across the top, C5's rise
    # and the C1/C2 intervals across the bottom right -- so no in-axes position is free,
    # and `lower right` was itself hiding C5 between 32k and 64k.
    fig.legend(handles=condition_legend_handles(), loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8)
    save(fig, "F6_block_b_satisfaction_vs_budget")


# --------------------------------------------------------------------------- #
# F7 — the expose crossover view: Delta = S_C1 - S_C3 vs complexity, one line per cap
# --------------------------------------------------------------------------- #
def figure_f7(delta: dict[tuple[int, str], dict[str, float]]) -> None:
    """The central expose question in one figure. Delta > 0 single agent ahead,
    Delta < 0 MAS ahead; the hypothesised crossover is Delta going + at Low to - at
    High WITHIN a cap. One line per budget cap shows this never happens; the caps
    separate vertically, which is the budget-dependent transition."""
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    band_x = np.arange(len(BANDS))
    cap_color = {16000: "#BDD7E7", 32000: "#6BAED6", 64000: "#3182BD", 128000: "#08519C"}
    cap_marker = {16000: "o", 32000: "s", 64000: "^", 128000: "D"}
    dodge = {16000: -0.06, 32000: -0.02, 64000: 0.02, 128000: 0.06}
    for cap in CAPS:
        means = [delta[(cap, b)]["mean"] for b in BANDS]
        lo = [delta[(cap, b)]["mean"] - delta[(cap, b)]["lo"] for b in BANDS]
        hi = [delta[(cap, b)]["hi"] - delta[(cap, b)]["mean"] for b in BANDS]
        ax.errorbar(band_x + dodge[cap], means, yerr=[lo, hi], color=cap_color[cap],
                    marker=cap_marker[cap], ms=6, linewidth=1.8, elinewidth=1.0,
                    capsize=2.5, label=f"cap {CAP_LABEL[cap]}")
    ax.axhline(0.0, color="#333333", linewidth=1.0, zorder=1)  # zero reference line
    ax.set_xticks(band_x)
    ax.set_xticklabels(["low", "medium", "high"])
    ax.set_xlabel("complexity band (conflict-graph density)")
    ax.set_ylabel(r"$\Delta = S_{C1} - S_{C3}$")
    ax.set_xlim(-0.4, len(BANDS) - 0.6)
    # Sign meaning is essential to read the figure.
    ax.text(-0.36, ax.get_ylim()[1] * 0.9, "single agent ahead (Δ > 0)", fontsize=8,
            color="#666666", va="top")
    ax.text(-0.36, ax.get_ylim()[0] * 0.9, "MAS ahead (Δ < 0)", fontsize=8,
            color="#666666", va="bottom")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              title="equal token budget", frameon=False)
    save(fig, "F7_c1_vs_c3_crossover_delta")


def main() -> int:
    set_style()
    rows = load_runs()
    sat_a = satisfaction_block_a(rows)
    sat_b = satisfaction_block_b(rows)
    stats = block_a_cell_stats(rows)
    figure_f1(sat_a)
    figure_f2(load_confirmatory_deltas())
    figure_f3(sat_a)
    figure_f4(stats)
    figure_f5(stats)
    figure_f6(sat_b)
    figure_f7(load_c1c3_expose_delta())
    print(f"wrote F3-F6 to {MAIN_DIR.relative_to(_REPO_ROOT)}")
    print(f"wrote F1, F2, F7 (frozen-analysis appendix) to "
          f"{APPENDIX_DIR.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
