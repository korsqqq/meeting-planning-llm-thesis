# scratchpad/analysis.py
"""Supervisor-style analysis of the held-out matrix (198 instances x 5 conditions x 4 caps).

Reads only committed artefacts (runs.csv, contrasts.csv, summary.json, candidates.csv),
computes descriptive statistics, a small set of exploratory tests, and renders figures.
Writes figures (PNG) and a numbers.json into OUT_DIR.

Nothing here is a confirmatory claim; the frozen analysis is reproduced, not replaced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report_out")
FIG = OUT / "figs"
FIG.mkdir(parents=True, exist_ok=True)

RUNS = REPO / "results/exports/heldout/runs.csv"
CONTRASTS = REPO / "results/analysis/heldout/contrasts.csv"
SUMMARY = REPO / "results/analysis/heldout/summary.json"
CAND = {
    8: REPO / "results/calibration/heldout_n8/candidates.csv",
    4: REPO / "results/calibration/heldout_n4/candidates.csv",
    5: REPO / "results/calibration/heldout_n5/candidates.csv",
    6: REPO / "results/calibration/heldout_n6/candidates.csv",
}

CONDS = ["c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3"]
SHORT = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
         "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
LEGEND = {"c1_react": "C1 ReAct", "c2_verify_revise": "C2 verify/revise",
          "c3_mas": "C3 hierarchical MAS", "c4_planner_critic": "C4 planner+critic",
          "c5_best_of_3": "C5 best-of-3"}
COLOR = {"c1_react": "#4D4D4D", "c2_verify_revise": "#0072B2", "c3_mas": "#D55E00",
         "c4_planner_critic": "#009E73", "c5_best_of_3": "#CC79A7"}
MARK = {"c1_react": "o", "c2_verify_revise": "s", "c3_mas": "^", "c4_planner_critic": "D",
        "c5_best_of_3": "v"}
CAPS = [16000, 32000, 64000, 128000]
CAPL = {16000: "16k", 32000: "32k", 64000: "64k", 128000: "128k"}
BANDS = ["low", "medium", "high"]
BANDL = {"low": "Low D", "medium": "Medium D", "high": "High D"}
SIZES = [4, 5, 6]
B = 10000
SEED = 20260827
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "legend.fontsize": 8.5, "figure.dpi": 130,
    "axes.spines.top": False, "axes.spines.right": False,
})

NUM: dict = {}


def boot_ci(x: np.ndarray, b: int = B) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


# ----------------------------------------------------------------------------- load
df = pd.read_csv(RUNS)
det = df.details_json.apply(json.loads)
df["latency"] = det.apply(lambda d: d["latency_seconds"])
df["thinking"] = det.apply(lambda d: d["tokens"]["thinking_tokens"])
df["input"] = det.apply(lambda d: d["tokens"]["input_tokens"])
df["answer"] = det.apply(lambda d: d["tokens"]["answer_tokens"])
df["n_proposals"] = det.apply(lambda d: d["n_proposals"])
df["n_valid_proposals"] = det.apply(lambda d: d["n_valid_proposals"])
df["tightness"] = det.apply(lambda d: d["tightness"])
df["overlap"] = det.apply(lambda d: d["overlap"])
df["travel"] = det.apply(lambda d: d["travel_structure"])
df["optimality"] = det.apply(lambda d: bool(d["score"]["optimality"])).astype(int)
df["cap_util"] = det.apply(lambda d: d["cap_utilisation"])
df["vne"] = ((df.valid == 1) & (df.n_meetings > 0)).astype(int)
df["achieved"] = df.n_meetings
df["band"] = df.band.fillna("")

cand = pd.concat([pd.read_csv(p).assign(n_pool=n) for n, p in CAND.items()])
feat_cols = ["instance_id", "D", "higher_order_gap_H", "alpha_reachable", "max_degree",
             "edge_share_top1", "largest_component", "n_max_independent_sets",
             "triangle_violations", "individually_reachable_count"]
df = df.merge(cand[feat_cols], on="instance_id", how="left")
assert df.higher_order_gap_H.notna().all(), "H join failed"
df["H"] = df.higher_order_gap_H.astype(int)

A = df[df.block == "A"].copy()
Bb = df[df.block == "B"].copy()
NUM["n_runs"] = int(len(df))
NUM["n_instances"] = int(df.instance_id.nunique())

# ----------------------------------------------------------------------------- cell means
rows = []
for c in CONDS:
    for cap in CAPS:
        for band in BANDS:
            x = A[(A.condition == c) & (A.cap == cap) & (A.band == band)].satisfaction.values
            lo, hi = boot_ci(x)
            sub = A[(A.condition == c) & (A.cap == cap) & (A.band == band)]
            rows.append(dict(cond=SHORT[c], cap=cap, band=band, n=len(x), mean=x.mean(),
                             sd=x.std(ddof=1), ci_lo=lo, ci_hi=hi,
                             vne=sub.vne.mean(), opt=sub.optimality.mean(),
                             tokens=sub.tokens_total.mean(), calls=sub.n_calls.mean(),
                             latency=sub.latency.mean(), achieved=sub.achieved.mean()))
        x = A[(A.condition == c) & (A.cap == cap)].satisfaction.values
        lo, hi = boot_ci(x)
        sub = A[(A.condition == c) & (A.cap == cap)]
        rows.append(dict(cond=SHORT[c], cap=cap, band="all", n=len(x), mean=x.mean(),
                         sd=x.std(ddof=1), ci_lo=lo, ci_hi=hi, vne=sub.vne.mean(),
                         opt=sub.optimality.mean(), tokens=sub.tokens_total.mean(),
                         calls=sub.n_calls.mean(), latency=sub.latency.mean(),
                         achieved=sub.achieved.mean()))
cells = pd.DataFrame(rows)
cells["sat_per_1k"] = cells["mean"] / cells["tokens"] * 1000
cells.to_csv(OUT / "cells_blockA.csv", index=False)
NUM["cells"] = cells.round(4).to_dict(orient="records")

# Block B cells
rowsB = []
for c in CONDS:
    for cap in CAPS:
        for n in SIZES:
            sub = Bb[(Bb.condition == c) & (Bb.cap == cap) & (Bb.n_people == n)]
            x = sub.satisfaction.values
            lo, hi = boot_ci(x)
            rowsB.append(dict(cond=SHORT[c], cap=cap, n_people=n, n=len(x), mean=x.mean(),
                              ci_lo=lo, ci_hi=hi, vne=sub.vne.mean(), opt=sub.optimality.mean(),
                              tokens=sub.tokens_total.mean(), calls=sub.n_calls.mean()))
cellsB = pd.DataFrame(rowsB)
cellsB["sat_per_1k"] = cellsB["mean"] / cellsB["tokens"] * 1000
cellsB.to_csv(OUT / "cells_blockB.csv", index=False)
NUM["cellsB"] = cellsB.round(4).to_dict(orient="records")


def cell(c, cap, band="all", col="mean"):
    return float(cells[(cells.cond == c) & (cells.cap == cap) & (cells.band == band)][col].iloc[0])


# ----------------------------------------------------------------------------- FIG 1 surface heatmaps
fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.3), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for ax, c in zip(axes, CONDS):
    M = np.array([[cell(SHORT[c], cap, band) for cap in CAPS] for band in BANDS])
    im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(3):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < 0.55 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS])
    ax.set_yticks(range(3)); ax.set_yticklabels(["Low", "Med", "High"] if c == CONDS[0] else [])
    ax.set_title(LEGEND[c], fontsize=9.5, color=COLOR[c])
    ax.set_xlabel("budget cap")
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel("conflict-density band")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
cb.set_label("mean satisfaction")
fig.savefig(FIG / "fig01_surface_heatmaps.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 2 budget-response per band
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True, constrained_layout=True)
x = np.arange(4)
for ax, band in zip(axes, BANDS + ["all"]):
    for k, c in enumerate(CONDS):
        sub = cells[(cells.cond == SHORT[c]) & (cells.band == band)].sort_values("cap")
        off = (k - 2) * 0.06
        ax.errorbar(x + off, sub["mean"], yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]],
                    color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.6, capsize=2, label=LEGEND[c])
    ax.set_xticks(x); ax.set_xticklabels([CAPL[c_] for c_ in CAPS])
    ax.set_title(BANDL.get(band, "All bands pooled (n = 150)"))
    ax.set_ylim(-0.03, 1.03); ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("budget cap (log2 spacing)")
axes[0].set_ylabel("mean satisfaction (95% bootstrap CI)")
axes[-1].legend(loc="upper left", frameon=False)
fig.savefig(FIG / "fig02_budget_response_bands.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 3 band effect at fixed cap (transposed)
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), sharey=True, constrained_layout=True)
for ax, cap in zip(axes, CAPS):
    for c in CONDS:
        sub = cells[(cells.cond == SHORT[c]) & (cells.cap == cap) & (cells.band != "all")]
        sub = sub.set_index("band").loc[BANDS]
        ax.errorbar(range(3), sub["mean"], yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]],
                    color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.6, capsize=2, label=LEGEND[c])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_title(f"cap {CAPL[cap]}"); ax.set_ylim(-0.03, 1.03); ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("pairwise conflict density band")
axes[0].set_ylabel("mean satisfaction")
axes[0].legend(frameon=False, loc="upper left")
fig.savefig(FIG / "fig03_band_effect_per_cap.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- marginal gains per doubling (paired)
gain_rows = []
for c in CONDS:
    for band in BANDS + ["all"]:
        sub = A[A.condition == c] if band == "all" else A[(A.condition == c) & (A.band == band)]
        piv = sub.pivot(index="instance_id", columns="cap", values="satisfaction")
        tok = sub.pivot(index="instance_id", columns="cap", values="tokens_total")
        for lo_cap, hi_cap in zip(CAPS[:-1], CAPS[1:]):
            d = (piv[hi_cap] - piv[lo_cap]).values
            lo, hi = boot_ci(d)
            changed = float((tok[hi_cap] != tok[lo_cap]).mean())
            gain_rows.append(dict(cond=SHORT[c], band=band, step=f"{CAPL[lo_cap]}->{CAPL[hi_cap]}",
                                  gain=d.mean(), ci_lo=lo, ci_hi=hi, improved=int((d > 0).sum()),
                                  worsened=int((d < 0).sum()), unchanged=int((d == 0).sum()),
                                  exec_changed=changed))
gains = pd.DataFrame(gain_rows)
gains.to_csv(OUT / "gains_per_doubling.csv", index=False)
NUM["gains"] = gains.round(4).to_dict(orient="records")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True, constrained_layout=True)
steps = ["16k->32k", "32k->64k", "64k->128k"]
for ax, band in zip(axes, BANDS + ["all"]):
    for k, c in enumerate(CONDS):
        sub = gains[(gains.cond == SHORT[c]) & (gains.band == band)].set_index("step").loc[steps]
        xs = np.arange(3) + (k - 2) * 0.15
        ax.bar(xs, sub.gain, width=0.14, color=COLOR[c], label=LEGEND[c])
        ax.errorbar(xs, sub.gain, yerr=[sub.gain - sub.ci_lo, sub.ci_hi - sub.gain], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(steps)
    ax.set_title(BANDL.get(band, "All bands pooled")); ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("budget doubling")
axes[0].set_ylabel("paired gain in mean satisfaction")
axes[-1].legend(frameon=False, loc="upper left")
fig.savefig(FIG / "fig04_marginal_gain_per_doubling.png", bbox_inches="tight")
plt.close(fig)

# activation budget and plateau
act_rows = []
for c in CONDS:
    for band in BANDS + ["all"]:
        sub = cells[(cells.cond == SHORT[c]) & (cells.band == band)].sort_values("cap")
        act = next((int(r.cap) for r in sub.itertuples() if r.vne >= 0.5), None)
        top = sub[sub.cap == 128000]["mean"].iloc[0]
        plateau = next((int(r.cap) for r in sub.itertuples() if top > 0 and r.mean >= 0.9 * top), None)
        act_rows.append(dict(cond=SHORT[c], band=band, activation_cap=act, plateau_cap=plateau,
                             sat_128k=top))
activation = pd.DataFrame(act_rows)
NUM["activation"] = activation.to_dict(orient="records")

# ----------------------------------------------------------------------------- FIG 5 cost-quality trajectories + utilisation
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), constrained_layout=True)
ax = axes[0]
for c in CONDS:
    sub = cells[(cells.cond == SHORT[c]) & (cells.band == "all")].sort_values("cap")
    ax.plot(sub.tokens / 1000, sub["mean"], color=COLOR[c], marker=MARK[c], ms=5, lw=1.6, label=LEGEND[c])
    for r in sub.itertuples():
        ax.annotate(CAPL[r.cap], (r.tokens / 1000, r.mean), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=COLOR[c])
ax.set_xlabel("mean realised tokens per run (thousands)"); ax.set_ylabel("mean satisfaction (Block A)")
ax.set_title("Cost-quality trajectory (points = caps)"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
for c in CONDS:
    sub = cells[(cells.cond == SHORT[c]) & (cells.band == "all")].sort_values("cap")
    ax.plot(range(4), sub.tokens / np.array(CAPS), color=COLOR[c], marker=MARK[c], ms=5, lw=1.6)
ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS]); ax.set_ylim(0, 1.05)
ax.set_ylabel("mean cap utilisation (tokens / cap)"); ax.set_xlabel("budget cap"); ax.grid(alpha=0.25)
ax.set_title("How much of the cap is actually spent")
ax = axes[2]
for c in CONDS:
    sub = cells[(cells.cond == SHORT[c]) & (cells.band == "all")].sort_values("cap")
    ax.plot(range(4), sub.sat_per_1k * 1000, color=COLOR[c], marker=MARK[c], ms=5, lw=1.6)
ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS])
ax.set_ylabel("satisfaction per 1M realised tokens"); ax.set_xlabel("budget cap"); ax.grid(alpha=0.25)
ax.set_title("Efficiency (ratio of means)")
fig.savefig(FIG / "fig05_cost_quality.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 6 instance x architecture matrix
inst = (A[A.condition == "c1_react"][["instance_id", "band", "conflict_pairs", "H", "optimum", "travel"]]
        .drop_duplicates().copy())
inst["band_rank"] = inst.band.map({"low": 0, "medium": 1, "high": 2})
inst = inst.sort_values(["band_rank", "conflict_pairs", "H", "instance_id"]).reset_index(drop=True)
order = inst.instance_id.tolist()
fig, axes = plt.subplots(1, 4, figsize=(13.5, 7.2), constrained_layout=True, sharey=True)
cmap6 = mcolors.ListedColormap(["#f0f0f0", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"])
bounds = [-0.01, 0.01, 0.30, 0.40, 0.60, 0.70, 0.80, 1.01]  # 0 | .25 | .33 | .5 | .67 | .75 | 1
norm6 = mcolors.BoundaryNorm(bounds, cmap6.N)
for ax, cap in zip(axes, CAPS):
    piv = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="satisfaction")
    M = piv.loc[order, CONDS].values
    ax.imshow(M, cmap=cmap6, norm=norm6, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[c] for c in CONDS])
    ax.set_title(f"cap {CAPL[cap]}")
    for b in (50, 100):
        ax.axhline(b - 0.5, color="black", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_yticks([25, 75, 125]); axes[0].set_yticklabels(["Low D\n(k=0-1)", "Medium D\n(k=7)", "High D\n(k=13-16)"])
axes[0].set_ylabel("150 held-out instances, n = 8 (sorted by band, conflict pairs, H)")
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap6(i)) for i in range(7)]
fig.legend(handles, ["0 (no valid non-empty plan)", "0.25", "0.33", "0.50", "0.67", "0.75", "1.0 (optimal)"],
           loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.savefig(FIG / "fig06_instance_matrix.png", bbox_inches="tight")
plt.close(fig)

# solved-by-how-many distribution
sb_rows = []
for cap in CAPS:
    piv = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="vne")
    k = piv[CONDS].sum(axis=1)
    counts = k.value_counts().reindex(range(6), fill_value=0)
    sb_rows.append(dict(cap=cap, **{f"solved_by_{i}": int(counts[i]) for i in range(6)}))
    piv_s = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="satisfaction")
    best = piv_s[CONDS].max(axis=1)
    uniq = {}
    for c in CONDS:
        is_best = (piv_s[c] == best) & (best > 0)
        uniq[SHORT[c] + "_best_or_tied"] = int(is_best.sum())
        others = piv_s[[o for o in CONDS if o != c]].max(axis=1)
        uniq[SHORT[c] + "_strictly_best"] = int(((piv_s[c] > others) & (best > 0)).sum())
    sb_rows[-1].update(uniq)
    sb_rows[-1]["oracle_of_5_mean"] = float(best.mean())
solved_by = pd.DataFrame(sb_rows)
NUM["solved_by"] = solved_by.to_dict(orient="records")

# ----------------------------------------------------------------------------- FIG 7 task space: D x H, structure
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
for ax, cap in zip(axes[:2], [64000, 128000]):
    piv = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="satisfaction")
    d = (piv["c3_mas"] - piv["c1_react"]).rename("d31").reset_index().merge(inst, on="instance_id")
    jx = rng.uniform(-0.35, 0.35, len(d)); jy = rng.uniform(-0.25, 0.25, len(d))
    sc = ax.scatter(d.conflict_pairs + jx, d.H + jy, c=d.d31, cmap="RdBu_r", vmin=-1, vmax=1,
                    s=28, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("binding conflict pairs k (D = k/28)"); ax.set_ylabel("higher-order gap H = alpha - O")
    ax.set_title(f"C3 - C1 per instance, cap {CAPL[cap]}")
    ax.grid(alpha=0.2)
cb = fig.colorbar(sc, ax=axes[:2], shrink=0.85, pad=0.01); cb.set_label("satisfaction(C3) - satisfaction(C1)")
ax = axes[2]
# mean satisfaction by H at 128k and 64k per condition (pooled bands)
for c in CONDS:
    sub = A[(A.condition == c) & (A.cap == 128000) & (A.H >= 0)].groupby("H").satisfaction.mean()
    ax.plot(sub.index, sub.values, color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.5, label=LEGEND[c])
    sub = A[(A.condition == c) & (A.cap == 64000) & (A.H >= 0)].groupby("H").satisfaction.mean()
    ax.plot(sub.index, sub.values, color=COLOR[c], marker=MARK[c], ms=3.5, lw=1.0, ls="--", alpha=0.6)
hcounts = A[(A.condition == "c1_react") & (A.cap == 64000) & (A.H >= 0)].H.value_counts().sort_index()
ax.set_xticks(hcounts.index); ax.set_xticklabels([f"{h}\n(n={n_})" for h, n_ in hcounts.items()], fontsize=8)
ax.set_xlabel("higher-order gap H (solid 128k, dashed 64k); H = -1 instance omitted")
ax.set_ylabel("mean satisfaction")
ax.set_title("Satisfaction against H, bands pooled"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=7.5)
fig.savefig(FIG / "fig07_task_space.png", bbox_inches="tight")
plt.close(fig)

# feature associations (descriptive)
assoc = []
for cap in [64000, 128000]:
    for c in CONDS:
        sub = A[(A.condition == c) & (A.cap == cap)]
        for f in ["conflict_pairs", "H", "tightness", "overlap", "optimum", "n_max_independent_sets", "max_degree"]:
            r, p = stats.spearmanr(sub[f], sub.satisfaction)
            assoc.append(dict(cap=cap, cond=SHORT[c], feature=f, spearman=r, p=p))
assoc = pd.DataFrame(assoc)
assoc.to_csv(OUT / "feature_assoc.csv", index=False)
NUM["assoc"] = assoc.round(4).to_dict(orient="records")

travel_tab = (A[A.cap.isin([64000, 128000])].groupby(["cap", "travel", "condition"]).satisfaction.mean()
              .unstack("condition")[CONDS].rename(columns=SHORT).round(3))
travel_n = A[(A.cap == 64000) & (A.condition == "c1_react")].travel.value_counts()
NUM["travel_tab"] = {f"{cap}|{tr}": row.to_dict() for (cap, tr), row in travel_tab.iterrows()}
NUM["travel_n"] = travel_n.to_dict()
NUM["travel_by_band"] = (A[(A.cap == 64000) & (A.condition == "c1_react")]
                         .groupby(["band", "travel"]).size().unstack(fill_value=0).to_dict())

# H x band stratified means at 64k/128k
hb = []
for cap in [64000, 128000]:
    for band in BANDS:
        for hs, mask in [("H=0", A.H == 0), ("H>=1", A.H >= 1)]:
            sub = A[(A.cap == cap) & (A.band == band) & mask]
            n = sub[sub.condition == "c1_react"].shape[0]
            if n == 0:
                continue
            row = dict(cap=cap, band=band, H=hs, n=n)
            for c in CONDS:
                row[SHORT[c]] = float(sub[sub.condition == c].satisfaction.mean())
            hb.append(row)
NUM["H_band"] = pd.DataFrame(hb).round(3).to_dict(orient="records")

# ----------------------------------------------------------------------------- FIG 8 ranks: Friedman + CD diagram per cap
fried = []
for cap in CAPS:
    for band in BANDS + ["all"]:
        sub = A[A.cap == cap] if band == "all" else A[(A.cap == cap) & (A.band == band)]
        piv = sub.pivot(index="instance_id", columns="condition", values="satisfaction")[CONDS]
        M = piv.values
        n, k = M.shape
        ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, M)  # 1 = best
        mean_rank = ranks.mean(axis=0)
        if np.allclose(M, M[0, 0]):
            chi2, p, W = np.nan, np.nan, 0.0
        else:
            chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
            W = chi2 / (n * (k - 1))
        cd = 2.728 * np.sqrt(k * (k + 1) / (6 * n))  # Nemenyi q_0.05, k=5
        fried.append(dict(cap=cap, band=band, n=n, chi2=chi2, p=p, kendall_W=W, CD=cd,
                          **{f"rank_{SHORT[c]}": float(mean_rank[j]) for j, c in enumerate(CONDS)}))
fried = pd.DataFrame(fried)
fried.to_csv(OUT / "friedman.csv", index=False)
NUM["friedman"] = fried.round(4).to_dict(orient="records")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.9), constrained_layout=True)
for ax, cap in zip(axes, CAPS):
    r = fried[(fried.cap == cap) & (fried.band == "all")].iloc[0]
    ranks = {SHORT[c]: r[f"rank_{SHORT[c]}"] for c in CONDS}
    cd = r.CD
    ax.set_xlim(5.2, 0.8); ax.set_ylim(-0.2, 3.4)
    ax.axhline(3.0, color="black", lw=1)
    for t in range(1, 6):
        ax.plot([t, t], [3.0, 3.12], color="black", lw=1); ax.text(t, 3.2, str(t), ha="center", fontsize=8)
    order_c = sorted(CONDS, key=lambda c: ranks[SHORT[c]])
    ys = [2.5, 2.1, 1.7, 1.3, 0.9]
    for y, c in zip(ys, order_c):
        rv = ranks[SHORT[c]]
        ax.plot([rv, rv], [3.0, y], color=COLOR[c], lw=1.2)
        ax.text(rv, y - 0.05, f"{SHORT[c]} ({rv:.2f})", ha="center", va="top", fontsize=8, color=COLOR[c])
    # CD groups: chain consecutive archs within CD
    vals = [ranks[SHORT[c]] for c in order_c]
    i = 0; level = 0.35
    groups = []
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            groups.append((vals[i], vals[j]))
        i = j + 1 if j > i else i + 1
    for gi, (a, b_) in enumerate(groups):
        ax.plot([a - 0.04, b_ + 0.04], [level - 0.12 * gi] * 2, color="black", lw=2.5, alpha=0.7)
    ax.plot([1.0, 1.0 + cd], [3.35, 3.35], color="grey", lw=2); ax.text(1.0 + cd / 2, 3.42, f"CD={cd:.2f}", ha="center", fontsize=7, color="grey")
    ax.set_title(f"cap {CAPL[cap]}  (W = {r.kendall_W:.2f})", fontsize=9.5)
    ax.axis("off")
fig.suptitle("Mean rank of the five architectures over 150 paired instances (1 = best); bars join architectures within the Nemenyi critical difference", fontsize=9)
fig.savefig(FIG / "fig08_rank_cd.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 9 pairwise matrix (from frozen contrasts.csv)
con = pd.read_csv(CONTRASTS)
conA = con[con.block == "A"].copy()
conA["band"] = conA.stratum.str.replace("band=", "")
pairs = ["C1->C2", "C1->C5", "C1->C4", "C1->C3", "C2->C5", "C2->C4", "C2->C3", "C5->C4", "C5->C3", "C4->C3"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True, sharey=True)
for ax, band in zip(axes, BANDS):
    M = np.zeros((10, 4)); S = np.empty((10, 4), dtype=object)
    for i, p in enumerate(pairs):
        for j, cap in enumerate(CAPS):
            r = conA[(conA.contrast == p) & (conA.band == band) & (conA.cap == cap)].iloc[0]
            M[i, j] = r.mean_delta
            S[i, j] = "*" if (r.ci_low > 0 or r.ci_high < 0) else ""
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    for i in range(10):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:+.2f}{S[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.4 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS])
    ax.set_yticks(range(10)); ax.set_yticklabels(pairs)
    ax.set_title(BANDL[band])
    for s in ax.spines.values():
        s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
cb.set_label("mean paired delta = sat(later) - sat(earlier);  * = 95% bootstrap CI excludes 0 (descriptive)")
fig.savefig(FIG / "fig09_pairwise_matrix.png", bbox_inches="tight")
plt.close(fig)

# count of CI-excluding-zero cells per pair
NUM["pair_ci_counts"] = {}
for p in pairs:
    sub = conA[conA.contrast == p]
    NUM["pair_ci_counts"][p] = dict(
        pos=int(((sub.ci_low > 0)).sum()), neg=int((sub.ci_high < 0).sum()), total=int(len(sub)))

# ----------------------------------------------------------------------------- FIG 10 recover / lose vs C1
tr_rows = []
for cap in CAPS:
    piv_v = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="vne")
    piv_s = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="satisfaction")
    for c in CONDS[1:]:
        tr_rows.append(dict(cap=cap, cond=SHORT[c],
                            recovered=int(((piv_v[c] == 1) & (piv_v["c1_react"] == 0)).sum()),
                            lost=int(((piv_v[c] == 0) & (piv_v["c1_react"] == 1)).sum()),
                            both=int(((piv_v[c] == 1) & (piv_v["c1_react"] == 1)).sum()),
                            neither=int(((piv_v[c] == 0) & (piv_v["c1_react"] == 0)).sum()),
                            sat_higher=int((piv_s[c] > piv_s["c1_react"]).sum()),
                            sat_lower=int((piv_s[c] < piv_s["c1_react"]).sum()),
                            sat_tied=int((piv_s[c] == piv_s["c1_react"]).sum())))
trans = pd.DataFrame(tr_rows)
NUM["transitions"] = trans.to_dict(orient="records")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, sharey=True)
for ax, cap in zip(axes, [64000, 128000]):
    sub = trans[trans.cap == cap]
    ys = np.arange(4)
    ax.barh(ys, sub.sat_higher, color="#2C7BB6", label="X ahead of C1 (satisfaction)")
    ax.barh(ys, -sub.sat_lower, color="#D7191C", label="X behind C1")
    for y, r in zip(ys, sub.itertuples()):
        ax.text(r.sat_higher + 1, y, f"+{r.sat_higher}", va="center", fontsize=8)
        ax.text(-r.sat_lower - 1, y, f"-{r.sat_lower}", va="center", ha="right", fontsize=8)
        ax.text(0, y + 0.42, f"tied {r.sat_tied}", ha="center", fontsize=7, color="grey")
    ax.set_yticks(ys); ax.set_yticklabels(sub.cond); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-110, 110); ax.set_title(f"cap {CAPL[cap]}: instances where X beats / trails C1 (of 150)")
    ax.set_xlabel("number of paired instances (satisfaction strictly higher / lower than C1)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.06))
fig.savefig(FIG / "fig10_paired_vs_c1.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 11 token composition + termination
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
for ax, cap in zip(axes[:2], [64000, 128000]):
    comp = A[A.cap == cap].groupby("condition")[["input", "thinking", "answer"]].mean().loc[CONDS]
    bottoms = np.zeros(5)
    for col, colr, lab in [("input", "#BBBBBB", "input (re-encoded context)"), ("thinking", "#555555", "thinking"),
                           ("answer", "#E69F00", "answer")]:
        ax.bar(range(5), comp[col] / 1000, bottom=bottoms / 1000, color=colr, label=lab)
        bottoms += comp[col].values
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[c] for c in CONDS])
    ax.set_ylabel("mean tokens per run (thousands)"); ax.set_title(f"Where the budget goes, cap {CAPL[cap]}")
    ax.axhline(cap / 1000, color="red", lw=0.8, ls="--")
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, fontsize=8)
ax = axes[2]
term = A.groupby(["condition", "cap"]).termination.value_counts(normalize=True).unstack(fill_value=0)
xs = np.arange(20)
labels = []
for i, c in enumerate(CONDS):
    for j, cap in enumerate(CAPS):
        r = term.loc[(c, cap)]
        x0 = i * 4 + j
        b0 = 0
        for cat, colr in [("agent_finish", "#1a9850"), ("budget", "#fdae61"), ("aborted", "#d73027")]:
            v = r.get(cat, 0.0)
            ax.bar(x0, v, bottom=b0, color=colr, width=0.85)
            b0 += v
        labels.append(CAPL[cap])
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=90, fontsize=7)
for i, c in enumerate(CONDS):
    ax.text(i * 4 + 1.5, 1.04, SHORT[c], ha="center", fontsize=9, color=COLOR[c])
ax.set_ylim(0, 1.12); ax.set_ylabel("share of runs"); ax.set_title("Termination: voluntary finish / budget / aborted")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in ["#1a9850", "#fdae61", "#d73027"]],
          labels=["agent_finish", "budget", "aborted"], frameon=False, fontsize=7.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), ncol=3)
fig.savefig(FIG / "fig11_tokens_termination.png", bbox_inches="tight")
plt.close(fig)
NUM["termination"] = {f"{SHORT[c]}|{cap}": term.loc[(c, cap)].round(3).to_dict() for c in CONDS for cap in CAPS}
comp_all = A.groupby(["condition", "cap"])[["input", "thinking", "answer"]].mean().round(0)
NUM["token_comp"] = {f"{SHORT[c]}|{cap}": comp_all.loc[(c, cap)].to_dict() for c in CONDS for cap in CAPS}

# ----------------------------------------------------------------------------- FIG 12 Block B
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True, constrained_layout=True)
for ax, n in zip(axes[:3], SIZES):
    for k, c in enumerate(CONDS):
        sub = cellsB[(cellsB.cond == SHORT[c]) & (cellsB.n_people == n)].sort_values("cap")
        off = (k - 2) * 0.06
        ax.errorbar(np.arange(4) + off, sub["mean"], yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]],
                    color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.6, capsize=2, label=LEGEND[c])
    ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS])
    ax.set_title(f"Block B, n = {n} (16 instances, O = 3)"); ax.set_ylim(-0.03, 1.03); ax.grid(axis="y", alpha=0.25)
    ax.set_xlabel("budget cap")
ax = axes[3]
for k, c in enumerate(CONDS):
    sub = cells[(cells.cond == SHORT[c]) & (cells.band == "all")].sort_values("cap")
    ax.errorbar(np.arange(4) + (k - 2) * 0.06, sub["mean"], yerr=[sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]],
                color=COLOR[c], marker=MARK[c], ms=4.5, lw=1.6, capsize=2, label=LEGEND[c])
ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS]); ax.grid(axis="y", alpha=0.25)
ax.set_title("Reference: Block A, n = 8, bands pooled (O = 3/4)"); ax.set_xlabel("budget cap")
axes[0].set_ylabel("mean satisfaction (95% bootstrap CI)")
axes[3].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.savefig(FIG / "fig12_block_b.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- FIG 13 valid-non-empty & optimality surfaces
fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.2), constrained_layout=True)
for row, (col, lab) in enumerate([("vne", "valid non-empty rate"), ("opt", "optimality rate")]):
    for ax, c in zip(axes[row], CONDS):
        M = np.array([[cell(SHORT[c], cap, band, col) for cap in CAPS] for band in BANDS])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma" if row else "cividis", aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] < 0.5 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS] if row else [])
        ax.set_yticks(range(3)); ax.set_yticklabels(["Low", "Med", "High"] if c == CONDS[0] else [])
        if row == 0:
            ax.set_title(LEGEND[c], fontsize=9.5, color=COLOR[c])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[row][0].set_ylabel(lab)
fig.savefig(FIG / "fig13_vne_opt_surfaces.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- frozen summary reproduce
summ = json.load(open(SUMMARY))
NUM["frozen_keys"] = list(summ.keys())[:40]

# ----------------------------------------------------------------------------- FIG 14 static 3D (fallback for the interactive panel)
fig = plt.figure(figsize=(13.5, 3.2))
for i, c in enumerate(CONDS):
    ax = fig.add_subplot(1, 5, i + 1, projection="3d")
    X, Y = np.meshgrid(np.arange(4), np.arange(3))
    Z = np.array([[cell(SHORT[c], cap, band) for cap in CAPS] for band in BANDS])
    ax.plot_surface(X, Y, Z, color=COLOR[c], alpha=0.85, edgecolor="white", linewidth=0.5,
                    rstride=1, cstride=1, shade=True)
    ax.set_zlim(0, 1)
    ax.set_xticks(range(4)); ax.set_xticklabels([CAPL[c_] for c_ in CAPS], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Low", "Med", "High"], fontsize=7)
    ax.set_zticks([0, 0.5, 1.0]); ax.tick_params(axis="z", labelsize=7)
    ax.set_title(LEGEND[c], fontsize=9, color=COLOR[c], pad=0)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.25, 1, 0.8))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.06)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
fig.savefig(FIG / "fig14_surface_3d_static.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- plotly 3D (optional)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    figp = go.Figure()
    for c in CONDS:
        Z = [[cell(SHORT[c], cap, band) for cap in CAPS] for band in BANDS]
        figp.add_trace(go.Surface(z=Z, x=[0, 1, 2, 3], y=[0, 1, 2], name=LEGEND[c], showscale=False,
                                  opacity=0.75, colorscale=[[0, COLOR[c]], [1, COLOR[c]]], showlegend=True))
    figp.update_layout(scene=dict(
        xaxis=dict(title="budget cap", tickvals=[0, 1, 2, 3], ticktext=["16k", "32k", "64k", "128k"]),
        yaxis=dict(title="conflict-density band", tickvals=[0, 1, 2], ticktext=["Low", "Medium", "High"]),
        zaxis=dict(title="mean satisfaction", range=[0, 1])),
        margin=dict(l=0, r=0, t=30, b=0), height=560, legend=dict(orientation="h"),
        title="Interactive: satisfaction surface per architecture over budget x band (Block A, n = 8)")
    html3d = pio.to_html(figp, include_plotlyjs=False, full_html=False, div_id="surface3d")
    (OUT / "fig3d.html").write_text(html3d, encoding="utf-8")
except Exception as e:  # pragma: no cover
    print("plotly failed:", e)

# ----------------------------------------------------------------------------- structure of the instance space
inst_feat = A[(A.condition == "c1_react") & (A.cap == 64000)]
NUM["instance_space"] = {
    "spearman_k_H": float(stats.spearmanr(inst_feat.conflict_pairs, inst_feat.H)[0]),
    "spearman_k_tightness": float(stats.spearmanr(inst_feat.conflict_pairs, inst_feat.tightness)[0]),
    "spearman_k_overlap": float(stats.spearmanr(inst_feat.conflict_pairs, inst_feat.overlap)[0]),
    "H_by_band": {b: inst_feat[inst_feat.band == b].H.value_counts().sort_index().to_dict() for b in BANDS},
    "tightness_by_band": {b: inst_feat[inst_feat.band == b].tightness.value_counts().sort_index().to_dict() for b in BANDS},
    "overlap_by_band": {b: inst_feat[inst_feat.band == b].overlap.value_counts().sort_index().to_dict() for b in BANDS},
    "optimum_by_band": {b: inst_feat[inst_feat.band == b].optimum.value_counts().sort_index().to_dict() for b in BANDS},
    "H_mean_by_band": {b: float(inst_feat[inst_feat.band == b].H.mean()) for b in BANDS},
}
# absolute meetings achieved (not normalised) per cond x cap x band
NUM["achieved"] = {f"{SHORT[c]}|{cap}|{band}": float(A[(A.condition == c) & (A.cap == cap) & (A.band == band)].achieved.mean())
                   for c in CONDS for cap in CAPS for band in BANDS}
# per-instance oracle-of-five vs best single architecture
NUM["oracle5"] = {}
for cap in CAPS:
    piv_s = A[A.cap == cap].pivot(index="instance_id", columns="condition", values="satisfaction")[CONDS]
    NUM["oracle5"][cap] = dict(oracle_of_5=float(piv_s.max(axis=1).mean()),
                               best_single=SHORT[piv_s.mean().idxmax()], best_single_mean=float(piv_s.mean().max()))
# latency ratio C3/C1 and tokens ratio
NUM["cost_ratios"] = {cap: dict(tokens_c3_over_c1=cell("C3", cap, "all", "tokens") / cell("C1", cap, "all", "tokens"),
                                latency_c3_over_c1=cell("C3", cap, "all", "latency") / cell("C1", cap, "all", "latency"),
                                latency_c5_over_c1=cell("C5", cap, "all", "latency") / cell("C1", cap, "all", "latency"))
                      for cap in CAPS}

# ----------------------------------------------------------------------------- headline numbers
NUM["headline"] = {
    "pooled_mean_by_cond_cap": {f"{SHORT[c]}|{cap}": round(cell(SHORT[c], cap), 3) for c in CONDS for cap in CAPS},
    "block_a_16k_max": float(A[A.cap == 16000].groupby("condition").satisfaction.mean().max()),
    "block_a_32k_range": [float(A[A.cap == 32000].groupby("condition").satisfaction.mean().min()),
                          float(A[A.cap == 32000].groupby("condition").satisfaction.mean().max())],
}
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, default=float)
print("cells\n", cells[cells.band == "all"].pivot(index="cond", columns="cap", values="mean").round(3))
print("\nactivation\n", activation)
print("\nfriedman\n", fried[["cap", "band", "n", "chi2", "p", "kendall_W", "CD"] + [f"rank_{SHORT[c]}" for c in CONDS]].round(3))
print("\nsolved_by\n", solved_by)
print("\ntransitions\n", trans)
print("\ngains(all)\n", gains[gains.band == "all"].round(3))
print("\nH x band\n", pd.DataFrame(NUM["H_band"]))
print("\ntravel\n", travel_tab, "\n", travel_n, "\n", NUM["travel_by_band"])
print("\nassoc 128k\n", assoc[assoc.cap == 128000].pivot(index="feature", columns="cond", values="spearman").round(2))
print("\nassoc 64k\n", assoc[assoc.cap == 64000].pivot(index="feature", columns="cond", values="spearman").round(2))
print("\npair CI counts", NUM["pair_ci_counts"])
print("\ntoken comp 64k", {k: v for k, v in NUM["token_comp"].items() if "64000" in k})
print("done ->", OUT)
