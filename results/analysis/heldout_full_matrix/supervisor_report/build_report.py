# scratchpad/build_report.py
"""Assemble the supervisor report (HTML) from report_out/{numbers.json, *.csv, figs/*.png, fig3d.html}
and the frozen summary.json. Every number in the prose is read from those artefacts."""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
DEST = Path(sys.argv[3])
DEST.mkdir(parents=True, exist_ok=True)

NUM = json.load(open(OUT / "numbers.json"))
cells = pd.read_csv(OUT / "cells_blockA.csv")
cellsB = pd.read_csv(OUT / "cells_blockB.csv")
gains = pd.read_csv(OUT / "gains_per_doubling.csv")
fried = pd.read_csv(OUT / "friedman.csv")
summ = json.load(open(REPO / "results/analysis/heldout/summary.json"))
sha = (REPO / "heldout_results_3960.sha256").read_text().strip()

CONDS = ["C1", "C2", "C3", "C4", "C5"]
CAPS = [16000, 32000, 64000, 128000]
CAPL = {16000: "16k", 32000: "32k", 64000: "64k", 128000: "128k"}
BANDS = ["low", "medium", "high"]
BANDL = {"low": "Low", "medium": "Medium", "high": "High"}
CCOL = {"C1": "#4D4D4D", "C2": "#0072B2", "C3": "#D55E00", "C4": "#009E73", "C5": "#CC79A7"}


def cell(c, cap, band="all", col="mean"):
    return float(cells[(cells.cond == c) & (cells.cap == cap) & (cells.band == band)][col].iloc[0])


def gain(c, step, band="all", col="gain"):
    return float(gains[(gains.cond == c) & (gains.step == step) & (gains.band == band)][col].iloc[0])


def fr(cap, band="all", col="kendall_W"):
    return float(fried[(fried.cap == cap) & (fried.band == band)][col].iloc[0])


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{100 * x:.0f}%"


def img(name, alt):
    data = base64.b64encode((OUT / "figs" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def badge(kind):
    lab = {"desc": "descriptive", "conf": "confirmatory · frozen", "expl": "exploratory · post hoc"}[kind]
    return f'<span class="badge badge-{kind}">{lab}</span>'


def figure(num, name, title, look, obs, kind="desc", extra=""):
    return f"""
<figure class="fig" id="fig{num}">
  <div class="fig-head"><span class="fignum">Fig. {num}</span> <span class="figtitle">{title}</span> {badge(kind)}</div>
  {img(name, title)}
  <figcaption>
    <p><b>На что смотреть.</b> {look}</p>
    <p><b>Что видно.</b> {obs}</p>
    {extra}
  </figcaption>
</figure>"""


def table(headers, rows, caption=None, cls="", align_first=True):
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for r in rows:
        tds = "".join(f"<td>{v}</td>" for v in r)
        body += f"<tr>{tds}</tr>"
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<div class="tbl-wrap"><table class="tbl {cls}">{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def cond_span(c):
    return f'<span class="cond" style="--cc:{CCOL[c]}">{c}</span>'


# --------------------------------------------------------------------------- tables
def matrix_table(cap):
    rows = []
    for band in BANDS:
        r = [BANDL[band]]
        for c in CONDS:
            m = cell(c, cap, band); lo = cell(c, cap, band, "ci_lo"); hi = cell(c, cap, band, "ci_hi")
            r.append(f"<b>{f2(m)}</b> <span class='ci'>[{f2(lo)}, {f2(hi)}]</span>")
        rows.append(r)
    r = ["<i>pooled (150)</i>"]
    for c in CONDS:
        m = cell(c, cap); lo = cell(c, cap, "all", "ci_lo"); hi = cell(c, cap, "all", "ci_hi")
        r.append(f"<b>{f2(m)}</b> <span class='ci'>[{f2(lo)}, {f2(hi)}]</span>")
    rows.append(r)
    return table(["band"] + [cond_span(c) for c in CONDS], rows,
                 caption=f"Cap {CAPL[cap]} — mean satisfaction, 95% percentile-bootstrap CI (instances resampled, 10 000, seed 20260827). n = 50 per band cell.")


def pooled_table():
    rows = []
    for c in CONDS:
        r = [cond_span(c)]
        for cap in CAPS:
            r.append(f"{f2(cell(c, cap))} <span class='ci'>[{f2(cell(c, cap, 'all', 'ci_lo'))}, {f2(cell(c, cap, 'all', 'ci_hi'))}]</span>")
        rows.append(r)
    return table(["arch"] + [CAPL[c] for c in CAPS], rows, caption="Block A pooled over bands (150 instances per cell): mean satisfaction with 95% bootstrap CI.")


def cost_table():
    rows = []
    for c in CONDS:
        for cap in CAPS:
            rows.append([cond_span(c), CAPL[cap], f"{cell(c, cap, 'all', 'tokens') / 1000:.1f}k",
                         pct(cell(c, cap, 'all', 'tokens') / cap), f"{cell(c, cap, 'all', 'calls'):.1f}",
                         f"{cell(c, cap, 'all', 'latency'):.0f}", f2(cell(c, cap, 'all', 'vne')),
                         f2(cell(c, cap, 'all', 'opt')), f"{cell(c, cap, 'all', 'sat_per_1k') * 1000:.1f}"])
    return table(["arch", "cap", "tokens/run", "cap used", "calls", "latency s", "valid non-empty", "optimal", "sat / 1M tok"],
                 rows, caption="Block A, realised cost and secondary outcomes per architecture × cap (means over 150 runs). Latency is the harness sum of model-call wall time on a shared cluster.", cls="dense")


def activation_table():
    act = pd.DataFrame(NUM["activation"])
    rows = []
    for c in CONDS:
        r = [cond_span(c)]
        for band in BANDS + ["all"]:
            a = act[(act.cond == c) & (act.band == band)].iloc[0]
            ac = "—" if pd.isna(a.activation_cap) else CAPL[int(a.activation_cap)]
            pl = "—" if pd.isna(a.plateau_cap) else CAPL[int(a.plateau_cap)]
            r.append(f"{ac} / {pl}")
        rows.append(r)
    return table(["arch", "Low", "Medium", "High", "pooled"], rows,
                 caption="Activation cap / plateau cap. Activation = smallest cap at which ≥ 50% of runs end with a valid non-empty plan. Plateau = smallest cap reaching ≥ 90% of that architecture's own 128k mean. ‘—’ = never reached within the ladder.")


def gains_table():
    rows = []
    for c in CONDS:
        r = [cond_span(c)]
        for step in ["16k->32k", "32k->64k", "64k->128k"]:
            g = gain(c, step); lo = gain(c, step, "all", "ci_lo"); hi = gain(c, step, "all", "ci_hi")
            ex = gain(c, step, "all", "exec_changed")
            r.append(f"{g:+.2f} <span class='ci'>[{lo:+.2f}, {hi:+.2f}]</span><br><span class='ci'>changed: {pct(ex)}</span>")
        rows.append(r)
    return table(["arch", "16k → 32k", "32k → 64k", "64k → 128k"], rows,
                 caption="Paired gain in mean satisfaction per budget doubling (Block A pooled, same 150 instances at both caps; 95% bootstrap CI). ‘changed’ = share of runs whose token total differs between the two caps, i.e. runs the extra budget actually altered.")


def friedman_table():
    rows = []
    for cap in CAPS:
        for band in ["all"] + BANDS:
            r = fried[(fried.cap == cap) & (fried.band == band)].iloc[0]
            p = "—" if pd.isna(r.p) else ("< 0.0001" if r.p < 1e-4 else f"{r.p:.4f}")
            ranks = " · ".join(f"{c} {r[f'rank_{c}']:.2f}" for c in sorted(CONDS, key=lambda c_: r[f"rank_{c_}"]))
            rows.append([CAPL[cap], "pooled" if band == "all" else BANDL[band], int(r.n), f"{r.chi2:.1f}", p, f2(r.kendall_W), ranks])
    return table(["cap", "stratum", "n", "χ²(4)", "p", "Kendall W", "mean ranks (1 = best)"], rows,
                 caption="Friedman test across the five architectures on paired instances, with Kendall's W as effect size (0 = indistinguishable, 1 = identical ordering on every instance). Exploratory, not registered; ties handled by average ranks.", cls="dense")


def frozen_table():
    rows = []
    for c in summ["confirmatory"]["contrasts"]:
        b = c["bands"]
        rows.append([c["contrast"], f"{b['low']['mean_delta']:+.3f}", f"{b['medium']['mean_delta']:+.3f}", f"{b['high']['mean_delta']:+.3f}",
                     f"{c['crossover']['p_iut']:.4f}", f"{c['crossover']['holm']['p_holm']:.4f}",
                     f"{c['interaction']['delta_statistic']:+.3f}", f"{c['interaction']['p_two_sided']:.4f}",
                     f"{c['interaction']['holm']['p_holm']:.4f}", "no" if not c["crossover"]["established"] else "yes"])
    return table(["contrast", "δ Low", "δ Medium", "δ High", "p IUT", "p IUT Holm", "Δ = δ_High − δ_Low", "p interaction", "p int. Holm", "crossover"], rows,
                 caption="Frozen confirmatory family at cap 64 000 (analysed 2026-09-01, commit b84959f): five registered contrasts, δ = sat(later) − sat(earlier); sign-flip permutation on the mean, 99 999 permutations; Holm across the ten p-values. Reproduced from summary.json, not recomputed.", cls="dense")


def blockb_table():
    rows = []
    for n in [4, 5, 6]:
        for c in CONDS:
            r = [f"n = {n}" if c == "C1" else "", cond_span(c)]
            for cap in CAPS:
                x = cellsB[(cellsB.cond == c) & (cellsB.n_people == n) & (cellsB.cap == cap)].iloc[0]
                r.append(f"{x['mean']:.2f} <span class='ci'>[{x.ci_lo:.2f}, {x.ci_hi:.2f}]</span>")
            rows.append(r)
    return table(["size", "arch", "16k", "32k", "64k", "128k"], rows,
                 caption="Block B (16 instances per size, all at oracle optimum O = 3): mean satisfaction with 95% bootstrap CI. n is a generator parameter, not a complexity band.", cls="dense")


def transitions_table():
    tr = pd.DataFrame(NUM["transitions"])
    rows = []
    for cap in [64000, 128000]:
        for c in ["C2", "C3", "C4", "C5"]:
            r = tr[(tr.cap == cap) & (tr.cond == c)].iloc[0]
            rows.append([CAPL[cap], cond_span(c), f"+{r.sat_higher}", f"−{r.sat_lower}", r.sat_tied, f"+{r.recovered}", f"−{r.lost}"])
    return table(["cap", "X", "X > C1 (sat)", "X < C1 (sat)", "tied", "X valid, C1 empty", "X empty, C1 valid"], rows,
                 caption="Paired instance-level comparison of each architecture X against C1 on the same 150 instances: satisfaction strictly higher / lower / tied, and transitions in ‘has a valid non-empty plan’.")


def travel_table():
    tt = NUM["travel_tab"]; tn = NUM["travel_n"]; tb = NUM["travel_by_band"]
    rows = []
    for cap in [64000, 128000]:
        for tr in ["clustered", "line", "random", "uniform"]:
            row = tt[f"{cap}|{tr}"]
            rows.append([CAPL[cap], tr, f"{tn[tr]} ({tb[tr]['low']}/{tb[tr]['medium']}/{tb[tr]['high']})"] +
                        [f2(row[c]) for c in CONDS] + [f"{row['C3'] - row['C1']:+.2f}"])
    return table(["cap", "travel structure", "n instances (L/M/H)"] + [cond_span(c) for c in CONDS] + ["C3 − C1"], rows,
                 caption="Mean satisfaction by travel structure of the instance (Block A). Travel structure is a generator parameter and is not balanced across bands; this is an association, not a controlled contrast.", cls="dense")


def hband_table():
    hb = pd.DataFrame(NUM["H_band"])
    rows = []
    for r in hb.itertuples():
        rows.append([CAPL[int(r.cap)], BANDL[r.band], r.H, r.n] + [f2(getattr(r, c)) for c in CONDS] + [f"{r.C3 - r.C1:+.2f}"])
    return table(["cap", "band", "H stratum", "n"] + [cond_span(c) for c in CONDS] + ["C3 − C1"], rows,
                 caption="Mean satisfaction by band × higher-order-gap stratum (H = α_reachable − O). The Low band has no H = 0 instance at all, which is the recorded imbalance of the design.", cls="dense")


def solved_table():
    sb = pd.DataFrame(NUM["solved_by"]); o5 = NUM["oracle5"]
    rows = []
    for r in sb.itertuples():
        cap = int(r.cap)
        rows.append([CAPL[cap], r.solved_by_0, r.solved_by_1, r.solved_by_2, r.solved_by_3, r.solved_by_4, r.solved_by_5,
                     " · ".join(f"{c} {getattr(r, f'{c}_strictly_best')}" for c in CONDS),
                     f"{o5[str(cap)]['oracle_of_5']:.2f} vs {o5[str(cap)]['best_single']} {o5[str(cap)]['best_single_mean']:.2f}"])
    return table(["cap", "solved by 0", "1", "2", "3", "4", "all 5", "strictly best (count of instances)", "oracle-of-5 vs best single"], rows,
                 caption="How many of the five architectures produce a valid non-empty plan on the same instance; which architecture is strictly best per instance; and the mean of the per-instance best satisfaction (an oracle selector over the five) against the best single architecture.", cls="dense")


# --------------------------------------------------------------------------- numbers for prose
W = {cap: fr(cap) for cap in CAPS}
o5 = NUM["oracle5"]
ratios = NUM["cost_ratios"]
isp = NUM["instance_space"]
sb = {int(r["cap"]): r for r in NUM["solved_by"]}
tr = {(int(r["cap"]), r["cond"]): r for r in NUM["transitions"]}
tok = NUM["token_comp"]
fam = summ["confirmatory"]["family"]
min_holm = min(h["p_holm"] for h in fam)
best_int = min(fam, key=lambda h: h["p_raw"] if h["hypothesis"].endswith("interaction") else 9)

cube_data = {c: [[cell(c, cap, band) for cap in CAPS] for band in BANDS] for c in CONDS}
CUBE3D = (Path(__file__).parent / "cube3d.html").read_text(encoding="utf-8")     .replace("__DATA__", json.dumps(cube_data))     .replace("__COLOR__", json.dumps(CCOL))     .replace("__CAPS__", json.dumps([CAPL[c] for c in CAPS]))     .replace("__BANDS__", json.dumps([BANDL[b] for b in BANDS]))

# --------------------------------------------------------------------------- CSS
CSS = """
:root{
  --ground:#F6F7F5; --panel:#FFFFFF; --sunk:#ECEFEC; --ink:#1C2126; --soft:#414B52; --muted:#6B767D;
  --rule:#D6DBD9; --accent:#0F6E74; --accent-ink:#0B565B; --amber:#8A6210; --amber-bg:#F7EEDA;
  --teal-bg:#E2F0F0; --grey-bg:#ECEFEC; --warn:#9A3B1E; --warn-bg:#F8E7E1;
  --f-display:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --ground:#12171A; --panel:#1A2125; --sunk:#151B1F; --ink:#E6EAE8; --soft:#C3CBCF; --muted:#8F9BA2;
  --rule:#2B353A; --accent:#5FC1C7; --accent-ink:#8ED6DA; --amber:#E0B25A; --amber-bg:#2E2610;
  --teal-bg:#12302F; --grey-bg:#1F2A2F; --warn:#F0A088; --warn-bg:#3A1F16; }}
:root[data-theme="dark"]{
  --ground:#12171A; --panel:#1A2125; --sunk:#151B1F; --ink:#E6EAE8; --soft:#C3CBCF; --muted:#8F9BA2;
  --rule:#2B353A; --accent:#5FC1C7; --accent-ink:#8ED6DA; --amber:#E0B25A; --amber-bg:#2E2610;
  --teal-bg:#12302F; --grey-bg:#1F2A2F; --warn:#F0A088; --warn-bg:#3A1F16; }
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--f-body);font-size:16.5px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:var(--accent-ink)}
.page{display:grid;grid-template-columns:230px minmax(0,1fr);gap:40px;max-width:1440px;margin:0 auto;padding:32px 28px 80px}
@media (max-width:960px){.page{grid-template-columns:1fr;gap:16px} nav.toc{position:static;max-height:none}}
nav.toc{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;font-size:13.5px;padding-right:8px}
nav.toc .eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
nav.toc ol{list-style:none;margin:0;padding:0}
nav.toc li{margin:0 0 6px}
nav.toc a{color:var(--soft);text-decoration:none;display:block;padding:2px 0 2px 10px;border-left:2px solid var(--rule)}
nav.toc a:hover,nav.toc a:focus{color:var(--accent-ink);border-left-color:var(--accent)}
nav.toc li.sub a{padding-left:22px;font-size:12.5px}
main{min-width:0}
header.masthead{border-bottom:1px solid var(--rule);padding-bottom:22px;margin-bottom:34px}
header .eyebrow{font-family:var(--f-mono);font-size:12px;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
h1{font-family:var(--f-display);font-weight:600;font-size:40px;line-height:1.12;margin:0 0 12px;text-wrap:balance;letter-spacing:-.01em}
.lede{font-size:19px;color:var(--soft);max-width:72ch;margin:0 0 16px}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;font-family:var(--f-mono);font-size:12.5px;color:var(--muted)}
h2{font-family:var(--f-display);font-weight:600;font-size:28px;line-height:1.2;margin:56px 0 14px;padding-top:12px;border-top:1px solid var(--rule);text-wrap:balance}
h2 .num{color:var(--accent);font-family:var(--f-mono);font-size:15px;font-weight:500;margin-right:10px;vertical-align:middle}
h3{font-family:var(--f-display);font-weight:600;font-size:21px;margin:32px 0 8px;text-wrap:balance}
h4{font-size:16px;margin:22px 0 6px;text-transform:uppercase;letter-spacing:.06em;color:var(--soft);font-weight:600}
p,li{max-width:78ch}
main p{margin:0 0 12px}
ul,ol{padding-left:22px;margin:0 0 14px}
li{margin:0 0 6px}
b{font-weight:600}
code,.mono{font-family:var(--f-mono);font-size:.88em;background:var(--sunk);padding:1px 5px;border-radius:3px}
.cond{font-family:var(--f-mono);font-weight:600;color:var(--cc);white-space:nowrap}
.ci{color:var(--muted);font-size:.85em;font-family:var(--f-mono)}
.badge{display:inline-block;font-family:var(--f-mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;border-radius:3px;vertical-align:middle;margin-left:6px}
.badge-desc{background:var(--grey-bg);color:var(--soft)}
.badge-conf{background:var(--teal-bg);color:var(--accent-ink)}
.badge-expl{background:var(--amber-bg);color:var(--amber)}
.fig{margin:26px 0 34px;padding:14px 16px 10px;background:var(--panel);border:1px solid var(--rule);border-radius:4px}
.fig img{width:100%;height:auto;display:block;border-radius:2px}
.fig-head{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:0 0 10px;font-size:14px}
.fignum{font-family:var(--f-mono);font-weight:600;color:var(--accent)}
.figtitle{font-weight:600}
figcaption{font-size:14.5px;color:var(--soft);margin-top:12px}
figcaption p{margin:0 0 6px;max-width:none}
.tbl-wrap{overflow-x:auto;margin:14px 0 22px;border:1px solid var(--rule);border-radius:4px;background:var(--panel)}
table.tbl{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
table.tbl caption{caption-side:bottom;text-align:left;font-size:13px;color:var(--muted);padding:8px 12px;border-top:1px solid var(--rule)}
table.tbl th,table.tbl td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule)}
table.tbl th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--sunk);font-weight:600;white-space:nowrap}
table.tbl tbody tr:last-child td{border-bottom:none}
table.tbl.dense{font-size:13px}
table.tbl.dense th,table.tbl.dense td{padding:5px 8px}
.callout{border-left:3px solid var(--accent);background:var(--panel);padding:12px 16px;margin:18px 0;border-radius:0 4px 4px 0}
.callout.amber{border-left-color:var(--amber)}
.callout.warn{border-left-color:var(--warn)}
.callout p:last-child{margin:0}
.finding{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:16px 18px;margin:14px 0}
.finding h3{margin:0 0 8px;font-size:19px}
.finding .grid{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;font-size:14.5px}
.finding .grid dt{color:var(--muted);font-family:var(--f-mono);font-size:12px;text-transform:uppercase;letter-spacing:.05em;padding-top:2px}
.finding .grid dd{margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0 20px}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:10px 12px}
.kpi .v{font-family:var(--f-mono);font-size:22px;font-weight:600;color:var(--accent-ink)}
.kpi .l{font-size:12.5px;color:var(--muted)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:900px){.two{grid-template-columns:1fr}}
.plot3d{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:8px;margin:20px 0}
.small{font-size:13.5px;color:var(--muted)}
.cube-wrap{position:relative}
#cube3d{width:100%;height:auto;display:block;color:var(--muted);cursor:grab;touch-action:none;outline:none;-webkit-user-select:none;user-select:none}
#cube3d.is-dragging{cursor:grabbing}
#cube3d:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#cube3d .cube-scaffold line{stroke:var(--rule);stroke-width:1}
#cube3d text.cube-tick{font-family:var(--f-mono);font-size:11px;fill:currentColor}
#cube3d text.cube-axis{font-family:var(--f-body);font-size:12.5px;fill:currentColor}
.cube-bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:4px 4px 6px;border-top:1px solid var(--rule);margin-top:4px}
.cube-legend{display:flex;flex-wrap:wrap;gap:6px}
.cube-chip{display:inline-flex;align-items:center;gap:6px;font-family:var(--f-mono);font-size:12px;font-weight:600;
  background:var(--sunk);color:var(--ink);border:1px solid var(--rule);border-radius:999px;padding:3px 10px;cursor:pointer}
.cube-chip .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.cube-chip.is-off{opacity:.42}
.cube-chip:hover{border-color:var(--accent)}
.cube-btn{margin-left:auto;font-family:var(--f-body);font-size:12.5px;background:none;color:var(--accent-ink);
  border:1px solid var(--rule);border-radius:3px;padding:3px 10px;cursor:pointer}
.cube-btn:hover{border-color:var(--accent)}
hr{border:0;border-top:1px solid var(--rule);margin:30px 0}
.narr ol li{margin-bottom:10px}
"""

# --------------------------------------------------------------------------- body sections
S = []  # sections as html strings

S.append(f"""
<header class="masthead">
  <p class="eyebrow">Supervisor reading · held-out matrix · Qwen3-32B-AWQ · {NUM['n_runs']} runs / {NUM['n_instances']} instances</p>
  <h1>Five Architectures, Four Budgets</h1>
  <p class="lede">Как пять систем (C1–C5) превращают один и тот же токен-бюджет в satisfaction на пространстве задач
  «бюджет × структурная сложность × размер». Сначала данные, потом интерпретация, потом каркас Results/Discussion для Bachelorarbeit.</p>
  <div class="meta">
    <span>data: results/exports/heldout/runs.csv</span>
    <span>frozen analysis: results/analysis/heldout (b84959f)</span>
    <span>sha256 (3960 docs): {sha.split()[0][:16]}…</span>
    <span>built 2026-09-03</span>
  </div>
</header>

<div class="callout">
<p><b>Статус этого документа.</b> Это не пре-регистрация и не замена замороженного конфирматорного анализа. Всё, что здесь помечено
{badge('desc')}, описывает выборку; {badge('conf')} — воспроизведено из замороженного <code>summary.json</code> без пересчёта;
{badge('expl')} — новые тесты, добавленные мной постфактум, и в тексте тезиса они могут идти только как exploratory.
Рамка чтения — та, которую ты просил: пять систем на равных, четыре бюджета на равных, вопрос «как эффективно каждая архитектура
использует данный бюджет», а не «кто лучше».</p>
</div>
""")

# ---- 1. understanding
S.append(f"""
<h2 id="s1"><span class="num">1</span>Что за эксперимент</h2>
<p><b>Research question (как зарегистрирована).</b> При равном токен-бюджете существует ли порог сложности, начиная с которого иерархическая
MAS обгоняет одиночного ReAct-агента на meeting planning, и стоит ли выигрыш координационных издержек? В замороженном плане это
превратилось в тест на <i>crossover</i> по оси парной конфликтной плотности D. Дескриптивная рамка (amendment 2026-09-02) шире:
полная матрица 5 × 4 × (3 + 3), все клетки на равных.</p>

<h3>Условия</h3>
{table(["id", "что это", "что варьируется относительно соседа"], [
 [cond_span("C1"), "ReAct, один агент, один контекст", "baseline"],
 [cond_span("C2"), "ReAct + фиксированный Draft→Verify→Revise ×2 (reflection-only, без tools в revise)", "C1 + mandated self-verification в том же контексте"],
 [cond_span("C3"), "иерархическая MAS: детерминированный split k=2 по travel-матрице → два C1-воркера на под-инстансах (квоты 3/8·W каждый, изолированные контексты) → zero-token aggregator (fallback + draft + pool) → один fresh-context critic над pool", "C4 + декомпозиция и её координационный интерфейс"],
 [cond_span("C4"), "один planner на полном инстансе (квота 3/4·W) + один fresh-context critic над pool его же предложений, без декомпозиции", "C2 + независимый критик вместо self-verification (не однофакторный контраст)"],
 [cond_span("C5"), "Best-of-3: три независимых C1-траектории по W/3 каждая (seeds 42/43/44), детерминированный validator-gated selector, одна финализация", "C1 + иначе потраченный тот же бюджет: три коротких поиска вместо одного длинного"],
], caption="W = cap − 256 (finalisation reserve). Все условия: те же три tools, тот же ledger/guard, те же инстансы, тот же tokenizer, тот же decoding profile (T=0.6, top_p=0.95, top_k=20, thinking ON). Все токены — input, thinking, answer, inter-agent messages — идут в один бюджет.")}

<h3>Факторы и пространство задач</h3>
<ul>
<li><b>Бюджет (cap):</b> 16k / 32k / 64k / 128k, геометрическая лестница. Важно: окно модели 32 768, поэтому больший cap покупает <i>больше вызовов</i>, а не длиннее вызовы.</li>
<li><b>Block A, n = 8, 150 инстансов:</b> три полосы парной конфликтной плотности D = k/28 — Low (k = 0–1), Medium (k = 7), High (k = 13–16), по 50; гистограмма оптимума O одинаковая во всех полосах (32 × O=3, 18 × O=4). Только это — зарегистрированная ось «сложности».</li>
<li><b>Block B, n = 4/5/6, по 16 инстансов, все O = 3:</b> ось <i>размера</i>; по решению проекта n — параметр генератора, не уровень сложности, и Block B анализируется отдельно.</li>
<li><b>Ковариаты инстанса (записаны, не сбалансированы):</b> higher-order gap H = α_reachable − O, tightness, overlap, travel structure (clustered/line/random/uniform), концентрация конфликтов.</li>
</ul>
<p><b>Метрика.</b> satisfaction = achieved/optimum, 0 при любом нарушении; финальный план по конструкции всегда валиден или пуст, поэтому
информативная «успешность» — <i>valid non-empty rate</i>. Вторичные: optimality rate, realised tokens, calls, latency, satisfaction/1k.
Значения satisfaction живут на решётке {{0, ¼, ⅓, ½, ⅔, ¾, 1}} — много нулей и связок; это определяет, какие тесты уместны.</p>

<h3>Что этими данными реально можно проверить</h3>
<ol>
<li>Зависит ли ранжирование пяти систем от бюджета (architecture × budget)?</li>
<li>Зависит ли оно от D-полосы (architecture × complexity) — зарегистрированный crossover?</li>
<li>Какой бюджет «активирует» каждую архитектуру, где у неё плато и предельная отдача от удвоения?</li>
<li>Сколько бюджета фактически тратится и какова эффективность на реально потраченный токен?</li>
<li>На уровне инстансов: одни и те же ли задачи решают разные архитектуры или разные (комплементарность, цена декомпозиции)?</li>
<li>Как результат меняется с размером n (Block B), и как он связан со структурными ковариатами (H, travel structure).</li>
</ol>
<p class="small">Чего проверить нельзя: перенос на другую модель (8B исключена до пилота), другую стратегию split, N ≠ 3 для best-of-N, влияние квантования.</p>
""")

# ---- 2. variables & structure of the instance space
S.append(f"""
<h2 id="s2"><span class="num">2</span>Структура данных, о которой надо знать до графиков</h2>
<p>Три вещи в устройстве выборки определяют, как читать всё дальше.</p>
<ol>
<li><b>Полностью парный дизайн.</b> Каждый из 198 инстансов решён всеми пятью системами при всех четырёх cap — 20 наблюдений на инстанс. Поэтому любое сравнение архитектур или бюджетов делается <i>внутри инстанса</i>, а разброс между инстансами (очень большой) уходит из знаменателя.</li>
<li><b>Ось D одномерна на практике и склеена с H и tightness.</b> Spearman(k, H) = {isp['spearman_k_H']:.2f}, Spearman(k, tightness) = {isp['spearman_k_tightness']:+.2f}, Spearman(k, overlap) = {isp['spearman_k_overlap']:.2f}. Средний H по полосам: {isp['H_mean_by_band']['low']:.2f} / {isp['H_mean_by_band']['medium']:.2f} / {isp['H_mean_by_band']['high']:.2f}; в Low band <i>нет ни одного</i> инстанса с H = 0, в High band их 28 из 50. «Low D» одновременно означает «High H и низкая tightness». Это записанное ограничение дизайна (THESIS_DECISIONS §3), и оно означает, что эффект полосы нельзя приписать одной парной плотности.</li>
<li><b>Один прогон на клетку.</b> Гарнес детерминирован (prefix caching off, seed 42; воспроизводимость до токена подтверждена на пробах), так что повторов нет и «шум» — это только межинстансная вариация. Все интервалы здесь — bootstrap по инстансам.</li>
</ol>
""")

# ---- 3. figures
S.append(f"""
<h2 id="s3"><span class="num">3</span>Данные: минимальный набор представлений</h2>
<p>Порядок: сначала поверхность architecture × budget × band (три взгляда на один и тот же куб), потом бюджетная динамика,
потом уровень инстансов и пространство задач, потом стоимость, потом Block B. Всё, что я счёл второстепенным, ушло в приложение (§10).</p>

<h3 id="s3a">3.1 Поверхность architecture × budget × band</h3>
{figure(1, "fig01_surface_heatmaps.png", "Mean satisfaction, band × cap, one panel per architecture (Block A, n = 8)",
 "Одна цветовая шкала на все пять панелей. Читай слева направо внутри панели (эффект бюджета) и сверху вниз (эффект полосы), потом сравнивай панели между собой.",
 f"Градиент почти целиком горизонтальный: бюджет доминирует. Вертикальный градиент внутри панели слабый и одинаковый по знаку у всех пяти (Low чуть выше). Панели различаются не формой, а <i>яркостью справа</i>: C3 достигает {f2(cell('C3',128000,'low'))} в Low/128k, C1 — {f2(cell('C1',128000,'low'))}. Столбец 16k пуст у всех; столбец 32k почти неотличим между C1, C2, C3 (C4 и особенно C5 ниже).",
 kind="desc")}

{figure(2, "fig02_budget_response_bands.png", "Budget-response curves per band, 95% bootstrap CI",
 "Наклон каждой линии между соседними cap — предельная отдача от удвоения; пересечения линий — смена лидера с бюджетом; пересечение <i>внутри</i> панели между полосами здесь не видно, поэтому четвёртая панель (pooled) почти совпадает с любой из трёх.",
 f"Две группы кривых. C1/C2 растут рано (16k→32k→64k) и потом выходят на плато; C3/C4 стартуют с той же точки, но между 32k и 64k делают скачок ~+0.5 и обгоняют. C5 лежит на нуле до 64k и делает свой скачок только на 128k. Порядок при 64k и 128k один и тот же во всех трёх полосах: C3 > C4 ≳ C2 > C1 (C5 меняет место). Интервалы у C1 самые широкие — бимодальность (0 или почти 1).",
 kind="desc")}

{figure(3, "fig03_band_effect_per_cap.png", "The same surface transposed: satisfaction against band, one panel per cap",
 "Здесь проверяется зарегистрированный crossover: он выглядел бы как пересечение двух линий между Low и High. Параллельность линий = отсутствие architecture × D взаимодействия.",
 f"При 64k и 128k линии почти параллельны и не пересекаются; при 32k все пять лежат в узкой полосе 0.0–0.25 с перекрывающимися интервалами. Наклон по D у всех отрицательный и небольшой (для C3 при 128k {f2(cell('C3',128000,'low'))} → {f2(cell('C3',128000,'high'))}, для C1 {f2(cell('C1',128000,'low'))} → {f2(cell('C1',128000,'high'))}). Единственная линия с заметно <i>растущим</i> относительным зазором к C1 — C2 (и это единственное взаимодействие с raw p = {best_int['p_raw']:.3f} в замороженном анализе, Holm {best_int['p_holm']:.2f}).",
 kind="desc")}

<div class="plot3d" id="fig3d">
<div class="fig-head"><span class="fignum">Fig. 3-i</span> <span class="figtitle">Интерактивный куб: satisfaction над плоскостью budget × band, по одной поверхности на архитектуру</span> {badge('desc')}</div>
<p class="small">Те же 60 средних, что на Fig. 1, но как рельеф. Тяни мышью — вращение; клик по чипу — скрыть или показать архитектуру; «сброс» возвращает исходный ракурс. Рендер собственный (inline SVG), без внешних библиотек и без WebGL, поэтому работает офлайн и в печати. Для основного текста тезиса 3D всё равно не рекомендую: Fig. 1–3 читаются надёжнее; это картинка для защиты и для собственного разглядывания куба.</p>
{CUBE3D}
<div id="cube3d-fallback" hidden>
  {img("fig14_surface_3d_static.png", "Static 3D satisfaction surfaces, one per architecture")}
  <p class="small">Статическая версия (интерактив не запустился): те же 60 средних как пять отдельных поверхностей, одна шкала z 0–1 на все панели.</p>
</div>
</div>

<h3 id="s3b">3.2 Бюджетная динамика: активация, плато, предельная отдача</h3>
{figure(4, "fig04_marginal_gain_per_doubling.png", "Paired gain in mean satisfaction per budget doubling, by band",
 "Высота столбца — сколько satisfaction в среднем покупает удвоение cap <i>на тех же инстансах</i>; интервал — bootstrap по парным разностям. Ищи, у какой архитектуры «главный» столбец на каком шаге.",
 f"У каждой архитектуры есть один доминирующий шаг: C1 — 16k→32k ({gain('C1','16k->32k'):+.2f}) и 32k→64k ({gain('C1','32k->64k'):+.2f}), затем почти ноль ({gain('C1','64k->128k'):+.2f}); C3 и C4 — 32k→64k ({gain('C3','32k->64k'):+.2f} и {gain('C4','32k->64k'):+.2f}); C5 — 64k→128k ({gain('C5','64k->128k'):+.2f}). После своего главного шага C4 почти не растёт ({gain('C4','64k->128k'):+.2f}), C3 продолжает ({gain('C3','64k->128k'):+.2f}). Ни одно удвоение не ухудшает средний результат (парные ухудшения единичны — гарнес хранит best_plan_so_far).",
 kind="expl")}

{activation_table()}
{gains_table()}

<div class="callout amber"><p><b>Оговорка про «changed».</b> Удвоение cap меняет исполнение только тех прогонов, которые упирались в нижний cap. Для C1 на шаге 64k→128k изменились лишь {pct(gain('C1','64k->128k','all','exec_changed'))} прогонов: остальные завершились добровольно ниже 64k и воспроизвели себя байт в байт. Поэтому «C1 не растёт с бюджетом» — это не «C1 не умеет тратить», а «C1 сам останавливается, оставив половину бюджета» (см. Fig. 11).</p></div>

<h3 id="s3c">3.3 Стоимость и эффективность</h3>
{figure(5, "fig05_cost_quality.png", "Cost-quality trajectories, cap utilisation, efficiency",
 "Левая панель — главная для рамки «эффективность использования бюджета»: точка = (реально потраченные токены, satisfaction), линия соединяет четыре cap одной архитектуры. Выше и левее — лучше. Средняя панель: доля cap, которую система фактически тратит. Правая: satisfaction на миллион реально потраченных токенов.",
 f"C3 и C4 лежат выше и левее всех при любом бюджете от 64k: C3 при 64k тратит {cell('C3',64000,'all','tokens')/1000:.0f}k токенов (меньше, чем C1 с {cell('C1',64000,'all','tokens')/1000:.0f}k) и даёт {f2(cell('C3',64000))} против {f2(cell('C1',64000))}. Отношение токенов C3/C1 = {ratios['64000']['tokens_c3_over_c1']:.2f} при 64k и {ratios['128000']['tokens_c3_over_c1']:.2f} при 128k. Все, кроме C5, тратят при 128k меньше половины cap — потолок поведенческий, а не бюджетный. C5 — единственная, кто всегда выжигает cap; её эффективность на токен худшая на всей лестнице.",
 kind="desc")}
{cost_table()}

{figure(6, "fig13_vne_opt_surfaces.png", "Valid non-empty rate (top) and optimality rate (bottom), band × cap per architecture",
 "Два «слоя» под satisfaction: верхний — доля прогонов, где вообще есть непустой валидный план; нижний — доля прогонов, достигших оптимума. Разница между слоями — качество планов, когда они есть.",
 f"Верхний слой объясняет большую часть Fig. 1: при 64k C4 и C3 имеют план в {pct(cell('C4',64000,'all','vne'))} и {pct(cell('C3',64000,'all','vne'))} прогонов, C1 — в {pct(cell('C1',64000,'all','vne'))}. Нижний слой различает C3 и C4: при равной «активации» C3 доводит до оптимума {pct(cell('C3',128000,'all','opt'))} прогонов при 128k, C4 — {pct(cell('C4',128000,'all','opt'))}. Т.е. C4 почти так же часто <i>находит</i> план, но реже <i>полный</i>: fresh-context critic без декомпозиции даёт валидность, а не длину.",
 kind="desc")}

<h3 id="s3d">3.4 Уровень инстансов и пространство задач</h3>
{figure(7, "fig06_instance_matrix.png", "Instance × architecture satisfaction, one panel per cap (150 instances, sorted by band, k, H)",
 "Каждая строка — инстанс, каждый столбец — архитектура, цвет — satisfaction на решётке. Смотри на (а) горизонтальные полосы — инстансы, которые решают все или никто; (б) вертикальные — архитектуры, которые решают почти всё; (в) «дырки» в столбце C3, которых нет в столбце C1 — цена декомпозиции.",
 f"16k — пустой лист (у {sb[16000]['solved_by_0']} из 150 инстансов нет ни одного валидного плана). 32k — разреженный узор без вертикальной структуры: кто что решил, почти случайно. 64k — появляются вертикальные столбцы C3 и C4, но при этом видны инстансы, где C1 решил, а C3 нет. 128k — столбец C3 почти сплошной; остаётся {sb[128000]['solved_by_0']} инстанс, не решённый никем, и {sb[128000]['solved_by_5']} инстансов, решённых всеми пятью. Полосы D видны слабо: в High band столбцы такие же плотные, как в Low.",
 kind="desc")}
{solved_table()}

{figure(8, "fig10_paired_vs_c1.png", "Per-instance comparison of each architecture against C1 (satisfaction strictly higher / lower)",
 "Синяя часть — инстансы, где X строго выше C1; красная — где строго ниже. Асимметрия красной части между C2/C4 и C3 — это то, что декомпозиция стоит.",
 f"При 64k C3 обгоняет C1 на {tr[(64000,'C3')]['sat_higher']} инстансах и проигрывает на {tr[(64000,'C3')]['sat_lower']}; C4 — {tr[(64000,'C4')]['sat_higher']} / {tr[(64000,'C4')]['sat_lower']}; C2 — {tr[(64000,'C2')]['sat_higher']} / {tr[(64000,'C2')]['sat_lower']} (ни одного проигрыша). При 128k проигрыши C3 сокращаются до {tr[(128000,'C3')]['sat_lower']}. C5 при 64k проигрывает C1 на {tr[(64000,'C5')]['sat_lower']} инстансах, при 128k ситуация переворачивается ({tr[(128000,'C5')]['sat_higher']} / {tr[(128000,'C5')]['sat_lower']}).",
 kind="desc")}
{transitions_table()}

{figure(9, "fig07_task_space.png", "Task space: k × H per instance coloured by C3 − C1 (64k, 128k); satisfaction against H",
 "Левые панели — где в пространстве (парная плотность k, higher-order gap H) лежат инстансы и где C3 впереди (красный) или позади (синий) C1. Правая — средняя satisfaction всех пяти как функция H (сплошная 128k, пунктир 64k).",
 f"Пространство задач — три изолированных кластера, а не плоскость: Low band сидит при H ∈ {{4,5}}, High — при H ∈ {{0,1}}. Внутри каждого кластера красное и синее перемешаны без видимой структуры по H. Правая панель: satisfaction <i>растёт</i> с H у всех пяти архитектур, т.е. на этой выборке инстансы с большим higher-order gap легче для агентов, а не сложнее — потому что H сцеплен с низкой tightness. Порядок архитектур при каждом H одинаков.",
 kind="desc")}
{hband_table()}
{travel_table()}

<h3 id="s3e">3.5 Полная матрица парных контрастов</h3>
{figure(10, "fig09_pairwise_matrix.png", "All ten pairwise paired differences × four caps, one panel per band (from the frozen contrasts.csv)",
 "Знак = sat(later) − sat(earlier) в bookkeeping-порядке C1 < C2 < C5 < C4 < C3 (это не ранжирование). Звёздочка — описательный 95% bootstrap CI не накрывает ноль. Ищи клетки, где знак меняется <i>по строке</i> (эффект бюджета) и где он менялся бы <i>между панелями</i> (эффект D — его нет).",
 f"По строкам знак меняется у всех пар с участием C5 (минус при 32k/64k, плюс при 128k) и у C1→C4 / C2→C4 (минус при 32k, плюс от 64k). Между панелями знак и величина почти не меняются: одна и та же строка читается одинаково в Low, Medium и High. Пара C4→C3 — единственная «архитектурная» пара с нулевым эффектом при 64k во всех полосах ({NUM['pair_ci_counts']['C4->C3']['pos']} из 12 клеток со звёздочкой, все при 128k).",
 kind="desc")}

<h3 id="s3f">3.6 Куда уходит бюджет и как заканчиваются прогоны</h3>
{figure(11, "fig11_tokens_termination.png", "Token composition (input / thinking / answer) at 64k and 128k; termination shares per architecture × cap",
 "Слева: из чего состоит потраченный бюджет; серая часть — повторно кодируемый контекст (цена «памяти» одиночного агента и цена сообщений в MAS). Справа: как заканчивается прогон — добровольно, по бюджету или аварийно (пустой ответ после think).",
 f"Координационный overhead C3 в токенах не виден: доля input у C3 ({tok['C3|64000']['input']/1000:.1f}k при 64k) ниже, чем у C1 ({tok['C1|64000']['input']/1000:.1f}k) — изолированные короткие контексты воркеров дешевле, чем один длинный контекст, который C1 пересылает каждый вызов. Thinking у всех ~32k при 64k; C5 тратит больше всех на input (три траектории). Справа: при 16k/32k доминируют aborted (пустой post-think) у C1/C2/C4; C3 при 128k завершается добровольно в 100% прогонов; C1 при 64k+ тоже — с половиной бюджета в остатке.",
 kind="desc")}

<h3 id="s3g">3.7 Block B: размер задачи</h3>
{figure(12, "fig12_block_b.png", "Block B: satisfaction against cap for n = 4, 5, 6 (O = 3), with Block A pooled as reference",
 "Сравнивай форму кривых между панелями: где C1 выходит на плато и на каком уровне; сдвигается ли «точка активации» с n.",
 f"Форма та же, что в Block A, но смещена влево: при n = 4 всё активируется уже на 32k, C1 достигает плато {cellsB[(cellsB.cond=='C1')&(cellsB.n_people==4)&(cellsB.cap==64000)]['mean'].iloc[0]:.2f} и дальше не растёт ни при 64k, ни при 128k (буквально те же прогоны). C3 при n = 4–6 доходит до 0.83–0.94; порядок C3 > C4 ≳ C2 > C1 при 64k+ повторяется. Единственная кривая, форма которой зависит от n, — C5: при n = 4 она догоняет C3 к 128k, при n = 6 — нет.",
 kind="desc")}
{blockb_table()}
""")

# ---- 4. quantitative results tables
S.append(f"""
<h2 id="s4"><span class="num">4</span>Основные количественные результаты (таблицы для Results)</h2>
<p>Четыре таблицы, по одной на cap — это и есть «главная таблица» тезиса; в тексте достаточно pooled-строки и одной таблицы стоимости.</p>
{pooled_table()}
{matrix_table(16000)}
{matrix_table(32000)}
{matrix_table(64000)}
{matrix_table(128000)}
""")

# ---- 5. statistics
S.append(f"""
<h2 id="s5"><span class="num">5</span>Статистика: только необходимое</h2>
<p>Данные парные, решётчатые, с тяжёлыми связками и нулями. Из этого следует: никаких t-тестов и ANOVA на средних; никаких смешанных моделей —
вопросы, на которые они бы ответили, здесь закрываются стратификацией (парность уже убирает межинстансную дисперсию). Пять инструментов, каждый под один вопрос.</p>

{table(["метод", "зачем", "какой вопрос", "что позволяет заключить", "статус"], [
 ["Percentile bootstrap по инстансам (10 000, seed 20260827) для средних клеток", "неопределённость каждой из 60 + 60 клеток без параметрических допущений", "насколько точна средняя satisfaction клетки", "интервал; не тест", badge("desc")],
 ["Парные bootstrap-CI для всех 10 пар × 3 полосы × 4 cap (замороженный contrasts.csv)", "полная матрица разностей на равных", "отличается ли пара архитектур в данной клетке", "описательно: CI не накрывает ноль — «в выборке различие есть»; без поправки на 120 клеток", badge("desc")],
 ["Замороженная конфирматорная семья при 64k: 5 контрастов × (IUT crossover + interaction), sign-flip permutation, Holm по 10 p", "единственный зарегистрированный тест RQ", "есть ли crossover по D; есть ли architecture × D", f"0 из 10 отвержений; минимальный Holm-p = {min_holm:.2f}; crossover не установлен ни для одной пары", badge("conf")],
 ["Friedman по 5 архитектурам на парных инстансах + Kendall's W + Nemenyi CD (Demšar 2006)", "омнибус «различимы ли пять систем вообще при этом бюджете» и размер эффекта", "architecture × budget: при каком cap ранжирование появляется", "W ≈ 0 при 16k, слабое при 32k, сильное при 64k/128k; группы внутри CD", badge("expl")],
 ["Парные разности между соседними cap (те же инстансы) с bootstrap-CI + доля «изменённых» прогонов", "предельная отдача от удвоения по архитектурам", "где активация, где плато", "точка активации и точка насыщения для каждой архитектуры", badge("expl")],
], caption="Что делать не нужно: регрессия satisfaction на (arch × cap × band × H × travel) — ковариаты сцеплены с полосой почти детерминированно (§2), модель не разделит D от H и tightness; стратифицированные таблицы §3.4 показывают ровно то, что модель могла бы, и честнее.")}

<h3>5.1 Замороженный конфирматорный результат</h3>
{frozen_table()}
<p>Читать ровно так, как записано в плане: crossover не установлен ни для одной из пяти пар; отсутствие отвержения не есть доказательство отсутствия crossover.
Единственный сигнал ниже 0.05 до поправки — interaction для C1→C2 (δ растёт с D: {summ['confirmatory']['contrasts'][0]['bands']['low']['mean_delta']:+.2f} / {summ['confirmatory']['contrasts'][0]['bands']['medium']['mean_delta']:+.2f} / {summ['confirmatory']['contrasts'][0]['bands']['high']['mean_delta']:+.2f}), после Holm {best_int['p_holm']:.2f}. Это можно упомянуть как направление, не как результат.</p>

<h3>5.2 Различимы ли пять систем: Friedman / Kendall's W / CD</h3>
{figure(13, "fig08_rank_cd.png", "Critical-difference diagrams per cap: mean rank of the five architectures over 150 paired instances",
 "Положение метки — средний ранг (1 = лучший); горизонтальная перекладина внизу соединяет архитектуры, чей средний ранг отличается меньше, чем на CD (Nemenyi, α = 0.05). W в заголовке — согласованность ранжирования между инстансами.",
 f"16k: W = {W[16000]:.2f}, все пять внутри одной перекладины — неразличимы. 32k: W = {W[32000]:.2f}, единственное разделение — C5 отстаёт; C1/C2/C3/C4 в одной группе. 64k: W = {W[64000]:.2f}, три яруса — {{C3, C4}} < {{C2, C1}} < C5. 128k: W = {W[128000]:.2f}, C3 один впереди, {{C4, C2, C5}} в одной группе, C1 отдельно позади. Ранжирование появляется с бюджетом, а не с D — по полосам картина та же (таблица ниже).",
 kind="expl")}
{friedman_table()}
<p class="small">Nemenyi при тяжёлых связках консервативен неравномерно; перекладины — ориентир, не доказательство. Для тезиса достаточно Friedman-p и W; CD-диаграмма — удобная одна картинка вместо десяти парных тестов.</p>
""")

# ---- 6. central findings
S.append(f"""
<h2 id="s6"><span class="num">6</span>Центральные findings</h2>
<p>Пять. Для каждого: что наблюдается / статистическое свидетельство / интерпретация / гипотеза — раздельно, чтобы в тексте не смешались.</p>

<div class="finding" id="f1">
<h3>F1. Бюджет — доминирующий фактор; ниже 64k архитектура не имеет значения</h3>
<dl class="grid">
<dt>наблюдение</dt><dd>При 16k все пять систем на нуле (максимум клетки {f2(NUM['headline']['block_a_16k_max'])}); при 32k все в диапазоне {f2(NUM['headline']['block_a_32k_range'][0])}–{f2(NUM['headline']['block_a_32k_range'][1])} с перекрывающимися CI, и номинально впереди одиночные C2/C1. Первое разделение — на 64k.</dd>
<dt>свидетельство</dt><dd>Friedman: W = {W[16000]:.2f} (p = {fr(16000,'all','p'):.2f}) при 16k; W = {W[32000]:.2f} при 32k; W = {W[64000]:.2f} при 64k. Парные CI: при 16k ни одна из 30 клеток-пар не отделяется от нуля; при 32k только пары с C5 и C1/C2→C4 (в минус).</dd>
<dt>интерпретация</dt><dd>При окне 32 768 токенов и thinking ON один вызов стоит 2–5k; 16k — это 10–12 вызовов, которых не хватает даже на сбор фактов через tools. Это свойство пары модель–задача, а не архитектур. Условия с зарезервированными фазами (C3-квоты, C4-квота, C5-трети) при 32k даже проигрывают: фиксированная структура «съедает» и без того недостаточный поиск — ровно то, что было записано до пилота как ожидание для C2/C3.</dd>
<dt>устойчивость</dt><dd>Высокая: повторяется во всех трёх полосах и во всех трёх размерах Block B (со сдвигом: при n = 4 «нуль» только на 16k).</dd>
<dt>для RQ</dt><dd>Порог, который есть в данных, — <b>бюджетный</b>, не сложностной.</dd>
</dl></div>

<div class="finding" id="f2">
<h3>F2. От 64k ранжирование стабильно и не зависит от D: C3 > C4 ≳ C2 > C1; C5 зависит от cap</h3>
<dl class="grid">
<dt>наблюдение</dt><dd>При 64k и 128k порядок средних одинаков в Low, Medium и High. C1→C3: +{cell('C3',64000,'low')-cell('C1',64000,'low'):.2f}/+{cell('C3',64000,'medium')-cell('C1',64000,'medium'):.2f}/+{cell('C3',64000,'high')-cell('C1',64000,'high'):.2f} при 64k и +{cell('C3',128000,'low')-cell('C1',128000,'low'):.2f}/+{cell('C3',128000,'medium')-cell('C1',128000,'medium'):.2f}/+{cell('C3',128000,'high')-cell('C1',128000,'high'):.2f} при 128k. Наклон по D у всех архитектур небольшой и одного знака.</dd>
<dt>свидетельство</dt><dd>Конфирматорно: 0/10 Holm-отвержений — ни crossover, ни architecture × D не установлены. Описательно: C1→C3 CI не накрывает ноль в 6 из 6 клеток 64k/128k; C4→C3 — только при 128k (3/3); C1→C2 — 6/6. Friedman по полосам при 64k: W = {fr(64000,'low'):.2f}/{fr(64000,'medium'):.2f}/{fr(64000,'high'):.2f} — одинаковая сила эффекта в каждой полосе.</dd>
<dt>интерпретация</dt><dd>Зарегистрированная гипотеза H1 («одиночный агент лучше на низкой сложности») не нашла региона: ни в Low D при n = 8, ни в Block B при n = 4–6, ни в dev-калибровках (три независимых попытки, все записаны). H2 в форме «преимущество MAS растёт с D» тоже не поддержана — преимущество есть, но <i>плоское</i>. Эффект полосы существует (Low легче для всех), но он не модулирует архитектуру.</dd>
<dt>гипотеза</dt><dd>Парная конфликтная плотность при фиксированном n = 8 и матчинге по O — слишком слабое изменение сложности для агента: то, что агент реально «чувствует», — размер контекста (n) и tightness, а они здесь либо зафиксированы, либо сцеплены с D в противоположную сторону (H). Ось D конструктивно валидна для solver-структуры, но не является осью трудности для LLM.</dd>
<dt>для RQ</dt><dd>Ответ на «есть ли порог сложности» — <b>не найден в исследованном диапазоне</b>; это полноценный отрицательный результат с прозрачной силой (50 инстансов на полосу, 4 cap).</dd>
</dl></div>

<div class="finding" id="f3">
<h3>F3. У каждой архитектуры свой бюджетный режим: активация, плато, насыщение</h3>
<dl class="grid">
<dt>наблюдение</dt><dd>C1: активируется на 32k–64k, плато {f2(cell('C1',64000))}→{f2(cell('C1',128000))}, при 128k тратит {pct(cell('C1',128000,'all','tokens')/128000)} cap и завершается добровольно в {pct(NUM['termination']['C1|128000'].get('agent_finish',0))} прогонов. C3/C4: активация на 64k (+{gain('C3','32k->64k'):.2f}/+{gain('C4','32k->64k'):.2f}); дальше C4 насыщается (+{gain('C4','64k->128k'):.2f}), C3 продолжает (+{gain('C3','64k->128k'):.2f}). C5: активация только на 128k (+{gain('C5','64k->128k'):.2f}), т.е. когда каждая из трёх траекторий получает ≈ {128000/3/1000:.0f}k — больше, чем C1 фактически тратит при 64k.</dd>
<dt>свидетельство</dt><dd>Парные разности между cap с CI (таблица §3.2); доля изменённых прогонов: C1 на шаге 64k→128k изменил {pct(gain('C1','64k->128k','all','exec_changed'))}, C3 — {pct(gain('C3','64k->128k','all','exec_changed'))}, C5 — {pct(gain('C5','64k->128k','all','exec_changed'))}.</dd>
<dt>интерпретация</dt><dd>«Равный бюджет» — это равный <i>потолок</i>, а не равная трата. C1 ограничен не бюджетом, а собственным решением остановиться (или пустым ответом) — дополнительные токены он не конвертирует. C3 конвертирует: квоты воркеров растут с cap, и каждый воркер продолжает поиск. Отсюда практическое правило: сравнение архитектур «при равном бюджете» осмысленно только в режиме, где хотя бы одна из них бюджетом ограничена; здесь это 32k–64k.</dd>
<dt>гипотеза</dt><dd>Плато C1 — это предел одной длинной траектории в одном контексте (деградация использования длинного контекста в смысле Tran &amp; Kiela), а не предел модели: та же модель в C3 на под-инстансах и в C5 при 128k доходит существенно выше.</dd>
</dl></div>

<div class="finding" id="f4">
<h3>F4. Координация в токенах бесплатна, в латентности — нет; C5 — худший способ потратить бюджет ниже 128k</h3>
<dl class="grid">
<dt>наблюдение</dt><dd>Токены C3/C1: {ratios['16000']['tokens_c3_over_c1']:.2f} (16k), {ratios['32000']['tokens_c3_over_c1']:.2f} (32k), {ratios['64000']['tokens_c3_over_c1']:.2f} (64k), {ratios['128000']['tokens_c3_over_c1']:.2f} (128k). Latency C3/C1: {ratios['64000']['latency_c3_over_c1']:.2f} при 64k, {ratios['128000']['latency_c3_over_c1']:.2f} при 128k. Эффективность (sat/1M realised tokens) при 64k: C3 {cell('C3',64000,'all','sat_per_1k')*1000:.1f}, C4 {cell('C4',64000,'all','sat_per_1k')*1000:.1f}, C2 {cell('C2',64000,'all','sat_per_1k')*1000:.1f}, C1 {cell('C1',64000,'all','sat_per_1k')*1000:.1f}, C5 {cell('C5',64000,'all','sat_per_1k')*1000:.1f}. Доля input у C3 ниже, чем у C1.</dd>
<dt>свидетельство</dt><dd>Средние по 150 прогонам; эффективность — ratio of means (описательно). Латентность — сумма времени вызовов на общем кластере, только относительное чтение.</dd>
<dt>интерпретация</dt><dd>Прямой ответ на вторую половину RQ: <b>выигрыш стоит координационных издержек</b>, потому что издержек в логических токенах нет — короткие изолированные контексты дешевле одного длинного. Цена появляется во времени при 128k (последовательные воркеры + критик) и в потере на отдельных инстансах (F5). C5 показывает, что «просто больше сэмплов» под равным бюджетом — не конкурент до тех пор, пока каждый сэмпл не получит полный рабочий бюджет одиночного агента; это уточняет Parmar et al. (Best-of-N сильнейший) до «при неограниченном бюджете».</dd>
</dl></div>

<div class="finding" id="f5">
<h3>F5. На уровне инстансов архитектуры частично комплементарны; декомпозиция имеет цену на отдельных задачах</h3>
<dl class="grid">
<dt>наблюдение</dt><dd>При 64k C3 проигрывает C1 на {tr[(64000,'C3')]['sat_lower']} из 150 инстансов, при 128k — на {tr[(128000,'C3')]['sat_lower']}; C2 и C4 не проигрывают C1 практически никогда ({tr[(64000,'C2')]['sat_lower']} и {tr[(64000,'C4')]['sat_lower']} при 64k). Oracle-of-5 (лучшая из пяти на каждом инстансе) даёт {o5['64000']['oracle_of_5']:.2f} при 64k и {o5['128000']['oracle_of_5']:.2f} при 128k против {o5['64000']['best_single_mean']:.2f} / {o5['128000']['best_single_mean']:.2f} у лучшей одиночной (C3). Преимущество C3 над C1 самое большое на clustered travel structure ({NUM['travel_tab']['64000|clustered']['C3']-NUM['travel_tab']['64000|clustered']['C1']:+.2f} при 64k) и самое малое на random ({NUM['travel_tab']['64000|random']['C3']-NUM['travel_tab']['64000|random']['C1']:+.2f}).</dd>
<dt>свидетельство</dt><dd>Парные счётчики (таблица §3.4); travel-структура не сбалансирована по полосам — только ассоциация.</dd>
<dt>интерпретация</dt><dd>Механизм цены записан заранее в дизайне C3: pool-restricted critic не может добавить человека, которого ни один воркер не предложил, а фиксированный географический split иногда разрывает пару, которую одиночный агент ставил вместе. Ассоциация с travel-структурой согласуется с этим: split по travel-матрице хорошо ложится на clustered и плохо — на random (где кластеров нет).</dd>
<dt>гипотеза</dt><dd>Преимущество декомпозиции зависит от того, насколько структура split совпадает со структурой ограничений задачи. Проверяемо на существующих данных только как ассоциация; для утверждения нужен сбалансированный по travel-структуре дизайн (future work).</dd>
</dl></div>

<h3>Неожиданные и отрицательные результаты, которые надо обсуждать явно</h3>
<ul>
<li><b>Нет региона, где одиночный агент лучше.</b> Проверялось четыре раза (Low D n = 8; dev-калибровки n = 4–6 при 8k–32k; n = 4 при 64k; held-out Block B). Отрицательный результат с фиксированной заранее интерпретационной рамкой (§3 amendment 2026-08-09: не редизайнить ось).</li>
<li><b>C5 «схлопывается» при ≤ 64k</b> — артефакт деления бюджета, предсказанный 128k-пробой на dev; при 128k C5 догоняет C2/C4 и в Low band почти C3.</li>
<li><b>Self-verification (C2) даёт скромный, но никогда не отрицательный прирост</b> — при том, что литература (Zheng 2024; Kambhampati 2024) ожидает вред от самокоррекции. Здесь вред структурно невозможен (validator-gated best_plan_so_far), поэтому измеряется только польза; это надо сказать прямо, иначе результат прочитают как опровержение литературы.</li>
<li><b>C4 ≈ C3 при 64k, C4 &lt; C3 при 128k.</b> При 64k декомпозиция ничего не добавляет к fresh-context critic; расхождение появляется, когда воркерам есть куда расти. Единственный намёк на «порог», и он бюджетный.</li>
<li><b>Satisfaction растёт с H</b>, т.е. диагностика «скрытой» сложности ведёт себя противоположно интуиции — потому что сцеплена с tightness. Это аргумент для Limitations, а не finding о H.</li>
</ul>
""")

# ---- 7. limitations
S.append(f"""
<h2 id="s7"><span class="num">7</span>Limitations, которые действительно важны</h2>
<ol>
<li><b>Одна модель, INT4 AWQ.</b> Все абсолютные уровни могут быть занижены квантованием; относительные — нет, но перенос на другие модели не показан (8B исключена: никогда не предлагает план). Ke et al.: сила sub-agent меняет знак MAS-эффекта — здесь это не варьировалось.</li>
<li><b>Ось D одномерна и сцеплена с H, tightness и overlap</b> (§2). Утверждения «эффект полосы» — это утверждения о пучке коррелированных свойств; «порог сложности» проверен только вдоль этого пучка.</li>
<li><b>Один дизайн MAS.</b> Фиксированный split k = 2 по travel-матрице, pool-restricted critic, без ревизионного раунда. C3-FI ablation (full-information critic) не запускалась, потому что триггер не сработал; вопрос «ограничивает ли pool» остаётся открытым, и F5 его касается.</li>
<li><b>Treatment fidelity C2.</b> На dev-прогонах verify/revise-стадии достигались примерно в половине прогонов при n = 8 (draft съедает бюджет; у C2 единственной нет резервированной фазы). Контраст C1→C2 измеряет «mandated verification, administered in part of runs».</li>
<li><b>Один прогон на клетку.</b> Обосновано детерминизмом гарнеса; но вариация от сэмплинга при другом seed не измерена, и «инстансная» дисперсия включает её.</li>
<li><b>Вторичные cap не зарегистрированы как конфирматорные.</b> Всё, что сказано про 16k/32k/128k и про architecture × budget, — описательно/exploratory. Регистрировался только 64k.</li>
<li><b>Latency</b> — на общем кластере, с последовательными воркерами; не сравнима с параллельной реализацией (Xu et al.: гомогенный workflow можно исполнить одним агентом с KV-cache reuse — логические токены здесь считаются иначе).</li>
<li><b>Бенчмарк генерированный</b>; NATURAL PLAN как external validation не запущен. Выводы — о семействе OPTW-инстансов с n ≤ 8.</li>
<li><b>C5 не оптимизирован по N</b> и по разбиению бюджета; «Best-of-3 при W/3» — одна точка, не кривая.</li>
</ol>
""")

# ---- 8. narratives
S.append(f"""
<h2 id="s8"><span class="num">8</span>Каркас Bachelorarbeit из этих результатов</h2>

<h3>Главный ответ, в одной фразе</h3>
<div class="callout"><p>При равном токен-бюджете на meeting planning с Qwen3-32B порог, разделяющий архитектуры, оказался <b>бюджетным, а не сложностным</b>:
ниже 64k все пять систем неразличимы (и одиночные номинально впереди), от 64k иерархическая MAS впереди во всех полосах конфликтной плотности и при всех размерах,
тратя при этом не больше токенов, чем одиночный агент; сложностного crossover в исследованном диапазоне нет, и единственный найденный «порог» для декомпозиции
(C4 → C3) тоже открывается бюджетом (128k).</p></div>

<h3>Results (порядок изложения, ~12–14 страниц)</h3>
<div class="narr"><ol>
<li><b>Completeness и структура выборки</b> (½ стр.): 198 × 5 × 4 = 3960, audit passed; таблица ковариат по полосам (H, tightness, overlap, travel) — честно показать сцепку. <i>Таблица</i>: §2-style covariate table.</li>
<li><b>Поверхность architecture × budget × band</b> (3 стр.): Fig. 1 (heatmaps) как открывающая картинка, Fig. 2 (curves per band); главная таблица — pooled + четыре cap-таблицы (§4), из них в текст pooled, остальные в приложение или одну сводную.</li>
<li><b>Зарегистрированный конфирматорный анализ</b> (2 стр.): таблица §5.1 и Fig. 3 (transposed) как её иллюстрация; вывод «crossover не установлен»; medium band descriptive.</li>
<li><b>Бюджетная динамика</b> (2–3 стр.): Fig. 4, таблица activation/plateau, таблица gains с долей изменённых прогонов; Friedman/W/CD (Fig. 13) как один exploratory омнибус.</li>
<li><b>Стоимость и эффективность</b> (2 стр.): Fig. 5, таблица стоимости, Fig. 11 (композиция токенов и termination) — здесь отвечает вторая половина RQ.</li>
<li><b>Уровень инстансов</b> (2 стр.): Fig. 7 (instance matrix), таблица «solved by / oracle-of-5», Fig. 8 (paired vs C1); стратификация по H и travel как таблицы.</li>
<li><b>Block B</b> (1–1½ стр.): Fig. 12 + таблица; формулировка «размер сдвигает точку активации влево, порядок не меняет».</li>
</ol></div>

<h3>Что — в основной текст, что — в приложение</h3>
{table(["в основной текст", "в приложение"], [
 ["Fig. 1, 2, 4, 5, 7, 8, 12, 13; pooled table; cost table; frozen confirmatory table; activation table", "Fig. 3 (transposed), 6 (validity/optimality surfaces), 9 (task space), 10 (full pairwise matrix), 11; четыре cap-таблицы полностью; Friedman по полосам; H × band; travel × arch; Block B полная таблица; contrasts.csv целиком; frozen report.md как есть"],
], caption="Ориентир: 8 фигур и 5 таблиц в Results; всё остальное — appendix с перекрёстными ссылками.")}

<h3>Discussion (порядок аргументов)</h3>
<div class="narr"><ol>
<li><b>Бюджетный порог вместо сложностного.</b> Свести F1 и F3: почему 32k — пол (окно 32k, thinking, стоимость вызова), почему C1 не растёт (добровольный stop, половина бюджета в остатке), почему C3/C4 растут (зарезервированные фазы конвертируют cap). Связать с Tran &amp; Kiela: их «single ≥ MAS при равном thinking-budget» здесь воспроизводится только <i>ниже</i> порога; выше — нет, и это ровно их же условие «MAS помогает, когда эффективное использование контекста одиночного агента деградирует».</li>
<li><b>Почему crossover по D не найден.</b> F2 + §2: ось конструктивно валидна для solver-структуры и не является осью трудности для агента; Amonkar/Zheng — деградация с числом ограничений есть у всех (наклон по D одинаков), но она не взаимодействует с архитектурой. Сказать прямо: «на этой оси, в этом диапазоне, при этом n — эквивалентность архитектур по форме зависимости от сложности».</li>
<li><b>Что даёт декомпозиция сверх критика.</b> C4 vs C3: при 64k ≈ 0, при 128k +0.2; связать с Ke et al. (Depth — контрпример; здесь задача последовательная, но split по географии делает под-задачи короче) и с F5 (цена split на отдельных инстансах, ассоциация с travel-структурой; L3 — partition, не FM-2.4).</li>
<li><b>Sampling baseline.</b> C5 и Parmar: Best-of-N сильнейший только когда N не делит бюджет; при равном бюджете — худший ниже 128k. Не оптимизировали N — сказать.</li>
<li><b>Self-verification.</b> C2 «никогда не вредит» — по конструкции (gate); литература про вред самокоррекции относится к системам без внешнего валидатора (Kambhampati: soundness должна приходить извне — здесь так и сделано).</li>
<li><b>Стоимость.</b> Токены vs латентность vs KV-cache (Xu; L4): логическая цена координации нулевая, физическая — до ×1.3 по времени при 128k.</li>
<li><b>Limitations</b> (§7) и что бы изменил в дизайне: варьировать n как основную ось при фиксированной D-полосе; сбалансировать travel-структуру; C3-FI; второй seed на подвыборке.</li>
</ol></div>

<h3>Итоговый ответ на RQ (формулировка для Conclusion)</h3>
<p>«Under an equal token budget, no complexity threshold was found in the studied range: the hierarchical system does not become
competitive <i>as conflict density rises</i>, it becomes competitive <i>as the budget rises</i> — from 64 000 tokens upward it leads in every
density band and at every instance size, while spending no more tokens than the single agent. Below that budget the five systems are
statistically indistinguishable. The gain is worth the coordination overhead in tokens, which is nil, and costs up to 30% in latency at the
largest budget; on roughly one instance in ten the fixed decomposition loses a plan the single agent finds.»</p>
""")

# ---- 9. literature map
S.append(f"""
<h2 id="s9"><span class="num">9</span>Как результаты ложатся на использованную литературу</h2>
{table(["источник", "что утверждает / предсказывает", "что показали данные", "как использовать"], [
 ["Tran &amp; Kiela 2026", "при равном thinking-budget single ≥ MAS (DPI); MAS помогает при деградации контекста одиночного агента", "single ≥ MAS только при ≤ 32k; от 64k MAS впереди, C1 останавливается с неиспользованным бюджетом", "расширение на tool-based planning: их null держится в режиме бюджетного голода и ломается там, где одиночный контекст перестаёт конвертировать бюджет"],
 ["Ke et al. 2026 (MAS-Orchestra)", "MAS не выигрывает на Depth; выигрыш на краю компетентности sub-agent", "выигрыш есть на последовательной задаче, но через географический split (короче цепочки на воркера), и он плоский по D", "L1: объяснить, почему decomposition по локациям превращает depth в parallel"],
 ["Tang et al. 2025", "выигрыш debate-MAS растёт со сложностью (depth доминирует)", "у non-redundant иерархии выигрыш не растёт с D", "L2: их механизм (redundancy) в C3 отсутствует; результат согласуется, не противоречит"],
 ["Xu et al. 2026 (OneFlow)", "гомогенный workflow можно исполнить одним агентом; KV-cache экономит", "один агент (C1) фактически не исполняет такой поиск; логический ledger не видит KV-cache", "L4: разделить логическую и физическую стоимость"],
 ["Parmar et al. 2025 (PlanGEN)", "Best-of-N — сильнейший вариант на NATURAL PLAN", "при равном бюджете Best-of-3 худший до 128k", "L5: сильнейший при неограниченном бюджете; sampling-baseline закрыт, N не оптимизирован"],
 ["Kambhampati et al. 2024 (LLM-Modulo)", "LLM не самоверифицируется; soundness — снаружи", "внешний validator — единственный источник валидности; C2/критики не могут вредить", "объясняет, почему self-verification здесь «не вредит» и почему C4 ≈ C3 при 64k (критик даёт валидность, не длину)"],
 ["Cemri et al. 2025 (MAST)", "MAS умирают от termination/verification", "termination harness-owned; C3 при 128k 100% agent_finish; цена — missing-candidate (partition, не FM-2.4)", "trace-level error analysis для F5"],
 ["Zheng et al. 2024 (NATURAL PLAN)", "< 10% при ≥ 8 людях; self-correction вредит", "при n = 8 одиночный агент ~0.3–0.4, MAS ~0.8 при 128k; self-correction не вредит по конструкции", "внешняя точка отсчёта; оговорка про validator gate"],
 ["Amonkar et al. 2025", "деградация с числом constraints у всех подходов (Qwen3-32B на Meeting Planning)", "наклон по D одинаков у всех пяти", "подтверждение направления; отсутствие взаимодействия — новое"],
 ["Xie et al. 2024 (TravelPlanner)", "агенты не держат глобальные constraints; ReAct недостаточен", "C1 действительно проигрывает всем структурированным вариантам от 64k", "мотивация baseline"],
], caption="Ни одно из утверждений литературы не «опровергается»; несколько уточняются условием равного бюджета и наличием внешнего валидатора.", cls="dense")}
""")

# ---- 10. appendix pointers & extra ideas
S.append(f"""
<h2 id="s10"><span class="num">10</span>Дополнительно: идеи и что я бы ещё сделал (по убыванию ценности)</h2>
<ol>
<li><b>«Effective budget» как ось вместо cap</b> — на графике Fig. 5 архитектуры уже сравниваются по реально потраченным токенам; в тексте это снимает возражение «равный cap ≠ равная трата». Достаточно одной фразы и одной фигуры.</li>
<li><b>Второй seed на подвыборке</b> (например, 30 инстансов × 5 × {{64k}}) — единственный способ отделить межинстансную вариацию от сэмплинговой; дёшево, и делает Limitation 5 количественной.</li>
<li><b>C3-FI на 64k/128k для High band</b> — закрывает вопрос «pool ограничивает?» и напрямую объясняет F5. Триггер ablation формально не сработал, поэтому только как post-hoc exploratory.</li>
<li><b>Trace-level MAST-разметка</b> для 23 инстансов, где C3 проиграл C1 при 64k: это готовая error analysis на 2–3 страницы и самый убедительный материал для Discussion про цену декомпозиции.</li>
<li><b>Не делать:</b> регрессии по ковариатам (сцепка), новые семьи тестов на вторичных cap (нечего регистрировать постфактум), 3D в тексте.</li>
</ol>

<h3>Файлы</h3>
<ul>
<li>Фигуры этого отчёта (PNG) лежат рядом с ним в <code>figs/</code>; скрипт, который их строит из committed CSV, — <code>make_figures.py</code> (pandas/matplotlib/scipy; bootstrap seed 20260827, тот же, что у замороженного анализатора).</li>
<li>Замороженные артефакты не менялись: <code>results/analysis/heldout/</code>, <code>results/exports/heldout/</code>.</li>
</ul>
""")

TOC = """
<nav class="toc" aria-label="Contents">
<p class="eyebrow">Содержание</p>
<ol>
<li><a href="#s1">1 · Что за эксперимент</a></li>
<li><a href="#s2">2 · Структура данных</a></li>
<li><a href="#s3">3 · Данные</a></li>
<li class="sub"><a href="#s3a">3.1 поверхность</a></li>
<li class="sub"><a href="#s3b">3.2 бюджетная динамика</a></li>
<li class="sub"><a href="#s3c">3.3 стоимость</a></li>
<li class="sub"><a href="#s3d">3.4 инстансы и пространство задач</a></li>
<li class="sub"><a href="#s3e">3.5 парные контрасты</a></li>
<li class="sub"><a href="#s3f">3.6 токены и termination</a></li>
<li class="sub"><a href="#s3g">3.7 Block B</a></li>
<li><a href="#s4">4 · Таблицы результатов</a></li>
<li><a href="#s5">5 · Статистика</a></li>
<li><a href="#s6">6 · Central findings</a></li>
<li><a href="#s7">7 · Limitations</a></li>
<li><a href="#s8">8 · Каркас Results / Discussion</a></li>
<li><a href="#s9">9 · Литература</a></li>
<li><a href="#s10">10 · Идеи и файлы</a></li>
</ol>
</nav>
"""

HEAD = """<title>Five Architectures, Four Budgets</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
"""

page = f"""{HEAD}<style>{CSS}</style>
<div class="page">
{TOC}
<main>
{''.join(S)}
</main>
</div>
"""
(DEST / "index.html").write_text(page, encoding="utf-8")
# also a standalone full-html copy for local opening
(DEST / "report_standalone.html").write_text("<!doctype html><html lang=\"ru\"><head>" + HEAD + "<style>" + CSS + "</style></head><body>" +
                                             f'<div class="page">{TOC}<main>{"".join(S)}</main></div></body></html>', encoding="utf-8")
print("written", DEST / "index.html", len(page) / 1e6, "MB")
