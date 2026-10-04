# Appendix figures to the frozen confirmatory analysis

These five figures are **specific to the 64 000 cap or to the C1–C3 pair**. They belong to
the frozen confirmatory analysis in this directory and are **not part of the descriptive
results**, which cover all four budgets and all five architectures on equal footing and live
in `../../heldout_full_matrix/`.

The separation is the decision of `THESIS_DECISIONS.md` §5, amendment 2026-09-02: no cap and
no pair of conditions may appear to hold a special place in the descriptive presentation.
Nothing here is withdrawn — the confirmatory analysis stands, with its registered scope and
its registered result — but it is reported as one analysis among the material, not as the
frame the descriptive results are read through.

Canonical format is **vector PDF**; the `.png` next to each is a preview only. No embedded
title and no on-figure method note: those belong in the Overleaf `\caption{}`.

Regenerate from committed CSVs — no cluster, no new statistics:

```
uv run --with matplotlib --with numpy python -m scripts.plot_heldout_figures   # F1, F2, F7
uv run --with matplotlib --with numpy python -m scripts.build_results_data     # I, J
```

Inputs: `results/exports/heldout/runs.csv`, `results/analysis/heldout/contrasts.csv` and
`results/analysis/heldout/summary.json`. Descriptive bootstrap where error bars appear:
percentile interval of the mean, resampling instances, 10 000 resamples, seed 20260827 —
the same construction the frozen analyser uses. **No hypothesis test is computed by either
script.**

**PDF bytes are not a reproducibility criterion.** A re-render on another machine can differ
by a few hundred bytes at an identical matplotlib version, because PDF packaging — font
subsetting in particular — depends on the surrounding fonttools and Pillow builds. The
rendering is unaffected: the PNG previews regenerate byte-identically. Verify against the
source data, the numbers and the PNGs, and do not regenerate PDFs merely to make bytes
agree. Toolchain last used: Python 3.12.13, matplotlib 3.11.1, numpy 2.5.2, Pillow 12.3.0,
fonttools 4.64.0 — the same note and table are in
`../../heldout_full_matrix/figures/README.md`.

| file | shows |
|---|---|
| `F1_satisfaction_by_condition_band_64k` | mean satisfaction per architecture × band at cap 64 000 |
| `F2_registered_contrasts_forest_64k` | the five registered contrasts at 64 000 as δ_low, δ_high with 95% CIs |
| `F7_c1_vs_c3_crossover_delta` | Δ = S_C1 − S_C3 against complexity band, one line per cap, zero line |
| `I_frozen_forest_64k` | the five registered contrasts at 64 000 from the frozen output, with the frozen intervals |
| `J_c1_minus_c3_delta` | Δ = S_C1 − S_C3 against band, one line per cap, from the frozen `contrasts.csv` |

## Captions (method and standing)

- **F1.** Mean satisfaction by architecture (C1–C5) and complexity band at the 64 000 cap
  (n = 8, 50 instances per band). Error bars are 95% percentile bootstrap intervals of the
  mean, descriptive; no confirmatory claim attaches to a single bar.
- **F2.** The five pre-specified architectural contrasts at cap 64 000, each as a paired
  mean difference δ = satisfaction(later) − satisfaction(earlier) in the Low and High bands
  (95% bootstrap CI, from the frozen analysis output). A crossover requires δ_low < 0 and
  δ_high > 0 for the same contrast; no contrast meets both conditions, and none of the ten
  confirmatory hypotheses rejected after Holm correction. The interval shown is descriptive
  and is not the confirmatory test.
- **F7.** Difference in mean satisfaction between the single ReAct agent and the
  hierarchical MAS, Δ = S_C1 − S_C3 (exposé convention: Δ > 0 = single agent ahead), against
  complexity band, one line per equal-budget cap (n = 8; error bars are the frozen
  descriptive 95% bootstrap intervals, re-signed). The hypothesised crossover would be a
  single line moving from Δ > 0 at Low to Δ < 0 at High; no line does.
- **I.** The five registered contrasts at the 64 000 cap: for each, δ_low and δ_high with
  the frozen descriptive 95% bootstrap intervals and a vertical line at zero.
- **J.** Δ = S_C1 − S_C3, the arithmetic sign conversion of the frozen `C1->C3` rows,
  against complexity band, one line per cap, with the frozen descriptive intervals
  re-signed. No verbal annotations.

*A single figure describes this sample. Establishing an ordering would require a test, and
no test is added here.*
