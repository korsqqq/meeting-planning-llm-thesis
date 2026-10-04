# Descriptive figures — the full experimental matrix

Every figure here covers **all four budget caps and all five architectures on equal
footing**, with Block A resolved by density band and Block B resolved by task size. This
is the figure set of the main descriptive results, fixed by `THESIS_DECISIONS.md` §5,
amendment 2026-09-02.

Figures specific to one cap or to one pair of conditions are **not here**. They are the
appendix of the frozen confirmatory analysis, in `../../heldout/figures/`.

Canonical format is **vector PDF**; the `.png` next to each is a preview. No embedded title
and no on-figure method note: those belong in the Overleaf `\caption{}`.

## Regenerating

```
uv run --with matplotlib --with numpy python -m scripts.build_results_data     # B, C, D, E, H, K, L, M
uv run --with matplotlib --with numpy python -m scripts.plot_heldout_figures   # F3-F6
uv run --with matplotlib python -m scripts.build_contact_sheet                 # CONTACT_SHEET
```

Sources: `results/exports/heldout/runs.csv`, `results/exports/heldout/aggregates.csv` and
`results/analysis/heldout/contrasts.csv`. Before writing anything, `build_results_data`
asserts that the per-cell means it computes reproduce the committed `aggregates.csv`.
Descriptive bootstrap where error bars appear: percentile interval of the mean, resampling
instances, 10 000 resamples, seed 20260827. **No hypothesis test, p-value, correction or
decision about significance is computed by any of these scripts.**

### How reproducibility is checked, and what the check does not cover

The content check is the numeric one: the `aggregates.csv` assertion above, plus the PNG
previews, which regenerate **byte-identically**.

**PDF bytes are not a reproducibility criterion.** The same figure re-rendered on another
machine can differ by a few hundred bytes at an identical matplotlib version, because the
PDF packaging — font subsetting in particular — depends on the surrounding fonttools and
Pillow builds. The rendering is unaffected, which is what the byte-identical PNGs
demonstrate. So verify figures against the source data, the numbers and the PNGs, and do
not treat a PDF byte difference as a discrepancy or regenerate PDFs merely to make bytes
agree.

The toolchain these artifacts were last built with, recorded so a difference can be
attributed rather than guessed at:

| package | version |
|---|---|
| Python | 3.12.13 |
| matplotlib | 3.11.1 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |
| fonttools | 4.64.0 |

## The main set

One figure per question. `CONTACT_SHEET.png` tiles the whole set for judging it as a set.

| file | what is plotted |
|---|---|
| `F3_satisfaction_vs_budget_bands` | Mean satisfaction against the budget cap (log2 x), one panel per band, one line per architecture, with descriptive bootstrap intervals. The headline surface. |
| `B_satisfaction_by_complexity` | Mean satisfaction against band, one panel per cap, one line per architecture. The same surface transposed, so the band effect at a fixed budget is read directly. |
| `C_tokens_by_arch_budget` | Mean tokens per run against the cap, one line per architecture, over the 150 Block A runs per cap. |
| `D_calls_by_arch_budget` | Mean model calls against the cap, one line per architecture. |
| `E_latency_by_arch_budget` | Median `latency_seconds` against the cap; whiskers span the 25th–75th percentile. |
| `F4_efficiency_satisfaction_vs_tokens` | Mean satisfaction against mean tokens actually spent, one panel per cap, with descriptive intervals. Upper-left is more efficient. |
| `H_optimality_by_arch_budget` | Optimality rate against the cap, one line per architecture. |
| `L_efficiency_by_budget` | Satisfaction per 1000 tokens against the cap, one panel per band. Ratio of means, the definition of RESULTS_DATA §12. |
| `F5_validity_and_termination_diagnostics` | Valid-non-empty rate and non-voluntary-stop share against the cap. Diagnostics. |
| `F6_block_b_satisfaction_vs_budget` | Block B satisfaction against the cap, one panel per size (n = 4, 5, 6). |
| `M_block_b_metrics_by_size` | Block B, the six metrics besides satisfaction: rows are metrics, columns are n = 4, 5, 6. The size axis is never averaged away. |

## Supplementary

| file | what is plotted |
|---|---|
| `K_pairwise_matrix_heatmap` | All ten pairwise contrasts × four caps, one panel per band; each cell is the mean paired difference δ = satisfaction(later) − satisfaction(earlier), annotated, on a diverging scale centred at zero. |
| `CONTACT_SHEET` | Every figure above on one sheet. Not a result; a tool for judging the set. |

`K` is dense by construction — ten pairs at four budgets in three bands is 120 numbers — and
exists so the complete matrix is visible in one place rather than because it reads well at
figure scale. The readable views of the same material are F3, B and L. The five contrasts of
amendment B2 keep their registered orientation in `K` but are not marked out: in the
descriptive presentation the ten pairs are shown together.

## Figures that were removed, and why

Two scripts had grown overlapping series. The contact sheet made the overlap visible, and
three figures were true duplicates of another figure that showed the same thing with more
information. They are deleted rather than kept for the sake of existing references:

| removed | superseded by | reason |
|---|---|---|
| `A_satisfaction_by_budget` | `F3_satisfaction_vs_budget_bands` | identical surface; F3 also carries the descriptive bootstrap intervals |
| `F_satisfaction_vs_tokens` | `F4_efficiency_satisfaction_vs_tokens` | identical points; F4 also carries the intervals |
| `G_valid_nonempty_by_arch_budget` | `F5_validity_and_termination_diagnostics` | F5's left panel is G; F5 also carries the non-voluntary-stop share |

## Captions (method and standing)

- **F3.** Mean satisfaction against the token budget cap, by architecture and complexity
  band (n = 8, 50 instances per cell; 95% bootstrap error bars, descriptive).
- **B.** Mean satisfaction against complexity band, one panel per budget cap.
- **C, D, E.** Realised cost per run — tokens, model calls and latency — against the budget
  cap. Read beside F3; neither is a test.
- **F4.** Mean satisfaction against mean tokens actually spent, one panel per cap. Upper
  left is more efficient.
- **H.** Optimality rate: the share of runs whose plan reached the oracle optimum.
- **L.** Satisfaction per 1000 tokens, resolved by band. A derived descriptive quantity, not
  an outcome the design controls.
- **F5.** Structural diagnostics: the valid-non-empty rate — the informative success rate,
  since every final plan is valid by construction and an empty plan is valid — and the share
  of runs that stopped by budget exhaustion or abort rather than a voluntary finish.
- **F6, M.** Block B (n = 4, 5, 6; 16 instances per size). `n` is a generator parameter and
  never a complexity level, so this block forms no confirmatory contrast and is reported
  separately **by size**: the point of the block is what changes as the task gets smaller,
  which pooling would hide. F6 gives satisfaction by size; M gives the other six metrics,
  one column per size.
- **K.** The complete pairwise matrix, supplementary.

*Every figure here describes this sample. An ordering visible in one of them is a
description, not a finding; establishing it would require a test, and none is computed.*
