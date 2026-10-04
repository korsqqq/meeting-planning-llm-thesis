# SUPERSEDED — initial power model

**Status: SUPERSEDED.** The initial power model omitted structural no-plan zeros and did not
include the `sigma = 0` boundary.

This directory holds the first power simulation, specified in `fedbdb9` and run on
2026-08-09 by `scripts/power_iut.py`. It selected **N = 35** per band. That number is **not
adopted**; it is kept because the reasoning that replaced it only makes sense next to what
it replaced.

## What it produced

Selected N = 35. Worst target-effect cell: base mean 0.50, `sigma = 0.5`, all optimum 3 —
joint power 0.834, Monte-Carlo interval [0.818, 0.850], confirmed at 0.823 with the
registered 10 000 bootstrap resamples. No cell required escalation. Analytic check of that
cell agreed: 0.820.

Other scenarios at N = 35: weak effect worst 0.199, strong 1.000, asymmetric 0.469.

## Why it is superseded

**No structural failure mode.** The outcome model generated achieved meetings from a
binomial around a moderate probability and had no way to represent a run that produces no
plan at all. In the formal pilot that was the dominant source of zeros: at cap 64 000 only
0.467 of C1 runs held any valid plan, so more than half scored zero not by planning badly
but by never proposing. Omitting that mode is not conservative in a known direction — it
changes both the variance and the tie structure, which is what governs power here.

**The binding case sat on the edge of the grid.** The worst configuration was the smallest
`sigma` in the grid, and this is systematic rather than accidental: larger heterogeneity
pushes the success probability toward 0 and 1, where the binomial variance is smaller and
the paired difference is less noisy. The true worst case is therefore `sigma → 0`, which the
grid did not contain. A hand check at `sigma = 0` put the joint power near 0.80, so the
selection rested on a boundary the grid never evaluated.

**No Type-I validation.** The percentile bootstrap decision rule was never checked against a
boundary null, so nothing established that the procedure holds its nominal level on data
this discrete at this sample size.

## What replaced it

`results/analysis/power_iut_v2/`, specified before it ran, keeping the latent-logistic model
and adding: `sigma = 0` in the grid, a hurdle regime for structural no-plan zeros with the
target effect kept unconditional, and mandatory Type-I validation at two boundary nulls with
a stop rule if the bootstrap proves anticonservative.
