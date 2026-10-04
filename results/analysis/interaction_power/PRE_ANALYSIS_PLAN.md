# Pre-analysis plan — crossover and interaction power

**Frozen 2026-08-27, before the computation was run.** No number in this file is a
result. It is committed separately from any output so that what was decided in
advance is separable from what was found.

Companion to `../empirical_power/` (main-effect calibration and power, 2026-08-26)
and to the amendment of 2026-08-10 that replaced the synthetic power model with
estimates from real development runs.

---

## 0. Standing constraints

| | |
|---|---|
| Data | Development runs only: 60 `n = 8` instances at cap 64000, fully paired across C1–C5, 20 per band. Plus the 12-instance 128k probe where stated. |
| Held-out | Seeds `100000+` are **not read, not generated, not inspected**. None exist. |
| Sample sizes examined | `N = 50, 60, 65` per band, and nothing else. |
| Test selection | By Type-I calibration and pre-computed power **only**. Never by which test yields a smaller p-value on development data. |
| Reporting | Every power figure is a **plug-in estimate**: the resampling population is 20 observed differences per band. Resampling 20 points up to 65 redistributes information, it does not create any. Uncertainty must be reported alongside every point estimate. |

---

## 1. Sign convention

    δ_b = satisfaction(C3) − satisfaction(C1)   within band b

This is the **negation** of the exposé's `delta = S_C1 − S_C3`. Both appear in the
project record; every number produced under this plan uses the δ above, and says so.

Expected crossover, in this convention:

    δ_low < 0     and     δ_high > 0

that is, the single agent ahead where structure is sparse, the hierarchy ahead
where it is dense.

---

## 2. What the development data already says about direction

Recorded here because it determines the method, not because it is a result of this
analysis. In the development runs δ is **positive in all three bands**
(+0.379 low, +0.242 medium, +0.408 high): C3 is ahead everywhere, and the
low-complexity sanity check fails. This is already in the record.

**Consequence.** Crossover power cannot be estimated by resampling the observed
differences, because the observed pattern contains no crossover: a plug-in estimate
would return power ≈ 0 by construction and would say nothing about the design's
ability to detect a crossover that exists. Crossover power is therefore computed
under an **assumed alternative**, with the development data supplying only the noise
structure — spread, tie mass, zero mass and the discreteness of the satisfaction
lattice.

---

## 3. Construction of the assumed crossover alternative

`satisfaction = achieved / O` with `O ∈ {3, 4}`, so differences live on a coarse
lattice. Adding a constant shift would move the distribution off that lattice and
quietly change the tie structure that is the whole reason this analysis exists.

Instead, the alternative is built by **sign reversal**, which preserves the lattice,
the magnitudes, the tie mass and the zero mass exactly:

| component | construction |
|---|---|
| `δ_high` | resample the observed high-band differences unchanged (already positive) |
| `δ_low` | resample the observed low-band differences **negated** (positive → negative) |
| `δ_medium` | resample the observed medium-band differences, negated or unchanged per the scale grid |

**Effect-size sensitivity.** The above is scale 1.0 — a crossover as large as the
effect actually observed. Power is also reported at scales **0.75 and 0.5**, applied
by thinning: with probability `1 − s` an individual difference is replaced by 0.
Thinning keeps every non-zero value on the lattice and shrinks the mean by exactly
`s`, at the cost of raising the tie rate — which is the conservative direction.

---

## 4. Hypotheses and tests

### 4.1 Primary architecture contrast

**C1 vs C3.** C2, C4 and C5 are secondary throughout and are reported without
inflating or deflating any primary claim.

### 4.2 Primary cap

**One primary cap: 64000.** It is the only cap with a complete five-condition
paired development set, and the registered C1 budget gate showed 8k/16k/32k sit at
or near the floor at `n = 8`. **128000 is a pre-declared secondary cap**; its
results are reported and never pooled with the primary. Because there is exactly
one primary cap, no correction across caps arises for the primary claim.

### 4.3 Crossover — intersection-union test

Evidence of crossover requires **both** directional components to pass at α = 0.05,
one-sided in the pre-declared direction:

    H0_low  : δ_low  ≥ 0        rejected in favour of δ_low  < 0
    H0_high : δ_high ≤ 0        rejected in favour of δ_high > 0

Crossover is declared only if both reject. An intersection-union test has size
≤ α **without multiplicity correction**, because the rejection region is the
intersection: that property is the reason for choosing this form.

**A significant `Architecture × D` interaction is NOT accepted as evidence of
crossover.** An interaction can be significant with δ of one sign throughout — which
is precisely the pattern the development data shows.

### 4.4 Interaction — High vs Low contrast

    Δ = δ_high − δ_low

tested two-sided at α = 0.05. This measures whether the architectural effect
*changes* with D, which is a weaker and different claim than a change of sign.

### 4.5 Medium band

Used to describe the **shape and location** of the transition. No definition of
crossover involving the medium band may be introduced after seeing results.

### 4.6 Candidate tests

Both are exact under the paired symmetry null by construction, so neither can
repeat the percentile bootstrap's calibration failure:

* `perm` — sign-flip permutation test on the mean difference
* `sign` — exact binomial sign test on non-tied pairs

The percentile bootstrap is **excluded as primary** on the evidence of
`../empirical_power/`: it exceeded nominal 0.05 in six of seven cells, worst where
ties were heaviest. It may still be reported as a descriptive interval.

Type-I is re-measured for both under the sign-flip null at each `N`, and both are
required to hold α before either may be selected.

---

## 5. Multiplicity policy

| family | members | policy |
|---|---|---|
| Primary crossover | the IUT | none required — IUT size ≤ α by construction |
| Primary interaction | `Δ = δ_high − δ_low` | single test, no correction |
| Per-band primary | `δ_low`, `δ_medium`, `δ_high` for C1 vs C3 | Holm across the three bands |
| Secondary | C2, C4, C5 contrasts | Holm within the secondary family; never pooled with primary |

Power is reported both uncorrected and after Holm, so the cost of the correction is
visible rather than assumed.

---

## 6. Uncertainty of the power estimates

Each band supplies 20 observed differences. Every power figure is therefore a
plug-in estimate whose own sampling error must be shown.

**Method.** Nested resampling. Draw a pseudo-population of 20 differences by
resampling the observed 20 with replacement, estimate power from that
pseudo-population, and repeat. Report the median and the 10th–90th percentile band
of the resulting power distribution.

A point estimate is never reported without that band. Where the band spans the
decision threshold, the plan requires that to be stated in those words rather than
resolved by the point estimate.

---

## 7. H sensitivity

The higher-order gap `H` is systematically unbalanced across bands; this is an
existing recorded limitation, not something this analysis repairs. Pre-declared
sensitivity: repeat the primary IUT and the High-vs-Low contrast within strata of
`H` (`H = 0` against `H ≥ 1`) and report whether the conclusion is unchanged.
Declared as sensitivity only — it is not a separate confirmatory claim, and it
cannot promote or demote the primary result.

---

## 8. Decision rule for N — fixed before the table exists

Choose the **smallest** `N ∈ {50, 60, 65}` satisfying **both**:

1. **Primary per-band power.** For C1 vs C3, power ≥ 0.80 in every band after Holm,
   evaluated at the **10th percentile** of the nested-resampling uncertainty band,
   not at the point estimate.
2. **Crossover power.** IUT power ≥ 0.80 under the assumed crossover alternative at
   scale 1.0, again at the 10th percentile.

If no `N` in the set satisfies both, the shortfall is reported explicitly with the
`N` that would be required, and **the largest value is not silently adopted**. A
design that cannot reach the stated power at any admissible `N` is a finding about
the design, and the response is a recorded decision, not an adjustment of the rule.

The band capacity measured on the development calibration pool was 65 per band
(41 at `O = 3`, 24 at `O = 4`). Whether the held-out pool supports the chosen `N`
is unknown until that pool is built, which cannot happen before this plan closes.

---

## 9. What this analysis may not do

* read, generate or inspect any held-out seed
* choose a test by its p-value on development data
* introduce a crossover definition not stated in section 4 above
* report a power point estimate without its uncertainty band
* adopt an `N` outside `{50, 60, 65}`

---

# AMENDMENT — 2026-08-27

Recorded **before any output of this analysis was read**. The computation had not
been started when these three defects were raised; nothing here is a reaction to a
number.

## A1. One estimand, and the sign test is demoted

The plan above listed `perm` and `sign` as two candidate tests to be compared on
power. That was wrong, and it hid an estimand problem behind a test-selection
question.

**The primary estimand is fixed here: the mean paired difference `δ_b`.**

A sign-flip permutation test on the mean targets that estimand. The binomial sign
test does not — it targets `P(δ > 0)`, a different functional that coincides with
the mean only under conditions this data does not supply. Two tests of two
different hypotheses cannot compete on power, and selecting between them by power
would amount to selecting the estimand by power, which inverts the order this
project works in.

Consequently:

* **Primary test: sign-flip permutation on the mean.** Sole test of the primary
  estimand, for the IUT components and for `Δ`.
* **Sign test: sensitivity only.** Reported as robustness on a related functional,
  explicitly labelled as answering a different question. It may not promote,
  demote or replace a primary result, and it plays no part in the `N` rule.
* The percentile bootstrap remains excluded as primary and may appear only as a
  descriptive interval.

This also disposes of an observation flagged on 2026-08-26 — that the sign test
appeared more powerful than the permutation test in the medium band. That is no
longer a comparison the design makes.

## A2. The medium band is descriptive, and the contradiction is removed

Section 4.5 called the medium band descriptive while section 5 put it in the Holm
family and section 8 let it drive `N`. Those cannot all hold. Resolved as follows.

**The medium band is descriptive.** It receives an effect estimate and an interval,
no confirmatory test, no membership of any Holm family, and **no role in the `N`
rule**. Letting the band with the smallest observed effect dictate the sample size
for a claim that is defined over Low and High would size the experiment for a
question it does not ask.

The primary confirmatory family is therefore exactly two members:

| # | test | correction |
|---|---|---|
| 1 | IUT crossover: `δ_low < 0` and `δ_high > 0` | none needed — IUT size ≤ α by construction |
| 2 | `Δ = δ_high − δ_low`, two-sided | — |

Holm is applied **across those two primary tests**. `δ_low` and `δ_high` are the
IUT's own components and are reported as estimates with intervals, not as separate
confirmatory tests; doing both would count the same evidence twice. Section 5's
three-band Holm row is superseded by this table. C2, C4 and C5 remain a separate
secondary family with Holm inside it.

**Section 8 is amended to match.** Choose the smallest `N ∈ {50, 60, 65}` such that,
at the 10th percentile of the nested-resampling band, under the assumed crossover
alternative at scale 1.0, and after Holm across the two primary tests:

1. IUT crossover power ≥ 0.80, and
2. power for `Δ` ≥ 0.80.

The medium band does not enter this rule. The refusal to silently adopt the largest
admissible `N` stands.

## A3. What the permutation test actually assumes

Section 4.6 said both candidates were "exact under the paired symmetry null by
construction". That is true but was stated loosely enough to be read as a stronger
claim than it is, so it is made precise here.

**The null for which the sign-flip test is exact** is that each paired difference is
distributed symmetrically about zero. Under that null the `2^n` sign assignments are
equiprobable, so the reference distribution is the exact conditional distribution of
the statistic and the test holds its level in finite samples.

**It is not a distribution-free exact test of the mean.** Symmetry about zero
implies a mean of zero; a mean of zero does not imply symmetry. If the differences
are asymmetric with mean zero, the sign-flip null is false, the equiprobability
argument does not apply, and the level is no longer guaranteed by construction —
validity there is asymptotic, not exact. Satisfaction differences are bounded,
lattice-valued and can be skewed, so symmetry is a substantive assumption about
this data and not a formality.

Stated consequences, all pre-declared:

* The primary claim is made **against the symmetry null**, and will be worded that
  way rather than as an unqualified statement about the mean.
* The one-sided IUT components use the same sign-flip reference distribution with a
  one-tailed statistic.
* The Type-I rates already reported in `../empirical_power/` were measured under the
  sign-flip null. That is the null for which `perm` is exact, so those figures
  validate the percentile bootstrap against a null the bootstrap's own target
  implies — a fair comparison — but they do **not** establish `perm`'s level under a
  mean-zero-but-asymmetric null. No such claim is made.
* **Added pre-declared diagnostic:** report the skewness of the observed differences
  per band alongside the power table. Where asymmetry is material, that is stated as
  a limitation of the primary test rather than repaired after the fact.

## A4. Order of operations

This amendment is committed before the interaction-power computation is run. No
output of that computation had been produced or read when it was written. Held-out
seeds remain untouched.

---

# AMENDMENT 2 — 2026-08-27, simplification

Two things in the plan above were over-built for a bachelor thesis and are
withdrawn here. This amendment replaces them and closes the pre-registration.
No held-out seed has been inspected, and none of what follows is a reaction to
held-out data.

## B1. The `N` decision rule is withdrawn; `N = 50`

`N = 50` per `n = 8` D-band, as already registered in the frozen held-out design
of 2026-08-26. It was the main-design target before any of this analysis existed.

The rule "smallest `N ∈ {50, 60, 65}` with 10th-percentile power ≥ 0.80" is
withdrawn as unnecessary over-engineering for this scope, not because it failed.
Its result stands in the record either way (`RESULTS.md`): no `N` in that grid met
it, the binding quantity was the crossover IUT's 10th percentile, and 65 would
have cost about 30% more GPU time without meeting the threshold.

No further power analysis will be run and no further development data collected.

## B2. C1-vs-C3 is no longer the sole primary contrast

Section 4.1 above made C1 vs C3 the primary architecture contrast with C2, C4 and
C5 secondary. That is withdrawn. It narrowed the thesis to one pairwise crossover
when the question is whether the *relative standing of the architectures* changes
with task complexity.

The final experiment compares all five architectures C1–C5 across all four budget
caps 16k / 32k / 64k / 128k, and the analysis follows that design.

### Confirmatory family

Five architecture-motivated contrasts, written as `later − earlier` along the
direction of increasing structure:

| # | contrast | `δ` |
|---|---|---|
| 1 | C1 → C2 | `sat(C2) − sat(C1)` |
| 2 | C2 → C4 | `sat(C4) − sat(C2)` |
| 3 | C4 → C3 | `sat(C3) − sat(C4)` |
| 4 | C1 → C5 | `sat(C5) − sat(C1)` |
| 5 | C5 → C3 | `sat(C3) − sat(C5)` |

Each contrast carries the same two pre-declared tests, at the primary cap:

* **Interaction.** `Δ = δ_high − δ_low`, two-sided. Does the contrast change with
  complexity?
* **Crossover.** Intersection-union: `δ_low < 0` and `δ_high > 0`, each one-sided
  at α. Does it change sign? The direction is declared uniformly for all five —
  the more structured architecture behind at low D, ahead at high D — because
  that is the single hypothesis the thesis tests.

**Multiplicity: Holm across all ten confirmatory p-values** (five contrasts × two
tests). One family, one rule. The IUT needs no internal correction; Holm applies
to its combined p-value like any other.

### Complete comparison matrix

All ten pairwise contrasts among C1–C5, in every band and at every cap, are
reported with effect estimates and intervals. The five above are the ones given
an explicit architectural reading; the remaining five complete the matrix and are
descriptive. Nothing outside the confirmatory family carries a confirmatory claim.

### Unchanged from the plan above

* Primary cap 64000; the other three caps are run, reported in the matrix, and
  pre-declared secondary. This is the existing registered decision and is what
  keeps the confirmatory family at ten tests rather than forty.
* Primary estimand: the mean paired difference (amendment A1). Sign-flip
  permutation on the mean is the primary test; the sign test is sensitivity on a
  different functional.
* Label permutation for `Δ`, exact under exchangeability of the two bands.
* The medium band stays descriptive (amendment A2). Its second rationale — that it
  should not drive `N` — is moot now that no `N` rule exists; the first stands.
* Permutation assumptions and the per-band skewness diagnostic (amendment A3).
* Block B (`n = 4/5/6` at `O = 3`) is descriptive. `n` is a generator parameter,
  never a complexity level, so Block B is not a D-band and forms no confirmatory
  contrast.

## B3. How a null crossover result must be reported

The crossover question stays confirmatory. Its limitation is stated in advance,
in these terms:

> Failure to establish crossover statistically is not evidence that no crossover
> exists.

The registered power figures are the reason this has to be said plainly rather
than left implied. At `N = 50` and an assumed crossover the size of the observed
architecture effect, the design's crossover power has a median of 0.985 but a
10th-percentile floor of 0.600; at half that effect the floor is 0.374. A
non-rejection is therefore consistent both with no crossover and with a real
crossover the design could not resolve, and the write-up may not choose between
those two readings on the strength of a null result.

This applies to each of the five contrasts separately. Nothing in this plan
licenses the reverse claim — that the architectures do not reorder with
complexity — from an absence of rejections.
