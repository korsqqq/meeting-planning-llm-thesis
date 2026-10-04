# Formal pilot — pre-registered design-validation analysis

**This is a pilot, not a confirmatory test of the research question.** The pilot carries ten instances per complexity level on seeds 10005+; the held-out main experiment (seeds 100000+) is the confirmatory one and requires at least fifty instances per level and cap. Everything below is design validation, and it is allowed to conclude that the instrument does not measure what the design assumed.

Pre-registered matrix (THESIS_DECISIONS §3): 3 conditions × 4 caps × 30 instances = 360 runs, one run per cell. Subset `29d836cb56141f3a29ac3fa9670e662c123586e967f2d5ae1280925c2228edbd`, binning `e6a365481dd4a485867c62377ac5bc6dc58b277113ccd207605a0258d5cba306`, model Qwen/Qwen3-32B-AWQ, window 32768, prefix caching disabled.

**Provenance.** Frozen behavioural code: `bc968bb`. Executing commit recorded in every run: `916434a`. The inspected difference between them is additive pre-registration plumbing (`--write-expected-runs`) that returns before the first run; the execution path is unchanged.

**Sign convention (expose).** `delta = S_C1 − S_C3`, single agent minus multi-agent. A **positive** delta means the single agent scored higher; a **negative** delta means the hierarchy did. H1 expects delta > 0 on easy instances, H2 expects delta < 0 on hard ones, and the crossover the research question asks about is a + to − sign change as complexity rises.

**Inference preserves the pairing.** Every instance is solved by all three conditions at the same cap. The bootstrap resamples instances, carrying whole paired differences; the interaction test permutes complexity labels while each paired difference stays intact. With ten instances per level and cap, intervals are wide and the interaction test is a diagnostic rather than a hypothesis test.


## Primary outcome and the expose's secondary metrics

| condition | cap | sat | achieved | optimum | optimality | feasible | non-empty plan | tokens | cap used | exhausted | sat/1k | calls | latency s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c1_react | 8000 | 0.000 | 0.000 | 3.767 | 0.000 | 1.000 | 0.000 | 7674.367 | 0.959 | 13 | 0.000 | 7.700 | 145.346 |
| c1_react | 16000 | 0.067 | 0.200 | 3.767 | 0.067 | 1.000 | 0.067 | 15550.767 | 0.972 | 7 | 0.004 | 10.833 | 311.833 |
| c1_react | 32000 | 0.250 | 0.867 | 3.767 | 0.233 | 1.000 | 0.267 | 28549.367 | 0.892 | 5 | 0.009 | 14.167 | 604.281 |
| c1_react | 64000 | 0.436 | 1.433 | 3.767 | 0.367 | 1.000 | 0.467 | 38933.733 | 0.608 | 0 | 0.011 | 15.933 | 853.481 |
| c2_verify_revise | 8000 | 0.000 | 0.000 | 3.767 | 0.000 | 1.000 | 0.000 | 7674.367 | 0.959 | 13 | 0.000 | 7.700 | 147.296 |
| c2_verify_revise | 16000 | 0.067 | 0.200 | 3.767 | 0.067 | 1.000 | 0.067 | 15747.467 | 0.984 | 8 | 0.004 | 10.933 | 319.330 |
| c2_verify_revise | 32000 | 0.306 | 1.200 | 3.767 | 0.233 | 1.000 | 0.333 | 30767.300 | 0.962 | 10 | 0.010 | 15.033 | 629.745 |
| c2_verify_revise | 64000 | 0.556 | 1.900 | 3.767 | 0.433 | 1.000 | 0.600 | 49213.233 | 0.769 | 5 | 0.011 | 19.200 | 974.245 |
| c3_mas | 8000 | 0.000 | 0.000 | 3.767 | 0.000 | 1.000 | 0.000 | 5754.867 | 0.719 | 0 | 0.000 | 6.833 | 128.531 |
| c3_mas | 16000 | 0.000 | 0.000 | 3.767 | 0.000 | 1.000 | 0.000 | 11701.600 | 0.731 | 0 | 0.000 | 12.100 | 242.121 |
| c3_mas | 32000 | 0.489 | 1.800 | 3.767 | 0.300 | 1.000 | 0.633 | 24452.867 | 0.764 | 0 | 0.020 | 17.900 | 528.763 |
| c3_mas | 64000 | 0.789 | 2.967 | 3.767 | 0.533 | 1.000 | 0.967 | 34420.333 | 0.538 | 0 | 0.023 | 20.333 | 779.473 |

**Feasibility is 1.000 everywhere and carries no information about the agent.** The harness finalises whatever plan survives validation, and an empty plan is vacuously feasible, so the rate is 1 by construction (§5). The informative counterpart is the share of runs holding a non-empty plan, in the column beside it. `final_invalid_reasons` is empty in every cell for the same reason; the rejection taxonomy that does carry information is over proposals, further below.

**Latency is wall-clock on a shared machine** with thermal throttling active throughout and one GPU released partway through the sweep, after which the remaining work ran alone. It is reported because the expose asks for it, and it must not be read as a clean efficiency comparison between conditions.


## Paired contrasts


### delta = S(c1_react) − S(c3_mas)

delta = S_c1_react - S_c3_mas; delta < 0 means c3_mas scored higher.

| cap | n | mean delta | median | 95% CI (paired bootstrap) | CI excl. 0 | mean Δtokens | sign test p |
|---|---|---|---|---|---|---|---|
| 8000 | 30 | 0.000 | 0.000 | [+0.000, +0.000] | no | 1919.500 | n/a |
| 16000 | 30 | 0.067 | 0.000 | [+0.000, +0.167] | no | 3849.167 | 0.5000 |
| 32000 | 30 | -0.239 | 0.000 | [-0.403, -0.075] | yes | 4096.500 | 0.0129 |
| 64000 | 30 | -0.353 | -0.464 | [-0.526, -0.173] | yes | 4513.400 | 0.0026 |

| cap | level | n | mean delta | 95% CI | CI excl. 0 |
|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | medium | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | medium | 10 | 0.200 | [+0.000, +0.500] | no |
| 16000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 32000 | easy | 10 | -0.300 | [-0.600, +0.000] | no |
| 32000 | medium | 10 | -0.350 | [-0.625, -0.100] | yes |
| 32000 | hard | 10 | -0.067 | [-0.317, +0.233] | no |
| 64000 | easy | 10 | -0.460 | [-0.767, -0.133] | yes |
| 64000 | medium | 10 | -0.457 | [-0.690, -0.215] | yes |
| 64000 | hard | 10 | -0.142 | [-0.458, +0.200] | no |

Interaction with complexity — all three levels, each paired difference kept intact and only the level labels permuted:

| cap | omnibus H | omnibus p | pre-registered decreasing-trend p | observed direction | two-sided trend p | easy − hard | pre-reg. positive p | ρ(complexity, delta) |
|---|---|---|---|---|---|---|---|---|
| 8000 | 0.000 | 1.0000 | 1.0000 | flat | 1.0000 | 0.000 | 1.0000 | n/a |
| 16000 | 0.774 | 0.3090 | 0.6646 | flat | 1.0000 | 0.000 | 0.6646 | -0.095 |
| 32000 | 1.241 | 0.4784 | 0.8353 | delta rises with complexity | 0.3464 | -0.233 | 0.8712 | 0.179 |
| 64000 | 2.387 | 0.2966 | 0.9351 | delta rises with complexity | 0.1356 | -0.318 | 0.9164 | 0.219 |

**The pre-registered alternative is delta FALLING with complexity** — H1 puts the single agent ahead on easy, H2 puts the hierarchy ahead on hard, so the crossover is a `+ → −` change. The one-sided columns above test that alternative and only that one. Where the observed direction is the opposite, the pre-registered p is necessarily large, and the p for the rising direction is recorded in `summary.json` as exploratory: it tests an alternative nobody registered, and quoting it as evidence about H1/H2 would be evaluating the hypothesis against its own contradiction.

A large p here does **not** establish that the difference is independent of complexity. With ten instances per level these tests have power only against a large effect, so a null result leaves the question open rather than answering it. Where the point estimates trend against the pre-registered crossover, the defensible statement is that the estimates lean that way and the pilot provides insufficient evidence to establish any trend.


### delta = S(c1_react) − S(c2_verify_revise)

delta = S_c1_react - S_c2_verify_revise; delta < 0 means c2_verify_revise scored higher.

| cap | n | mean delta | median | 95% CI (paired bootstrap) | CI excl. 0 | mean Δtokens | sign test p |
|---|---|---|---|---|---|---|---|
| 8000 | 30 | 0.000 | 0.000 | [+0.000, +0.000] | no | 0.000 | n/a |
| 16000 | 30 | 0.000 | 0.000 | [+0.000, +0.000] | no | -196.700 | n/a |
| 32000 | 30 | -0.056 | 0.000 | [-0.139, +0.000] | no | -2217.933 | 0.5000 |
| 64000 | 30 | -0.119 | 0.000 | [-0.256, +0.003] | no | -10279.500 | 0.2188 |

| cap | level | n | mean delta | 95% CI | CI excl. 0 |
|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | medium | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | medium | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 32000 | easy | 10 | -0.167 | [-0.417, +0.000] | no |
| 32000 | medium | 10 | 0.000 | [+0.000, +0.000] | no |
| 32000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 64000 | easy | 10 | -0.167 | [-0.417, +0.000] | no |
| 64000 | medium | 10 | -0.267 | [-0.533, +0.000] | no |
| 64000 | hard | 10 | 0.075 | [+0.000, +0.225] | no |

Interaction with complexity — all three levels, each paired difference kept intact and only the level labels permuted:

| cap | omnibus H | omnibus p | pre-registered decreasing-trend p | observed direction | two-sided trend p | easy − hard | pre-reg. positive p | ρ(complexity, delta) |
|---|---|---|---|---|---|---|---|---|
| 8000 | 0.000 | 1.0000 | 1.0000 | flat | 1.0000 | 0.000 | 1.0000 | n/a |
| 16000 | 0.000 | 1.0000 | 1.0000 | flat | 1.0000 | 0.000 | 1.0000 | n/a |
| 32000 | 0.774 | 0.3160 | 1.0000 | delta rises with complexity | 0.2123 | -0.167 | 1.0000 | 0.315 |
| 64000 | 2.220 | 0.0983 | 0.9141 | delta rises with complexity | 0.1839 | -0.242 | 0.9300 | 0.198 |

**The pre-registered alternative is delta FALLING with complexity** — H1 puts the single agent ahead on easy, H2 puts the hierarchy ahead on hard, so the crossover is a `+ → −` change. The one-sided columns above test that alternative and only that one. Where the observed direction is the opposite, the pre-registered p is necessarily large, and the p for the rising direction is recorded in `summary.json` as exploratory: it tests an alternative nobody registered, and quoting it as evidence about H1/H2 would be evaluating the hypothesis against its own contradiction.

A large p here does **not** establish that the difference is independent of complexity. With ten instances per level these tests have power only against a large effect, so a null result leaves the question open rather than answering it. Where the point estimates trend against the pre-registered crossover, the defensible statement is that the estimates lean that way and the pilot provides insufficient evidence to establish any trend.


### delta = S(c2_verify_revise) − S(c3_mas)

delta = S_c2_verify_revise - S_c3_mas; delta < 0 means c3_mas scored higher.

| cap | n | mean delta | median | 95% CI (paired bootstrap) | CI excl. 0 | mean Δtokens | sign test p |
|---|---|---|---|---|---|---|---|
| 8000 | 30 | 0.000 | 0.000 | [+0.000, +0.000] | no | 1919.500 | n/a |
| 16000 | 30 | 0.067 | 0.000 | [+0.000, +0.167] | no | 4045.867 | 0.5000 |
| 32000 | 30 | -0.183 | 0.000 | [-0.364, -0.006] | yes | 6314.433 | 0.1185 |
| 64000 | 30 | -0.233 | -0.083 | [-0.389, -0.073] | yes | 14792.900 | 0.0075 |

| cap | level | n | mean delta | 95% CI | CI excl. 0 |
|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | medium | 10 | 0.000 | [+0.000, +0.000] | no |
| 8000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | easy | 10 | 0.000 | [+0.000, +0.000] | no |
| 16000 | medium | 10 | 0.200 | [+0.000, +0.500] | no |
| 16000 | hard | 10 | 0.000 | [+0.000, +0.000] | no |
| 32000 | easy | 10 | -0.133 | [-0.500, +0.217] | no |
| 32000 | medium | 10 | -0.350 | [-0.625, -0.100] | yes |
| 32000 | hard | 10 | -0.067 | [-0.317, +0.233] | no |
| 64000 | easy | 10 | -0.293 | [-0.567, -0.026] | yes |
| 64000 | medium | 10 | -0.190 | [-0.430, +0.015] | no |
| 64000 | hard | 10 | -0.217 | [-0.517, +0.133] | no |

Interaction with complexity — all three levels, each paired difference kept intact and only the level labels permuted:

| cap | omnibus H | omnibus p | pre-registered decreasing-trend p | observed direction | two-sided trend p | easy − hard | pre-reg. positive p | ρ(complexity, delta) |
|---|---|---|---|---|---|---|---|---|
| 8000 | 0.000 | 1.0000 | 1.0000 | flat | 1.0000 | 0.000 | 1.0000 | n/a |
| 16000 | 0.774 | 0.3090 | 0.6646 | flat | 1.0000 | 0.000 | 0.6646 | -0.095 |
| 32000 | 1.226 | 0.5006 | 0.5680 | delta rises with complexity | 0.8781 | -0.067 | 0.6194 | 0.029 |
| 64000 | 0.736 | 0.6875 | 0.6691 | delta rises with complexity | 0.6871 | -0.076 | 0.6451 | 0.079 |

**The pre-registered alternative is delta FALLING with complexity** — H1 puts the single agent ahead on easy, H2 puts the hierarchy ahead on hard, so the crossover is a `+ → −` change. The one-sided columns above test that alternative and only that one. Where the observed direction is the opposite, the pre-registered p is necessarily large, and the p for the rising direction is recorded in `summary.json` as exploratory: it tests an alternative nobody registered, and quoting it as evidence about H1/H2 would be evaluating the hypothesis against its own contradiction.

A large p here does **not** establish that the difference is independent of complexity. With ten instances per level these tests have power only against a large effect, so a null result leaves the question open rather than answering it. Where the point estimates trend against the pre-registered crossover, the defensible statement is that the estimates lean that way and the pilot provides insufficient evidence to establish any trend.


## Crossover check on the primary contrast

H1: delta > 0 on easy (single agent ahead); H2: delta < 0 on hard; crossover is + on easy turning - on hard within a cap.

| level | 8000 sign | 16000 sign | 32000 sign | 64000 sign | caps whose CI excludes 0 |
|---|---|---|---|---|---|
| easy | 0 | 0 | - | - | ['64000'] |
| medium | 0 | + | - | - | ['32000', '64000'] |
| hard | 0 | 0 | - | - | none |

## Pilot diagnostic: the low-complexity sanity check fails

The expose specifies that on the easiest level the single agent should be at least on a par with the hierarchy, and that a violation points at the measurement design, the experimental setup or the evaluation protocol rather than at a finding. The pilot violates it: on easy instances the hierarchy is ahead at both working caps, and the margin there is at least as large as on hard.

This diagnostic does **not** rest on the interaction tests. It is a statement about the easy level on its own, where the paired interval excludes zero. The interaction tests are underpowered at ten instances per level and neither establish nor rule out a complexity dependence; the sanity check fails regardless of how that question is eventually resolved.

Two properties of the level assignment bear on this and are recorded in §3 rather than repaired, because repairing them after seeing the scores would mean choosing the selection from its results:

| level | mean oracle optimum | C1 satisfaction at 64k | C1 meetings at 64k | C3 − C1 delta at 64k |
|---|---|---|---|---|
| easy | 5.500 | 0.333 | 1.700 | -0.460 |
| medium | 3.300 | 0.400 | 1.100 | -0.457 |
| hard | 2.500 | 0.575 | 1.500 | -0.142 |

The baseline's own satisfaction does not fall as the level rises, which is the first sign that the axis is not ordering difficulty as intended.


The oracle optimum falls as structural conflict rises, so the denominator of the primary metric shrinks with the level it is meant to index: one meeting is worth far more on hard than on easy. The hard level additionally contains no four-person instances and is nearly determined by `tightness`. Whether the complexity axis creates the rising-difficulty ordering the research question assumes is therefore open, and it is the decision the main experiment must settle before it runs.


## Proposal taxonomy by role — diagnostic only

These rates describe internal behaviour and are **not** the evidence for why one condition outscores another. A C1 proposal addresses the whole task; a C3 worker proposal addresses a decomposed sub-instance; a C3 critic proposal has already passed the aggregator gate and is close to valid by construction. Pooling them would compare generation against generation-plus-filtered-selection, and even split by role the worker and C1 numbers are not the same measurement. Establishing what causes the difference is what C4 is for.

| condition | cap | role | attempts | validity |
|---|---|---|---|---|
| c1_react | 16000 | react_step | 5 | 0.400 |
| c1_react | 32000 | react_step | 23 | 0.348 |
| c1_react | 64000 | react_step | 43 | 0.372 |
| c2_verify_revise | 16000 | react_step | 5 | 0.400 |
| c2_verify_revise | 32000 | react_step | 23 | 0.348 |
| c2_verify_revise | 32000 | revise | 11 | 0.636 |
| c2_verify_revise | 64000 | react_step | 50 | 0.300 |
| c2_verify_revise | 64000 | revise | 43 | 0.698 |
| c3_mas | 16000 | worker_b | 1 | 0.000 |
| c3_mas | 32000 | worker_a | 18 | 0.944 |
| c3_mas | 32000 | worker_b | 13 | 0.769 |
| c3_mas | 32000 | critic | 19 | 0.947 |
| c3_mas | 64000 | worker_a | 27 | 0.815 |
| c3_mas | 64000 | worker_b | 27 | 0.704 |
| c3_mas | 64000 | critic | 29 | 1.000 |

| condition | cap | pooled attempts | pooled validity | acceptance | rejection reasons |
|---|---|---|---|---|---|
| c1_react | 8000 | 0 | n/a | n/a | {} |
| c1_react | 16000 | 5 | 0.400 | 0.400 | {'travel_infeasible': 3} |
| c1_react | 32000 | 23 | 0.348 | 0.348 | {'travel_infeasible': 13, 'window_violation': 6} |
| c1_react | 64000 | 43 | 0.372 | 0.349 | {'travel_infeasible': 23, 'window_violation': 11, 'overlapping_meeting': 1} |
| c2_verify_revise | 8000 | 0 | n/a | n/a | {} |
| c2_verify_revise | 16000 | 5 | 0.400 | 0.400 | {'travel_infeasible': 3} |
| c2_verify_revise | 32000 | 34 | 0.441 | 0.294 | {'travel_infeasible': 17, 'window_violation': 6} |
| c2_verify_revise | 64000 | 93 | 0.484 | 0.204 | {'travel_infeasible': 41, 'window_violation': 27, 'overlapping_meeting': 2} |
| c3_mas | 8000 | 0 | n/a | n/a | {} |
| c3_mas | 16000 | 1 | 0.000 | 0.000 | {'travel_infeasible': 1} |
| c3_mas | 32000 | 50 | 0.900 | 0.660 | {'travel_infeasible': 4, 'window_violation': 1} |
| c3_mas | 64000 | 83 | 0.843 | 0.602 | {'travel_infeasible': 11, 'window_violation': 4} |

## What each condition actually executed

Counted from call roles, which record the answer directly. An earlier version of this analysis inferred C2 exposure from identical token totals; identical totals are consistent with a stage never running but do not demonstrate it.

| condition | cap | role | calls | calls/run | runs reached | share |
|---|---|---|---|---|---|---|
| c1_react | 8000 | react_step | 201 | 6.700 | 30 | 1.000 |
| c1_react | 16000 | react_step | 295 | 9.833 | 30 | 1.000 |
| c1_react | 32000 | react_step | 395 | 13.167 | 30 | 1.000 |
| c1_react | 64000 | react_step | 448 | 14.933 | 30 | 1.000 |
| c2_verify_revise | 8000 | react_step | 201 | 6.700 | 30 | 1.000 |
| c2_verify_revise | 16000 | react_step | 295 | 9.833 | 30 | 1.000 |
| c2_verify_revise | 16000 | verify | 2 | 0.067 | 2 | 0.067 |
| c2_verify_revise | 16000 | revise | 1 | 0.033 | 1 | 0.033 |
| c2_verify_revise | 32000 | react_step | 395 | 13.167 | 30 | 1.000 |
| c2_verify_revise | 32000 | verify | 15 | 0.500 | 9 | 0.300 |
| c2_verify_revise | 32000 | revise | 11 | 0.367 | 7 | 0.233 |
| c2_verify_revise | 64000 | react_step | 454 | 15.133 | 30 | 1.000 |
| c2_verify_revise | 64000 | verify | 48 | 1.600 | 25 | 0.833 |
| c2_verify_revise | 64000 | revise | 44 | 1.467 | 24 | 0.800 |
| c3_mas | 8000 | worker_a | 91 | 3.033 | 30 | 1.000 |
| c3_mas | 8000 | worker_b | 84 | 2.800 | 30 | 1.000 |
| c3_mas | 16000 | worker_a | 168 | 5.600 | 30 | 1.000 |
| c3_mas | 16000 | worker_b | 165 | 5.500 | 30 | 1.000 |
| c3_mas | 32000 | worker_a | 244 | 8.133 | 30 | 1.000 |
| c3_mas | 32000 | worker_b | 244 | 8.133 | 30 | 1.000 |
| c3_mas | 32000 | critic | 19 | 0.633 | 19 | 0.633 |
| c3_mas | 64000 | worker_a | 274 | 9.133 | 30 | 1.000 |
| c3_mas | 64000 | worker_b | 277 | 9.233 | 30 | 1.000 |
| c3_mas | 64000 | critic | 29 | 0.967 | 29 | 0.967 |

## Termination and budget

`aborted` denotes a grant that expired inside a reasoning block, leaving nothing after the closing think tag; the loop then finalises with the plan it already holds, and across the whole matrix a run holds a non-empty final plan exactly when it holds a non-empty best plan. It is collapsed with `budget` as budget-limited. That is a property of the loop and is **not** the same as the ledger's `budget_exhausted` flag, reported beside it: a condition can stop without exhausting its grant, and the two columns are given separately so no mechanism is read off one of them alone.

| condition | cap | termination | budget-limited | ledger exhausted | tokens left | finalisation mismatch |
|---|---|---|---|---|---|---|
| c1_react | 8000 | {'aborted': 17, 'budget': 13} | 30 | 13 | 325.633 | 0 |
| c1_react | 16000 | {'aborted': 21, 'agent_finish': 2, 'budget': 7} | 28 | 7 | 449.233 | 0 |
| c1_react | 32000 | {'agent_finish': 9, 'aborted': 16, 'budget': 5} | 21 | 5 | 3450.633 | 3 |
| c1_react | 64000 | {'agent_finish': 26, 'aborted': 4} | 4 | 0 | 25066.267 | 6 |
| c2_verify_revise | 8000 | {'aborted': 17, 'budget': 13} | 30 | 13 | 325.633 | 0 |
| c2_verify_revise | 16000 | {'aborted': 21, 'budget': 8, 'agent_finish': 1} | 29 | 8 | 252.533 | 0 |
| c2_verify_revise | 32000 | {'budget': 10, 'aborted': 16, 'agent_finish': 4} | 26 | 10 | 1232.700 | 4 |
| c2_verify_revise | 64000 | {'agent_finish': 21, 'budget': 5, 'aborted': 4} | 9 | 5 | 14786.767 | 9 |
| c3_mas | 8000 | {'agent_finish': 18, 'aborted': 12} | 12 | 0 | 2245.133 | 0 |
| c3_mas | 16000 | {'agent_finish': 19, 'aborted': 11} | 11 | 0 | 4298.400 | 0 |
| c3_mas | 32000 | {'agent_finish': 13, 'aborted': 17} | 17 | 0 | 7547.133 | 5 |
| c3_mas | 64000 | {'agent_finish': 28, 'aborted': 2} | 2 | 0 | 29579.667 | 11 |

## Level tables


### c1_react

| cap | level | n | sat | achieved | optimum | optimality | attempts | runs w/ valid | budget-limited |
|---|---|---|---|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 0 | 0.000 | 10 |
| 8000 | medium | 10 | 0.000 | 0.000 | 3.300 | 0.000 | 0 | 0.000 | 10 |
| 8000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 10 |
| 16000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 1 | 0.000 | 8 |
| 16000 | medium | 10 | 0.200 | 0.600 | 3.300 | 0.200 | 4 | 0.200 | 10 |
| 16000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 10 |
| 32000 | easy | 10 | 0.250 | 1.200 | 5.500 | 0.200 | 9 | 0.300 | 4 |
| 32000 | medium | 10 | 0.400 | 1.100 | 3.300 | 0.400 | 10 | 0.400 | 7 |
| 32000 | hard | 10 | 0.100 | 0.300 | 2.500 | 0.100 | 4 | 0.100 | 10 |
| 64000 | easy | 10 | 0.333 | 1.700 | 5.500 | 0.200 | 12 | 0.400 | 1 |
| 64000 | medium | 10 | 0.400 | 1.100 | 3.300 | 0.400 | 17 | 0.400 | 2 |
| 64000 | hard | 10 | 0.575 | 1.500 | 2.500 | 0.500 | 14 | 0.600 | 1 |

### c2_verify_revise

| cap | level | n | sat | achieved | optimum | optimality | attempts | runs w/ valid | budget-limited |
|---|---|---|---|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 0 | 0.000 | 10 |
| 8000 | medium | 10 | 0.000 | 0.000 | 3.300 | 0.000 | 0 | 0.000 | 10 |
| 8000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 10 |
| 16000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 1 | 0.000 | 9 |
| 16000 | medium | 10 | 0.200 | 0.600 | 3.300 | 0.200 | 4 | 0.200 | 10 |
| 16000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 10 |
| 32000 | easy | 10 | 0.417 | 2.200 | 5.500 | 0.200 | 19 | 0.500 | 6 |
| 32000 | medium | 10 | 0.400 | 1.100 | 3.300 | 0.400 | 11 | 0.400 | 10 |
| 32000 | hard | 10 | 0.100 | 0.300 | 2.500 | 0.100 | 4 | 0.100 | 10 |
| 64000 | easy | 10 | 0.500 | 2.700 | 5.500 | 0.200 | 28 | 0.600 | 1 |
| 64000 | medium | 10 | 0.667 | 1.800 | 3.300 | 0.600 | 31 | 0.700 | 4 |
| 64000 | hard | 10 | 0.500 | 1.200 | 2.500 | 0.500 | 34 | 0.500 | 4 |

### c3_mas

| cap | level | n | sat | achieved | optimum | optimality | attempts | runs w/ valid | budget-limited |
|---|---|---|---|---|---|---|---|---|---|
| 8000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 0 | 0.000 | 3 |
| 8000 | medium | 10 | 0.000 | 0.000 | 3.300 | 0.000 | 0 | 0.000 | 5 |
| 8000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 4 |
| 16000 | easy | 10 | 0.000 | 0.000 | 5.500 | 0.000 | 1 | 0.000 | 3 |
| 16000 | medium | 10 | 0.000 | 0.000 | 3.300 | 0.000 | 0 | 0.000 | 5 |
| 16000 | hard | 10 | 0.000 | 0.000 | 2.500 | 0.000 | 0 | 0.000 | 3 |
| 32000 | easy | 10 | 0.550 | 2.700 | 5.500 | 0.300 | 23 | 0.800 | 4 |
| 32000 | medium | 10 | 0.750 | 2.300 | 3.300 | 0.600 | 20 | 0.800 | 5 |
| 32000 | hard | 10 | 0.167 | 0.400 | 2.500 | 0.000 | 7 | 0.300 | 8 |
| 64000 | easy | 10 | 0.793 | 4.300 | 5.500 | 0.600 | 31 | 1.000 | 0 |
| 64000 | medium | 10 | 0.857 | 2.700 | 3.300 | 0.500 | 29 | 1.000 | 0 |
| 64000 | hard | 10 | 0.717 | 1.900 | 2.500 | 0.500 | 23 | 0.900 | 2 |

## Limitations

- **Pilot, not confirmatory.** Ten instances per level and cap. The bootstrap intervals are correspondingly wide, and the interaction analysis is a diagnostic. No claim here is a test of the research question.
- **A non-significant interaction is not a finding of no interaction.** The omnibus and trend tests above have power only against a large effect at this sample size. Where they return a large p, the correct reading is that the pilot does not resolve whether the condition difference depends on complexity, not that it does not.
- One run per cell. Determinism removes sampling variance, so a cell is a point estimate of a deterministic system rather than a mean over repetitions.
- The hard level contains no four-person instances and is nearly determined by `tightness`, so level, instance size and that generator knob are partially confounded. Recorded rather than repaired.
- Oracle optimum falls with level, so the primary metric's denominator shrinks along the axis it indexes. Absolute meetings are reported beside satisfaction for that reason.
- Categorical levels come from this pool's own boundaries and are not comparable with the calibration sweep's levels.
- One GPU was released partway through, and the remaining shard was replayed against the other endpoint. Both served the same model revision under the same flags and cross-endpoint determinism was verified separately; the endpoint distribution is recorded in the audit block rather than treated as uniform.
- Associations between instance properties and outcomes are descriptive: the generator varies several properties together, so none is manipulated alone.

