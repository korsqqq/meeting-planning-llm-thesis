# Post-gate architecture diagnostic — n = 8, cap 64 000

**Development diagnostic, descriptive.** It cannot repair the C1 budget gate: that gate reads C1 only by design, so a C2 or C3 result is not evidence about it in either direction. No cap is selected here.

Runs: 240 diagnostic + 60 C1 baseline, model `Qwen/Qwen3-32B-AWQ`, subset `911e4864d6fb`, paired on the identical instance. Completeness audit passed.

> **Mixed provenance:** the diagnostic runs carry 3 different git commits.


## Validator-approved non-empty plan rate

| condition | Low | Medium | High | pooled |
|---|---|---|---|---|
| c1_react | 9/20 = 0.45 ✗ | 9/20 = 0.45 ✗ | 5/20 = 0.25 ✗ | 23/60 = 0.38 |
| c2_verify_revise | 14/20 = 0.70 ✓ | 10/20 = 0.50 ✓ | 9/20 = 0.45 ✗ | 33/60 = 0.55 |
| c3_mas | 16/20 = 0.80 ✓ | 14/20 = 0.70 ✓ | 14/20 = 0.70 ✓ | 44/60 = 0.73 |
| c4_planner_critic | 16/20 = 0.80 ✓ | 14/20 = 0.70 ✓ | 15/20 = 0.75 ✓ | 45/60 = 0.75 |
| c5_best_of_3 | 0/20 = 0.00 ✗ | 5/20 = 0.25 ✗ | 1/20 = 0.05 ✗ | 6/60 = 0.10 |

`✓` marks a band at or above the reference rate of 50%, reused from the registered gate. It is a reference point here, not a decision rule.


## Reading, fixed before the outcomes

**NOT_A_UNIVERSAL_TASK_WIDE_FLOOR** — at least one structured condition reaches the reference rate in every band, so the n = 8 floor is not universal across architectures.

The C1 budget gate remains failed. It read C1 only, by design, so this diagnostic is not evidence about it in either direction; no primary cap is selected here and the main experiment does not start on this result.


## Satisfaction, cost and behaviour (pooled)

| condition | mean sat | zeros | mean tokens | cap used | calls | proposal validity |
|---|---|---|---|---|---|---|
| c1_react | 0.300 | 37/60 | 48673 | 0.76 | 18.8 | 0.247 |
| c2_verify_revise | 0.418 | 27/60 | 55141 | 0.86 | 20.7 | 0.375 |
| c3_mas | 0.643 | 16/60 | 45274 | 0.71 | 23.3 | 0.691 |
| c4_planner_critic | 0.489 | 15/60 | 45160 | 0.71 | 18.3 | 0.500 |
| c5_best_of_3 | 0.079 | 54/60 | 62905 | 0.98 | 36.6 | 0.162 |

## Paired against the completed C1 arm

| condition | recovered where C1 failed | lost where C1 succeeded | same | mean Δ satisfaction |
|---|---|---|---|---|
| c2_verify_revise | 10 | 0 | 50 | +0.118 |
| c3_mas | 28 | 7 | 25 | +0.343 |
| c4_planner_critic | 26 | 4 | 30 | +0.189 |
| c5_best_of_3 | 2 | 19 | 39 | -0.221 |

Counts only. No hypothesis test is performed by this diagnostic.


## C2 stage reach

| stage | runs reaching it once | runs reaching it twice | mean calls |
|---|---|---|---|
| verify | 35/60 | 27/60 | 1.033 |
| revise | 32/60 | 24/60 | 0.933 |

Runs completing both full cycles: 24/60. C2 is the only structured condition whose later stages have no reserved budget, so a stage that was never reached is a fidelity fact about the run, not a model choice.


## C3 by role

| role | proposal attempts | valid | validity rate |
|---|---|---|---|
| workers | 106 | 61 | 0.576 |
| critic | 43 | 42 | 0.977 |

The critic ran in 44/60 runs (0.73). worker and critic rates are not comparable with each other or with C1: workers solve a decomposed sub-problem and the critic is validator-gated.


## c4_planner_critic — condition diagnostics

Log-only quantities the condition records about its own structure. A degenerate condition shows up here as a number rather than being inferred from the rate.

| quantity | value |
|---|---|
| `n_with_diagnostics` | 60 |
| `mean_critic_room` | 22886.65 |
| `mean_evidence_people_covered` | 6.4167 |
| `mean_evidence_travel_coverage` | 0.2765 |
| `mean_fallback_size` | 0.5667 |
| `mean_headroom` | 2.3 |
| `mean_known_pairs_shown` | 3.3 |
| `mean_planner_quota` | 47808.0 |
| `mean_planner_spent` | 40857.35 |
| `mean_pool_size` | 2.8667 |
| `mean_unknown_pairs_shown` | 10.5333 |
| `runs_critic_improved` | 32 |
| `rate_critic_improved` | 0.5333 |
| `runs_critic_reached` | 49 |
| `rate_critic_reached` | 0.8167 |

## c5_best_of_3 — condition diagnostics

Log-only quantities the condition records about its own structure. A degenerate condition shows up here as a number rather than being inferred from the rate.

| quantity | value |
|---|---|
| `n_with_diagnostics` | 60 |
| `mean_n_distinct_products` | 1.1 |
| `mean_winner_attempt_index` | 0.1 |
| `winner_attempt_counts` | {'0': 56, '1': 2, '2': 2} |
| `distinct_product_counts` | {'1': 54, '2': 6, '3': 0} |

Distinct final products against the outcome. Equal marginal counts would not establish a run-by-run correspondence; this table does.

| distinct products | satisfaction = 0 | satisfaction > 0 |
|---|---|---|
| 1 | 54 | 0 |
| 2 | 0 | 6 |

## Why proposals were rejected

| condition | overlapping_meeting | travel_infeasible | window_violation |
|---|---|---|---|
| c1_react | 5 | 57 | 40 |
| c2_verify_revise | 8 | 72 | 54 |
| c3_mas | 0 | 39 | 17 |
| c4_planner_critic | 2 | 46 | 37 |
| c5_best_of_3 | 0 | 26 | 26 |

Multi-label over rejected proposals, so a row can exceed the number of rejections.

