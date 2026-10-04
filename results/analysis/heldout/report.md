# Held-out experiment — confirmatory analysis

Plan: `results/analysis/interaction_power/PRE_ANALYSIS_PLAN.md (frozen 2026-08-27; amendment 2026-08-27 A1-A4; amendment 2 2026-08-27 B1-B3)`.

Held-out confirmatory analysis. The confirmatory family is five architecture-motivated contrasts at the primary cap 64000, each with an interaction test and a crossover IUT, Holm across all ten p-values. Everything else in this document -- the other three caps, the other five pairwise contrasts, the medium band, Block B, the sign test and the bootstrap intervals -- is descriptive or sensitivity and carries no confirmatory claim.

Sign: delta = satisfaction(later) - satisfaction(earlier). The five confirmatory orientations are the REGISTERED ones (amendment B2) and are listed under `confirmatory_orientations`. The direction of the remaining five pairs follows a technical convention whose only purpose is to give each pair one stable sign in the descriptive matrix; it encodes no hierarchy and no ranking of the architectures. Registered expectation for the five: delta_low < 0 and delta_high > 0.

Registered orientations of the confirmatory five (amendment B2):

- `C1->C2` — delta = sat(c2_verify_revise) - sat(c1_react)
- `C2->C4` — delta = sat(c4_planner_critic) - sat(c2_verify_revise)
- `C4->C3` — delta = sat(c3_mas) - sat(c4_planner_critic)
- `C1->C5` — delta = sat(c5_best_of_3) - sat(c1_react)
- `C5->C3` — delta = sat(c3_mas) - sat(c5_best_of_3)

Every other pair is oriented by a bookkeeping convention (`c1_react < c2_verify_revise < c5_best_of_3 < c4_planner_critic < c3_mas`) that exists only to keep one sign per pair. Bookkeeping only: it fixes the direction of each unordered pair so a contrast keeps one sign throughout. Not a hierarchy, not a ranking, and never a basis for reading a result.

3960 runs over 198 instances, model `Qwen/Qwen3-32B-AWQ`. Completeness audit: 3960/3960 verified, passed = True.

## The confirmatory family — primary cap 64000

Ten p-values, one Holm correction, α = 0.05, 99999 permutations.

| contrast | δ low | δ medium | δ high | p IUT | p IUT (Holm) | Δ | p interaction | p interaction (Holm) |
|---|---|---|---|---|---|---|---|---|
| C1->C2 | +0.052 | +0.115 | +0.183 | 1.0000 | 1.0000 | +0.132 | 0.0115 | 0.1146 |
| C2->C4 | +0.192 | +0.097 | +0.138 | 0.9991 | 1.0000 | -0.053 | 0.4952 | 1.0000 |
| C4->C3 | +0.048 | +0.092 | +0.023 | 0.7971 | 1.0000 | -0.025 | 0.7728 | 1.0000 |
| C1->C5 | -0.327 | -0.250 | -0.148 | 0.9925 | 1.0000 | +0.178 | 0.0594 | 0.5348 |
| C5->C3 | +0.618 | +0.553 | +0.493 | 1.0000 | 1.0000 | -0.125 | 0.1649 | 1.0000 |

*The medium column is descriptive and enters no test and no Holm family.*

### Crossover

| contrast | δ low < 0 | δ high > 0 | IUT rejects after Holm | crossover |
|---|---|---|---|---|
| C1->C2 | False | True | False | not established |
| C2->C4 | False | True | False | not established |
| C4->C3 | False | True | False | not established |
| C1->C5 | True | False | False | not established |
| C5->C3 | False | True | False | not established |

**No crossover is established.** Failure to establish crossover statistically is not evidence that no crossover exists. A non-rejection is consistent both with no crossover and with a real crossover this design could not resolve, and this report does not choose between those readings.

### Per-band estimates, with the pre-declared skewness diagnostic

| contrast | band | n | mean δ | 95% bootstrap CI (descriptive) | ties | skew |
|---|---|---|---|---|---|---|
| C1->C2 | low | 50 | +0.052 | [+0.005, +0.110] | 92% | +3.81 |
| C1->C2 | medium | 50 | +0.115 | [+0.052, +0.188] | 82% | +1.73 |
| C1->C2 | high | 50 | +0.183 | [+0.105, +0.267] | 70% | +1.20 |
| C2->C4 | low | 50 | +0.192 | [+0.077, +0.307] | 50% | +0.12 |
| C2->C4 | medium | 50 | +0.097 | [-0.007, +0.202] | 58% | +0.03 |
| C2->C4 | high | 50 | +0.138 | [+0.048, +0.232] | 46% | -0.01 |
| C4->C3 | low | 50 | +0.048 | [-0.070, +0.162] | 42% | -0.76 |
| C4->C3 | medium | 50 | +0.092 | [-0.047, +0.230] | 28% | -0.05 |
| C4->C3 | high | 50 | +0.023 | [-0.085, +0.128] | 32% | -0.24 |
| C1->C5 | low | 50 | -0.327 | [-0.465, -0.192] | 56% | -0.22 |
| C1->C5 | medium | 50 | -0.250 | [-0.363, -0.140] | 64% | -0.50 |
| C1->C5 | high | 50 | -0.148 | [-0.267, -0.035] | 64% | -0.77 |
| C5->C3 | low | 50 | +0.618 | [+0.495, +0.737] | 24% | -0.59 |
| C5->C3 | medium | 50 | +0.553 | [+0.433, +0.665] | 24% | -0.69 |
| C5->C3 | high | 50 | +0.493 | [+0.372, +0.615] | 26% | -0.43 |

The sign-flip permutation is exact under symmetry of the paired differences about zero, not merely a zero mean. Skewness is reported per band so that assumption is auditable rather than assumed away.

### Sign test — sensitivity only

A different functional, `P(δ > 0)`. It may not promote, demote or replace a primary result.

| contrast | p IUT (sign test) | p IUT (primary permutation) |
|---|---|---|
| C1->C2 | 1.0000 | 1.0000 |
| C2->C4 | 0.9980 | 0.9991 |
| C4->C3 | 0.9879 | 0.7971 |
| C1->C5 | 0.9846 | 0.9925 |
| C5->C3 | 1.0000 | 1.0000 |

## H sensitivity

Pre-declared sensitivity (plan section 7). Declared as sensitivity only: it cannot promote or demote the primary result. Delta uses the unequal-size label permutation because the strata are not balanced by design.

| stratum | contrast | n low | n high | mean δ low | mean δ high | p IUT | p interaction |
|---|---|---|---|---|---|---|---|
| H_eq_0 | C1->C2 | 0 | 28 | — | +0.208 | — | — |
| H_eq_0 | C2->C4 | 0 | 28 | — | +0.098 | — | — |
| H_eq_0 | C4->C3 | 0 | 28 | — | +0.066 | — | — |
| H_eq_0 | C1->C5 | 0 | 28 | — | -0.098 | — | — |
| H_eq_0 | C5->C3 | 0 | 28 | — | +0.470 | — | — |
| H_ge_1 | C1->C2 | 50 | 21 | +0.052 | +0.159 | 1.0000 | 0.0786 |
| H_ge_1 | C2->C4 | 50 | 21 | +0.192 | +0.175 | 0.9991 | 0.8678 |
| H_ge_1 | C4->C3 | 50 | 21 | +0.048 | -0.032 | 0.7964 | 0.4595 |
| H_ge_1 | C1->C5 | 50 | 21 | -0.327 | -0.222 | 0.9919 | 0.4017 |
| H_ge_1 | C5->C3 | 50 | 21 | +0.618 | +0.524 | 1.0000 | 0.4465 |

## Complete comparison matrix — descriptive

All ten pairwise contrasts, every band and size, every cap. The five given an architectural reading are marked; the rest complete the matrix, in the bookkeeping direction described above and with no hierarchy implied. No cell here carries a confirmatory claim, and the three non-primary caps are pre-declared secondary.

The full matrix is written to `contrasts.csv`. The primary-cap Block A rows are reproduced here.

| contrast | band | mean δ | 95% CI | n | family |
|---|---|---|---|---|---|
| C1->C2 | low | +0.052 | [+0.005, +0.110] | 50 | confirmatory |
| C1->C2 | medium | +0.115 | [+0.052, +0.188] | 50 | confirmatory |
| C1->C2 | high | +0.183 | [+0.105, +0.267] | 50 | confirmatory |
| C1->C5 | low | -0.327 | [-0.465, -0.192] | 50 | confirmatory |
| C1->C5 | medium | -0.250 | [-0.363, -0.140] | 50 | confirmatory |
| C1->C5 | high | -0.148 | [-0.267, -0.035] | 50 | confirmatory |
| C1->C4 | low | +0.243 | [+0.133, +0.352] | 50 | descriptive |
| C1->C4 | medium | +0.212 | [+0.115, +0.310] | 50 | descriptive |
| C1->C4 | high | +0.322 | [+0.232, +0.407] | 50 | descriptive |
| C1->C3 | low | +0.292 | [+0.113, +0.460] | 50 | descriptive |
| C1->C3 | medium | +0.303 | [+0.145, +0.457] | 50 | descriptive |
| C1->C3 | high | +0.345 | [+0.232, +0.463] | 50 | descriptive |
| C2->C5 | low | -0.378 | [-0.515, -0.243] | 50 | descriptive |
| C2->C5 | medium | -0.365 | [-0.473, -0.258] | 50 | descriptive |
| C2->C5 | high | -0.332 | [-0.440, -0.223] | 50 | descriptive |
| C2->C4 | low | +0.192 | [+0.077, +0.307] | 50 | confirmatory |
| C2->C4 | medium | +0.097 | [-0.007, +0.202] | 50 | confirmatory |
| C2->C4 | high | +0.138 | [+0.048, +0.232] | 50 | confirmatory |
| C2->C3 | low | +0.240 | [+0.063, +0.412] | 50 | descriptive |
| C2->C3 | medium | +0.188 | [+0.025, +0.348] | 50 | descriptive |
| C2->C3 | high | +0.162 | [+0.048, +0.275] | 50 | descriptive |
| C5->C4 | low | +0.570 | [+0.470, +0.667] | 50 | descriptive |
| C5->C4 | medium | +0.462 | [+0.353, +0.568] | 50 | descriptive |
| C5->C4 | high | +0.470 | [+0.373, +0.563] | 50 | descriptive |
| C5->C3 | low | +0.618 | [+0.495, +0.737] | 50 | confirmatory |
| C5->C3 | medium | +0.553 | [+0.433, +0.665] | 50 | confirmatory |
| C5->C3 | high | +0.493 | [+0.372, +0.615] | 50 | confirmatory |
| C4->C3 | low | +0.048 | [-0.070, +0.162] | 50 | confirmatory |
| C4->C3 | medium | +0.092 | [-0.047, +0.230] | 50 | confirmatory |
| C4->C3 | high | +0.023 | [-0.085, +0.128] | 50 | confirmatory |

## Standing of the parts

- **Medium band.** Descriptive (amendment A2). It receives an effect estimate and an interval, no confirmatory test, and no membership of the Holm family.
- **Block B (`n = 4/5/6`).** Descriptive (amendment B2). `n` is a generator parameter and never a complexity level, so Block B is not a D band and forms no confirmatory contrast.
- **Secondary caps.** 16000, 32000 and 128000 are run and reported, never pooled with the primary cap, and carry no confirmatory test.
- **Bootstrap.** Descriptive interval only.

---

*Failure to establish crossover statistically is not evidence that no crossover exists. This applies to each of the five contrasts separately, and nothing here licenses the reverse claim.*
