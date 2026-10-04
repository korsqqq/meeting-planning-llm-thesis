# Held-out stage extraction — verification report

An extraction, not an analysis: no hypothesis test, p-value, correction or significance claim is computed anywhere in this pass, and no LLM was called.

Completeness audit: 3960/3960 verified, passed = True.

3960 runs -> 13379 stage rows, 6005 proposal rows.

## Agreement with the stored documents

| condition | runs | final plan matches | call totals match | token totals match |
|---|---|---|---|---|
| c1_react | 792 | 792/792 | 792/792 | 792/792 |
| c2_verify_revise | 792 | 792/792 | 792/792 | 792/792 |
| c3_mas | 792 | 792/792 | 792/792 | 792/792 |
| c4_planner_critic | 792 | 792/792 | 792/792 | 792/792 |
| c5_best_of_3 | 792 | 792/792 | 792/792 | 792/792 |

Every rebuilt final plan equals the stored `final_plan`, meeting for meeting. The reconstruction reproduces what the harness did.

## The reconstruction checked against a recorded field

Only C4 records `critic_improved`, so only there can the rebuilt critic improvement be compared with what the harness itself concluded. Where the comparison is possible it is made; where it is not, the row says so rather than reporting agreement it did not test.

| condition | runs where checkable | rebuilt agrees with recorded |
|---|---|---|
| c1_react | 0 | not recorded |
| c2_verify_revise | 0 | not recorded |
| c3_mas | 0 | not recorded |
| c4_planner_critic | 792 | 792/792 |
| c5_best_of_3 | 0 | not recorded |

## Where the numbers come from

| condition | runs with a diagnostics block | stage quality reconstructed | stage quality unknown |
|---|---|---|---|
| c1_react | 0/792 | 792 | 0 |
| c2_verify_revise | 0/792 | 1448 | 26 |
| c3_mas | 0/792 | 1236 | 0 |
| c4_planner_critic | 792/792 | 1239 | 0 |
| c5_best_of_3 | 792/792 | 0 | 0 |

`unknown` is left empty in the CSV and is never written as zero: a stage whose proposals cannot be assigned to a cycle unambiguously has no measured quality, which is a different fact from a stage that measured no improvement.

## Stages observed

| condition | stage | rows | runs with at least one proposal |
|---|---|---|---|
| c1_react | finalize | 792 | 0 |
| c1_react | react_step | 792 | 534 |
| c2_verify_revise | finalize | 792 | 0 |
| c2_verify_revise | react_step | 792 | 534 |
| c2_verify_revise | revise | 682 | 677 |
| c2_verify_revise | verify | 718 | 0 |
| c3_mas | aggregate | 792 | 0 |
| c3_mas | critic | 444 | 440 |
| c3_mas | finalize | 792 | 0 |
| c3_mas | worker_a | 792 | 420 |
| c3_mas | worker_b | 792 | 424 |
| c4_planner_critic | critic | 447 | 444 |
| c4_planner_critic | finalize | 792 | 0 |
| c4_planner_critic | planner | 792 | 447 |
| c5_best_of_3 | bon_attempt_0 | 792 | 224 |
| c5_best_of_3 | bon_attempt_1 | 792 | 225 |
| c5_best_of_3 | bon_attempt_2 | 792 | 255 |
| c5_best_of_3 | finalize | 792 | 0 |

## Proposal outcomes

Counted over proposal attempts, not over runs. Interval-style summaries over runs belong to the analysis step, not to this extraction.

| condition | accepted | invalid_constraints | valid_not_longer |
|---|---|---|---|
| c1_react | 242 | 616 | 22 |
| c2_verify_revise | 333 | 887 | 324 |
| c3_mas | 711 | 348 | 452 |
| c4_planner_critic | 449 | 517 | 185 |
| c5_best_of_3 | 264 | 634 | 21 |

---

*Every column in the two tables is either read from a recorded field or rebuilt from recorded fields, and the `*_source` columns say which. Nothing here establishes a cause.*
