# Budget calibration — the registered primary-cap rule applied

The rule was fixed in THESIS_DECISIONS section 5 before this development set existed, and the ladder was named explicitly before any run was launched. This document applies it; it does not choose it.

Run set: **240** runs, condition `c1_react`, model `Qwen/Qwen3-32B-AWQ`, subset `911e4864d6fb`, seeds 30000–30063. Completeness audit passed.

**Success**, structurally defined: the final plan is validator-approved *and* contains at least one meeting. Nothing else enters the decision — not satisfaction, not the optimality gap, not the proposal taxonomy.


## The quantity the rule reads

| cap | Low | Medium | High | all bands ≥ 50% |
|---|---|---|---|---|
| 8000 | 0/20 = 0.00 ✗ | 0/20 = 0.00 ✗ | 0/20 = 0.00 ✗ | no |
| 16000 | 0/20 = 0.00 ✗ | 0/20 = 0.00 ✗ | 1/20 = 0.05 ✗ | no |
| 32000 | 0/20 = 0.00 ✗ | 2/20 = 0.10 ✗ | 1/20 = 0.05 ✗ | no |
| 64000 | 9/20 = 0.45 ✗ | 9/20 = 0.45 ✗ | 5/20 = 0.25 ✗ | no |

## Verdict

### NO CAP SELECTED

No rung of the frozen ladder reaches 50% qualifying runs in all three density bands.

**budget calibration is declared failed and the main experiment does not start; falling back to the top rung because nothing qualified is not available**

The bands, the estimand and the inference rule are unaffected: what failed is the budget at which this single-agent baseline can be run at all, which is a finding about the condition and the instance size, not about the design.


## What the budget bought — non-binding

Reported because the next step reads it, and because it describes the runs. None of it entered the decision above.

| cap | band | mean sat | zeros | non-empty plan | mean tokens | cap used | proposal validity |
|---|---|---|---|---|---|---|---|
| 8000 | low | 0.000 | 20/20 | 0.00 | 7617 | 0.95 | — |
| 8000 | medium | 0.000 | 20/20 | 0.00 | 7736 | 0.97 | — |
| 8000 | high | 0.000 | 20/20 | 0.00 | 7671 | 0.96 | — |
| 16000 | low | 0.000 | 20/20 | 0.00 | 15701 | 0.98 | — |
| 16000 | medium | 0.000 | 20/20 | 0.00 | 15761 | 0.98 | 0.000 |
| 16000 | high | 0.033 | 19/20 | 0.05 | 15460 | 0.97 | 0.200 |
| 32000 | low | 0.000 | 20/20 | 0.00 | 30522 | 0.95 | 0.000 |
| 32000 | medium | 0.087 | 18/20 | 0.10 | 30944 | 0.97 | 0.143 |
| 32000 | high | 0.033 | 19/20 | 0.05 | 28060 | 0.88 | 0.091 |
| 64000 | low | 0.362 | 11/20 | 0.45 | 49127 | 0.77 | 0.310 |
| 64000 | medium | 0.358 | 11/20 | 0.45 | 50203 | 0.78 | 0.281 |
| 64000 | high | 0.179 | 15/20 | 0.25 | 46691 | 0.73 | 0.156 |
