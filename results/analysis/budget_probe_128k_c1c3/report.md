# Budget probe — 64 000 vs 128 000, paired within instance

**Exploratory development probe. Descriptive, not a hypothesis test, not a gate.** No threshold is introduced and no cap is selected here.

48 runs over 12 frozen instances (4 per band), model `Qwen/Qwen3-32B-AWQ`, paired on the identical instance at both caps. Completeness audit passed.


## Satisfaction — the primary metric

| band | c1_react 64k | c1_react 128k | Δ c1_react | c3_mas 64k | c3_mas 128k | Δ c3_mas |
|---|---|---|---|---|---|---|
| low | 0.667 | 0.667 | +0.000 | 0.667 | 0.750 | +0.083 |
| medium | 0.167 | 0.167 | +0.000 | 0.792 | 0.792 | +0.000 |
| high | 0.000 | 0.000 | +0.000 | 0.604 | 0.792 | +0.188 |
| pooled | 0.278 | 0.278 | +0.000 | 0.688 | 0.778 | +0.090 |

## Was the extra budget used at all?

| condition | mean tokens 64k | mean tokens 128k | cap used 128k | runs over 64 000 | runs under 64 000 |
|---|---|---|---|---|---|
| c1_react | 43594 | 46967 | 0.37 | 4/12 | 8/12 |
| c3_mas | 42975 | 62168 | 0.49 | 5/12 | 7/12 |

If runs at 128 000 do not spend more than 64 000, the extra budget is unused and nothing else needs interpreting.


## Paired change, pooled

| condition | improved | worsened | unchanged | mean Δ satisfaction | gained a validated plan | lost one |
|---|---|---|---|---|---|---|
| c1_react | 0 | 0 | 12 | +0.000 | 0 | 0 |
| c3_mas | 3 | 0 | 9 | +0.090 | 1 | 0 |

Counts and means only; no hypothesis test is performed.


## Validated non-empty rate — secondary, diagnostic

A structural precondition for scoring at all, reported beside the primary metric and **never called performance or accuracy**.

| condition | 64k | 128k |
|---|---|---|
| c1_react | 4/12 = 0.33 | 4/12 = 0.33 |
| c3_mas | 10/12 = 0.83 | 11/12 = 0.92 |

## Cost and behaviour

| condition | cap | calls | proposal validity | termination |
|---|---|---|---|---|
| c1_react | 64000 | 17.5 | 0.235 | {'aborted': 3, 'agent_finish': 7, 'budget': 2} |
| c1_react | 128000 | 17.9 | 0.286 | {'aborted': 1, 'agent_finish': 11} |
| c3_mas | 64000 | 23.3 | 0.697 | {'aborted': 1, 'agent_finish': 11} |
| c3_mas | 128000 | 27.1 | 0.661 | {'agent_finish': 12} |

---

*4 instances per band supports description and ordering, not inference. No threshold is introduced and no cap is selected; whether 128000 enters the held-out design is a separate later decision.*

