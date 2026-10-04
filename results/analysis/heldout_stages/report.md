# Held-out stage tables — descriptive

Counts and means over the committed stage extraction. **No hypothesis test, p-value, correction, interval or significance claim.** Where a stage is followed by a better plan this is the **observed gain of the stage**, never its effect: nothing here identifies what would have happened without it.

Every count is over distinct runs. Block A is broken out by conflict-density band and Block B by task size; the `all` row pools them and is an overview only.

## 1. Runs that never produced a proposal

The outcome that dominates the low caps. Reported as an outcome, not as the established cause of the low scores.

**Block A — all**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 146/150 | 55/150 | 6/150 | 5/150 |
| c2_verify_revise | 146/150 | 54/150 | 4/150 | 3/150 |
| c3_mas | 143/150 | 102/150 | 4/150 | 0/150 |
| c4_planner_critic | 148/150 | 113/150 | 14/150 | 5/150 |
| c5_best_of_3 | 148/150 | 145/150 | 89/150 | 0/150 |

**Block A — low**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 49/50 | 17/50 | 2/50 | 1/50 |
| c2_verify_revise | 49/50 | 17/50 | 2/50 | 1/50 |
| c3_mas | 49/50 | 34/50 | 2/50 | 0/50 |
| c4_planner_critic | 49/50 | 38/50 | 4/50 | 1/50 |
| c5_best_of_3 | 50/50 | 49/50 | 32/50 | 0/50 |

**Block A — medium**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 48/50 | 21/50 | 3/50 | 3/50 |
| c2_verify_revise | 48/50 | 20/50 | 1/50 | 1/50 |
| c3_mas | 47/50 | 32/50 | 2/50 | 0/50 |
| c4_planner_critic | 49/50 | 37/50 | 9/50 | 3/50 |
| c5_best_of_3 | 50/50 | 49/50 | 31/50 | 0/50 |

**Block A — high**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 49/50 | 17/50 | 1/50 | 1/50 |
| c2_verify_revise | 49/50 | 17/50 | 1/50 | 1/50 |
| c3_mas | 47/50 | 36/50 | 0/50 | 0/50 |
| c4_planner_critic | 50/50 | 38/50 | 1/50 | 1/50 |
| c5_best_of_3 | 48/50 | 47/50 | 26/50 | 0/50 |

**Block B — all**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 40/48 | 6/48 | 0/48 | 0/48 |
| c2_verify_revise | 40/48 | 6/48 | 0/48 | 0/48 |
| c3_mas | 29/48 | 4/48 | 0/48 | 0/48 |
| c4_planner_critic | 44/48 | 20/48 | 1/48 | 0/48 |
| c5_best_of_3 | 48/48 | 43/48 | 5/48 | 0/48 |

**Block B — n=4**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 14/16 | 1/16 | 0/16 | 0/16 |
| c2_verify_revise | 14/16 | 1/16 | 0/16 | 0/16 |
| c3_mas | 7/16 | 0/16 | 0/16 | 0/16 |
| c4_planner_critic | 15/16 | 4/16 | 0/16 | 0/16 |
| c5_best_of_3 | 16/16 | 15/16 | 1/16 | 0/16 |

**Block B — n=5**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 10/16 | 3/16 | 0/16 | 0/16 |
| c2_verify_revise | 10/16 | 3/16 | 0/16 | 0/16 |
| c3_mas | 10/16 | 0/16 | 0/16 | 0/16 |
| c4_planner_critic | 13/16 | 8/16 | 1/16 | 0/16 |
| c5_best_of_3 | 16/16 | 12/16 | 1/16 | 0/16 |

**Block B — n=6**

| condition | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| c1_react | 16/16 | 2/16 | 0/16 | 0/16 |
| c2_verify_revise | 16/16 | 2/16 | 0/16 | 0/16 |
| c3_mas | 12/16 | 4/16 | 0/16 | 0/16 |
| c4_planner_critic | 16/16 | 8/16 | 0/16 | 0/16 |
| c5_best_of_3 | 16/16 | 16/16 | 3/16 | 0/16 |

## 2. What became of the critic

Four mutually exclusive fates per run, summing to the runs in the cell. C4 reads the harness's own `critic_improved`; C3 records no such field and is reconstructed, which the `source` column states.

Two shares are given. `improved_share_of_runs` describes the architecture as it ran; `improved_share_of_reached` describes the stage where it was actually invoked. They answer different questions and neither replaces the other.

**Block A**

| stratum | condition | cap | runs | not_reached | reached_no_proposal | proposed_not_improved | improved | improved_share_of_runs | improved_share_of_reached | source |
|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | 150 | 146 | 0 | 4 | 0 | 0.0 | 0.0 | reconstructed |
| all | c3_mas | 32000 | 150 | 121 | 0 | 29 | 0 | 0.0 | 0.0 | reconstructed |
| all | c3_mas | 64000 | 150 | 33 | 2 | 113 | 2 | 0.013 | 0.017 | reconstructed |
| all | c3_mas | 128000 | 150 | 11 | 2 | 128 | 9 | 0.06 | 0.065 | reconstructed |
| all | c4_planner_critic | 16000 | 150 | 148 | 0 | 1 | 1 | 0.007 | 0.5 | recorded |
| all | c4_planner_critic | 32000 | 150 | 113 | 0 | 14 | 23 | 0.153 | 0.622 | recorded |
| all | c4_planner_critic | 64000 | 150 | 14 | 1 | 52 | 83 | 0.553 | 0.61 | recorded |
| all | c4_planner_critic | 128000 | 150 | 5 | 1 | 70 | 74 | 0.493 | 0.51 | recorded |
| low | c3_mas | 16000 | 50 | 50 | 0 | 0 | 0 | 0.0 |  | reconstructed |
| low | c3_mas | 32000 | 50 | 37 | 0 | 13 | 0 | 0.0 | 0.0 | reconstructed |
| low | c3_mas | 64000 | 50 | 10 | 2 | 38 | 0 | 0.0 | 0.0 | reconstructed |
| low | c3_mas | 128000 | 50 | 1 | 2 | 46 | 1 | 0.02 | 0.02 | reconstructed |
| low | c4_planner_critic | 16000 | 50 | 49 | 0 | 1 | 0 | 0.0 | 0.0 | recorded |
| low | c4_planner_critic | 32000 | 50 | 38 | 0 | 6 | 6 | 0.12 | 0.5 | recorded |
| low | c4_planner_critic | 64000 | 50 | 4 | 0 | 22 | 24 | 0.48 | 0.522 | recorded |
| low | c4_planner_critic | 128000 | 50 | 1 | 0 | 26 | 23 | 0.46 | 0.469 | recorded |
| medium | c3_mas | 16000 | 50 | 48 | 0 | 2 | 0 | 0.0 | 0.0 | reconstructed |
| medium | c3_mas | 32000 | 50 | 41 | 0 | 9 | 0 | 0.0 | 0.0 | reconstructed |
| medium | c3_mas | 64000 | 50 | 12 | 0 | 38 | 0 | 0.0 | 0.0 | reconstructed |
| medium | c3_mas | 128000 | 50 | 6 | 0 | 42 | 2 | 0.04 | 0.045 | reconstructed |
| medium | c4_planner_critic | 16000 | 50 | 49 | 0 | 0 | 1 | 0.02 | 1.0 | recorded |
| medium | c4_planner_critic | 32000 | 50 | 37 | 0 | 5 | 8 | 0.16 | 0.615 | recorded |
| medium | c4_planner_critic | 64000 | 50 | 9 | 0 | 20 | 21 | 0.42 | 0.512 | recorded |
| medium | c4_planner_critic | 128000 | 50 | 3 | 1 | 25 | 21 | 0.42 | 0.447 | recorded |
| high | c3_mas | 16000 | 50 | 48 | 0 | 2 | 0 | 0.0 | 0.0 | reconstructed |
| high | c3_mas | 32000 | 50 | 43 | 0 | 7 | 0 | 0.0 | 0.0 | reconstructed |
| high | c3_mas | 64000 | 50 | 11 | 0 | 37 | 2 | 0.04 | 0.051 | reconstructed |
| high | c3_mas | 128000 | 50 | 4 | 0 | 40 | 6 | 0.12 | 0.13 | reconstructed |
| high | c4_planner_critic | 16000 | 50 | 50 | 0 | 0 | 0 | 0.0 |  | recorded |
| high | c4_planner_critic | 32000 | 50 | 38 | 0 | 3 | 9 | 0.18 | 0.75 | recorded |
| high | c4_planner_critic | 64000 | 50 | 1 | 1 | 10 | 38 | 0.76 | 0.776 | recorded |
| high | c4_planner_critic | 128000 | 50 | 1 | 0 | 19 | 30 | 0.6 | 0.612 | recorded |

**Block B**

| stratum | condition | cap | runs | not_reached | reached_no_proposal | proposed_not_improved | improved | improved_share_of_runs | improved_share_of_reached | source |
|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | 48 | 30 | 0 | 18 | 0 | 0.0 | 0.0 | reconstructed |
| all | c3_mas | 32000 | 48 | 6 | 0 | 35 | 7 | 0.146 | 0.167 | reconstructed |
| all | c3_mas | 64000 | 48 | 1 | 0 | 38 | 9 | 0.188 | 0.191 | reconstructed |
| all | c3_mas | 128000 | 48 | 0 | 0 | 39 | 9 | 0.188 | 0.188 | reconstructed |
| all | c4_planner_critic | 16000 | 48 | 44 | 1 | 0 | 3 | 0.062 | 0.75 | recorded |
| all | c4_planner_critic | 32000 | 48 | 20 | 0 | 8 | 20 | 0.417 | 0.714 | recorded |
| all | c4_planner_critic | 64000 | 48 | 1 | 0 | 22 | 25 | 0.521 | 0.532 | recorded |
| all | c4_planner_critic | 128000 | 48 | 0 | 0 | 26 | 22 | 0.458 | 0.458 | recorded |
| n=4 | c3_mas | 16000 | 16 | 7 | 0 | 9 | 0 | 0.0 | 0.0 | reconstructed |
| n=4 | c3_mas | 32000 | 16 | 0 | 0 | 9 | 7 | 0.438 | 0.438 | reconstructed |
| n=4 | c3_mas | 64000 | 16 | 0 | 0 | 9 | 7 | 0.438 | 0.438 | reconstructed |
| n=4 | c3_mas | 128000 | 16 | 0 | 0 | 9 | 7 | 0.438 | 0.438 | reconstructed |
| n=4 | c4_planner_critic | 16000 | 16 | 15 | 0 | 0 | 1 | 0.062 | 1.0 | recorded |
| n=4 | c4_planner_critic | 32000 | 16 | 4 | 0 | 6 | 6 | 0.375 | 0.5 | recorded |
| n=4 | c4_planner_critic | 64000 | 16 | 0 | 0 | 10 | 6 | 0.375 | 0.375 | recorded |
| n=4 | c4_planner_critic | 128000 | 16 | 0 | 0 | 10 | 6 | 0.375 | 0.375 | recorded |
| n=5 | c3_mas | 16000 | 16 | 10 | 0 | 6 | 0 | 0.0 | 0.0 | reconstructed |
| n=5 | c3_mas | 32000 | 16 | 0 | 0 | 16 | 0 | 0.0 | 0.0 | reconstructed |
| n=5 | c3_mas | 64000 | 16 | 0 | 0 | 14 | 2 | 0.125 | 0.125 | reconstructed |
| n=5 | c3_mas | 128000 | 16 | 0 | 0 | 14 | 2 | 0.125 | 0.125 | reconstructed |
| n=5 | c4_planner_critic | 16000 | 16 | 13 | 1 | 0 | 2 | 0.125 | 0.667 | recorded |
| n=5 | c4_planner_critic | 32000 | 16 | 8 | 0 | 1 | 7 | 0.438 | 0.875 | recorded |
| n=5 | c4_planner_critic | 64000 | 16 | 1 | 0 | 5 | 10 | 0.625 | 0.667 | recorded |
| n=5 | c4_planner_critic | 128000 | 16 | 0 | 0 | 7 | 9 | 0.562 | 0.562 | recorded |
| n=6 | c3_mas | 16000 | 16 | 13 | 0 | 3 | 0 | 0.0 | 0.0 | reconstructed |
| n=6 | c3_mas | 32000 | 16 | 6 | 0 | 10 | 0 | 0.0 | 0.0 | reconstructed |
| n=6 | c3_mas | 64000 | 16 | 1 | 0 | 15 | 0 | 0.0 | 0.0 | reconstructed |
| n=6 | c3_mas | 128000 | 16 | 0 | 0 | 16 | 0 | 0.0 | 0.0 | reconstructed |
| n=6 | c4_planner_critic | 16000 | 16 | 16 | 0 | 0 | 0 | 0.0 |  | recorded |
| n=6 | c4_planner_critic | 32000 | 16 | 8 | 0 | 1 | 7 | 0.438 | 0.875 | recorded |
| n=6 | c4_planner_critic | 64000 | 16 | 0 | 0 | 7 | 9 | 0.562 | 0.562 | recorded |
| n=6 | c4_planner_critic | 128000 | 16 | 0 | 0 | 9 | 7 | 0.438 | 0.438 | recorded |

## 3. The observed gain of the critic stage, and its price

`mean_sat_gain_*` averages per-run differences in satisfaction rather than differencing two means: satisfaction is a ratio to a per-instance optimum and the optima differ across instances. As in table 2, the two denominators are both reported.

**Block A**

| stratum | condition | cap | runs | runs_critic_ran | mean_meetings_before | mean_meetings_after | mean_added_when_critic_ran | mean_added_over_all_runs | mean_sat_gain_when_critic_ran | mean_sat_gain_over_all_runs | mean_critic_tokens | mean_critic_latency_s | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | 150 | 4 | 1.5 | 1.5 | 0.0 | 0.0 | 0.0 | 0.0 | 1129.0 | 10.322 | reconstructed |
| all | c3_mas | 32000 | 150 | 29 | 2.379 | 2.379 | 0.0 | 0.0 | 0.0 | 0.0 | 1675.414 | 16.972 | reconstructed |
| all | c3_mas | 64000 | 150 | 117 | 2.786 | 2.803 | 0.017 | 0.013 | 0.004 | 0.0033 | 4947.538 | 60.987 | reconstructed |
| all | c3_mas | 128000 | 150 | 139 | 2.914 | 2.978 | 0.065 | 0.06 | 0.018 | 0.0167 | 8391.871 | 107.649 | reconstructed |
| all | c4_planner_critic | 16000 | 150 | 2 | 1.0 | 2.0 | 1.0 | 0.013 | 0.25 | 0.0033 | 1823.0 | 18.27 | recorded |
| all | c4_planner_critic | 32000 | 150 | 37 | 0.649 | 1.568 | 0.919 | 0.227 | 0.268 | 0.0661 | 3163.946 | 35.954 | recorded |
| all | c4_planner_critic | 64000 | 150 | 136 | 1.037 | 2.169 | 1.132 | 1.027 | 0.335 | 0.3033 | 3865.809 | 45.197 | recorded |
| all | c4_planner_critic | 128000 | 150 | 145 | 1.262 | 2.207 | 0.945 | 0.913 | 0.28 | 0.2711 | 4271.738 | 50.528 | recorded |
| low | c3_mas | 16000 | 50 | 0 |  |  |  | 0.0 |  | 0.0 |  |  | reconstructed |
| low | c3_mas | 32000 | 50 | 13 | 2.923 | 2.923 | 0.0 | 0.0 | 0.0 | 0.0 | 1815.846 | 18.322 | reconstructed |
| low | c3_mas | 64000 | 50 | 40 | 3.15 | 3.15 | 0.0 | 0.0 | 0.0 | 0.0 | 6837.575 | 86.831 | reconstructed |
| low | c3_mas | 128000 | 50 | 49 | 3.204 | 3.224 | 0.02 | 0.02 | 0.007 | 0.0067 | 10306.51 | 133.918 | reconstructed |
| low | c4_planner_critic | 16000 | 50 | 1 | 2.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1240.0 | 11.097 | recorded |
| low | c4_planner_critic | 32000 | 50 | 12 | 1.0 | 1.583 | 0.583 | 0.14 | 0.181 | 0.0433 | 3031.333 | 34.229 | recorded |
| low | c4_planner_critic | 64000 | 50 | 46 | 1.413 | 2.5 | 1.087 | 1.0 | 0.315 | 0.29 | 3576.587 | 41.051 | recorded |
| low | c4_planner_critic | 128000 | 50 | 49 | 1.551 | 2.531 | 0.98 | 0.96 | 0.282 | 0.2767 | 3699.878 | 42.538 | recorded |
| medium | c3_mas | 16000 | 50 | 2 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 918.5 | 7.939 | reconstructed |
| medium | c3_mas | 32000 | 50 | 9 | 2.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1676.222 | 17.363 | reconstructed |
| medium | c3_mas | 64000 | 50 | 38 | 2.684 | 2.684 | 0.0 | 0.0 | 0.0 | 0.0 | 4274.395 | 51.726 | reconstructed |
| medium | c3_mas | 128000 | 50 | 44 | 2.795 | 2.841 | 0.045 | 0.04 | 0.013 | 0.0117 | 7265.364 | 92.412 | reconstructed |
| medium | c4_planner_critic | 16000 | 50 | 1 | 0.0 | 2.0 | 2.0 | 0.04 | 0.5 | 0.01 | 2406.0 | 25.444 | recorded |
| medium | c4_planner_critic | 32000 | 50 | 13 | 0.538 | 1.538 | 1.0 | 0.26 | 0.282 | 0.0733 | 3460.538 | 39.495 | recorded |
| medium | c4_planner_critic | 64000 | 50 | 41 | 1.073 | 2.073 | 1.0 | 0.82 | 0.291 | 0.2383 | 3635.707 | 41.884 | recorded |
| medium | c4_planner_critic | 128000 | 50 | 47 | 1.298 | 2.106 | 0.809 | 0.76 | 0.241 | 0.2267 | 4455.319 | 53.533 | recorded |
| high | c3_mas | 16000 | 50 | 2 | 2.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1339.5 | 12.705 | reconstructed |
| high | c3_mas | 32000 | 50 | 7 | 1.857 | 1.857 | 0.0 | 0.0 | 0.0 | 0.0 | 1413.571 | 13.963 | reconstructed |
| high | c3_mas | 64000 | 50 | 39 | 2.513 | 2.564 | 0.051 | 0.04 | 0.013 | 0.01 | 3664.923 | 43.504 | reconstructed |
| high | c3_mas | 128000 | 50 | 46 | 2.717 | 2.848 | 0.13 | 0.12 | 0.034 | 0.0317 | 7429.891 | 94.242 | reconstructed |
| high | c4_planner_critic | 16000 | 50 | 0 |  |  |  | 0.0 |  | 0.0 |  |  | recorded |
| high | c4_planner_critic | 32000 | 50 | 12 | 0.417 | 1.583 | 1.167 | 0.28 | 0.34 | 0.0817 | 2975.25 | 33.842 | recorded |
| high | c4_planner_critic | 64000 | 50 | 49 | 0.653 | 1.939 | 1.286 | 1.26 | 0.389 | 0.3817 | 4329.857 | 51.862 | recorded |
| high | c4_planner_critic | 128000 | 50 | 49 | 0.939 | 1.98 | 1.041 | 1.02 | 0.316 | 0.31 | 4667.51 | 55.636 | recorded |

**Block B**

| stratum | condition | cap | runs | runs_critic_ran | mean_meetings_before | mean_meetings_after | mean_added_when_critic_ran | mean_added_over_all_runs | mean_sat_gain_when_critic_ran | mean_sat_gain_over_all_runs | mean_critic_tokens | mean_critic_latency_s | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | 48 | 18 | 1.333 | 1.333 | 0.0 | 0.0 | 0.0 | 0.0 | 1072.556 | 9.731 | reconstructed |
| all | c3_mas | 32000 | 48 | 42 | 2.024 | 2.19 | 0.167 | 0.146 | 0.056 | 0.0486 | 3907.452 | 47.225 | reconstructed |
| all | c3_mas | 64000 | 48 | 47 | 2.426 | 2.617 | 0.191 | 0.188 | 0.064 | 0.0625 | 6976.489 | 88.151 | reconstructed |
| all | c3_mas | 128000 | 48 | 48 | 2.5 | 2.688 | 0.188 | 0.188 | 0.062 | 0.0625 | 7549.417 | 95.619 | reconstructed |
| all | c4_planner_critic | 16000 | 48 | 4 | 0.0 | 1.0 | 1.0 | 0.083 | 0.333 | 0.0278 | 2919.75 | 32.498 | recorded |
| all | c4_planner_critic | 32000 | 48 | 28 | 0.857 | 2.214 | 1.357 | 0.792 | 0.452 | 0.2639 | 3603.821 | 41.443 | recorded |
| all | c4_planner_critic | 64000 | 48 | 47 | 1.319 | 2.34 | 1.021 | 1.0 | 0.34 | 0.3333 | 4095.83 | 48.03 | recorded |
| all | c4_planner_critic | 128000 | 48 | 48 | 1.479 | 2.375 | 0.896 | 0.896 | 0.299 | 0.2986 | 4382.375 | 51.784 | recorded |
| n=4 | c3_mas | 16000 | 16 | 9 | 1.222 | 1.222 | 0.0 | 0.0 | 0.0 | 0.0 | 1015.778 | 9.069 | reconstructed |
| n=4 | c3_mas | 32000 | 16 | 16 | 1.812 | 2.25 | 0.438 | 0.438 | 0.146 | 0.1458 | 4584.5 | 55.965 | reconstructed |
| n=4 | c3_mas | 64000 | 16 | 16 | 2.312 | 2.75 | 0.438 | 0.438 | 0.146 | 0.1458 | 6125.688 | 76.185 | reconstructed |
| n=4 | c3_mas | 128000 | 16 | 16 | 2.312 | 2.75 | 0.438 | 0.438 | 0.146 | 0.1458 | 6125.688 | 76.139 | reconstructed |
| n=4 | c4_planner_critic | 16000 | 16 | 1 | 0.0 | 1.0 | 1.0 | 0.062 | 0.333 | 0.0208 | 2037.0 | 20.641 | recorded |
| n=4 | c4_planner_critic | 32000 | 16 | 12 | 1.5 | 2.417 | 0.917 | 0.688 | 0.306 | 0.2292 | 3377.333 | 38.488 | recorded |
| n=4 | c4_planner_critic | 64000 | 16 | 16 | 1.875 | 2.562 | 0.688 | 0.688 | 0.229 | 0.2292 | 3425.562 | 39.257 | recorded |
| n=4 | c4_planner_critic | 128000 | 16 | 16 | 1.875 | 2.562 | 0.688 | 0.688 | 0.229 | 0.2292 | 3425.562 | 39.121 | recorded |
| n=5 | c3_mas | 16000 | 16 | 6 | 1.5 | 1.5 | 0.0 | 0.0 | 0.0 | 0.0 | 1130.667 | 10.393 | reconstructed |
| n=5 | c3_mas | 32000 | 16 | 16 | 2.125 | 2.125 | 0.0 | 0.0 | 0.0 | 0.0 | 4306.125 | 53.075 | reconstructed |
| n=5 | c3_mas | 64000 | 16 | 16 | 2.375 | 2.5 | 0.125 | 0.125 | 0.042 | 0.0417 | 7073.25 | 89.287 | reconstructed |
| n=5 | c3_mas | 128000 | 16 | 16 | 2.375 | 2.5 | 0.125 | 0.125 | 0.042 | 0.0417 | 7073.25 | 89.238 | reconstructed |
| n=5 | c4_planner_critic | 16000 | 16 | 3 | 0.0 | 1.0 | 1.0 | 0.188 | 0.333 | 0.0625 | 3214.0 | 36.45 | recorded |
| n=5 | c4_planner_critic | 32000 | 16 | 8 | 0.375 | 2.25 | 1.875 | 0.938 | 0.625 | 0.3125 | 4349.375 | 51.015 | recorded |
| n=5 | c4_planner_critic | 64000 | 16 | 15 | 0.933 | 2.267 | 1.333 | 1.25 | 0.444 | 0.4167 | 3958.733 | 46.312 | recorded |
| n=5 | c4_planner_critic | 128000 | 16 | 16 | 1.25 | 2.375 | 1.125 | 1.125 | 0.375 | 0.375 | 4031.625 | 47.133 | recorded |
| n=6 | c3_mas | 16000 | 16 | 3 | 1.333 | 1.333 | 0.0 | 0.0 | 0.0 | 0.0 | 1126.667 | 10.393 | reconstructed |
| n=6 | c3_mas | 32000 | 16 | 10 | 2.2 | 2.2 | 0.0 | 0.0 | 0.0 | 0.0 | 2186.3 | 23.879 | reconstructed |
| n=6 | c3_mas | 64000 | 16 | 15 | 2.6 | 2.6 | 0.0 | 0.0 | 0.0 | 0.0 | 7780.8 | 99.704 | reconstructed |
| n=6 | c3_mas | 128000 | 16 | 16 | 2.812 | 2.812 | 0.0 | 0.0 | 0.0 | 0.0 | 9449.312 | 121.479 | reconstructed |
| n=6 | c4_planner_critic | 16000 | 16 | 0 |  |  |  | 0.0 |  | 0.0 |  |  | recorded |
| n=6 | c4_planner_critic | 32000 | 16 | 8 | 0.375 | 1.875 | 1.5 | 0.75 | 0.5 | 0.25 | 3198.0 | 36.306 | recorded |
| n=6 | c4_planner_critic | 64000 | 16 | 16 | 1.125 | 2.188 | 1.062 | 1.062 | 0.354 | 0.3542 | 4894.625 | 58.411 | recorded |
| n=6 | c4_planner_critic | 128000 | 16 | 16 | 1.312 | 2.188 | 0.875 | 0.875 | 0.292 | 0.2917 | 5689.938 | 69.097 | recorded |

## 4. What room the critic had

`headroom = pool_size − fallback_size`: candidates the critic could add on top of the fallback. `room_to_optimum = optimum − fallback_size`: how much more the oracle proves attainable on the instance.

**A positive headroom is not an available improvement.** Extra candidates still have to fit the travel and window constraints alongside the people already scheduled, and the pool need not contain a set that reaches the optimum. The table therefore says what room existed, not what was achievable.

Where headroom is zero the critic cannot improve at all, so those runs are separated rather than averaged in. `mean_travel_coverage` sits inside the strata because in these data coverage falls as pool size rises — an observation about this sample, not an identity — so a cross-tabulation against coverage alone would reproduce the headroom axis inverted.

**Block A**

| stratum | condition | cap | headroom_stratum | runs_critic_ran | improved | improved_share | mean_pool_size | mean_fallback_size | mean_headroom | mean_room_to_optimum | share_fallback_at_optimum | mean_travel_coverage | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | headroom = 0 | 4 | 0 | 0.0 | 1.5 | 1.5 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| all | c3_mas | 32000 | headroom = 0 | 29 | 0 | 0.0 | 2.379 | 2.379 | 0.0 | 1.0 | 0.31 |  | reconstructed |
| all | c3_mas | 64000 | headroom = 0 | 80 | 0 | 0.0 | 2.562 | 2.562 | 0.0 | 0.775 | 0.438 |  | reconstructed |
| all | c3_mas | 64000 | headroom >= 1 | 37 | 2 | 0.054 | 6.054 | 3.27 | 2.784 | 0.297 | 0.73 |  | reconstructed |
| all | c3_mas | 128000 | headroom = 0 | 45 | 0 | 0.0 | 2.489 | 2.489 | 0.0 | 0.822 | 0.422 |  | reconstructed |
| all | c3_mas | 128000 | headroom >= 1 | 94 | 9 | 0.096 | 5.777 | 3.117 | 2.66 | 0.277 | 0.734 |  | reconstructed |
| all | c4_planner_critic | 16000 | headroom = 0 | 1 | 0 | 0.0 | 2.0 | 2.0 | 0.0 | 2.0 | 0.0 | 0.333 | recorded |
| all | c4_planner_critic | 16000 | headroom >= 1 | 1 | 1 | 1.0 | 3.0 | 0.0 | 3.0 | 4.0 | 0.0 | 0.167 | recorded |
| all | c4_planner_critic | 32000 | headroom = 0 | 9 | 0 | 0.0 | 2.444 | 2.444 | 0.0 | 0.778 | 0.333 | 0.306 | recorded |
| all | c4_planner_critic | 32000 | headroom >= 1 | 28 | 23 | 0.821 | 3.25 | 0.071 | 3.179 | 3.429 | 0.0 | 0.193 | recorded |
| all | c4_planner_critic | 64000 | headroom = 0 | 41 | 0 | 0.0 | 2.756 | 2.756 | 0.0 | 0.488 | 0.561 | 0.33 | recorded |
| all | c4_planner_critic | 64000 | headroom >= 1 | 95 | 83 | 0.874 | 3.958 | 0.295 | 3.663 | 3.116 | 0.032 | 0.204 | recorded |
| all | c4_planner_critic | 128000 | headroom = 0 | 43 | 0 | 0.0 | 2.791 | 2.791 | 0.0 | 0.442 | 0.581 | 0.337 | recorded |
| all | c4_planner_critic | 128000 | headroom >= 1 | 102 | 74 | 0.725 | 4.343 | 0.618 | 3.725 | 2.794 | 0.088 | 0.201 | recorded |
| low | c3_mas | 32000 | headroom = 0 | 13 | 0 | 0.0 | 2.923 | 2.923 | 0.0 | 0.462 | 0.538 |  | reconstructed |
| low | c3_mas | 64000 | headroom = 0 | 25 | 0 | 0.0 | 2.88 | 2.88 | 0.0 | 0.36 | 0.64 |  | reconstructed |
| low | c3_mas | 64000 | headroom >= 1 | 15 | 0 | 0.0 | 7.067 | 3.6 | 3.467 | 0.067 | 0.933 |  | reconstructed |
| low | c3_mas | 128000 | headroom = 0 | 15 | 0 | 0.0 | 2.733 | 2.733 | 0.0 | 0.4 | 0.6 |  | reconstructed |
| low | c3_mas | 128000 | headroom >= 1 | 34 | 1 | 0.029 | 6.588 | 3.412 | 3.176 | 0.059 | 0.941 |  | reconstructed |
| low | c4_planner_critic | 16000 | headroom = 0 | 1 | 0 | 0.0 | 2.0 | 2.0 | 0.0 | 2.0 | 0.0 | 0.333 | recorded |
| low | c4_planner_critic | 32000 | headroom = 0 | 4 | 0 | 0.0 | 2.5 | 2.5 | 0.0 | 0.5 | 0.5 | 0.292 | recorded |
| low | c4_planner_critic | 32000 | headroom >= 1 | 8 | 6 | 0.75 | 3.375 | 0.25 | 3.125 | 3.125 | 0.0 | 0.142 | recorded |
| low | c4_planner_critic | 64000 | headroom = 0 | 16 | 0 | 0.0 | 2.812 | 2.812 | 0.0 | 0.375 | 0.625 | 0.326 | recorded |
| low | c4_planner_critic | 64000 | headroom >= 1 | 30 | 24 | 0.8 | 4.2 | 0.667 | 3.533 | 2.767 | 0.1 | 0.214 | recorded |
| low | c4_planner_critic | 128000 | headroom = 0 | 17 | 0 | 0.0 | 2.941 | 2.941 | 0.0 | 0.294 | 0.706 | 0.323 | recorded |
| low | c4_planner_critic | 128000 | headroom >= 1 | 32 | 23 | 0.719 | 4.281 | 0.812 | 3.469 | 2.594 | 0.156 | 0.212 | recorded |
| medium | c3_mas | 16000 | headroom = 0 | 2 | 0 | 0.0 | 1.0 | 1.0 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| medium | c3_mas | 32000 | headroom = 0 | 9 | 0 | 0.0 | 2.0 | 2.0 | 0.0 | 1.222 | 0.111 |  | reconstructed |
| medium | c3_mas | 64000 | headroom = 0 | 27 | 0 | 0.0 | 2.444 | 2.444 | 0.0 | 0.963 | 0.259 |  | reconstructed |
| medium | c3_mas | 64000 | headroom >= 1 | 11 | 0 | 0.0 | 5.636 | 3.273 | 2.364 | 0.182 | 0.818 |  | reconstructed |
| medium | c3_mas | 128000 | headroom = 0 | 17 | 0 | 0.0 | 2.529 | 2.529 | 0.0 | 0.824 | 0.353 |  | reconstructed |
| medium | c3_mas | 128000 | headroom >= 1 | 27 | 2 | 0.074 | 5.407 | 2.963 | 2.444 | 0.407 | 0.593 |  | reconstructed |
| medium | c4_planner_critic | 16000 | headroom >= 1 | 1 | 1 | 1.0 | 3.0 | 0.0 | 3.0 | 4.0 | 0.0 | 0.167 | recorded |
| medium | c4_planner_critic | 32000 | headroom = 0 | 3 | 0 | 0.0 | 2.333 | 2.333 | 0.0 | 1.0 | 0.0 | 0.306 | recorded |
| medium | c4_planner_critic | 32000 | headroom >= 1 | 10 | 8 | 0.8 | 3.8 | 0.0 | 3.8 | 3.6 | 0.0 | 0.16 | recorded |
| medium | c4_planner_critic | 64000 | headroom = 0 | 15 | 0 | 0.0 | 2.6 | 2.6 | 0.0 | 0.667 | 0.333 | 0.364 | recorded |
| medium | c4_planner_critic | 64000 | headroom >= 1 | 26 | 21 | 0.808 | 4.231 | 0.192 | 4.038 | 3.231 | 0.0 | 0.184 | recorded |
| medium | c4_planner_critic | 128000 | headroom = 0 | 17 | 0 | 0.0 | 2.529 | 2.529 | 0.0 | 0.706 | 0.294 | 0.38 | recorded |
| medium | c4_planner_critic | 128000 | headroom >= 1 | 30 | 21 | 0.7 | 4.5 | 0.6 | 3.9 | 2.833 | 0.1 | 0.187 | recorded |
| high | c3_mas | 16000 | headroom = 0 | 2 | 0 | 0.0 | 2.0 | 2.0 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| high | c3_mas | 32000 | headroom = 0 | 7 | 0 | 0.0 | 1.857 | 1.857 | 0.0 | 1.714 | 0.143 |  | reconstructed |
| high | c3_mas | 64000 | headroom = 0 | 28 | 0 | 0.0 | 2.393 | 2.393 | 0.0 | 0.964 | 0.429 |  | reconstructed |
| high | c3_mas | 64000 | headroom >= 1 | 11 | 2 | 0.182 | 5.091 | 2.818 | 2.273 | 0.727 | 0.364 |  | reconstructed |
| high | c3_mas | 128000 | headroom = 0 | 13 | 0 | 0.0 | 2.154 | 2.154 | 0.0 | 1.308 | 0.308 |  | reconstructed |
| high | c3_mas | 128000 | headroom >= 1 | 33 | 6 | 0.182 | 5.242 | 2.939 | 2.303 | 0.394 | 0.636 |  | reconstructed |
| high | c4_planner_critic | 32000 | headroom = 0 | 2 | 0 | 0.0 | 2.5 | 2.5 | 0.0 | 1.0 | 0.5 | 0.333 | recorded |
| high | c4_planner_critic | 32000 | headroom >= 1 | 10 | 9 | 0.9 | 2.6 | 0.0 | 2.6 | 3.5 | 0.0 | 0.267 | recorded |
| high | c4_planner_critic | 64000 | headroom = 0 | 10 | 0 | 0.0 | 2.9 | 2.9 | 0.0 | 0.4 | 0.8 | 0.287 | recorded |
| high | c4_planner_critic | 64000 | headroom >= 1 | 39 | 38 | 0.974 | 3.59 | 0.077 | 3.513 | 3.308 | 0.0 | 0.21 | recorded |
| high | c4_planner_critic | 128000 | headroom = 0 | 9 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.222 | 0.889 | 0.281 | recorded |
| high | c4_planner_critic | 128000 | headroom >= 1 | 40 | 30 | 0.75 | 4.275 | 0.475 | 3.8 | 2.925 | 0.025 | 0.202 | recorded |

**Block B**

| stratum | condition | cap | headroom_stratum | runs_critic_ran | improved | improved_share | mean_pool_size | mean_fallback_size | mean_headroom | mean_room_to_optimum | share_fallback_at_optimum | mean_travel_coverage | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | c3_mas | 16000 | headroom = 0 | 18 | 0 | 0.0 | 1.333 | 1.333 | 0.0 | 1.667 | 0.0 |  | reconstructed |
| all | c3_mas | 32000 | headroom = 0 | 26 | 0 | 0.0 | 1.769 | 1.769 | 0.0 | 1.231 | 0.192 |  | reconstructed |
| all | c3_mas | 32000 | headroom >= 1 | 16 | 7 | 0.438 | 4.312 | 2.438 | 1.875 | 0.562 | 0.438 |  | reconstructed |
| all | c3_mas | 64000 | headroom = 0 | 13 | 0 | 0.0 | 1.923 | 1.923 | 0.0 | 1.077 | 0.308 |  | reconstructed |
| all | c3_mas | 64000 | headroom >= 1 | 34 | 9 | 0.265 | 4.529 | 2.618 | 1.912 | 0.382 | 0.618 |  | reconstructed |
| all | c3_mas | 128000 | headroom = 0 | 9 | 0 | 0.0 | 1.889 | 1.889 | 0.0 | 1.111 | 0.222 |  | reconstructed |
| all | c3_mas | 128000 | headroom >= 1 | 39 | 9 | 0.231 | 4.538 | 2.641 | 1.897 | 0.359 | 0.641 |  | reconstructed |
| all | c4_planner_critic | 16000 | headroom >= 1 | 4 | 3 | 0.75 | 3.25 | 0.0 | 3.25 | 3.0 | 0.0 | 0.083 | recorded |
| all | c4_planner_critic | 32000 | headroom = 0 | 7 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.274 | recorded |
| all | c4_planner_critic | 32000 | headroom >= 1 | 21 | 20 | 0.952 | 3.667 | 0.143 | 3.524 | 2.857 | 0.048 | 0.201 | recorded |
| all | c4_planner_critic | 64000 | headroom = 0 | 14 | 0 | 0.0 | 2.929 | 2.929 | 0.0 | 0.071 | 0.929 | 0.304 | recorded |
| all | c4_planner_critic | 64000 | headroom >= 1 | 33 | 25 | 0.758 | 3.939 | 0.636 | 3.303 | 2.364 | 0.212 | 0.206 | recorded |
| all | c4_planner_critic | 128000 | headroom = 0 | 14 | 0 | 0.0 | 2.929 | 2.929 | 0.0 | 0.071 | 0.929 | 0.304 | recorded |
| all | c4_planner_critic | 128000 | headroom >= 1 | 34 | 22 | 0.647 | 4.029 | 0.882 | 3.147 | 2.118 | 0.294 | 0.213 | recorded |
| n=4 | c3_mas | 16000 | headroom = 0 | 9 | 0 | 0.0 | 1.222 | 1.222 | 0.0 | 1.778 | 0.0 |  | reconstructed |
| n=4 | c3_mas | 32000 | headroom = 0 | 5 | 0 | 0.0 | 1.0 | 1.0 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| n=4 | c3_mas | 32000 | headroom >= 1 | 11 | 7 | 0.636 | 4.0 | 2.182 | 1.818 | 0.818 | 0.182 |  | reconstructed |
| n=4 | c3_mas | 64000 | headroom = 0 | 1 | 0 | 0.0 | 1.0 | 1.0 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| n=4 | c3_mas | 64000 | headroom >= 1 | 15 | 7 | 0.467 | 4.0 | 2.4 | 1.6 | 0.6 | 0.4 |  | reconstructed |
| n=4 | c3_mas | 128000 | headroom = 0 | 1 | 0 | 0.0 | 1.0 | 1.0 | 0.0 | 2.0 | 0.0 |  | reconstructed |
| n=4 | c3_mas | 128000 | headroom >= 1 | 15 | 7 | 0.467 | 4.0 | 2.4 | 1.6 | 0.6 | 0.4 |  | reconstructed |
| n=4 | c4_planner_critic | 16000 | headroom >= 1 | 1 | 1 | 1.0 | 3.0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.083 | recorded |
| n=4 | c4_planner_critic | 32000 | headroom = 0 | 5 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.283 | recorded |
| n=4 | c4_planner_critic | 32000 | headroom >= 1 | 7 | 6 | 0.857 | 3.714 | 0.429 | 3.286 | 2.571 | 0.143 | 0.212 | recorded |
| n=4 | c4_planner_critic | 64000 | headroom = 0 | 8 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.333 | recorded |
| n=4 | c4_planner_critic | 64000 | headroom >= 1 | 8 | 6 | 0.75 | 3.75 | 0.75 | 3.0 | 2.25 | 0.25 | 0.242 | recorded |
| n=4 | c4_planner_critic | 128000 | headroom = 0 | 8 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.333 | recorded |
| n=4 | c4_planner_critic | 128000 | headroom >= 1 | 8 | 6 | 0.75 | 3.75 | 0.75 | 3.0 | 2.25 | 0.25 | 0.242 | recorded |
| n=5 | c3_mas | 16000 | headroom = 0 | 6 | 0 | 0.0 | 1.5 | 1.5 | 0.0 | 1.5 | 0.0 |  | reconstructed |
| n=5 | c3_mas | 32000 | headroom = 0 | 11 | 0 | 0.0 | 1.727 | 1.727 | 0.0 | 1.273 | 0.0 |  | reconstructed |
| n=5 | c3_mas | 32000 | headroom >= 1 | 5 | 0 | 0.0 | 5.0 | 3.0 | 2.0 | 0.0 | 1.0 |  | reconstructed |
| n=5 | c3_mas | 64000 | headroom = 0 | 4 | 0 | 0.0 | 1.5 | 1.5 | 0.0 | 1.5 | 0.0 |  | reconstructed |
| n=5 | c3_mas | 64000 | headroom >= 1 | 12 | 2 | 0.167 | 4.583 | 2.667 | 1.917 | 0.333 | 0.667 |  | reconstructed |
| n=5 | c3_mas | 128000 | headroom = 0 | 4 | 0 | 0.0 | 1.5 | 1.5 | 0.0 | 1.5 | 0.0 |  | reconstructed |
| n=5 | c3_mas | 128000 | headroom >= 1 | 12 | 2 | 0.167 | 4.583 | 2.667 | 1.917 | 0.333 | 0.667 |  | reconstructed |
| n=5 | c4_planner_critic | 16000 | headroom >= 1 | 3 | 2 | 0.667 | 3.333 | 0.0 | 3.333 | 3.0 | 0.0 | 0.083 | recorded |
| n=5 | c4_planner_critic | 32000 | headroom = 0 | 1 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.25 | recorded |
| n=5 | c4_planner_critic | 32000 | headroom >= 1 | 7 | 7 | 1.0 | 4.143 | 0.0 | 4.143 | 3.0 | 0.0 | 0.162 | recorded |
| n=5 | c4_planner_critic | 64000 | headroom = 0 | 5 | 0 | 0.0 | 2.8 | 2.8 | 0.0 | 0.2 | 0.8 | 0.267 | recorded |
| n=5 | c4_planner_critic | 64000 | headroom >= 1 | 10 | 10 | 1.0 | 3.9 | 0.0 | 3.9 | 3.0 | 0.0 | 0.205 | recorded |
| n=5 | c4_planner_critic | 128000 | headroom = 0 | 5 | 0 | 0.0 | 2.8 | 2.8 | 0.0 | 0.2 | 0.8 | 0.267 | recorded |
| n=5 | c4_planner_critic | 128000 | headroom >= 1 | 11 | 9 | 0.818 | 3.909 | 0.545 | 3.364 | 2.455 | 0.182 | 0.226 | recorded |
| n=6 | c3_mas | 16000 | headroom = 0 | 3 | 0 | 0.0 | 1.333 | 1.333 | 0.0 | 1.667 | 0.0 |  | reconstructed |
| n=6 | c3_mas | 32000 | headroom = 0 | 10 | 0 | 0.0 | 2.2 | 2.2 | 0.0 | 0.8 | 0.5 |  | reconstructed |
| n=6 | c3_mas | 64000 | headroom = 0 | 8 | 0 | 0.0 | 2.25 | 2.25 | 0.0 | 0.75 | 0.5 |  | reconstructed |
| n=6 | c3_mas | 64000 | headroom >= 1 | 7 | 0 | 0.0 | 5.571 | 3.0 | 2.571 | 0.0 | 1.0 |  | reconstructed |
| n=6 | c3_mas | 128000 | headroom = 0 | 4 | 0 | 0.0 | 2.5 | 2.5 | 0.0 | 0.5 | 0.5 |  | reconstructed |
| n=6 | c3_mas | 128000 | headroom >= 1 | 12 | 0 | 0.0 | 5.167 | 2.917 | 2.25 | 0.083 | 0.917 |  | reconstructed |
| n=6 | c4_planner_critic | 32000 | headroom = 0 | 1 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.25 | recorded |
| n=6 | c4_planner_critic | 32000 | headroom >= 1 | 7 | 7 | 1.0 | 3.143 | 0.0 | 3.143 | 3.0 | 0.0 | 0.229 | recorded |
| n=6 | c4_planner_critic | 64000 | headroom = 0 | 1 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.25 | recorded |
| n=6 | c4_planner_critic | 64000 | headroom >= 1 | 15 | 9 | 0.6 | 4.067 | 1.0 | 3.067 | 2.0 | 0.333 | 0.187 | recorded |
| n=6 | c4_planner_critic | 128000 | headroom = 0 | 1 | 0 | 0.0 | 3.0 | 3.0 | 0.0 | 0.0 | 1.0 | 0.25 | recorded |
| n=6 | c4_planner_critic | 128000 | headroom >= 1 | 15 | 7 | 0.467 | 4.267 | 1.2 | 3.067 | 1.8 | 0.4 | 0.189 | recorded |

---

*C2's per-cycle detail is limited by 26 stage rows whose proposals could not be assigned to a cycle unambiguously; those carry no quality and are not counted as zero. C2's end-to-end quality is unaffected. Nothing in these tables shows why one architecture spends its budget better than another.*
