# Structural calibration — pre-specified diagnostics

Read-only over the frozen candidate pool. **No band is chosen here.** This report contains no Low/Medium/High constant and no search for the best-separated or best-matched triple of `D` regions; the bands are decided against the admissibility rules in THESIS_DECISIONS §3 and fixed in a dated amendment.

Pool: 9600 candidates at `n_people = 8`, seeds 20000–20099, collector commit `2f3f2108`. Since `n` is fixed, `D` takes exactly 29 values and every table is cut on the exact conflict-pair count — no bin width was chosen by anyone.


## A. Does the axis exist

`D` quartiles: 0.000 / 0.000 / 0.000 / 0.357 / 1.000

- `D = 0`: **5209** candidates
- `0 < D < 1`: **3925**
- `D = 1`: **466**

| pairs | D | count | cumulative share |
|---|---|---|---|
| 0 | 0.000 | 5209 | 0.543 |
| 1 | 0.036 | 363 | 0.580 |
| 2 | 0.071 | 283 | 0.610 |
| 3 | 0.107 | 202 | 0.631 |
| 4 | 0.143 | 237 | 0.656 |
| 5 | 0.179 | 209 | 0.677 |
| 6 | 0.214 | 188 | 0.697 |
| 7 | 0.250 | 179 | 0.716 |
| 8 | 0.286 | 171 | 0.733 |
| 9 | 0.321 | 154 | 0.750 |
| 10 | 0.357 | 193 | 0.770 |
| 11 | 0.393 | 151 | 0.785 |
| 12 | 0.429 | 136 | 0.799 |
| 13 | 0.464 | 130 | 0.813 |
| 14 | 0.500 | 102 | 0.824 |
| 15 | 0.536 | 118 | 0.836 |
| 16 | 0.571 | 106 | 0.847 |
| 17 | 0.607 | 74 | 0.855 |
| 18 | 0.643 | 85 | 0.864 |
| 19 | 0.679 | 91 | 0.873 |
| 20 | 0.714 | 83 | 0.882 |
| 21 | 0.750 | 103 | 0.892 |
| 22 | 0.786 | 87 | 0.901 |
| 23 | 0.821 | 77 | 0.909 |
| 24 | 0.857 | 89 | 0.919 |
| 25 | 0.893 | 103 | 0.929 |
| 26 | 0.929 | 109 | 0.941 |
| 27 | 0.964 | 102 | 0.952 |
| 28 | 1.000 | 466 | 1.000 |

**`D` by travel structure**

| value | n | min | q1 | median | q3 | max |
|---|---|---|---|---|---|---|
| uniform | 2400 | 0.000 | 0.000 | 0.000 | 0.393 | 1.000 |
| clustered | 2400 | 0.000 | 0.000 | 0.000 | 0.357 | 1.000 |
| line | 2400 | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 |
| random | 2400 | 0.000 | 0.000 | 0.000 | 0.357 | 1.000 |

**`D` by tightness**

| value | n | min | q1 | median | q3 | max |
|---|---|---|---|---|---|---|
| 0.2 | 1600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4 | 1600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.6 | 1600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.8 | 1600 | 0.000 | 0.000 | 0.071 | 0.143 | 0.643 |
| 0.9 | 1600 | 0.000 | 0.214 | 0.357 | 0.679 | 1.000 |
| 1.0 | 1600 | 0.036 | 0.429 | 0.750 | 1.000 | 1.000 |

**`D` by overlap**

| value | n | min | q1 | median | q3 | max |
|---|---|---|---|---|---|---|
| 0.2 | 2400 | 0.000 | 0.000 | 0.000 | 0.179 | 0.786 |
| 0.5 | 2400 | 0.000 | 0.000 | 0.000 | 0.250 | 1.000 |
| 0.8 | 2400 | 0.000 | 0.000 | 0.000 | 0.536 | 1.000 |
| 1.0 | 2400 | 0.000 | 0.000 | 0.000 | 0.786 | 1.000 |

## B. Conflict density against the oracle optimum

The central question: can well-separated `D` coexist with a comparable optimum?

Spearman(`D`, optimum) = **-0.869**; Spearman(`D`, optimum/n) = -0.869

**Which `D` are available at each optimum**

| optimum | n | D min | D median | D max | distinct pair counts | pair range |
|---|---|---|---|---|---|---|
| 1 | 466 | 1.000 | 1.000 | 1.000 | 1 | [28, 28] |
| 2 | 975 | 0.143 | 0.786 | 0.964 | 24 | [4, 27] |
| 3 | 1392 | 0.000 | 0.393 | 0.857 | 25 | [0, 24] |
| 4 | 1262 | 0.000 | 0.143 | 0.571 | 17 | [0, 16] |
| 5 | 1292 | 0.000 | 0.000 | 0.393 | 12 | [0, 11] |
| 6 | 1183 | 0.000 | 0.000 | 0.179 | 6 | [0, 5] |
| 7 | 1482 | 0.000 | 0.000 | 0.107 | 4 | [0, 3] |
| 8 | 1548 | 0.000 | 0.000 | 0.000 | 1 | [0, 0] |

**Which optima occur at each pair count**

| pairs | n | optimum min | median | max | mean optimum/n |
|---|---|---|---|---|---|
| 0 | 5209 | 3.000 | 7.000 | 8.000 | 0.824 |
| 1 | 363 | 3.000 | 5.000 | 7.000 | 0.595 |
| 2 | 283 | 3.000 | 4.000 | 7.000 | 0.534 |
| 3 | 202 | 3.000 | 4.000 | 7.000 | 0.532 |
| 4 | 237 | 2.000 | 4.000 | 6.000 | 0.513 |
| 5 | 209 | 2.000 | 4.000 | 6.000 | 0.496 |
| 6 | 188 | 2.000 | 4.000 | 5.000 | 0.485 |
| 7 | 179 | 2.000 | 4.000 | 5.000 | 0.468 |
| 8 | 171 | 2.000 | 4.000 | 5.000 | 0.456 |
| 9 | 154 | 2.000 | 3.000 | 5.000 | 0.428 |
| 10 | 193 | 2.000 | 3.000 | 5.000 | 0.420 |
| 11 | 151 | 2.000 | 3.000 | 5.000 | 0.394 |
| 12 | 136 | 2.000 | 3.000 | 4.000 | 0.384 |
| 13 | 130 | 2.000 | 3.000 | 4.000 | 0.370 |
| 14 | 102 | 2.000 | 3.000 | 4.000 | 0.375 |
| 15 | 118 | 2.000 | 3.000 | 4.000 | 0.353 |
| 16 | 106 | 2.000 | 3.000 | 4.000 | 0.357 |
| 17 | 74 | 2.000 | 3.000 | 3.000 | 0.326 |
| 18 | 85 | 2.000 | 2.000 | 3.000 | 0.300 |
| 19 | 91 | 2.000 | 2.000 | 3.000 | 0.287 |
| 20 | 83 | 2.000 | 2.000 | 3.000 | 0.271 |
| 21 | 103 | 2.000 | 2.000 | 3.000 | 0.260 |
| 22 | 87 | 2.000 | 2.000 | 3.000 | 0.257 |
| 23 | 77 | 2.000 | 2.000 | 3.000 | 0.253 |
| 24 | 89 | 2.000 | 2.000 | 3.000 | 0.251 |
| 25 | 103 | 2.000 | 2.000 | 2.000 | 0.250 |
| 26 | 109 | 2.000 | 2.000 | 2.000 | 0.250 |
| 27 | 102 | 2.000 | 2.000 | 2.000 | 0.250 |
| 28 | 466 | 1.000 | 1.000 | 1.000 | 0.125 |

## C. Is a region of `D` effectively one generator knob

Spearman(`D`, tightness) = **0.881**; Spearman(`D`, overlap) = 0.173

| pairs | n | distinct tightness | dominant tightness | its share | distinct structures |
|---|---|---|---|---|---|
| 0 | 5209 | 5 | 0.2 | 0.307 | 4 |
| 1 | 363 | 3 | 0.8 | 0.948 | 4 |
| 2 | 283 | 3 | 0.8 | 0.866 | 4 |
| 3 | 202 | 3 | 0.8 | 0.644 | 4 |
| 4 | 237 | 3 | 0.8 | 0.498 | 4 |
| 5 | 209 | 3 | 0.9 | 0.569 | 4 |
| 6 | 188 | 3 | 0.9 | 0.532 | 4 |
| 7 | 179 | 3 | 0.9 | 0.587 | 4 |
| 8 | 171 | 3 | 0.9 | 0.538 | 4 |
| 9 | 154 | 3 | 0.9 | 0.442 | 4 |
| 10 | 193 | 3 | 0.9 | 0.420 | 4 |
| 11 | 151 | 3 | 1.0 | 0.556 | 4 |
| 12 | 136 | 3 | 1.0 | 0.537 | 4 |
| 13 | 130 | 3 | 1.0 | 0.546 | 4 |
| 14 | 102 | 3 | 1.0 | 0.451 | 4 |
| 15 | 118 | 3 | 1.0 | 0.576 | 4 |
| 16 | 106 | 3 | 0.9 | 0.481 | 4 |
| 17 | 74 | 3 | 0.9 | 0.513 | 4 |
| 18 | 85 | 3 | 0.9 | 0.600 | 4 |
| 19 | 91 | 2 | 0.9 | 0.637 | 4 |
| 20 | 83 | 2 | 0.9 | 0.711 | 4 |
| 21 | 103 | 2 | 0.9 | 0.660 | 4 |
| 22 | 87 | 2 | 0.9 | 0.713 | 4 |
| 23 | 77 | 2 | 0.9 | 0.558 | 4 |
| 24 | 89 | 2 | 1.0 | 0.584 | 4 |
| 25 | 103 | 2 | 1.0 | 0.699 | 4 |
| 26 | 109 | 2 | 1.0 | 0.679 | 4 |
| 27 | 102 | 2 | 1.0 | 0.794 | 4 |
| 28 | 466 | 2 | 1.0 | 0.964 | 4 |

The §3 diversity rule asks for at least two tightness values, at least two travel structures, and no single tightness above 60 % — applied to the final bands, not to individual pair counts. This table is the input to that check.


## D. Graph structure behind a high `D`

| quantity | min | q1 | median | q3 | max | Spearman with D |
|---|---|---|---|---|---|---|
| `alpha_reachable` | 1.000 | 4.000 | 8.000 | 8.000 | 8.000 | -0.994 |
| `higher_order_gap_H` | -1.000 | 0.000 | 1.000 | 2.000 | 5.000 | -0.342 |
| `max_degree` | 0.000 | 0.000 | 0.000 | 4.000 | 7.000 | 0.996 |
| `edge_share_top1` | 0.250 | 0.304 | 0.385 | 0.500 | 1.000 | -0.922 |
| `edge_share_top2` | 0.500 | 0.600 | 0.727 | 1.000 | 2.000 | -0.938 |
| `largest_component` | 1.000 | 1.000 | 1.000 | 7.000 | 8.000 | 0.989 |
| `n_max_independent_sets` | 1.000 | 1.000 | 1.000 | 2.000 | 18.000 | 0.772 |
| `individually_reachable_count` | 5.000 | 8.000 | 8.000 | 8.000 | 8.000 | -0.060 |

Candidates with fewer than 8 individually reachable people: **65**. These cannot reach the top of the `D` range, since the conflict graph is built over the reachable set only.

| pairs | n | alpha | H | max degree | edge share top1 | max ind. sets |
|---|---|---|---|---|---|---|
| 0 | 5209 | 8.000 | 1.409 | 0.000 | n/a | 1.000 |
| 1 | 363 | 6.989 | 2.231 | 1.000 | 1.000 | 2.000 |
| 2 | 283 | 6.569 | 2.300 | 1.587 | 0.793 | 2.240 |
| 3 | 202 | 6.094 | 1.837 | 2.109 | 0.703 | 2.792 |
| 4 | 237 | 5.662 | 1.561 | 2.443 | 0.611 | 3.017 |
| 5 | 209 | 5.364 | 1.392 | 2.770 | 0.554 | 3.211 |
| 6 | 188 | 5.074 | 1.192 | 3.043 | 0.507 | 2.894 |
| 7 | 179 | 4.710 | 0.967 | 3.357 | 0.480 | 3.626 |
| 8 | 171 | 4.409 | 0.760 | 3.538 | 0.442 | 3.573 |
| 9 | 154 | 4.149 | 0.727 | 3.792 | 0.421 | 4.123 |
| 10 | 193 | 4.042 | 0.679 | 4.145 | 0.414 | 4.072 |
| 11 | 151 | 3.682 | 0.530 | 4.437 | 0.403 | 4.636 |
| 12 | 136 | 3.574 | 0.500 | 4.544 | 0.379 | 4.375 |
| 13 | 130 | 3.454 | 0.492 | 4.739 | 0.364 | 4.208 |
| 14 | 102 | 3.598 | 0.598 | 5.059 | 0.361 | 3.039 |
| 15 | 118 | 3.271 | 0.449 | 5.203 | 0.347 | 3.644 |
| 16 | 106 | 3.226 | 0.368 | 5.613 | 0.351 | 3.425 |
| 17 | 74 | 3.041 | 0.432 | 5.865 | 0.345 | 4.162 |
| 18 | 85 | 3.094 | 0.694 | 6.188 | 0.344 | 3.165 |
| 19 | 91 | 3.055 | 0.758 | 6.450 | 0.340 | 3.187 |
| 20 | 83 | 2.928 | 0.759 | 6.494 | 0.325 | 2.928 |
| 21 | 103 | 2.612 | 0.534 | 6.582 | 0.314 | 3.864 |
| 22 | 87 | 2.563 | 0.506 | 6.851 | 0.311 | 3.437 |
| 23 | 77 | 2.351 | 0.325 | 6.948 | 0.302 | 3.597 |
| 24 | 89 | 2.146 | 0.135 | 6.966 | 0.290 | 3.562 |
| 25 | 103 | 2.087 | 0.087 | 7.000 | 0.280 | 2.825 |
| 26 | 109 | 2.000 | 0.000 | 7.000 | 0.269 | 2.000 |
| 27 | 102 | 2.000 | 0.000 | 7.000 | 0.259 | 1.000 |
| 28 | 466 | 1.000 | 0.000 | 7.000 | 0.250 | 8.000 |

A high `D` together with a high `alpha_reachable` and a high `edge_share_top1` is the hub case: many conflicts sitting on one person, and a trivial answer.


## E. Triangle inequality and the sign of `H`

Ordered triples examined per instance: 504.

| structure | n | violations median | max | mean rate | instances with any | H<0 | H=0 | H>0 |
|---|---|---|---|---|---|---|---|---|
| uniform | 2400 | 0.000 | 6.000 | 0.002 | 936 | 0 | 681 | 1719 |
| clustered | 2400 | 2.000 | 14.000 | 0.006 | 1584 | 0 | 1050 | 1350 |
| line | 2400 | 6.000 | 34.000 | 0.014 | 2160 | 0 | 1065 | 1335 |
| random | 2400 | 74.000 | 101.000 | 0.144 | 2400 | 2 | 985 | 1413 |

Overall: `H < 0` in **2** candidates, `H = 0` in 3781, `H > 0` in 5817. Negative values are reported, never clipped: they record that feasibility is not monotone under deletion.


## What this report deliberately does not do

- It names no band and applies no threshold.
- It does not search for the triple of `D` regions that would look best on separation, optimum matching or knob balance. That search would select the design from its own outcome.
- It draws no conclusion about whether Plan A is admissible. That decision is made against the rules already written in §3 and recorded in a dated amendment.

