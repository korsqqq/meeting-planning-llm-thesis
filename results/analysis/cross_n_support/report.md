# Cross-size structural support — lower-complexity calibration

**Read-only. Selects nothing, fixes no optimum histogram, cuts no band.** The numbers below exist so that the optimum distribution for the 12-per-size sample is chosen from measured capacity rather than from intuition.

## Pools

| n | candidates | seeds | C(n,2) | k range | D range | conflict-free | audit |
|---|---|---|---|---|---|---|---|
| 4 | 9600 | 40000–40099 | 6 | 0–6 | 0.000–1.000 | 63.4% | passed |
| 5 | 9600 | 50000–50099 | 10 | 0–10 | 0.000–1.000 | 59.9% | passed |
| 6 | 9600 | 60000–60099 | 15 | 0–15 | 0.000–1.000 | 57.2% | passed |
| 8 | 9600 | 30000–30099 | 28 | 0–28 | 0.000–1.000 | 54.6% | passed |

## Optimum distribution per size

| n | O=1 | O=2 | O=3 | O=4 | O=5 | O=6 | O=7 | O=8 |
|---|---|---|---|---|---|---|---|---|
| 4 | 790 | 1848 | 1836 | 5126 | — | — | — | — |
| 5 | 639 | 1522 | 1662 | 1624 | 4153 | — | — | — |
| 6 | 546 | 1208 | 1647 | 1416 | 1503 | 3280 | — | — |
| 8 | 470 | 1009 | 1352 | 1212 | 1316 | 1154 | 1513 | 1574 |

## Cross-tab: size × optimum

| n | O | count | O/n | k min | k med | k max | D med | H min | H med | H max | tightness values | structures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 790 | 0.25 | 1 | 6 | 6 | 1.000 | 0 | 0 | 0 | 3 | 4 |
| 4 | 2 | 1848 | 0.5 | 0 | 3 | 5 | 0.500 | 0 | 0 | 2 | 4 | 4 |
| 4 | 3 | 1836 | 0.75 | 0 | 1 | 4 | 0.167 | -1 | 0 | 1 | 5 | 4 |
| 4 | 4 | 5126 | 1.0 | 0 | 0 | 1 | 0.000 | -1 | 0 | 0 | 6 | 4 |
| 5 | 1 | 639 | 0.2 | 3 | 10 | 10 | 1.000 | 0 | 0 | 0 | 2 | 4 |
| 5 | 2 | 1522 | 0.4 | 0 | 6 | 9 | 0.600 | 0 | 0 | 3 | 3 | 4 |
| 5 | 3 | 1662 | 0.6 | 0 | 2 | 7 | 0.200 | -1 | 1 | 2 | 4 | 4 |
| 5 | 4 | 1624 | 0.8 | 0 | 0 | 3 | 0.000 | -1 | 1 | 1 | 5 | 4 |
| 5 | 5 | 4153 | 1.0 | 0 | 0 | 1 | 0.000 | -1 | 0 | 0 | 6 | 4 |
| 6 | 1 | 546 | 0.167 | 15 | 15 | 15 | 1.000 | 0 | 0 | 0 | 2 | 4 |
| 6 | 2 | 1208 | 0.333 | 0 | 11 | 14 | 0.733 | 0 | 0 | 4 | 3 | 4 |
| 6 | 3 | 1647 | 0.5 | 0 | 4 | 12 | 0.267 | -1 | 0 | 3 | 4 | 4 |
| 6 | 4 | 1416 | 0.667 | 0 | 0 | 7 | 0.000 | -1 | 2 | 2 | 5 | 4 |
| 6 | 5 | 1503 | 0.833 | 0 | 0 | 3 | 0.000 | -1 | 1 | 1 | 6 | 4 |
| 6 | 6 | 3280 | 1.0 | 0 | 0 | 0 | 0.000 | 0 | 0 | 0 | 5 | 4 |
| 8 | 1 | 470 | 0.125 | 28 | 28 | 28 | 1.000 | 0 | 0 | 0 | 2 | 4 |
| 8 | 2 | 1009 | 0.25 | 3 | 22 | 27 | 0.786 | 0 | 0 | 4 | 3 | 4 |
| 8 | 3 | 1352 | 0.375 | 0 | 10 | 24 | 0.357 | -1 | 1 | 5 | 3 | 4 |
| 8 | 4 | 1212 | 0.5 | 0 | 4 | 15 | 0.143 | -1 | 2 | 4 | 4 | 4 |
| 8 | 5 | 1316 | 0.625 | 0 | 0 | 10 | 0.000 | -1 | 3 | 3 | 5 | 4 |
| 8 | 6 | 1154 | 0.75 | 0 | 0 | 8 | 0.000 | -1 | 2 | 2 | 5 | 4 |
| 8 | 7 | 1513 | 0.875 | 0 | 0 | 3 | 0.000 | 0 | 1 | 1 | 5 | 4 |
| 8 | 8 | 1574 | 1.0 | 0 | 0 | 0 | 0.000 | 0 | 0 | 0 | 3 | 4 |

## Matched variant — common support O ∈ {3, 4}

| n | capacity O=3 | capacity O=4 |
|---|---|---|
| 4 | 1836 | 5126 |
| 5 | 1662 | 1624 |
| 6 | 1647 | 1416 |
| 8 | 1352 | 1212 |

* joint capacity {'3': 1352, '4': 1212}, binding size per optimum {'3': 8, '4': 8}
* scaled to 12 per size: {'3': 6, '4': 6} — capacity sufficient: True
* diversity rules cannot be met at: no size — all clear

## Matched variant — common support O = {3} only

| n | capacity O=3 |
|---|---|
| 4 | 1836 |
| 5 | 1662 |
| 6 | 1647 |
| 8 | 1352 |

* joint capacity {'3': 1352}, binding size per optimum {'3': 8}
* scaled to 12 per size: {'3': 12} — capacity sufficient: True
* diversity rules cannot be met at: no size — all clear

## Ceiling analysis

| n | O | O/n | meets everyone | satisfaction step | H ceiling (n−O) | H max seen | H mean | mean unreachable |
|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 0.5 | no | 0.500 | 2 | 2 | 0.344 | 0.015 |
| 4 | 3 | 0.75 | no | 0.333 | 1 | 1 | 0.446 | 0.008 |
| 4 | 4 | 1.0 | yes | 0.250 | 0 | 0 | -0.001 | 0.0 |
| 5 | 2 | 0.4 | no | 0.500 | 3 | 3 | 0.35 | 0.017 |
| 5 | 3 | 0.6 | no | 0.333 | 2 | 2 | 0.721 | 0.011 |
| 5 | 4 | 0.8 | no | 0.250 | 1 | 1 | 0.752 | 0.002 |
| 6 | 2 | 0.333 | no | 0.500 | 4 | 4 | 0.425 | 0.013 |
| 6 | 3 | 0.5 | no | 0.333 | 3 | 3 | 0.857 | 0.022 |
| 6 | 4 | 0.667 | no | 0.250 | 2 | 2 | 1.244 | 0.004 |
| 8 | 2 | 0.25 | no | 0.500 | 6 | 4 | 0.538 | 0.005 |
| 8 | 3 | 0.375 | no | 0.333 | 5 | 5 | 1.342 | 0.04 |
| 8 | 4 | 0.5 | no | 0.250 | 4 | 4 | 1.936 | 0.016 |

## Sanity check — the rejected high-optimum option at n = 8

**READ-ONLY SANITY CHECK -- NOT A SELECTION CANDIDATE**

O >= 6 does not exist in the frozen high band, so admitting it puts the compared groups at different optima and reintroduces the confound the matching rule removes.

* `n = 8, conflict pairs <= 1, O >= 6` → 4164 candidates, optimum histogram {6: 1082, 7: 1508, 8: 1574}, mean H 0.864, mean unreachable 0.0
* for contrast `n = 8, conflict pairs <= 1, O in {3, 4}` → 491 candidates, H histogram {3: 108, 4: 361, 5: 22}, mean H 3.825

## The frozen n = 8 development subset, for comparison

* `D:\thesis-mas-planning\.claude\worktrees\heuristic-lamarr-ce26c0\results\manifests\subset__bands_budget_dev__911e4864d6fb.json` (`911e4864d6fb`), 60 instances
* optimum histogram {3: 36, 4: 24}, identical per band: {'easy': {3: 12, 4: 8}, 'medium': {3: 12, 4: 8}, 'hard': {3: 12, 4: 8}}
* k range [0, 15], tightness {0.6: 8, 0.8: 19, 0.9: 21, 1.0: 12}

---

*Nothing above is frozen. The optimum distribution for the 36 instances is a separate dated decision, taken after this output has been read and recorded in THESIS_DECISIONS before any instance is selected.*
