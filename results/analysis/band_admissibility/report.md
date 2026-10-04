# Band admissibility — proposed design against the frozen calibration pool

The bands below were **proposed, not derived here**. This report applies the criteria recorded in THESIS_DECISIONS §3 and reports pass or fail. It does not search for bands and does not suggest a repair: a failing design triggers the recorded fallback rather than an adjustment, because widening a band until it passes would choose the design from its own outcome.

Structural stratum: `n_people = 8`, oracle optimum `O = 4`, higher-order gap `H = 0` — 364 calibration candidates.

Buffer pair counts, deliberately unused: [5, 6, 11, 12].


## Per band

| band | pairs | D | n | ≥50 | distinct tightness | ≥2 | dominant share | ≤60% | structures | ≥2 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Low | 1–4 | 0.036–0.143 | 9 | **FAIL** | 2 | pass | 0.667 | **FAIL** | 4 | pass | **FAIL** |
| Medium | 7–10 | 0.250–0.357 | 237 | pass | 2 | pass | 0.523 | pass | 4 | pass | PASS |
| High | 13–16 | 0.464–0.571 | 24 | **FAIL** | 2 | pass | 0.792 | **FAIL** | 4 | pass | **FAIL** |

### Composition

**Low** — tightness {'0.9': 3, '1.0': 6}, structures {'clustered': 1, 'line': 4, 'random': 3, 'uniform': 1}, overlap {'0.2': 9}

**Medium** — tightness {'0.9': 113, '1.0': 124}, structures {'clustered': 70, 'line': 45, 'random': 61, 'uniform': 61}, overlap {'0.2': 146, '0.5': 91}

**High** — tightness {'0.9': 5, '1.0': 19}, structures {'clustered': 10, 'line': 5, 'random': 5, 'uniform': 4}, overlap {'0.2': 14, '0.5': 6, '0.8': 4}


## Identical composition across bands

Each band passing the diversity rules separately does not stop the bands differing from one another, which would move the confound between bands instead of inside them. This asks whether the strata common to all three hold enough candidates to draw the same joint (tightness, travel structure) distribution in each.

- shared strata: **3**
- capacity for an identical allocation: **7** per band
- target: 50 per band
- feasible: **no**, short by 43


## Verdict

**NOT ADMISSIBLE** under the pre-registered criteria.

The recorded response is the fallback in §3, not a revision of these bands. Moving a boundary until the check passes would make the design a function of the check.

The calibration pool is support evidence, not the sample. The held-out instances are generated from the main seed range and their manifest is built separately.

