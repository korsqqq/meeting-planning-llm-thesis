# C1 calibration sweep — analysis

These runs are **calibration evidence, not held-out pilot evidence**: one of the
thirty instances (`uniform-n4-t20-o20-s10000`) had been run and inspected before the
manifest was frozen. They inform the budget ladder and describe mechanisms; they
are not a pre-registered basis for comparing C1 with C2/C3.

Levels below are the OLD binning (medium <= 5). The formal manifest cuts its own
boundaries (medium <= 6), so categorical levels are **not** comparable across the
two, and no table here mixes them.


## All runs (n = 120 runs)

| cap | sat | optimum | achieved | abs gap | tokens | att/run | parse | validity | acceptance | runs w/ valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 8000 | 0.011 | 3.700 | 0.033 | 3.667 | 7659.500 | 0.033 | 1.000 | 1.000 | 1.000 | 0.033 |
| 16000 | 0.011 | 3.700 | 0.033 | 3.667 | 15781.067 | 0.300 | 1.000 | 0.111 | 0.111 | 0.033 |
| 32000 | 0.300 | 3.700 | 0.967 | 2.733 | 29660.400 | 1.000 | 1.000 | 0.333 | 0.333 | 0.333 |
| 64000 | 0.502 | 3.700 | 1.667 | 2.033 | 40630.633 | 1.700 | 1.000 | 0.333 | 0.333 | 0.533 |

**32k vs 64k, paired on 30 instances**: improved 7, tied 23, worsened 0; mean Δsat 0.202, median 0.000; mean Δattempts 0.700; mean Δtokens 10970.233; 64k token quartiles [28641.8, 39295.5, 52597.0]; share above 32k tokens 0.633, above 48k 0.267.


## Excluding uniform-n4-t20-o20-s10000 (n = 116 runs)

| cap | sat | optimum | achieved | abs gap | tokens | att/run | parse | validity | acceptance | runs w/ valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 8000 | 0.011 | 3.690 | 0.035 | 3.655 | 7668.828 | 0.035 | 1.000 | 1.000 | 1.000 | 0.035 |
| 16000 | 0.011 | 3.690 | 0.035 | 3.655 | 15778.897 | 0.276 | 1.000 | 0.125 | 0.125 | 0.035 |
| 32000 | 0.276 | 3.690 | 0.862 | 2.828 | 29810.069 | 0.966 | 1.000 | 0.321 | 0.321 | 0.310 |
| 64000 | 0.484 | 3.690 | 1.586 | 2.103 | 41158.586 | 1.690 | 1.000 | 0.327 | 0.327 | 0.517 |

**32k vs 64k, paired on 29 instances**: improved 7, tied 22, worsened 0; mean Δsat 0.208, median 0.000; mean Δattempts 0.724; mean Δtokens 11348.517; 64k token quartiles [28977.5, 39886.0, 53345.0]; share above 32k tokens 0.655, above 48k 0.276.


## Level tables (old binning), full set


### cap 8000

| level | n | sat | optimum | achieved | abs gap | att | validity | runs w/ valid | termination |
|---|---|---|---|---|---|---|---|---|---|
| easy | 10 | 0.000 | 5.500 | 0.000 | 5.500 | 0 | n/a | 0.000 | {'budget': 8, 'aborted': 2} |
| medium | 10 | 0.000 | 3.600 | 0.000 | 3.600 | 0 | n/a | 0.000 | {'aborted': 8, 'budget': 2} |
| hard | 10 | 0.033 | 2.000 | 0.100 | 1.900 | 1 | 1.000 | 0.100 | {'aborted': 6, 'budget': 4} |

Context cross-tab: limited+length 0, limited+stop 0, not-limited+length 18, limited calls 0 of 225. A limited call had a reduced ALLOWANCE; only finish_reason=length is actual truncation.


### cap 16000

| level | n | sat | optimum | achieved | abs gap | att | validity | runs w/ valid | termination |
|---|---|---|---|---|---|---|---|---|---|
| easy | 10 | 0.000 | 5.500 | 0.000 | 5.500 | 5 | 0.000 | 0.000 | {'aborted': 8, 'budget': 2} |
| medium | 10 | 0.000 | 3.600 | 0.000 | 3.600 | 1 | 0.000 | 0.000 | {'aborted': 9, 'budget': 1} |
| hard | 10 | 0.033 | 2.000 | 0.100 | 1.900 | 3 | 0.333 | 0.100 | {'aborted': 9, 'budget': 1} |

Context cross-tab: limited+length 0, limited+stop 0, not-limited+length 26, limited calls 0 of 332. A limited call had a reduced ALLOWANCE; only finish_reason=length is actual truncation.


### cap 32000

| level | n | sat | optimum | achieved | abs gap | att | validity | runs w/ valid | termination |
|---|---|---|---|---|---|---|---|---|---|
| easy | 10 | 0.400 | 5.500 | 1.600 | 3.900 | 11 | 0.364 | 0.400 | {'agent_finish': 4, 'budget': 2, 'aborted': 4} |
| medium | 10 | 0.267 | 3.600 | 0.700 | 2.900 | 7 | 0.429 | 0.300 | {'agent_finish': 3, 'aborted': 4, 'budget': 3} |
| hard | 10 | 0.233 | 2.000 | 0.600 | 1.400 | 12 | 0.250 | 0.300 | {'aborted': 6, 'agent_finish': 3, 'budget': 1} |

Spearman over n = 30 runs (wide interval at this size, reported
as description, not as an effect size): complexity vs satisfaction -0.093, vs achieved -0.143, vs absolute gap -0.312, vs oracle optimum -0.778.


Context cross-tab: limited+length 0, limited+stop 0, not-limited+length 14, limited calls 0 of 426. A limited call had a reduced ALLOWANCE; only finish_reason=length is actual truncation.


### cap 64000

| level | n | sat | optimum | achieved | abs gap | att | validity | runs w/ valid | termination |
|---|---|---|---|---|---|---|---|---|---|
| easy | 10 | 0.471 | 5.500 | 2.100 | 3.400 | 17 | 0.294 | 0.500 | {'agent_finish': 9, 'aborted': 1} |
| medium | 10 | 0.600 | 3.600 | 2.000 | 1.600 | 13 | 0.538 | 0.600 | {'agent_finish': 6, 'aborted': 4} |
| hard | 10 | 0.433 | 2.000 | 0.900 | 1.100 | 21 | 0.238 | 0.500 | {'agent_finish': 8, 'aborted': 1, 'budget': 1} |

Spearman over n = 30 runs (wide interval at this size, reported
as description, not as an effect size): complexity vs satisfaction -0.002, vs achieved -0.200, vs absolute gap -0.230, vs oracle optimum -0.778.


Context cross-tab: limited+length 2, limited+stop 406, not-limited+length 3, limited calls 408 of 482. A limited call had a reduced ALLOWANCE; only finish_reason=length is actual truncation.


## Formal subset audit (composition only — no runs exist for it yet)

- subset `29d836cb5614` from manifest `e6a365481dd4`
- counts {'easy': 10, 'medium': 10, 'hard': 10}, 10/10/10: True
- duplicate instance ids: none
- overlap with the superseded subset: none

| level | n | strata (n_people-travel) | seeds | complexity min/med/max | optimum mean/med/min/max |
|---|---|---|---|---|---|
| easy | 10 | {'4-clustered': 2, '4-uniform': 2, '6-clustered': 2, '6-uniform': 2, '8-clustered': 1, '8-uniform': 1} | [10005] | [0, 0.0, 0] | 5.5 / 6.0 / 4 / 8 |
| medium | 10 | {'4-clustered': 2, '4-uniform': 2, '6-clustered': 2, '6-uniform': 2, '8-clustered': 1, '8-uniform': 1} | [10005] | [1, 5.0, 6] | 3.3 / 3.0 / 1 / 5 |
| hard | 10 | {'6-clustered': 3, '6-uniform': 3, '8-clustered': 2, '8-uniform': 2} | [10005, 10006, 10007] | [7, 12.5, 26] | 2.5 / 2.0 / 1 / 4 |

**Coverage limitation, recorded not fixed.** The formal hard level contains no
n=4 instances: with the new pool's boundary (hard > 6) no four-person instance
qualifies as hard, so the round-robin cannot place one. Level and n_people are
therefore partially confounded at the top level. The manifest is NOT regenerated
to repair this -- doing so would mean choosing the selection after seeing its
composition.

