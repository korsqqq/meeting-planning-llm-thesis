# `n = 4` budget follow-up — 32000 against 64000

**exploratory development follow-up; descriptive only. No hypothesis test, no threshold, no gate, no cap selection.** The same frozen twelve instances at both caps, all at oracle optimum 3. Twelve paired instances: description and ordering, not inference.

Completeness audit of the 64000 arm: 24/24 verified, passed = True.

## The four cells

| metric | C1 @32000 | C3 @32000 | C1 @64000 | C3 @64000 |
|---|---|---|---|---|
| mean satisfaction | 0.500 | 0.722 | 0.583 | 0.917 |
| validated non-empty | 7/12 | 12/12 | 8/12 | 12/12 |
| mean tokens | 25581 | 22546 | 29158 | 26708 |
| mean calls | 13.0 | 16.5 | 13.8 | 17.2 |
| cap used | 0.80 | 0.70 | 0.46 | 0.42 |
| proposal validity | 0.5714 | 0.9677 | 0.7391 | 0.9167 |
| termination | {'aborted': 4, 'agent_finish': 7, 'budget': 1} | {'agent_finish': 12} | {'agent_finish': 11, 'budget': 1} | {'agent_finish': 12} |

## Architecture contrast at each cap

Sign: `delta = C1 - C3`, so a **negative** delta means the hierarchy is ahead.

| cap | Δ (C1−C3) | C1 ahead | C3 ahead | tied | pairs |
|---|---|---|---|---|---|
| 32000 | -0.222 | 3 | 6 | 3 | 12 |
| 64000 | -0.333 | 1 | 6 | 5 | 12 |

## Was the extra budget used, and by how many runs?

The 32768-token window clamps every call at both caps, so a larger cap buys **more calls, not longer ones**. A run that reproduces its token total exactly ran the same trajectory at both caps and is **not informative** about the extra budget; a run whose total changed was altered by the cap and is.

| condition | execution changed | reproduced exactly | exceeded 32 000 | mean Δ tokens |
|---|---|---|---|---|
| c1_react | 5/12 | 7 | 5 | +3576 |
| c3_mas | 6/12 | 6 | 0 | +4163 |

*Exceeding 32 000 is sufficient evidence of a constraint but not necessary.* C1 spends one undivided working budget and diverges only when it would have run out. C3 subdivides the cap: its worker quota is `3/8 * (cap - 256)`, 11 904 against 23 904, so a worker behaves differently long before any total approaches 32 000. The changed-execution column is the one to read.

## Paired change from 32000 to 64000, per condition

| condition | improved | worsened | unchanged | mean Δ satisfaction | gained a validated plan | lost one | execution changed |
|---|---|---|---|---|---|---|---|
| c1_react | 1 | 0 | 11 | +0.083 | 1 | 0 | 5/12 |
| c3_mas | 4 | 0 | 8 | +0.194 | 0 | 0 | 6/12 |

Counts and means only; no hypothesis test is performed. Read every mean against the changed-execution count in the last column: the runs that reproduced exactly contribute a zero to the mean without being evidence either way.

## The reading, fixed before these runs existed

Case obtained: **`c1_below_c3`**. Registered consequence, quoted from the pre-registration of 2026-08-17, THESIS_DECISIONS section 3:

> C3 keeps the descriptive ordering even after the cap is doubled on the lowest frozen structural region. The lower-complexity calibration is then closed, and no easier development pool is built in response.

---

*No threshold is introduced, no cap is selected and no gate is applied. Twelve development instances support description and ordering only.*
