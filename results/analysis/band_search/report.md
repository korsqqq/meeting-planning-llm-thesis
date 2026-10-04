# Three-band design search — exhaustive, outcome-blind

This search answers whether the **previously registered** design is reachable. The earlier rejection concerned one proposal that was stricter than anything registered, so it could not settle the question.

**Status of the ingredients.** Previously registered: separated `D` levels, at least 50 per band, at least two tightness values and two travel structures per band, no tightness above 60%, comparable `O` distribution. Prospectively fixed before this search ran, with the structural pool already seen and no new LLM outcome seen: "comparable" means an identical `O` histogram, the requirements hold jointly on the selected sample, bands are contiguous with at least one unused value between them, and the tie-break below. Post-specified and non-binding: `H`, `overlap`, `alpha_reachable`, conflict concentration, triangle statistics — none of these can fail a design, enter the objective, or break a tie.

Triples considered: **593775**; pruned before the solver: 73956; decided by CP-SAT: 657; admissible at the deciding separation: **308**.

Enumeration: descending separation, stopped at the first admissible level; triples with a smaller gap cannot win the tie-break.


## Verdict: FEASIBLE

Selected by the fixed tie-break — largest minimum gap (5), then largest joint capacity (65), then lexicographic order.

| band | pairs | D | joint capacity |
|---|---|---|---|
| 1 | 0–1 | 0.000–0.036 | 65 |
| 2 | 7–7 | 0.250–0.250 | 65 |
| 3 | 13–16 | 0.464–0.571 | 65 |

Identical `O` histogram across the three bands: `{'3': 41, '4': 24}`


## Non-binding diagnostics for the selected design

Reported because they describe the design, not because they justified it. An unattractive value here is a residual limitation carried into the analysis, never a reason to search again.

| band | pairs | raw n | mean alpha | mean edge share top1 | H counts | overlap counts |
|---|---|---|---|---|---|---|
| band1 | 0–1 | 5572 | 7.934 | 1.0 | {0: 1555, 1: 1545, 2: 1193, 3: 912, 4: 351, 5: 16} | {'0.2': 1542, '0.5': 1483, '0.8': 1326, '1.0': 1221} |
| band2 | 7–7 | 179 | 4.709 | 0.4796 | {-1: 1, 0: 74, 1: 51, 2: 37, 3: 15, 4: 1} | {'0.2': 73, '0.5': 55, '0.8': 18, '1.0': 33} |
| band3 | 13–16 | 456 | 3.386 | 0.3561 | {0: 284, 1: 132, 2: 35, 3: 5} | {'0.2': 67, '0.5': 184, '0.8': 156, '1.0': 49} |
