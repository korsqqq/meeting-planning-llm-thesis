# Joint support of `D`, oracle optimum and higher-order gap

> **This diagnostic is post-specified.** It was not part of the analysis committed before the calibration pool was read. It is motivated by a result from a diagnostic that *was* pre-registered: the higher-order gap `H` is largest exactly where the pairwise density `D` is smallest, so bands cut on `D` alone would move `H` in the opposite direction and confound the two.

`H` is treated here as a matching variable alongside the oracle optimum `O`. It does **not** enter the definition of complexity, and no composite of the form `D + H` is formed. The design tests pairwise interaction density and holds the higher-order part comparable so that it cannot explain the outcome.

**No band is selected here.** No threshold, no Low/Medium/High constant, and no search for the cell or triple of cells that would look best.

Pool: 9600 candidates at `n_people = 8`. Spearman(`D`, `H`) = **-0.342**; Spearman(`H`, optimum) = -0.083.


## How `H` is distributed at each optimum

| optimum | H=-1 | H=0 | H=1 | H=2 | H=3 | H=4 | H=5 |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 466 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 616 | 235 | 93 | 24 | 6 | 1 |
| 3 | 0 | 590 | 259 | 233 | 198 | 96 | 16 |
| 4 | 0 | 364 | 207 | 160 | 205 | 326 | 0 |
| 5 | 2 | 147 | 158 | 192 | 793 | 0 | 0 |
| 6 | 0 | 41 | 90 | 1052 | 0 | 0 | 0 |
| 7 | 0 | 9 | 1473 | 0 | 0 | 0 | 0 |
| 8 | 0 | 1548 | 0 | 0 | 0 | 0 | 0 |

## Common support: how wide is `D` inside a fixed (optimum, `H`) cell

A cell is usable for the intended contrast only if it holds enough candidates across a wide enough span of `D`. Both columns are reported; neither is thresholded here.

| cell | n | pair span | distinct pairs | D min | D max | distinct tightness | dominant share | structures | ρ(D, tightness) |
|---|---|---|---|---|---|---|---|---|---|
| `O=1,H=0` | 466 | 0 | 1 | 1.000 | 1.000 | 2 | 0.964 | 4 | n/a |
| `O=2,H=0` | 616 | 20 | 19 | 0.250 | 0.964 | 2 | 0.744 | 4 | -0.056 |
| `O=2,H=1` | 235 | 11 | 12 | 0.500 | 0.893 | 2 | 0.996 | 4 | 0.103 |
| `O=2,H=2` | 93 | 13 | 13 | 0.321 | 0.786 | 2 | 0.753 | 4 | 0.707 |
| `O=2,H=3` | 24 | 10 | 10 | 0.250 | 0.607 | 2 | 0.750 | 2 | 0.683 |
| `O=2,H=4` | 6 | 7 | 6 | 0.143 | 0.393 | 1 | 1.000 | 1 | n/a |
| `O=2,H=5` | 1 | 0 | 1 | 0.179 | 0.179 | 1 | 1.000 | 1 | n/a |
| `O=3,H=0` | 590 | 22 | 22 | 0.071 | 0.857 | 2 | 0.795 | 4 | -0.224 |
| `O=3,H=1` | 259 | 14 | 15 | 0.143 | 0.643 | 2 | 0.811 | 4 | 0.018 |
| `O=3,H=2` | 233 | 12 | 13 | 0.107 | 0.536 | 2 | 0.721 | 4 | 0.352 |
| `O=3,H=3` | 198 | 9 | 10 | 0.071 | 0.393 | 2 | 0.955 | 4 | 0.218 |
| `O=3,H=4` | 96 | 5 | 6 | 0.036 | 0.214 | 2 | 0.948 | 4 | 0.156 |
| `O=3,H=5` | 16 | 0 | 1 | 0.000 | 0.000 | 2 | 0.875 | 3 | n/a |
| `O=4,H=0` | 364 | 15 | 15 | 0.036 | 0.571 | 3 | 0.563 | 4 | 0.186 |
| `O=4,H=1` | 207 | 9 | 10 | 0.107 | 0.429 | 2 | 0.841 | 4 | 0.483 |
| `O=4,H=2` | 160 | 6 | 7 | 0.071 | 0.286 | 2 | 0.700 | 4 | 0.646 |
| `O=4,H=3` | 205 | 3 | 4 | 0.036 | 0.143 | 2 | 0.966 | 4 | 0.171 |
| `O=4,H=4` | 326 | 0 | 1 | 0.000 | 0.000 | 2 | 0.724 | 4 | n/a |
| `O=5,H=-1` | 2 | 2 | 2 | 0.250 | 0.321 | 2 | 0.500 | 1 | n/a |
| `O=5,H=0` | 147 | 8 | 9 | 0.107 | 0.393 | 3 | 0.816 | 4 | 0.351 |
| `O=5,H=1` | 158 | 8 | 8 | 0.036 | 0.321 | 2 | 0.696 | 4 | 0.617 |
| `O=5,H=2` | 192 | 4 | 5 | 0.036 | 0.179 | 2 | 0.880 | 4 | 0.416 |
| `O=5,H=3` | 793 | 0 | 1 | 0.000 | 0.000 | 4 | 0.731 | 4 | n/a |
| `O=6,H=0` | 41 | 3 | 4 | 0.071 | 0.179 | 2 | 0.902 | 4 | 0.342 |
| `O=6,H=1` | 90 | 4 | 5 | 0.000 | 0.143 | 2 | 0.811 | 4 | 0.352 |
| `O=6,H=2` | 1052 | 0 | 1 | 0.000 | 0.000 | 5 | 0.460 | 4 | n/a |
| `O=7,H=0` | 9 | 2 | 3 | 0.036 | 0.107 | 2 | 0.667 | 3 | 0.545 |
| `O=7,H=1` | 1473 | 0 | 1 | 0.000 | 0.000 | 5 | 0.523 | 4 | n/a |
| `O=8,H=0` | 1548 | 0 | 1 | 0.000 | 0.000 | 4 | 0.753 | 4 | n/a |

## What this report does not do

- It selects no band and applies no threshold.
- It does not rank cells or search for the best triple. That search would choose the design from its own outcome.
- It does not redefine complexity. `D` remains the axis; `O` and `H` are matching variables.

The decision this feeds is which of the three outcomes recorded in §3 applies: common support is adequate and `D` bands are cut with `O` and `H` balanced; `H` cannot be balanced and the axis is renamed to pairwise interaction density with `H` reported separately; or `D` and `H` are structurally opposed with no common support, and the one-dimensional scale is abandoned for a `(D, H)` representation.

