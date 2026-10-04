# Status: exploratory, superseded — binds nothing

This directory holds the corrected Type-I validation of the bootstrap decision rule, run at
`N = 95` on 2026-08-10 and completed after the decision to stop sizing the experiment from a
synthetic model (THESIS_DECISIONS section 5, amendment of 2026-08-10).

It was already running when that decision was taken and was allowed to finish rather than be
discarded. Its data-generating process is the same synthetic latent-logistic hurdle model
whose assumptions about agent behaviour are not empirically established, so **no number here
is binding**, on `N`, on the choice of test, or on whether the main experiment starts.

Result, for the record: verdict ANTICONSERVATIVE, one cell of 216 surviving Holm at 0.0589
against a nominal 0.05, family mean 0.0517. The cell is `sigma = 0, lambda = 0` — the corner
with the fewest distinct outcome values.

Why it is kept rather than deleted: the question it raises does not depend on the invented
parameters. Real satisfaction is `achieved / O` with `O` in {3, 4}, so per-instance
differences will also be coarse and tie-heavy, and a percentile bootstrap on such a sample
can miss its nominal level. The registered response is unchanged — if this reproduces on
real development data, the statistical test changes and `N` does not.

Companion exploratory directories: `../power_iut` (v1, N = 35) and `../power_iut_v2`
(v2, N = 95). Neither number is adopted.
