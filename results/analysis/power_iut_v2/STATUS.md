# Status: exploratory, superseded — binds nothing

This directory holds a synthetic power simulation for the intersection-union test. On
2026-08-10 the project stopped determining the sample size from this model
(THESIS_DECISIONS section 5, amendment of that date).

**The N it reports is not adopted.** `N = 35` from v1 and `N = 95` from v2 differ by nearly a
factor of three, and the whole of that difference comes from assumptions nobody has measured:
how much instance heterogeneity there is on the logit scale, how often an agent produces no
plan at all, and how the two interact. A sample size is only as good as the variance estimate
under it.

Nothing here is deleted. The two runs, their disagreement and the defect found in the first
Type-I rule are the audit trail for why the route to `N` changed.

The size will instead come from real development runs: budget-dev fixes the cap, then C1 and
C3 development runs establish the real spread, the real zero rate and the real tie rate, and
`N` is frozen from those before the held-out range opens.

See also `../type_one_validation/STATUS.md`.
