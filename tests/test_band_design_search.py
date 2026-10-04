# tests/test_band_design_search.py
"""Offline tests for the exhaustive band-design search.

The search decides whether the registered design is reachable, so the tests concentrate on
the two ways it could lie: declaring a design admissible when no balanced sample exists,
and letting a non-binding diagnostic influence the outcome. The enumeration and tie-break
are checked exactly, because both are supposed to remove discretion rather than hide it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.search_band_design import (
    MAX_PAIRS,
    NON_BINDING,
    Counts,
    band_triples,
    joint_allocation,
    main,
    _survives_prefilter,
    optimum_capacity,
    search,
    separation,
)
from tests.test_structural_analysis import row, write_pool


def pool(spec) -> list[dict]:
    """spec: list of (pairs, optimum, tightness, structure, count, H)."""
    rows, i = [], 0
    for pairs, optimum, tight, struct, count, *rest in spec:
        H = rest[0] if rest else 0
        for _ in range(count):
            rows.append(row(i, pairs=pairs, optimum=optimum, tightness=tight,
                            structure=struct, H=H))
            i += 1
    return rows


# --------------------------------------------------------------------------- #
# Enumeration and separation.
# --------------------------------------------------------------------------- #
def test_triples_are_ordered_disjoint_and_really_separated():
    for bands in band_triples(max_k=8):
        (a1, b1), (a2, b2), (a3, b3) = bands
        assert a1 <= b1 < a2 <= b2 < a3 <= b3
        assert a2 - b1 - 1 >= 1 and a3 - b2 - 1 >= 1


def test_width_one_bands_are_allowed():
    assert ((0, 0), (2, 2), (4, 4)) in set(band_triples(max_k=4))


def test_separation_is_the_minimum_gap():
    assert separation([(0, 1), (3, 4), (9, 10)]) == 1
    assert separation([(0, 1), (5, 6), (11, 12)]) == 3


def test_enumeration_covers_the_full_conflict_range():
    seen = {k for bands in band_triples(max_k=MAX_PAIRS) for lo, hi in bands
            for k in (lo, hi)}
    assert min(seen) == 0 and max(seen) == MAX_PAIRS


# --------------------------------------------------------------------------- #
# The joint feasibility test -- the reason raw per-band checks are not enough.
# --------------------------------------------------------------------------- #
def test_bands_that_pass_separately_can_still_fail_jointly():
    """The hole the joint test closes.

    Every band holds plenty of candidates and two tightness values, and the optimum
    histograms can be matched. But the candidates that make the histograms match all sit at
    one tightness, so no selected sample satisfies diversity and matching at once.
    """
    rows = pool([
        # band 1: matching optimum 3 only at tightness 1.0; its variety is at optimum 5
        (1, 3, 1.0, "uniform", 60), (1, 5, 0.8, "line", 60),
        # band 2: same shape
        (5, 3, 1.0, "uniform", 60), (5, 7, 0.8, "line", 60),
        # band 3: same shape
        (9, 3, 1.0, "uniform", 60), (9, 8, 0.8, "line", 60),
    ])
    counts = Counts(rows)
    avails = [counts.interval(1, 1), counts.interval(5, 5), counts.interval(9, 9)]
    assert optimum_capacity(counts, avails) >= 50      # the cheap check says yes
    ok, _m, _plan = joint_allocation(counts, avails, minimum=50)
    assert ok is False                                  # the joint test says no


def test_a_genuinely_balanced_pool_is_admissible():
    spec = []
    for k in (1, 5, 9):
        for t, s in ((0.8, "uniform"), (1.0, "line")):
            for o in (3, 4):
                spec.append((k, o, t, s, 20))
    counts = Counts(pool(spec))
    avails = [counts.interval(k, k) for k in (1, 5, 9)]
    ok, m, plan = joint_allocation(counts, avails, minimum=50)
    assert ok and m >= 50
    # The histogram is shared by construction, so it is identical in all three bands.
    assert sum(plan["optimum_histogram"].values()) == m


def test_the_sixty_percent_rule_binds_on_the_selected_sample():
    """One tightness dominating the pool must not be selectable past the cap."""
    spec = []
    for k in (1, 5, 9):
        spec.append((k, 3, 1.0, "uniform", 100))   # dominant
        spec.append((k, 3, 0.8, "line", 5))        # scarce
    counts = Counts(pool(spec))
    avails = [counts.interval(k, k) for k in (1, 5, 9)]
    ok, m, _ = joint_allocation(counts, avails, minimum=None, maximise=True)
    assert ok
    # capacity is capped by the scarce value: 5 scarce -> at most 5/0.4 = 12
    assert m <= 12


def test_two_structures_are_required_in_the_selected_sample():
    spec = [(k, 3, t, "uniform", 60) for k in (1, 5, 9) for t in (0.8, 1.0)]
    counts = Counts(pool(spec))
    avails = [counts.interval(k, k) for k in (1, 5, 9)]
    ok, _m, _ = joint_allocation(counts, avails, minimum=50)
    assert ok is False   # only one travel structure exists anywhere


# --------------------------------------------------------------------------- #
# Verdict, tie-break and the standing of diagnostics.
# --------------------------------------------------------------------------- #
def _balanced_spec(ks, per=20, H=0):
    return [(k, o, t, s, per, H) for k in ks for o in (3, 4)
            for t, s in ((0.8, "uniform"), (1.0, "line"))]


def test_no_feasible_design_is_reported_rather_than_repaired():
    rows = pool([(1, 3, 1.0, "uniform", 200), (5, 4, 1.0, "uniform", 200),
                 (9, 5, 1.0, "uniform", 200)])   # disjoint optima, one tightness
    out = search(rows, target=50, max_k=12)
    assert out["verdict"] == "NO_FEASIBLE_DESIGN"
    assert out["selected"] is None
    assert out["n_admissible"] == 0


def test_the_tie_break_prefers_the_largest_gap_then_capacity_then_order():
    rows = pool(_balanced_spec([0, 2, 4, 6, 8, 10]))
    out = search(rows, target=50, max_k=12)
    assert out["verdict"] == "FEASIBLE"
    sel = out["selected"]
    # Among admissible triples the widest achievable gap must win.
    assert sel["separation"] == max(sel["separation"], out["tie_break"]["max_separation"])
    assert sel["separation"] == out["tie_break"]["max_separation"]
    assert sel["joint_capacity"] == out["tie_break"]["max_joint_capacity"]


def test_a_diagnostic_cannot_change_the_verdict():
    """H differs wildly between two otherwise identical pools; the choice must not move."""
    base = _balanced_spec([0, 2, 4, 6, 8, 10], H=0)
    skewed = [(k, o, t, s, n, (0 if k < 5 else 4)) for (k, o, t, s, n, _h) in base]
    a = search(pool(base), target=50, max_k=12)["selected"]["bands"]
    b = search(pool(skewed), target=50, max_k=12)["selected"]["bands"]
    assert a == b


def test_diagnostics_are_declared_non_binding_in_the_artifacts(tmp_path):
    rows = pool(_balanced_spec([0, 2, 4, 6, 8, 10]))
    out = tmp_path / "out"
    assert main(["--pool", str(write_pool(tmp_path, rows)), "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(doc["non_binding_diagnostics"]) == set(NON_BINDING)
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "non-binding" in text
    assert "never a reason to search again" in text
    assert "overlap" in text and "H" in text


# --------------------------------------------------------------------------- #
# The optimisation must change the cost and nothing else.
#
# `search` prunes with cheap tests and walks separations downwards, stopping at the first
# level that yields anything. Both are claims about semantics, not just speed: the pruning
# must never reject a triple that a full check would accept, and the early stop is only
# valid because the tie-break maximises separation first. These tests hold the fast path to
# a brute-force reference on a small pool.
# --------------------------------------------------------------------------- #
def _brute_force(rows, target=50, max_k=10):
    """Every triple tested, then the full tie-break. The reference implementation."""
    counts = Counts(rows)
    admissible = []
    for bands in band_triples(max_k=max_k):
        avails = [counts.interval(lo, hi) for lo, hi in bands]
        ok, _m, _p = joint_allocation(counts, avails, minimum=target)
        if ok:
            admissible.append(bands)
    if not admissible:
        return None
    best_gap = max(separation(b) for b in admissible)
    finalists = [b for b in admissible if separation(b) == best_gap]
    caps = {}
    for b in finalists:
        avails = [counts.interval(lo, hi) for lo, hi in b]
        _ok, cap, _p = joint_allocation(counts, avails, minimum=target, maximise=True)
        caps[b] = cap
    best_cap = max(caps.values())
    finalists = sorted(b for b in finalists if caps[b] == best_cap)
    return {"bands": [list(x) for x in finalists[0]], "separation": best_gap,
            "joint_capacity": best_cap, "n_at_gap": len(
                [b for b in admissible if separation(b) == best_gap])}


def test_the_fast_search_returns_exactly_what_brute_force_returns():
    rows = pool(_balanced_spec([0, 1, 3, 4, 6, 7, 9, 10], per=12))
    fast = search(rows, target=50, max_k=10)
    slow = _brute_force(rows, target=50, max_k=10)
    assert slow is not None and fast["verdict"] == "FEASIBLE"
    assert fast["selected"]["bands"] == slow["bands"]
    assert fast["selected"]["separation"] == slow["separation"]
    assert fast["selected"]["joint_capacity"] == slow["joint_capacity"]
    # every admissible triple at the deciding separation was enumerated, so the capacity
    # maximised inside that level is the true maximum there
    assert fast["n_admissible_at_selected_separation"] == slow["n_at_gap"]


def test_the_cheap_prefilter_never_rejects_an_admissible_triple():
    """It is a necessary condition only: whatever it drops, the solver would also drop."""
    rows = pool(_balanced_spec([0, 1, 3, 4, 6, 7, 9, 10], per=12)
                + [(2, 3, 1.0, "uniform", 200), (5, 8, 1.0, "uniform", 200)])
    counts = Counts(rows)
    checked = dropped = 0
    for bands in band_triples(max_k=10):
        checked += 1
        if _survives_prefilter(counts, bands, 50):
            continue
        dropped += 1
        avails = [counts.interval(lo, hi) for lo, hi in bands]
        ok, _m, _p = joint_allocation(counts, avails, minimum=50)
        assert ok is False, f"prefilter dropped an admissible triple {bands}"
    assert checked > 0 and dropped > 0     # the test would be vacuous otherwise


def test_no_feasible_design_still_exhausts_every_separation():
    """The early stop cannot produce a false negative: it only fires on a hit."""
    rows = pool([(1, 3, 1.0, "uniform", 200), (5, 4, 1.0, "uniform", 200),
                 (9, 5, 1.0, "uniform", 200)])
    out = search(rows, target=50, max_k=10)
    assert out["verdict"] == "NO_FEASIBLE_DESIGN"
    assert _brute_force(rows, target=50, max_k=10) is None
    # nothing was left untested: every surviving triple reached the solver
    assert out["triples_considered"] == out["triples_pruned_before_solver"] + \
        out["triples_sent_to_solver"]


def test_the_enumeration_strategy_is_declared_in_the_artifact():
    rows = pool(_balanced_spec([0, 2, 4, 6, 8, 10]))
    out = search(rows, target=50, max_k=12)
    assert "descending separation" in out["enumeration"]
    assert "cannot win the tie-break" in out["enumeration"]


def _executable_source(path: str) -> str:
    """Source with comments and string literals removed.

    Grepping the raw file matches the prose that explains what the script must not do,
    which is exactly the text these checks are meant to be independent of. Only executable
    tokens are inspected.
    """
    import io
    import tokenize
    out = []
    with open(path, encoding="utf-8") as fh:
        for tok in tokenize.generate_tokens(io.StringIO(fh.read()).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                out.append(tok.string)
    return " ".join(out)


def test_the_search_never_reads_an_llm_quantity():
    src = _executable_source("scripts/search_band_design.py")
    for forbidden in ("satisfaction", "agents", "llm_client", "LLMClient", "c1_react",
                      "c3_mas", "proposal"):
        assert forbidden not in src


def test_overlap_never_appears_in_a_binding_path():
    """It may be reported, but must not appear in a constraint or in the tie-break."""
    src = _executable_source("scripts/search_band_design.py")
    binding = src.split("def diagnose")[0]      # everything before the diagnostics block
    assert "overlap" not in binding
