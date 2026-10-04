# tests/test_lowcx_manifest.py
"""Offline tests for the lower-complexity selector.

Written and committed before the selector was ever run against a real pool. Two properties
carry the design and most of what follows attacks one of them: the optimum is fixed at 3 so
the satisfaction denominator is identical at every size, and nothing an agent produced may
reach the choice of instances. The rest pin the diversity rules, the seed ranges and the
level labels, which must describe conflict density and never task size.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_lowcx_manifest import (
    FIXED_OPTIMUM,
    REQUIRED_STRUCTURE_HISTOGRAM,
    MAX_TIGHTNESS_SHARE_DEN,
    MAX_TIGHTNESS_SHARE_NUM,
    PER_SIZE,
    SelectionError,
    audit_report,
    boundaries_for,
    build_manifests,
    canonical_key,
    eligible,
    level_for,
    load_pool,
    main,
    max_pairs,
    select,
    verify,
)
from scripts.build_pilot_manifest import content_hash

FIELDS = [
    "instance_id", "seed", "n_people", "tightness", "overlap", "travel_structure",
    "individually_reachable_count", "conflicting_pairs", "D", "D_reachable",
    "oracle_optimum", "optimum_over_n", "proven_optimal", "solver_status",
    "alpha_reachable", "higher_order_gap_H", "n_max_independent_sets", "max_degree",
    "edge_share_top1", "edge_share_top2", "largest_component", "triangle_violations",
    "triangle_triples_checked", "solve_seconds",
]

STRUCTURES = ("uniform", "clustered", "line", "random")
TIGHTNESS = (0.6, 0.8, 0.9, 1.0)


def row(n, o, *, seed, k=0, t=0.8, overlap=0.5, structure="uniform", alpha=None,
        proven=True):
    alpha = (o if alpha is None else alpha)
    return {
        "instance_id": f"{structure}-n{n}-t{int(t * 100)}-o{int(overlap * 100)}-s{seed}",
        "seed": seed, "n_people": n, "tightness": t, "overlap": overlap,
        "travel_structure": structure, "individually_reachable_count": n,
        "conflicting_pairs": k, "D": round(k / max_pairs(n), 6), "D_reachable": "",
        "oracle_optimum": o, "optimum_over_n": round(o / n, 6),
        "proven_optimal": proven, "solver_status": "OPTIMAL", "alpha_reachable": alpha,
        "higher_order_gap_H": alpha - o, "n_max_independent_sets": 1, "max_degree": 0,
        "edge_share_top1": "", "edge_share_top2": "", "largest_component": 1,
        "triangle_violations": 0, "triangle_triples_checked": 24, "solve_seconds": 0.01,
    }


def pool_rows(n, base_seed, *, count=40):
    """A pool with real variety: several optima, four structures, several tightness values.

    The cycles are deliberately coprime. An earlier version tied the optimum to `i % 2` and
    the structure to `i % 4`, which made the eligible half of the pool carry only two of the
    four structures -- so the fixture, not the selector, decided that an equal structure
    split was unreachable.
    """
    out = []
    for i in range(count):
        out.append(row(n, 2 if i % 5 == 0 else FIXED_OPTIMUM, seed=base_seed + i,
                       k=i % 3, t=TIGHTNESS[i % 3],
                       overlap=(0.2, 0.5, 0.8, 1.0)[i % 4],
                       structure=STRUCTURES[i % len(STRUCTURES)],
                       alpha=FIXED_OPTIMUM + (i % 2)))
    return out


def write_pool(dirpath: Path, rows, *, audit_ok=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    with (dirpath / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (dirpath / "pool_meta.json").write_text(json.dumps({
        "schema_version": "structural_calibration/1.0", "git_commit": "abc123",
        "n_completed": len(rows), "seed_start": min(r["seed"] for r in rows),
        "seed_end": max(r["seed"] for r in rows),
        "audit": {"all_checks_passed": audit_ok},
    }), encoding="utf-8")
    return dirpath


def parsed_pool(n, base_seed, **kw):
    """Pool rows in the shape `load_pool` returns, sorted as `eligible` sorts them."""
    return eligible([{**r, "proven_optimal": r["proven_optimal"]}
                     for r in pool_rows(n, base_seed, **kw)])


# --------------------------------------------------------------------------- #
# The fixed optimum.
# --------------------------------------------------------------------------- #
def test_only_the_fixed_optimum_is_eligible():
    rows = pool_rows(4, 40000)
    got = eligible(rows)
    assert got, "fixture produced no eligible candidate"
    assert {r["oracle_optimum"] for r in got} == {FIXED_OPTIMUM}


def test_the_fixed_optimum_is_three_so_the_denominator_is_identical_everywhere():
    """satisfaction = achieved / O. One optimum at every size makes the metric's four
    attainable values the same set, not merely a comparable one."""
    assert FIXED_OPTIMUM == 3


def test_an_unproven_optimum_is_never_eligible():
    rows = [row(4, FIXED_OPTIMUM, seed=40000 + i, proven=False) for i in range(20)]
    assert eligible(rows) == []


def test_every_selected_instance_carries_the_fixed_optimum():
    pools = {n: parsed_pool(n, base) for n, base in ((4, 40000), (5, 50000), (6, 60000))}
    chosen = select(pools)
    for n, rows in chosen.items():
        assert len(rows) == PER_SIZE
        assert {r["oracle_optimum"] for r in rows} == {FIXED_OPTIMUM}


# --------------------------------------------------------------------------- #
# Diversity, unchanged from the n = 8 design.
# --------------------------------------------------------------------------- #
def test_the_selection_meets_every_diversity_rule():
    pools = {n: parsed_pool(n, base) for n, base in ((4, 40000), (5, 50000), (6, 60000))}
    for n, rows in select(pools).items():
        tight = Counter(r["tightness"] for r in rows)
        assert len(tight) >= 2, f"n={n} used one tightness value"
        for t, c in tight.items():
            assert MAX_TIGHTNESS_SHARE_DEN * c <= MAX_TIGHTNESS_SHARE_NUM * PER_SIZE, \
                f"n={n} tightness {t} is {c}/{PER_SIZE}"
        assert len({r["travel_structure"] for r in rows}) >= 2


# --------------------------------------------------------------------------- #
# Travel structure: exact, and the same at every size.
# --------------------------------------------------------------------------- #
def test_the_required_histogram_is_an_equal_split_over_all_four_structures():
    assert REQUIRED_STRUCTURE_HISTOGRAM == {"clustered": 3, "line": 3, "random": 3,
                                            "uniform": 3}
    assert sum(REQUIRED_STRUCTURE_HISTOGRAM.values()) == PER_SIZE


def test_every_size_gets_exactly_that_histogram():
    pools = {n: parsed_pool(n, base) for n, base in ((4, 40000), (5, 50000), (6, 60000))}
    for n, rows in select(pools).items():
        got = Counter(r["travel_structure"] for r in rows)
        assert dict(got) == REQUIRED_STRUCTURE_HISTOGRAM, f"n={n} got {dict(got)}"


def test_the_histogram_is_identical_across_sizes():
    """Changing n must not change travel topology, or the two vary together."""
    pools = {n: parsed_pool(n, base) for n, base in ((4, 40000), (5, 50000), (6, 60000))}
    got = {n: dict(Counter(r["travel_structure"] for r in rows))
           for n, rows in select(pools).items()}
    assert len({tuple(sorted(h.items())) for h in got.values()}) == 1


def test_the_rejected_eleven_clustered_shape_is_now_unreachable():
    """The first selection returned 11 clustered + 1 line at every size. That shape
    satisfies the older 'at least 2 structures' rule and must now be refused."""
    pools = {4: parsed_pool(4, 40000)}
    rows = select(pools)[4]
    assert Counter(r["travel_structure"] for r in rows)["clustered"] == 3


def test_a_pool_that_cannot_fill_one_structure_is_refused_not_rebalanced():
    rows = [row(4, FIXED_OPTIMUM, seed=40000 + i, t=TIGHTNESS[i % 4],
                structure=("clustered", "line", "random")[i % 3]) for i in range(60)]
    with pytest.raises(SelectionError):
        select({4: eligible(rows)})       # no `uniform` candidate exists


def test_a_structure_outside_the_histogram_cannot_enter():
    rows = ([row(4, FIXED_OPTIMUM, seed=40000 + i, t=TIGHTNESS[i % 4],
                 structure=STRUCTURES[i % 4]) for i in range(40)]
            + [row(4, FIXED_OPTIMUM, seed=40100 + i, t=0.9, structure="grid")
               for i in range(20)])
    chosen = select({4: eligible(rows)})[4]
    assert "grid" not in {r["travel_structure"] for r in chosen}


def test_one_instance_per_seed():
    pools = {n: parsed_pool(n, base) for n, base in ((4, 40000), (5, 50000), (6, 60000))}
    chosen = select(pools)
    seeds = [r["seed"] for rows in chosen.values() for r in rows]
    assert len(seeds) == len(set(seeds))


def test_a_pool_that_cannot_meet_the_rules_is_refused_not_relaxed():
    """Every eligible candidate at one tightness: the 60% rule cannot be met, and the
    recorded response is to fail rather than to drop the rule."""
    rows = [row(4, FIXED_OPTIMUM, seed=40000 + i, t=0.8, structure=STRUCTURES[i % 4])
            for i in range(40)]
    with pytest.raises(SelectionError):
        select({4: eligible(rows)})


def test_too_few_candidates_is_an_error_with_the_count_in_it():
    rows = [row(4, FIXED_OPTIMUM, seed=40000 + i, t=TIGHTNESS[i % 4]) for i in range(5)]
    with pytest.raises(SelectionError, match="only 5 candidates"):
        select({4: eligible(rows)})


# --------------------------------------------------------------------------- #
# Determinism and the canonical order.
# --------------------------------------------------------------------------- #
def test_the_selection_is_reproducible():
    def run():
        pools = {n: parsed_pool(n, b) for n, b in ((4, 40000), (5, 50000), (6, 60000))}
        return {n: [r["instance_id"] for r in rows] for n, rows in select(pools).items()}
    assert run() == run()


def test_the_canonical_order_ignores_every_structural_property():
    """Order is seed, structure, tightness, overlap -- nothing about difficulty."""
    a = row(4, 3, seed=40000, k=0, alpha=3)
    b = dict(a, conflicting_pairs=5, higher_order_gap_H=9, alpha_reachable=12)
    assert canonical_key(a) == canonical_key(b)


def test_the_rule_prefers_earlier_candidates_in_canonical_order():
    """With the rules satisfiable many ways, the chosen set is the earliest one."""
    rows = eligible([row(4, FIXED_OPTIMUM, seed=40000 + i, t=TIGHTNESS[i % 4],
                         structure=STRUCTURES[i % 4]) for i in range(40)])
    chosen = select({4: rows})[4]
    assert [r["instance_id"] for r in chosen] == \
           [r["instance_id"] for r in rows[:PER_SIZE]]


# --------------------------------------------------------------------------- #
# Level labels describe density, never size.
# --------------------------------------------------------------------------- #
def test_the_boundaries_reproduce_the_frozen_design_at_n_eight():
    assert boundaries_for(8) == {"b1_easy_max": 1, "b2_medium_max": 7}


def test_the_boundaries_are_the_same_densities_at_every_size():
    assert boundaries_for(4) == {"b1_easy_max": 0, "b2_medium_max": 1}
    assert boundaries_for(5) == {"b1_easy_max": 0, "b2_medium_max": 2}
    assert boundaries_for(6) == {"b1_easy_max": 0, "b2_medium_max": 3}


def test_a_dense_small_instance_outranks_a_sparse_larger_one():
    """The behaviour the design requires: a level is a statement about conflict density,
    so `n = 5` at five conflicting pairs is hard while `n = 6` at none is easy."""
    assert level_for(5, boundaries_for(5)) == "hard"
    assert level_for(0, boundaries_for(6)) == "easy"


def test_the_level_of_every_entry_follows_from_its_conflict_count():
    pools = {4: parsed_pool(4, 40000)}
    chosen = select(pools)[4]
    _b, subset = build_manifests(4, chosen, {"_dir": "x"}, {"rule": "test"})
    for level, rows in subset["instances"].items():
        for e in rows:
            assert e["level"] == level == level_for(e["complexity_metric"],
                                                    boundaries_for(4))


# --------------------------------------------------------------------------- #
# Verification reads the documents, not the search.
# --------------------------------------------------------------------------- #
def build_one(n=4, base=40000):
    chosen = select({n: parsed_pool(n, base)})[n]
    return build_manifests(n, chosen, {"_dir": "x"}, {"rule": "test"})


def test_a_clean_pair_verifies():
    binning, subset = build_one()
    assert verify(binning, subset) == []


def test_a_wrong_optimum_is_caught():
    binning, subset = build_one()
    subset["instances"]["easy"][0]["optimum"] = 4
    assert any("expected 3" in p for p in verify(binning, subset))


def test_a_mislabelled_level_is_caught():
    binning, subset = build_one()
    victim = next(rows for rows in subset["instances"].values() if rows)[0]
    victim["complexity_metric"] = 99
    assert any("level" in p for p in verify(binning, subset))


def test_a_duplicate_seed_is_caught():
    binning, subset = build_one()
    rows = next(r for r in subset["instances"].values() if len(r) >= 2)
    rows[1]["seed"] = rows[0]["seed"]
    assert any("more than once" in p for p in verify(binning, subset))


def test_a_seed_outside_the_reserved_range_is_caught():
    binning, subset = build_one()
    next(r for r in subset["instances"].values() if r)[0]["seed"] = 100001
    assert any("outside" in p for p in verify(binning, subset))


def test_tampered_boundaries_are_caught():
    binning, subset = build_one()
    binning["boundaries"] = {"b1_easy_max": 9, "b2_medium_max": 99}
    assert any("boundaries" in p for p in verify(binning, subset))


def test_the_content_hashes_reproduce():
    binning, subset = build_one()
    for doc in (binning, subset):
        assert doc["content_hash"] == content_hash(
            {k: v for k, v in doc.items() if k not in ("content_hash", "timestamp_utc")})


# --------------------------------------------------------------------------- #
# No agent quantity may reach the selection.
# --------------------------------------------------------------------------- #
def test_the_selector_never_reads_an_agent_quantity():
    import tokenize
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_lowcx_manifest.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING)).lower()
    for banned in ("satisfaction", "tokens_total", "termination", "run_result",
                   "c1_react", "c3_mas", "final_plan", "transcript", "score"):
        assert banned not in code, f"the selector reads {banned!r}"


def test_the_held_out_range_is_not_reachable_from_this_selector():
    from scripts.build_lowcx_manifest import SEED_RANGE_FOR_SIZE, SEED_RANGES
    for name in SEED_RANGE_FOR_SIZE.values():
        lo, hi = SEED_RANGES[name]
        assert hi < 100000


# --------------------------------------------------------------------------- #
# The audit report carries what the amendment has to show.
# --------------------------------------------------------------------------- #
def test_the_audit_reports_the_distributions_the_amendment_needs():
    chosen = select({4: parsed_pool(4, 40000)})[4]
    _b, subset = build_manifests(4, chosen, {"_dir": "x"}, {"rule": "test"})
    rep = audit_report(4, chosen, subset)
    for key in ("instance_ids", "optimum", "conflict_pairs", "higher_order_gap_H",
                "tightness", "travel_structure", "content_hash", "density_D", "levels"):
        assert key in rep, f"audit is missing {key}"
    assert rep["optimum"] == {"3": PER_SIZE}
    assert len(rep["instance_ids"]) == PER_SIZE


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_verified_manifests(tmp_path):
    pools = []
    for n, base in ((4, 40000), (5, 50000), (6, 60000)):
        d = write_pool(tmp_path / f"n{n}", pool_rows(n, base))
        pools += ["--pool", f"{n}={d}"]
    out = tmp_path / "manifests"
    assert main([*pools, "--out", str(out)]) == 0
    subsets = sorted(out.glob("subset__lowcx_n*.json"))
    assert len(subsets) == 3
    total = 0
    for p in subsets:
        doc = json.loads(p.read_text(encoding="utf-8"))
        entries = [e for rows in doc["instances"].values() for e in rows]
        total += len(entries)
        assert {e["optimum"] for e in entries} == {FIXED_OPTIMUM}
    assert total == 3 * PER_SIZE
    audit = json.loads((out / "audit__lowcx.json").read_text(encoding="utf-8"))
    assert sorted(audit["per_size_report"]) == ["4", "5", "6"]


def test_the_cli_refuses_to_overwrite(tmp_path):
    d = write_pool(tmp_path / "n4", pool_rows(4, 40000))
    out = tmp_path / "m"
    assert main(["--pool", f"4={d}", "--out", str(out)]) == 0
    with pytest.raises(SelectionError, match="refusing to overwrite"):
        main(["--pool", f"4={d}", "--out", str(out)])


def test_the_cli_refuses_a_pool_that_failed_its_audit(tmp_path):
    d = write_pool(tmp_path / "n4", pool_rows(4, 40000), audit_ok=False)
    with pytest.raises(SelectionError, match="completeness audit"):
        main(["--pool", f"4={d}", "--out", str(tmp_path / "m")])


def test_the_cli_refuses_a_size_with_no_reserved_range(tmp_path):
    d = write_pool(tmp_path / "n7", [row(7, FIXED_OPTIMUM, seed=70000)])
    with pytest.raises(SystemExit):
        main(["--pool", f"7={d}", "--out", str(tmp_path / "m")])


def test_a_pool_of_the_wrong_size_is_refused(tmp_path):
    d = write_pool(tmp_path / "n4", pool_rows(5, 50000))
    with pytest.raises(SelectionError, match="expected n_people=4"):
        load_pool(d, 4)


def test_a_wrong_structure_histogram_is_caught_by_verification():
    binning, subset = build_one()
    victim = next(r for r in subset["instances"].values() if r)[0]
    victim["cell"] = dict(victim["cell"], travel_structure="clustered")
    assert any("travel structures" in p for p in verify(binning, subset))


# --------------------------------------------------------------------------- #
# The frozen n = 8 path is not touched by any of this.
# --------------------------------------------------------------------------- #
def test_the_n8_selector_still_carries_its_own_rules():
    """This extension lives in its own file. The band selector must be unchanged: same
    bands, same reference optimum histogram, same n."""
    from scripts import build_band_manifest as b8
    assert b8.N_PEOPLE == 8
    assert b8.BANDS == (("low", "easy", 0, 1), ("medium", "medium", 7, 7),
                        ("high", "hard", 13, 16))
    assert b8.REFERENCE_OPTIMUM_HISTOGRAM == {3: 41, 4: 24}
    assert b8.MIN_TRAVEL_STRUCTURES == 2      # the older, weaker rule stays as it was
    assert not hasattr(b8, "REQUIRED_STRUCTURE_HISTOGRAM")


def test_the_frozen_n8_subset_is_untouched_on_disk():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/manifests/subset__bands_budget_dev__911e4864d6fb.json"
    if not path.exists():
        pytest.skip("frozen n=8 subset not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert content_hash(doc) == doc["content_hash"]
    assert doc["content_hash"].startswith("911e4864d6fb")
    entries = [e for rows in doc["instances"].values() for e in rows]
    assert len(entries) == 60


def test_each_document_is_named_after_its_own_hash(tmp_path):
    """The runner locates the binning document by the hash the subset points at, so a
    binning file named after the subset's hash is undiscoverable."""
    import re
    pools = []
    for n, base in ((4, 40000), (5, 50000), (6, 60000)):
        d = write_pool(tmp_path / f"n{n}", pool_rows(n, base))
        pools += ["--pool", f"{n}={d}"]
    out = tmp_path / "m"
    assert main([*pools, "--out", str(out)]) == 0
    for path in out.glob("*__lowcx_n*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        stamped = re.search(r"__([0-9a-f]{12})\.json$", path.name).group(1)
        assert doc["content_hash"].startswith(stamped), path.name


def test_the_runner_can_discover_the_binning_document(tmp_path):
    """The discovery the sweep actually performs: glob for the subset's manifest_hash."""
    pools = []
    for n, base in ((4, 40000), (5, 50000), (6, 60000)):
        d = write_pool(tmp_path / f"n{n}", pool_rows(n, base))
        pools += ["--pool", f"{n}={d}"]
    out = tmp_path / "m"
    assert main([*pools, "--out", str(out)]) == 0
    for sub in out.glob("subset__lowcx_n*.json"):
        doc = json.loads(sub.read_text(encoding="utf-8"))
        found = list(sub.parent.glob(f"binning__*__{doc['manifest_hash'][:12]}*.json"))
        assert len(found) == 1, f"{sub.name}: found {len(found)} binning documents"
