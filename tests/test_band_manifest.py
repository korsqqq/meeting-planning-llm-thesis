# tests/test_band_manifest.py
"""Offline tests for the band-matched instance selector.

The selector can fail in ways that still produce a plausible-looking manifest: bands that
match on the optimum only among the listed values, a diversity rule satisfied on the raw
band but not on the sample, two instances sharing a seed, or a level label that no longer
tracks the band it stands for. Each of those has a test here, on synthetic pools rather
than on the real one, so a failure points at the rule and not at the data.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.build_band_manifest import (
    BANDS,
    LEVEL_BOUNDARIES,
    MAX_TIGHTNESS_SHARE_DEN,
    MAX_TIGHTNESS_SHARE_NUM,
    N_PEOPLE,
    REFERENCE_OPTIMUM_HISTOGRAM,
    SelectionError,
    SeedRangeError,
    build_manifests,
    canonical_key,
    choose,
    eligible,
    histogram_ladder,
    main,
    scaled_histogram,
    select,
    verify,
)
from scripts.build_pilot_manifest import content_hash
from scripts.collect_structural_calibration import SEED_RANGES
from scripts.run_pilot_sweep import level_for_metric


# --------------------------------------------------------------------------- #
# Synthetic pools.
# --------------------------------------------------------------------------- #
def _row(seed, pairs, optimum, tightness, structure, overlap=0.2):
    return {
        "instance_id": f"{structure}-n8-t{int(tightness * 100)}-s{seed}-k{pairs}",
        "seed": seed, "n_people": N_PEOPLE, "tightness": tightness, "overlap": overlap,
        "travel_structure": structure, "conflicting_pairs": pairs,
        "oracle_optimum": optimum, "proven_optimal": True,
        "alpha_reachable": optimum, "higher_order_gap_H": 0,
    }


def make_pool(per_band=40, seed0=30000):
    """A pool wide enough that every requirement can be met, and none is met by accident."""
    rows, seed = [], seed0
    for _band, _level, lo, hi in BANDS:
        for i in range(per_band):
            rows.append(_row(
                seed=seed,
                pairs=lo + (i % (hi - lo + 1)),
                optimum=3 if i % 3 else 4,
                tightness=(0.8, 0.9, 1.0)[i % 3],
                structure=("uniform", "clustered", "line", "random")[i % 4],
            ))
            seed += 1
    return rows


# --------------------------------------------------------------------------- #
# The shared optimum histogram.
# --------------------------------------------------------------------------- #
def test_the_scaled_histogram_keeps_the_frozen_composition():
    """41:24 out of 65 at twenty per band is 13 and 7, by largest remainder."""
    assert scaled_histogram(20) == {3: 13, 4: 7}
    assert sum(scaled_histogram(20).values()) == 20


@pytest.mark.parametrize("total", [1, 5, 20, 33, 65, 95])
def test_the_scaled_histogram_always_sums_to_the_requested_size(total):
    assert sum(scaled_histogram(total).values()) == total


def test_at_the_reference_size_the_scaling_is_the_identity():
    assert scaled_histogram(sum(REFERENCE_OPTIMUM_HISTOGRAM.values())) == \
        REFERENCE_OPTIMUM_HISTOGRAM


def test_the_ladder_starts_at_the_scaled_reference_and_moves_away_from_it():
    ladder = list(histogram_ladder(20, [2, 3, 4]))
    assert ladder[0] == scaled_histogram(20)

    def distance(h):
        keys = set(h) | set(scaled_histogram(20))
        return sum(abs(h.get(k, 0) - scaled_histogram(20).get(k, 0)) for k in keys)

    assert [distance(h) for h in ladder] == sorted(distance(h) for h in ladder)


def test_the_ladder_is_complete_and_free_of_duplicates():
    ladder = list(histogram_ladder(4, [2, 3]))
    assert len(ladder) == len({tuple(sorted(h.items())) for h in ladder})
    assert all(sum(h.values()) == 4 for h in ladder)
    assert len(ladder) == 5                    # (4,0) (3,1) (2,2) (1,3) (0,4)


# --------------------------------------------------------------------------- #
# Eligibility.
# --------------------------------------------------------------------------- #
def test_only_the_three_frozen_bands_are_eligible():
    rows = [_row(30000, k, 3, 0.9, "uniform") for k in range(0, 20)]
    pools = eligible(rows)
    assert {r["conflicting_pairs"] for r in pools["low"]} == {0, 1}
    assert {r["conflicting_pairs"] for r in pools["medium"]} == {7}
    assert {r["conflicting_pairs"] for r in pools["high"]} == {13, 14, 15, 16}
    # the gaps really are gaps
    assert not any(r["conflicting_pairs"] in {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}
                   for rows_ in pools.values() for r in rows_)


def test_an_unproven_optimum_cannot_enter_a_design_matched_on_the_optimum():
    r = _row(30000, 7, 3, 0.9, "uniform")
    r["proven_optimal"] = False
    assert eligible([r])["medium"] == []


def test_a_different_n_people_is_refused():
    r = _row(30000, 7, 3, 0.9, "uniform")
    r["n_people"] = 6
    assert eligible([r])["medium"] == []


def test_candidates_are_ordered_by_the_collector_walk():
    rows = [_row(30001, 7, 3, 0.9, "uniform"), _row(30000, 7, 3, 0.9, "uniform")]
    assert [r["seed"] for r in eligible(rows)["medium"]] == [30000, 30001]
    assert canonical_key(rows[0])[0] == 30001


# --------------------------------------------------------------------------- #
# The selection itself.
# --------------------------------------------------------------------------- #
def test_the_selection_meets_every_frozen_requirement_at_once():
    chosen, prov = choose(eligible(make_pool()), 10)
    assert prov["used_the_scaled_reference"] is True
    histograms = {b: tuple(sorted(Counter(r["oracle_optimum"] for r in rows).items()))
                  for b, rows in chosen.items()}
    assert len(set(histograms.values())) == 1
    for band, rows in chosen.items():
        assert len(rows) == 10
        assert len({r["tightness"] for r in rows}) >= 2
        assert len({r["travel_structure"] for r in rows}) >= 2
        for _t, c in Counter(r["tightness"] for r in rows).items():
            assert MAX_TIGHTNESS_SHARE_DEN * c <= MAX_TIGHTNESS_SHARE_NUM * len(rows)


def test_no_seed_is_used_twice_even_when_one_seed_could_serve_two_bands():
    """A seed appears in several bands only because the pool reuses it; the manifest may
    not, because the bootstrap treats instances as independent units."""
    rows = []
    for seed in range(30000, 30030):
        for _band, _lv, lo, _hi in BANDS:
            for i, (t, s) in enumerate([(0.8, "uniform"), (0.9, "clustered"),
                                        (1.0, "line")]):
                rows.append(_row(seed, lo, 3 if i else 4, t, s, overlap=0.2 + 0.1 * i))
    chosen, _ = choose(eligible(rows), 5)
    seeds = [r["seed"] for band in chosen.values() for r in band]
    assert len(seeds) == 15
    assert len(set(seeds)) == 15


def test_the_selection_is_the_earliest_admissible_one():
    """Fifteen admissible candidates per band, ten needed: the first ten must win."""
    chosen, _ = choose(eligible(make_pool(per_band=15)), 10)
    for band, _lv, _lo, _hi in BANDS:
        pool = eligible(make_pool(per_band=15))[band]
        picked = {r["instance_id"] for r in chosen[band]}
        ranks = sorted(i for i, r in enumerate(pool) if r["instance_id"] in picked)
        assert sum(ranks) <= sum(range(10)) + 6      # near-earliest under the constraints
        assert ranks[0] == 0


def test_selection_is_reproducible():
    a, _ = choose(eligible(make_pool()), 10)
    b, _ = choose(eligible(make_pool()), 10)
    assert {k: [r["instance_id"] for r in v] for k, v in a.items()} == \
           {k: [r["instance_id"] for r in v] for k, v in b.items()}


def test_a_band_that_cannot_supply_the_size_is_refused_not_shrunk():
    rows = [r for r in make_pool() if r["conflicting_pairs"] != 7][:20]
    with pytest.raises(SelectionError):
        choose(eligible(rows), 10)


def test_a_pool_with_only_one_travel_structure_is_refused():
    rows = [r for r in make_pool(per_band=60) if r["travel_structure"] == "uniform"]
    with pytest.raises(SelectionError):
        choose(eligible(rows), 10)


def test_the_sixty_percent_tightness_cap_binds_on_the_selected_sample():
    """The band is dominated by one tightness; a valid selection must dilute it."""
    rows = []
    seed = 30000
    for _band, _lv, lo, _hi in BANDS:
        for i in range(60):
            rows.append(_row(seed, lo, 3 if i % 3 else 4,
                             0.9 if i < 45 else 1.0,
                             ("uniform", "clustered")[i % 2]))
            seed += 1
    chosen, _ = choose(eligible(rows), 10)
    for rows_ in chosen.values():
        top = Counter(r["tightness"] for r in rows_).most_common(1)[0][1]
        assert MAX_TIGHTNESS_SHARE_DEN * top <= MAX_TIGHTNESS_SHARE_NUM * 10


def test_optima_outside_the_shared_histogram_never_enter():
    rows = make_pool()
    for r in rows[::7]:
        r["oracle_optimum"] = 5
    chosen = select(eligible(rows), 10, {3: 7, 4: 3})
    assert chosen is not None
    assert {r["oracle_optimum"] for band in chosen.values() for r in band} == {3, 4}


# --------------------------------------------------------------------------- #
# The emitted documents.
# --------------------------------------------------------------------------- #
@pytest.fixture
def manifests():
    chosen, prov = choose(eligible(make_pool()), 10)
    return build_manifests(chosen, per_band=10, seed_range="budget_dev",
                           pool_meta={"git_commit": "x", "n_completed": 120},
                           provenance=prov)


def test_the_emitted_pair_passes_its_own_verification(manifests):
    binning, subset = manifests
    assert verify(binning, subset, 10, "budget_dev") == []


def test_both_hashes_reproduce_and_the_subset_points_at_the_binning(manifests):
    binning, subset = manifests
    assert content_hash(binning) == binning["content_hash"]
    assert content_hash(subset) == subset["content_hash"]
    assert subset["manifest_hash"] == binning["content_hash"]


def test_a_tampered_manifest_fails_verification(manifests):
    binning, subset = manifests
    subset["instances"]["easy"][0]["complexity_metric"] = 9
    problems = verify(binning, subset, 10, "budget_dev")
    assert any("outside the low band" in p for p in problems)
    assert any("content hash does not reproduce" in p for p in problems)


def test_verification_catches_a_repeated_seed(manifests):
    binning, subset = manifests
    subset["instances"]["hard"][1]["seed"] = subset["instances"]["hard"][0]["seed"]
    assert any("used more than once" in p for p in verify(binning, subset, 10,
                                                          "budget_dev"))


def test_verification_catches_histograms_that_stopped_matching(manifests):
    binning, subset = manifests
    subset["instances"]["hard"][0]["optimum"] = 9
    assert any("optimum histograms differ" in p
               for p in verify(binning, subset, 10, "budget_dev"))


def test_verification_catches_a_boundary_that_no_longer_tracks_the_bands(manifests):
    binning, subset = manifests
    binning["boundaries"] = {"b1_easy_max": 7, "b2_medium_max": 12}
    assert any("does not map onto level" in p
               for p in verify(binning, subset, 10, "budget_dev"))


def test_a_seed_outside_the_reserved_range_is_refused():
    chosen, prov = choose(eligible(make_pool(seed0=20000)), 10)
    with pytest.raises(SeedRangeError):
        build_manifests(chosen, per_band=10, seed_range="budget_dev", pool_meta={},
                        provenance=prov)


# --------------------------------------------------------------------------- #
# Compatibility with the frozen sweep runner.
# --------------------------------------------------------------------------- #
def test_the_boundaries_reproduce_the_band_membership_the_runner_will_derive(manifests):
    """The runner re-derives each level from the metric and refuses a mismatch, so the
    boundaries must be exactly right for every conflict count the bands contain."""
    _binning, subset = manifests
    for _band, level, lo, hi in BANDS:
        for k in range(lo, hi + 1):
            assert level_for_metric(k, LEVEL_BOUNDARIES).value == level
    for level, rows in subset["instances"].items():
        for e in rows:
            assert level_for_metric(e["complexity_metric"],
                                    LEVEL_BOUNDARIES).value == level


def test_the_boundaries_sit_in_the_frozen_gaps_not_inside_a_band():
    edges = {k for _b, _lv, lo, hi in BANDS for k in (lo, hi)}
    assert LEVEL_BOUNDARIES["b1_easy_max"] in edges          # 1, the top of Low
    assert LEVEL_BOUNDARIES["b2_medium_max"] in edges        # 7, the whole of Medium


def test_the_subset_carries_every_field_the_runner_reads(manifests):
    _binning, subset = manifests
    for key in ("content_hash", "manifest_hash", "grid_name", "instances"):
        assert key in subset
    for rows in subset["instances"].values():
        for e in rows:
            assert {"instance_id", "seed", "level", "optimum", "complexity_metric",
                    "cell"} <= set(e)
            assert {"n_people", "tightness", "overlap",
                    "travel_structure"} <= set(e["cell"])


# --------------------------------------------------------------------------- #
# The seed ranges.
# --------------------------------------------------------------------------- #
def test_only_the_frozen_held_out_blocks_are_reachable_from_this_selector():
    """Superseded form of "held-out is unreachable". The 2026-08-26 amendment opened four
    named blocks; everything else about the guarantee is unchanged."""
    assert "main_heldout" not in SEED_RANGES
    assert {k: v for k, v in SEED_RANGES.items() if v[1] >= 100000} == {
    "held_out_n8": (100_000, 109_999),
    "held_out_n4": (110_000, 119_999),
    "held_out_n5": (120_000, 129_999),
    "held_out_n6": (130_000, 139_999),
}
    assert max(hi for _lo, hi in SEED_RANGES.values()) < 140000


def test_budget_dev_is_the_range_the_amendment_reserved():
    assert SEED_RANGES["budget_dev"] == (30000, 39999)


def test_the_ranges_do_not_overlap():
    spans = sorted(SEED_RANGES.values())
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))


# --------------------------------------------------------------------------- #
# No LLM quantity can reach the selection.
# --------------------------------------------------------------------------- #
def test_the_selector_never_reads_an_agent_quantity():
    """Guards the rule in section 3: selecting instances by observed model behaviour would
    tune the complexity axis to the thing under test."""
    import tokenize
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_band_manifest.py"
    with path.open("rb") as fh:
        code = "".join(
            tok.string for tok in tokenize.tokenize(fh.readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING)
        )
    for banned in ("satisfaction", "token", "llm", "vllm", "budget_cap", "react",
                   "condition", "agent", "transcript"):
        assert banned not in code.lower(), f"selector source mentions {banned!r}"


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_a_pair_that_verifies(tmp_path):
    import csv as _csv
    pool = tmp_path / "pool"
    pool.mkdir()
    rows = make_pool()
    fields = list(rows[0])
    with (pool / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (pool / "pool_meta.json").write_text(
        json.dumps({"git_commit": "abc", "n_completed": len(rows),
                    "seed_start": 30000, "seed_end": 30119}), encoding="utf-8")

    out = tmp_path / "manifests"
    assert main(["--pool", str(pool), "--per-band", "10", "--out", str(out)]) == 0
    subsets = list(out.glob("subset__bands_budget_dev__*.json"))
    binnings = list(out.glob("binning__bands_budget_dev__*.json"))
    assert len(subsets) == 1 and len(binnings) == 1
    subset = json.loads(subsets[0].read_text(encoding="utf-8"))
    binning = json.loads(binnings[0].read_text(encoding="utf-8"))
    assert verify(binning, subset, 10, "budget_dev") == []
    assert sum(len(v) for v in subset["instances"].values()) == 30
    assert (out / "report__bands_budget_dev.md").exists()


def test_the_cli_refuses_to_overwrite_an_existing_manifest(tmp_path):
    import csv as _csv
    pool = tmp_path / "pool"
    pool.mkdir()
    rows = make_pool()
    with (pool / "candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (pool / "pool_meta.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "manifests"
    assert main(["--pool", str(pool), "--per-band", "10", "--out", str(out)]) == 0
    with pytest.raises(SystemExit):
        main(["--pool", str(pool), "--per-band", "10", "--out", str(out)])
