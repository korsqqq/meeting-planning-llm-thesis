# tests/test_probe_subset.py
"""Offline tests for the frozen 12-instance budget-probe subset.

The probe rests on one property: the twelve instances were chosen without looking at how C1
or C3 already did on them. If that fails, the 64 000 arm of a paired comparison stops being
a measured baseline and becomes a chosen one. Most of the tests below defend that, and the
rest pin the invariants the amendment states.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.build_pilot_manifest import content_hash
from scripts.build_probe_subset import (
    LEVELS,
    OPTIMA,
    PER_OPTIMUM,
    ProbeSelectionError,
    audit,
    build,
    main,
    select,
)


def entry(iid, level, optimum, *, seed, k=7, tightness=0.9, structure="uniform"):
    return {
        "instance_id": iid, "seed": seed, "level": level, "optimum": optimum,
        "complexity_metric": k,
        "cell": {"n_people": 8, "tightness": tightness, "overlap": 0.5,
                 "travel_structure": structure},
    }


def make_parent(per_level=20):
    """A parent shaped like the real one: interleaved optima, ascending seeds."""
    payload = {
        "schema_version": "pilot_subset/1.0",
        "manifest_hash": "b" * 64,
        "manifest_schema_version": "pilot_manifest/1.0",
        "grid_name": "bands_budget_dev",
        "seed_range_name": "budget_dev",
        "instances": {},
    }
    seed = 30000
    ks = {"easy": 0, "medium": 7, "hard": 14}
    for lv in LEVELS:
        rows = []
        for i in range(per_level):
            rows.append(entry(f"uniform-n8-t90-o50-s{seed}", lv, 3 if i % 2 else 4,
                              seed=seed, k=ks[lv],
                              structure=("uniform", "clustered", "line", "random")[i % 4]))
            seed += 1
        payload["instances"][lv] = rows
    payload["content_hash"] = content_hash(payload)
    return payload


# --------------------------------------------------------------------------- #
# The rule.
# --------------------------------------------------------------------------- #
def test_four_per_band_split_two_and_two_on_the_optimum():
    chosen = select(make_parent())
    assert set(chosen) == set(LEVELS)
    for lv in LEVELS:
        assert len(chosen[lv]) == len(OPTIMA) * PER_OPTIMUM == 4
        assert Counter(e["optimum"] for e in chosen[lv]) == {3: 2, 4: 2}


def test_the_rule_takes_the_first_matches_in_parent_order():
    parent = make_parent()
    chosen = select(parent)
    for lv in LEVELS:
        rows = parent["instances"][lv]
        for optimum in OPTIMA:
            expected = [e["instance_id"] for e in rows
                        if e["optimum"] == optimum][:PER_OPTIMUM]
            got = [e["instance_id"] for e in chosen[lv] if e["optimum"] == optimum]
            assert got == expected


def test_the_selection_is_reproducible():
    a, b = select(make_parent()), select(make_parent())
    assert {lv: [e["instance_id"] for e in a[lv]] for lv in LEVELS} == \
           {lv: [e["instance_id"] for e in b[lv]] for lv in LEVELS}


def test_a_parent_that_cannot_supply_the_split_is_refused_not_relaxed():
    parent = make_parent()
    for e in parent["instances"]["hard"]:
        e["optimum"] = 3                       # no O = 4 left in that band
    parent["content_hash"] = content_hash(parent)
    with pytest.raises(ProbeSelectionError, match="need 2"):
        select(parent)


# --------------------------------------------------------------------------- #
# Selection, never construction.
# --------------------------------------------------------------------------- #
def test_every_entry_is_copied_verbatim_from_the_parent():
    parent = make_parent()
    subset = build(parent, 128000)
    by_id = {e["instance_id"]: e for lv in LEVELS for e in parent["instances"][lv]}
    for lv in LEVELS:
        for e in subset["instances"][lv]:
            assert e == by_id[e["instance_id"]]


def test_a_modified_entry_fails_the_audit():
    parent = make_parent()
    subset = build(parent, 128000)
    subset["instances"]["easy"][0] = dict(subset["instances"]["easy"][0], optimum=9)
    with pytest.raises(ProbeSelectionError, match="differs from the parent entry"):
        audit(subset, parent)


def test_the_subset_points_at_the_same_binning_manifest():
    parent = make_parent()
    subset = build(parent, 128000)
    assert subset["manifest_hash"] == parent["manifest_hash"]
    assert subset["parent_subset_hash"] == parent["content_hash"]


# --------------------------------------------------------------------------- #
# The audit.
# --------------------------------------------------------------------------- #
def test_a_complete_subset_passes():
    parent = make_parent()
    report = audit(build(parent, 128000), parent)
    assert report["all_checks_passed"] is True
    assert report["n_instances"] == 12
    assert report["per_band"] == {"low": 4, "medium": 4, "high": 4}
    assert report["distinct_seeds"] == 12
    assert report["held_out_seeds_present"] is False


def test_a_held_out_seed_fails_the_audit():
    parent = make_parent()
    subset = build(parent, 128000)
    subset["instances"]["easy"][0] = dict(subset["instances"]["easy"][0], seed=100001)
    with pytest.raises(ProbeSelectionError, match="development range|held-out"):
        audit(subset, parent)


def test_an_unbalanced_optimum_histogram_fails_the_audit():
    parent = make_parent()
    subset = build(parent, 128000)
    subset["instances"]["medium"] = subset["instances"]["medium"][:1] * 4
    with pytest.raises(ProbeSelectionError):
        audit(subset, parent)


def test_the_content_hash_reproduces():
    parent = make_parent()
    subset = build(parent, 128000)
    assert content_hash(subset) == subset["content_hash"]


# --------------------------------------------------------------------------- #
# No agent outcome may reach the selection.
# --------------------------------------------------------------------------- #
def test_the_selector_never_reads_an_agent_outcome():
    """The property the probe rests on: the twelve were chosen without seeing how C1 or
    C3 did on them."""
    import tokenize
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_probe_subset.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))
    for banned in ("satisfaction", "feasib", "termination", "tokens_total", "nonempty",
                   "score", "run_result", "c1_react", "c3_mas"):
        assert banned not in code.lower(), f"probe selector reads {banned!r}"


def test_the_rule_ignores_everything_except_position_and_optimum():
    """Permuting the fields a selection may not read leaves the choice unchanged."""
    parent = make_parent()
    before = {lv: [e["instance_id"] for e in select(parent)[lv]] for lv in LEVELS}
    for lv in LEVELS:
        for i, e in enumerate(parent["instances"][lv]):
            e["cell"] = dict(e["cell"], tightness=[0.6, 0.8, 0.9, 1.0][i % 4])
            e["complexity_metric"] = 99 - i
    after = {lv: [e["instance_id"] for e in select(parent)[lv]] for lv in LEVELS}
    assert before == after


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_a_subset_that_audits(tmp_path):
    parent_path = tmp_path / "parent.json"
    parent = make_parent()
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    out = tmp_path / "manifests"
    assert main(["--parent", str(parent_path), "--out", str(out)]) == 0
    written = list(out.glob("subset__budget_probe_128k__*.json"))
    assert len(written) == 1
    subset = json.loads(written[0].read_text(encoding="utf-8"))
    assert audit(subset, parent)["all_checks_passed"] is True
    assert subset["probe_cap"] == 128000


def test_the_cli_refuses_a_parent_whose_hash_does_not_reproduce(tmp_path):
    parent = make_parent()
    # Edited after hashing. The field must really change: entry 0 already has
    # optimum 4 in this fixture, so setting it to 4 would leave the hash intact and the
    # test would pass while checking nothing.
    parent["instances"]["easy"][0]["optimum"] = 9
    p = tmp_path / "parent.json"
    p.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(SystemExit, match="has been modified"):
        main(["--parent", str(p), "--out", str(tmp_path / "m")])


def test_the_cli_refuses_to_overwrite(tmp_path):
    p = tmp_path / "parent.json"
    p.write_text(json.dumps(make_parent()), encoding="utf-8")
    out = tmp_path / "m"
    assert main(["--parent", str(p), "--out", str(out)]) == 0
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--parent", str(p), "--out", str(out)])


def test_the_real_probe_subset_matches_the_amendment():
    """The twelve ids recorded in THESIS_DECISIONS are the twelve in the manifest."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root / "results/manifests/subset__budget_probe_128k__76f3bc0f0654.json"
    if not path.exists():           # the manifest travels with the repo; skip if cut out
        pytest.skip("probe subset manifest not present")
    subset = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "easy": ["random-n8-t80-o100-s30000", "random-n8-t80-o80-s30002",
                 "clustered-n8-t60-o100-s30001", "clustered-n8-t60-o100-s30004"],
        "medium": ["line-n8-t90-o80-s30039", "line-n8-t80-o100-s30040",
                   "uniform-n8-t90-o50-s30036", "clustered-n8-t90-o20-s30041"],
        "hard": ["random-n8-t90-o80-s30015", "clustered-n8-t90-o80-s30017",
                 "uniform-n8-t100-o50-s30024", "uniform-n8-t90-o50-s30027"],
    }
    got = {lv: [e["instance_id"] for e in subset["instances"][lv]] for lv in LEVELS}
    assert got == expected
    assert content_hash(subset) == subset["content_hash"]
    assert subset["content_hash"].startswith("76f3bc0f0654")
