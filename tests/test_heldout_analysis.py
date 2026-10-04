# tests/test_heldout_analysis.py
"""Offline tests for the held-out confirmatory analyser.

The analyser executes a plan that was frozen before the data existed, so the tests
target the places where a silent drift from that plan would still produce a readable
report: the orientation of the contrasts, the size and membership of the Holm family,
the interaction statistic, the IUT's decision rule, and the refusal to analyse a set
that is not the registered design. Determinism is tested too -- a permutation test
whose p-value moves between runs would make the artifact unreproducible.

No held-out document is read here. Every fixture is synthetic.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from scripts.analyse_heldout import (
    CONFIRMATORY,
    HELD_OUT_SEED_FLOOR,
    LEVEL_TO_BAND,
    ORIENTATION_ORDER,
    PRIMARY_CAP,
    AnalysisRefused,
    build,
    check_inputs,
    confirmatory_family,
    holm,
    label_perm_unequal,
    load_instances,
    load_records,
    main,
    orient,
    paired_deltas,
    render,
)
from scripts.analyse_interaction_power import ALPHA, holm2

MODEL = "Qwen/Qwen3-32B-AWQ"
MODEL_TAG = MODEL.replace("/", "-")
CAPS = (16000, 32000, 64000, 128000)
CONDS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
SUBSET_HASH, BINNING_HASH = "s" * 64, "b" * 64
PERMS = 999
BOOT = 500          # DESCRIPTIVE interval only; the registered default is 10000

# Ten instances per band. Equal bands are what the registered label permutation for
# Delta requires, and ten is the smallest size at which the sign-flip test can reject
# after Holm across ten hypotheses at all: with n identical differences the smallest
# attainable one-sided p is 2^-n, and Holm needs it below alpha/10 = 0.005.
PER_BAND = 10
PER_SIZE = 3


# --------------------------------------------------------------------------- #
# Fixture world.
# --------------------------------------------------------------------------- #
def instance_id(n: int, seed: int, structure: str = "uniform") -> str:
    return f"{structure}-n{n}-t80-o50-s{seed}"


def subset_doc(n_people: int, base_seed: int, per_level: dict[str, int]):
    instances: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    seed = base_seed
    for level, count in per_level.items():
        for _ in range(count):
            instances[level].append({
                "instance_id": instance_id(n_people, seed),
                "seed": seed,
                "level": level,
                "optimum": 3,
                "complexity_metric": 0 if level == "easy" else 7,
                "cell": {"n_people": n_people, "tightness": 0.8, "overlap": 0.5,
                         "travel_structure": "uniform"},
            })
            seed += 1
    return {"schema_version": "pilot_subset/1.0", "n_people": n_people,
            "manifest_hash": BINNING_HASH, "content_hash": "c" * 64,
            "counts": {k: len(v) for k, v in instances.items()},
            "instances": instances}


def run_doc(condition: str, iid: str, cap: int, satisfaction: float):
    rid = f"{condition}__{iid}__cap{cap}__{MODEL_TAG}"
    return {
        "schema_version": "harness_slice/1.0",
        "run_result": {
            "run_id": rid, "instance_id": iid, "condition": condition, "cap": cap,
            "model": MODEL, "seed": 42,
            "score": {"satisfaction": satisfaction, "valid": True},
            "tokens": {"total": 1000},
            "calls": [{}] * 3,
            "final_plan": {"meetings": [{}]},
        },
        "agent": {"termination": "agent_finish", "proposals": [{"valid": True}]},
        "pilot_sweep": {"subset_hash": SUBSET_HASH, "binning_hash": BINNING_HASH,
                        "git_commit": "abc123def456"},
    }


def satisfaction_for(condition: str, band: str | None) -> float:
    """A clean crossover in Block A: C1 ahead at low, C3 ahead at high.

    Deliberately separable so the tests can assert a decision rather than a number.
    The position in ORIENTATION_ORDER is used only to spread the five synthetic
    conditions apart; it carries no meaning here and none is asserted.
    """
    rung = (ORIENTATION_ORDER.index(condition)
            / (len(ORIENTATION_ORDER) - 1))                 # 0.0 .. 1.0
    if band == "low":
        return round(1.0 - rung, 4)
    if band == "high":
        return round(rung, 4)
    return 0.5                                             # medium, and Block B


def write_world(tmp_path: Path, *, satisfaction=satisfaction_for, seed_base=None,
                per_band=PER_BAND):
    """A complete miniature of the registered design: Block A bands + Block B sizes."""
    base = HELD_OUT_SEED_FLOOR if seed_base is None else seed_base
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    subsets, pools, expected_runs = [], [], []

    block_a = subset_doc(8, base, {"easy": per_band, "medium": per_band,
                                   "hard": per_band})
    docs = [(block_a, 8)]
    for i, n in enumerate((4, 5, 6)):
        docs.append((subset_doc(n, base + 10000 * (i + 1),
                                {"easy": PER_SIZE, "medium": 0, "hard": 0}), n))

    for doc, n in docs:
        path = tmp_path / f"subset_n{n}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        subsets.append(path)

        pool = tmp_path / f"pool_n{n}.csv"
        lines = ["instance_id,seed,higher_order_gap_H"]
        for k, (level, rows) in enumerate(doc["instances"].items()):
            for j, e in enumerate(rows):
                lines.append(f"{e['instance_id']},{e['seed']},{(j + k) % 2}")
        pool.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pools.append(pool)

        for level, rows in doc["instances"].items():
            band = LEVEL_TO_BAND[level] if n == 8 else None
            for e in rows:
                for cap in CAPS:
                    for cond in CONDS:
                        d = run_doc(cond, e["instance_id"], cap,
                                    satisfaction(cond, band))
                        (runs / f"{d['run_result']['run_id']}.json").write_text(
                            json.dumps(d), encoding="utf-8")
                        expected_runs.append({
                            "run_id": d["run_result"]["run_id"], "condition": cond,
                            "instance_id": e["instance_id"], "cap": cap,
                            "level": level})

    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({
        "schema_version": "expected_runs/1.0", "subset_hash": SUBSET_HASH,
        "binning_hash": BINNING_HASH, "model": MODEL, "runs": expected_runs,
    }), encoding="utf-8")
    return {"runs": runs, "subsets": subsets, "pools": pools, "expected": expected}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return write_world(tmp_path_factory.mktemp("heldout"))


@pytest.fixture(scope="module")
def loaded(world):
    instances = load_instances(world["subsets"])
    from scripts.analyse_heldout import load_higher_order_gap
    gaps = load_higher_order_gap(world["pools"])
    rows = load_records(world["runs"], instances)
    return instances, gaps, rows


@pytest.fixture(scope="module")
def summary(loaded):
    instances, gaps, rows = loaded
    integrity = check_inputs(rows, instances, gaps)
    audit_stub = {"n_expected": len(rows), "n_verified": len(rows),
                  "all_checks_passed": True}
    return build(rows, instances, gaps, audit_stub, integrity, perms=PERMS, boot=BOOT)


# --------------------------------------------------------------------------- #
# The plan's shape: orientation, family size, membership.
# --------------------------------------------------------------------------- #
def test_the_orientation_convention_agrees_with_the_registered_five():
    """The five orientations are the plan's; the convention must not contradict them.

    Written out rather than derived, so that a change to ORIENTATION_ORDER -- which
    is bookkeeping and carries no hierarchy -- cannot silently redefine what
    amendment B2 registered.
    """
    registered = (
        ("C1->C2", "c1_react", "c2_verify_revise"),
        ("C2->C4", "c2_verify_revise", "c4_planner_critic"),
        ("C4->C3", "c4_planner_critic", "c3_mas"),
        ("C1->C5", "c1_react", "c5_best_of_3"),
        ("C5->C3", "c5_best_of_3", "c3_mas"),
    )
    assert CONFIRMATORY == registered
    for _, earlier, later in registered:
        assert orient(earlier, later) == (earlier, later)
        assert orient(later, earlier) == (earlier, later)


def test_the_confirmatory_family_is_exactly_ten_hypotheses(summary):
    conf = summary["confirmatory"]
    assert conf["n_hypotheses"] == 10
    assert len(conf["family"]) == 10
    assert len(conf["contrasts"]) == 5
    kinds = {h["hypothesis"].split("|")[1] for h in conf["family"]}
    assert kinds == {"crossover_iut", "interaction"}


def test_the_medium_band_is_in_no_holm_family(summary):
    """Amendment A2: descriptive, no confirmatory test, no Holm membership."""
    labels = [h["hypothesis"] for h in summary["confirmatory"]["family"]]
    assert not any("medium" in lab for lab in labels)
    for res in summary["confirmatory"]["contrasts"]:
        medium = res["bands"]["medium"]
        assert medium["mean_delta"] is not None      # estimated
        assert "p_value" not in medium and "holm" not in medium


def test_only_the_primary_cap_carries_confirmatory_tests(summary):
    assert summary["primary_cap"] == PRIMARY_CAP
    assert sorted(summary["secondary_caps"]) == [16000, 32000, 128000]
    for res in summary["confirmatory"]["contrasts"]:
        assert res["cap"] == PRIMARY_CAP
    for row in summary["comparison_matrix"]:
        assert "p_iut" not in row and "p_two_sided" not in row


def test_block_b_is_descriptive_and_never_a_band(loaded, summary):
    instances, _, rows = loaded
    for info in instances.values():
        assert (info["band"] is None) == (info["n_people"] != 8)
    assert all(r["block"] == "A"
               for r in rows if r["band"] is not None)
    strata = {row["stratum"] for row in summary["comparison_matrix"]
              if row["block"] == "B"}
    assert strata == {"n=4", "n=5", "n=6"}


def test_the_matrix_covers_all_ten_pairs_at_every_cap(summary):
    contrasts = {row["contrast"] for row in summary["comparison_matrix"]}
    assert len(contrasts) == 10
    caps = {row["cap"] for row in summary["comparison_matrix"]}
    assert caps == set(CAPS)
    flagged = {row["contrast"] for row in summary["comparison_matrix"]
               if row["in_confirmatory_family"]}
    assert flagged == {label for label, *_ in CONFIRMATORY}


# --------------------------------------------------------------------------- #
# Holm.
# --------------------------------------------------------------------------- #
def test_holm_agrees_with_the_existing_two_hypothesis_implementation():
    """`holm2` is the repository's existing rule; the general form must not diverge."""
    grid = [0.001, 0.01, 0.02, 0.024, 0.026, 0.05, 0.2, 0.9]
    for p_a in grid:
        for p_b in grid:
            mine = holm([p_a, p_b], ALPHA)
            ref_a, ref_b = holm2(np.array([p_a]), np.array([p_b]), ALPHA)
            assert mine[0]["reject"] == bool(ref_a[0])
            assert mine[1]["reject"] == bool(ref_b[0])


def test_holm_is_step_down_over_ten_hypotheses():
    p = [0.001, 0.004, 0.03, 0.04, 0.05, 0.06, 0.5, 0.6, 0.7, 0.8]
    out = holm(p, ALPHA)
    assert out[0]["p_holm"] == pytest.approx(0.01)      # 10 * 0.001
    assert out[1]["p_holm"] == pytest.approx(0.036)     # 9 * 0.004
    assert out[0]["reject"] and out[1]["reject"]
    assert not out[2]["reject"]                         # 8 * 0.03 = 0.24


def test_holm_rejects_nothing_once_the_smallest_p_fails():
    out = holm([0.03, 0.031, 0.04], ALPHA)
    assert [o["reject"] for o in out] == [False, False, False]


def test_holm_adjusted_values_never_decrease_with_rank():
    p = [0.2, 0.001, 0.05, 0.9, 0.004]
    out = holm(p, ALPHA)
    ordered = sorted(zip(p, [o["p_holm"] for o in out]))
    adjusted = [a for _, a in ordered]
    assert adjusted == sorted(adjusted)
    assert all(a <= 1.0 for a in adjusted)


# --------------------------------------------------------------------------- #
# The interaction statistic and the IUT.
# --------------------------------------------------------------------------- #
def test_the_interaction_statistic_is_high_minus_low(summary):
    for res in summary["confirmatory"]["contrasts"]:
        expected = (res["bands"]["high"]["mean_delta"]
                    - res["bands"]["low"]["mean_delta"])
        assert res["interaction"]["delta_statistic"] == pytest.approx(expected,
                                                                     abs=1e-4)


def test_the_delta_is_later_minus_earlier(loaded):
    _, _, rows = loaded
    pairs = paired_deltas([r for r in rows if r["block"] == "A"],
                          "c1_react", "c3_mas", cap=PRIMARY_CAP,
                          keep=lambda r: r["band"] == "high")
    # At the high band the fixture puts C3 at 1.0 and C1 at 0.0.
    assert pairs and all(d == pytest.approx(1.0) for _, d in pairs)
    low = paired_deltas([r for r in rows if r["block"] == "A"],
                        "c1_react", "c3_mas", cap=PRIMARY_CAP,
                        keep=lambda r: r["band"] == "low")
    assert low and all(d == pytest.approx(-1.0) for _, d in low)


def test_the_iut_is_the_maximum_of_its_two_components(summary):
    for res in summary["confirmatory"]["contrasts"]:
        cross = res["crossover"]
        assert cross["p_iut"] == pytest.approx(
            max(cross["p_low_one_sided"], cross["p_high_one_sided"]), abs=1e-9)


def test_a_clean_crossover_is_established(summary):
    """The fixture is built with C1 ahead at low and C3 ahead at high."""
    conf = summary["confirmatory"]
    assert conf["any_crossover_established"]
    for res in conf["contrasts"]:
        assert res["bands"]["low"]["mean_delta"] < 0
        assert res["bands"]["high"]["mean_delta"] > 0
        assert res["crossover"]["components_directionally_consistent"]
        assert res["crossover"]["established"]


def test_no_crossover_when_both_bands_favour_the_same_side(tmp_path):
    """The pattern the development data showed: the hierarchy ahead everywhere."""
    def always_hierarchy(condition, band):
        return round(ORIENTATION_ORDER.index(condition)
                     / (len(ORIENTATION_ORDER) - 1), 4)

    w = write_world(tmp_path, satisfaction=always_hierarchy)
    instances = load_instances(w["subsets"])
    from scripts.analyse_heldout import load_higher_order_gap
    gaps = load_higher_order_gap(w["pools"])
    rows = load_records(w["runs"], instances)
    conf = confirmatory_family(rows, perms=PERMS, boot=BOOT,
                               rng=np.random.default_rng(20260827))
    assert not conf["any_crossover_established"]
    for res in conf["contrasts"]:
        assert res["bands"]["low"]["mean_delta"] > 0        # wrong side for the IUT
        assert not res["crossover"]["components_directionally_consistent"]
        assert not res["crossover"]["established"]
    assert check_inputs(rows, instances, gaps)["n_instances"] > 0


def test_an_interaction_can_be_significant_without_a_crossover(tmp_path):
    """Plan 4.3: an interaction is not accepted as evidence of a crossover."""
    def one_sided_but_growing(condition, band):
        rung = (ORIENTATION_ORDER.index(condition)
                / (len(ORIENTATION_ORDER) - 1))
        return round((0.1 if band == "low" else 0.9) * rung, 4)

    w = write_world(tmp_path, satisfaction=one_sided_but_growing)
    instances = load_instances(w["subsets"])
    rows = load_records(w["runs"], instances)
    conf = confirmatory_family(rows, perms=PERMS, boot=BOOT,
                               rng=np.random.default_rng(20260827))
    for res in conf["contrasts"]:
        assert res["interaction"]["delta_statistic"] > 0     # the effect grows with D
        assert res["bands"]["low"]["mean_delta"] >= 0        # but never changes sign
        assert not res["crossover"]["established"]
    assert not conf["any_crossover_established"]


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #
def test_permutation_p_values_are_deterministic(loaded):
    """Same inputs, same seed, same artifact -- twice."""
    instances, gaps, rows = loaded
    integrity = check_inputs(rows, instances, gaps)
    stub = {"n_expected": len(rows), "n_verified": len(rows),
            "all_checks_passed": True}
    first = build(rows, instances, gaps, stub, integrity, perms=PERMS, boot=BOOT)
    second = build(rows, instances, gaps, stub, integrity, perms=PERMS, boot=BOOT)
    for key in ("confirmatory", "h_sensitivity", "comparison_matrix"):
        assert json.dumps(first[key], sort_keys=True) == \
               json.dumps(second[key], sort_keys=True)


def test_the_label_permutation_is_deterministic_for_a_fixed_seed():
    low, high = [0.0, 0.0, 1.0], [1.0, 1.0]
    a = label_perm_unequal(low, high, 500, np.random.default_rng(7))
    b = label_perm_unequal(low, high, 500, np.random.default_rng(7))
    assert a == b


def test_the_unequal_label_permutation_matches_exhaustive_enumeration():
    """The generalisation used only by the H strata must be the same construction."""
    low, high = [0.0, 0.0], [1.0, 1.0, 1.0]
    pooled = low + high
    obs = abs(np.mean(high) - np.mean(low))
    hits = 0
    total = 0
    for idx in combinations(range(len(pooled)), len(low)):
        lo = [pooled[i] for i in idx]
        hi = [pooled[i] for i in range(len(pooled)) if i not in idx]
        total += 1
        if abs(np.mean(hi) - np.mean(lo)) >= obs - 1e-12:
            hits += 1
    exact = hits / total
    estimate = label_perm_unequal(low, high, 20000, np.random.default_rng(11))
    assert estimate == pytest.approx(exact, abs=0.02)


# --------------------------------------------------------------------------- #
# The gate.
# --------------------------------------------------------------------------- #
def test_a_development_seed_is_refused(tmp_path):
    w = write_world(tmp_path, seed_base=30000)
    instances = load_instances(w["subsets"])
    from scripts.analyse_heldout import load_higher_order_gap
    gaps = load_higher_order_gap(w["pools"])
    rows = load_records(w["runs"], instances)
    with pytest.raises(AnalysisRefused, match="below the held-out floor"):
        check_inputs(rows, instances, gaps)


def test_a_missing_H_pool_is_an_error_not_an_omission(loaded):
    instances, _, rows = loaded
    with pytest.raises(AnalysisRefused, match="higher_order_gap_H"):
        check_inputs(rows, instances, {})


def test_an_incomplete_matrix_is_refused(loaded):
    instances, gaps, rows = loaded
    with pytest.raises(AnalysisRefused, match="expected"):
        check_inputs(rows[:-1], instances, gaps)


def test_a_duplicated_cell_is_refused(loaded):
    instances, gaps, rows = loaded
    with pytest.raises(AnalysisRefused, match="more than once"):
        check_inputs(list(rows) + [rows[0]], instances, gaps)


def test_two_models_are_refused(loaded):
    instances, gaps, rows = loaded
    mixed = [dict(r) for r in rows]
    mixed[0]["model"] = "another/model"
    with pytest.raises(AnalysisRefused, match="more than one model"):
        check_inputs(mixed, instances, gaps)


def test_unequal_bands_are_refused_rather_than_analysed(tmp_path):
    """The registered Delta permutation is exact for two equally sized bands."""
    w = write_world(tmp_path)
    instances = load_instances(w["subsets"])
    rows = load_records(w["runs"], instances)
    victim = next(r["instance_id"] for r in rows if r["band"] == "high")
    trimmed = [r for r in rows if r["instance_id"] != victim]
    with pytest.raises(AnalysisRefused, match="bands are"):
        confirmatory_family(trimmed, perms=PERMS, boot=BOOT,
                            rng=np.random.default_rng(20260827))


def test_the_cli_refuses_an_incomplete_sweep(tmp_path):
    w = write_world(tmp_path)
    next(w["runs"].glob("*.json")).unlink()
    args = ["--runs", str(w["runs"]), "--expected", str(w["expected"]),
            "--out", str(tmp_path / "out"), "--perms", str(PERMS),
            "--bootstrap", str(BOOT)]
    for s in w["subsets"]:
        args += ["--subset", str(s)]
    for p in w["pools"]:
        args += ["--pool", str(p)]
    with pytest.raises(AnalysisRefused, match="completeness audit failed"):
        main(args)


# --------------------------------------------------------------------------- #
# Outputs.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_the_three_documents(tmp_path):
    w = write_world(tmp_path)
    out = tmp_path / "out"
    args = ["--runs", str(w["runs"]), "--expected", str(w["expected"]),
            "--out", str(out), "--perms", str(PERMS), "--bootstrap", str(BOOT)]
    for s in w["subsets"]:
        args += ["--subset", str(s)]
    for p in w["pools"]:
        args += ["--pool", str(p)]
    assert main(args) == 0
    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    assert (out / "contrasts.csv").exists()
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["confirmatory"]["n_hypotheses"] == 10
    header = (out / "contrasts.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "in_confirmatory_family" in header and "mean_delta" in header


def test_the_report_carries_the_pre_declared_standing(summary):
    text = render(summary)
    assert "confirmatory" in text.lower()
    assert "Holm across all ten" in text or "one Holm correction" in text
    assert "SENSITIVITY ONLY" in text or "sensitivity" in text.lower()
    assert "descriptive" in text.lower()


def test_a_null_crossover_report_states_its_limitation(tmp_path):
    def always_hierarchy(condition, band):
        return round(ORIENTATION_ORDER.index(condition)
                     / (len(ORIENTATION_ORDER) - 1), 4)

    w = write_world(tmp_path, satisfaction=always_hierarchy)
    instances = load_instances(w["subsets"])
    from scripts.analyse_heldout import load_higher_order_gap
    gaps = load_higher_order_gap(w["pools"])
    rows = load_records(w["runs"], instances)
    integrity = check_inputs(rows, instances, gaps)
    stub = {"n_expected": len(rows), "n_verified": len(rows),
            "all_checks_passed": True}
    text = render(build(rows, instances, gaps, stub, integrity, perms=PERMS,
                        boot=BOOT))
    assert ("Failure to establish crossover statistically is not evidence that no "
            "crossover exists.") in text


def test_the_h_sensitivity_covers_both_strata(summary):
    strata = summary["h_sensitivity"]["strata"]
    assert set(strata) == {"H_eq_0", "H_ge_1"}
    for entries in strata.values():
        assert {e["contrast"] for e in entries} == {label for label, *_ in CONFIRMATORY}
    assert "cannot promote or demote" in summary["h_sensitivity"]["standing"]
