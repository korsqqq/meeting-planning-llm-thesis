# tests/test_budget_dev_analysis.py
"""Offline tests for the budget-calibration analyser, on synthetic run documents.

The analyser exists to apply one registered rule, so the tests concentrate on the ways it
could apply a different one without looking wrong: pooling the bands instead of requiring
each, taking a qualifying rung other than the smallest, inventing a cap when none
qualifies, or letting satisfaction leak into a decision that is supposed to read one binary.
"""

from __future__ import annotations

import json

import pytest

from scripts.analyse_budget_dev import (
    BANDS,
    EXPECTED_BINNING_HASH,
    EXPECTED_SUBSET_HASH,
    FROZEN_LADDER,
    N_INSTANCES_PER_BAND,
    SUCCESS_THRESHOLD,
    AuditError,
    audit,
    load_runs,
    main,
    select_caps,
    success_rates,
)

LEVEL_OF_BAND = {"low": "easy", "medium": "medium", "high": "hard"}


def doc_for(instance_id, band, cap, *, meetings, feasible=True, satisfaction=0.0,
            seed=30000, subset=EXPECTED_SUBSET_HASH, binning=EXPECTED_BINNING_HASH,
            condition="c1_react", model="Qwen/Qwen3-32B-AWQ"):
    plan = {"meetings": [{"person": f"p{i}"} for i in range(meetings)]}
    return {
        "run_result": {
            "run_id": f"{condition}-{instance_id}-{cap}",
            "condition": condition, "instance_id": instance_id, "cap": cap,
            "level": LEVEL_OF_BAND[band], "model": model, "seed": seed,
            "score": {"solver_optimum": 3, "n_valid_meetings": meetings,
                      "satisfaction": satisfaction, "valid": feasible},
            "tokens": {"total": cap // 2, "budget_exhausted": False},
            "calls": [], "final_plan": plan,
        },
        "agent": {"termination": "agent_finish", "n_steps": 4, "proposals": []},
        "pilot_sweep": {"subset_hash": subset, "binning_hash": binning},
        "instance": {"complexity_metric": 7, "generator_params":
                     {"tightness": 0.9, "travel_structure": "uniform"}},
    }


def write_set(tmp_path, success):
    """A full 240-run set. `success(band, cap, i)` decides whether run i qualifies."""
    d = tmp_path / "runs"
    d.mkdir(parents=True)
    seed = 30000
    for band in BANDS:
        for i in range(N_INSTANCES_PER_BAND):
            iid = f"uniform-n8-t90-o50-s{seed}"
            for cap in FROZEN_LADDER:
                ok = success(band, cap, i)
                doc = doc_for(iid, band, cap, meetings=2 if ok else 0,
                              satisfaction=0.67 if ok else 0.0, seed=seed)
                (d / f"{doc['run_result']['run_id']}.json").write_text(
                    json.dumps(doc), encoding="utf-8")
            seed += 1
    return d


# --------------------------------------------------------------------------- #
# The success definition.
# --------------------------------------------------------------------------- #
def test_an_empty_plan_never_counts_however_good_the_score(tmp_path):
    """Both halves of the registered definition are applied, not just the convenient one."""
    d = tmp_path / "r"
    d.mkdir()
    doc = doc_for("i", "low", 8000, meetings=0, feasible=True, satisfaction=1.0)
    (d / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    assert load_runs(d)[0]["counts_for_cap_rule"] is False


def test_an_infeasible_plan_never_counts_however_many_meetings(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    doc = doc_for("i", "low", 8000, meetings=5, feasible=False, satisfaction=0.9)
    (d / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    assert load_runs(d)[0]["counts_for_cap_rule"] is False


def test_one_meeting_is_enough(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    doc = doc_for("i", "low", 8000, meetings=1, feasible=True, satisfaction=0.0)
    (d / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    assert load_runs(d)[0]["counts_for_cap_rule"] is True


# --------------------------------------------------------------------------- #
# The rule reads bands separately.
# --------------------------------------------------------------------------- #
def test_a_pooled_majority_does_not_qualify_when_one_band_is_on_the_floor(tmp_path):
    """The defect the 'separately' clause exists to prevent: 100/100/0 pools to 67%."""
    def success(band, cap, i):
        return band != "high"
    rows = load_runs(write_set(tmp_path, success))
    rates = success_rates(rows)
    pooled = sum(r["counts_for_cap_rule"] for r in rows if r["cap"] == 32000) / 60
    assert pooled > SUCCESS_THRESHOLD
    assert rates[32000]["all_bands_meet_threshold"] is False
    assert select_caps(rates)["verdict"] == "NO_CAP"


def test_exactly_half_passes_and_one_run_short_does_not(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: i < 10))
    assert success_rates(rows)[8000]["per_band"]["low"]["meets_threshold"] is True
    rows = load_runs(write_set(tmp_path / "b", lambda b, c, i: i < 9))
    assert success_rates(rows)[8000]["per_band"]["low"]["meets_threshold"] is False


def test_the_worst_band_is_reported(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: b != "high" or i < 2))
    assert success_rates(rows)[64000]["worst_band"] == "high"


# --------------------------------------------------------------------------- #
# Cap selection.
# --------------------------------------------------------------------------- #
def test_the_smallest_qualifying_rung_wins_not_the_best_looking_one(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: c >= 16000 and i < 15))
    d = select_caps(success_rates(rows))
    assert d["primary_cap"] == 16000
    assert d["qualifying_caps"] == [16000, 32000, 64000]


def test_the_secondary_is_the_preceding_rung(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: c >= 32000))
    d = select_caps(success_rates(rows))
    assert (d["primary_cap"], d["secondary_cap"]) == (32000, 16000)


def test_the_secondary_goes_up_when_the_primary_is_the_lowest_rung(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: True))
    d = select_caps(success_rates(rows))
    assert (d["primary_cap"], d["secondary_cap"]) == (8000, 16000)


def test_no_qualifying_rung_yields_no_cap_and_never_the_top_one(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: i < 5))
    d = select_caps(success_rates(rows))
    assert d["verdict"] == "NO_CAP"
    assert d["primary_cap"] is None and d["secondary_cap"] is None
    assert "does not start" in d["consequence"]
    assert str(max(FROZEN_LADDER)) not in json.dumps(d)


def test_a_rung_qualifying_only_above_the_primary_does_not_change_the_choice(tmp_path):
    rows = load_runs(write_set(tmp_path, lambda b, c, i: c != 32000 and i < 18))
    d = select_caps(success_rates(rows))
    assert d["primary_cap"] == 8000


# --------------------------------------------------------------------------- #
# The audit.
# --------------------------------------------------------------------------- #
def test_a_complete_set_passes(tmp_path):
    report = audit(load_runs(write_set(tmp_path, lambda b, c, i: True)))
    assert report["all_checks_passed"] is True
    assert report["n_runs"] == 240
    assert report["instances_per_band"] == {b: N_INSTANCES_PER_BAND for b in BANDS}


def test_a_missing_run_fails_the_audit(tmp_path):
    d = write_set(tmp_path, lambda b, c, i: True)
    next(iter(sorted(d.glob("*.json")))).unlink()
    with pytest.raises(AuditError, match="expected 240 runs"):
        audit(load_runs(d))


def test_a_foreign_subset_fails_the_audit(tmp_path):
    d = write_set(tmp_path, lambda b, c, i: True)
    p = sorted(d.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["pilot_sweep"]["subset_hash"] = "0" * 64
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AuditError, match="subset hash"):
        audit(load_runs(d))


def test_a_second_condition_fails_the_audit(tmp_path):
    d = write_set(tmp_path, lambda b, c, i: True)
    p = sorted(d.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["condition"] = "c3_mas"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AuditError, match="conditions beyond"):
        audit(load_runs(d))


def test_a_seed_outside_development_fails_the_audit(tmp_path):
    d = write_set(tmp_path, lambda b, c, i: True)
    p = sorted(d.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["seed"] = 100001
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AuditError, match="development seed range"):
        audit(load_runs(d))


def test_a_duplicated_cell_fails_the_audit(tmp_path):
    d = write_set(tmp_path, lambda b, c, i: True)
    p = sorted(d.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["run_id"] += "-copy"
    (d / "copy.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AuditError):
        audit(load_runs(d))


# --------------------------------------------------------------------------- #
# The decision reads one binary and nothing else.
# --------------------------------------------------------------------------- #
def test_satisfaction_cannot_move_the_decision(tmp_path):
    """Same qualifying pattern, satisfaction inverted: the verdict must be identical."""
    a = load_runs(write_set(tmp_path / "a", lambda b, c, i: c >= 32000))
    d_a = select_caps(success_rates(a))
    for r in a:
        r["satisfaction"] = 1.0 - r["satisfaction"]
    assert select_caps(success_rates(a)) == d_a


def test_the_exit_code_distinguishes_a_selected_cap_from_none(tmp_path):
    out = tmp_path / "o"
    d = write_set(tmp_path, lambda b, c, i: c >= 16000)
    assert main(["--runs", str(d), "--out", str(out)]) == 0
    d2 = write_set(tmp_path / "x", lambda b, c, i: False)
    assert main(["--runs", str(d2), "--out", str(out / "2")]) == 2


def test_the_written_documents_carry_the_rule_and_the_audit(tmp_path):
    out = tmp_path / "o"
    d = write_set(tmp_path, lambda b, c, i: c >= 16000)
    main(["--runs", str(d), "--out", str(out)])
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["registered_rule"]["threshold"] == SUCCESS_THRESHOLD
    assert doc["registered_rule"]["per_band"] == "each band separately, never pooled"
    assert doc["audit"]["all_checks_passed"] is True
    assert doc["decision"]["primary_cap"] == 16000
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Primary cap" in report and "16000" in report


def test_the_report_states_the_consequence_when_no_cap_qualifies(tmp_path):
    out = tmp_path / "o"
    d = write_set(tmp_path, lambda b, c, i: False)
    main(["--runs", str(d), "--out", str(out)])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "NO CAP SELECTED" in report
    assert "does not start" in report
