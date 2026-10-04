# tests/test_n4_budget_probe.py
"""Offline tests for the n = 4 budget follow-up analyser.

Written alongside the analyser while the 24 runs were still executing, so neither could have
been shaped by the outcome. What must hold whatever the numbers are: the sign convention,
the informativeness count that stops twelve runs being read as twelve observations, and the
interpretation rule staying the one the pre-registration fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness import DOCUMENT_SCHEMA_VERSION
from scripts.analyse_n4_budget_probe import (
    BASELINE_CAP,
    PROBE_CAP,
    REGISTERED_READINGS,
    AnalysisRefused,
    architecture_contrast,
    budget_change,
    build,
    load_instances,
    load_records,
    main,
    registered_reading,
    render,
    summarise,
)

MODEL = "Qwen/Qwen3-32B-AWQ"
CONDS = ("c1_react", "c3_mas")


def iid(seed):
    return f"uniform-n4-t80-o50-s{seed}"


def subset_doc(count=12):
    entries = [{
        "instance_id": iid(40000 + i), "seed": 40000 + i, "level": "easy", "optimum": 3,
        "complexity_metric": 0,
        "cell": {"n_people": 4, "tightness": 0.8, "overlap": 0.5,
                 "travel_structure": "uniform"},
    } for i in range(count)]
    return {"schema_version": "pilot_subset/1.0", "n_people": 4, "fixed_optimum": 3,
            "manifest_hash": "b" * 64, "content_hash": "c" * 64,
            "instances": {"easy": entries, "medium": [], "hard": []}}


def run_doc(condition, instance_id, cap, *, satisfaction, tokens, meetings=1, valid=True):
    rid = f"{condition}__{instance_id}__cap{cap}__{MODEL.replace('/', '-')}"
    return {
        # The real value, not a guess: the completeness audit checks it, and a fixture that
        # invents one tests the audit's rejection path instead of the analyser.
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "run_result": {
            "run_id": rid, "instance_id": instance_id, "condition": condition, "cap": cap,
            "model": MODEL, "seed": 42,
            "score": {"satisfaction": satisfaction, "valid": valid},
            "tokens": {"total": tokens}, "calls": [{}] * 7,
            "final_plan": {"meetings": [{}] * meetings},
        },
        "agent": {"termination": "agent_finish", "proposals": [{"valid": True}]},
        "pilot_sweep": {"subset_hash": "c" * 64, "binning_hash": "b" * 64,
                        "git_commit": "abc123def456"},
    }


def write_world(tmp_path, *, sat=None, tok=None, count=12):
    """A complete little follow-up: 12 instances, two caps, two conditions."""
    sub = tmp_path / "subset.json"
    doc = subset_doc(count)
    sub.write_text(json.dumps(doc), encoding="utf-8")
    d32, d64 = tmp_path / "r32", tmp_path / "r64"
    for d in (d32, d64):
        d.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(doc["instances"]["easy"]):
        for cap, run_dir in ((BASELINE_CAP, d32), (PROBE_CAP, d64)):
            for c in CONDS:
                s = sat(i, cap, c) if sat else (0.5 if c == "c1_react" else 1.0)
                t = tok(i, cap, c) if tok else (25_000 if cap == BASELINE_CAP else 40_000)
                doc_ = run_doc(c, e["instance_id"], cap, satisfaction=s, tokens=t)
                (run_dir / f"{doc_['run_result']['run_id']}.json").write_text(
                    json.dumps(doc_), encoding="utf-8")
    return sub, d32, d64


def expected_doc(d64: Path):
    return {"schema_version": "expected_runs/1.0", "subset_hash": "c" * 64,
            "binning_hash": "b" * 64, "model": MODEL,
            "runs": [{"run_id": p.stem, "condition": p.stem.split("__")[0],
                      "instance_id": p.stem.split("__")[1],
                      "cap": int(p.stem.split("__")[2].removeprefix("cap")),
                      "level": "easy"} for p in sorted(d64.glob("*.json"))]}


# --------------------------------------------------------------------------- #
# The sign convention.
# --------------------------------------------------------------------------- #
def test_the_architecture_delta_is_c1_minus_c3():
    rows = [{"instance_id": "i", "condition": "c1_react", "satisfaction": 1.0},
            {"instance_id": "i", "condition": "c3_mas", "satisfaction": 0.0}]
    assert architecture_contrast(rows)["mean_delta_satisfaction"] == 1.0


def test_a_hierarchy_advantage_is_negative():
    rows = [{"instance_id": "i", "condition": "c1_react", "satisfaction": 0.0},
            {"instance_id": "i", "condition": "c3_mas", "satisfaction": 1.0}]
    a = architecture_contrast(rows)
    assert a["mean_delta_satisfaction"] == -1.0 and a["instances_c3_ahead"] == 1


# --------------------------------------------------------------------------- #
# Informativeness: the whole point of the count.
# --------------------------------------------------------------------------- #
def rec(instance_id, sat, tokens, *, valid=True):
    return {"instance_id": instance_id, "satisfaction": sat, "tokens_total": tokens,
            "nonempty_validated": valid, "over_baseline_cap": tokens > BASELINE_CAP}


def test_a_run_that_stayed_under_the_lower_cap_is_not_informative():
    """It executed an identical configuration at both caps, so it says nothing about the
    extra budget."""
    base = [rec("i1", 0.5, 20_000)]
    probe = [rec("i1", 0.5, 20_000)]
    g = budget_change(base, probe)
    assert g["runs_with_changed_execution"] == 0
    assert g["identical_token_totals"] == 1
    assert g["runs_exceeding_baseline_cap"] == 0


def test_a_run_that_exceeded_the_lower_cap_is_informative():
    g = budget_change([rec("i1", 0.5, 31_000)], [rec("i1", 1.0, 45_000)])
    assert g["runs_exceeding_baseline_cap"] == 1
    assert g["runs_with_changed_execution"] == 1
    assert g["improved"] == 1 and g["mean_delta_satisfaction"] == 0.5


def test_a_run_that_changed_without_crossing_the_lower_cap_still_counts():
    """The C3 case: it subdivides the cap, so a worker's quota grows and the run executes
    differently while every total stays under 32000. Counting only the runs that crossed
    the lower cap would report zero informative runs for a condition that plainly changed."""
    g = budget_change([rec("i1", 0.0, 22_000)], [rec("i1", 1.0, 27_000)])
    assert g["runs_exceeding_baseline_cap"] == 0
    assert g["runs_with_changed_execution"] == 1
    assert g["identical_token_totals"] == 0


def test_the_informative_count_is_independent_of_the_mean():
    """Eight unchanged runs under the cap and four improved above it: the mean is computed
    over twelve, and the report must be able to say only four were informative."""
    base = [rec(f"i{i}", 0.0, 20_000) for i in range(8)] + \
           [rec(f"j{i}", 0.0, 31_000) for i in range(4)]
    probe = [rec(f"i{i}", 0.0, 20_000) for i in range(8)] + \
            [rec(f"j{i}", 1.0, 50_000) for i in range(4)]
    g = budget_change(base, probe)
    assert g["n_pairs"] == 12
    assert g["runs_exceeding_baseline_cap"] == 4
    assert g["runs_with_changed_execution"] == 4
    assert g["improved"] == 4 and g["unchanged"] == 8


def test_gained_and_lost_validated_plans_are_counted_separately():
    base = [rec("i1", 0.0, 31_000, valid=False), rec("i2", 1.0, 31_000, valid=True)]
    probe = [rec("i1", 1.0, 40_000, valid=True), rec("i2", 0.0, 40_000, valid=False)]
    g = budget_change(base, probe)
    assert (g["gained_a_validated_plan"], g["lost_a_validated_plan"]) == (1, 1)


# --------------------------------------------------------------------------- #
# The interpretation rule may not be reinvented.
# --------------------------------------------------------------------------- #
def test_the_two_registered_readings_are_the_only_ones():
    assert set(REGISTERED_READINGS) == {"c1_at_or_above_c3", "c1_below_c3"}


def test_equality_counts_as_the_c1_at_or_above_case():
    assert registered_reading(0.5, 0.5)["case"] == "c1_at_or_above_c3"


def test_a_hierarchy_lead_selects_the_closing_case():
    r = registered_reading(0.4, 0.7)
    assert r["case"] == "c1_below_c3"
    assert "no easier development pool is built" in r["licenses"]


def test_the_candidate_regime_is_never_called_a_crossover():
    text = REGISTERED_READINGS["c1_at_or_above_c3"]
    assert "NOT a demonstrated statistical crossover" in text


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #
def test_an_instance_at_another_optimum_is_refused(tmp_path):
    doc = subset_doc(1)
    doc["instances"]["easy"][0]["optimum"] = 4
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AnalysisRefused, match="matched at 3"):
        load_instances(p)


def test_records_from_another_cap_are_ignored(tmp_path):
    sub, d32, d64 = write_world(tmp_path)
    stray = run_doc("c1_react", iid(40000), 16000, satisfaction=1.0, tokens=15_000)
    (d32 / f"{stray['run_result']['run_id']}.json").write_text(
        json.dumps(stray), encoding="utf-8")
    rows = load_records(d32, load_instances(sub), BASELINE_CAP)
    assert {r["cap"] for r in rows} == {BASELINE_CAP} and len(rows) == 24


def test_an_incomplete_probe_arm_is_refused(tmp_path):
    sub, d32, d64 = write_world(tmp_path)
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(expected_doc(d64)), encoding="utf-8")
    next(d64.glob("*.json")).unlink()
    with pytest.raises(AnalysisRefused, match="completeness audit failed"):
        main(["--runs-32k", str(d32), "--runs-64k", str(d64), "--subset", str(sub),
              "--expected", str(exp), "--out", str(tmp_path / "o")])


# --------------------------------------------------------------------------- #
# The report's wording.
# --------------------------------------------------------------------------- #
def built(tmp_path, **kw):
    sub, d32, d64 = write_world(tmp_path, **kw)
    instances = load_instances(sub)
    return build(load_records(d32, instances, BASELINE_CAP),
                 load_records(d64, instances, PROBE_CAP), instances,
                 {"n_expected": 24, "n_verified": 24, "n_missing": 0, "n_incompatible": 0,
                  "n_unexpected_files": 0, "all_checks_passed": True})


def test_the_report_states_the_informativeness_caveat(tmp_path):
    text = render(built(tmp_path))
    assert "more calls, not longer ones" in text
    assert "not necessary" in text
    assert "execution changed" in text


def test_the_report_quotes_the_registered_reading(tmp_path):
    text = render(built(tmp_path))
    assert "fixed before these runs existed" in text
    assert "pre-registration of 2026-08-17" in text


def test_the_report_selects_no_cap_and_no_threshold(tmp_path):
    s = built(tmp_path)
    assert s["selects_a_cap"] is False and s["introduces_a_threshold"] is False
    text = render(s)
    assert "no cap is selected" in text
    claims = "\n".join(l for l in text.splitlines() if "not" not in l.lower())
    for banned in ("significant", "p =", "proves"):
        assert banned not in claims


def test_an_empty_cell_reports_zero_rather_than_dividing():
    assert summarise([])["n"] == 0


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_writes_both_artifacts(tmp_path):
    sub, d32, d64 = write_world(tmp_path)
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(expected_doc(d64)), encoding="utf-8")
    out = tmp_path / "o"
    assert main(["--runs-32k", str(d32), "--runs-64k", str(d64), "--subset", str(sub),
                 "--expected", str(exp), "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["n_instances"] == 12
    assert doc["caps"] == [BASELINE_CAP, PROBE_CAP]
    assert doc["registered_reading"]["case"] in REGISTERED_READINGS
    assert "execution changed" in (out / "report.md").read_text(encoding="utf-8")
