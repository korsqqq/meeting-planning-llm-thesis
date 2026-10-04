# tests/test_lowcx_calibration.py
"""Offline tests for the lower-complexity calibration analyser.

Three things must hold no matter what the numbers turn out to be: the sign of the contrast
follows the exposé, an incomplete sweep is refused rather than summarised, and the report
never promotes the validated non-empty rate to a performance claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyse_lowcx_calibration import (
    AnalysisRefused,
    build,
    load_instances,
    load_records,
    main,
    paired,
    render,
    summarise,
    worker_quota,
)

MODEL = "Qwen/Qwen3-32B-AWQ"
CAPS = (8000, 16000, 32000)
CONDS = ("c1_react", "c3_mas")


def instance_id(n, seed, structure="uniform"):
    return f"{structure}-n{n}-t80-o50-s{seed}"


def subset_doc(n, base_seed, count=2):
    entries = [{
        "instance_id": instance_id(n, base_seed + i), "seed": base_seed + i,
        "level": "easy", "optimum": 3, "complexity_metric": i,
        "cell": {"n_people": n, "tightness": 0.8, "overlap": 0.5,
                 "travel_structure": "uniform"},
    } for i in range(count)]
    return {"schema_version": "pilot_subset/1.0", "n_people": n, "fixed_optimum": 3,
            "manifest_hash": "b" * 64, "content_hash": "c" * 64,
            "instances": {"easy": entries, "medium": [], "hard": []}}


def run_doc(condition, iid, cap, *, satisfaction, meetings=1, tokens=1000, valid=True):
    rid = f"{condition}__{iid}__cap{cap}__{MODEL.replace('/', '-')}"
    return {
        "schema_version": "run/1.0",
        "run_result": {
            "run_id": rid, "instance_id": iid, "condition": condition, "cap": cap,
            "model": MODEL, "seed": 42,
            "score": {"satisfaction": satisfaction, "valid": valid},
            "tokens": {"total": tokens},
            "calls": [{}] * 5,
            "final_plan": {"meetings": [{}] * meetings},
        },
        "agent": {"termination": "agent_finish", "proposals": [{"valid": True}]},
        "pilot_sweep": {"subset_hash": "s" * 64, "binning_hash": "b" * 64,
                        "git_commit": "abc123def456"},
    }


def write_world(tmp_path, *, satisfaction=None, sizes=(4, 5, 6)):
    """A complete little sweep: 2 instances per size, 3 caps, 2 conditions."""
    subs, runs = [], tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        doc = subset_doc(n, n * 10000)
        p = tmp_path / f"subset_n{n}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        subs.append(p)
        for e in doc["instances"]["easy"]:
            for cap in CAPS:
                for c in CONDS:
                    sat = (satisfaction(n, cap, c) if satisfaction
                           else (1.0 if c == "c1_react" else 0.5))
                    d = run_doc(c, e["instance_id"], cap, satisfaction=sat)
                    (runs / f"{d['run_result']['run_id']}.json").write_text(
                        json.dumps(d), encoding="utf-8")
    return subs, runs


# --------------------------------------------------------------------------- #
# The sign convention.
# --------------------------------------------------------------------------- #
def test_the_delta_is_c1_minus_c3():
    rows = [
        {"instance_id": "i1", "cap": 8000, "condition": "c1_react",
         "satisfaction": 1.0, "tokens_total": 100},
        {"instance_id": "i1", "cap": 8000, "condition": "c3_mas",
         "satisfaction": 0.0, "tokens_total": 200},
    ]
    p = paired(rows)
    assert p["mean_delta_satisfaction"] == 1.0
    assert p["instances_c1_ahead"] == 1 and p["instances_c3_ahead"] == 0


def test_a_hierarchy_advantage_is_negative():
    """The exposé's sign: negative means the hierarchy is ahead."""
    rows = [
        {"instance_id": "i1", "cap": 8000, "condition": "c1_react",
         "satisfaction": 0.0, "tokens_total": 100},
        {"instance_id": "i1", "cap": 8000, "condition": "c3_mas",
         "satisfaction": 1.0, "tokens_total": 100},
    ]
    p = paired(rows)
    assert p["mean_delta_satisfaction"] == -1.0
    assert p["instances_c3_ahead"] == 1


def test_pooling_over_caps_pairs_within_cap_not_across_it():
    """The bug this guards: keying on instance_id alone lets the same instance at three
    caps collapse to one record, so a pooled contrast silently reports one cap."""
    rows = []
    for cap, c1, c3 in ((8000, 0.0, 0.0), (16000, 0.0, 1.0), (32000, 1.0, 1.0)):
        rows.append({"instance_id": "i1", "cap": cap, "condition": "c1_react",
                     "satisfaction": c1, "tokens_total": cap})
        rows.append({"instance_id": "i1", "cap": cap, "condition": "c3_mas",
                     "satisfaction": c3, "tokens_total": cap})
    p = paired(rows)
    assert p["n_pairs"] == 3, "one instance at three caps is three paired observations"
    # The function rounds to four places, so compare on that scale rather than exactly.
    assert p["mean_delta_satisfaction"] == pytest.approx(-1 / 3, abs=1e-4)
    assert (p["instances_c3_ahead"], p["instances_tied"]) == (1, 2)


def test_an_unpaired_instance_is_dropped_from_the_contrast():
    rows = [{"instance_id": "i1", "cap": 8000, "condition": "c1_react",
             "satisfaction": 1.0, "tokens_total": 1}]
    assert paired(rows)["n_pairs"] == 0


# --------------------------------------------------------------------------- #
# The worker quota, reported so starvation stays visible.
# --------------------------------------------------------------------------- #
def test_the_worker_quota_matches_the_locked_split():
    assert worker_quota(8000) == 2904
    assert worker_quota(16000) == 5904
    assert worker_quota(32000) == 11904


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #
def test_an_instance_at_another_optimum_is_refused(tmp_path):
    doc = subset_doc(4, 40000)
    doc["instances"]["easy"][0]["optimum"] = 4
    p = tmp_path / "s.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(AnalysisRefused, match="matched at 3"):
        load_instances([p])


def test_a_missing_run_stops_the_analysis(tmp_path):
    subs, runs = write_world(tmp_path)
    next(runs.glob("*.json")).unlink()
    instances = load_instances(subs)
    rows = load_records(runs, instances)
    assert len(rows) < len(instances) * len(CAPS) * len(CONDS)


def test_records_outside_the_frozen_pool_are_ignored(tmp_path):
    subs, runs = write_world(tmp_path)
    stray = run_doc("c1_react", "uniform-n8-t80-o50-s30000", 8000, satisfaction=1.0)
    (runs / f"{stray['run_result']['run_id']}.json").write_text(
        json.dumps(stray), encoding="utf-8")
    rows = load_records(runs, load_instances(subs))
    assert all(r["instance_id"] != "uniform-n8-t80-o50-s30000" for r in rows)


def test_another_condition_is_ignored(tmp_path):
    subs, runs = write_world(tmp_path)
    other = run_doc("c5_best_of_3", instance_id(4, 40000), 8000, satisfaction=1.0)
    (runs / f"{other['run_result']['run_id']}.json").write_text(
        json.dumps(other), encoding="utf-8")
    rows = load_records(runs, load_instances(subs))
    assert {r["condition"] for r in rows} == set(CONDS)


# --------------------------------------------------------------------------- #
# Cells.
# --------------------------------------------------------------------------- #
def test_an_empty_cell_reports_zero_rather_than_dividing():
    assert summarise([])["n"] == 0


def test_the_summary_carries_the_diagnostics_that_separate_starvation(tmp_path):
    subs, runs = write_world(tmp_path)
    instances = load_instances(subs)
    rows = load_records(runs, instances)
    s = build(rows, instances, {"n_expected": len(rows), "n_verified": len(rows),
                                "n_missing": 0, "n_incompatible": 0,
                                "n_unexpected_files": 0, "all_checks_passed": True})
    cell = s["cells"]["n4|8000|c3_mas"]
    for key in ("mean_cap_utilisation", "termination", "mean_calls", "mean_tokens"):
        assert key in cell
    assert s["c3_worker_quota_by_cap"]["8000"] == 2904


# --------------------------------------------------------------------------- #
# Wording the report is not allowed to drift on.
# --------------------------------------------------------------------------- #
def test_the_report_keeps_the_secondary_metric_secondary(tmp_path):
    subs, runs = write_world(tmp_path)
    instances = load_instances(subs)
    rows = load_records(runs, instances)
    text = render(build(rows, instances,
                        {"n_expected": len(rows), "n_verified": len(rows), "n_missing": 0,
                         "n_incompatible": 0, "n_unexpected_files": 0,
                         "all_checks_passed": True}))
    assert "secondary, diagnostic" in text
    assert "Never** described as performance" in text
    assert "no cap is selected" in text
    assert "not inference" in text
    # A line that FORBIDS a word is not a line that uses it. Scanning the whole report
    # flagged the prohibition itself -- a guard that fails on an honest disclaimer and
    # passes on a silent removal of it.
    claims = "\n".join(line for line in text.splitlines()
                       if "never" not in line.lower())
    for banned in ("accuracy", "significant", "p =", "crossover threshold"):
        assert banned not in claims, f"report claims {banned!r}"


def test_the_report_states_that_n_is_not_a_complexity_level(tmp_path):
    subs, runs = write_world(tmp_path)
    instances = load_instances(subs)
    rows = load_records(runs, instances)
    text = render(build(rows, instances,
                        {"n_expected": len(rows), "n_verified": len(rows), "n_missing": 0,
                         "n_incompatible": 0, "n_unexpected_files": 0,
                         "all_checks_passed": True}))
    assert "generator parameter" in text
    assert "not identified" in text


# --------------------------------------------------------------------------- #
# End to end.
# --------------------------------------------------------------------------- #
def test_the_cli_refuses_an_incomplete_sweep(tmp_path):
    subs, runs = write_world(tmp_path)
    victim = next(runs.glob("*.json"))
    expected = {"schema_version": "expected_runs/1.0", "subset_hash": "s" * 64,
                "binning_hash": "b" * 64, "model": MODEL,
                "runs": [{"run_id": p.stem,
                          "condition": p.stem.split("__")[0],
                          "instance_id": p.stem.split("__")[1],
                          "cap": int(p.stem.split("__")[2].removeprefix("cap")),
                          "level": "easy"} for p in sorted(runs.glob("*.json"))]}
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(expected), encoding="utf-8")
    victim.unlink()
    args = ["--runs", str(runs), "--expected", str(exp), "--out", str(tmp_path / "o")]
    for s in subs:
        args += ["--subset", str(s)]
    with pytest.raises(AnalysisRefused, match="completeness audit failed"):
        main(args)
