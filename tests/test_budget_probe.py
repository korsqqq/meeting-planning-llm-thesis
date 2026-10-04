# tests/test_budget_probe.py
"""Offline tests for the paired 64 000 -> 128 000 budget probe analyser.

The probe is only meaningful if every comparison is genuinely paired on the same instance
at both caps, and only honest if nothing in it can be mistaken for a gate. The tests below
attack both, plus the quantity that can answer the probe's question on its own -- whether
the extra budget was spent at all.
"""

from __future__ import annotations

import json

import pytest

from scripts.analyse_budget_probe import (
    BANDS,
    BASELINE_CAP,
    DEFAULT_CONDITIONS,
    N_PER_BAND,
    PROBE_CAP,
    ProbeAuditError,
    analyse,
    audit,
    load_probe_instances,
    load_runs,
    main,
)
from scripts.analyse_budget_probe import EXPECTED_BINNING_HASH as BINNING

LEVEL_OF_BAND = {"low": "easy", "medium": "medium", "high": "hard"}


def doc_for(iid, band, cond, cap, *, sat, meetings=None, tokens=None, seed=30000,
            binning=BINNING, model="Qwen/Qwen3-32B-AWQ", commit="fb597ad", calls=3,
            termination="agent_finish"):
    if meetings is None:
        meetings = 2 if sat > 0 else 0
    if tokens is None:
        tokens = cap // 2
    return {
        "run_result": {
            "run_id": f"{cond}__{iid}__cap{cap}", "condition": cond, "instance_id": iid,
            "cap": cap, "level": LEVEL_OF_BAND[band], "model": model, "seed": seed,
            "score": {"solver_optimum": 3, "n_valid_meetings": meetings,
                      "satisfaction": sat, "valid": True},
            "tokens": {"total": tokens, "budget_exhausted": False},
            "calls": [{"role": "react_step"}] * calls,
            "final_plan": {"meetings": [{"person_id": f"p{i}"} for i in range(meetings)]},
        },
        "agent": {"termination": termination, "n_steps": 4, "proposals": []},
        "pilot_sweep": {"subset_hash": "x" * 64, "binning_hash": binning,
                        "git_commit": commit},
    }


@pytest.fixture
def probe(tmp_path):
    """A frozen 12-instance subset plus a directory of runs, both caps, all conditions."""
    instances = []
    seed = 30000
    subset = {"instances": {}}
    for band in BANDS:
        rows = []
        for _ in range(N_PER_BAND):
            iid = f"uniform-n8-t90-o50-s{seed}"
            rows.append({"instance_id": iid, "seed": seed, "optimum": 3})
            instances.append((iid, band, seed))
            seed += 1
        subset["instances"][LEVEL_OF_BAND[band]] = rows
    sp = tmp_path / "subset.json"
    sp.write_text(json.dumps(subset), encoding="utf-8")
    return sp, instances, tmp_path


def write_runs(tmp_path, instances, sat_fn, *, conditions=DEFAULT_CONDITIONS,
               tokens_fn=None, **kw):
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    for cond in conditions:
        for cap in (BASELINE_CAP, PROBE_CAP):
            for i, (iid, band, seed) in enumerate(instances):
                tokens = tokens_fn(cond, cap, i) if tokens_fn else cap // 2
                doc = doc_for(iid, band, cond, cap, sat=sat_fn(cond, cap, i),
                              tokens=tokens, seed=seed, **kw)
                (d / f"{cond}_{cap}_{iid}.json").write_text(json.dumps(doc),
                                                            encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Pairing and the audit.
# --------------------------------------------------------------------------- #
def test_a_complete_probe_passes(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    rep = audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)
    assert rep["all_checks_passed"] is True
    assert rep["n_instances"] == 12
    assert rep["n_runs"] == 12 * 3 * 2
    assert rep["instances_per_band"] == {b: N_PER_BAND for b in BANDS}


def test_a_missing_arm_fails(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    next(iter(sorted(d.glob("c3_mas_128000_*.json")))).unlink()
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    with pytest.raises(ProbeAuditError, match="c3_mas at 128000"):
        audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)


def test_an_instance_present_at_one_cap_only_fails(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    p = sorted(d.glob("c1_react_128000_*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["instance_id"] = "uniform-n8-t90-o50-s39999"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    with pytest.raises(ProbeAuditError):
        audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)


def test_a_foreign_binning_hash_fails(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5, binning="0" * 64)
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    with pytest.raises(ProbeAuditError, match="binning hash"):
        audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)


def test_a_second_model_fails(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    p = sorted(d.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["model"] = "Qwen/Qwen3-8B-AWQ"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    with pytest.raises(ProbeAuditError, match="more than one model"):
        audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)


def test_runs_outside_the_probe_subset_are_ignored_not_fatal(probe):
    """The 64 000 arms live in directories holding 60 instances; only 12 are the probe."""
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    extra = doc_for("uniform-n8-t90-o50-s39998", "low", "c1_react", BASELINE_CAP,
                    sat=1.0, seed=39998)
    (d / "outsider.json").write_text(json.dumps(extra), encoding="utf-8")
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    assert audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)["n_runs"] == 72


def test_runs_at_other_caps_are_ignored(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    iid, band, seed = instances[0]
    (d / "cap32k.json").write_text(
        json.dumps(doc_for(iid, band, "c1_react", 32000, sat=0.9, seed=seed)),
        encoding="utf-8")
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    assert audit(rows, load_probe_instances(sp), DEFAULT_CONDITIONS)["all_checks_passed"]


# --------------------------------------------------------------------------- #
# The paired change.
# --------------------------------------------------------------------------- #
def test_the_paired_delta_is_computed_within_instance(probe):
    sp, instances, tmp = probe
    # 128k is better on the first half of every band, worse on the rest.
    d = write_runs(tmp, instances,
                   lambda c, cap, i: (0.75 if cap == PROBE_CAP else 0.25) if i % 2 == 0
                   else (0.25 if cap == PROBE_CAP else 0.75))
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    pr = analyse(rows, DEFAULT_CONDITIONS)["by_condition"]["c1_react"]["paired"]["pooled"]
    assert pr["n_paired"] == 12
    assert pr["improved"] == 6 and pr["worsened"] == 6 and pr["unchanged"] == 0
    assert pr["mean_delta_satisfaction"] == 0.0


def test_no_change_is_counted_as_unchanged(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    pr = analyse(rows, DEFAULT_CONDITIONS)["by_condition"]["c3_mas"]["paired"]["pooled"]
    assert (pr["improved"], pr["worsened"], pr["unchanged"]) == (0, 0, 12)
    assert pr["mean_delta_satisfaction"] == 0.0


def test_gaining_and_losing_a_validated_plan_are_counted_separately(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances,
                   lambda c, cap, i: (0.5 if cap == PROBE_CAP else 0.0) if i < 3
                   else (0.0 if cap == PROBE_CAP else 0.5) if i < 5 else 0.5)
    rows = load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS)
    pr = analyse(rows, DEFAULT_CONDITIONS)["by_condition"]["c1_react"]["paired"]["pooled"]
    assert pr["gained_a_validated_plan"] == 3
    assert pr["lost_a_validated_plan"] == 2


def test_bands_are_paired_separately(probe):
    sp, instances, tmp = probe
    # only the high band improves
    d = write_runs(tmp, instances,
                   lambda c, cap, i: 0.75 if (cap == PROBE_CAP and i >= 8) else 0.25)
    a = analyse(load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS),
                DEFAULT_CONDITIONS)
    paired = a["by_condition"]["c1_react"]["paired"]
    assert paired["low"]["mean_delta_satisfaction"] == 0.0
    assert paired["medium"]["mean_delta_satisfaction"] == 0.0
    assert paired["high"]["mean_delta_satisfaction"] == 0.5
    assert paired["high"]["n_paired"] == N_PER_BAND


# --------------------------------------------------------------------------- #
# The quantity that can answer the question on its own.
# --------------------------------------------------------------------------- #
def test_runs_that_never_exceed_the_old_cap_are_counted_both_ways(probe):
    sp, instances, tmp = probe
    # At 128k: the first 4 spend more than 64000, the rest do not.
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5,
                   tokens_fn=lambda c, cap, i: (70000 if i < 4 else 40000)
                   if cap == PROBE_CAP else 50000)
    a = analyse(load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS),
                DEFAULT_CONDITIONS)
    b128 = a["by_condition"]["c1_react"]["by_cap"][str(PROBE_CAP)]
    assert b128["runs_over_64000_tokens"] == 4
    assert b128["runs_under_64000_tokens"] == 8
    assert b128["runs_over_64000_tokens"] + b128["runs_under_64000_tokens"] == b128["n"]


def test_cap_utilisation_is_relative_to_each_runs_own_cap(probe):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5,
                   tokens_fn=lambda c, cap, i: cap // 2)
    a = analyse(load_runs([d], load_probe_instances(sp), DEFAULT_CONDITIONS),
                DEFAULT_CONDITIONS)
    e = a["by_condition"]["c1_react"]["by_cap"]
    assert e[str(BASELINE_CAP)]["mean_cap_utilisation"] == 0.5
    assert e[str(PROBE_CAP)]["mean_cap_utilisation"] == 0.5
    assert e[str(PROBE_CAP)]["mean_tokens"] == 2 * e[str(BASELINE_CAP)]["mean_tokens"]


# --------------------------------------------------------------------------- #
# Nothing here may read as a gate.
# --------------------------------------------------------------------------- #
def test_the_probe_selects_no_cap_and_introduces_no_test(probe, tmp_path):
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5)
    out = tmp_path / "o"
    assert main(["--runs", str(d), "--subset", str(sp), "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["selects_a_cap"] is False
    reading = doc["analysis"]["reading"]
    assert "not a hypothesis test" in reading["status"]
    assert reading["primary_metric"] == "satisfaction"
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "not a gate" in report
    assert "never called performance" in report


def test_the_analyser_introduces_no_hypothesis_test():
    import tokenize
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyse_budget_probe.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))
    for banned in ("bootstrap", "permutation", "pvalue", "p_value", "wilcoxon",
                   "ttest", "kruskal", "ci_low", "threshold", "meets_"):
        assert banned not in code.lower(), f"probe analyser mentions {banned!r}"


def test_only_the_requested_conditions_are_analysed(probe):
    """C5 lands later than C1 and C3, so a partial probe must still be analysable."""
    sp, instances, tmp = probe
    d = write_runs(tmp, instances, lambda c, cap, i: 0.5,
                   conditions=("c1_react", "c3_mas"))
    conds = ("c1_react", "c3_mas")
    rows = load_runs([d], load_probe_instances(sp), conds)
    assert audit(rows, load_probe_instances(sp), conds)["n_runs"] == 48
    assert set(analyse(rows, conds)["by_condition"]) == set(conds)
