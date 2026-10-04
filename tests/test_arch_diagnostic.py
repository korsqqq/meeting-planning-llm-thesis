# tests/test_arch_diagnostic.py
"""Offline tests for the post-gate architecture diagnostic, on synthetic documents.

The diagnostic has one job it must not quietly fail at: staying a description that cannot
be mistaken for a repair of the C1 budget gate. The tests therefore cover the audit that
makes the comparison meaningful at all, the definition of the reported quantity, and the
absence of anything that would turn counting into inference.
"""

from __future__ import annotations

import json

import pytest

from scripts.analyse_arch_diagnostic import (
    BANDS,
    DIAGNOSTIC_CAP,
    DEFAULT_DIAGNOSTIC_CONDITIONS as DIAGNOSTIC_CONDITIONS,
    KNOWN_DIAGNOSTIC_CONDITIONS,
    EXPECTED_BINNING_HASH,
    EXPECTED_SUBSET_HASH,
    N_INSTANCES_PER_BAND,
    REFERENCE_RATE,
    DiagnosticAuditError,
    analyse,
    audit,
    load_runs,
    main,
)

LEVEL_OF_BAND = {"low": "easy", "medium": "medium", "high": "hard"}


def doc_for(instance_id, band, condition, *, cap=DIAGNOSTIC_CAP, meetings=2,
            feasible=True, satisfaction=0.67, seed=30000, calls=(), proposals=(),
            subset=EXPECTED_SUBSET_HASH, binning=EXPECTED_BINNING_HASH,
            model="Qwen/Qwen3-32B-AWQ", git_commit="abc123", termination="agent_finish"):
    return {
        "run_result": {
            "run_id": f"{condition}__{instance_id}__cap{cap}",
            "condition": condition, "instance_id": instance_id, "cap": cap,
            "level": LEVEL_OF_BAND[band], "model": model, "seed": seed,
            "score": {"solver_optimum": 3, "n_valid_meetings": meetings,
                      "satisfaction": satisfaction, "valid": feasible},
            "tokens": {"total": cap // 2, "budget_exhausted": False},
            "calls": [{"role": r} for r in calls],
            "final_plan": {"meetings": [{"person_id": f"p{i}"} for i in range(meetings)]},
        },
        "agent": {"termination": termination, "n_steps": 4,
                  "proposals": list(proposals)},
        "pilot_sweep": {"subset_hash": subset, "binning_hash": binning,
                        "git_commit": git_commit},
    }


def prop(role, valid, reasons=()):
    return {"role": role, "parsed": True, "valid": valid, "reasons": list(reasons),
            "accepted_into_best_plan": valid}


def instances():
    out, seed = [], 30000
    for band in BANDS:
        for _ in range(N_INSTANCES_PER_BAND):
            out.append((f"uniform-n8-t90-o50-s{seed}", band, seed))
            seed += 1
    return out


def write_sets(tmp_path, success, *, c1_success=lambda band, i: False,
               calls_for=lambda cond: (), props_for=lambda cond: ()):
    """A complete diagnostic set plus the C1 arm, in two directories."""
    diag_dir, c1_dir = tmp_path / "diag", tmp_path / "c1"
    diag_dir.mkdir(parents=True)
    c1_dir.mkdir(parents=True)
    for i, (iid, band, seed) in enumerate(instances()):
        ok = c1_success(band, i)
        d = doc_for(iid, band, "c1_react", meetings=2 if ok else 0,
                    satisfaction=0.67 if ok else 0.0, seed=seed)
        (c1_dir / f"c1_{iid}.json").write_text(json.dumps(d), encoding="utf-8")
        for cond in DIAGNOSTIC_CONDITIONS:
            ok = success(cond, band, i)
            d = doc_for(iid, band, cond, meetings=2 if ok else 0,
                        satisfaction=0.67 if ok else 0.0, seed=seed,
                        calls=calls_for(cond), proposals=props_for(cond))
            (diag_dir / f"{cond}_{iid}.json").write_text(json.dumps(d), encoding="utf-8")
    return diag_dir, c1_dir


# --------------------------------------------------------------------------- #
# The reported quantity.
# --------------------------------------------------------------------------- #
def test_an_empty_plan_is_not_counted_even_though_it_is_feasible(tmp_path):
    """The reason a generic feasibility rate would be useless here."""
    d = tmp_path / "r"
    d.mkdir()
    doc = doc_for("i", "low", "c2_verify_revise", meetings=0, feasible=True,
                  satisfaction=0.0)
    (d / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    row = load_runs(d)[0]
    assert row["feasible"] is True
    assert row["nonempty_validated"] is False


def test_an_infeasible_plan_with_meetings_is_not_counted(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    doc = doc_for("i", "low", "c3_mas", meetings=4, feasible=False, satisfaction=0.9)
    (d / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    assert load_runs(d)[0]["nonempty_validated"] is False


# --------------------------------------------------------------------------- #
# The audit.
# --------------------------------------------------------------------------- #
def test_a_complete_set_passes(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    rep = audit(load_runs(diag), load_runs(c1))
    assert rep["all_checks_passed"] is True
    assert rep["n_diagnostic_runs"] == 120
    assert rep["n_c1_baseline_runs"] == 60
    assert rep["single_provenance_commit"] is True


def test_a_missing_run_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    sorted(diag.glob("*.json"))[0].unlink()
    with pytest.raises(DiagnosticAuditError, match="expected 120 diagnostic runs"):
        audit(load_runs(diag), load_runs(c1))


def test_only_one_condition_present_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    for p in diag.glob("c3_mas_*.json"):
        p.unlink()
    with pytest.raises(DiagnosticAuditError, match="conditions"):
        audit(load_runs(diag), load_runs(c1))


def test_a_wrong_cap_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["cap"] = 32000
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError, match="caps"):
        audit(load_runs(diag), load_runs(c1))


def test_a_foreign_subset_hash_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["pilot_sweep"]["subset_hash"] = "0" * 64
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError, match="subset hash"):
        audit(load_runs(diag), load_runs(c1))


def test_a_duplicate_run_id_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    (diag / "copy.json").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError):
        audit(load_runs(diag), load_runs(c1))


def test_an_instance_the_c1_arm_does_not_cover_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("c2_verify_revise_*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["instance_id"] = "uniform-n8-t90-o50-s39999"
    doc["run_result"]["run_id"] += "-moved"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError, match="missing here|not in the C1 arm"):
        audit(load_runs(diag), load_runs(c1))


def test_a_held_out_seed_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["seed"] = 100001
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError, match="development seed range"):
        audit(load_runs(diag), load_runs(c1))


def test_a_second_model_fails(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["run_result"]["model"] = "Qwen/Qwen3-8B-AWQ"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(DiagnosticAuditError, match="more than one model"):
        audit(load_runs(diag), load_runs(c1))


def test_mixed_provenance_is_reported_rather_than_fatal(tmp_path):
    """Recorded, not raised: the runs still exist and the fact must be visible."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["pilot_sweep"]["git_commit"] = "deadbeef"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rep = audit(load_runs(diag), load_runs(c1))
    assert rep["single_provenance_commit"] is False
    assert len(rep["diagnostic_git_commits"]) == 2


def test_c1_runs_at_other_caps_are_ignored_not_fatal(tmp_path):
    """The C1 directory holds the whole ladder; only the 64k arm is compared."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    iid, band, seed = instances()[0]
    extra = doc_for(iid, band, "c1_react", cap=8000, meetings=0, satisfaction=0.0,
                    seed=seed)
    (c1 / "c1_extra.json").write_text(json.dumps(extra), encoding="utf-8")
    rep = audit(load_runs(diag), load_runs(c1))
    assert rep["n_c1_baseline_runs"] == 60


# --------------------------------------------------------------------------- #
# The reading, fixed in advance.
# --------------------------------------------------------------------------- #
def test_one_condition_clearing_every_band_gives_the_not_universal_reading(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: c == "c3_mas" and i % 2 == 0)
    a = analyse(load_runs(diag), load_runs(c1))
    assert a["reaches_reference_in_every_band"]["c3_mas"] is True
    assert a["reaches_reference_in_every_band"]["c2_verify_revise"] is False
    assert a["reading"]["verdict"] == "NOT_A_UNIVERSAL_TASK_WIDE_FLOOR"


def test_clearing_only_two_bands_is_not_enough(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: c == "c3_mas" and b != "high" and i % 2 == 0)
    a = analyse(load_runs(diag), load_runs(c1))
    assert a["reaches_reference_in_every_band"]["c3_mas"] is False
    assert a["reading"]["verdict"] == "N8_REMAINS_PROBLEMATIC"


def test_neither_condition_clearing_gives_the_problematic_reading(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: i < 5)
    a = analyse(load_runs(diag), load_runs(c1))
    assert a["reading"]["verdict"] == "N8_REMAINS_PROBLEMATIC"
    assert "reconsidered" in a["reading"]["statement"]


def test_exactly_the_reference_rate_counts_as_reaching_it(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: i % 2 == 0)
    a = analyse(load_runs(diag), load_runs(c1))
    for cond in DIAGNOSTIC_CONDITIONS:
        for b in BANDS:
            blk = a["by_condition"][cond]["by_band"][b]
            assert blk["nonempty_validated_rate"] == REFERENCE_RATE
            assert blk["at_least_reference_rate"] is True


def test_every_output_says_it_cannot_repair_the_c1_gate(tmp_path):
    out = tmp_path / "o"
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    assert main(["--diag-runs", str(diag), "--c1-runs", str(c1),
                 "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert doc["cannot_repair_the_c1_gate"] is True
    assert "not evidence about it" in \
        doc["analysis"]["reading"]["does_not_repair_the_c1_gate"]
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "cannot repair the C1 budget gate" in report
    assert "No cap is selected here" in report


def test_the_diagnostic_introduces_no_hypothesis_test():
    """Guards the registered constraint: this script counts, pairs and describes."""
    import tokenize
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyse_arch_diagnostic.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))
    for banned in ("bootstrap", "permutation", "p_value", "pvalue", "wilcoxon",
                   "mannwhitney", "ttest", "kruskal", "jonckheere", "ci_low"):
        assert banned not in code.lower(), f"diagnostic source mentions {banned!r}"


# --------------------------------------------------------------------------- #
# Condition-specific reporting.
# --------------------------------------------------------------------------- #
def test_c2_stage_reach_counts_verify_and_revise_calls(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: True,
        calls_for=lambda cond: (("react_step", "verify", "revise", "verify", "revise",
                                 "finalize") if cond == "c2_verify_revise" else ()))
    a = analyse(load_runs(diag), load_runs(c1))
    sr = a["by_condition"]["c2_verify_revise"]["stage_reach"]
    assert sr["runs_with_verify_at_least_1"] == 60
    assert sr["runs_with_revise_at_least_2"] == 60
    assert sr["runs_reaching_full_two_cycles"] == 60
    assert sr["mean_verify_calls"] == 2.0


def test_c2_stage_reach_sees_a_run_that_stopped_after_one_cycle(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: True,
        calls_for=lambda cond: (("react_step", "verify", "revise", "finalize")
                                if cond == "c2_verify_revise" else ()))
    sr = analyse(load_runs(diag), load_runs(c1))[
        "by_condition"]["c2_verify_revise"]["stage_reach"]
    assert sr["runs_with_verify_at_least_1"] == 60
    assert sr["runs_with_verify_at_least_2"] == 0
    assert sr["runs_reaching_full_two_cycles"] == 0


def test_c3_validity_is_split_by_worker_and_critic(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: True,
        calls_for=lambda cond: (("worker_a", "worker_b", "critic", "finalize")
                                if cond == "c3_mas" else ()),
        props_for=lambda cond: ((prop("worker_a", False, ["travel_infeasible"]),
                                 prop("worker_b", True),
                                 prop("critic", True))
                                if cond == "c3_mas" else ()))
    rs = analyse(load_runs(diag), load_runs(c1))["by_condition"]["c3_mas"]["role_split"]
    assert rs["worker_proposal_attempts"] == 120      # 2 per run x 60
    assert rs["worker_proposal_valid"] == 60
    assert rs["worker_validity_rate"] == 0.5
    assert rs["critic_proposal_attempts"] == 60
    assert rs["critic_validity_rate"] == 1.0
    assert rs["runs_where_critic_ran"] == 60
    assert rs["critic_reach_rate"] == 1.0


def test_c3_critic_reach_falls_when_the_critic_is_skipped(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: True,
        calls_for=lambda cond: (("worker_a", "worker_b", "finalize")
                                if cond == "c3_mas" else ()))
    rs = analyse(load_runs(diag), load_runs(c1))["by_condition"]["c3_mas"]["role_split"]
    assert rs["runs_where_critic_ran"] == 0
    assert rs["critic_reach_rate"] == 0.0
    assert rs["critic_validity_rate"] is None


def test_the_rejection_taxonomy_is_multi_label(tmp_path):
    diag, c1 = write_sets(
        tmp_path, lambda c, b, i: True,
        props_for=lambda cond: (prop("react_step", False,
                                     ["travel_infeasible", "window_violation"]),))
    pooled = analyse(load_runs(diag), load_runs(c1))[
        "by_condition"]["c2_verify_revise"]["pooled"]
    assert pooled["rejection_reasons"] == {"travel_infeasible": 60,
                                           "window_violation": 60}
    assert pooled["proposal_attempts"] == 60
    assert pooled["proposal_valid"] == 0


# --------------------------------------------------------------------------- #
# Pairing against C1.
# --------------------------------------------------------------------------- #
def test_pairing_counts_recoveries_and_losses_on_the_same_instance(tmp_path):
    # C1 succeeds on the first five of each band; C2 succeeds on all.
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True,
                          c1_success=lambda b, i: i % N_INSTANCES_PER_BAND < 5)
    pr = analyse(load_runs(diag), load_runs(c1))[
        "by_condition"]["c2_verify_revise"]["paired_vs_c1"]
    assert pr["n_paired"] == 60
    assert pr["recovered_where_c1_failed"] == 45
    assert pr["lost_where_c1_succeeded"] == 0
    assert pr["same_outcome"] == 15


def test_pairing_reports_losses_too(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: False,
                          c1_success=lambda b, i: True)
    pr = analyse(load_runs(diag), load_runs(c1))[
        "by_condition"]["c3_mas"]["paired_vs_c1"]
    assert pr["recovered_where_c1_failed"] == 0
    assert pr["lost_where_c1_succeeded"] == 60
    assert pr["mean_satisfaction_delta_vs_c1"] < 0


# --------------------------------------------------------------------------- #
# Extending the diagnostic to later conditions must not disturb the frozen one.
# --------------------------------------------------------------------------- #
def _add_condition(diag_dir, cond, success):
    for i, (iid, band, seed) in enumerate(instances()):
        ok = success(band, i)
        d = doc_for(iid, band, cond, meetings=2 if ok else 0,
                    satisfaction=0.67 if ok else 0.0, seed=seed, git_commit="later")
        d["agent"]["diagnostics"] = {"pool_size": 3, "fallback_size": 1, "headroom": 2,
                                     "critic_reached": ok, "critic_improved": ok,
                                     "evidence_travel_coverage": 0.5}
        (diag_dir / f"{cond}_{iid}.json").write_text(json.dumps(d), encoding="utf-8")


def test_the_default_condition_set_is_the_frozen_pair():
    assert DIAGNOSTIC_CONDITIONS == ("c2_verify_revise", "c3_mas")
    assert set(DIAGNOSTIC_CONDITIONS) <= set(KNOWN_DIAGNOSTIC_CONDITIONS)
    assert "c4_planner_critic" in KNOWN_DIAGNOSTIC_CONDITIONS


def test_adding_c4_to_the_directory_does_not_change_the_frozen_reading(tmp_path):
    """The C2/C3 verdict is recorded; a later condition in the same directory must not
    be able to move it."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: c == "c3_mas" and i % 2 == 0)
    before = analyse(load_runs(diag), load_runs(c1))
    _add_condition(diag, "c4_planner_critic", lambda b, i: True)
    rows = [r for r in load_runs(diag) if r["condition"] in DIAGNOSTIC_CONDITIONS]
    after = analyse(rows, load_runs(c1))
    assert before == after
    assert audit(rows, load_runs(c1))["all_checks_passed"] is True


def test_a_declared_condition_that_is_absent_fails_the_audit(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    conds = (*DIAGNOSTIC_CONDITIONS, "c4_planner_critic")
    with pytest.raises(DiagnosticAuditError, match="expected 180 diagnostic runs"):
        audit(load_runs(diag), load_runs(c1), conds)


def test_four_conditions_are_analysed_and_paired_together(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    _add_condition(diag, "c4_planner_critic", lambda b, i: i < 12)
    conds = (*DIAGNOSTIC_CONDITIONS, "c4_planner_critic")
    rows = load_runs(diag)
    rep = audit(rows, load_runs(c1), conds)
    assert rep["n_diagnostic_runs"] == 180
    a = analyse(rows, load_runs(c1), conds)
    assert set(a["by_condition"]) == {"c1_react", *conds}
    assert a["by_condition"]["c4_planner_critic"]["paired_vs_c1"]["n_paired"] == 60
    assert set(a["reaches_reference_in_every_band"]) == set(conds)


def test_condition_diagnostics_are_summarised_for_c4(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    _add_condition(diag, "c4_planner_critic", lambda b, i: i < 12)
    conds = (*DIAGNOSTIC_CONDITIONS, "c4_planner_critic")
    a = analyse(load_runs(diag), load_runs(c1), conds)
    cd = a["by_condition"]["c4_planner_critic"]["condition_diagnostics"]
    assert cd["n_with_diagnostics"] == 60
    assert cd["mean_pool_size"] == 3.0 and cd["mean_headroom"] == 2.0
    # `i` runs 0..59 over the flat instance list, so `i < 12` marks 12 runs.
    assert cd["runs_critic_reached"] == 12 and cd["rate_critic_improved"] == 0.2
    assert cd["mean_evidence_travel_coverage"] == 0.5


def test_provenance_is_reported_per_condition(tmp_path):
    """Conditions run weeks apart carry different commits, which is not a defect; a
    condition split across commits is."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    _add_condition(diag, "c4_planner_critic", lambda b, i: True)
    conds = (*DIAGNOSTIC_CONDITIONS, "c4_planner_critic")
    rep = audit(load_runs(diag), load_runs(c1), conds)
    assert rep["git_commit_per_condition"]["c4_planner_critic"] == ["later"]
    assert rep["git_commit_per_condition"]["c3_mas"] == ["abc123"]
    assert rep["conditions_with_split_provenance"] == []
    assert rep["single_provenance_commit"] is False


def test_a_condition_split_across_commits_is_flagged(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    p = sorted(diag.glob("c3_mas_*.json"))[0]
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["pilot_sweep"]["git_commit"] = "deadbeef"
    p.write_text(json.dumps(doc), encoding="utf-8")
    rep = audit(load_runs(diag), load_runs(c1))
    assert rep["conditions_with_split_provenance"] == ["c3_mas"]


def test_the_cli_refuses_an_unknown_condition(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    with pytest.raises(SystemExit, match="unknown diagnostic condition"):
        main(["--diag-runs", str(diag), "--c1-runs", str(c1),
              "--out", str(tmp_path / "o"), "--conditions", "c9_wat"])


def test_the_cli_reproduces_the_frozen_pair_by_default(tmp_path):
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    _add_condition(diag, "c4_planner_critic", lambda b, i: False)
    out = tmp_path / "o"
    assert main(["--diag-runs", str(diag), "--c1-runs", str(c1),
                 "--out", str(out)]) == 0
    doc = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(doc["analysis"]["by_condition"]) == {"c1_react", *DIAGNOSTIC_CONDITIONS}
    assert doc["audit"]["n_diagnostic_runs"] == 120


def test_the_distinct_product_histogram_covers_each_run_once(tmp_path):
    """It once counted every run twice: the mean stayed right while the buckets
    doubled, which is the failure mode that reads as plausible."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    for i, (iid, band, seed) in enumerate(instances()):
        d = doc_for(iid, band, "c5_best_of_3", seed=seed, git_commit="later")
        d["agent"]["diagnostics"] = {
            "n_distinct_products": 1 if i % 10 else 2,
            "winner_attempt_index": 0 if i % 10 else 1,
        }
        (diag / f"c5_{iid}.json").write_text(json.dumps(d), encoding="utf-8")
    conds = (*DIAGNOSTIC_CONDITIONS, "c5_best_of_3")
    cd = analyse(load_runs(diag), load_runs(c1), conds)[
        "by_condition"]["c5_best_of_3"]["condition_diagnostics"]
    assert sum(cd["distinct_product_counts"].values()) == 60
    assert cd["distinct_product_counts"] == {"1": 54, "2": 6, "3": 0}
    assert sum(cd["winner_attempt_counts"].values()) == 60
    assert cd["mean_n_distinct_products"] == round(66 / 60, 4)


def test_distinct_products_are_cross_tabulated_against_the_outcome(tmp_path):
    """Equal marginal counts do not establish a run-by-run correspondence, so the
    collapse claim needs the joint table, not two margins that happen to match."""
    diag, c1 = write_sets(tmp_path, lambda c, b, i: True)
    for i, (iid, band, seed) in enumerate(instances()):
        # 50 runs: one product and zero satisfaction. 4: one product but a positive
        # score -- the case that would make "same count" a false inference. 6: two
        # products, positive.
        if i < 50:
            n_products, sat = 1, 0.0
        elif i < 54:
            n_products, sat = 1, 0.5
        else:
            n_products, sat = 2, 0.5
        d = doc_for(iid, band, "c5_best_of_3", seed=seed, satisfaction=sat,
                    meetings=0 if sat == 0.0 else 2, git_commit="later")
        d["agent"]["diagnostics"] = {"n_distinct_products": n_products,
                                     "winner_attempt_index": 0}
        (diag / f"c5_{iid}.json").write_text(json.dumps(d), encoding="utf-8")
    conds = (*DIAGNOSTIC_CONDITIONS, "c5_best_of_3")
    cd = analyse(load_runs(diag), load_runs(c1), conds)[
        "by_condition"]["c5_best_of_3"]["condition_diagnostics"]
    assert cd["distinct_product_counts"] == {"1": 54, "2": 6, "3": 0}
    assert cd["distinct_products_by_outcome"] == {
        "1": {"satisfaction_zero": 50, "satisfaction_positive": 4},
        "2": {"satisfaction_zero": 0, "satisfaction_positive": 6},
    }
    # The margins alone would have suggested 54 == 54; the joint table shows 50.
    total_zero = sum(c["satisfaction_zero"]
                     for c in cd["distinct_products_by_outcome"].values())
    assert total_zero == 50 and cd["distinct_product_counts"]["1"] == 54
