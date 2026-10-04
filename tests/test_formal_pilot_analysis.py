# tests/test_formal_pilot_analysis.py
"""Offline tests for the formal-pilot analyser: no runs, no endpoint, synthetic documents.

Three things are worth testing here and the rest follows from them. The integrity gate,
because every table is an average over the run set and a set that is short or mixed would
still produce a readable table. The sign convention, because the expose defines
`delta = S_C1 - S_C3` and an inverted delta would invert H1/H2 in the text. And the
pairing, because both inferential procedures are only valid if the bootstrap resamples
whole paired differences and the permutation keeps each difference intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyse_formal_pilot import (
    CONDITIONS,
    EXPECTED_BINNING_HASH,
    EXPECTED_CAPS,
    EXPECTED_INSTANCES,
    EXPECTED_SUBSET_HASH,
    FormalPilotIntegrityError,
    aggregate,
    contrast,
    integrity_audit,
    interaction_tests,
    load_runs,
    main,
    paired_bootstrap_ci,
    sign_test,
    spearman,
    stage_exposure,
)
from scripts.run_pilot_sweep import content_hash

COMMIT = "916434ac8d27a4c889ee754c779d1dfeaf47e29a"
LEVELS = ["easy"] * 10 + ["medium"] * 10 + ["hard"] * 10
# Reversed crossover on purpose: the hierarchy leads on easy and trails on hard, so a
# test that silently flipped the sign convention would still look plausible unless the
# direction is asserted explicitly.
SAT = {
    "easy": {"c1_react": 0.2, "c2_verify_revise": 0.3, "c3_mas": 0.8},
    "medium": {"c1_react": 0.4, "c2_verify_revise": 0.4, "c3_mas": 0.6},
    "hard": {"c1_react": 0.6, "c2_verify_revise": 0.5, "c3_mas": 0.4},
}
ROLE_CALLS = {
    "c1_react": [("react_step", 3), ("finalize", 1)],
    "c2_verify_revise": [("react_step", 3), ("verify", 2), ("revise", 1), ("finalize", 1)],
    "c3_mas": [("worker_a", 2), ("worker_b", 2), ("critic", 1), ("finalize", 1)],
}
ROLE_PROPS = {
    "c1_react": [("react_step", 2, 1)],
    "c2_verify_revise": [("react_step", 2, 1), ("revise", 1, 1)],
    # The critic's single attempt is always valid: it only emits what already passed the
    # aggregator gate. Pooling it with the workers is the confound the split exists for.
    "c3_mas": [("worker_a", 2, 1), ("worker_b", 2, 0), ("critic", 1, 1)],
}


def make_doc(*, condition, instance_id, cap, level, satisfaction, optimum=4,
             complexity=3, tokens=None, subset=EXPECTED_SUBSET_HASH,
             binning=EXPECTED_BINNING_HASH, commit=COMMIT, n_people=4,
             travel="uniform", termination="agent_finish", exhausted=False) -> dict:
    total = cap - 156 if tokens is None else tokens
    achieved = round(satisfaction * optimum)
    calls = []
    for role, k in ROLE_CALLS[condition]:
        calls += [{"input_tokens": 100, "thinking_tokens": 50, "answer_tokens": 10,
                   "finish_reason": "stop", "context_limited": False, "role": role,
                   "latency_seconds": 1.5}] * k
    props, step = [], 0
    for role, k, n_valid in ROLE_PROPS[condition]:
        for i in range(k):
            step += 1
            valid = i < n_valid
            props.append({"parsed": True, "valid": valid,
                          "accepted_into_best_plan": valid, "role": role, "step": step,
                          "n_meetings": achieved, "condition": condition,
                          "reasons": [] if valid else ["travel_infeasible"]})
    return {
        "run_result": {
            "run_id": f"{condition}__{instance_id}__cap{cap}__M",
            "condition": condition, "instance_id": instance_id, "cap": cap,
            "level": level, "model": "Qwen/Qwen3-32B-AWQ", "seed": 10005,
            "latency_seconds": 42.0, "calls": calls,
            "score": {"valid": True, "satisfaction": satisfaction,
                      "n_valid_meetings": achieved, "solver_optimum": optimum,
                      "optimality": achieved == optimum, "invalid_reasons": []},
            "tokens": {"total": total, "cap": cap, "budget_exhausted": exhausted,
                       "n_calls": len(calls)},
            "final_plan": {"meetings": [{} for _ in range(achieved)]},
            "best_plan_so_far": {"meetings": [{} for _ in range(achieved)]},
        },
        "agent": {"termination": termination, "proposals": props, "n_steps": len(calls),
                  "finalization_mismatch": False, "empty_turns": 0},
        "instance": {"complexity_metric": complexity,
                     "generator_params": {"n_people": n_people, "travel_structure": travel,
                                          "tightness": 100, "overlap": 20}},
        "pilot_sweep": {"subset_hash": subset, "binning_hash": binning,
                        "git_commit": commit, "max_model_len": 32768,
                        "base_url": "http://localhost:8000/v1"},
        "usage_audit": {"has_endpoint_usage": True,
                        "totals": {"internal_total_tokens": total,
                                   "endpoint_total_tokens": total}},
        "transcript_sanity": {"think_leak": False},
    }


def build_matrix(tmp_path: Path, mutate=None) -> Path:
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    i = 0
    for k in range(EXPECTED_INSTANCES):
        iid = f"uniform-n4-t100-o20-s{10005 + k}"
        level = LEVELS[k]
        for cond in CONDITIONS:
            for cap in sorted(EXPECTED_CAPS):
                # A small per-instance jitter so the bootstrap has something to resample.
                sat = min(1.0, max(0.0, SAT[level][cond] + 0.01 * (k % 5)))
                doc = make_doc(condition=cond, instance_id=iid, cap=cap, level=level,
                               satisfaction=sat, complexity=k)
                if mutate is not None:
                    mutate(doc, i)
                (run_dir / f"{doc['run_result']['run_id']}.json").write_text(
                    json.dumps(doc), encoding="utf-8")
                i += 1
    return run_dir


def write_expected(run_dir: Path, path: Path, monkeypatch) -> Path:
    """A self-consistent expected-runs manifest, with the pinned digest pointed at it."""
    ids = sorted(json.loads(p.read_text(encoding="utf-8"))["run_result"]["run_id"]
                 for p in run_dir.glob("*.json"))
    payload = {"schema_version": "expected_runs/1.0", "runs": [{"run_id": i} for i in ids]}
    payload["content_hash"] = content_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("scripts.analyse_formal_pilot.EXPECTED_RUNS_MANIFEST_HASH",
                        payload["content_hash"])
    return path


# --------------------------------------------------------------------------- #
# Sign convention -- the expose defines delta = S_C1 - S_C3.
# --------------------------------------------------------------------------- #
def test_delta_is_single_agent_minus_multi_agent(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    c = contrast(rows, "c1_react", "c3_mas", 32000)
    assert c["minuend"] == "c1_react" and c["subtrahend"] == "c3_mas"
    # Easy: the hierarchy leads, so the expose delta must be NEGATIVE there.
    assert c["by_level"]["easy"]["delta_mean"] < 0
    # Hard: the single agent leads, so it must be POSITIVE.
    assert c["by_level"]["hard"]["delta_mean"] > 0


def test_contrast_states_its_own_direction(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    c = contrast(rows, "c1_react", "c3_mas", 64000)
    assert "S_c1_react - S_c3_mas" in c["definition"]
    assert "c3_mas scored higher" in c["definition"]


# --------------------------------------------------------------------------- #
# Pairing.
# --------------------------------------------------------------------------- #
def test_bootstrap_resamples_paired_differences():
    """Constant differences must give a degenerate interval, not a spread.

    If the two conditions were resampled independently, between-instance variance would
    leak back in and the interval would open up even though every pair differs by the
    same amount.
    """
    out = paired_bootstrap_ci([0.5] * 10)
    assert out["mean"] == 0.5
    assert out["ci_low"] == 0.5 and out["ci_high"] == 0.5
    assert out["excludes_zero"] is True


def test_bootstrap_is_deterministic_and_flags_intervals_covering_zero():
    d = [0.4, -0.3, 0.0, 0.2, -0.5, 0.1, 0.0, -0.1, 0.3, -0.2]
    a, b = paired_bootstrap_ci(d), paired_bootstrap_ci(d)
    assert a == b
    assert a["ci_low"] < 0 < a["ci_high"] and a["excludes_zero"] is False


def test_bootstrap_undefined_below_two_pairs():
    out = paired_bootstrap_ci([0.5])
    assert out["ci_low"] is None and out["excludes_zero"] is False


def test_omnibus_covers_all_three_levels_not_only_the_extremes():
    """A middle level that departs from both extremes must register."""
    rows = [(0.0, "easy", 1)] * 6 + [(0.9, "medium", 5)] * 6 + [(0.0, "hard", 9)] * 6
    out = interaction_tests(rows, permutations=2000)
    assert out["levels_present"] == ["easy", "medium", "hard"]
    # The extreme contrast sees nothing; the omnibus does.
    assert out["extreme_contrast"]["statistic_easy_minus_hard"] == pytest.approx(0.0)
    assert out["extreme_contrast"]["p_two_sided"] > 0.5
    assert out["omnibus"]["p"] < 0.01


def test_a_rising_delta_does_not_get_credited_to_the_preregistered_falling_trend():
    """The direction that matters: H1/H2 predict delta FALLING with complexity.

    Data that rises instead is evidence against the registered alternative, so the
    one-sided pre-registered p must be LARGE. Reporting the opposite direction's small p
    as though it evaluated H1/H2 would test the hypothesis against its own contradiction.
    """
    rows = [(-0.6, "easy", 1)] * 5 + [(0.0, "medium", 5)] * 5 + [(0.6, "hard", 9)] * 5
    out = interaction_tests(rows, permutations=2000)
    tr = out["trend"]
    assert tr["observed_direction"] == "delta rises with complexity"
    assert "falls with complexity" in tr["preregistered_direction"]
    assert tr["statistic"] > tr["null_expectation"]
    assert tr["p_one_sided_preregistered_decreasing"] > 0.95
    assert tr["p_one_sided_opposite_increasing_exploratory"] < 0.01


def test_the_preregistered_trend_is_detected_when_the_data_actually_falls():
    rows = [(0.3, "easy", 1), (0.2, "easy", 2), (0.1, "medium", 5), (0.0, "medium", 6),
            (-0.1, "hard", 8), (-0.2, "hard", 9)] * 3
    out = interaction_tests(rows, permutations=2000)
    tr = out["trend"]
    assert tr["observed_direction"] == "delta falls with complexity"
    assert tr["p_one_sided_preregistered_decreasing"] < 0.05
    assert tr["p_one_sided_opposite_increasing_exploratory"] > 0.95


def test_the_extreme_contrast_is_also_directional():
    """Pre-registered alternative: delta larger on easy than on hard, i.e. positive."""
    rows = [(-0.5, "easy", 1)] * 5 + [(0.5, "hard", 9)] * 5
    ex = interaction_tests(rows, permutations=2000)["extreme_contrast"]
    assert ex["statistic_easy_minus_hard"] < 0
    assert ex["p_one_sided_preregistered_positive"] > 0.95
    assert ex["p_one_sided_opposite_negative_exploratory"] < 0.05


def test_interaction_finds_nothing_when_complexity_is_unrelated():
    rows = [(0.3, "easy", 1), (-0.3, "easy", 2), (0.3, "medium", 5), (-0.3, "medium", 6),
            (0.3, "hard", 8), (-0.3, "hard", 9)] * 3
    out = interaction_tests(rows, permutations=2000)
    assert out["omnibus"]["p"] > 0.5
    assert out["trend"]["p_two_sided"] > 0.5
    assert out["trend"]["p_one_sided_preregistered_decreasing"] > 0.2
    # And it must say so without claiming independence.
    assert "does not establish independence" in out["power_note"]


def test_interaction_is_deterministic():
    rows = [(0.6, "easy", 1)] * 4 + [(0.1, "medium", 5)] * 4 + [(-0.4, "hard", 9)] * 4
    assert interaction_tests(rows) == interaction_tests(rows)


def test_interaction_trend_uses_every_level(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    it = contrast(rows, "c1_react", "c3_mas", 64000)["interaction"]
    assert it["levels_present"] == ["easy", "medium", "hard"]
    # delta rises with complexity in the fixture, so the rank correlation is positive.
    assert it["spearman_complexity_vs_delta"] > 0.8
    assert it["trend"]["observed_direction"] == "delta rises with complexity"
    # Rising is the opposite of what H1/H2 registered, so their test must not fire.
    assert it["trend"]["p_one_sided_preregistered_decreasing"] > 0.9


# --------------------------------------------------------------------------- #
# Role separation.
# --------------------------------------------------------------------------- #
def test_proposal_validity_is_split_by_role(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    at = [r for r in rows if r["condition"] == "c3_mas" and r["cap"] == 64000]
    by_role = aggregate(at)["by_role"]
    assert set(by_role) == {"worker_a", "worker_b", "critic"}
    # The critic is valid by construction; pooling it would inflate the condition.
    assert by_role["critic"]["validity_rate"] == 1.0
    assert by_role["worker_b"]["validity_rate"] == 0.0
    assert by_role["worker_a"]["validity_rate"] == 0.5
    # And the pooled figure sits above the workers taken together, which is the confound:
    # workers manage 1 valid in 4 attempts, the pooled rate reads 2 in 5.
    worker_valid = by_role["worker_a"]["valid"] + by_role["worker_b"]["valid"]
    worker_attempts = by_role["worker_a"]["attempts"] + by_role["worker_b"]["attempts"]
    assert aggregate(at)["proposal_validity_rate"] > worker_valid / worker_attempts


def test_stage_exposure_counts_verify_and_revise_directly(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    at = [r for r in rows if r["condition"] == "c2_verify_revise" and r["cap"] == 8000]
    se = stage_exposure(at)
    assert se["verify"]["calls_total"] == 2 * EXPECTED_INSTANCES
    assert se["revise"]["share_runs_reached"] == 1.0
    assert "worker_a" not in se


def test_expose_secondaries_are_extracted(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    g = aggregate([r for r in rows if r["condition"] == "c1_react" and r["cap"] == 32000])
    assert g["feasibility_rate"] == 1.0
    assert g["optimality_rate"] is not None
    assert g["latency_mean_seconds"] == 42.0
    assert g["calls_mean"] == 4.0
    assert g["budget_exhausted_runs"] == 0
    assert g["tokens_remaining_mean"] == 156.0


# --------------------------------------------------------------------------- #
# Integrity gate.
# --------------------------------------------------------------------------- #
def test_complete_matrix_passes(tmp_path):
    audit = integrity_audit(load_runs(build_matrix(tmp_path)))
    assert audit["all_checks_passed"] is True
    assert audit["n_runs"] == 360
    assert audit["provenance"]["executing_commit"] == COMMIT


def test_a_missing_run_fails(tmp_path):
    run_dir = build_matrix(tmp_path)
    next(iter(sorted(run_dir.glob("c3_mas__*cap64000*.json")))).unlink()
    with pytest.raises(FormalPilotIntegrityError, match="expected 360 runs"):
        integrity_audit(load_runs(run_dir))


def test_a_half_finished_condition_fails(tmp_path):
    run_dir = build_matrix(tmp_path)
    for p in sorted(run_dir.glob("c3_mas__*"))[:30]:
        p.unlink()
    with pytest.raises(FormalPilotIntegrityError):
        integrity_audit(load_runs(run_dir))


def test_a_truncated_subset_hash_fails(tmp_path):
    """Pinning the full digest: a prefix match is not enough to prove the manifest."""
    def mutate(doc, i):
        if i == 7:
            doc["pilot_sweep"]["subset_hash"] = EXPECTED_SUBSET_HASH[:12]
    with pytest.raises(FormalPilotIntegrityError, match="subset_hash"):
        integrity_audit(load_runs(build_matrix(tmp_path, mutate)))


def test_a_split_repository_state_fails(tmp_path):
    def mutate(doc, i):
        if i > 300:
            doc["pilot_sweep"]["git_commit"] = "deadbeef" * 5
    with pytest.raises(FormalPilotIntegrityError, match="more than one repository state"):
        integrity_audit(load_runs(build_matrix(tmp_path, mutate)))


def test_ground_truth_drift_within_an_instance_fails(tmp_path):
    def mutate(doc, i):
        if doc["run_result"]["condition"] == "c3_mas":
            doc["run_result"]["score"]["solver_optimum"] = 99
    with pytest.raises(FormalPilotIntegrityError, match="optimum differs"):
        integrity_audit(load_runs(build_matrix(tmp_path, mutate)))


def test_a_token_ledger_mismatch_fails(tmp_path):
    def mutate(doc, i):
        if i == 3:
            doc["usage_audit"]["totals"]["endpoint_total_tokens"] = 1
    with pytest.raises(FormalPilotIntegrityError, match="internal != endpoint"):
        integrity_audit(load_runs(build_matrix(tmp_path, mutate)))


def test_a_think_leak_fails(tmp_path):
    def mutate(doc, i):
        if i == 11:
            doc["transcript_sanity"]["think_leak"] = True
    with pytest.raises(FormalPilotIntegrityError, match="think_leak"):
        integrity_audit(load_runs(build_matrix(tmp_path, mutate)))


def test_mixed_endpoints_are_recorded_not_rejected(tmp_path):
    def mutate(doc, i):
        if i % 2:
            doc["pilot_sweep"]["base_url"] = "http://localhost:8002/v1"
    audit = integrity_audit(load_runs(build_matrix(tmp_path, mutate)))
    assert len(audit["base_urls"]) == 2


def test_a_missing_expected_runs_manifest_fails(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    with pytest.raises(FormalPilotIntegrityError, match="expected-runs manifest not found"):
        integrity_audit(rows, tmp_path / "nope.json")


def test_a_tampered_manifest_fails_even_with_a_matching_stored_hash(tmp_path, monkeypatch):
    """The analyser recomputes rather than trusting the digest written in the file."""
    run_dir = build_matrix(tmp_path)
    rows = load_runs(run_dir)
    path = write_expected(run_dir, tmp_path / "expected.json", monkeypatch)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["runs"].append({"run_id": "smuggled__run"})  # stored content_hash left untouched
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FormalPilotIntegrityError, match="self-inconsistent"):
        integrity_audit(rows, path)


def test_a_manifest_that_is_not_the_pre_registered_one_fails(tmp_path):
    run_dir = build_matrix(tmp_path)
    rows = load_runs(run_dir)
    payload = {"runs": [{"run_id": r["run_id"]} for r in rows]}
    payload["content_hash"] = content_hash(payload)
    path = tmp_path / "other.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalPilotIntegrityError, match="pre-registered"):
        integrity_audit(rows, path)


# --------------------------------------------------------------------------- #
# Secondary tests kept as secondary.
# --------------------------------------------------------------------------- #
def test_sign_test_is_exact_and_excludes_ties():
    assert sign_test([1] * 6)["p_two_sided"] == pytest.approx(0.03125)
    out = sign_test([0, 0, 0, 1, 1, 1])
    assert (out["ties"], out["positive"]) == (3, 3)
    assert out["p_two_sided"] == pytest.approx(0.25)
    assert sign_test([0.0] * 30)["p_two_sided"] is None


def test_spearman_is_undefined_without_variation():
    assert spearman([1, 2, 3], [5, 5, 5]) is None
    assert spearman([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_rates_are_undefined_on_an_empty_denominator(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    at = [dict(r, attempts=0, parsed=0, valid=0, accepted=0, any_valid=False,
               **{f"attempts_{k}": 0 for k in ("react_step", "revise", "worker_a",
                                               "worker_b", "critic")})
          for r in rows if r["condition"] == "c1_react" and r["cap"] == 8000]
    g = aggregate(at)
    assert g["proposal_validity_rate"] is None
    assert g["by_role"] == {}
    assert g["share_runs_any_valid"] == 0.0


def test_aborted_and_budget_collapse_into_budget_limited(tmp_path):
    rows = load_runs(build_matrix(tmp_path))
    at = [dict(r, termination=t) for r, t in
          zip(rows[:4], ("aborted", "budget", "agent_finish", "max_steps"))]
    assert aggregate(at)["budget_limited"] == 2


# --------------------------------------------------------------------------- #
# Artifacts.
# --------------------------------------------------------------------------- #
def test_artifacts_are_written_and_byte_reproducible(tmp_path, monkeypatch):
    run_dir = build_matrix(tmp_path)
    exp = write_expected(run_dir, tmp_path / "expected.json", monkeypatch)
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    for out in (out_a, out_b):
        assert main(["--runs", str(run_dir), "--expected-runs", str(exp),
                     "--out", str(out)]) == 0
    for name in ("summary.json", "runs.csv", "report.md"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
    assert b"\r\n" not in (out_a / "runs.csv").read_bytes()
    assert b"\r\n" not in (out_a / "report.md").read_bytes()


def test_report_states_what_it_is_and_is_not(tmp_path, monkeypatch):
    run_dir = build_matrix(tmp_path)
    exp = write_expected(run_dir, tmp_path / "expected.json", monkeypatch)
    out = tmp_path / "out"
    main(["--runs", str(run_dir), "--expected-runs", str(exp), "--out", str(out)])
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "confirmatory" not in text.split("not a confirmatory")[0]
    assert "pilot, not a confirmatory test" in text
    assert "S_C1 − S_C3" in text
    assert "low-complexity sanity check fails" in text
    assert "diagnostic" in text
    assert "Feasibility is 1.000 everywhere" in text
    # A null interaction must never be written up as evidence of no interaction.
    assert "not a finding of no interaction" in text
    assert "does **not** establish that the difference is independent" in text
    assert "omnibus H" in text
    # The one-sided column must be labelled as testing the registered direction, and the
    # opposite direction must be named exploratory rather than quoted as evidence.
    assert "pre-registered decreasing-trend p" in text
    assert "The pre-registered alternative is delta FALLING with complexity" in text
    assert "exploratory" in text
    # The two commits must be distinguishable, not blurred into one provenance line.
    assert "Frozen behavioural code" in text and "Executing commit" in text


def test_refuses_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        main(["--runs", str(empty), "--out", str(tmp_path / "out")])
