# tests/test_audit_expected_runs.py
"""Offline tests for the completeness audit.

The audit's whole value is that it fails when a file count would pass, so the tests are
mostly ways of being incomplete that `ls | wc -l` cannot see: a missing run, a truncated
document, a result from a different frozen selection, a stray file from another sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_expected_runs import AuditFailed, audit, load_expected, main

MODEL = "Qwen/Qwen3-32B-AWQ"
SUBSET, BINNING = "a" * 64, "b" * 64


def run_id(condition, instance_id, cap):
    return f"{condition}__{instance_id}__cap{cap}__{MODEL.replace('/', '-')}"


def expected_doc(instances, caps=(8000, 16000), conditions=("c1_react", "c3_mas")):
    runs = [{"run_id": run_id(c, i, cap), "condition": c, "instance_id": i, "cap": cap,
             "level": "easy"}
            for c in conditions for cap in caps for i in instances]
    return {"schema_version": "expected_runs/1.0", "subset_hash": SUBSET,
            "binning_hash": BINNING, "model": MODEL, "caps": list(caps),
            "conditions": list(conditions), "n_expected_runs": len(runs), "runs": runs}


def write_expected(path: Path, instances, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected_doc(instances, **kw)), encoding="utf-8")
    return path


def write_result(runs_dir: Path, r, *, subset=SUBSET, binning=BINNING, schema="run/1.0",
                 truncate=False, **overrides):
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{r['run_id']}.json"
    if truncate:
        path.write_text('{"schema_version": "run/1.0", "run_res', encoding="utf-8")
        return path
    doc = {
        "schema_version": schema,
        "run_result": {
            "run_id": r["run_id"], "instance_id": r["instance_id"],
            "condition": r["condition"], "cap": r["cap"],
            "model": MODEL, "level": r["level"], "seed": 1, "satisfaction": 0.5,
            "achieved_reward": 1, "oracle_optimum": 3, "feasible": True,
            "termination": "agent_finish", **overrides,
        },
        "pilot_sweep": {"subset_hash": subset, "binning_hash": binning},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def schema_version(monkeypatch):
    """Pin the document schema the audit demands, so these tests do not break when the
    harness's own version moves."""
    import scripts.run_pilot_sweep as rps
    monkeypatch.setattr(rps, "DOCUMENT_SCHEMA_VERSION", "run/1.0")
    monkeypatch.setattr(rps, "_REQUIRED_RUN_RESULT_FIELDS",
                        ("run_id", "instance_id", "condition", "cap"))
    return "run/1.0"


def complete_setup(tmp_path, schema_version, instances=("i1", "i2")):
    exp = write_expected(tmp_path / "m" / "expected.json", instances)
    runs = tmp_path / "runs"
    for r in json.loads(exp.read_text())["runs"]:
        write_result(runs, r)
    return [exp], runs


# --------------------------------------------------------------------------- #
# The happy path.
# --------------------------------------------------------------------------- #
def test_a_complete_sweep_passes(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    rep = audit(load_expected(exp), runs)
    assert rep["all_checks_passed"] is True
    assert rep["n_verified"] == rep["n_expected"] == 8
    assert rep["n_missing"] == rep["n_incompatible"] == rep["n_unexpected_files"] == 0


def test_the_report_breaks_down_by_condition_and_cap(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    rep = audit(load_expected(exp), runs)
    assert rep["verified_by_condition_and_cap"] == {
        "c1_react@8000": 2, "c1_react@16000": 2,
        "c3_mas@8000": 2, "c3_mas@16000": 2,
    }


def test_several_manifests_are_pooled(tmp_path, schema_version):
    a = write_expected(tmp_path / "m" / "a.json", ("i1",))
    b = write_expected(tmp_path / "m" / "b.json", ("i2",))
    runs = tmp_path / "runs"
    for p in (a, b):
        for r in json.loads(p.read_text())["runs"]:
            write_result(runs, r)
    rep = audit(load_expected([a, b]), runs)
    assert rep["all_checks_passed"] is True
    assert rep["n_expected"] == 8
    assert set(rep["verified_by_source_manifest"]) == {"a.json", "b.json"}


# --------------------------------------------------------------------------- #
# Ways of being incomplete that a file count cannot see.
# --------------------------------------------------------------------------- #
def test_one_missing_run_fails(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    next(runs.glob("*.json")).unlink()
    rep = audit(load_expected(exp), runs)
    assert rep["all_checks_passed"] is False
    assert rep["n_missing"] == 1


def test_a_truncated_document_fails(tmp_path, schema_version):
    exp = write_expected(tmp_path / "m" / "expected.json", ("i1", "i2"))
    runs = tmp_path / "runs"
    rows = json.loads(exp.read_text())["runs"]
    for r in rows[1:]:
        write_result(runs, r)
    write_result(runs, rows[0], truncate=True)
    rep = audit(load_expected([exp]), runs)
    assert rep["all_checks_passed"] is False
    assert rep["n_incompatible"] == 1
    assert rep["n_missing"] == 0          # the file exists; a count would have passed


def test_a_result_from_another_frozen_selection_fails(tmp_path, schema_version):
    """The failure the hashes exist to catch: right shape, wrong dataset."""
    exp = write_expected(tmp_path / "m" / "expected.json", ("i1", "i2"))
    runs = tmp_path / "runs"
    rows = json.loads(exp.read_text())["runs"]
    for r in rows[1:]:
        write_result(runs, r)
    write_result(runs, rows[0], subset="c" * 64)
    rep = audit(load_expected([exp]), runs)
    assert rep["all_checks_passed"] is False
    assert rep["n_incompatible"] == 1
    assert any("subset_hash" in m for m in rep["incompatible"])


def test_a_cap_mismatch_inside_the_document_fails(tmp_path, schema_version):
    exp = write_expected(tmp_path / "m" / "expected.json", ("i1",))
    runs = tmp_path / "runs"
    rows = json.loads(exp.read_text())["runs"]
    for r in rows[1:]:
        write_result(runs, r)
    write_result(runs, rows[0], cap=99999)
    rep = audit(load_expected([exp]), runs)
    assert rep["n_incompatible"] == 1


def test_a_stray_file_from_another_sweep_fails(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    (runs / "c5_best_of_3__other__cap64000__Qwen-Qwen3-32B-AWQ.json").write_text(
        "{}", encoding="utf-8")
    rep = audit(load_expected(exp), runs)
    assert rep["all_checks_passed"] is False
    assert rep["n_unexpected_files"] == 1


def test_a_run_id_in_two_manifests_is_refused(tmp_path, schema_version):
    a = write_expected(tmp_path / "m" / "a.json", ("i1",))
    b = write_expected(tmp_path / "m" / "b.json", ("i1",))
    with pytest.raises(AuditFailed, match="more than one"):
        load_expected([a, b])


def test_an_empty_manifest_is_refused(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"subset_hash": SUBSET, "binning_hash": BINNING,
                             "model": MODEL, "runs": []}), encoding="utf-8")
    with pytest.raises(AuditFailed, match="no runs listed"):
        load_expected([p])


# --------------------------------------------------------------------------- #
# The audit decides nothing about the experiment.
# --------------------------------------------------------------------------- #
def test_the_audit_reads_no_outcome():
    import tokenize
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_expected_runs.py"
    with path.open("rb") as fh:
        code = "".join(tok.string for tok in tokenize.tokenize(fh.readline)
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING)).lower()
    for banned in ("satisfaction", "achieved_reward", "mean", "threshold", "gate"):
        assert banned not in code, f"the completeness audit reads {banned!r}"


# --------------------------------------------------------------------------- #
# Exit codes -- the audit is meant to be usable as a gate in a shell.
# --------------------------------------------------------------------------- #
def test_the_cli_exits_zero_when_complete(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    out = tmp_path / "audit.json"
    assert main(["--expected", str(exp[0]), "--runs", str(runs), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["all_checks_passed"] is True


def test_the_cli_exits_nonzero_when_incomplete(tmp_path, schema_version):
    exp, runs = complete_setup(tmp_path, schema_version)
    next(runs.glob("*.json")).unlink()
    assert main(["--expected", str(exp[0]), "--runs", str(runs)]) == 1
