# tests/test_pilot_export.py
"""Offline tests for the thesis-facing pilot export.

The export exists so that data collected under a superseded complexity axis cannot be
silently plotted as if it came from the current one. So the tests check the arithmetic, and
they check that both warnings survive in the artifacts.
"""

from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

import pytest

from scripts.export_pilot_dataset import (
    EXPECTED_INSTANCES,
    EXPECTED_RUNS,
    PilotExportError,
    calls_table,
    instances_table,
    load,
    main,
    proposals_table,
)


def doc(instance="inst-a", condition="c1_react", cap=32000, n_people=8, pairs=7,
        level="medium", optimum=4, proposals=None, calls=None, seed=10005):
    return {
        "run_result": {
            "run_id": f"{condition}__{instance}__cap{cap}__M",
            "instance_id": instance, "condition": condition, "cap": cap,
            "level": level, "seed": seed,
            "score": {"solver_optimum": optimum, "satisfaction": 0.5,
                      "n_valid_meetings": 2, "valid": True, "optimality": False,
                      "invalid_reasons": []},
            "calls": calls if calls is not None else [
                {"role": "react_step", "input_tokens": 100, "thinking_tokens": 50,
                 "answer_tokens": 10, "finish_reason": "stop", "context_limited": False,
                 "requested_max_tokens": 900, "effective_max_tokens": 900,
                 "latency_seconds": 1.2}],
        },
        "agent": {"proposals": proposals if proposals is not None else [
            {"role": "react_step", "step": 3, "n_meetings": 2, "parsed": True,
             "valid": False, "accepted_into_best_plan": False,
             "reasons": ["travel_infeasible", "window_violation"]}]},
        "instance": {"complexity_metric": pairs,
                     "generator_params": {"n_people": n_people, "tightness": 1.0,
                                          "overlap": 0.2, "travel_structure": "uniform"}},
    }


def archive(tmp_path, docs):
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    for i, doc_ in enumerate(docs):
        (d / f"{i:04d}.json").write_text(json.dumps(doc_), encoding="utf-8")
    return d


def full_archive(tmp_path):
    """A structurally valid archive: 30 instances x 12 cells."""
    docs = []
    for i in range(EXPECTED_INSTANCES):
        for cond in ("c1_react", "c2_verify_revise", "c3_mas"):
            for cap in (8000, 16000, 32000, 64000):
                docs.append(doc(instance=f"inst-{i:02d}", condition=cond, cap=cap,
                                pairs=i % 15, seed=10005 + i))
    return archive(tmp_path, docs)


# --------------------------------------------------------------------------- #
# The archive must be the frozen pilot.
# --------------------------------------------------------------------------- #
def test_a_short_archive_is_refused(tmp_path):
    with pytest.raises(PilotExportError, match=f"expected {EXPECTED_RUNS} run documents"):
        load(archive(tmp_path, [doc()]))


def test_a_wrong_instance_count_is_refused(tmp_path):
    docs = [doc(instance="only-one", cap=c) for c in range(EXPECTED_RUNS)]
    with pytest.raises(PilotExportError, match="distinct instances"):
        load(archive(tmp_path, docs))


def test_inconsistent_instance_facts_are_refused(tmp_path):
    """The same instance must look identical in every run it appears in."""
    docs = [doc(instance=f"inst-{i:02d}", cap=c) for i in range(EXPECTED_INSTANCES)
            for c in (8000, 16000, 32000, 64000)] * 3
    docs[5] = doc(instance=docs[5]["run_result"]["instance_id"], pairs=99)
    with pytest.raises(PilotExportError, match="instance facts differ"):
        instances_table(docs)


# --------------------------------------------------------------------------- #
# The retrospective density.
# --------------------------------------------------------------------------- #
def test_retrospective_density_normalises_by_the_instance_size():
    """The whole point of the normalisation: comparable across the pilot's n in {4,6,8}."""
    rows = instances_table([doc(instance="a", n_people=8, pairs=7),
                            doc(instance="b", n_people=4, pairs=3)])
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["a"]["D_retrospective"] == pytest.approx(7 / comb(8, 2), abs=1e-6)
    assert by_id["b"]["D_retrospective"] == pytest.approx(3 / comb(4, 2), abs=1e-6)


def test_the_old_level_is_carried_under_a_name_that_says_so():
    rows = instances_table([doc(level="hard")])
    assert rows[0]["level_old_binning"] == "hard"
    assert "level" not in rows[0]        # the bare name must not survive


# --------------------------------------------------------------------------- #
# The two views that exist nowhere else.
# --------------------------------------------------------------------------- #
def test_proposals_keep_the_role_that_produced_them():
    """Pooling worker and critic attempts was the error this table prevents."""
    d = doc(proposals=[
        {"role": "worker_a", "step": 2, "n_meetings": 3, "parsed": True, "valid": True,
         "accepted_into_best_plan": True, "reasons": []},
        {"role": "critic", "step": 9, "n_meetings": 4, "parsed": True, "valid": True,
         "accepted_into_best_plan": True, "reasons": []}])
    rows = proposals_table([d])
    assert [r["role"] for r in rows] == ["worker_a", "critic"]
    assert [r["attempt_index"] for r in rows] == [0, 1]


def test_rejection_reasons_survive_as_a_multi_label_field():
    rows = proposals_table([doc()])
    assert rows[0]["reasons"] == "travel_infeasible;window_violation"


def test_calls_carry_the_token_split_and_the_context_clamp():
    rows = calls_table([doc()])
    r = rows[0]
    assert (r["input_tokens"], r["thinking_tokens"], r["answer_tokens"]) == (100, 50, 10)
    assert r["finish_reason"] == "stop" and r["context_limited"] is False
    assert r["role"] == "react_step" and r["latency_seconds"] == 1.2


def test_an_empty_table_is_refused_rather_than_written(tmp_path):
    from scripts.export_pilot_dataset import write_csv
    with pytest.raises(PilotExportError, match="empty table"):
        write_csv(tmp_path / "x.csv", [])


# --------------------------------------------------------------------------- #
# The warnings must reach the artifacts.
# --------------------------------------------------------------------------- #
def test_the_readme_states_both_warnings(tmp_path):
    out, readme = tmp_path / "out", tmp_path / "README.md"
    assert main(["--runs", str(full_archive(tmp_path)), "--out", str(out),
                 "--readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "`level` here is the OLD binning" in text
    assert "not comparable" in text
    assert "`D_retrospective` is descriptive" in text
    assert "not evidence for anything" in text
    # and the practical instruction that protects the held-out range
    assert "Do not reuse these 30 instances" in text


def test_the_tables_are_written_and_byte_reproducible(tmp_path):
    runs = full_archive(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        main(["--runs", str(runs), "--out", str(out), "--readme", str(out / "R.md")])
    for name in ("pilot_instances.csv", "pilot_proposals.csv", "pilot_calls.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
        assert b"\r\n" not in (a / name).read_bytes()
    rows = list(csv.DictReader((a / "pilot_instances.csv").open(encoding="utf-8")))
    assert len(rows) == EXPECTED_INSTANCES


def test_the_export_does_not_touch_the_archive(tmp_path):
    runs = full_archive(tmp_path)
    before = {p.name: p.read_bytes() for p in runs.glob("*.json")}
    main(["--runs", str(runs), "--out", str(tmp_path / "o"),
          "--readme", str(tmp_path / "R.md")])
    after = {p.name: p.read_bytes() for p in runs.glob("*.json")}
    assert before == after
