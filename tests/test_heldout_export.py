# tests/test_heldout_export.py
"""Offline tests for the descriptive held-out export.

The export exists to be looked at before the analysis is read, so the tests check the
two things that would make it misleading: that it carries exactly the verified runs in
a stable shape, and that it stays descriptive -- no p-value, no decision, nothing that
could be mistaken for the confirmatory analysis.

No held-out document is read here. Every fixture is synthetic.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from importlib.util import find_spec
from pathlib import Path

import pytest

from scripts.export_heldout_results import (
    COLUMNS,
    DETAILS,
    SCHEMA,
    SQLITE_TABLE,
    ExportRefused,
    aggregate,
    build_row,
    collect,
    main,
)

MODEL = "Qwen/Qwen3-32B-AWQ"
MODEL_TAG = MODEL.replace("/", "-")
CAPS = (16000, 32000, 64000, 128000)
CONDS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
SUBSET_HASH, BINNING_HASH = "s" * 64, "b" * 64
FLOOR = 100_000


def subset_doc(n_people: int, base_seed: int, per_level: dict[str, int]):
    instances: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    seed = base_seed
    for level, count in per_level.items():
        for _ in range(count):
            instances[level].append({
                "instance_id": f"uniform-n{n_people}-t80-o50-s{seed}",
                "seed": seed, "level": level, "optimum": 3,
                "complexity_metric": 0 if level == "easy" else 7,
                "cell": {"n_people": n_people, "tightness": 0.8, "overlap": 0.5,
                         "travel_structure": "uniform"},
            })
            seed += 1
    return {"schema_version": "pilot_subset/1.0", "n_people": n_people,
            "manifest_hash": BINNING_HASH, "content_hash": "c" * 64,
            "instances": instances}


def run_doc(condition: str, iid: str, cap: int, satisfaction: float):
    rid = f"{condition}__{iid}__cap{cap}__{MODEL_TAG}"
    return {
        "schema_version": "harness_slice/1.0",
        "timestamp_utc": "2026-08-30T12:00:00+00:00",
        "finalization_reserve": 256,
        "seeds": {"inference": 42},
        "provenance": {"git_commit": "abc123def456"},
        "instance": {"n_people": 8},
        "oracle": {"optimum": 3},
        "transcript_sanity": {"think_leak": False},
        "usage_audit": {"per_call": [{"x": 1}], "delta": 0},
        "agent": {
            "termination": "agent_finish",
            "proposals": [{"valid": True}, {"valid": False}],
            "transcript": "a very long transcript " * 200,
        },
        "run_result": {
            "run_id": rid, "instance_id": iid, "condition": condition, "cap": cap,
            "model": MODEL, "seed": 42, "level": "easy",
            "score": {"satisfaction": satisfaction, "valid": True,
                      "n_valid_meetings": 2, "solver_optimum": 3},
            "tokens": {"total": 1234, "thinking": 900},
            "calls": [{"i": 0}, {"i": 1}, {"i": 2}],
            "final_plan": {"meetings": [{}, {}]},
            "best_plan_so_far": {"meetings": [{}, {}]},
            "sampling": {"temperature": 0.6},
            "latency_seconds": 12.5,
            "notes": None,
        },
        "pilot_sweep": {"subset_hash": SUBSET_HASH, "binning_hash": BINNING_HASH,
                        "git_commit": "abc123def456"},
    }


def write_world(tmp_path: Path, *, seed_base: int = FLOOR, per_band: int = 2):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    subsets, expected_runs = [], []

    docs = [(subset_doc(8, seed_base, {"easy": per_band, "medium": per_band,
                                       "hard": per_band}), 8)]
    for i, n in enumerate((4, 5, 6)):
        docs.append((subset_doc(n, seed_base + 10000 * (i + 1),
                                {"easy": 2, "medium": 0, "hard": 0}), n))

    for doc, n in docs:
        path = tmp_path / f"subset_n{n}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        subsets.append(path)
        for rows in doc["instances"].values():
            for e in rows:
                for cap in CAPS:
                    for cond in CONDS:
                        d = run_doc(cond, e["instance_id"], cap, 0.5)
                        (runs / f"{d['run_result']['run_id']}.json").write_text(
                            json.dumps(d), encoding="utf-8")
                        expected_runs.append({
                            "run_id": d["run_result"]["run_id"], "condition": cond,
                            "instance_id": e["instance_id"], "cap": cap,
                            "level": e["level"]})

    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({
        "schema_version": "expected_runs/1.0", "subset_hash": SUBSET_HASH,
        "binning_hash": BINNING_HASH, "model": MODEL, "runs": expected_runs,
    }), encoding="utf-8")
    return {"runs": runs, "subsets": subsets, "expected": expected,
            "n_runs": len(expected_runs)}


def cli(world, out: Path, extra: list[str] | None = None) -> list[str]:
    args = ["--runs", str(world["runs"]), "--expected", str(world["expected"]),
            "--out", str(out)]
    for s in world["subsets"]:
        args += ["--subset", str(s)]
    return args + (extra or [])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    world = write_world(tmp)
    out = tmp / "out"
    assert main(cli(world, out)) == 0
    with (out / "runs.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with (out / "aggregates.csv").open(encoding="utf-8", newline="") as fh:
        aggregates = list(csv.DictReader(fh))
    manifest = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
    with sqlite3.connect(out / "heldout_results.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        db_rows = [dict(r) for r in
                   conn.execute(f"SELECT * FROM {SQLITE_TABLE}")]
    conn.close()
    return {"world": world, "out": out, "rows": rows, "aggregates": aggregates,
            "manifest": manifest, "db_rows": db_rows}


# --------------------------------------------------------------------------- #
# Shape.
# --------------------------------------------------------------------------- #
def test_one_row_per_verified_run(exported):
    assert len(exported["rows"]) == exported["world"]["n_runs"]
    assert exported["manifest"]["n_runs"] == exported["world"]["n_runs"]
    assert len({r["run_id"] for r in exported["rows"]}) == len(exported["rows"])


def test_the_csv_header_is_exactly_the_declared_schema(exported):
    """The count is whatever SCHEMA says; what matters is that all three agree."""
    declared = [name for name, _type, _doc in SCHEMA]
    assert list(COLUMNS) == declared
    assert list(exported["rows"][0]) == declared
    assert [c["name"] for c in exported["manifest"]["schema"]] == declared
    assert exported["manifest"]["n_columns"] == len(declared)
    assert DETAILS in declared and "source_json" in declared


def test_every_row_names_a_source_document_that_exists(exported):
    """A row must be enough to find the full run, including what was not copied."""
    for row in exported["rows"]:
        source = Path(row["source_json"])
        assert source.exists(), source
        assert source.stem == row["run_id"]
        doc = json.loads(source.read_text(encoding="utf-8"))
        assert doc["run_result"]["run_id"] == row["run_id"]
        # the blocks the row does not carry are recoverable from it
        assert doc["agent"]["transcript"]
        assert len(doc["run_result"]["calls"]) == int(row["n_calls"])


def test_the_sqlite_table_holds_the_same_rows_as_the_csv(exported):
    """Same rows, same schema, one primary key per run."""
    csv_rows, db_rows = exported["rows"], exported["db_rows"]
    assert len(db_rows) == len(csv_rows)
    assert list(db_rows[0]) == [name for name, _t, _d in SCHEMA]
    by_id = {r["run_id"]: r for r in db_rows}
    assert len(by_id) == len(db_rows)
    for row in csv_rows:
        db = by_id[row["run_id"]]
        for name, _type, _doc in SCHEMA:
            assert str(db[name]) == row[name], (row["run_id"], name)


def test_re_exporting_does_not_append_a_second_copy(tmp_path):
    """The database is rebuilt, not added to."""
    world = write_world(tmp_path)
    out = tmp_path / "out"
    assert main(cli(world, out)) == 0
    assert main(cli(world, out)) == 0
    with sqlite3.connect(out / "heldout_results.sqlite") as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM {SQLITE_TABLE}").fetchone()[0]
    conn.close()
    assert count == world["n_runs"]


def test_the_row_order_is_deterministic(exported):
    ordered = sorted(exported["rows"],
                     key=lambda r: (r["condition"], int(r["cap"]), r["instance_id"]))
    assert [r["run_id"] for r in exported["rows"]] == [r["run_id"] for r in ordered]


def test_details_json_is_json_and_names_what_it_leaves_out(exported):
    details = json.loads(exported["rows"][0][DETAILS])
    assert details["n_proposals"] == 2 and details["n_valid_proposals"] == 1
    assert details["tokens"]["thinking"] == 900
    assert set(details["not_copied"]) == {"agent.transcript_and_proposal_records",
                                          "run_result.calls"}


def test_the_large_blocks_are_not_duplicated_into_every_row(exported):
    """Their absence must be visible in the manifest, not silent."""
    blob = exported["rows"][0][DETAILS]
    assert "a very long transcript" not in blob
    assert exported["manifest"]["details_json_excludes"]
    assert exported["rows"][0]["n_calls"] == "3"


def test_block_b_rows_carry_no_band(exported):
    for row in exported["rows"]:
        assert (row["band"] == "") == (row["block"] == "B")
        assert (row["block"] == "A") == (row["n_people"] == "8")


# --------------------------------------------------------------------------- #
# Aggregates: counts and averages only.
# --------------------------------------------------------------------------- #
def test_the_four_requested_groupings_are_present_plus_the_cross(exported):
    groupings = {a["grouping"] for a in exported["aggregates"]}
    assert groupings == {"condition", "cap", "band", "n_people",
                         "condition x cap x stratum"}


def test_each_grouping_accounts_for_every_run(exported):
    total = exported["world"]["n_runs"]
    per_grouping = {}
    for a in exported["aggregates"]:
        per_grouping.setdefault(a["grouping"], 0)
        per_grouping[a["grouping"]] += int(a["n_runs"])
    assert per_grouping["condition"] == total
    assert per_grouping["cap"] == total
    assert per_grouping["n_people"] == total
    assert per_grouping["condition x cap x stratum"] == total
    # The band grouping covers Block A only, which is what a D band means.
    block_a = sum(1 for r in exported["rows"] if r["block"] == "A")
    assert per_grouping["band"] == block_a


def test_aggregates_are_descriptive_statistics_only():
    rows = [{"condition": "c1_react", "cap": 16000, "block": "A", "band": "low",
             "n_people": 8, "satisfaction": 1.0, "valid": True, "n_meetings": 2,
             "tokens_total": 100, "n_calls": 3, "termination": "agent_finish"},
            {"condition": "c1_react", "cap": 16000, "block": "A", "band": "low",
             "n_people": 8, "satisfaction": 0.0, "valid": False, "n_meetings": 0,
             "tokens_total": 200, "n_calls": 5, "termination": "budget"}]
    out = aggregate(rows)
    by_condition = next(a for a in out if a["grouping"] == "condition")
    assert by_condition["n_runs"] == 2
    assert by_condition["mean_satisfaction"] == 0.5
    assert by_condition["valid_rate"] == 0.5
    assert by_condition["terminations"] == {"agent_finish": 1, "budget": 1}


def test_nothing_in_the_output_looks_like_a_test_result(exported):
    """The export must not be mistakable for the confirmatory analysis."""
    banned = ("p_value", "p_raw", "p_holm", "holm", "reject", "crossover", "iut",
              "confirmatory", "significan")
    surface = " ".join(exported["aggregates"][0]).lower() + " " + \
              " ".join(list(exported["rows"][0])).lower()
    for token in banned:
        assert token not in surface
    assert "no hypothesis test" in exported["manifest"]["standing"].lower()


# --------------------------------------------------------------------------- #
# Refusals.
# --------------------------------------------------------------------------- #
def test_an_incomplete_sweep_is_refused(tmp_path):
    world = write_world(tmp_path)
    next(world["runs"].glob("*.json")).unlink()
    with pytest.raises(ExportRefused, match="completeness audit failed"):
        main(cli(world, tmp_path / "out"))
    assert not (tmp_path / "out" / "runs.csv").exists()


def test_a_development_seed_is_refused(tmp_path):
    world = write_world(tmp_path, seed_base=30000)
    with pytest.raises(ExportRefused, match="below the held-out floor"):
        main(cli(world, tmp_path / "out"))


def test_an_instance_missing_from_the_subsets_is_refused(tmp_path):
    world = write_world(tmp_path)
    expected = json.loads(world["expected"].read_text(encoding="utf-8"))
    with pytest.raises(ExportRefused, match="in no subset manifest"):
        collect([{**expected["runs"][0], "instance_id": "ghost-n8-t80-o50-s199999",
                  "model": MODEL}], world["runs"], {}, {})


# --------------------------------------------------------------------------- #
# Parquet is optional, and its absence is reported rather than hidden.
# --------------------------------------------------------------------------- #
def test_the_parquet_status_is_always_recorded(exported):
    status = exported["manifest"]["parquet"]
    assert status.startswith("written") or status.startswith("not written")
    if status.startswith("not written"):
        assert not (exported["out"] / "runs.parquet").exists()
    else:
        assert (exported["out"] / "runs.parquet").exists()


def test_require_parquet_makes_a_missing_engine_fatal(tmp_path):
    world = write_world(tmp_path)
    out = tmp_path / "out"
    engine_available = all(find_spec(m) is not None
                           for m in ("pandas", "pyarrow"))
    if not engine_available:
        with pytest.raises(ExportRefused, match="not written"):
            main(cli(world, out, ["--require-parquet"]))
        return
    assert main(cli(world, out, ["--require-parquet"])) == 0
    assert (out / "runs.parquet").exists()


def test_build_row_keeps_the_frozen_structural_facts():
    info = {"instance_id": "uniform-n8-t80-o50-s100000", "seed": 100000,
            "n_people": 8, "block": "A", "level": "hard", "band": "high",
            "optimum": 3, "complexity_metric": 14, "tightness": 0.8, "overlap": 0.5,
            "travel_structure": "uniform", "subset": "subset_n8.json"}
    row = build_row(run_doc("c3_mas", info["instance_id"], 64000, 0.6667), info, 2,
                    "results/logs/heldout/some_run.json")
    assert row["band"] == "high" and row["conflict_pairs"] == 14
    assert row["source_json"] == "results/logs/heldout/some_run.json"
    assert row["valid"] == 1          # 0/1, so the CSV and the table read alike
    assert row["satisfaction"] == 0.6667 and row["n_meetings"] == 2
    details = json.loads(row[DETAILS])
    assert details["higher_order_gap_H"] == 2
    assert details["travel_structure"] == "uniform"
    assert details["cap_utilisation"] == round(1234 / 64000, 4)
