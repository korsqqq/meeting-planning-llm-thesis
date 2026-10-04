# scripts/export_heldout_results.py
"""Flatten the held-out runs into one descriptive table. CPU, no LLM, no inference.

Writes the same rows three ways -- `runs.csv`, `runs.parquet` (when an engine is
present) and `runs` in `heldout_results.sqlite` -- from ONE schema declaration, so
the three cannot drift apart.

    .venv-harness/bin/python -m scripts.export_heldout_results \
        --runs results/logs/heldout \
        --expected results/manifests/expected_runs__bands_held_out_n8.json \
        --expected results/manifests/expected_runs__heldout_n4.json \
        --expected results/manifests/expected_runs__heldout_n5.json \
        --expected results/manifests/expected_runs__heldout_n6.json \
        --subset results/manifests/subset__bands_held_out_n8__278ce0f2e8fe.json \
        --subset results/manifests/subset__heldout_n4__54f3f48d6232.json \
        --subset results/manifests/subset__heldout_n5__4b91dee07c34.json \
        --subset results/manifests/subset__heldout_n6__44418ababc22.json \
        --pool results/calibration/heldout_n8/candidates.csv \
        --out results/exports/heldout

To also write `runs.parquet`, layer a parquet engine on ephemerally -- pandas is in
`requirements/harness.lock.txt` but pyarrow is not, and the lock hash the HPC
provenance record depends on must not move. `--with` layers onto the ACTIVE
environment, so the venv is named through VIRTUAL_ENV and the interpreter is plain
`python`; pointing `uv run` at `.venv-harness/bin/python` directly would put pyarrow
somewhere that interpreter never looks (checked, not assumed):

    VIRTUAL_ENV=$PWD/.venv-harness uv run --with pyarrow \
        python -m scripts.export_heldout_results ...

This is the same pattern `scripts/hpc/bootstrap.sh` already uses to run inside a venv
without activating it.

WHAT THIS IS, AND IS NOT. A data export. It reads every verified run document and
writes one row per run, plus simple counts and averages. **It performs no hypothesis
test, computes no p-value, applies no correction, decides nothing and interprets
nothing.** The confirmatory analysis lives in `scripts/analyse_heldout.py` under the
plan frozen 2026-08-27; nothing here may be quoted as a result of that analysis, and
an ordering visible in these tables is a description of the sample, not a finding.

WHY IT EXISTS SEPARATELY. Looking at the data before reading a test is only safe when
the test was fixed in advance, which it was. Keeping the two in different files keeps
that separation visible: this script cannot express a decision even by accident,
because it contains nothing that could make one.

THE SCHEMA is declared once, in `SCHEMA` below: a name, a SQLite type and a
one-line meaning for every column. The CSV header, the SQLite DDL and the export
manifest are all generated from it, and the column count is whatever that
declaration says rather than a number stated in prose. `details_json` is a column
in it like any other and carries the rest of each document's small fields, so a
follow-up question does not need a second pass over disk.

WHAT `details_json` DOES NOT CARRY, and why. Two blocks are large and are left in the
run documents rather than duplicated into every row: the agent block (transcript and
per-proposal records, ~9 KB per run) and `run_result.calls` (per-call accounting,
~4 KB). Their sizes and counts are recorded instead -- `n_calls` is a column,
proposal counts are in `details_json` -- so their absence is visible rather than
silent.

EVERY ROW NAMES ITS SOURCE. `source_json` holds the path of the document the row
was built from, relative to the repository root where possible, and `run_id` is its
stem. The full run -- transcript, per-call accounting, everything not copied -- is
therefore always one `json.load` away from any row, in the CSV, the parquet and the
SQLite table alike. The run documents remain the source of truth.

THE BAND LABELS come from `scripts/analyse_heldout.py` rather than being restated
here, so the export and the analysis can never disagree about which instances are
Low, Medium or High.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyse_heldout import (  # noqa: E402
    HELD_OUT_SEED_FLOOR,
    LEVEL_TO_BAND,
    load_higher_order_gap,
    load_instances,
)
from scripts.audit_expected_runs import audit, load_expected  # noqa: E402
from scripts.run_pilot_sweep import result_path  # noqa: E402

SCHEMA_VERSION = "heldout_export/1.0"

# THE SCHEMA. One declaration, three outputs: the CSV header, the SQLite DDL and the
# manifest are all generated from it, so a column can never exist in one and not the
# others. (name, sqlite type, meaning). Anything not named here travels in
# `details_json`.
SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("run_id", "TEXT", "run identity and the stem of the source document"),
    ("condition", "TEXT", "c1_react | c2_verify_revise | c3_mas | c4_planner_critic "
                          "| c5_best_of_3"),
    ("cap", "INTEGER", "token budget cap of this run"),
    ("instance_id", "TEXT", "frozen instance the run was made on"),
    ("block", "TEXT", "A = the n=8 density design, B = the n=4/5/6 sizes"),
    ("band", "TEXT", "low | medium | high for Block A; empty for Block B, which is "
                     "not a density band"),
    ("n_people", "INTEGER", "generator parameter, never a complexity level"),
    ("optimum", "INTEGER", "oracle optimum of the instance, the denominator of "
                           "satisfaction"),
    ("conflict_pairs", "INTEGER", "binding conflict pairs of the instance"),
    ("satisfaction", "REAL", "achieved objective / oracle optimum; 0 if the plan is "
                             "invalid"),
    ("valid", "INTEGER", "1 if the final plan passed the validator, else 0"),
    ("n_meetings", "INTEGER", "meetings in the final plan"),
    ("tokens_total", "INTEGER", "all tokens booked to this run, thinking included"),
    ("n_calls", "INTEGER", "LLM calls; the per-call records stay in the document"),
    ("termination", "TEXT", "how the run ended, as the agent reported it"),
    ("source_json", "TEXT", "path of the run document this row was built from, "
                            "relative to the repository root where possible"),
    ("details_json", "TEXT", "the document's remaining small fields, as JSON"),
)
COLUMNS = tuple(name for name, _type, _doc in SCHEMA)
DETAILS = "details_json"
SQLITE_TABLE = "runs"

# Blocks deliberately not copied into every row. Recorded here so the exclusion is
# documented in the code that performs it, not only in prose.
NOT_COPIED = {
    "agent.transcript_and_proposal_records": "large; counts kept in details_json",
    "run_result.calls": "large; n_calls kept as a column",
}


class ExportRefused(RuntimeError):
    """The inputs are not the verified held-out set. Never worked around."""


def _plan_size(plan: Any) -> int:
    return len(plan.get("meetings") or []) if isinstance(plan, dict) else 0


# --------------------------------------------------------------------------- #
# Rows.
# --------------------------------------------------------------------------- #
def source_reference(path: Path) -> str:
    """The document's path, relative to the repository root where it lies inside it.

    Relative so the export stays readable after it is copied off the cluster;
    absolute otherwise, because an unresolvable reference is worse than a long one.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_row(doc: dict[str, Any], info: dict[str, Any], gap: int | None,
              source: str) -> dict[str, Any]:
    """One run document plus its frozen structural facts -> one flat row."""
    rr = doc["run_result"]
    score = rr.get("score") or {}
    tokens = rr.get("tokens") or {}
    agent = doc.get("agent") or {}
    proposals = agent.get("proposals") or []
    cap = rr["cap"]

    row = {
        "run_id": rr["run_id"],
        "condition": rr["condition"],
        "cap": cap,
        "instance_id": rr["instance_id"],
        "block": info["block"],
        "band": info["band"] or "",
        "n_people": info["n_people"],
        "optimum": info["optimum"],
        "conflict_pairs": info["complexity_metric"],
        "satisfaction": score.get("satisfaction"),
        # 0/1 rather than a bool: SQLite has no boolean type, and the CSV and the
        # table must read identically.
        "valid": int(bool(score.get("valid"))),
        "n_meetings": _plan_size(rr.get("final_plan")),
        "tokens_total": tokens.get("total"),
        "n_calls": len(rr.get("calls") or []),
        "termination": agent.get("termination"),
        "source_json": source,
    }

    details = {
        "seed": info["seed"],
        "level_label": info["level"],
        "model": rr.get("model"),
        "tightness": info["tightness"],
        "overlap": info["overlap"],
        "travel_structure": info["travel_structure"],
        "higher_order_gap_H": gap,
        "subset_manifest": info["subset"],
        "score": score,
        "tokens": tokens,
        "cap_utilisation": (round(tokens["total"] / cap, 4)
                            if tokens.get("total") is not None and cap else None),
        "latency_seconds": rr.get("latency_seconds"),
        "sampling": rr.get("sampling"),
        "final_plan": rr.get("final_plan"),
        "best_plan_so_far": rr.get("best_plan_so_far"),
        "n_proposals": len(proposals),
        "n_valid_proposals": sum(1 for p in proposals if p.get("valid")),
        "notes": rr.get("notes"),
        "run_timestamp": rr.get("timestamp") or doc.get("timestamp_utc"),
        "document_schema_version": doc.get("schema_version"),
        "finalization_reserve": doc.get("finalization_reserve"),
        "seeds": doc.get("seeds"),
        "provenance": doc.get("provenance"),
        "oracle": doc.get("oracle"),
        "instance_meta": doc.get("instance"),
        "transcript_sanity": doc.get("transcript_sanity"),
        "usage_audit_keys": sorted((doc.get("usage_audit") or {}).keys()),
        "pilot_sweep": doc.get("pilot_sweep"),
        "not_copied": NOT_COPIED,
    }
    row[DETAILS] = json.dumps(details, ensure_ascii=False, sort_keys=True)
    return row


def collect(expected: Sequence[dict[str, Any]], runs_dir: Path,
            instances: dict[str, dict[str, Any]], gaps: dict[str, int]
            ) -> list[dict[str, Any]]:
    """Every expected run, in a deterministic order, joined to its frozen facts."""
    rows: list[dict[str, Any]] = []
    for item in sorted(expected, key=lambda r: (r["condition"], r["cap"],
                                                r["instance_id"])):
        info = instances.get(item["instance_id"])
        if info is None:
            raise ExportRefused(
                f"{item['instance_id']} is in an expected-runs manifest but in no "
                "subset manifest; the export would carry a run whose structural facts "
                "are unknown")
        path = result_path(runs_dir, item["condition"], item["instance_id"],
                           item["cap"], item["model"])
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows.append(build_row(doc, info, gaps.get(item["instance_id"]),
                              source_reference(path)))
    return rows


# --------------------------------------------------------------------------- #
# Aggregates. Counts and averages -- no test, no interval, no decision.
# --------------------------------------------------------------------------- #
def _stratum(row: dict[str, Any]) -> str:
    return f"band={row['band']}" if row["block"] == "A" else f"n={row['n_people']}"


def describe_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sats = [r["satisfaction"] for r in rows if r["satisfaction"] is not None]
    toks = [r["tokens_total"] for r in rows if r["tokens_total"] is not None]
    return {
        "n_runs": len(rows),
        "mean_satisfaction": round(statistics.fmean(sats), 4) if sats else None,
        "median_satisfaction": round(statistics.median(sats), 4) if sats else None,
        "sd_satisfaction": (round(statistics.stdev(sats), 4)
                            if len(sats) > 1 else None),
        "min_satisfaction": round(min(sats), 4) if sats else None,
        "max_satisfaction": round(max(sats), 4) if sats else None,
        "valid_rate": (round(sum(1 for r in rows if r["valid"]) / len(rows), 4)
                       if rows else None),
        "nonempty_validated_rate": (
            round(sum(1 for r in rows if r["valid"] and r["n_meetings"] >= 1)
                  / len(rows), 4) if rows else None),
        "mean_tokens": round(statistics.fmean(toks)) if toks else None,
        "median_tokens": round(statistics.median(toks)) if toks else None,
        "mean_calls": round(statistics.fmean([r["n_calls"] for r in rows]), 2)
        if rows else None,
        "mean_meetings": round(statistics.fmean([r["n_meetings"] for r in rows]), 2)
        if rows else None,
        "terminations": dict(sorted(Counter(r["termination"] for r in rows).items())),
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Five simple groupings: the four asked for, plus the obvious cross of them."""
    groupings: list[tuple[str, Any]] = [
        ("condition", lambda r: (r["condition"], "", "")),
        ("cap", lambda r: ("", r["cap"], "")),
        ("band", lambda r: ("", "", f"band={r['band']}") if r["block"] == "A" else None),
        ("n_people", lambda r: ("", "", f"n={r['n_people']}")),
        ("condition x cap x stratum",
         lambda r: (r["condition"], r["cap"], _stratum(r))),
    ]
    out: list[dict[str, Any]] = []
    for name, key_of in groupings:
        buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = key_of(row)
            if key is not None:
                buckets[key].append(row)
        for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
            condition, cap, stratum = key
            out.append({"grouping": name, "condition": condition, "cap": cap,
                        "stratum": stratum, **describe_group(buckets[key])})
    return out


# --------------------------------------------------------------------------- #
# Writing.
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: Sequence[dict[str, Any]],
              fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False)
                                 if isinstance(v, (dict, list)) else v)
                             for k, v in row.items() if k in set(fields)})


def write_sqlite(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the `runs` table: the same rows as runs.csv, from the same SCHEMA.

    Standard-library sqlite3, no dependency added. The file is rebuilt from
    scratch on every run so a re-export cannot append a second copy of the sweep,
    and `run_id` is the primary key so a duplicate row is an error rather than a
    silently doubled count.
    """
    path.unlink(missing_ok=True)
    columns = ", ".join(
        f'{name} {sql_type}' + (' PRIMARY KEY' if name == 'run_id' else '')
        for name, sql_type, _doc in SCHEMA)
    placeholders = ", ".join("?" for _ in SCHEMA)
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE {SQLITE_TABLE} ({columns})')
        conn.executemany(
            f'INSERT INTO {SQLITE_TABLE} VALUES ({placeholders})',
            [tuple(row[name] for name in COLUMNS) for row in rows])
    conn.close()


def write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write runs.parquet if an engine is available. Returns a status string.

    pandas is in the frozen harness lock; a parquet engine is not. Rather than add
    one -- which would move the lock hash the HPC provenance record depends on -- the
    absence is reported with the command that supplies it ephemerally.
    """
    try:
        import pandas as pd
    except ImportError:
        return "not written: pandas is not importable"
    try:
        pd.DataFrame(list(rows)).to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:
        return (f"not written: no parquet engine ({exc.__class__.__name__}). "
                "Re-run as `VIRTUAL_ENV=$PWD/.venv-harness uv run --with pyarrow "
                "python -m scripts.export_heldout_results ...` to produce it.")
    return f"written: {path}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--expected", action="append", type=Path, required=True)
    ap.add_argument("--subset", action="append", type=Path, required=True)
    ap.add_argument("--pool", action="append", type=Path, default=None,
                    help="structural pool CSV; supplies higher_order_gap_H if given")
    ap.add_argument("--out", type=Path, default=Path("results/exports/heldout"))
    ap.add_argument("--require-parquet", action="store_true",
                    help="fail if no parquet engine is available")
    args = ap.parse_args(argv)

    # The same precondition the analysis uses: a partial sweep still exports readable
    # rows, and a reader cannot tell the difference from the file.
    expected = load_expected(args.expected)
    audit_report = audit(expected, args.runs)
    if not audit_report["all_checks_passed"]:
        raise ExportRefused(
            f"completeness audit failed: {audit_report['n_missing']} missing, "
            f"{audit_report['n_incompatible']} incompatible, "
            f"{audit_report['n_unexpected_files']} unexpected. Nothing is exported.")

    instances = load_instances(args.subset)
    gaps = load_higher_order_gap(args.pool) if args.pool else {}
    stray = sorted(i["instance_id"] for i in instances.values()
                   if i["seed"] < HELD_OUT_SEED_FLOOR)
    if stray:
        raise ExportRefused(
            f"{len(stray)} instance(s) carry a seed below the held-out floor "
            f"{HELD_OUT_SEED_FLOOR}, e.g. {stray[:3]}. This export is the held-out set.")

    rows = collect(expected, args.runs, instances, gaps)
    aggregates = aggregate(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "runs.csv", rows, COLUMNS)
    write_sqlite(args.out / "heldout_results.sqlite", rows)
    parquet_status = write_parquet(args.out / "runs.parquet", rows)
    keys = ["grouping", "condition", "cap", "stratum"]
    agg_fields = keys + ([k for k in aggregates[0] if k not in keys]
                         if aggregates else [])
    write_csv(args.out / "aggregates.csv", aggregates, agg_fields)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "standing": (
            "Descriptive export. No hypothesis test, no p-value, no correction and no "
            "decision is computed here. The confirmatory analysis is "
            "scripts/analyse_heldout.py under the plan frozen 2026-08-27; an ordering "
            "visible in these tables describes the sample and is not a finding."),
        "runs_dir": str(args.runs),
        "n_runs": len(rows),
        "n_instances": len(instances),
        "schema": [{"name": name, "sqlite_type": sql_type, "meaning": doc}
                   for name, sql_type, doc in SCHEMA],
        "n_columns": len(SCHEMA),
        "sqlite_table": SQLITE_TABLE,
        "details_json_excludes": NOT_COPIED,
        "source_document_reference": (
            "every row carries `source_json`, the path of the document it was built "
            "from; `run_id` is that file's stem. The full run, including the "
            "transcript and the per-call records that are not copied here, is "
            "recoverable from it."),
        "band_alias": dict(LEVEL_TO_BAND),
        "conditions": sorted({r["condition"] for r in rows}),
        "caps": sorted({r["cap"] for r in rows}),
        "strata": sorted({_stratum(r) for r in rows}),
        "completeness_audit": {
            "n_expected": audit_report["n_expected"],
            "n_verified": audit_report["n_verified"],
            "all_checks_passed": audit_report["all_checks_passed"],
        },
        "parquet": parquet_status,
        "higher_order_gap_H_available": bool(gaps),
    }
    (args.out / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.require_parquet and not parquet_status.startswith("written"):
        raise ExportRefused(parquet_status)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    print(f"exported {len(rows)} runs over {len(instances)} instances")
    print(f"  runs.csv                {len(SCHEMA)} columns (schema in the manifest)")
    print(f"  heldout_results.sqlite  table {SQLITE_TABLE!r}, same rows")
    print(f"  runs.parquet            {parquet_status}")
    print(f"  aggregates.csv          {len(aggregates)} rows over "
          f"{len({a['grouping'] for a in aggregates})} groupings")
    print("  export_manifest.json")
    print("\ndescriptive export only: no test, no p-value, no decision")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
