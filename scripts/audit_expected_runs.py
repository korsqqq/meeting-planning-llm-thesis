# scripts/audit_expected_runs.py
"""Completeness audit: every expected run has a verified result on disk. CPU only.

    python -m scripts.audit_expected_runs \
        --expected results/manifests/expected_runs__lowcx_n4.json \
        --expected results/manifests/expected_runs__lowcx_n5.json \
        --expected results/manifests/expected_runs__lowcx_n6.json \
        --runs results/logs/lowcx_calib

The registered completion criterion: **analysis may begin only once every run id in the
expected-runs manifests has a verified result on disk.** This script is that check and
nothing else -- it reads no satisfaction, computes no mean and decides no threshold. It
exists so that "the sweep finished" is a proved statement rather than a file count.

A file count is not completeness. A truncated document, a result carried over from a
different frozen selection, or one written before a schema change would all satisfy
`ls | wc -l` and then quietly enter an analysis. Each expected run is therefore re-opened
and checked against the manifest it claims to come from, using the sweep runner's own
`verify_completed` so that the audit and the resume logic cannot drift apart.

Unexpected files in the results directory are reported too. A stray document means either
that the directory is shared with another sweep -- which breaks the "kept separate" rule --
or that a run was produced under parameters nobody recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_pilot_sweep import (  # noqa: E402
    ResultIncompatible,
    result_path,
    verify_completed,
)

SCHEMA_VERSION = "expected_runs_audit/1.0"


class AuditFailed(RuntimeError):
    """The sweep is not complete, or a result is not the run it claims to be."""


def load_expected(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Flatten the expected-runs documents, carrying each run's manifest hashes with it."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        runs = doc.get("runs")
        if not isinstance(runs, list) or not runs:
            raise AuditFailed(f"{p.name}: no runs listed")
        for r in runs:
            if r["run_id"] in seen:
                raise AuditFailed(f"{r['run_id']} appears in more than one expected-runs "
                                  "manifest; the audit would then pass on one file twice")
            seen.add(r["run_id"])
            out.append({**r, "subset_hash": doc["subset_hash"],
                        "binning_hash": doc["binning_hash"], "model": doc["model"],
                        "source": p.name})
    return out


def audit(expected: Sequence[dict[str, Any]], runs_dir: Path) -> dict[str, Any]:
    missing: list[str] = []
    incompatible: list[str] = []
    ok: list[dict[str, Any]] = []

    for r in expected:
        path = result_path(runs_dir, r["condition"], r["instance_id"], r["cap"], r["model"])
        if not path.exists():
            missing.append(r["run_id"])
            continue
        try:
            verify_completed(
                path, run_id=r["run_id"], instance_id=r["instance_id"],
                condition=r["condition"], cap=r["cap"],
                subset_hash=r["subset_hash"], binning_hash=r["binning_hash"],
            )
        except ResultIncompatible as exc:
            incompatible.append(str(exc))
            continue
        ok.append(r)

    on_disk = {p.stem for p in runs_dir.glob("*.json")}
    unexpected = sorted(on_disk - {r["run_id"] for r in expected})

    cells = Counter((r["condition"], r["cap"]) for r in ok)
    return {
        "schema_version": SCHEMA_VERSION,
        "audited_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_dir": str(runs_dir),
        "n_expected": len(expected),
        "n_verified": len(ok),
        "n_missing": len(missing),
        "n_incompatible": len(incompatible),
        "n_unexpected_files": len(unexpected),
        "missing": missing[:50],
        "incompatible": incompatible[:50],
        "unexpected_files": unexpected[:50],
        "verified_by_condition_and_cap": {
            f"{c}@{cap}": n for (c, cap), n in sorted(cells.items())},
        "verified_by_source_manifest": dict(sorted(Counter(
            r["source"] for r in ok).items())),
        "models": sorted({r["model"] for r in expected}),
        "all_checks_passed": not (missing or incompatible or unexpected),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expected", action="append", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="optional path for the audit document")
    args = ap.parse_args(argv)

    expected = load_expected(args.expected)
    report = audit(expected, args.runs)

    print(f"expected {report['n_expected']} runs over {len(args.expected)} manifest(s)")
    print(f"verified {report['n_verified']}  missing {report['n_missing']}  "
          f"incompatible {report['n_incompatible']}  "
          f"unexpected files {report['n_unexpected_files']}")
    for cell, n in report["verified_by_condition_and_cap"].items():
        print(f"  {cell:<24} {n}")
    for label, items in (("missing", report["missing"]),
                         ("incompatible", report["incompatible"]),
                         ("unexpected", report["unexpected_files"])):
        for item in items[:10]:
            print(f"  {label}: {item}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"written: {args.out}")

    if not report["all_checks_passed"]:
        print("\nAUDIT FAILED — analysis may not begin. Every expected run must have a "
              "verified result, and nothing else may be in the directory.")
        return 1
    print("\naudit passed — every expected run has a verified result on disk")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
