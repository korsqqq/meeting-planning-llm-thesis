# scripts/hpc/summarise_runs.py
"""Summarise the run documents produced by one HPC job into a single JSON.

Written for the runtime-calibration job, and reused by the array job later. It
reads whatever run documents are already on disk, so calling it repeatedly
during a job leaves a usable summary even if the job is killed at its walltime.

This produces **timing** figures for scheduling the held-out sweep. It is not an
analysis: satisfaction is carried through only so a run that produced nothing can
be told apart from one that worked, and it must never be reported as an outcome.

Usage::

    python scripts/hpc/summarise_runs.py --runs-dir <dir> --out <summary.json> \\
        [--label <text>] [--extra key=value ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def read_run(path: str) -> dict:
    """Flatten one run document down to the fields a scheduler cares about."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rr = doc.get("run_result") or {}
    agent = doc.get("agent") or {}
    tokens = rr.get("tokens") or {}
    score = rr.get("score") or {}
    calls = rr.get("calls") or []
    return {
        "file": os.path.basename(path),
        "instance_id": rr.get("instance_id"),
        "level": rr.get("level"),
        "condition": rr.get("condition"),
        "cap": rr.get("cap"),
        "latency_seconds": rr.get("latency_seconds"),
        "tokens_total": tokens.get("total"),
        "tokens_input": tokens.get("input_tokens"),
        "tokens_thinking": tokens.get("thinking_tokens"),
        "tokens_answer": tokens.get("answer_tokens"),
        "budget_exhausted": tokens.get("budget_exhausted"),
        # tokens.n_calls is the ledger's own count; len(calls) is the trace length.
        "n_calls": tokens.get("n_calls") if tokens.get("n_calls") is not None else len(calls),
        "termination": agent.get("termination"),
        # Carried for sanity only -- see the module docstring.
        "satisfaction": score.get("satisfaction"),
    }


def group_key(run: dict) -> str:
    return f"{run.get('condition')}@{run.get('cap')}"


def summarise(runs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(group_key(run), []).append(run)

    by_group = {}
    for key, items in sorted(groups.items()):
        lat = [r["latency_seconds"] for r in items if isinstance(r.get("latency_seconds"), (int, float))]
        tok = [r["tokens_total"] for r in items if isinstance(r.get("tokens_total"), (int, float))]
        cal = [r["n_calls"] for r in items if isinstance(r.get("n_calls"), (int, float))]
        by_group[key] = {
            "n_runs": len(items),
            "latency_seconds": {"values": lat, "mean": _mean(lat), "max": max(lat) if lat else None},
            "tokens_total": {"mean": _mean(tok), "max": max(tok) if tok else None},
            "n_calls": {"mean": _mean(cal), "max": max(cal) if cal else None},
            "termination": dict(Counter(r.get("termination") for r in items)),
        }

    all_lat = [r["latency_seconds"] for r in runs if isinstance(r.get("latency_seconds"), (int, float))]
    all_tok = [r["tokens_total"] for r in runs if isinstance(r.get("tokens_total"), (int, float))]
    return {
        "n_runs": len(runs),
        "by_condition_cap": by_group,
        "overall": {
            "latency_seconds": {
                "sum": sum(all_lat) if all_lat else None,
                "mean": _mean(all_lat),
                "max": max(all_lat) if all_lat else None,
            },
            "tokens_total": {"sum": sum(all_tok) if all_tok else None, "mean": _mean(all_tok)},
            "termination": dict(Counter(r.get("termination") for r in runs)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--extra", action="append", default=[],
                        help="key=value pairs recorded verbatim (repeatable)")
    args = parser.parse_args()

    paths = sorted(glob.glob(str(args.runs_dir / "*.json")))
    runs, unreadable = [], []
    for path in paths:
        try:
            runs.append(read_run(path))
        except Exception as exc:                      # pragma: no cover - defensive
            unreadable.append({"file": os.path.basename(path), "error": str(exc)})

    extra = {}
    for item in args.extra:
        key, _, value = item.partition("=")
        extra[key] = value

    payload = {
        "schema_version": "hpc_runtime_summary/1.0",
        "standing": (
            "runtime calibration on already-used development instances. "
            "Timing only -- NOT a thesis outcome, not an analysis, not held-out data."
        ),
        "label": args.label,
        "runs_dir": str(args.runs_dir),
        "extra": extra,
        "summary": summarise(runs),
        "runs": runs,
        "unreadable": unreadable,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    s = payload["summary"]
    print(f"  summarised {s['n_runs']} run(s) -> {args.out}")
    for key, g in s["by_condition_cap"].items():
        lat = g["latency_seconds"]
        mean_s = f"{lat['mean']:.1f}" if lat["mean"] is not None else "n/a"
        max_s = f"{lat['max']:.1f}" if lat["max"] is not None else "n/a"
        tok_s = f"{g['tokens_total']['mean']:.0f}" if g["tokens_total"]["mean"] is not None else "n/a"
        print(f"    {key:<28} n={g['n_runs']}  mean {mean_s}s  max {max_s}s  "
              f"tokens {tok_s}  {g['termination']}")
    if unreadable:
        print(f"    WARNING: {len(unreadable)} unreadable file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
