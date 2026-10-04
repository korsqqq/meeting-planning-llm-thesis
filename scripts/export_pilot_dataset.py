# scripts/export_pilot_dataset.py
"""Thesis-facing tables derived from the frozen formal-pilot runs. CPU only, read-only.

    python -m scripts.export_pilot_dataset

The 360 run documents are the archive; they are complete but not something anyone can plot
from. `results/analysis/formal_pilot/runs.csv` already gives one row per run, so this script
does not duplicate it. It emits the three views that exist nowhere else and that the thesis
chapters will need:

* `pilot_instances.csv` -- one row per instance (30), with the structural facts and the
  **retrospective** density `D = conflicting pairs / C(n_people, 2)`.
* `pilot_proposals.csv` -- one row per proposal attempt, carrying the role that produced it,
  so worker, critic and revise attempts can be separated rather than pooled.
* `pilot_calls.csv` -- one row per LLM call, with role, token split, finish reason, context
  clamping and latency.

All three key on `run_id` or `instance_id` and join to `runs.csv`.

**Two warnings this export exists to make impossible to miss**, both written into the
README it produces:

The `level` column in the pilot data is the **old** binning, superseded on 2026-08-09. It is
not comparable with the pairwise-conflict-density bands frozen for the main experiment, and
a chapter that plots the two together is plotting two different axes.

`D_retrospective` re-expresses the pilot on the axis adopted afterwards. It is computable
because every run records `complexity_metric` and `n_people`, and it is genuinely useful for
showing what the pilot looked like on the later axis. It is **descriptive only**: these
instances were never selected under that axis, the pilot spans three values of `n`, and no
confirmatory claim may rest on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from math import comb
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "pilot_dataset/1.0"
EXPECTED_RUNS = 360
EXPECTED_INSTANCES = 30


class PilotExportError(RuntimeError):
    """The archive is not the frozen pilot; the export must not proceed."""


def load(run_dir: Path) -> list[dict[str, Any]]:
    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(run_dir.glob("*.json"))]
    if len(docs) != EXPECTED_RUNS:
        raise PilotExportError(
            f"expected {EXPECTED_RUNS} run documents under {run_dir}, found {len(docs)}")
    ids = {d["run_result"]["instance_id"] for d in docs}
    if len(ids) != EXPECTED_INSTANCES:
        raise PilotExportError(
            f"expected {EXPECTED_INSTANCES} distinct instances, found {len(ids)}")
    return docs


def instances_table(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for d in docs:
        rr, inst = d["run_result"], d.get("instance", {})
        iid = rr["instance_id"]
        gp = inst.get("generator_params", {})
        n = gp.get("n_people")
        k = inst.get("complexity_metric")
        row = {
            "instance_id": iid,
            "seed": rr["seed"],
            "n_people": n,
            "tightness": gp.get("tightness"),
            "overlap": gp.get("overlap"),
            "travel_structure": gp.get("travel_structure"),
            "conflicting_pairs": k,
            # Retrospective: the axis adopted after this pilot, applied to it for
            # description only. See the README.
            "D_retrospective": (round(k / comb(n, 2), 6)
                                if k is not None and n and n >= 2 else None),
            "solver_optimum": rr["score"]["solver_optimum"],
            "level_old_binning": rr["level"],
        }
        if iid in rows and rows[iid] != row:
            raise PilotExportError(
                f"{iid}: instance facts differ between runs; the archive is inconsistent")
        rows[iid] = row
    return [rows[k] for k in sorted(rows)]


def proposals_table(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        rr = d["run_result"]
        for i, p in enumerate(d["agent"].get("proposals", [])):
            out.append({
                "run_id": rr["run_id"],
                "instance_id": rr["instance_id"],
                "condition": rr["condition"],
                "cap": rr["cap"],
                "attempt_index": i,
                "role": p.get("role"),
                "step": p.get("step"),
                "n_meetings": p.get("n_meetings"),
                "parsed": p.get("parsed"),
                "valid": p.get("valid"),
                "accepted_into_best_plan": p.get("accepted_into_best_plan"),
                "reasons": ";".join(p.get("reasons") or []),
            })
    return out


def calls_table(docs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        rr = d["run_result"]
        for i, c in enumerate(rr.get("calls", [])):
            out.append({
                "run_id": rr["run_id"],
                "instance_id": rr["instance_id"],
                "condition": rr["condition"],
                "cap": rr["cap"],
                "call_index": i,
                "role": c.get("role"),
                "input_tokens": c.get("input_tokens"),
                "thinking_tokens": c.get("thinking_tokens"),
                "answer_tokens": c.get("answer_tokens"),
                "finish_reason": c.get("finish_reason"),
                "context_limited": c.get("context_limited"),
                "requested_max_tokens": c.get("requested_max_tokens"),
                "effective_max_tokens": c.get("effective_max_tokens"),
                "latency_seconds": c.get("latency_seconds"),
            })
    return out


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        raise PilotExportError(f"refusing to write an empty table to {path}")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


README = """# Formal pilot — archived dataset

360 runs: 3 conditions (C1 ReAct, C2 verify/revise, C3 hierarchical MAS) x 4 budget caps
(8k, 16k, 32k, 64k) x 30 instances, one run per cell, `Qwen/Qwen3-32B-AWQ` with prefix
caching disabled. Completed 2026-08-07; the raw documents in this directory have not been
modified since and are not to be modified.

## Read this before plotting anything

**`level` here is the OLD binning.** It was superseded on 2026-08-09. The main experiment
uses three bands of pairwise conflict density at a fixed `n_people = 8` (THESIS_DECISIONS
section 3, amendment of that date). The two are different axes cut on different pools, and
putting them on one chart compares things that are not comparable. In the exported tables
the column is named `level_old_binning` for that reason.

**`D_retrospective` is descriptive.** It re-expresses each pilot instance on the axis adopted
afterwards, `conflicting pairs / C(n_people, 2)`, which is computable because every run
records both quantities. It is useful for showing what the pilot looked like on the later
axis. It is not evidence for anything: these instances were never selected under that axis,
the pilot spans `n` in {4, 6, 8} while the main experiment fixes `n = 8`, and the oracle
optimum was never matched across densities here.

## What is where

| Path | Contents |
|---|---|
| `results/formal_pilot/*.json` | the 360 run documents, including agent transcripts |
| `results/formal_pilot/sweep_logs/` | stdout of both sweep shards: the exact launch commands and timings |
| `results/analysis/formal_pilot/runs.csv` | one row per run, 60 columns |
| `results/analysis/formal_pilot/report.md` | the corrected analysis |
| `results/analysis/formal_pilot/pilot_instances.csv` | one row per instance, with `D_retrospective` |
| `results/analysis/formal_pilot/pilot_proposals.csv` | one row per proposal attempt, with the role that produced it |
| `results/analysis/formal_pilot/pilot_calls.csv` | one row per LLM call, with tokens, finish reason and latency |

The tables join on `run_id` and `instance_id`.

## What the pilot established, and what it did not

Established: C3 ahead of C1 at caps 32k and 64k, paired bootstrap intervals excluding zero,
while spending fewer tokens. Not established: any dependence of that advantage on complexity,
in either direction. The principal diagnostic was that the low-complexity sanity check failed
— the hierarchy led on the easiest level with interval support — which is what caused the
complexity axis to be rebuilt. Full numbers and the correction history are in
THESIS_DECISIONS section 3.

Do not reuse these 30 instances for the main experiment. They come from the pilot seed range
10005+, which is consumed; the main experiment draws from 100000+.

## Provenance

Generated by `scripts/export_pilot_dataset.py`. Re-running it reproduces the tables byte for
byte, so `git status` staying clean after a rerun is a working reproducibility check.
"""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/formal_pilot"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/formal_pilot"))
    ap.add_argument("--readme", type=Path, default=Path("results/formal_pilot/README.md"))
    args = ap.parse_args(argv)

    docs = load(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    digests = {
        "pilot_instances.csv": write_csv(args.out / "pilot_instances.csv",
                                         instances_table(docs)),
        "pilot_proposals.csv": write_csv(args.out / "pilot_proposals.csv",
                                         proposals_table(docs)),
        "pilot_calls.csv": write_csv(args.out / "pilot_calls.csv", calls_table(docs)),
    }
    with args.readme.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(README)

    print(f"instances : {len(instances_table(docs)):>5}")
    print(f"proposals : {len(proposals_table(docs)):>5}")
    print(f"calls     : {len(calls_table(docs)):>5}")
    for name, digest in digests.items():
        print(f"  {name:<24} sha256 {digest[:16]}...")
    print(f"written: {args.out}/pilot_*.csv and {args.readme}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
