# src/harness/result_writer.py
"""Serialise one harness run into a stable, inspectable JSON document.

The canonical per-run record is the Layer-0 `RunResult` (src/schemas/result.py);
this module does not redefine any of its fields. The document wraps that record
with the harness-side context RunResult deliberately does not carry: the oracle
block (kept out of the agent by design), the instance provenance, the transcript,
and reproducibility metadata (all three seeds, git commit/branch).

Layout (top-level keys, stable order):

    schema_version   "harness_slice/1.0"
    run_id           mirrors run_result.run_id (also the filename stem)
    timestamp_utc    mirrors run_result.timestamp
    condition / cap / model_label / finalization_reserve
    seeds            {generator, solver, inference} -- logged separately on purpose
    provenance       {git_commit, git_branch} (null when git is unavailable)
    instance         task identity + knobs + measured complexity
    oracle           optimum / status / proven_optimal / latency (harness-only data)
    agent            transcript + step count (artifacts not in RunResult)
    usage_audit      internal ledger vs endpoint-reported usage (audit only)
    transcript_sanity think-leak / oracle-vocabulary diagnostics (audit only)
    run_result       the full canonical RunResult dump (tokens, calls, score, plans)

Values are never invented: anything unavailable is stored as null.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from src.agents.react_core import ReactResult
from src.core.llm_client import SEED as INFERENCE_SEED
from src.oracle import SOLVER_SEED, OracleSolution
from src.schemas import Instance, RunResult

__all__ = ["DOCUMENT_SCHEMA_VERSION", "build_document", "write_result"]

DOCUMENT_SCHEMA_VERSION = "harness_slice/1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_value(*args: str) -> str | None:
    """One git plumbing value, or None if git/repo is unavailable (never raises)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_document(
    *,
    run_result: RunResult,
    instance: Instance,
    oracle: OracleSolution,
    oracle_latency_seconds: float,
    agent: ReactResult,
    finalization_reserve: int,
    usage_audit: dict[str, Any] | None = None,
    sanity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-ready document for one run (pure, no I/O).

    `usage_audit` (internal ledger vs endpoint-reported usage) and `sanity`
    (transcript think-leak / oracle-vocabulary scan) are audit-only diagnostic
    blocks; they are stored verbatim and default to None when not computed.
    """
    return {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "run_id": run_result.run_id,
        "timestamp_utc": run_result.timestamp,
        "condition": run_result.condition.value,
        "cap": run_result.cap,
        "model_label": run_result.model,
        "finalization_reserve": finalization_reserve,
        "seeds": {
            "generator": instance.seed,
            "solver": SOLVER_SEED,
            "inference": INFERENCE_SEED,
        },
        "provenance": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "instance": {
            "instance_id": instance.instance_id,
            "generator_params": instance.generator_params.model_dump(mode="json"),
            "n_people": len(instance.people),
            "travel_structure": instance.generator_params.travel_structure.value,
            "waiting_allowed": instance.waiting_allowed,
            "start_time": instance.start_time,
            "end_of_day": instance.end_of_day,
            "meeting_duration": instance.meeting_duration,
            "complexity_metric": instance.complexity_metric,
            "level": instance.level.value if instance.level is not None else None,
        },
        "oracle": {
            "optimum": oracle.optimum,
            "status": oracle.status,
            "proven_optimal": oracle.proven_optimal,
            "latency_seconds": oracle_latency_seconds,
        },
        "agent": {
            "n_steps": agent.n_steps,
            # Integrity diagnostic: the terminal emit parsed but did not round-trip
            # best_plan_so_far, so the structural plan was scored instead. Cannot affect
            # the score; the pilot reports its rate (THESIS_DECISIONS section 7).
            "finalization_mismatch": agent.finalization_mismatch,
            "transcript": agent.transcript,
        },
        "usage_audit": usage_audit,
        "transcript_sanity": sanity,
        "run_result": run_result.model_dump(mode="json"),
    }


def write_result(document: dict[str, Any], output_dir: Path) -> Path:
    """Write the document as `<output_dir>/<run_id>.json` (idempotent overwrite)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{document['run_id']}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path
