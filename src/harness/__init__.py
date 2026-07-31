# src/harness/__init__.py
"""Harness layer: run instances through conditions and log structured results.

Current scope is the offline vertical slice (one instance, one condition, one
JSON document): `runner.run_single_instance` wires
generator/handcrafted Instance -> CP-SAT oracle -> C1/C2 -> hidden
validator/scorer -> RunResult -> `result_writer` JSON. The sweep runner (run
matrix, resume, aggregation) is a later layer on top of the same entry point.

The oracle stays strictly on this side of the boundary: nothing computed here is
ever fed into an agent's prompts, tools, or observations.
"""

from __future__ import annotations

from .result_writer import DOCUMENT_SCHEMA_VERSION, build_document, write_result
from .runner import SUPPORTED_CONDITIONS, HarnessRun, run_single_instance
from .sanity import ORACLE_TERMS, transcript_sanity
from .smoke_client import OfflineSmokeClient

__all__ = [
    "run_single_instance",
    "HarnessRun",
    "SUPPORTED_CONDITIONS",
    "build_document",
    "write_result",
    "DOCUMENT_SCHEMA_VERSION",
    "OfflineSmokeClient",
    "transcript_sanity",
    "ORACLE_TERMS",
]
