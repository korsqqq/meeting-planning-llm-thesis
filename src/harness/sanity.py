# src/harness/sanity.py
"""Transcript sanity diagnostics for (live-)smoke runs. Read-only, no behaviour.

Produces a JSON-serialisable block answering the manual-checkpoint questions:

  * did raw `<think>`/`</think>` reasoning text leak into the conversation
    history (it must never -- react_core appends only the post-think answer);
  * do oracle/scorer vocabulary terms appear anywhere the agent could see them.

Both checks are lexical DIAGNOSTICS, not gates: a live model may innocently say
"score" or "complexity" in its own prose, so hits mean "read this transcript by
hand", not "the run is invalid". The structural guarantees live in the code and
its tests; this block just makes a live transcript cheap to triage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["ORACLE_TERMS", "transcript_sanity"]

# Vocabulary that should never originate from the HARNESS side of the boundary.
ORACLE_TERMS = (
    "optimum",
    "solver",
    "cp-sat",
    "score",
    "validity",
    "satisfaction",
    "complexity",
    "oracle",
)


def transcript_sanity(transcript: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Scan a ReactResult transcript; return a diagnostic block (never raises)."""
    think_open: list[int] = []
    think_close: list[int] = []
    oracle_hits: list[dict[str, Any]] = []

    for i, message in enumerate(transcript):
        content = message.get("content", "")
        if "<think>" in content:
            think_open.append(i)
        if "</think>" in content:
            think_close.append(i)
        low = content.lower()
        for term in ORACLE_TERMS:
            if term in low:
                oracle_hits.append({"message_index": i, "term": term,
                                    "role": message.get("role")})

    think_leak = bool(think_open or think_close)
    return {
        "n_messages": len(transcript),
        "think_leak": think_leak,
        "think_open_message_indices": think_open,
        "think_close_message_indices": think_close,
        "oracle_term_hits": oracle_hits,
        "clean": not think_leak and not oracle_hits,
    }
