# src/core/usage_audit.py
"""Endpoint-usage extraction and internal-vs-endpoint token comparison. AUDIT ONLY.

The experimental budget source is, and stays, the INTERNAL ledger
(`src/core/budget.py`, counted with the Qwen tokenizer -- THESIS_DECISIONS §2).
Endpoint-reported usage is captured purely so a live smoke run can AUDIT that
accounting:

  * the internal OUTPUT count includes the `<think>`/`</think>` tag tokens and
    the whitespace between the two parts (the split is at the last `</think>`,
    everything generated is booked), so the `endpoint - internal` delta should
    be ~0 -- exactly 0 when the endpoint returns generated token ids
    (`return_token_ids`), small residuals only when raw text segments have to
    be re-tokenised (Ollama, older vLLM);
  * a large or negative delta signals a counting bug and must be investigated
    before the pilot.

Nothing here enforces anything. Every function returns a JSON-serialisable
record; missing endpoint usage is recorded honestly (`has_endpoint_usage:
false`, null fields), never faked or interpolated. Reasoning-token detail
fields (`completion_tokens_details.reasoning_tokens`) may be missing depending
on endpoint, model and vLLM version -- their absence is normal, not an error.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.schemas import CallRecord, TokenUsage

__all__ = ["extract_endpoint_usage", "compare_call", "audit_run"]


# --------------------------------------------------------------------------- #
# Tolerant accessors (OpenAI SDK objects, plain dicts, SimpleNamespace, ...).
# --------------------------------------------------------------------------- #
def _dig(obj: Any, *keys: str) -> Any:
    """Walk `obj` by keys, treating each level as a mapping or an attribute bag."""
    cur = obj
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-serialisable copy (non-serialisable leaves become str)."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def _raw_usage(usage: Any) -> Any:
    """Preserve the endpoint's raw usage object in a JSON-serialisable form."""
    if isinstance(usage, dict):
        return _json_safe(usage)
    for dump in ("model_dump", "to_dict"):
        fn = getattr(usage, dump, None)
        if callable(fn):
            try:
                return _json_safe(fn())
            except Exception:
                pass
    try:
        return _json_safe(vars(usage))
    except Exception:
        return str(usage)


# --------------------------------------------------------------------------- #
# Extraction (one endpoint response -> one record, or None if usage is absent).
# --------------------------------------------------------------------------- #
def extract_endpoint_usage(
    usage: Any,
    *,
    model: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any] | None:
    """Normalise an endpoint `usage` object into a flat record, or None.

    Handles OpenAI-style attribute objects, dict-style usage, and nested
    `completion_tokens_details` / `prompt_tokens_details` blocks. Never raises
    on a shape it does not know -- unknown fields simply stay in `raw`.
    """
    if usage is None:
        return None
    record = {
        "prompt_tokens": _as_int(_dig(usage, "prompt_tokens")),
        "completion_tokens": _as_int(_dig(usage, "completion_tokens")),
        "total_tokens": _as_int(_dig(usage, "total_tokens")),
        "reasoning_tokens": _as_int(
            _dig(usage, "completion_tokens_details", "reasoning_tokens")
        ),
        "cached_tokens": _as_int(_dig(usage, "prompt_tokens_details", "cached_tokens")),
        "model": model,
        "finish_reason": finish_reason,
        "raw": _raw_usage(usage),
    }
    if all(
        record[k] is None
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        # An object that carries no recognisable counts is as good as no usage;
        # returning None keeps `has_endpoint_usage` honest.
        return None
    return record


# --------------------------------------------------------------------------- #
# Comparison.
# --------------------------------------------------------------------------- #
def _endpoint_total(endpoint: dict[str, Any], notes: list[str]) -> int | None:
    total = endpoint.get("total_tokens")
    if total is not None:
        return total
    prompt, completion = endpoint.get("prompt_tokens"), endpoint.get("completion_tokens")
    if prompt is not None and completion is not None:
        notes.append("endpoint total_tokens reconstructed from prompt + completion")
        return prompt + completion
    return None


def compare_call(
    *,
    internal_input_tokens: int,
    internal_thinking_tokens: int,
    internal_answer_tokens: int,
    endpoint: dict[str, Any] | None,
    role: str | None = None,
) -> dict[str, Any]:
    """One call: internal ledger counts vs the endpoint's report (if any)."""
    internal_total = (
        internal_input_tokens + internal_thinking_tokens + internal_answer_tokens
    )
    notes: list[str] = []
    result: dict[str, Any] = {
        "role": role,
        "internal_input_tokens": internal_input_tokens,
        "internal_thinking_tokens": internal_thinking_tokens,
        "internal_answer_tokens": internal_answer_tokens,
        "internal_total_tokens": internal_total,
        "endpoint_prompt_tokens": None,
        "endpoint_completion_tokens": None,
        "endpoint_total_tokens": None,
        "endpoint_reasoning_tokens": None,
        "delta_total_tokens": None,
        "relative_delta_total": None,
        "has_endpoint_usage": endpoint is not None,
        "notes": notes,
    }
    if endpoint is None:
        notes.append("endpoint usage missing for this call")
        return result

    result["endpoint_prompt_tokens"] = endpoint.get("prompt_tokens")
    result["endpoint_completion_tokens"] = endpoint.get("completion_tokens")
    result["endpoint_reasoning_tokens"] = endpoint.get("reasoning_tokens")
    endpoint_total = _endpoint_total(endpoint, notes)
    result["endpoint_total_tokens"] = endpoint_total
    if endpoint_total is not None:
        delta = endpoint_total - internal_total
        result["delta_total_tokens"] = delta
        if internal_total > 0:
            result["relative_delta_total"] = delta / internal_total
    return result


def _sum_or_none(values: list[int | None]) -> int | None:
    if not values or any(v is None for v in values):
        return None
    return sum(values)  # type: ignore[arg-type]


def audit_run(
    *,
    tokens: TokenUsage,
    calls: Sequence[CallRecord],
    endpoint_records: Sequence[dict[str, Any] | None] | None,
) -> dict[str, Any]:
    """Whole-run audit block: per-call comparisons + totals. JSON-serialisable.

    `endpoint_records` is the client's usage log slice for THIS run: one entry
    per completion call in order, None where the endpoint reported no usage, or
    None altogether for clients that do not log usage (offline scripted ones).
    """
    notes: list[str] = []
    records = list(endpoint_records) if endpoint_records is not None else None
    has_usage = bool(records) and any(r is not None for r in records)

    aligned: bool | None = None
    if records is not None:
        aligned = len(records) == len(calls)
        if not aligned:
            notes.append(
                f"endpoint record count ({len(records)}) != internal call count "
                f"({len(calls)}); per-call alignment skipped"
            )

    per_call = [
        compare_call(
            internal_input_tokens=call.input_tokens,
            internal_thinking_tokens=call.thinking_tokens,
            internal_answer_tokens=call.answer_tokens,
            endpoint=records[i] if (records is not None and aligned) else None,
            role=call.role,
        )
        for i, call in enumerate(calls)
    ]

    if has_usage:
        usable = [r for r in records if r is not None]
        endpoint_prompt = _sum_or_none([r.get("prompt_tokens") for r in usable])
        endpoint_completion = _sum_or_none([r.get("completion_tokens") for r in usable])
        endpoint_total = _sum_or_none([r.get("total_tokens") for r in usable])
        if endpoint_total is None and endpoint_prompt is not None and endpoint_completion is not None:
            endpoint_total = endpoint_prompt + endpoint_completion
            notes.append("endpoint total reconstructed from prompt + completion sums")
        if any(r is None for r in records):
            notes.append("some calls returned no endpoint usage; sums cover the rest")
        notes.append(
            "internal output counts include <think> tags and inter-part whitespace "
            "(split at the last </think>); expect ~0 delta, small residuals only "
            "when text segments are re-tokenised instead of using endpoint token ids"
        )
    else:
        endpoint_prompt = endpoint_completion = endpoint_total = None
        notes.append(
            "no endpoint usage available (offline/scripted client or endpoint "
            "returned none); internal ledger stands alone"
        )

    delta = endpoint_total - tokens.total if endpoint_total is not None else None
    return {
        "has_endpoint_usage": has_usage,
        "n_internal_calls": tokens.n_calls,
        "n_endpoint_records": len(records) if records is not None else None,
        "calls_aligned": aligned,
        "totals": {
            "internal_input_tokens": tokens.input_tokens,
            "internal_thinking_tokens": tokens.thinking_tokens,
            "internal_answer_tokens": tokens.answer_tokens,
            "internal_total_tokens": tokens.total,
            "endpoint_prompt_tokens": endpoint_prompt,
            "endpoint_completion_tokens": endpoint_completion,
            "endpoint_total_tokens": endpoint_total,
            "delta_total_tokens": delta,
            "relative_delta_total": (
                delta / tokens.total if delta is not None and tokens.total > 0 else None
            ),
        },
        "per_call": per_call,
        "notes": notes,
    }
