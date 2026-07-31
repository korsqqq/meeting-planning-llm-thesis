# src/core/llm_client.py
"""OpenAI-compatible client for the planning agents (vLLM in the main run, a local
Ollama GGUF for development).

Design choices that keep the budget control exact and the code portable (see
THESIS_DECISIONS.md sections 1-2):

  * Completions, not chat. We render the prompt ourselves with the Qwen3 chat
    template (`QwenTokenizer.render_input`) and send the raw string to
    `/v1/completions`. The endpoint does NOT re-template it, so the input token count
    equals exactly what the model sees -- no double-templating, identical on vLLM and
    Ollama.
  * We self-parse the completion rather than relying on vLLM's `--reasoning-parser`
    (which only applies in chat mode). The split point is the LAST `</think>`
    delimiter -- robust to a missing opening `<think>` tag, matching the official
    Qwen3 parsing example (which splits on the closing token only). Everything up
    to and including the delimiter is thinking; the rest is the answer. Token
    counts come from the generated token ids when the endpoint returns them (vLLM
    `return_token_ids`), otherwise from the raw text segments via the same
    tokenizer. The tag tokens and inter-part whitespace are booked as thinking,
    so thinking + answer covers the full generated output and the ledger identity
    total = input + thinking + answer matches what the model actually generated.
  * Fixed sampling for thinking-ON deciding calls (temperature 0.6 / top_p 0.95 /
    top_k 20 / min_p 0 / seed 42). The terminal emit node calls with
    enable_thinking=False and a `guided_json` schema (vLLM); there the constrained
    decoding makes the sampling profile near-irrelevant.

The client is pure transport + token counting. It does NOT own the budget ledger:
the caller asks the guard for `max_new_tokens`, passes it as `max_tokens`, and books
the realised counts (returned on the response) into the shared `BudgetLedger`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .tokenizer import QwenTokenizer
from .usage_audit import extract_endpoint_usage

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "SEED",
    "END_THINK",
    "LLMResponse",
    "LLMClient",
    "split_thinking",
    "split_thinking_raw",
]

# Local Ollama by default; the main run sets VLLM_URL to the vLLM endpoint.
DEFAULT_BASE_URL = os.environ.get("VLLM_URL") or "http://localhost:11434/v1"
DEFAULT_MODEL = os.environ.get("VLLM_MODEL") or "qwen3:8b"

# Fixed sampling for thinking-ON deciding calls (THESIS_DECISIONS.md section 1).
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
SEED = 42

# Qwen3 reasoning delimiters. The split point is the LAST closing delimiter,
# per the official Qwen3 parsing example -- the opening tag is not guaranteed
# to appear in the generated text.
END_THINK = "</think>"
_OPEN_THINK = "<think>"


def split_thinking_raw(text: str) -> tuple[str, str]:
    """Split a completion into raw (thinking, answer) segments at the LAST `</think>`.

    The thinking segment includes the delimiter tags and any surrounding
    whitespace -- these tokens were generated and must be budget-counted. Shapes:
      * `...</think>rest`  -> everything up to and including the last `</think>`
        is thinking, `rest` is the answer (an opening `<think>` is NOT required);
      * `<think>...` with no closer (budget truncated mid-reasoning) -> all
        thinking, empty answer;
      * no tags at all (thinking-OFF call) -> empty thinking, all answer.
    """
    idx = text.rfind(END_THINK)
    if idx != -1:
        cut = idx + len(END_THINK)
        return text[:cut], text[cut:]
    if _OPEN_THINK in text:
        return text, ""
    return "", text


def split_thinking(text: str) -> tuple[str, str]:
    """Cleaned (thinking, answer) for downstream parsing: same split as
    `split_thinking_raw`, with the delimiter tags and edge whitespace stripped."""
    thinking_raw, answer_raw = split_thinking_raw(text)
    thinking = thinking_raw
    if thinking.endswith(END_THINK):
        thinking = thinking[: -len(END_THINK)]
    stripped = thinking.lstrip()
    if stripped.startswith(_OPEN_THINK):
        thinking = stripped[len(_OPEN_THINK):]
    return thinking.strip(), answer_raw.strip()


def _split_output_ids(output_ids: Sequence[int], end_think_id: int) -> tuple[int, int] | None:
    """(thinking_count, answer_count) split at the LAST `end_think_id` (the
    delimiter counts as thinking); None when the delimiter id is absent."""
    for i in range(len(output_ids) - 1, -1, -1):
        if output_ids[i] == end_think_id:
            return i + 1, len(output_ids) - i - 1
    return None


@dataclass(frozen=True)
class LLMResponse:
    """One completion, already split and token-counted.

    The `endpoint_*` fields mirror what the endpoint itself reported. AUDIT
    ONLY: the budget is enforced from the internal `*_tokens` counts above
    (Qwen tokenizer + BudgetLedger), never from these. All endpoint fields are
    optional and stay None when the endpoint returns no usage (offline
    scripted clients, some Ollama versions) -- a missing report is recorded
    honestly, never faked. `finish_reason` is already the endpoint's own value.
    Reasoning/cached detail fields may be absent depending on endpoint, model
    and vLLM version; that is normal, not an error.
    """

    thinking: str
    answer: str
    raw_text: str
    input_tokens: int
    thinking_tokens: int
    answer_tokens: int
    finish_reason: str | None
    endpoint_prompt_tokens: int | None = None
    endpoint_completion_tokens: int | None = None
    endpoint_total_tokens: int | None = None
    endpoint_reasoning_tokens: int | None = None
    endpoint_cached_tokens: int | None = None
    endpoint_usage_raw: dict[str, Any] | None = None
    endpoint_model: str | None = None
    usage_source: str | None = None  # "endpoint.usage" when the endpoint reported usage


class LLMClient:
    """Thin wrapper over an OpenAI-compatible `/v1/completions` endpoint."""

    def __init__(
        self,
        *,
        tokenizer: QwenTokenizer,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "ollama",   # ignored by Ollama; vLLM accepts any non-empty key
        timeout: float = 180.0,
        force_no_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.base_url = base_url
        # Local-debug escape hatch: Ollama's Qwen3 GGUF does its thinking internally and
        # returns empty text via /v1/completions, so thinking-ON decide steps come back
        # blank. Setting this renders every call thinking-OFF -- the model emits its
        # actions/JSON directly. Debug only; the vLLM run keeps thinking ON (the <think>
        # block is in the raw completion there and is parsed/counted normally).
        self.force_no_thinking = force_no_thinking
        # Per-call endpoint usage audit trail, in call order: one entry per
        # `complete()` -- the extracted usage record, or None when the endpoint
        # reported nothing for that call. The harness snapshots a slice of this
        # per run to build the usage_audit JSON block. Audit only: the internal
        # ledger stays the sole budget authority.
        self.usage_log: list[dict[str, Any] | None] = []
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def count_input(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Token count of the exact prompt that `complete` would send.

        The caller uses this to ask the budget guard for `max_new_tokens` BEFORE the
        call. It counts the rendered string itself, so it matches what the model sees.
        """
        think = enable_thinking and not self.force_no_thinking
        rendered = self.tokenizer.render_input(
            messages, tools=tools, enable_thinking=think
        )
        return self.tokenizer.count_text(rendered)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        enable_thinking: bool = True,
        tools: list[dict[str, Any]] | None = None,
        guided_json: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Render, call `/v1/completions`, split, and count one generation.

        `max_tokens` is the guard's `max_new_tokens` for this call. `guided_json`
        constrains the output to a JSON schema (vLLM `guided_json`; used for the
        terminal emit node with enable_thinking=False).
        """
        think = enable_thinking and not self.force_no_thinking
        rendered = self.tokenizer.render_input(
            messages, tools=tools, enable_thinking=think
        )
        input_tokens = self.tokenizer.count_text(rendered)

        # `think` is read by Ollama (disable its hidden reasoning); vLLM ignores it.
        # `return_token_ids` asks vLLM for the generated token ids so the
        # thinking/answer split can be counted exactly; other endpoints ignore it.
        extra_body: dict[str, Any] = {
            "top_k": TOP_K,
            "min_p": MIN_P,
            "think": think,
            "return_token_ids": True,
            "add_special_tokens": False,
        }
        if guided_json is not None:
            extra_body["guided_json"] = guided_json

        completion = self._client.completions.create(
            model=self.model,
            prompt=rendered,
            max_tokens=max(1, max_tokens),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            n=1,
            stop=stop,
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        text = choice.text or ""
        thinking, answer = split_thinking(text)
        thinking_tokens, answer_tokens = self._count_output(choice, text)

        # Endpoint-reported usage, if any (AUDIT ONLY -- the internal counts
        # below remain the budget source; see usage_audit.py). Extraction never
        # raises: missing usage, dict-style usage and missing reasoning-token
        # details (endpoint/model/vLLM-version dependent) all degrade to None.
        usage_record = extract_endpoint_usage(
            getattr(completion, "usage", None),
            model=getattr(completion, "model", None),
            finish_reason=choice.finish_reason,
        )
        self.usage_log.append(usage_record)

        return LLMResponse(
            thinking=thinking,
            answer=answer,
            raw_text=text,
            input_tokens=input_tokens,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            finish_reason=choice.finish_reason,
            endpoint_prompt_tokens=usage_record["prompt_tokens"] if usage_record else None,
            endpoint_completion_tokens=usage_record["completion_tokens"] if usage_record else None,
            endpoint_total_tokens=usage_record["total_tokens"] if usage_record else None,
            endpoint_reasoning_tokens=usage_record["reasoning_tokens"] if usage_record else None,
            endpoint_cached_tokens=usage_record["cached_tokens"] if usage_record else None,
            endpoint_usage_raw=usage_record["raw"] if usage_record else None,
            endpoint_model=usage_record["model"] if usage_record else None,
            usage_source="endpoint.usage" if usage_record else None,
        )

    def _count_output(self, choice: Any, text: str) -> tuple[int, int]:
        """(thinking_tokens, answer_tokens) for one generation, split at the LAST
        `</think>` -- the delimiter tags and inter-part whitespace count as
        thinking, so the two parts always sum to the full generated output.

        Preferred path: the endpoint's own generated token ids (vLLM
        `return_token_ids`), split at the delimiter's token id -- exact, no
        re-tokenisation. Fallback (Ollama, older vLLM, scripted clients): the raw
        text segments from `split_thinking_raw`, counted with the Qwen tokenizer.
        `</think>` is a single added token, so the segment counts compose exactly.
        """
        thinking_raw, answer_raw = split_thinking_raw(text)

        output_ids = getattr(choice, "token_ids", None)
        end_think_id = getattr(self.tokenizer, "end_think_token_id", None)
        if (
            end_think_id is not None
            and isinstance(output_ids, (list, tuple))
            and all(isinstance(t, int) for t in output_ids)
        ):
            ids = list(output_ids)
            split = _split_output_ids(ids, end_think_id)
            if split is not None:
                return split
            if not answer_raw:
                # No delimiter generated: truncated thinking (all thinking) or,
                # with no tags at all, a thinking-OFF call (all answer).
                thinking_count = len(ids) if thinking_raw else 0
                return thinking_count, len(ids) - thinking_count
            if not thinking_raw:
                return 0, len(ids)
            # `</think>` appears in the text but its id is absent from the ids:
            # ids and text disagree -- fall through to the text-segment count.

        thinking_tokens = self.tokenizer.count_text(thinking_raw) if thinking_raw else 0
        answer_tokens = self.tokenizer.count_text(answer_raw) if answer_raw else 0
        return thinking_tokens, answer_tokens
