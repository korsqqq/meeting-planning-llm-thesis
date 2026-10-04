# src/core/__init__.py
"""Agent-facing infrastructure called on every step (Layer 4).

  * tokenizer - Qwen3 token counting for the budget (input vs thinking vs answer).
  * budget    - the single shared ledger + guard.

Distinct from `src/harness/` (the later external batch-runner over the run matrix):
these are invoked by the agents themselves, not by the runner.

  * llm_client - OpenAI-compatible client (vLLM main run / local Ollama for dev).
"""

from __future__ import annotations

from .budget import DEFAULT_FINALIZATION_RESERVE, DEFAULT_MAX_CALL_TOKENS, BudgetLedger
from .llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    END_THINK,
    LLMClient,
    ContextWindowError,
    LLMResponse,
    split_thinking,
    split_thinking_raw,
)
from .tokenizer import DEFAULT_TOKENIZER, QwenTokenizer
from .usage_audit import audit_run, compare_call, extract_endpoint_usage

__all__ = [
    "QwenTokenizer",
    "DEFAULT_TOKENIZER",
    "BudgetLedger",
    "DEFAULT_FINALIZATION_RESERVE",
    "DEFAULT_MAX_CALL_TOKENS",
    "LLMClient",
    "ContextWindowError",
    "LLMResponse",
    "split_thinking",
    "split_thinking_raw",
    "END_THINK",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "extract_endpoint_usage",
    "compare_call",
    "audit_run",
]
