# tests/test_llm_client.py
"""LLMClient tests.

`split_thinking` is pure and tested offline. The end-to-end test hits a live
OpenAI-compatible endpoint ($VLLM_URL or local Ollama) and skips if none is
reachable -- it asserts the transport + counting + ledger balance, NOT that the
model produced a <think> block (that is endpoint/model dependent; Ollama's GGUF
does not surface thinking through /v1/completions, the vLLM run does).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.core import (
    BudgetLedger,
    LLMClient,
    QwenTokenizer,
    split_thinking,
    split_thinking_raw,
)


# --------------------------------------------------------------------------- #
# Offline: <think> splitting.
# --------------------------------------------------------------------------- #
def test_split_thinking_normal() -> None:
    thinking, answer = split_thinking("<think>weigh options</think>final plan")
    assert thinking == "weigh options"
    assert answer == "final plan"


def test_split_thinking_strips_whitespace() -> None:
    thinking, answer = split_thinking("<think>\n  reason \n</think>\n\n  the answer ")
    assert thinking == "reason"
    assert answer == "the answer"


def test_split_thinking_no_block_is_all_answer() -> None:
    # thinking-OFF terminal emit: no tags, everything is the answer.
    thinking, answer = split_thinking('{"meetings": []}')
    assert thinking == ""
    assert answer == '{"meetings": []}'


def test_split_thinking_unclosed_block_is_all_thinking() -> None:
    # Budget truncated generation mid-reasoning: no answer yet.
    thinking, answer = split_thinking("<think>still reasoning when the budget ran")
    assert thinking == "still reasoning when the budget ran"
    assert answer == ""


def test_split_thinking_missing_open_tag_is_still_thinking() -> None:
    # Qwen3 may omit the opening <think>; the official parsing example splits
    # on the closing delimiter only. Everything before it is reasoning.
    thinking, answer = split_thinking("reason without open tag</think>\nAction: x")
    assert thinking == "reason without open tag"
    assert answer == "Action: x"


def test_split_thinking_splits_at_last_closer() -> None:
    # A literal </think> inside the reasoning must not end the thinking early.
    thinking, answer = split_thinking("<think>a</think>b</think>final")
    assert thinking == "a</think>b"
    assert answer == "final"


def test_split_thinking_raw_keeps_tags_and_whitespace() -> None:
    # The raw segments are the budget-counting basis: the delimiter tags and the
    # whitespace between the parts stay inside the thinking segment.
    raw = "<think>\nreason\n</think>\n\nanswer"
    thinking_raw, answer_raw = split_thinking_raw(raw)
    assert thinking_raw == "<think>\nreason\n</think>"
    assert answer_raw == "\n\nanswer"
    assert thinking_raw + answer_raw == raw


# --------------------------------------------------------------------------- #
# Offline: token accounting of the thinking/answer split (stubbed transport,
# real Qwen tokenizer).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client() -> LLMClient:
    try:
        return LLMClient(tokenizer=QwenTokenizer())
    except Exception as exc:  # tokenizer download failure
        pytest.skip(f"tokenizer unavailable: {exc}")


def _stubbed_client(client: LLMClient, completion: Any) -> tuple[LLMClient, list[dict[str, Any]]]:
    """A fresh client whose transport returns `completion`; records call kwargs."""
    stub = LLMClient(tokenizer=client.tokenizer, model="stub",
                     base_url="http://localhost:9/v1")  # never contacted
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return completion

    stub._client = SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    return stub, calls


def test_complete_books_tag_tokens_as_thinking(client: LLMClient) -> None:
    # The <think>/</think> tags and inter-part whitespace were generated, so
    # they are budget: thinking + answer must cover the FULL raw output.
    raw = "<think>\nweigh the options\n</think>\n\nAction: finish"
    completion = SimpleNamespace(
        choices=[SimpleNamespace(text=raw, finish_reason="stop")], usage=None,
    )
    stub, calls = _stubbed_client(client, completion)
    resp = stub.complete([{"role": "user", "content": "go"}], max_tokens=64)

    tok = client.tokenizer
    assert resp.thinking_tokens + resp.answer_tokens == tok.count_text(raw)
    assert resp.thinking_tokens > tok.count_text("weigh the options")  # tags included
    assert resp.answer == "Action: finish"  # parsed answer stays clean
    # The request asks vLLM for generated token ids (ignored elsewhere).
    assert calls[0]["extra_body"]["return_token_ids"] is True


def test_complete_prefers_endpoint_token_ids(client: LLMClient) -> None:
    # When vLLM returns generated token ids, the split is done on the ids at the
    # last </think> id -- exact, no re-tokenisation of decoded text.
    end_id = client.tokenizer.end_think_token_id
    ids = [11, 22, 33, end_id, 44, 55]
    completion = SimpleNamespace(
        choices=[SimpleNamespace(text="<think>xx</think>yy", finish_reason="stop",
                                 token_ids=ids)],
        usage=None,
    )
    stub, _ = _stubbed_client(client, completion)
    resp = stub.complete([{"role": "user", "content": "go"}], max_tokens=64)
    assert resp.thinking_tokens == 4  # up to and including </think>
    assert resp.answer_tokens == 2


def test_complete_token_ids_truncated_thinking_is_all_thinking(client: LLMClient) -> None:
    # Length stop before </think>: the whole output is thinking, answer empty ->
    # the caller's empty-post-think rule routes straight to finalise.
    ids = [11, 22, 33, 44]
    completion = SimpleNamespace(
        choices=[SimpleNamespace(text="<think>cut off mid-reason", finish_reason="length",
                                 token_ids=ids)],
        usage=None,
    )
    stub, _ = _stubbed_client(client, completion)
    resp = stub.complete([{"role": "user", "content": "go"}], max_tokens=4)
    assert resp.thinking_tokens == 4 and resp.answer_tokens == 0
    assert resp.answer == ""


# --------------------------------------------------------------------------- #
# Live: end-to-end transport + counting + ledger balance.
# --------------------------------------------------------------------------- #


def test_live_completion_balances_ledger(client: LLMClient) -> None:
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Reply with exactly: OK"},
    ]
    ledger = BudgetLedger(cap=2000, finalization_reserve=128)
    input_tokens = client.count_input(messages)
    budget = ledger.max_new_tokens(input_tokens, finalizing=False)

    try:
        resp = client.complete(messages, max_tokens=min(budget, 64))
    except Exception as exc:  # no endpoint reachable
        pytest.skip(f"no live endpoint: {exc}")

    assert resp.input_tokens == input_tokens > 0
    assert resp.answer_tokens >= 0 and resp.thinking_tokens >= 0
    ledger.record_call(
        input_tokens=resp.input_tokens,
        thinking_tokens=resp.thinking_tokens,
        answer_tokens=resp.answer_tokens,
    )
    # The ledger's identity must hold and stay within the cap.
    assert ledger.total == resp.input_tokens + resp.thinking_tokens + resp.answer_tokens
    assert ledger.total <= ledger.cap
