# tests/test_context_window.py
"""Context-window guard: the deployment's physical limit, not a budget policy.

The budget guard says how much a call MAY spend; the window says how much the served
model can physically hold. The smaller of the two is what goes on the wire. These tests
pin down that the endpoint receives the effective limit rather than the requested one,
that the ledger and the instance cap are untouched by the clamp, that a prompt which
alone exceeds the window fails before any request is made, and that `max_model_len=None`
leaves the old behaviour exactly as it was for offline and scripted clients.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core import BudgetLedger, ContextWindowError, LLMClient


class _FakeCompletions:
    """Captures what the client actually sends, and answers with a fixed completion."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)

        class _Choice:
            text = "<think>t</think>done"
            finish_reason = "stop"

        class _Completion:
            choices = [_Choice()]
            usage = None

        return _Completion()


class _FakeOpenAI:
    def __init__(self, *a: Any, **kw: Any) -> None:
        self.completions = _FakeCompletions()


class _FixedTokenizer:
    """Counts a fixed number of input tokens; output counting falls back to text."""

    def __init__(self, input_tokens: int) -> None:
        self._input = input_tokens
        self.end_think_id = None

    def render_input(self, messages, *, tools=None, enable_thinking=True) -> str:
        return "PROMPT"

    def count_text(self, text: str) -> int:
        return self._input if text == "PROMPT" else max(1, len(text) // 4)

    def count_input(self, messages, *, tools=None, enable_thinking=True) -> int:
        return self._input


def _client(monkeypatch, *, input_tokens: int, max_model_len: int | None) -> LLMClient:
    import src.core.llm_client as mod
    monkeypatch.setattr(mod, "OpenAI", _FakeOpenAI)
    return LLMClient(
        tokenizer=_FixedTokenizer(input_tokens),  # type: ignore[arg-type]
        model="test-model", base_url="http://fake", max_model_len=max_model_len,
    )


def _sent(client: LLMClient) -> dict[str, Any]:
    return client._client.completions.calls[-1]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# The clamp itself.
# --------------------------------------------------------------------------- #
def test_top_rung_request_is_clamped_to_the_window(monkeypatch) -> None:
    # The case the guard exists for: at cap 64000 the budget guard grants ~63.5k, which
    # cannot fit a 32768-token window alongside the prompt.
    client = _client(monkeypatch, input_tokens=225, max_model_len=32768)
    resp = client.complete([{"role": "user", "content": "x"}], max_tokens=63519)

    assert _sent(client)["max_tokens"] == 32768 - 225   # what the endpoint received
    assert resp.requested_max_tokens == 63519
    assert resp.effective_max_tokens == 32768 - 225
    assert resp.context_limited is True


@pytest.mark.parametrize("cap_grant", [7519, 15519, 31519])
def test_lower_rungs_are_untouched_when_they_fit(monkeypatch, cap_grant: int) -> None:
    # 8k / 16k / 32k all fit the window with the prompt, so the guard must be invisible.
    client = _client(monkeypatch, input_tokens=225, max_model_len=32768)
    resp = client.complete([{"role": "user", "content": "x"}], max_tokens=cap_grant)

    assert _sent(client)["max_tokens"] == cap_grant
    assert resp.requested_max_tokens == cap_grant
    assert resp.effective_max_tokens == cap_grant
    assert resp.context_limited is False


def test_no_window_configured_keeps_the_previous_behaviour(monkeypatch) -> None:
    # Offline clients, unit tests and ad-hoc calls have no deployment behind them.
    client = _client(monkeypatch, input_tokens=225, max_model_len=None)
    resp = client.complete([{"role": "user", "content": "x"}], max_tokens=63519)

    assert _sent(client)["max_tokens"] == 63519
    assert resp.requested_max_tokens == 63519
    assert resp.effective_max_tokens == 63519
    assert resp.context_limited is False


def test_a_prompt_that_fills_the_window_fails_before_the_endpoint(monkeypatch) -> None:
    client = _client(monkeypatch, input_tokens=32768, max_model_len=32768)
    with pytest.raises(ContextWindowError, match="nothing left to generate"):
        client.complete([{"role": "user", "content": "x"}], max_tokens=100)
    assert client._client.completions.calls == []  # type: ignore[attr-defined]

    over = _client(monkeypatch, input_tokens=40000, max_model_len=32768)
    with pytest.raises(ContextWindowError):
        over.complete([{"role": "user", "content": "x"}], max_tokens=100)
    assert over._client.completions.calls == []  # type: ignore[attr-defined]


def test_a_nonsense_window_is_rejected_at_construction(monkeypatch) -> None:
    import src.core.llm_client as mod
    monkeypatch.setattr(mod, "OpenAI", _FakeOpenAI)
    with pytest.raises(ValueError, match="max_model_len must be > 0"):
        LLMClient(tokenizer=_FixedTokenizer(10), model="m",  # type: ignore[arg-type]
                  base_url="http://fake", max_model_len=0)


# --------------------------------------------------------------------------- #
# The clamp must not touch the budget.
# --------------------------------------------------------------------------- #
def test_clamping_leaves_the_instance_cap_and_ledger_alone(monkeypatch) -> None:
    # The ledger books REALISED tokens, so a call that was allowed less than the budget
    # granted simply leaves the remainder unspent -- the 64k cap is still a 64k cap.
    ledger = BudgetLedger(cap=64000, finalization_reserve=256)
    client = _client(monkeypatch, input_tokens=225, max_model_len=32768)

    granted = ledger.max_new_tokens(225, finalizing=False)
    assert granted == 64000 - 225 - 256          # the budget is unaware of the window

    resp = client.complete([{"role": "user", "content": "x"}], max_tokens=granted)
    assert resp.context_limited is True
    ledger.record_call(input_tokens=resp.input_tokens,
                       thinking_tokens=resp.thinking_tokens,
                       answer_tokens=resp.answer_tokens)

    assert ledger.cap == 64000
    assert ledger.total == resp.input_tokens + resp.thinking_tokens + resp.answer_tokens
    assert ledger.remaining == 64000 - ledger.total


# --------------------------------------------------------------------------- #
# The fields reach the run record, for reasoning AND finalisation.
# --------------------------------------------------------------------------- #
def test_call_records_carry_the_limits_for_reasoning_and_finalisation() -> None:
    from src.agents.single_agent.react import run_react
    from src.core import LLMResponse
    from tests.test_react import _instance  # the shared two-person fixture

    def _resp(answer: str, requested: int, effective: int) -> LLMResponse:
        return LLMResponse(
            thinking="t", answer=answer, raw_text=f"<think>t</think>{answer}",
            input_tokens=50, thinking_tokens=10, answer_tokens=5, finish_reason="stop",
            requested_max_tokens=requested, effective_max_tokens=effective,
            context_limited=effective < requested,
        )

    class _Client:
        def __init__(self) -> None:
            self._queue = [
                _resp("Action: finish", 60000, 32543),          # reasoning, clamped
                _resp('{"meetings": []}', 300, 300),            # finalisation, not clamped
            ]

        def count_input(self, messages, *, enable_thinking=True, tools=None) -> int:
            return 50

        def complete(self, messages, *, max_tokens, enable_thinking=True, tools=None,
                     guided_json=None, stop=None):
            return self._queue.pop(0)

    result = run_react(_instance(), _Client(), cap=64000)  # type: ignore[arg-type]

    assert [c.role for c in result.calls] == ["react_step", "finalize"]
    reasoning, finalisation = result.calls
    assert reasoning.requested_max_tokens == 60000
    assert reasoning.effective_max_tokens == 32543
    assert reasoning.context_limited is True
    # Finalisation obeys the same rule; here it fits, so it is recorded as unclamped.
    assert finalisation.requested_max_tokens == 300
    assert finalisation.effective_max_tokens == 300
    assert finalisation.context_limited is False


def test_scripted_clients_report_no_limiting_rather_than_inventing_it() -> None:
    from src.agents.single_agent.react import run_react
    from src.harness import OfflineSmokeClient
    from tests.test_react import _instance

    result = run_react(_instance(), OfflineSmokeClient(), cap=8000)  # type: ignore[arg-type]
    assert result.calls, "the offline client must still drive a full run"
    for call in result.calls:
        assert call.requested_max_tokens is None
        assert call.effective_max_tokens is None
        assert call.context_limited is False
