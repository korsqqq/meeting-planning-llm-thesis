# tests/test_budget.py
"""BudgetLedger tests -- pure integers, no GPU/network/tokenizer.

Covers the running totals, the guard arithmetic (reserve, routing, max_new_tokens),
the overspend guard, and the mapping onto the Layer-0 TokenUsage schema.
"""

from __future__ import annotations

import pytest

from src.core import BudgetLedger
from src.schemas import TokenUsage


def test_empty_ledger() -> None:
    led = BudgetLedger(cap=1000, finalization_reserve=200)
    assert led.total == 0
    assert led.remaining == 1000
    assert led.n_calls == 0
    assert not led.budget_exhausted


def test_construction_validation() -> None:
    with pytest.raises(ValueError):
        BudgetLedger(cap=0)
    with pytest.raises(ValueError):
        BudgetLedger(cap=100, finalization_reserve=100)  # reserve must be < cap
    with pytest.raises(ValueError):
        BudgetLedger(cap=100, finalization_reserve=-1)


def test_record_accumulates_and_balances() -> None:
    led = BudgetLedger(cap=10_000, finalization_reserve=256)
    led.record_call(input_tokens=120, thinking_tokens=300, answer_tokens=80)
    led.record_call(input_tokens=200, thinking_tokens=50, answer_tokens=40)
    assert led.input_tokens == 320
    assert led.thinking_tokens == 350
    assert led.answer_tokens == 120
    assert led.total == 320 + 350 + 120
    assert led.n_calls == 2

    usage = led.to_token_usage()
    assert isinstance(usage, TokenUsage)
    # The TokenUsage validator enforces total == input + thinking + answer <= cap.
    assert usage.total == led.total
    assert usage.thinking_tokens == 350


def test_max_new_tokens_respects_reserve() -> None:
    led = BudgetLedger(cap=1000, finalization_reserve=200)
    led.record_call(input_tokens=100, thinking_tokens=100, answer_tokens=0)  # total 200, remaining 800
    # Ordinary call: remaining(800) - input(100) - reserve(200) = 500.
    assert led.max_new_tokens(100, finalizing=False) == 500
    # Finalising call may use the reserve: remaining(800) - input(100) = 700.
    assert led.max_new_tokens(100, finalizing=True) == 700
    # Never negative.
    assert led.max_new_tokens(5000, finalizing=False) == 0


def test_should_finalize_routes_when_reasoning_room_gone() -> None:
    led = BudgetLedger(cap=1000, finalization_reserve=200, route_threshold=0)
    # Plenty of room left -> keep reasoning.
    assert not led.should_finalize(next_input_tokens=100)
    # Burn most of the budget: total 750, remaining 250.
    led.record_call(input_tokens=750, thinking_tokens=0, answer_tokens=0)
    # remaining(250) - input(100) - reserve(200) = -50 <= 0 -> finalize.
    assert led.should_finalize(next_input_tokens=100)


def test_overspend_is_rejected() -> None:
    led = BudgetLedger(cap=500, finalization_reserve=50)
    with pytest.raises(ValueError):
        led.record_call(input_tokens=400, thinking_tokens=80, answer_tokens=80)  # 560 > 500


def test_budget_exhausted_flag_and_usage_override() -> None:
    led = BudgetLedger(cap=300, finalization_reserve=50)
    led.record_call(input_tokens=200, thinking_tokens=60, answer_tokens=40)  # total 300
    assert led.budget_exhausted
    assert led.to_token_usage().budget_exhausted is True
    # Override: harness force-stopped for budget even with a remainder on the clock.
    led2 = BudgetLedger(cap=1000, finalization_reserve=50)
    led2.record_call(input_tokens=100, thinking_tokens=100, answer_tokens=0)
    assert not led2.budget_exhausted
    assert led2.to_token_usage(budget_exhausted=True).budget_exhausted is True


def test_reserve_is_single_constant_not_per_call() -> None:
    # The gap between a finalising and an ordinary grant is always exactly one
    # reserve, no matter how many calls have happened -- so C1 (many calls) and
    # C3/C4 (many calls across agents) get the same cap - reserve of working tokens.
    led = BudgetLedger(cap=10_000, finalization_reserve=200)
    led.record_call(input_tokens=100, thinking_tokens=100, answer_tokens=0)
    assert led.max_new_tokens(0, finalizing=True) - led.max_new_tokens(0, finalizing=False) == 200
    led.record_call(input_tokens=100, thinking_tokens=100, answer_tokens=0)
    assert led.max_new_tokens(0, finalizing=True) - led.max_new_tokens(0, finalizing=False) == 200


def test_guard_protocol_always_leaves_room_to_finalize() -> None:
    # Following the guard protocol (check should_finalize before each ordinary call),
    # the loop terminates and a final answer still fits under the cap.
    led = BudgetLedger(cap=2000, finalization_reserve=300, route_threshold=0)
    input_cost = 120
    while not led.should_finalize(next_input_tokens=input_cost):
        grant = led.max_new_tokens(input_cost, finalizing=False)
        led.record_call(input_tokens=input_cost, thinking_tokens=min(grant, 200), answer_tokens=0)
    assert led.n_calls >= 1
    final_answer = led.max_new_tokens(input_cost, finalizing=True)
    led.record_call(input_tokens=input_cost, thinking_tokens=0, answer_tokens=final_answer)
    assert led.total <= led.cap
