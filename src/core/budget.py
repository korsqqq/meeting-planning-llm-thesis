# src/core/budget.py
"""Single shared token budget: ledger + guard.

THESIS_DECISIONS.md section 2: one ledger per instance, shared by every call (the
same object for C1's many calls and C3's several agents). All tokens of all calls
count -- input + thinking + answer. A fixed slice is reserved so a final answer is
always producible, and the harness routes to the finalisation node once the budget
is too small for another reasoning call; on true exhaustion it force-stops and
returns best_plan_so_far.

The reserve is a single per-instance allowance, applied as a constant FLOOR on
`remaining` that ordinary calls cannot dip into -- it is NOT re-deducted on each call.
So C1 (many calls) and C3/C4 (many calls across several agents sharing this one
ledger) all get the same `cap - finalization_reserve` of working tokens; this is what
keeps "equal budget" actually equal across conditions, rather than penalising the
multi-step systems.

Framework-agnostic on purpose: LangGraph does not exist yet, so the future graph
nodes will call this object. It has no LangGraph dependency and is fully testable on
plain integers -- no GPU, no network, no tokenizer.

The guard side is advisory arithmetic: the harness asks `max_new_tokens(...)` before
a call and `should_finalize(...)` to decide routing; `record_call(...)` then books
the realised counts. record_call refuses a call that would push the total over the
cap, because the guard is supposed to have capped max_new_tokens first -- an overspend
means a harness bug, not a silently truncated ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas import TokenUsage

__all__ = ["DEFAULT_FINALIZATION_RESERVE", "DEFAULT_MAX_CALL_TOKENS", "BudgetLedger"]

# Tokens held back for the finalisation node so a plan can always be emitted.
# [PILOT] -- the exact value is confirmed once real answer sizes are observed.
DEFAULT_FINALIZATION_RESERVE = 256

# Candidate ceiling on what ONE ordinary reasoning call may generate, independent of
# how much budget is left. Motivation: without a ceiling the guard hands a single call
# the whole remainder, and a thinking block that never closes destroys all of it at
# once (Qwen3-8B-AWQ at cap 16000: one call spent 5083 tokens and returned nothing
# after </think>).
#
# NOT enabled by default, and the reason is measured, not theoretical. Enabling it at
# 2000 made things strictly worse: truncating a think block leaves nothing after
# </think>, which the section-2 rule reads as "the model produced nothing" and routes
# straight to finalisation. Both models then died on their ninth step with the budget
# still half full, and C1 on Qwen3-32B-AWQ fell from satisfaction 1.00 to 0.00 at cap
# 16000. A ceiling therefore cannot be introduced on its own: it needs a way to tell
# "the harness cut this call short" apart from "the agent had nothing to say".
# [PILOT] -- decide the ceiling and that distinction together, before the main run.
DEFAULT_MAX_CALL_TOKENS = 2000


@dataclass
class BudgetLedger:
    """Running token ledger for one instance, with the budget-guard arithmetic.

    `cap` is the budget; `finalization_reserve` is kept aside for the final answer;
    `route_threshold` is how little reasoning room must remain before the harness
    routes to the final node.
    """

    cap: int
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE
    route_threshold: int = 0
    # Per-call generation ceiling for ordinary reasoning calls. None (the default)
    # means unbounded: see DEFAULT_MAX_CALL_TOKENS for why a ceiling cannot be turned
    # on until truncation is distinguishable from an empty answer.
    max_call_tokens: int | None = None
    input_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    n_calls: int = 0

    def __post_init__(self) -> None:
        if self.cap <= 0:
            raise ValueError(f"cap must be > 0, got {self.cap}")
        if self.finalization_reserve < 0:
            raise ValueError(
                f"finalization_reserve must be >= 0, got {self.finalization_reserve}"
            )
        if self.finalization_reserve >= self.cap:
            raise ValueError(
                f"finalization_reserve ({self.finalization_reserve}) must be smaller "
                f"than cap ({self.cap})"
            )
        if self.route_threshold < 0:
            raise ValueError(f"route_threshold must be >= 0, got {self.route_threshold}")
        if self.max_call_tokens is not None and self.max_call_tokens <= 0:
            raise ValueError(
                f"max_call_tokens must be > 0 or None, got {self.max_call_tokens}"
            )

    # --- running totals --------------------------------------------------- #
    @property
    def total(self) -> int:
        return self.input_tokens + self.thinking_tokens + self.answer_tokens

    @property
    def remaining(self) -> int:
        return self.cap - self.total

    @property
    def budget_exhausted(self) -> bool:
        """True when no tokens are left at all."""
        return self.remaining <= 0

    # --- guard arithmetic ------------------------------------------------- #
    def max_new_tokens(self, input_tokens: int, *, finalizing: bool = False) -> int:
        """How many new tokens a call may generate, given its prompt size.

        Leaves the finalisation reserve untouched for ordinary calls; a finalising
        call may spend it. Never negative.

        Ordinary calls are additionally capped at `max_call_tokens`, so one runaway
        thinking block cannot consume the whole remaining budget. The finalising call
        is deliberately NOT capped: it runs with thinking OFF under a JSON schema, so
        it cannot ramble, and capping it would change the reserve arithmetic that
        keeps "equal budget" equal across conditions.
        """
        room = self.remaining - input_tokens
        if not finalizing:
            room -= self.finalization_reserve
            if self.max_call_tokens is not None:
                room = min(room, self.max_call_tokens)
        return max(0, room)

    def should_finalize(self, next_input_tokens: int = 0) -> bool:
        """True when an ordinary call no longer fits -> route to the final node."""
        return self.max_new_tokens(next_input_tokens, finalizing=False) <= self.route_threshold

    # --- booking ---------------------------------------------------------- #
    def record_call(
        self, *, input_tokens: int, thinking_tokens: int, answer_tokens: int
    ) -> None:
        """Book one model call's realised token counts into the ledger."""
        for name, value in (
            ("input_tokens", input_tokens),
            ("thinking_tokens", thinking_tokens),
            ("answer_tokens", answer_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        new_total = self.total + input_tokens + thinking_tokens + answer_tokens
        if new_total > self.cap:
            raise ValueError(
                f"call would overspend the budget: {new_total} > cap {self.cap}; "
                "cap max_new_tokens with the guard before issuing the call"
            )

        self.input_tokens += input_tokens
        self.thinking_tokens += thinking_tokens
        self.answer_tokens += answer_tokens
        self.n_calls += 1

    def to_token_usage(self, *, budget_exhausted: bool | None = None) -> TokenUsage:
        """Snapshot the ledger as the Layer-0 TokenUsage record.

        `budget_exhausted` defaults to the literal "nothing left" test; the harness
        may pass True explicitly when it force-stopped for budget reasons while a
        sub-reserve remainder was still on the clock.
        """
        exhausted = self.budget_exhausted if budget_exhausted is None else budget_exhausted
        return TokenUsage(
            cap=self.cap,
            total=self.total,
            input_tokens=self.input_tokens,
            thinking_tokens=self.thinking_tokens,
            answer_tokens=self.answer_tokens,
            n_calls=self.n_calls,
            budget_exhausted=exhausted,
        )
