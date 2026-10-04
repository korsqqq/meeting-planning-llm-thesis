# src/harness/smoke_client.py
"""Offline scripted client for the harness vertical slice. NOT a model.

Purpose: exercise the full plumbing (prompt -> tool loop -> budget ledger ->
finalisation -> validator/scorer -> JSON) deterministically, with no endpoint, no
GPU and no network. It follows the exact `LLMClient` call interface (`count_input`
/ `complete`) so the agent graphs run unchanged.

What it is NOT: evidence about model behaviour. Runs made with this client do not
close the live vLLM checkpoint, say nothing about thinking-mode budget dynamics,
and must never be reported as model results (`model_label` should name it
explicitly, e.g. "offline-smoke-client").

How it acts: a tiny rule-based planner over the SAME information a real model
would see -- it reads the task facts from the system prompt and the tool
Observations in the conversation, never from the Instance object or the oracle.
It lists people, probes them in listed order, proposes the first single meeting
that fits (by its own arithmetic over the observed window + travel time), then
finishes; under C2 it answers the verify prompt with a short reflection and
re-proposes its plan on revise. The terminal emit echoes the plan text it is
given as JSON. Token counts are deterministic word counts, NOT Qwen tokenizer
counts -- fine for plumbing, meaningless for calibration.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from src.core import LLMResponse

__all__ = ["OfflineSmokeClient"]

_AVAILABILITY_RE = re.compile(r"availability\((\w+)\)\s*=\s*(\{.*\})", re.DOTALL)
_TRAVEL_RE = re.compile(r"travel_time\(.+?->.+?\)\s*=\s*(\d+)\s*min")
_START_RE = re.compile(r"start at location '([^']+)' at minute (\d+)")
_DURATION_RE = re.compile(r"lasts exactly (\d+) minutes")
_END_OF_DAY_RE = re.compile(r"must end by minute (\d+)")


class OfflineSmokeClient:
    """Deterministic scripted stand-in for LLMClient (offline plumbing only)."""

    def __init__(self) -> None:
        # Task facts, learned only from the prompt/observations.
        self._start_loc: str | None = None
        self._start_time = 0
        self._duration = 0
        self._end_of_day = 0
        self._waiting_allowed = True
        # Probe state.
        self._queue: list[str] = []
        self._current: dict[str, Any] | None = None  # availability of the probed person
        self._plan: str | None = None                # "p0@25" once something fits
        self.n_completions = 0

    # ------------------------------------------------------------------ #
    # LLMClient interface.
    # ------------------------------------------------------------------ #
    def count_input(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool = True,
        tools: Any = None,
    ) -> int:
        """Deterministic pseudo-count: words + a flat per-message overhead."""
        return sum(len(m["content"].split()) + 4 for m in messages)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        enable_thinking: bool = True,
        tools: Any = None,
        guided_json: Any = None,
        stop: Any = None,
    ) -> LLMResponse:
        self.n_completions += 1
        self._learn_task_facts(messages)
        if guided_json is not None or not enable_thinking:
            answer = self._serialise_final(messages)
        else:
            answer = self._next_move(messages)
        return LLMResponse(
            thinking="",
            answer=answer,
            raw_text=answer,
            input_tokens=self.count_input(messages, enable_thinking=enable_thinking),
            thinking_tokens=0,
            answer_tokens=len(answer.split()),
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ #
    # Rule-based policy over the visible conversation.
    # ------------------------------------------------------------------ #
    def _learn_task_facts(self, messages: list[dict[str, Any]]) -> None:
        if self._start_loc is not None or not messages:
            return
        system = messages[0]["content"]
        if m := _START_RE.search(system):
            self._start_loc = m.group(1)
            self._start_time = int(m.group(2))
        if m := _DURATION_RE.search(system):
            self._duration = int(m.group(1))
        if m := _END_OF_DAY_RE.search(system):
            self._end_of_day = int(m.group(1))
        self._waiting_allowed = "You cannot wait" not in system

    def _next_move(self, messages: list[dict[str, Any]]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # C2 reflection turns.
        if "Verification step." in last_user:
            return (
                "Thought: I checked each window and travel time I observed; "
                "the current plan is the best I can justify with them."
            )
        if "Revision step." in last_user:
            if self._plan is not None:
                return f"Action: propose[{self._plan}]"
            return "No feasible meeting was found earlier; there is nothing to change."

        # C1 draft loop.
        if "People:" in last_user:
            people_part = last_user.split("People:", 1)[1].strip()
            self._queue = [p.strip() for p in people_part.split(",") if p.strip()]
            return self._probe_next_or_finish()
        if match := _AVAILABILITY_RE.search(last_user):
            self._current = ast.literal_eval(match.group(2))
            return (
                f"Action: get_travel_time[{self._start_loc}, "
                f"{self._current['location']}]"
            )
        if match := _TRAVEL_RE.search(last_user):
            move = self._try_schedule(travel_minutes=int(match.group(1)))
            if move is not None:
                return move
            return self._probe_next_or_finish()
        if "Recorded a plan" in last_user:
            return "Action: finish"
        # Initial "Plan the route now" turn (and any unrecognised nudge).
        return "Action: list_people[]"

    def _probe_next_or_finish(self) -> str:
        if self._queue:
            return f"Action: get_availability[{self._queue.pop(0)}]"
        return "Action: finish"

    def _try_schedule(self, *, travel_minutes: int) -> str | None:
        """Propose the probed person if one meeting fits, using only observed facts."""
        person = self._current
        self._current = None
        if person is None:
            return None
        arrival = self._start_time + travel_minutes
        start = max(arrival, person["window_start"]) if self._waiting_allowed else arrival
        if start < person["window_start"]:
            return None  # cannot wait for the window to open
        end = start + self._duration
        if end > person["window_end"] or end > self._end_of_day:
            return None
        self._plan = f"{person['person_id']}@{start}"
        return f"Action: propose[{self._plan}]"

    def _serialise_final(self, messages: list[dict[str, Any]]) -> str:
        """Terminal emit: echo the 'Plan: ...' payload of the finalize prompt as JSON."""
        prompt = messages[-1]["content"] if messages else ""
        plan_line = prompt.split("\n", 1)[0]
        plan_text = plan_line.partition("Plan:")[2].strip()
        meetings = []
        if plan_text and plan_text != "(none)":
            for token in plan_text.split(","):
                pid, _, start = token.strip().partition("@")
                meetings.append(
                    f'{{"person_id": "{pid.strip()}", "start_time": {int(start)}}}'
                )
        return '{"meetings": [' + ", ".join(meetings) + "]}"
