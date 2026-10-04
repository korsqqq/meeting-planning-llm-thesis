# src/schemas/result.py
"""Run-result schema: one structured JSON record per run.

A run = one agent execution on one cell (condition x level x cap x instance).
The harness writes exactly one RunResult per run. Two design requirements from
THESIS_DECISIONS.md are encoded here:

  * `best_plan_so_far` is stored SEPARATELY from `final_plan` (hard requirement).
  * token accounting is a single ledger whose total equals
    input + thinking + answer, and thinking tokens are logged separately so
    `satisfaction / 1k tokens` can be computed both full-call and answer-only.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .answer import AnswerContract
from .common import Condition, InvalidReason, Level

RESULT_SCHEMA_VERSION = "result/1.0"


class SamplingConfig(BaseModel):
    """Snapshot of the decoding settings used for this run. Stored so every log
    is self-describing and the thinking/greedy decision is auditable."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float = 0.0
    thinking_enabled: bool = True
    seed: int


class CallRecord(BaseModel):
    """Token and timing breakdown for one model call. In MAS, `role` identifies
    which agent made the call (supervisor / worker_a / worker_b / aggregator /
    critic); in single-agent conditions it is a step label."""

    model_config = ConfigDict(extra="forbid")

    role: str
    input_tokens: int = Field(ge=0)      # prompt + history + tool schemas + inter-agent msgs
    thinking_tokens: int = Field(ge=0)   # tokens inside the <think> block
    answer_tokens: int = Field(ge=0)     # tokens of the visible answer
    latency_seconds: float = Field(ge=0.0)
    # Why the endpoint stopped generating, verbatim ("stop", "length", ...). Diagnostic
    # only: nothing routes on it. It is what separates a generation cut short by the
    # per-call grant from a model that closed </think> and then emitted nothing -- the
    # two look identical in the token counts, and the empty-post-think rule (section 2)
    # treats them the same. Optional so that offline scripted clients, which have no
    # endpoint to report one, stay valid.
    finish_reason: str | None = None
    # Context-window accounting: what the budget guard granted, what was actually sent
    # after the deployment's window was applied, and whether the two differ. Diagnostic
    # only -- the ledger books the REALISED tokens, so a limited call simply leaves its
    # unspent grant in the budget. None on offline clients, which have no window.
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    context_limited: bool = False
    # C5 only (§4 C5): which of the three Best-of-3 search trajectories issued this call,
    # and the sampling seed it actually used. Optional and defaulting to None so every
    # document written before C5 existed, and every call in C1-C4, stays valid unchanged
    # -- the sweep runner's schema check and the existing analysers are unaffected. The
    # single finalisation call never carries them: it is outside the attempt sequence.
    attempt_index: int | None = None
    sampling_seed: int | None = None


class TokenUsage(BaseModel):
    """The single budget ledger for the whole instance. `total` must equal
    input + thinking + answer (all counted with the Qwen tokenizer)."""

    model_config = ConfigDict(extra="forbid")

    cap: int = Field(gt=0)
    total: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    thinking_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    n_calls: int = Field(ge=0)
    budget_exhausted: bool

    @model_validator(mode="after")
    def _ledger_balances(self) -> "TokenUsage":
        parts = self.input_tokens + self.thinking_tokens + self.answer_tokens
        if parts != self.total:
            raise ValueError(
                f"token ledger does not balance: input+thinking+answer={parts} "
                f"!= total={self.total}"
            )
        if self.total > self.cap:
            raise ValueError(f"total spent {self.total} exceeds cap {self.cap}")
        return self


class Score(BaseModel):
    """Evaluation outcome. `satisfaction` is the primary metric; an invalid plan
    scores 0 regardless of how many meetings it contained."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    satisfaction: float = Field(ge=0.0, le=1.0)
    n_valid_meetings: int = Field(ge=0)
    solver_optimum: int = Field(ge=0)
    optimality: bool                       # reached the solver optimum exactly
    invalid_reasons: list[InvalidReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _invalid_scores_zero(self) -> "Score":
        if not self.valid and self.satisfaction != 0.0:
            raise ValueError("invalid plan must have satisfaction == 0.0")
        if self.valid and self.invalid_reasons:
            raise ValueError("valid plan must have an empty invalid_reasons list")
        return self


class RunResult(BaseModel):
    """One full run record. This is the unit the analysis layer aggregates over."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RESULT_SCHEMA_VERSION
    run_id: str
    instance_id: str
    seed: int

    # Experimental cell.
    condition: Condition
    level: Level
    cap: int = Field(gt=0)
    model: str                              # e.g. "Qwen/Qwen3-32B-AWQ"
    sampling: SamplingConfig

    # Plans: the emitted answer and the best valid intermediate plan, kept apart.
    final_plan: AnswerContract | None
    best_plan_so_far: AnswerContract | None

    tokens: TokenUsage
    score: Score
    latency_seconds: float = Field(ge=0.0)
    calls: list[CallRecord] = Field(default_factory=list)

    timestamp: str                          # ISO 8601, UTC
    notes: str | None = None

    def sat_per_1k_total(self) -> float:
        """Satisfaction per 1000 total tokens (full call cost)."""
        if self.tokens.total == 0:
            return 0.0
        return self.score.satisfaction / self.tokens.total * 1000.0

    def sat_per_1k_answer(self) -> float:
        """Satisfaction per 1000 answer tokens (planning-only efficiency)."""
        if self.tokens.answer_tokens == 0:
            return 0.0
        return self.score.satisfaction / self.tokens.answer_tokens * 1000.0
