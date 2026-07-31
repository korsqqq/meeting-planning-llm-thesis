# src/schemas/answer.py
"""Answer contract: the exact plan an agent must emit.

Every condition (C1-C4) returns this same object. The validator and scorer read
it together with the Instance. Keeping it minimal makes parsing robust and lets
the JSON schema drive vLLM guided decoding:

    from schemas.answer import AnswerContract
    guided_json = AnswerContract.model_json_schema()

The meeting order is the list order. The meeting end time is implicit:
start_time + instance.meeting_duration. An empty `meetings` list is a valid,
well-formed answer (it scores 0 when the solver optimum is positive).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ANSWER_SCHEMA_VERSION = "answer/1.0"


class Meeting(BaseModel):
    """A single scheduled meeting: who, and when it starts (minutes)."""

    model_config = ConfigDict(extra="forbid")

    person_id: str
    start_time: int = Field(ge=0)


class AnswerContract(BaseModel):
    """The plan returned by an agent. Order of `meetings` is the visiting order."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ANSWER_SCHEMA_VERSION
    meetings: list[Meeting] = Field(default_factory=list)

    @classmethod
    def empty(cls) -> "AnswerContract":
        """An empty plan. Used as the initial best_plan_so_far before the agent
        has produced any valid intermediate plan."""
        return cls(meetings=[])
