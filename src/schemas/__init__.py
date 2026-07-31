# src/schemas/__init__.py
"""Layer-0 contracts for the thesis harness.

Three versioned schemas, all JSON-serialisable via pydantic v2:

  * Instance       - one OPTW meeting-planning task (generator -> solver/agents)
  * AnswerContract - the plan an agent emits (agents -> validator/scorer)
  * RunResult      - one structured log record (harness -> analysis)

Nothing else in the codebase should define these shapes; import them from here.
"""

from __future__ import annotations

from .answer import ANSWER_SCHEMA_VERSION, AnswerContract, Meeting
from .common import Condition, InvalidReason, Level
from .instance import (
    INSTANCE_SCHEMA_VERSION,
    GeneratorParams,
    Instance,
    Person,
    TravelStructure,
)
from .result import (
    RESULT_SCHEMA_VERSION,
    CallRecord,
    RunResult,
    SamplingConfig,
    Score,
    TokenUsage,
)

__all__ = [
    "Instance",
    "Person",
    "GeneratorParams",
    "TravelStructure",
    "INSTANCE_SCHEMA_VERSION",
    "AnswerContract",
    "Meeting",
    "ANSWER_SCHEMA_VERSION",
    "RunResult",
    "TokenUsage",
    "Score",
    "CallRecord",
    "SamplingConfig",
    "RESULT_SCHEMA_VERSION",
    "Condition",
    "Level",
    "InvalidReason",
]
