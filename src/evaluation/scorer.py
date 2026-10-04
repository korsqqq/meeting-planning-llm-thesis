# src/evaluation/scorer.py
"""Scoring: turn a plan + the oracle optimum into a Score. Correctness only.

THESIS_DECISIONS.md section 5. The scorer is CPU-only and deliberately knows nothing
about tokens -- the efficiency metric (satisfaction / 1k) joins Score with TokenUsage
at aggregation (RunResult), never here, so this module stays a pure correctness check.

Three correctness invariants:

  * Same objective on both sides. `satisfaction = achieved_objective / solver_optimum`,
    and the achieved objective is the SAME expression as the solver's objective. The
    OPTW here uses unit prizes (every met person = 1), so the objective is the count of
    met people and numerator and denominator are both that count. If prizes ever become
    non-uniform, BOTH the solver and this function must switch to the weighted sum
    together -- a raw count would then give the wrong partial credit.
  * Independent revalidation. The scorer never trusts the agent's self-report: it runs
    the hidden validator on the raw plan. Any hard violation makes the WHOLE plan
    invalid and scores 0 (no partial credit). Only a fully valid plan earns its objective.
  * optimum == 0. A degenerate instance where nothing is feasible: a valid plan is then
    necessarily empty and has achieved the (zero) optimum, so it scores 1.0 -- this
    avoids a 0/0 and is excluded from the primary analysis at aggregation (section 5).
"""

from __future__ import annotations

from src.oracle import validate
from src.schemas import AnswerContract, Instance, Score

__all__ = ["score_plan"]


def _achieved_objective(plan: AnswerContract) -> int:
    """Achieved objective under unit prizes: the number of met people.

    Mirrors the solver objective (maximise the sum of visit indicators). In a valid
    plan each listed person is met exactly once, so the objective is len(meetings).
    """
    return len(plan.meetings)


def score_plan(
    instance: Instance,
    plan: AnswerContract | None,
    solver_optimum: int,
) -> Score:
    """Score a (possibly malformed) plan against the oracle optimum.

    `solver_optimum` must come from the oracle (`solve(instance).optimum`): the same
    objective this function measures. `plan is None` means the raw agent output could
    not be parsed (the validator returns MALFORMED).
    """
    if solver_optimum < 0:
        raise ValueError(f"solver_optimum must be >= 0, got {solver_optimum}")

    reasons = validate(instance, plan)  # independent hidden validator, not self-report
    if reasons:
        # Any hard violation -> whole plan invalid -> 0, no partial credit.
        return Score(
            valid=False,
            satisfaction=0.0,
            n_valid_meetings=0,
            solver_optimum=solver_optimum,
            optimality=False,
            invalid_reasons=reasons,
        )

    # Valid plan: `plan` is not None and every listed meeting is feasible.
    assert plan is not None  # validate(None) -> [MALFORMED], handled above
    achieved = _achieved_objective(plan)

    if solver_optimum == 0:
        # Nothing is feasible; a valid plan is empty and has reached the 0 optimum.
        return Score(
            valid=True,
            satisfaction=1.0,
            n_valid_meetings=achieved,  # 0
            solver_optimum=0,
            optimality=True,
            invalid_reasons=[],
        )

    if achieved > solver_optimum:
        # A feasible plan cannot beat the proven optimum; this means the solver and
        # the validator disagree -- a bug in one of them. Fail loudly.
        raise ValueError(
            f"valid plan meets {achieved} people but the proven optimum is "
            f"{solver_optimum}; solver/validator disagree -- stop and fix"
        )

    return Score(
        valid=True,
        satisfaction=achieved / solver_optimum,
        n_valid_meetings=achieved,
        solver_optimum=solver_optimum,
        optimality=achieved == solver_optimum,
        invalid_reasons=[],
    )
