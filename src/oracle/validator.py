# src/oracle/validator.py
"""Hidden hard-constraint validator (independent of the solver).

THESIS_DECISIONS.md section 5 / PROGRESS.md: a plan is valid only if it violates
no hard constraint; any violation makes the whole plan invalid and scores 0 (no
partial credit). This module is the single authority on validity. It is kept
INDEPENDENT of solver.py and brute_force.py so a bug in the oracle cannot hide a
bug in the validator or vice versa.

Unlike the oracle, the validator does not choose start times: it checks the
explicit `start_time` the agent put on every Meeting, in the order the meetings
are listed (list order == visiting order, per AnswerContract). It returns the
multi-label list of InvalidReason values (empty list == valid).

Reason mapping (see schemas/common.py InvalidReason):
  * MALFORMED          -- the raw output could not be parsed (answer is None).
  * UNKNOWN_PERSON     -- a meeting names a person not in the instance.
  * DUPLICATE_MEETING  -- the same person is met more than once.
  * WINDOW_VIOLATION   -- a meeting falls outside the person's availability window.
  * DEADLINE_EXCEEDED  -- a meeting ends after end_of_day.
  * OVERLAPPING_MEETING-- a meeting starts before the previous one ends.
  * TRAVEL_INFEASIBLE  -- too little time to travel from the previous stop; also
                          covers illegal idle time when waiting_allowed is False
                          (the start must equal the arrival time).
"""

from __future__ import annotations

from src.schemas import AnswerContract, Instance, InvalidReason, Person

__all__ = ["validate", "is_valid"]


def validate(instance: Instance, answer: AnswerContract | None) -> list[InvalidReason]:
    """Return the sorted, de-duplicated list of hard-constraint violations.

    An empty list means the plan is valid. `answer is None` signals a parse
    failure upstream and yields [MALFORMED].
    """
    if answer is None:
        return [InvalidReason.MALFORMED]

    reasons: set[InvalidReason] = set()
    by_id: dict[str, Person] = {p.person_id: p for p in instance.people}
    duration = instance.meeting_duration

    # Pass 1: identity checks (unknown person, duplicates).
    seen: set[str] = set()
    for m in answer.meetings:
        if m.person_id not in by_id:
            reasons.add(InvalidReason.UNKNOWN_PERSON)
        if m.person_id in seen:
            reasons.add(InvalidReason.DUPLICATE_MEETING)
        seen.add(m.person_id)

    # Pass 2: per-meeting window and deadline checks.
    for m in answer.meetings:
        person = by_id.get(m.person_id)
        if person is None:
            continue  # already flagged UNKNOWN_PERSON; no window to check against
        end = m.start_time + duration
        if m.start_time < person.window_start or end > person.window_end:
            reasons.add(InvalidReason.WINDOW_VIOLATION)
        if end > instance.end_of_day:
            reasons.add(InvalidReason.DEADLINE_EXCEEDED)

    # Pass 3: sequential travel / overlap checks. The depot is the virtual stop
    # before the first meeting (location start_location, available from start_time).
    prev_loc: str | None = instance.start_location
    prev_end = instance.start_time
    for idx, m in enumerate(answer.meetings):
        person = by_id.get(m.person_id)
        if person is None:
            # Cannot reason about travel across an unknown location: break the
            # chain and resume from the next known meeting.
            prev_loc = None
            prev_end = m.start_time + duration
            continue
        if prev_loc is None:
            # The previous stop was unknown; skip this hop's travel check.
            prev_loc = person.location
            prev_end = m.start_time + duration
            continue

        travel = instance.travel_time(prev_loc, person.location)
        required = prev_end + travel

        if idx > 0 and m.start_time < prev_end:
            # Two real meetings overlap in time.
            reasons.add(InvalidReason.OVERLAPPING_MEETING)
        elif m.start_time < required:
            # Not enough time to arrive (covers the depot -> first-meeting hop too).
            reasons.add(InvalidReason.TRAVEL_INFEASIBLE)
        elif not instance.waiting_allowed and m.start_time != required:
            # Idle time is not allowed: the start must equal the arrival time.
            reasons.add(InvalidReason.TRAVEL_INFEASIBLE)

        prev_loc = person.location
        prev_end = m.start_time + duration

    return sorted(reasons, key=lambda r: r.value)


def is_valid(instance: Instance, answer: AnswerContract | None) -> bool:
    """True iff the plan violates no hard constraint."""
    return not validate(instance, answer)
