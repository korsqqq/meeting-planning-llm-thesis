# src/schemas/instance.py
"""Instance schema: one meeting-planning task (Orienteering Problem with Time
Windows).

An instance is produced by the generator and is the single object that the
solver, the validator, and every agent condition read. It carries the raw task
only. The complexity metric and level are filled in AFTER the solver runs and
the pilot fixes the bins (THESIS_DECISIONS.md section 3), so they are optional
and None at generation time.

All times are integer minutes from the start of the day. Times never use floats.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import Level

INSTANCE_SCHEMA_VERSION = "instance/1.0"


class TravelStructure(StrEnum):
    """How the travel-time matrix is shaped by the generator. Held as an explicit
    field so the complexity diagnostic and the analysis can group by it."""

    UNIFORM = "uniform"        # roughly equal distances
    CLUSTERED = "clustered"    # tight geographic clusters, far between clusters
    LINE = "line"             # locations along a line / corridor
    RANDOM = "random"


class GeneratorParams(BaseModel):
    """The knobs the generator was called with. Stored verbatim for
    reproducibility; complexity is measured from the solved instance, NOT
    inferred from these."""

    model_config = ConfigDict(extra="forbid")

    n_people: int = Field(ge=1)
    tightness: float = Field(ge=0.0, le=1.0)   # how tight the availability windows are
    overlap: float = Field(ge=0.0, le=1.0)     # how much the windows overlap
    travel_structure: TravelStructure


class Person(BaseModel):
    """One person who can be met, with their location and availability window."""

    model_config = ConfigDict(extra="forbid")

    person_id: str
    location: str
    window_start: int = Field(ge=0)  # minutes; earliest the meeting may start
    window_end: int = Field(ge=0)    # minutes; latest the meeting may end

    @model_validator(mode="after")
    def _window_ordered(self) -> "Person":
        if self.window_end <= self.window_start:
            raise ValueError(
                f"person {self.person_id}: window_end ({self.window_end}) must be "
                f"greater than window_start ({self.window_start})"
            )
        return self


class Instance(BaseModel):
    """A single meeting-planning task.

    Construction order for the harness: the generator emits an Instance with
    `complexity_metric` and `level` set to None; the oracle solves it and the
    pilot binning fills those two fields in place before the main sweep.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = INSTANCE_SCHEMA_VERSION
    instance_id: str
    seed: int
    generator_params: GeneratorParams

    start_location: str
    start_time: int = Field(ge=0)
    end_of_day: int = Field(ge=0)      # hard deadline; every meeting must end by this
    meeting_duration: int = Field(gt=0)  # fixed, identical for all meetings
    waiting_allowed: bool
    tie_break: str                     # deterministic rule between equally good routes

    locations: list[str]
    people: list[Person]
    # travel_times[a][b] = minutes to go from location a to location b.
    travel_times: dict[str, dict[str, int]]

    # Filled after solving / pilot binning. None at generation time.
    complexity_metric: int | None = Field(
        default=None,
        description="Count of binding pairwise conflict pairs in the conflict "
        "graph (solver-verified). None until the instance is solved.",
    )
    level: Level | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "Instance":
        if self.end_of_day <= self.start_time:
            raise ValueError("end_of_day must be greater than start_time")

        loc_set = set(self.locations)

        if self.start_location not in loc_set:
            raise ValueError(f"start_location {self.start_location!r} not in locations")

        for p in self.people:
            if p.location not in loc_set:
                raise ValueError(
                    f"person {p.person_id}: location {p.location!r} not in locations"
                )

        # The travel matrix must cover every ordered pair of locations, so a bug
        # in the generator surfaces here rather than as a KeyError mid-run.
        for a in self.locations:
            if a not in self.travel_times:
                raise ValueError(f"travel_times missing row for location {a!r}")
            row = self.travel_times[a]
            missing = loc_set - set(row)
            if missing:
                raise ValueError(
                    f"travel_times[{a!r}] missing destinations: {sorted(missing)}"
                )

        # Person ids must be unique.
        ids = [p.person_id for p in self.people]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate person_id values in people")

        return self

    def travel_time(self, a: str, b: str) -> int:
        """Travel time in minutes from location a to b."""
        return self.travel_times[a][b]

    def person(self, person_id: str) -> Person:
        for p in self.people:
            if p.person_id == person_id:
                return p
        raise KeyError(f"unknown person_id {person_id!r}")
