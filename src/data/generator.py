# src/data/generator.py
"""Deterministic generator for meeting-planning instances (OPTW).

Given the four canonical knobs from GeneratorParams (n_people, tightness, overlap,
travel_structure) plus a seed, this emits a fully-formed `Instance` with
`complexity_metric` and `level` left None -- those are filled later by the oracle
and the pilot binning (see complexity.py and THESIS_DECISIONS.md section 3).

Design intent -- the knobs must actually move the binding-conflict count so the
pilot sees a spread from easy to hard (the complexity is MEASURED post-solve, not
assumed):

  * tightness in [0,1]  -> window length. 0 = windows span the whole day (slack,
    easy); 1 = windows barely hold one meeting (no slack, hard).
  * overlap in [0,1]    -> temporal clustering of windows. 0 = window starts spread
    across the day (staggered, easy to do sequentially); 1 = all windows centred on
    the same time (everyone available at once -> far-apart pairs conflict).
  * travel_structure    -> spatial layout, hence the travel matrix:
        UNIFORM   points spread over a square (metric, symmetric)
        CLUSTERED two far-apart clusters (metric, symmetric; matches the C3 split)
        LINE      points along a corridor (metric, symmetric)
        RANDOM    arbitrary asymmetric matrix (no triangle inequality)

Tight windows + high overlap + clustered/line travel => many conflicts (hard);
wide windows + low overlap + uniform travel => few conflicts (easy).

The day length, meeting duration and coordinate scale are generator-level config
with defaults; they are deliberately NOT part of GeneratorParams (which stores only
the four knobs the schema fixes) and are pilot-tunable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

from src.schemas import GeneratorParams, Instance, Person, TravelStructure

__all__ = [
    "DEFAULT_START_TIME",
    "DEFAULT_END_OF_DAY",
    "DEFAULT_MEETING_DURATION",
    "RECOMMENDED_MAX_N",
    "generate_instance",
    "generate_dataset",
]

# Oracle-tractability ceiling. The CP-SAT optimum is proven exactly and fast for
# n <= 9 across every knob setting (the worst case -- clustered travel with medium
# windows -- closes in under ~2s). Past ~10 people the visit-count proof blows up on
# those configs. Complexity in this study is driven by binding conflicts, NOT by
# headcount (THESIS_DECISIONS.md section 3), so staying at or below this size costs
# nothing methodologically. n is not hard-capped here; exceeding it just risks the
# oracle raising on the densest instances.
RECOMMENDED_MAX_N = 9

# --- generator-level defaults (pilot-tunable, not part of GeneratorParams) ----- #
DEFAULT_START_TIME = 0
DEFAULT_END_OF_DAY = 480          # minutes (an 8-hour planning day)
DEFAULT_MEETING_DURATION = 30     # minutes, identical for every meeting

_COORD_SPAN = 100.0               # people live in a [0, span] x [0, span] map
_CLUSTER_SPREAD = 8.0             # std-dev of a CLUSTERED point around its centre
_LINE_JITTER = 3.0               # std-dev off the LINE corridor
_RANDOM_TRAVEL_MIN = 5            # RANDOM structure: travel-time bounds (minutes)
_RANDOM_TRAVEL_MAX = 100

_START_LOC = "start"


def _people_locations(n_people: int) -> list[str]:
    return [f"loc_{i}" for i in range(n_people)]


def _coordinates(
    rng: random.Random, people_locs: Sequence[str], structure: TravelStructure
) -> dict[str, tuple[float, float]]:
    """Lay out the depot and one point per person according to `structure`."""
    coords: dict[str, tuple[float, float]] = {}
    half = _COORD_SPAN / 2.0

    if structure is TravelStructure.UNIFORM:
        coords[_START_LOC] = (half, half)
        for loc in people_locs:
            coords[loc] = (rng.uniform(0, _COORD_SPAN), rng.uniform(0, _COORD_SPAN))

    elif structure is TravelStructure.CLUSTERED:
        centres = [(0.2 * _COORD_SPAN, half), (0.8 * _COORD_SPAN, half)]
        coords[_START_LOC] = (half, half)  # depot sits between the two clusters
        for i, loc in enumerate(people_locs):
            cx, cy = centres[i % 2]
            coords[loc] = (rng.gauss(cx, _CLUSTER_SPREAD), rng.gauss(cy, _CLUSTER_SPREAD))

    elif structure is TravelStructure.LINE:
        coords[_START_LOC] = (0.0, 0.0)  # one end of the corridor
        for loc in people_locs:
            coords[loc] = (rng.uniform(0, _COORD_SPAN), rng.gauss(0.0, _LINE_JITTER))

    else:  # pragma: no cover - RANDOM has no coordinates
        raise ValueError(f"{structure} has no coordinate layout")

    return coords


def _build_travel(
    rng: random.Random, locations: Sequence[str], structure: TravelStructure
) -> dict[str, dict[str, int]]:
    """Travel-time matrix over all ordered location pairs (diagonal 0)."""
    if structure is TravelStructure.RANDOM:
        # Arbitrary, possibly asymmetric, no triangle inequality -- a stress case.
        return {
            a: {
                b: 0 if a == b else rng.randint(_RANDOM_TRAVEL_MIN, _RANDOM_TRAVEL_MAX)
                for b in locations
            }
            for a in locations
        }

    coords = _coordinates(rng, locations[1:], structure)
    travel: dict[str, dict[str, int]] = {}
    for a in locations:
        row: dict[str, int] = {}
        for b in locations:
            row[b] = 0 if a == b else max(1, round(math.dist(coords[a], coords[b])))
        travel[a] = row
    return travel


def _build_people(
    rng: random.Random,
    *,
    n_people: int,
    tightness: float,
    overlap: float,
    start_time: int,
    end_of_day: int,
    meeting_duration: int,
) -> list[Person]:
    """One availability window per person, shaped by tightness and overlap."""
    day_span = end_of_day - start_time

    # tightness -> window length (1 = just holds a meeting, 0 = whole day).
    window_length = meeting_duration + round((1.0 - tightness) * (day_span - meeting_duration))
    window_length = max(meeting_duration, min(window_length, day_span))

    # overlap -> how widely window starts are spread around the day's midpoint.
    available_span = day_span - window_length          # range a start may occupy
    centre = start_time + available_span / 2.0
    spread = (1.0 - overlap) * available_span

    people: list[Person] = []
    for i in range(n_people):
        if spread > 0:
            ws = centre + rng.uniform(-spread / 2.0, spread / 2.0)
        else:
            ws = centre
        ws_int = int(round(ws))
        ws_int = max(start_time, min(ws_int, start_time + available_span))
        people.append(
            Person(
                person_id=f"p{i}",
                location=f"loc_{i}",
                window_start=ws_int,
                window_end=ws_int + window_length,
            )
        )
    return people


def _default_instance_id(
    structure: TravelStructure, n_people: int, tightness: float, overlap: float, seed: int
) -> str:
    return (
        f"{structure.value}-n{n_people}"
        f"-t{int(round(tightness * 100))}-o{int(round(overlap * 100))}-s{seed}"
    )


def generate_instance(
    *,
    n_people: int,
    tightness: float,
    overlap: float,
    travel_structure: TravelStructure,
    seed: int,
    meeting_duration: int = DEFAULT_MEETING_DURATION,
    start_time: int = DEFAULT_START_TIME,
    end_of_day: int = DEFAULT_END_OF_DAY,
    waiting_allowed: bool = True,
    tie_break: str = "earliest_start_then_person_id",
    instance_id: str | None = None,
) -> Instance:
    """Build one deterministic instance. Same args + seed => identical instance.

    `complexity_metric` and `level` are left None; fill them via complexity.py after
    solving.
    """
    if not 0.0 <= tightness <= 1.0:
        raise ValueError(f"tightness must be in [0, 1], got {tightness}")
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}")
    if n_people < 1:
        raise ValueError(f"n_people must be >= 1, got {n_people}")
    if meeting_duration <= 0:
        raise ValueError(f"meeting_duration must be > 0, got {meeting_duration}")
    if end_of_day <= start_time:
        raise ValueError("end_of_day must be greater than start_time")

    rng = random.Random(seed)

    locations = [_START_LOC, *_people_locations(n_people)]
    travel_times = _build_travel(rng, locations, travel_structure)
    people = _build_people(
        rng,
        n_people=n_people,
        tightness=tightness,
        overlap=overlap,
        start_time=start_time,
        end_of_day=end_of_day,
        meeting_duration=meeting_duration,
    )

    if instance_id is None:
        instance_id = _default_instance_id(
            travel_structure, n_people, tightness, overlap, seed
        )

    return Instance(
        instance_id=instance_id,
        seed=seed,
        generator_params=GeneratorParams(
            n_people=n_people,
            tightness=tightness,
            overlap=overlap,
            travel_structure=travel_structure,
        ),
        start_location=_START_LOC,
        start_time=start_time,
        end_of_day=end_of_day,
        meeting_duration=meeting_duration,
        waiting_allowed=waiting_allowed,
        tie_break=tie_break,
        locations=locations,
        people=people,
        travel_times=travel_times,
    )


def generate_dataset(
    *,
    n_people_values: Iterable[int],
    tightness_values: Iterable[float],
    overlap_values: Iterable[float],
    structures: Iterable[TravelStructure],
    seeds: Iterable[int],
    **instance_kwargs: object,
) -> list[Instance]:
    """Cartesian sweep of the knob grid x seeds -> a list of instances.

    Used by the pilot to produce the 200+ instances whose complexity distribution
    fixes the easy/medium/hard boundaries (THESIS_DECISIONS.md section 3). Iteration
    order is deterministic.
    """
    instances: list[Instance] = []
    for structure in structures:
        for n_people in n_people_values:
            for tightness in tightness_values:
                for overlap in overlap_values:
                    for seed in seeds:
                        instances.append(
                            generate_instance(
                                n_people=n_people,
                                tightness=tightness,
                                overlap=overlap,
                                travel_structure=structure,
                                seed=seed,
                                **instance_kwargs,  # type: ignore[arg-type]
                            )
                        )
    return instances
