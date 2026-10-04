# src/agents/multi_agent/supervisor.py
"""C3 supervisor: the deterministic split into two geographic clusters.

THESIS_DECISIONS section 4, C3-1/C3-2 (LOCKED): the supervisor is a deterministic
control node, not an LLM agent. It computes the fixed k=2 split from the travel
matrix (the Instance carries no coordinates), builds the two sub-instances, and
enforces the empty-cluster rule. It issues no model calls and spends zero tokens.

Split algorithm (C3-1, verbatim):
  1. symmetrise distances: d(x, y) = (travel[x][y] + travel[y][x]) / 2 over the
     people's locations (depot excluded);
  2. seeds = the pair of people whose locations maximise d; ties by the
     lexicographically smallest sorted (person_id, person_id) pair;
  3. labels: cluster A is seeded by the smaller person_id of the winning pair;
  4. assignment: every other person joins the seed nearer by d; distance ties -> A;
  5. n = 1: the single person forms cluster A and cluster B is empty (the empty
     cluster's worker is skipped by the caller; quotas never reallocate).

A sub-instance is the parent restricted to one cluster: its people, the depot plus
their locations (parent order preserved), the travel matrix restricted to those
locations; every scalar field copied verbatim. The partition is hard -- each person
occurs in exactly one sub-instance -- and restriction changes no constraint on the
remaining people, so a plan valid in a sub-instance is valid in the full instance
(asserted by tests; the merge relies on it).
"""

from __future__ import annotations

from src.schemas import Instance, Person

__all__ = ["split_people", "make_sub_instance", "split_instance"]


def _sym_dist(instance: Instance, loc_x: str, loc_y: str) -> float:
    """Symmetrised travel distance between two locations (C3-1 step 1)."""
    return (instance.travel_time(loc_x, loc_y) + instance.travel_time(loc_y, loc_x)) / 2.0


def split_people(instance: Instance) -> tuple[list[Person], list[Person]]:
    """Deterministic k=2 split of the people (C3-1). Returns (cluster A, cluster B),
    each in parent instance order. Cluster B is empty iff the instance has one person."""
    people = instance.people
    if len(people) == 1:
        return list(people), []

    # Seeds: the farthest pair by symmetrised distance; ties resolved by iterating
    # the id-sorted pairs in order and keeping the FIRST maximum, which is exactly
    # the lexicographically smallest sorted (person_id, person_id) pair.
    by_id = sorted(people, key=lambda p: p.person_id)
    seed_a: Person | None = None
    seed_b: Person | None = None
    best_d = -1.0
    for i in range(len(by_id)):
        for j in range(i + 1, len(by_id)):
            d = _sym_dist(instance, by_id[i].location, by_id[j].location)
            if d > best_d:
                best_d = d
                seed_a, seed_b = by_id[i], by_id[j]  # by_id order => smaller id is A
    assert seed_a is not None and seed_b is not None

    cluster_a: list[Person] = []
    cluster_b: list[Person] = []
    for p in people:  # parent order preserved within each cluster
        if p.person_id == seed_a.person_id:
            cluster_a.append(p)
        elif p.person_id == seed_b.person_id:
            cluster_b.append(p)
        else:
            d_a = _sym_dist(instance, p.location, seed_a.location)
            d_b = _sym_dist(instance, p.location, seed_b.location)
            (cluster_a if d_a <= d_b else cluster_b).append(p)  # ties -> A
    return cluster_a, cluster_b


def make_sub_instance(
    instance: Instance, cluster: list[Person], label: str
) -> Instance | None:
    """Restrict the parent instance to one cluster (C3-1). None for an empty cluster."""
    if not cluster:
        return None

    keep_locs = {instance.start_location} | {p.location for p in cluster}
    locations = [loc for loc in instance.locations if loc in keep_locs]
    travel_times = {
        a: {b: instance.travel_times[a][b] for b in locations} for a in locations
    }
    return Instance(
        instance_id=f"{instance.instance_id}::{label}",
        seed=instance.seed,
        generator_params=instance.generator_params,
        start_location=instance.start_location,
        start_time=instance.start_time,
        end_of_day=instance.end_of_day,
        meeting_duration=instance.meeting_duration,
        waiting_allowed=instance.waiting_allowed,
        tie_break=instance.tie_break,
        locations=locations,
        people=list(cluster),
        travel_times=travel_times,
        # complexity_metric / level stay None: a sub-instance is an in-memory
        # working object, never a solved or logged experimental instance.
    )


def split_instance(instance: Instance) -> tuple[Instance | None, Instance | None]:
    """The supervisor's whole job: (sub-instance A, sub-instance B)."""
    cluster_a, cluster_b = split_people(instance)
    return (
        make_sub_instance(instance, cluster_a, "A"),
        make_sub_instance(instance, cluster_b, "B"),
    )
