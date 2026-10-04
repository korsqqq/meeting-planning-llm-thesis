# tests/test_stage_tables.py
"""Offline tests for the descriptive stage tables.

The tables feed a Discussion section, so the tests target the places where a silent
default or a collapsed grouping would still produce a readable number: an unknown
turning into zero, a stratum being pooled away, and the two denominators being
conflated into one.
"""

from __future__ import annotations

import pytest

from scripts.analyse_stage_metrics import (
    POOLED,
    AnalysisRefused,
    Runs,
    opportunity_row,
    required,
    strata,
    table_critic_gain,
    table_critic_outcomes,
    table_no_proposal,
)


def stage(run, role, condition="c4_planner_critic", cap="64000", block="A",
          band="high", n_people="8", optimum="4", **kw):
    row = {
        "run_id": run, "instance_id": run.split("__")[1] if "__" in run else run,
        "condition": condition, "cap": cap, "block": block, "band": band,
        "n_people": n_people, "optimum": optimum, "conflict_pairs": "12",
        "stage_role": role, "stage_order": "0", "cycle_index": "0",
        "attempt_index": "", "n_calls": "1", "input_tokens": "10",
        "thinking_tokens": "20", "answer_tokens": "5", "stage_tokens": "35",
        "latency_seconds": "1.0", "n_context_limited": "0", "finish_reasons": "{}",
        "best_meetings_before": "1", "best_meetings_after": "2", "meetings_added": "1",
        "sat_before": "0.25", "sat_after": "0.5", "sat_delta": "0.25",
        "quality_source": "reconstructed", "n_proposals": "1", "n_parsed": "1",
        "n_valid": "1", "n_accepted": "1", "stage_outcome": "accepted",
        "pool_size": "4", "fallback_size": "1", "headroom": "3", "critic_room": "100",
        "planner_quota": "1000", "planner_spent": "900",
        "evidence_people_covered": "4", "evidence_travel_coverage": "0.25",
        "known_pairs_shown": "3", "unknown_pairs_shown": "9",
        "critic_reached": "1", "critic_improved": "1", "winner_attempt_index": "",
        "n_distinct_products": "", "information_source": "direct",
        "worker_a_meetings": "", "worker_b_meetings": "", "fallback_from": "",
    }
    row.update(kw)
    return row


def world(rows, proposals=()):
    return Runs(rows, [{"run_id": r} for r in proposals])


# --------------------------------------------------------------------------- #
# Missing is not zero.
# --------------------------------------------------------------------------- #
def test_a_ran_stage_without_a_value_stops_the_calculation():
    """`_int(x) or 0` used to turn this into a measured zero."""
    with pytest.raises(AnalysisRefused, match="not zero"):
        required(stage("r1", "critic", meetings_added=""), "meetings_added", "r1")


def test_the_gain_table_refuses_rather_than_defaulting():
    rows = [stage("r1", "critic", meetings_added=""), stage("r1", "planner")]
    with pytest.raises(AnalysisRefused):
        table_critic_gain(world(rows))


def test_a_critic_that_never_ran_contributes_a_real_zero():
    """No critic stage at all is a measured absence, and belongs in the all-runs mean."""
    rows = [stage("r1", "planner"), stage("r2", "planner"), stage("r2", "critic")]
    table = table_critic_gain(world(rows))
    cell = next(r for r in table if r["stratum"] == POOLED
                and r["condition"] == "c4_planner_critic" and r["cap"] == "64000")
    assert cell["runs"] == 2 and cell["runs_critic_ran"] == 1
    assert cell["mean_added_when_critic_ran"] == 1.0     # over the one that ran
    assert cell["mean_added_over_all_runs"] == 0.5       # over both


# --------------------------------------------------------------------------- #
# Both denominators, kept apart.
# --------------------------------------------------------------------------- #
def test_the_two_shares_are_reported_separately():
    rows = [stage("r1", "planner"), stage("r2", "planner"), stage("r2", "critic")]
    table = table_critic_outcomes(world(rows))
    cell = next(r for r in table if r["stratum"] == POOLED
                and r["condition"] == "c4_planner_critic" and r["cap"] == "64000")
    assert cell["not_reached"] == 1 and cell["improved"] == 1
    assert cell["improved_share_of_runs"] == 0.5        # the architecture as it ran
    assert cell["improved_share_of_reached"] == 1.0     # the stage where invoked


def test_satisfaction_gain_averages_per_run_differences():
    """Optima differ per instance, so the mean of deltas is the quantity."""
    rows = [stage("r1", "critic", optimum="4", sat_delta="0.25"),
            stage("r2", "critic", optimum="8", sat_delta="0.125"),
            stage("r1", "planner"), stage("r2", "planner")]
    cell = next(r for r in table_critic_gain(world(rows))
                if r["stratum"] == POOLED and r["cap"] == "64000")
    # The mean of the two per-run deltas, at the table's three-decimal resolution.
    # Differencing two means of `sat_after` and `sat_before` would answer a different
    # question here, because the two runs have different optima.
    assert cell["mean_sat_gain_when_critic_ran"] == round((0.25 + 0.125) / 2, 3)


# --------------------------------------------------------------------------- #
# Strata are not pooled away.
# --------------------------------------------------------------------------- #
def test_block_a_is_split_by_band_and_block_b_by_size():
    assert [name for name, _ in strata("A")] == [POOLED, "low", "medium", "high"]
    assert [name for name, _ in strata("B")] == [POOLED, "n=4", "n=5", "n=6"]


def test_a_band_difference_survives_into_the_tables():
    rows = [stage("r1", "planner", band="low"), stage("r2", "planner", band="high")]
    table = table_no_proposal(world(rows, proposals=["r1"]))
    pooled = next(r for r in table if r["stratum"] == POOLED
                  and r["condition"] == "c4_planner_critic")
    low = next(r for r in table if r["stratum"] == "low")
    high = next(r for r in table if r["stratum"] == "high")
    assert pooled["runs_without_any_proposal"] == 1 and pooled["runs"] == 2
    assert low["runs_without_any_proposal"] == 0     # r1 proposed
    assert high["runs_without_any_proposal"] == 1    # r2 did not


# --------------------------------------------------------------------------- #
# Room, and what it does not claim.
# --------------------------------------------------------------------------- #
def test_room_is_read_from_the_aggregate_for_c3_and_the_critic_for_c4():
    c4 = world([stage("r1", "critic", pool_size="5", fallback_size="1")])
    assert opportunity_row(c4, "r1", "c4_planner_critic")["headroom"] == 4

    c3 = world([stage("r2", "aggregate", condition="c3_mas", pool_size="6",
                      fallback_size="4", critic_improved=""),
                stage("r2", "critic", condition="c3_mas", pool_size="6",
                      fallback_size="4", critic_improved="")])
    row = opportunity_row(c3, "r2", "c3_mas")
    assert row["headroom"] == 2 and row["pool_size"] == 6


def test_room_to_optimum_and_the_already_optimal_flag():
    runs = world([stage("r1", "critic", fallback_size="4", optimum="4"),
                  stage("r2", "critic", fallback_size="1", optimum="4")])
    at_optimum = opportunity_row(runs, "r1", "c4_planner_critic")
    below = opportunity_row(runs, "r2", "c4_planner_critic")
    assert at_optimum["room_to_optimum"] == 0 and at_optimum["fallback_at_optimum"]
    assert below["room_to_optimum"] == 3 and not below["fallback_at_optimum"]
