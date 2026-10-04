# tests/test_harness_vertical_slice.py
"""Vertical-slice harness tests (offline, scripted clients, no endpoint).

What must hold:
  * the pipeline instance -> oracle -> C1 -> hidden scorer -> RunResult works and
    the numbers are the oracle's / scorer's, not the agent's self-report;
  * the JSON document is written where asked, with the stable top-level shape;
  * nothing oracle- or scorer-derived ever reaches the agent's conversation;
  * C3/C4 fail loudly as intentionally-not-implemented;
  * C2 dispatches through the same entry point;
  * reruns are deterministic up to timestamps/latencies.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core import LLMResponse
from src.harness import STEP_TOKEN_FLOOR, run_single_instance, steps_for_cap
from src.schemas import (
    Condition,
    GeneratorParams,
    Instance,
    InvalidReason,
    Level,
    Person,
    RunResult,
    TravelStructure,
)

# --------------------------------------------------------------------------- #
# Handcrafted instances (feasibility known by construction, not via the solver).
# --------------------------------------------------------------------------- #
def _instance() -> Instance:
    """Two compatible people: p0@10 then p1@60 both fit -> optimum 2, 0 conflicts."""
    return Instance(
        instance_id="harness-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=300,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=[
            Person(person_id="p0", location="A", window_start=0, window_end=100),
            Person(person_id="p1", location="B", window_start=0, window_end=200),
        ],
        travel_times={
            "S": {"S": 0, "A": 10, "B": 10},
            "A": {"S": 10, "A": 0, "B": 20},
            "B": {"S": 10, "A": 20, "B": 0},
        },
    )


def _conflicted_instance() -> Instance:
    """a and b are each reachable alone but never together -> 1 binding conflict."""
    return Instance(
        instance_id="harness-conflict",
        seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=1.0, overlap=1.0,
            travel_structure=TravelStructure.RANDOM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=300,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=[
            Person(person_id="a", location="A", window_start=0, window_end=40),
            Person(person_id="b", location="B", window_start=0, window_end=40),
        ],
        travel_times={
            "S": {"S": 0, "A": 10, "B": 10},
            "A": {"S": 10, "A": 0, "B": 100},
            "B": {"S": 10, "A": 100, "B": 0},
        },
    )


# --------------------------------------------------------------------------- #
# Scripted client (same shape as in test_react / test_react_verify_revise).
# --------------------------------------------------------------------------- #
class _ScriptedClient:
    """Plays back scripted answers; records every message it is ever shown."""

    def __init__(self, answers: list[str], *, input_tokens: int = 50):
        self._answers = answers
        self.i = 0
        self._input = input_tokens
        self.seen_contents: list[str] = []

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        self.seen_contents.extend(m["content"] for m in messages)
        return self._input

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        self.seen_contents.extend(m["content"] for m in messages)
        text = self._answers[self.i]
        self.i += 1
        return LLMResponse(
            thinking="", answer=text, raw_text=text,
            input_tokens=self._input, thinking_tokens=0,
            answer_tokens=len(text.split()), finish_reason="stop",
        )


def _c1_script_partial_plan() -> list[str]:
    """Draft proposes the 1-meeting plan, finishes; finalize echoes it."""
    return [
        "Action: propose[p0@10]",
        "Action: finish",
        '{"meetings": [{"person_id": "p0", "start_time": 10}]}',
    ]


# --------------------------------------------------------------------------- #
# 1. The slice runs end to end and the record is oracle/scorer-grounded.
# --------------------------------------------------------------------------- #
def test_harness_runs_c1_vertical_slice() -> None:
    client = _ScriptedClient(_c1_script_partial_plan())
    run = run_single_instance(
        instance=_instance(), client=client, condition="c1_react",
        cap=4000, model_label="scripted-test",
    )

    assert isinstance(run.run_result, RunResult)
    # Oracle block: computed harness-side, proven.
    assert run.oracle.optimum == 2
    assert run.oracle.proven_optimal and run.oracle.status == "OPTIMAL"
    assert run.oracle_latency_seconds >= 0.0
    # Complexity + the pre-registered easy anchor (0 conflicts).
    assert run.instance.complexity_metric == 0
    assert run.run_result.level is Level.EASY
    # Score: 1 of 2 meetings -> partial credit through the hidden scorer.
    score = run.run_result.score
    assert score.valid and score.satisfaction == 0.5 and not score.optimality
    assert score.n_valid_meetings == 1 and score.solver_optimum == 2
    # Token usage and calls came from the ledger.
    tokens = run.run_result.tokens
    assert tokens.n_calls == 3 and tokens.total > 0 and tokens.cap == 4000
    assert [c.role for c in run.run_result.calls] == ["react_step", "react_step", "finalize"]
    assert run.run_result.latency_seconds >= 0.0
    assert run.json_path is None  # no output_dir requested


# --------------------------------------------------------------------------- #
# 2. JSON writing.
# --------------------------------------------------------------------------- #
def test_harness_writes_json(tmp_path) -> None:
    client = _ScriptedClient(_c1_script_partial_plan())
    run = run_single_instance(
        instance=_instance(), client=client, condition="c1_react",
        cap=4000, model_label="scripted-test", output_dir=tmp_path,
    )

    assert run.json_path is not None and run.json_path.exists()
    assert run.json_path.name == f"{run.run_result.run_id}.json"
    doc = json.loads(run.json_path.read_text(encoding="utf-8"))

    for key in (
        "schema_version", "run_id", "timestamp_utc", "condition", "cap",
        "model_label", "finalization_reserve", "seeds", "provenance",
        "instance", "oracle", "agent", "run_result",
    ):
        assert key in doc, f"missing top-level key {key!r}"

    assert doc["condition"] == "c1_react"
    assert doc["model_label"] == "scripted-test"
    assert doc["seeds"] == {"generator": 0, "solver": 0, "inference": 42}
    assert doc["instance"]["n_people"] == 2
    assert doc["instance"]["complexity_metric"] == 0
    assert doc["oracle"]["optimum"] == 2 and doc["oracle"]["proven_optimal"] is True
    assert doc["run_result"]["score"]["satisfaction"] == 0.5
    assert doc["run_result"]["tokens"]["n_calls"] == 3
    assert isinstance(doc["agent"]["transcript"], list) and doc["agent"]["transcript"]


# --------------------------------------------------------------------------- #
# 3. No oracle/scorer leakage into anything the agent ever sees.
# --------------------------------------------------------------------------- #
def test_oracle_not_leaked_to_agent_transcript() -> None:
    client = _ScriptedClient(_c1_script_partial_plan())
    run = run_single_instance(
        instance=_instance(), client=client, condition="c1_react",
        cap=4000, model_label="scripted-test",
    )

    banned = ("optimum", "solver", "cp-sat", "score", "validity",
              "satisfaction", "complexity", "oracle")
    # Everything the client was ever shown (prompt renders + completion calls)...
    for content in client.seen_contents:
        low = content.lower()
        for word in banned:
            assert word not in low, f"agent prompt leaked {word!r}: {content[:80]}"
    # ...and the persisted transcript.
    for message in run.agent.transcript:
        low = message["content"].lower()
        for word in banned:
            assert word not in low, f"transcript leaked {word!r}: {message}"


# --------------------------------------------------------------------------- #
# 4. Every declared condition has a runner; an undeclared one fails explicitly.
# --------------------------------------------------------------------------- #
def test_every_declared_condition_is_dispatchable() -> None:
    """C4 was implemented on 2026-08-12, so the dispatch now covers the whole enum.
    A member without a runner would be a silently unrunnable condition."""
    from src.harness.runner import SUPPORTED_CONDITIONS
    assert set(SUPPORTED_CONDITIONS) == set(Condition)


def test_unknown_condition_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        run_single_instance(
            instance=_instance(), client=_ScriptedClient([]), condition="c9_wat",
            cap=4000, model_label="scripted-test",
        )


# --------------------------------------------------------------------------- #
# 5. The hidden scorer judges the final plan, not the agent's claim.
# --------------------------------------------------------------------------- #
def test_finalization_cannot_alter_the_scored_plan() -> None:
    # The terminal emit disobeys: it returns a DIFFERENT plan (adding p1 at 45, which
    # needs travel until 60). The section-2 invariant is enforced in code -- a parseable
    # emit is accepted only when it round-trips best_plan_so_far -- so the disobedient
    # emit is discarded, the structural plan is scored, and the divergence is recorded.
    #
    # NOTE: before the 2026-07-26 fix this test asserted the OPPOSITE (that the altered
    # plan was scored invalid). That encoded the drift bug as expected behaviour.
    client = _ScriptedClient([
        "Action: propose[p0@10]",
        "Action: finish",
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p1", "start_time": 45}]}',
    ])
    run = run_single_instance(
        instance=_instance(), client=client, condition="c1_react",
        cap=4000, model_label="scripted-test",
    )

    assert run.agent.finalization_mismatch is True
    # The scored plan is the structural best, never the altered emit.
    assert run.run_result.final_plan == run.run_result.best_plan_so_far
    assert [(m.person_id, m.start_time)
            for m in run.run_result.final_plan.meetings] == [("p0", 10)]
    score = run.run_result.score
    assert score.valid and score.satisfaction == 0.5
    assert score.invalid_reasons == []


def test_hidden_gate_keeps_an_invalid_proposal_out_of_the_score() -> None:
    # The agent's own PROPOSAL is infeasible (p1 at 45 needs travel until 60). The hidden
    # validator gate refuses it, so best_plan_so_far stays empty and the run scores an
    # honest 0.0 on a valid empty plan -- the agent's self-report never enters the score.
    client = _ScriptedClient([
        "Action: propose[p0@10, p1@45]",
        "Action: finish",
        '{"meetings": []}',
    ])
    run = run_single_instance(
        instance=_instance(), client=client, condition="c1_react",
        cap=4000, model_label="scripted-test",
    )

    assert run.run_result.best_plan_so_far.meetings == []
    assert run.agent.finalization_mismatch is False  # emit round-tripped the empty plan
    score = run.run_result.score
    assert score.valid and score.satisfaction == 0.0
    assert score.solver_optimum == 2


# --------------------------------------------------------------------------- #
# 6a. C2 dispatches through the same entry point.
# --------------------------------------------------------------------------- #
def test_c2_condition_through_harness() -> None:
    client = _ScriptedClient([
        "Action: propose[p0@10]",                        # draft
        "Action: finish",                                # draft -> verify
        "The plan could still add p1.",                  # verify 1
        "Action: propose[p0@10, p1@60]",                 # revise 1
        "Nothing further to improve.",                   # verify 2
        "Action: propose[p0@10, p1@60]",                 # revise 2 -> stop
        '{"meetings": [{"person_id": "p0", "start_time": 10}, '
        '{"person_id": "p1", "start_time": 60}]}',       # finalize
    ])
    run = run_single_instance(
        instance=_instance(), client=client, condition=Condition.C2_VERIFY_REVISE,
        cap=8000, model_label="scripted-test",
    )

    roles = [c.role for c in run.run_result.calls]
    assert roles.count("verify") == 2 and roles.count("revise") == 2
    assert run.run_result.condition is Condition.C2_VERIFY_REVISE
    assert run.run_result.score.valid
    assert run.run_result.score.satisfaction == 1.0
    assert run.run_result.score.optimality


# --------------------------------------------------------------------------- #
# 6a. The step cap follows the budget, so the budget is the binding constraint.
# --------------------------------------------------------------------------- #
def test_steps_for_cap_scales_with_the_budget() -> None:
    # Monotone in the cap, and never zero even for an absurdly small budget.
    assert steps_for_cap(2_000) == 2_000 // STEP_TOKEN_FLOOR
    assert steps_for_cap(64_000) == 64_000 // STEP_TOKEN_FLOOR
    assert steps_for_cap(64_000) > steps_for_cap(16_000) > steps_for_cap(2_000)
    assert steps_for_cap(1) == 1

    # The floor must stay conservative: the cheapest ReAct step observed live cost
    # roughly 430 tokens, so the derived cap has to exceed what the budget can pay for.
    assert steps_for_cap(16_000) > 16_000 // 430


def test_runner_derives_max_steps_when_not_given(tmp_path) -> None:
    # An explicit max_steps still wins (the smoke script and the tests rely on it);
    # omitting it must not silently fall back to a constant that binds before tokens.
    run = run_single_instance(
        instance=_instance(), client=_ScriptedClient(_c1_script_partial_plan()),
        condition="c1_react", cap=16_000, model_label="scripted-test",
        output_dir=tmp_path,
    )
    assert run.run_result.tokens.cap == 16_000
    # The scripted client stops on its own well before either limit; what matters is
    # that the derived cap is far above the number of steps the budget can fund.
    assert steps_for_cap(16_000) >= 64


# --------------------------------------------------------------------------- #
# 6b. Determinism: reruns agree except timestamps/latencies.
# --------------------------------------------------------------------------- #
def test_document_deterministic_modulo_time(tmp_path) -> None:
    def _run(subdir: str):
        return run_single_instance(
            instance=_instance(), client=_ScriptedClient(_c1_script_partial_plan()),
            condition="c1_react", cap=4000, model_label="scripted-test",
            output_dir=tmp_path / subdir,
        )

    doc1 = json.loads(_run("a").json_path.read_text(encoding="utf-8"))
    doc2 = json.loads(_run("b").json_path.read_text(encoding="utf-8"))

    assert doc1["run_id"] == doc2["run_id"]
    assert doc1["instance"] == doc2["instance"]
    assert doc1["agent"]["transcript"] == doc2["agent"]["transcript"]
    for block in ("optimum", "status", "proven_optimal"):
        assert doc1["oracle"][block] == doc2["oracle"][block]
    for block in ("tokens", "score", "final_plan", "best_plan_so_far"):
        assert doc1["run_result"][block] == doc2["run_result"][block]


# --------------------------------------------------------------------------- #
# 6c. Honest level policy: conflicted instances need pilot binning.
# --------------------------------------------------------------------------- #
def test_conflicted_instance_requires_explicit_level() -> None:
    script = ["Action: finish", '{"meetings": []}']
    with pytest.raises(ValueError, match="pilot binning"):
        run_single_instance(
            instance=_conflicted_instance(), client=_ScriptedClient(script),
            condition="c1_react", cap=4000, model_label="scripted-test",
        )
    # With an explicit level the same instance runs.
    run = run_single_instance(
        instance=_conflicted_instance(), client=_ScriptedClient(script),
        condition="c1_react", cap=4000, model_label="scripted-test",
        level=Level.MEDIUM,
    )
    assert run.instance.complexity_metric == 1
    assert run.run_result.level is Level.MEDIUM
    assert run.oracle.optimum == 1
    assert run.run_result.score.valid  # empty plan is valid; scores 0 of optimum 1
    assert run.run_result.score.satisfaction == 0.0


# --------------------------------------------------------------------------- #
# 7. CLI smoke: generated instance + offline smoke client, JSON on disk.
# --------------------------------------------------------------------------- #
def test_cli_offline_smoke(tmp_path, capsys) -> None:
    from scripts.run_vertical_slice import main

    rc = main(["--output-dir", str(tmp_path), "--seed", "0"])
    assert rc == 0

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["model_label"] == "offline-smoke-client"
    assert doc["run_result"]["score"]["valid"] is True
    # The smoke client plans one real meeting from tool observations alone.
    assert len(doc["run_result"]["final_plan"]["meetings"]) >= 1
    assert doc["run_result"]["score"]["satisfaction"] > 0.0

    out = capsys.readouterr().out
    assert "OFFLINE SCRIPTED-CLIENT SMOKE" in out  # honest banner, printed loudly
