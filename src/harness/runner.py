# src/harness/runner.py
"""Vertical-slice harness: one instance through one condition, end to end.

    generator/handcrafted Instance -> CP-SAT oracle (optimum + complexity metric)
        -> condition run (C1 / C2) -> hidden validator + scorer -> RunResult
        -> optional JSON log on disk (result_writer)

Scope (deliberate, THESIS_DECISIONS.md section 4 / PROGRESS):
  * All five conditions are implemented: c1_react, c2_verify_revise, c3_mas,
    c4_planner_critic and c5_best_of_3.
  * This is a single-run slice, not the sweep runner: no run matrix, no resume
    logic, no aggregation. Those come with the pilot.

Oracle isolation (hard invariant): the solver runs strictly harness-side, BEFORE
the agent, and nothing derived from it -- optimum, canonical plan, status,
complexity metric, level, score -- is passed into the agent run. The agent receives
the raw task Instance only; its prompts and observations are built exclusively from
that instance by the shared react core and the three read-only tools.

Level policy (honest by construction): a fresh instance has `level=None` until the
pilot fixes bin boundaries. The only boundary locked in advance is the anchor
"easy = 0 binding conflicts" (THESIS_DECISIONS.md section 3), so the slice derives
EASY for a 0-conflict instance and otherwise REFUSES to guess -- the caller must
pass an explicit `level` obtained from real pilot binning.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.agents.multi_agent.hierarchical import run_hierarchical
from src.agents.react_core import ReactResult
from src.agents.single_agent.best_of_3 import run_best_of_3
from src.agents.single_agent.planner_critic import run_planner_critic
from src.agents.single_agent.react import run_react
from src.agents.single_agent.react_verify_revise import run_verify_revise
from src.core import DEFAULT_FINALIZATION_RESERVE
from src.core.llm_client import MIN_P, SEED, TEMPERATURE, TOP_K, TOP_P
from src.core.usage_audit import audit_run
from src.data import solve_and_annotate
from src.evaluation import score_plan
from src.harness.sanity import transcript_sanity
from src.oracle import OracleSolution
from src.schemas import Condition, Instance, Level, RunResult, SamplingConfig

__all__ = ["SUPPORTED_CONDITIONS", "STEP_TOKEN_FLOOR", "HarnessRun", "steps_for_cap",
           "run_id_for", "run_single_instance"]

# Conservative lower bound on the token cost of one ReAct step, used to derive the
# step cap from the budget (see `steps_for_cap`). [PILOT] -- revisit once step costs
# are measured across both models and all complexity levels.
STEP_TOKEN_FLOOR = 250

# Every condition shares one call shape (instance, client, *, cap, max_steps,
# finalization_reserve, retry_on_empty). The dispatch is the only place that knows
# which runner belongs to which condition.
SUPPORTED_CONDITIONS: dict[Condition, object] = {
    Condition.C1_REACT: run_react,
    Condition.C2_VERIFY_REVISE: run_verify_revise,
    Condition.C3_MAS: run_hierarchical,
    Condition.C4_PLANNER_CRITIC: run_planner_critic,
    Condition.C5_BEST_OF_3: run_best_of_3,
}


@dataclass(frozen=True)
class HarnessRun:
    """Everything one slice run produced, RunResult plus harness-side context."""

    run_result: RunResult
    instance: Instance                # annotated copy (complexity_metric + level set)
    oracle: OracleSolution
    oracle_latency_seconds: float
    agent: ReactResult                # raw agent artifacts (transcript, best plan, calls)
    usage_audit: dict                 # internal-vs-endpoint token comparison (audit only)
    sanity: dict                      # think-leak / oracle-vocabulary transcript diagnostics
    json_path: Path | None            # where the JSON document was written, if requested


def run_id_for(
    condition: Condition | str, instance_id: str, cap: int, model_label: str
) -> str:
    """The run identity, and therefore the result filename stem.

    One formula, used both by the runner when it writes and by anything that needs to
    know in advance whether a run already exists (resume). The model belongs to the
    identity: without it two models over the same instance and cap collide on
    `<run_id>.json` and overwrite each other silently. The label is normalised because
    Hub ids contain '/'.
    """
    cond = condition.value if isinstance(condition, Condition) else str(condition)
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    return f"{cond}__{instance_id}__cap{cap}__{model_tag}"


def steps_for_cap(cap: int) -> int:
    """Step cap derived from the budget, so the BUDGET is what ends a run.

    The research question is about behaviour under an equal TOKEN budget, so the step
    counter must never be the binding constraint -- otherwise the independent variable
    is silently "number of steps". A fixed `max_steps=12` fails that: a live C1 run on
    Qwen3-32B-AWQ solved the EASIEST instance (n=4, zero conflicts) in 11 steps, one
    short of the cap, so any harder instance would have been cut by the counter rather
    than by tokens.

    `STEP_TOKEN_FLOOR` is a conservative lower bound on what one ReAct step costs
    (input + thinking + answer); the cheapest step observed live was ~430 tokens, so
    250 leaves a margin. The counter therefore stays what it is meant to be -- a
    runaway guard behind the budget and the `aborted` path, not an experimental knob.
    """
    return max(1, cap // STEP_TOKEN_FLOOR)


def _resolve_condition(condition: str | Condition) -> Condition:
    try:
        cond = Condition(condition)
    except ValueError as exc:
        valid = ", ".join(c.value for c in Condition)
        raise ValueError(f"unknown condition {condition!r}; expected one of: {valid}") from exc
    if cond not in SUPPORTED_CONDITIONS:  # pragma: no cover - every member is mapped
        raise NotImplementedError(
            f"condition {cond.value!r} has no runner on this dispatch."
        )
    return cond


def _resolve_level(instance: Instance, level: Level | None) -> Level:
    if level is not None:
        return level
    if instance.complexity_metric == 0:
        # The one pre-registered anchor: easy = 0 binding conflicts.
        return Level.EASY
    raise ValueError(
        f"instance {instance.instance_id!r} has complexity_metric="
        f"{instance.complexity_metric} but no level: medium/hard boundaries come "
        "from pilot binning (THESIS_DECISIONS.md section 3), not from a guess -- "
        "pass level= explicitly for a conflicted instance."
    )


def run_single_instance(
    *,
    instance: Instance,
    client,
    condition: str | Condition = Condition.C1_REACT,
    cap: int,
    model_label: str,
    finalization_reserve: int = DEFAULT_FINALIZATION_RESERVE,
    max_steps: int | None = None,
    retry_on_empty: bool = False,
    level: Level | None = None,
    output_dir: Path | str | None = None,
    notes: str | None = None,
) -> HarnessRun:
    """Run one instance through one condition and return the full record.

    `client` is anything with the LLMClient interface (`count_input` /
    `complete`); the offline slice passes a scripted client. `model_label` names
    what actually produced the tokens (e.g. "offline-smoke-client" or
    "Qwen/Qwen3-32B-AWQ") and is stored verbatim -- never claim a model that did
    not run. If `output_dir` is given, a JSON document is written there as
    `<run_id>.json` and the path is returned on the record.

    `max_steps=None` (the default) derives the step cap from the budget so that the
    budget, not the step counter, is what ends a run -- see `steps_for_cap`.

    `retry_on_empty` is the EXPERIMENTAL arm of the empty-turn A/B and defaults to the
    locked section-2 behaviour (False). Nothing in the sweep sets it; only
    `scripts/run_empty_retry_ab.py` does.
    """
    cond = _resolve_condition(condition)
    if max_steps is None:
        max_steps = steps_for_cap(cap)

    # --- harness-side ground truth (never enters the agent run) ------------- #
    t0 = time.perf_counter()
    annotated, oracle = solve_and_annotate(instance)
    oracle_latency = time.perf_counter() - t0
    annotated = annotated.model_copy(update={"level": _resolve_level(annotated, level)})

    # --- the agent run: raw task instance only ------------------------------ #
    # Snapshot the client's endpoint-usage audit trail so a reused client only
    # contributes THIS run's calls. Scripted/offline clients have no usage_log;
    # that is recorded honestly as "no endpoint usage", never faked.
    client_usage_log = getattr(client, "usage_log", None)
    usage_log_start = len(client_usage_log) if client_usage_log is not None else 0

    runner = SUPPORTED_CONDITIONS[cond]
    t1 = time.perf_counter()
    agent_result: ReactResult = runner(
        instance,
        client,
        cap=cap,
        max_steps=max_steps,
        finalization_reserve=finalization_reserve,
        retry_on_empty=retry_on_empty,
    )
    agent_latency = time.perf_counter() - t1

    endpoint_records = (
        list(client_usage_log[usage_log_start:]) if client_usage_log is not None else None
    )

    # --- hidden validation + scoring (independent of the agent's self-report) - #
    score = score_plan(instance, agent_result.final_plan, oracle.optimum)

    run_id = run_id_for(cond, annotated.instance_id, cap, model_label)
    run_result = RunResult(
        run_id=run_id,
        instance_id=annotated.instance_id,
        seed=annotated.seed,
        condition=cond,
        level=annotated.level,
        cap=cap,
        model=model_label,
        sampling=SamplingConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            seed=SEED,
            thinking_enabled=True,
        ),
        final_plan=agent_result.final_plan,
        best_plan_so_far=agent_result.best_plan_so_far,
        tokens=agent_result.tokens,
        score=score,
        latency_seconds=agent_latency,
        calls=agent_result.calls,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )

    # Audit-only diagnostics: internal ledger vs endpoint-reported usage, and the
    # transcript think-leak / oracle-vocabulary scan. Neither influences the run.
    usage_audit = audit_run(
        tokens=agent_result.tokens,
        calls=agent_result.calls,
        endpoint_records=endpoint_records,
    )
    sanity = transcript_sanity(agent_result.transcript)

    json_path: Path | None = None
    if output_dir is not None:
        # Imported here to keep runner importable without touching the filesystem
        # helpers (and to avoid a module cycle if the writer ever needs the runner).
        from src.harness.result_writer import build_document, write_result

        document = build_document(
            run_result=run_result,
            instance=annotated,
            oracle=oracle,
            oracle_latency_seconds=oracle_latency,
            agent=agent_result,
            finalization_reserve=finalization_reserve,
            usage_audit=usage_audit,
            sanity=sanity,
        )
        json_path = write_result(document, Path(output_dir))

    return HarnessRun(
        run_result=run_result,
        instance=annotated,
        oracle=oracle,
        oracle_latency_seconds=oracle_latency,
        agent=agent_result,
        usage_audit=usage_audit,
        sanity=sanity,
        json_path=json_path,
    )
