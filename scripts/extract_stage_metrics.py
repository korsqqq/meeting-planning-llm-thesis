# scripts/extract_stage_metrics.py
"""Per-stage and per-proposal extraction from the held-out run documents. CPU only.

    .venv-harness/bin/python -m scripts.extract_stage_metrics \
        --runs results/logs/heldout \
        --expected results/manifests/expected_runs__bands_held_out_n8.json \
        --expected results/manifests/expected_runs__heldout_n4.json \
        --expected results/manifests/expected_runs__heldout_n5.json \
        --expected results/manifests/expected_runs__heldout_n6.json \
        --subset results/manifests/subset__bands_held_out_n8__278ce0f2e8fe.json \
        --subset results/manifests/subset__heldout_n4__54f3f48d6232.json \
        --subset results/manifests/subset__heldout_n5__4b91dee07c34.json \
        --subset results/manifests/subset__heldout_n6__44418ababc22.json \
        --out results/exports/heldout_stages

WHAT THIS IS. An extraction, not an analysis. It reads what the harness already
recorded and flattens it into two tables plus a verification report. **No hypothesis
test, no p-value, no correction, no significance claim, and no new LLM call.** The
frozen confirmatory analysis and the descriptive package are read-only inputs here and
are never written.

THE STAGE KEY. Both `run_result.calls[]` and `agent.proposals[]` carry the same `role`
vocabulary, so the stage is the role and no mapping has to be invented:

    C1  react_step, finalize
    C2  react_step, verify, revise, finalize      (verify emits no proposal by design)
    C3  worker_a, worker_b, [aggregate], critic, finalize
    C4  planner, critic, finalize
    C5  bon_attempt_0/1/2, finalize               (proposals carry attempt_index too)

`aggregate` is bracketed because it makes no model call: it is a synthesised row that
carries the C3 fallback and candidate pool, which cost nothing and are reconstructed.

DIRECT vs RECONSTRUCTED, and why the distinction is kept in the data. C4 and C5 record
an `agent.diagnostics` block, so their critic and attempt facts are read out of fields.
C1, C2 and C3 have that block present but empty, so the same facts have to be rebuilt
from the proposal sequence. Every derived column is therefore paired with a `*_source`
column reading `direct`, `reconstructed` or `unknown`, and the report counts how often
each occurred. A reconstruction that cannot be made unambiguously is left EMPTY and
marked `unknown` -- never filled with zero, because "the critic proposed nothing" and
"we could not tell what the critic proposed" are different facts.

THE C3 RECONSTRUCTION, from `src/agents/multi_agent/hierarchical.py`. Each worker's
final sub-plan is the plan of its last proposal with `accepted_into_best_plan` true;
absent such a proposal the sub-plan is empty. The fallback is then

    fallback = B if len(B) > len(A) else A          # ties resolve to A

and it enters the same hidden gate as any propose: full-instance validator, adopted
only if strictly longer than the best held so far. The candidate pool is the union of
`person_id` over the two final sub-plans -- reconstructable because a proposal's `plan`
stores `person_id` per meeting, so no prompt text is parsed. The reconstruction is
checked, not assumed: the report compares the rebuilt final plan against the stored
`run_result.final_plan` for every run, and a mismatch is reported rather than absorbed.

WHAT THE VALIDATOR VERDICT IS AND IS NOT. `valid` on a proposal is the harness's own
verdict recorded at propose time. This script re-uses it and runs no validator of its
own, so nothing here can disagree with the frozen scoring. The one place a validator
verdict would be needed and is not available is whether a C3 fallback passed the
full-instance check; that is inferred from whether the plan became the best plan, and
the inference is marked as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyse_heldout import load_instances  # noqa: E402
from scripts.audit_expected_runs import audit, load_expected  # noqa: E402
from scripts.run_pilot_sweep import result_path  # noqa: E402

SCHEMA_VERSION = "heldout_stage_extraction/1.0"

# Roles that never carry a proposal. `verify` is reflection-only by design (C2), and
# `finalize` emits the final plan through the terminal node rather than a propose.
NON_PROPOSING_ROLES = ("verify", "finalize")
AGGREGATE_STAGE = "aggregate"

UNKNOWN = ""            # never 0: an absent measurement is not a measured zero.

# Every key the harness may record in `agent.diagnostics`, declared ONCE. The row
# template, the copy-out and STAGE_FIELDS are all generated from this tuple: the
# first version of this script spelled the list out in three places, they drifted,
# and `critic_improved` -- the one field that makes the C4 critic claim checkable
# against something other than our own reconstruction -- silently never reached the
# CSV. A key absent from a run's diagnostics stays UNKNOWN for that run.
DIRECT_INFO_KEYS = (
    # C4 (planner + critic)
    "pool_size", "fallback_size", "headroom", "critic_room", "planner_quota",
    "planner_spent", "evidence_people_covered", "evidence_travel_coverage",
    "known_pairs_shown", "unknown_pairs_shown", "critic_reached",
    "critic_improved",
    # C5 (best-of-3)
    "winner_attempt_index", "n_distinct_products",
)
# Recorded as booleans; written as 0/1 so the CSV and any database read alike.
BOOLEAN_INFO_KEYS = ("critic_reached", "critic_improved")

STAGE_FIELDS = [
    # keys for every downstream grouping
    "run_id", "instance_id", "condition", "cap", "block", "band", "n_people",
    "optimum", "conflict_pairs",
    # the stage itself
    "stage_role", "stage_order", "cycle_index", "attempt_index",
    # cost, all direct from run_result.calls
    "n_calls", "input_tokens", "thinking_tokens", "answer_tokens", "stage_tokens",
    "latency_seconds", "n_context_limited", "finish_reasons",
    # quality around the stage
    "best_meetings_before", "best_meetings_after", "meetings_added",
    "sat_before", "sat_after", "sat_delta", "quality_source",
    # proposal activity within the stage
    "n_proposals", "n_parsed", "n_valid", "n_accepted", "stage_outcome",
    # information available to the stage: every recorded diagnostics key, then the
    # provenance flag, then the C3 reconstruction detail.
    *DIRECT_INFO_KEYS, "information_source",
    "worker_a_meetings", "worker_b_meetings", "fallback_from",
]

PROPOSAL_FIELDS = [
    "run_id", "instance_id", "condition", "cap", "block", "band", "n_people",
    "optimum", "stage_role", "attempt_index", "proposal_index", "step",
    "parsed", "n_meetings", "valid", "in_pool", "accepted_into_best_plan",
    "outcome", "reasons", "n_reasons",
]


class ExtractionRefused(RuntimeError):
    """The inputs are not the verified held-out set. Never worked around."""


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _plan_people(plan: Any) -> tuple[tuple[Any, Any], ...]:
    """A plan as a comparable set of (person_id, start_time) pairs."""
    if not plan:
        return ()
    meetings = plan.get("meetings") if isinstance(plan, dict) else plan
    if not meetings:
        return ()
    return tuple(sorted((m.get("person_id"), m.get("start_time")) for m in meetings))


def _ratio(numerator: int | None, denominator: int | None) -> Any:
    """Satisfaction as achieved/optimum, or UNKNOWN when the optimum cannot divide.

    Unit prizes make the achieved objective the met-people count, which is the meeting
    count of a validated plan -- the same expression the scorer uses. An optimum of 0
    is excluded upstream and returns UNKNOWN here rather than a fabricated 1.0.
    """
    if numerator is None or not denominator:
        return UNKNOWN
    return round(numerator / denominator, 6)


def call_stages(calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Segment the call list into consecutive same-role blocks, in execution order.

    Consecutive rather than grouped: C2 can enter `revise` twice, and collapsing the
    two would destroy the very sequence the verify/revise question is about.
    """
    stages: list[dict[str, Any]] = []
    seen = Counter()
    for call in calls:
        role = call.get("role")
        if not stages or stages[-1]["stage_role"] != role:
            seen[role] += 1
            stages.append({"stage_role": role, "cycle_index": seen[role] - 1,
                           "calls": []})
        stages[-1]["calls"].append(call)
    for order, stage in enumerate(stages):
        stage["stage_order"] = order
    return stages


def cost_of(calls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token, call, latency and finish-reason totals. Every field direct."""
    inp = sum(c.get("input_tokens") or 0 for c in calls)
    think = sum(c.get("thinking_tokens") or 0 for c in calls)
    ans = sum(c.get("answer_tokens") or 0 for c in calls)
    return {
        "n_calls": len(calls),
        "input_tokens": inp, "thinking_tokens": think, "answer_tokens": ans,
        "stage_tokens": inp + think + ans,
        "latency_seconds": round(sum(c.get("latency_seconds") or 0.0 for c in calls), 3),
        "n_context_limited": sum(1 for c in calls if c.get("context_limited")),
        "finish_reasons": json.dumps(dict(sorted(
            Counter(c.get("finish_reason") for c in calls).items())), ensure_ascii=False),
    }


def proposal_outcome(p: dict[str, Any]) -> str:
    """The five distinguishable fates of one propose, in the order the gate applies.

    `out_of_pool` exists only where the harness recorded `in_pool` (the C3 and C4
    critics); elsewhere the field is absent and the label cannot arise, which is a
    property of the gate rather than a gap in the data.
    """
    if not p.get("parsed"):
        return "parse_failed"
    if not p.get("valid"):
        return "invalid_constraints"
    if p.get("in_pool") is False:
        return "out_of_pool"
    if p.get("accepted_into_best_plan"):
        return "accepted"
    return "valid_not_longer"


# --------------------------------------------------------------------------- #
# Per-run extraction.
# --------------------------------------------------------------------------- #
def worker_final_plans(proposals: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Each C3 worker's final sub-plan: its last proposal accepted into its own best.

    Mirrors `WorkerOutcome.sub_plan`, which is the worker's `best_plan_so_far` at the
    end of its loop. A worker that never had a proposal accepted finished with the
    empty plan, which is a measured empty, not an unknown.
    """
    out: dict[str, Any] = {}
    for role in ("worker_a", "worker_b"):
        accepted = [p for p in proposals
                    if p.get("role") == role and p.get("accepted_into_best_plan")]
        out[role] = accepted[-1].get("plan") if accepted else []
    return out


def c3_reconstruction(doc: dict[str, Any], proposals: Sequence[dict[str, Any]]
                      ) -> dict[str, Any]:
    """Fallback, candidate pool and critic outcome for C3, rebuilt from proposals.

    The rule is copied from `hierarchical.aggregate_node`: the longer worker sub-plan
    wins and a tie resolves to A. Whether the fallback then passed the full-instance
    validator is not recorded anywhere, so it is inferred from whether the run's best
    plan ended up equal to it; the caller marks the whole block reconstructed.
    """
    finals = worker_final_plans(proposals)
    a, b = finals["worker_a"] or [], finals["worker_b"] or []
    fallback_from = "worker_b" if len(b) > len(a) else "worker_a"
    fallback = b if fallback_from == "worker_b" else a
    pool = {m.get("person_id") for m in list(a) + list(b) if m.get("person_id") is not None}

    critic = [p for p in proposals if p.get("role") == "critic"]
    if not critic:
        critic_outcome = "not_reached_or_no_proposal"
    else:
        outcomes = [proposal_outcome(p) for p in critic]
        critic_outcome = "accepted" if "accepted" in outcomes else outcomes[-1]
    return {
        "worker_a_meetings": len(a),
        "worker_b_meetings": len(b),
        "fallback_from": fallback_from,
        "fallback_size": len(fallback),
        "pool_size": len(pool),
        "fallback_plan": fallback,
        "critic_outcome": critic_outcome,
    }


def run_level_plan(condition: str, proposals: Sequence[dict[str, Any]],
                   diagnostics: dict[str, Any], recon: dict[str, Any]) -> Any:
    """The plan the RUN ended with, rebuilt the way each architecture actually forms it.

    The naive "last accepted proposal" is right only where every propose feeds one
    shared best plan, which is C1, C2 and C4. It is wrong for the other two, and the
    smoke test against the development documents is what showed it:

    * **C3.** Worker proposals are accepted into their own sub-instance best, not into
      the run's. The run-level plan is the fallback, and then the critic's proposal if
      that one passed the gate. Taking the last accepted proposal picked worker B's
      plan whenever the two workers tied, where the aggregator's tie rule takes A --
      same meeting count, different people, and 14 development runs disagreed on
      exactly that.
    * **C5.** The three attempts are independent and the product is selected, so the
      run-level plan is the winner named by `winner_attempt_index`, read directly from
      the diagnostics rather than inferred.
    """
    if condition == "c5_best_of_3":
        products = diagnostics.get("attempt_products") or []
        winner = diagnostics.get("winner_attempt_index")
        if winner is not None:
            for product in products:
                if product.get("attempt_index") == winner:
                    return product.get("plan") or []
        return []
    if condition == "c3_mas":
        critic_accepted = [p for p in proposals
                           if p.get("role") == "critic"
                           and p.get("accepted_into_best_plan")]
        if critic_accepted:
            return critic_accepted[-1].get("plan") or []
        return recon.get("fallback_plan") or []
    accepted = [p for p in proposals if p.get("accepted_into_best_plan")]
    return (accepted[-1].get("plan") or []) if accepted else []


def running_best(proposals: Sequence[dict[str, Any]]) -> list[tuple[int, Any]]:
    """The best-plan trajectory: after each accepted proposal, its size and plan."""
    trajectory: list[tuple[int, Any]] = []
    best_len, best_plan = 0, []
    for p in proposals:
        if p.get("accepted_into_best_plan"):
            best_len, best_plan = p.get("n_meetings") or 0, p.get("plan") or []
        trajectory.append((best_len, best_plan))
    return trajectory


def extract_run(doc: dict[str, Any], info: dict[str, Any]
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """One document -> its stage rows, its proposal rows, and its verification facts."""
    rr = doc["run_result"]
    agent = doc.get("agent") or {}
    diagnostics = agent.get("diagnostics") or {}
    proposals = list(agent.get("proposals") or [])
    condition, cap = rr["condition"], rr["cap"]
    optimum = info["optimum"]

    keys = {
        "run_id": rr["run_id"], "instance_id": rr["instance_id"],
        "condition": condition, "cap": cap, "block": info["block"],
        "band": info["band"] or UNKNOWN, "n_people": info["n_people"],
        "optimum": optimum, "conflict_pairs": info["complexity_metric"],
    }

    # ---- proposal rows ---------------------------------------------------
    prop_rows: list[dict[str, Any]] = []
    for index, p in enumerate(proposals):
        reasons = p.get("reasons") or []
        prop_rows.append({
            **{k: keys[k] for k in ("run_id", "instance_id", "condition", "cap",
                                    "block", "band", "n_people", "optimum")},
            "stage_role": p.get("role"),
            "attempt_index": p.get("attempt_index", UNKNOWN)
            if p.get("attempt_index") is not None else UNKNOWN,
            "proposal_index": index,
            "step": p.get("step"),
            "parsed": int(bool(p.get("parsed"))),
            "n_meetings": p.get("n_meetings"),
            "valid": int(bool(p.get("valid"))),
            "in_pool": (int(bool(p["in_pool"])) if "in_pool" in p else UNKNOWN),
            "accepted_into_best_plan": int(bool(p.get("accepted_into_best_plan"))),
            "outcome": proposal_outcome(p),
            "reasons": json.dumps(reasons, ensure_ascii=False),
            "n_reasons": len(reasons),
        })

    # ---- stage rows ------------------------------------------------------
    role_cycles = Counter()
    for stage in call_stages(rr.get("calls") or []):
        role_cycles[stage["stage_role"]] += 1

    props_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in proposals:
        props_by_role[p.get("role")].append(p)

    # A repeating role is ambiguous only when its proposals cannot be split across its
    # cycles one for one in step order. C2's `revise` is the only role that repeats,
    # and when it proposed once per cycle the assignment is determined, not guessed;
    # blanking the whole role in that case would discard a measurement that exists.
    ambiguous_roles = {r for r, n in role_cycles.items()
                       if n > 1 and len(props_by_role.get(r, [])) != n}

    trajectory = running_best(proposals)
    prop_position = {id(p): i for i, p in enumerate(proposals)}

    stage_rows: list[dict[str, Any]] = []
    best_before_run = 0
    for stage in call_stages(rr.get("calls") or []):
        role, cycle = stage["stage_role"], stage["cycle_index"]
        all_of_role = props_by_role.get(role, [])
        if role_cycles[role] > 1 and role not in ambiguous_roles:
            # One proposal per cycle, in step order: cycle k owns proposal k.
            mine = [all_of_role[cycle]] if cycle < len(all_of_role) else []
        else:
            mine = list(all_of_role)
        ambiguous = role in ambiguous_roles and len(mine) > 0

        if role in NON_PROPOSING_ROLES:
            quality_source, before, after = "not_applicable", UNKNOWN, UNKNOWN
        elif ambiguous:
            quality_source, before, after = "unknown", UNKNOWN, UNKNOWN
        else:
            positions = [prop_position[id(p)] for p in mine]
            before = best_before_run
            after = max((trajectory[i][0] for i in positions), default=before)
            best_before_run = max(best_before_run, after)
            quality_source = "reconstructed"

        row = {
            **keys,
            "stage_role": role, "stage_order": stage["stage_order"],
            "cycle_index": cycle,
            "attempt_index": next((p["attempt_index"] for p in mine
                                   if p.get("attempt_index") is not None), UNKNOWN),
            **cost_of(stage["calls"]),
            "best_meetings_before": before,
            "best_meetings_after": after,
            "meetings_added": (after - before) if before != UNKNOWN
            and after != UNKNOWN else UNKNOWN,
            "sat_before": _ratio(before if before != UNKNOWN else None, optimum),
            "sat_after": _ratio(after if after != UNKNOWN else None, optimum),
            "sat_delta": UNKNOWN,
            "quality_source": quality_source,
            "n_proposals": len(mine),
            "n_parsed": sum(1 for p in mine if p.get("parsed")),
            "n_valid": sum(1 for p in mine if p.get("valid")),
            "n_accepted": sum(1 for p in mine if p.get("accepted_into_best_plan")),
            "stage_outcome": ("no_proposal" if not mine else
                             ("accepted" if any(p.get("accepted_into_best_plan")
                                                for p in mine)
                              else proposal_outcome(mine[-1]))),
            **{k: UNKNOWN for k in DIRECT_INFO_KEYS},
            "information_source": "none_recorded",
            "worker_a_meetings": UNKNOWN, "worker_b_meetings": UNKNOWN,
            "fallback_from": UNKNOWN,
        }
        if row["sat_before"] != UNKNOWN and row["sat_after"] != UNKNOWN:
            row["sat_delta"] = round(row["sat_after"] - row["sat_before"], 6)
        stage_rows.append(row)

    # ---- information: direct where the harness recorded it ---------------
    if diagnostics:
        direct = {k: (int(bool(diagnostics[k])) if k in BOOLEAN_INFO_KEYS
                      else diagnostics[k])
                  for k in DIRECT_INFO_KEYS if k in diagnostics}
        if direct:
            for row in stage_rows:
                row.update(direct)
                row["information_source"] = "direct"

    # ---- C3: the synthesised aggregate stage -----------------------------
    recon: dict[str, Any] = {}
    if condition == "c3_mas":
        recon = c3_reconstruction(doc, proposals)
        critic_at = next((i for i, r in enumerate(stage_rows)
                          if r["stage_role"] == "critic"), None)
        insert_at = critic_at if critic_at is not None else max(
            (i for i, r in enumerate(stage_rows)
             if r["stage_role"] == "finalize"), default=len(stage_rows))
        # Run-level quality, which is NOT the workers' quality. A worker's accepted
        # proposals move its own sub-instance best; the run's best plan is empty until
        # the aggregator adopts the fallback. Relabelled so the two are never summed
        # or compared as if they measured the same thing.
        for row in stage_rows:
            if row["stage_role"] in ("worker_a", "worker_b"):
                row["quality_source"] = "reconstructed_subinstance"
            elif row["stage_role"] == "critic":
                before = recon["fallback_size"]
                critic_gains = [p.get("n_meetings") or 0 for p in proposals
                                if p.get("role") == "critic"
                                and p.get("accepted_into_best_plan")]
                after = max([before, *critic_gains])
                row["best_meetings_before"] = before
                row["best_meetings_after"] = after
                row["meetings_added"] = after - before
                row["sat_before"] = _ratio(before, optimum)
                row["sat_after"] = _ratio(after, optimum)
                row["sat_delta"] = (round(row["sat_after"] - row["sat_before"], 6)
                                    if row["sat_before"] != UNKNOWN
                                    and row["sat_after"] != UNKNOWN else UNKNOWN)
                row["quality_source"] = "reconstructed"
        agg = {
            **keys, "stage_role": AGGREGATE_STAGE, "stage_order": UNKNOWN,
            "cycle_index": 0, "attempt_index": UNKNOWN,
            "n_calls": 0, "input_tokens": 0, "thinking_tokens": 0, "answer_tokens": 0,
            "stage_tokens": 0, "latency_seconds": 0.0, "n_context_limited": 0,
            "finish_reasons": json.dumps({}),
            "best_meetings_before": 0,
            "best_meetings_after": recon["fallback_size"],
            "meetings_added": recon["fallback_size"],
            "sat_before": _ratio(0, optimum),
            "sat_after": _ratio(recon["fallback_size"], optimum),
            "sat_delta": UNKNOWN,
            "quality_source": "reconstructed",
            "n_proposals": 0, "n_parsed": 0, "n_valid": 0, "n_accepted": 0,
            "stage_outcome": "no_proposal",
            **{k: UNKNOWN for k in DIRECT_INFO_KEYS},
            "pool_size": recon["pool_size"], "fallback_size": recon["fallback_size"],
            "information_source": "reconstructed",
            "worker_a_meetings": recon["worker_a_meetings"],
            "worker_b_meetings": recon["worker_b_meetings"],
            "fallback_from": recon["fallback_from"],
        }
        if agg["sat_before"] != UNKNOWN and agg["sat_after"] != UNKNOWN:
            agg["sat_delta"] = round(agg["sat_after"] - agg["sat_before"], 6)
        stage_rows.insert(insert_at, agg)
        for order, row in enumerate(stage_rows):
            row["stage_order"] = order

    # ---- C5: the attempts are recorded, so nothing about them is inferred -----
    if condition == "c5_best_of_3" and diagnostics.get("attempt_products"):
        products = {p.get("attempt_index"): p
                    for p in diagnostics["attempt_products"]}
        winner = diagnostics.get("winner_attempt_index")
        for row in stage_rows:
            role = row["stage_role"]
            if not role.startswith("bon_attempt_"):
                continue
            index = int(role.rsplit("_", 1)[1])
            product = products.get(index)
            if product is None:
                continue
            n = product.get("n_meetings")
            row["attempt_index"] = index
            row["best_meetings_before"] = 0
            row["best_meetings_after"] = n
            row["meetings_added"] = n
            row["sat_before"] = _ratio(0, optimum)
            row["sat_after"] = _ratio(n, optimum)
            row["sat_delta"] = (round(row["sat_after"] - row["sat_before"], 6)
                                if row["sat_after"] != UNKNOWN else UNKNOWN)
            # The attempt's product is a recorded field, not a rebuilt one.
            row["quality_source"] = "direct"
            row["stage_outcome"] = ("selected_winner" if index == winner
                                    else row["stage_outcome"])

    # ---- verification facts ---------------------------------------------
    rebuilt = _plan_people(run_level_plan(condition, proposals, diagnostics, recon))
    stored = _plan_people(rr.get("final_plan"))
    checks = {
        "run_id": rr["run_id"], "condition": condition, "cap": cap,
        "final_plan_matches": rebuilt == stored,
        "rebuilt_meetings": len(rebuilt), "stored_meetings": len(stored),
        "calls_match": sum(r["n_calls"] for r in stage_rows) == len(rr.get("calls") or []),
        "tokens_match": sum(r["stage_tokens"] for r in stage_rows)
        == (rr.get("tokens") or {}).get("total"),
        "stage_tokens_sum": sum(r["stage_tokens"] for r in stage_rows),
        "document_tokens_total": (rr.get("tokens") or {}).get("total"),
        "ambiguous_stage_quality": sum(1 for r in stage_rows
                                       if r["quality_source"] == "unknown"),
        "has_diagnostics": bool(diagnostics),
        **critic_agreement(stage_rows, diagnostics),
    }
    return stage_rows, prop_rows, checks


def critic_agreement(stage_rows: Sequence[dict[str, Any]],
                     diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Does the rebuilt critic improvement agree with the recorded one?

    Only C4 records `critic_improved`, so only C4 can answer it -- but where it can,
    the reconstruction is checked against a field rather than trusted. A run whose
    critic never ran contributes nothing either way, and is reported as such rather
    than counted as agreement.
    """
    if "critic_improved" not in diagnostics:
        return {"critic_improved_checkable": False,
                "critic_improved_agrees": UNKNOWN}
    critic = next((r for r in stage_rows if r["stage_role"] == "critic"), None)
    rebuilt = bool(critic and critic["meetings_added"] not in (UNKNOWN, None)
                   and int(critic["meetings_added"]) > 0)
    return {"critic_improved_checkable": True,
            "critic_improved_agrees": rebuilt == bool(diagnostics["critic_improved"])}


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def render_report(checks: Sequence[dict[str, Any]], stage_rows: Sequence[dict[str, Any]],
                  prop_rows: Sequence[dict[str, Any]], audit_report: dict[str, Any]
                  ) -> str:
    L: list[str] = []
    add = L.append
    by_condition = defaultdict(list)
    for c in checks:
        by_condition[c["condition"]].append(c)

    add("# Held-out stage extraction — verification report")
    add("")
    add("An extraction, not an analysis: no hypothesis test, p-value, correction or "
        "significance claim is computed anywhere in this pass, and no LLM was called.")
    add("")
    add(f"Completeness audit: {audit_report['n_verified']}/{audit_report['n_expected']} "
        f"verified, passed = {audit_report['all_checks_passed']}.")
    add("")
    add(f"{len(checks)} runs -> {len(stage_rows)} stage rows, "
        f"{len(prop_rows)} proposal rows.")
    add("")

    add("## Agreement with the stored documents")
    add("")
    add("| condition | runs | final plan matches | call totals match | token totals match |")
    add("|---|---|---|---|---|")
    for cond in sorted(by_condition):
        rows = by_condition[cond]
        n = len(rows)
        add(f"| {cond} | {n} | {sum(r['final_plan_matches'] for r in rows)}/{n} | "
            f"{sum(r['calls_match'] for r in rows)}/{n} | "
            f"{sum(r['tokens_match'] for r in rows)}/{n} |")
    add("")
    bad = [r for r in checks if not r["final_plan_matches"]]
    if bad:
        add(f"**{len(bad)} run(s) whose rebuilt final plan differs from the stored one.** "
            "Listed so the reconstruction can be corrected rather than trusted:")
        add("")
        add("| run_id | rebuilt | stored |")
        add("|---|---|---|")
        for r in bad[:40]:
            add(f"| `{r['run_id']}` | {r['rebuilt_meetings']} | {r['stored_meetings']} |")
        add("")
    else:
        add("Every rebuilt final plan equals the stored `final_plan`, meeting for "
            "meeting. The reconstruction reproduces what the harness did.")
        add("")

    add("## The reconstruction checked against a recorded field")
    add("")
    add("Only C4 records `critic_improved`, so only there can the rebuilt critic "
        "improvement be compared with what the harness itself concluded. Where the "
        "comparison is possible it is made; where it is not, the row says so rather "
        "than reporting agreement it did not test.")
    add("")
    add("| condition | runs where checkable | rebuilt agrees with recorded |")
    add("|---|---|---|")
    for cond in sorted(by_condition):
        rows = [r for r in by_condition[cond] if r["critic_improved_checkable"]]
        if not rows:
            add(f"| {cond} | 0 | not recorded |")
            continue
        add(f"| {cond} | {len(rows)} | "
            f"{sum(1 for r in rows if r['critic_improved_agrees'])}/{len(rows)} |")
    add("")

    add("## Where the numbers come from")
    add("")
    add("| condition | runs with a diagnostics block | stage quality reconstructed | "
        "stage quality unknown |")
    add("|---|---|---|---|")
    for cond in sorted(by_condition):
        rows = by_condition[cond]
        sr = [s for s in stage_rows if s["condition"] == cond]
        add(f"| {cond} | {sum(r['has_diagnostics'] for r in rows)}/{len(rows)} | "
            f"{sum(1 for s in sr if s['quality_source'] == 'reconstructed')} | "
            f"{sum(1 for s in sr if s['quality_source'] == 'unknown')} |")
    add("")
    add("`unknown` is left empty in the CSV and is never written as zero: a stage whose "
        "proposals cannot be assigned to a cycle unambiguously has no measured quality, "
        "which is a different fact from a stage that measured no improvement.")
    add("")

    add("## Stages observed")
    add("")
    add("Stage rows and distinct runs are reported separately because they differ "
        "wherever a stage repeats: C2 can enter `revise` twice, so rows carrying a "
        "proposal there exceed the runs involved. A single column would have been read "
        "as runs and would have been wrong by nearly a factor of two.")
    add("")
    add("| condition | stage | stage rows | rows with a proposal | "
        "distinct runs with a proposal |")
    add("|---|---|---|---|---|")
    seen = defaultdict(lambda: {"rows": 0, "with_proposal": 0, "runs": set()})
    for s in stage_rows:
        k = (s["condition"], s["stage_role"])
        seen[k]["rows"] += 1
        if s["n_proposals"]:
            seen[k]["with_proposal"] += 1
            seen[k]["runs"].add(s["run_id"])
    for (cond, role), v in sorted(seen.items()):
        add(f"| {cond} | {role} | {v['rows']} | {v['with_proposal']} | "
            f"{len(v['runs'])} |")
    add("")

    add("## Proposal outcomes")
    add("")
    add("Counted over proposal attempts, not over runs. Interval-style summaries over "
        "runs belong to the analysis step, not to this extraction.")
    add("")
    outcomes = sorted({p["outcome"] for p in prop_rows})
    add("| condition | " + " | ".join(outcomes) + " |")
    add("|---|" + "---|" * len(outcomes))
    for cond in sorted(by_condition):
        counts = Counter(p["outcome"] for p in prop_rows if p["condition"] == cond)
        add(f"| {cond} | " + " | ".join(str(counts.get(o, 0)) for o in outcomes) + " |")
    add("")
    add("---")
    add("")
    add("*Every column in the two tables is either read from a recorded field or "
        "rebuilt from recorded fields, and the `*_source` columns say which. Nothing "
        "here establishes a cause.*")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--expected", action="append", type=Path, required=True)
    ap.add_argument("--subset", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/exports/heldout_stages"))
    args = ap.parse_args(argv)

    expected = load_expected(args.expected)
    audit_report = audit(expected, args.runs)
    if not audit_report["all_checks_passed"]:
        raise ExtractionRefused(
            f"completeness audit failed: {audit_report['n_missing']} missing, "
            f"{audit_report['n_incompatible']} incompatible, "
            f"{audit_report['n_unexpected_files']} unexpected. Nothing is extracted.")

    instances = load_instances(args.subset)
    stage_rows: list[dict[str, Any]] = []
    prop_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for item in sorted(expected, key=lambda r: (r["condition"], r["cap"],
                                                r["instance_id"])):
        info = instances.get(item["instance_id"])
        if info is None:
            raise ExtractionRefused(
                f"{item['instance_id']} is in an expected-runs manifest but in no "
                "subset manifest; its band and optimum are unknown")
        path = result_path(args.runs, item["condition"], item["instance_id"],
                           item["cap"], item["model"])
        doc = json.loads(path.read_text(encoding="utf-8"))
        s, p, c = extract_run(doc, info)
        stage_rows.extend(s)
        prop_rows.extend(p)
        checks.append(c)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "stage_metrics.csv", stage_rows, STAGE_FIELDS)
    write_csv(args.out / "proposal_diagnostics.csv", prop_rows, PROPOSAL_FIELDS)
    (args.out / "extraction_report.md").write_text(
        render_report(checks, stage_rows, prop_rows, audit_report), encoding="utf-8")
    (args.out / "extraction_manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "standing": ("Extraction only. No hypothesis test, p-value, correction or "
                     "significance claim; no LLM call. Frozen outputs are read-only."),
        "runs_dir": str(args.runs),
        "n_runs": len(checks), "n_stage_rows": len(stage_rows),
        "n_proposal_rows": len(prop_rows),
        "stage_fields": STAGE_FIELDS, "proposal_fields": PROPOSAL_FIELDS,
        "unknown_marker": "empty string; never 0",
        "direct_info_keys": list(DIRECT_INFO_KEYS),
        "final_plan_mismatches": sum(1 for c in checks if not c["final_plan_matches"]),
        "critic_improved_disagreements": sum(
            1 for c in checks if c["critic_improved_checkable"]
            and not c["critic_improved_agrees"]),
        "token_total_mismatches": sum(1 for c in checks if not c["tokens_match"]),
        "call_total_mismatches": sum(1 for c in checks if not c["calls_match"]),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    print(f"{len(checks)} runs -> {len(stage_rows)} stage rows, "
          f"{len(prop_rows)} proposal rows")
    print(f"  final plan mismatches : {sum(1 for c in checks if not c['final_plan_matches'])}")
    print(f"  token total mismatches: {sum(1 for c in checks if not c['tokens_match'])}")
    print(f"  call total mismatches : {sum(1 for c in checks if not c['calls_match'])}")
    print(f"  ambiguous stage rows  : {sum(c['ambiguous_stage_quality'] for c in checks)}")
    checkable = [c for c in checks if c["critic_improved_checkable"]]
    print(f"  critic_improved agrees: "
          f"{sum(1 for c in checkable if c['critic_improved_agrees'])}/{len(checkable)}"
          "  (C4 only; the other conditions do not record the field)")
    print(f"\nwritten: {args.out}/stage_metrics.csv, proposal_diagnostics.csv, "
          "extraction_report.md, extraction_manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
