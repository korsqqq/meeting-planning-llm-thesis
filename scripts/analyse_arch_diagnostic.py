# scripts/analyse_arch_diagnostic.py
"""Post-gate architecture diagnostic at n = 8, cap 64 000. Description, not inference.

    python -m scripts.analyse_arch_diagnostic \
        --diag-runs results/logs/budget_dev_arch_diag \
        --c1-runs   results/logs/budget_dev \
        --out       results/analysis/budget_dev_arch_diag

**This cannot repair the C1 budget gate.** That gate reads C1 only, by design, so that the
treatment architecture cannot choose the budget; a C2 or C3 result is not evidence about it
in either direction. Nothing here selects a cap, and the main experiment does not start on
this script's authority. The verdict strings below say so, and a test asserts they do.

The question, registered before any outcome existed (THESIS_DECISIONS section 5, amendment
of 2026-08-11): on the same n = 8 development instances at 64 000, do C2 and C3 recover
validator-approved **non-empty** plans substantially more often than plain C1?

*Non-empty specifically.* An empty plan is vacuously feasible in this harness -- the scored
`final_plan` is `best_plan_so_far`, which only ever holds validator-approved plans -- so a
generic feasibility rate is ~1.0 by construction and carries no information. The quantity
reported here is `validator-approved AND at least one meeting`, the same structural
definition the budget gate used.

*The 50% figure is a reference point, not a decision rule.* It is reused from the registered
gate rather than chosen now, and no threshold is tuned from these outcomes. No hypothesis
test is introduced: this script counts, pairs and describes.

Fail-loud by construction. A run set that is short, duplicated, drawn from a different frozen
selection, or run at the wrong cap would still produce perfectly readable rates, and the
comparison would mean nothing. Every such condition raises instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "arch_diagnostic/1.0"

# The frozen selection these runs must have come from. Pinned so a different subset
# cannot be analysed by accident: the numbers would look identical and mean something else.
EXPECTED_SUBSET_HASH = (
    "911e4864d6fb8ea2a78ff465f8e55a286d3c48afc1b8403d5e63f231bf96fd91")
EXPECTED_BINNING_HASH = (
    "8e671cd13dbb262b638048eff58241226f09d5804373534789915bc302ba0dff")

DIAGNOSTIC_CAP = 64_000
# The default reproduces the frozen 2026-08-13 run byte for byte. Later conditions are
# added by naming them on the command line, never by changing this default: the C2/C3
# reading is recorded and must stay reproducible from the same directory.
DEFAULT_DIAGNOSTIC_CONDITIONS = ("c2_verify_revise", "c3_mas")
BASELINE_CONDITION = "c1_react"
# Every condition this diagnostic knows how to describe. Membership here does not mean a
# condition was run; the audit requires exactly the ones the caller declares.
KNOWN_DIAGNOSTIC_CONDITIONS = (
    "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
N_INSTANCES_PER_BAND = 20
BANDS = ("low", "medium", "high")
BAND_OF_LEVEL = {"easy": "low", "medium": "medium", "hard": "high"}
# Reused from the registered gate. Not a decision rule here, and not tuned from outcomes.
REFERENCE_RATE = 0.50

# Call roles the conditions emit, used for stage reach. Nothing routes on them.
C2_STAGE_ROLES = ("verify", "revise")
C3_SEARCH_ROLES = ("worker_a", "worker_b")
C3_CRITIC_ROLE = "critic"


class DiagnosticAuditError(RuntimeError):
    """The run set is not the one this diagnostic was registered against."""


def _plan_size(plan: Any) -> int:
    if not isinstance(plan, dict):
        return 0
    return max((len(v) for v in plan.values() if isinstance(v, list)), default=0)


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    """One row per result document, with the fields this diagnostic reports."""
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        score, toks = rr["score"], rr["tokens"]
        md = doc.get("pilot_sweep") or {}
        calls = rr.get("calls") or []
        props = agent.get("proposals") or []
        meetings = _plan_size(rr.get("final_plan"))

        reasons: Counter = Counter()
        for p in props:
            if not p["valid"]:
                reasons.update(p.get("reasons") or [])

        rows.append({
            "run_id": rr["run_id"],
            "condition": rr["condition"],
            "instance_id": rr["instance_id"],
            "cap": rr["cap"],
            "level": rr["level"],
            "band": BAND_OF_LEVEL.get(rr["level"], rr["level"]),
            "model": rr["model"],
            "seed": rr["seed"],
            "subset_hash": md.get("subset_hash"),
            "binning_hash": md.get("binning_hash"),
            "git_commit": md.get("git_commit"),
            "optimum": score["solver_optimum"],
            "achieved": score["n_valid_meetings"],
            "satisfaction": score["satisfaction"],
            "feasible": bool(score.get("valid")),
            "final_plan_meetings": meetings,
            # The reported quantity. Both halves applied as written; non-emptiness is the
            # operative one, since feasibility is ~1.0 by construction.
            "nonempty_validated": bool(score.get("valid")) and meetings >= 1,
            "tokens_total": toks["total"],
            "cap_utilisation": round(toks["total"] / rr["cap"], 4),
            "budget_exhausted": bool(toks.get("budget_exhausted")),
            "termination": agent.get("termination"),
            "n_steps": agent.get("n_steps"),
            "n_calls": len(calls),
            "call_roles": Counter(c.get("role") for c in calls),
            "proposal_attempts": len(props),
            "proposal_valid": sum(1 for p in props if p["valid"]),
            "proposals_by_role": Counter(p.get("role") for p in props),
            "valid_by_role": Counter(p.get("role") for p in props if p["valid"]),
            "rejection_reasons": reasons,
            # Condition-specific, log-only. Empty for conditions that write none.
            "diagnostics": agent.get("diagnostics") or {},
        })
    return rows


def audit(diag: Sequence[dict[str, Any]], c1: Sequence[dict[str, Any]],
          conditions: Sequence[str] = DEFAULT_DIAGNOSTIC_CONDITIONS) -> dict[str, Any]:
    """Prove the sets are what the amendment described, before any rate is read."""
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    n_expected = N_INSTANCES_PER_BAND * len(BANDS)          # 60 per condition
    check(len(diag) == n_expected * len(conditions),
          f"expected {n_expected * len(conditions)} diagnostic runs for "
          f"{list(conditions)}, found {len(diag)}")

    ids = {r["run_id"] for r in diag}
    check(len(ids) == len(diag), f"{len(diag) - len(ids)} duplicate run_id(s)")

    caps = {r["cap"] for r in diag}
    check(caps == {DIAGNOSTIC_CAP},
          f"diagnostic caps {sorted(caps)}, expected only {DIAGNOSTIC_CAP}")

    per_cond = Counter(r["condition"] for r in diag)
    check(set(per_cond) == set(conditions),
          f"conditions {sorted(per_cond)}, expected {sorted(conditions)}")
    for cond in conditions:
        check(per_cond.get(cond, 0) == n_expected,
              f"{cond}: {per_cond.get(cond, 0)} runs, expected exactly {n_expected}")

    for label, rows in (("diagnostic", diag), ("C1 baseline", c1)):
        subsets = {r["subset_hash"] for r in rows}
        check(subsets == {EXPECTED_SUBSET_HASH},
              f"{label} runs carry subset hash(es) "
              f"{sorted(h[:12] for h in subsets if h)}, expected "
              f"{EXPECTED_SUBSET_HASH[:12]}")
        binnings = {r["binning_hash"] for r in rows}
        check(binnings == {EXPECTED_BINNING_HASH},
              f"{label} runs carry binning hash(es) "
              f"{sorted(h[:12] for h in binnings if h)}, expected "
              f"{EXPECTED_BINNING_HASH[:12]}")
        seeds = {r["seed"] for r in rows}
        stray = sorted(s for s in seeds if not 30000 <= s <= 39999)
        check(not stray, f"{label}: {len(stray)} run(s) outside the development seed "
                         f"range: {stray[:8]}")

    models = {r["model"] for r in diag} | {r["model"] for r in c1}
    check(len(models) == 1,
          f"more than one model across the compared sets: {sorted(models)}")

    # The C1 comparison arm: exactly the 60 runs at this cap, no more.
    c1_at_cap = [r for r in c1 if r["cap"] == DIAGNOSTIC_CAP]
    check({r["condition"] for r in c1_at_cap} <= {BASELINE_CONDITION},
          f"C1 arm contains other conditions: "
          f"{sorted({r['condition'] for r in c1_at_cap})}")
    check(len(c1_at_cap) == n_expected,
          f"C1 baseline at cap {DIAGNOSTIC_CAP}: {len(c1_at_cap)} runs, expected "
          f"{n_expected}")

    # Pairing: every condition must cover the identical instance set.
    inst_c1 = {r["instance_id"] for r in c1_at_cap}
    for cond in conditions:
        inst = {r["instance_id"] for r in diag if r["condition"] == cond}
        check(len(inst) == n_expected,
              f"{cond}: {len(inst)} distinct instances, expected {n_expected}")
        missing = sorted(inst_c1 - inst)
        extra = sorted(inst - inst_c1)
        check(not missing, f"{cond}: {len(missing)} instance(s) present for C1 and "
                           f"missing here: {missing[:5]}")
        check(not extra, f"{cond}: {len(extra)} instance(s) not in the C1 arm: "
                         f"{extra[:5]}")

    for band in BANDS:
        for cond in conditions:
            n = len({r["instance_id"] for r in diag
                     if r["condition"] == cond and r["band"] == band})
            check(n == N_INSTANCES_PER_BAND,
                  f"{cond} / band {band}: {n} instances, expected "
                  f"{N_INSTANCES_PER_BAND}")

    if problems:
        raise DiagnosticAuditError(
            "architecture diagnostic failed its completeness audit:\n  - "
            + "\n  - ".join(problems))

    # Provenance is reported PER CONDITION. Conditions run weeks apart necessarily carry
    # different commits, and that is not a defect; a condition whose own runs are split
    # across commits is. Only the latter is worth flagging.
    per_commit = {c: sorted({r["git_commit"] for r in diag
                             if r["condition"] == c and r["git_commit"]})
                  for c in conditions}
    split = [c for c, cs in per_commit.items() if len(cs) > 1]
    commits = sorted({c for cs in per_commit.values() for c in cs})
    return {
        "n_diagnostic_runs": len(diag),
        "n_c1_baseline_runs": len(c1_at_cap),
        "conditions": sorted(per_cond),
        "cap": DIAGNOSTIC_CAP,
        "model": sorted(models)[0],
        "instances_per_band": N_INSTANCES_PER_BAND,
        "subset_hash": EXPECTED_SUBSET_HASH,
        "binning_hash": EXPECTED_BINNING_HASH,
        "paired_on": "identical instance_id across c1_react, c2_verify_revise, c3_mas",
        "diagnostic_git_commits": commits,
        "git_commit_per_condition": per_commit,
        "conditions_with_split_provenance": split,
        "single_provenance_commit": len(commits) <= 1,
        "all_checks_passed": True,
    }


def _block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Everything reported for one (condition, band) or (condition, pooled) cell."""
    if not rows:
        return {}
    n = len(rows)
    attempts = sum(r["proposal_attempts"] for r in rows)
    call_roles: Counter = Counter()
    props_role: Counter = Counter()
    valid_role: Counter = Counter()
    reasons: Counter = Counter()
    for r in rows:
        call_roles.update(r["call_roles"])
        props_role.update(r["proposals_by_role"])
        valid_role.update(r["valid_by_role"])
        reasons.update(r["rejection_reasons"])
    hits = sum(1 for r in rows if r["nonempty_validated"])
    return {
        "n": n,
        "nonempty_validated": hits,
        "nonempty_validated_rate": round(hits / n, 4),
        "at_least_reference_rate": bool(hits / n >= REFERENCE_RATE),
        "mean_satisfaction": round(sum(r["satisfaction"] for r in rows) / n, 4),
        "n_satisfaction_zero": sum(1 for r in rows if r["satisfaction"] == 0.0),
        "n_satisfaction_one": sum(1 for r in rows if r["satisfaction"] == 1.0),
        "mean_tokens": round(sum(r["tokens_total"] for r in rows) / n, 1),
        "mean_cap_utilisation": round(sum(r["cap_utilisation"] for r in rows) / n, 4),
        "n_budget_exhausted": sum(1 for r in rows if r["budget_exhausted"]),
        "mean_calls": round(sum(r["n_calls"] for r in rows) / n, 2),
        "call_roles": dict(sorted(call_roles.items())),
        "termination": dict(sorted(Counter(r["termination"] for r in rows).items())),
        "proposal_attempts": attempts,
        "proposal_valid": sum(r["proposal_valid"] for r in rows),
        "proposal_validity_rate": (round(sum(r["proposal_valid"] for r in rows) / attempts, 4)
                                   if attempts else None),
        "proposals_by_role": dict(sorted(props_role.items())),
        "valid_by_role": dict(sorted(valid_role.items())),
        "rejection_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def _c2_stage_reach(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How far C2 actually got through the fixed Draft -> V -> R -> V -> R structure."""
    out: dict[str, Any] = {"n": len(rows)}
    for role in C2_STAGE_ROLES:
        reached = sum(1 for r in rows if r["call_roles"].get(role, 0) >= 1)
        twice = sum(1 for r in rows if r["call_roles"].get(role, 0) >= 2)
        out[f"runs_with_{role}_at_least_1"] = reached
        out[f"runs_with_{role}_at_least_2"] = twice
        out[f"rate_{role}_at_least_1"] = round(reached / len(rows), 4) if rows else None
        out[f"mean_{role}_calls"] = (
            round(sum(r["call_roles"].get(role, 0) for r in rows) / len(rows), 3)
            if rows else None)
    out["runs_reaching_full_two_cycles"] = sum(
        1 for r in rows
        if r["call_roles"].get("verify", 0) >= 2 and r["call_roles"].get("revise", 0) >= 2)
    return out


def _c3_split(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C3 proposal validity split by worker versus critic, plus critic reach.

    Kept separate from the pooled rate on purpose: workers solve a decomposed
    sub-problem and the critic is gated by the validator, so the two are not
    comparable with each other, nor with C1's single-trajectory rate.
    """
    workers_att = workers_val = critic_att = critic_val = 0
    for r in rows:
        for role in C3_SEARCH_ROLES:
            workers_att += r["proposals_by_role"].get(role, 0)
            workers_val += r["valid_by_role"].get(role, 0)
        critic_att += r["proposals_by_role"].get(C3_CRITIC_ROLE, 0)
        critic_val += r["valid_by_role"].get(C3_CRITIC_ROLE, 0)
    critic_ran = sum(1 for r in rows if r["call_roles"].get(C3_CRITIC_ROLE, 0) >= 1)
    return {
        "n": len(rows),
        "worker_proposal_attempts": workers_att,
        "worker_proposal_valid": workers_val,
        "worker_validity_rate": round(workers_val / workers_att, 4) if workers_att else None,
        "critic_proposal_attempts": critic_att,
        "critic_proposal_valid": critic_val,
        "critic_validity_rate": round(critic_val / critic_att, 4) if critic_att else None,
        "runs_where_critic_ran": critic_ran,
        "critic_reach_rate": round(critic_ran / len(rows), 4) if rows else None,
        "note": "worker and critic rates are not comparable with each other or with C1: "
                "workers solve a decomposed sub-problem and the critic is validator-gated",
    }



def _condition_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean/positive counts over a condition's own log-only diagnostics block.

    C4 and C5 each record structure the others have no analogue for -- pool size and
    evidence coverage, trajectory products and the winning attempt. Averaged here so a
    degenerate condition is visible as a number rather than inferred from the rate.
    """
    blocks = [r["diagnostics"] for r in rows if r["diagnostics"]]
    if not blocks:
        return {"n_with_diagnostics": 0}
    out: dict[str, Any] = {"n_with_diagnostics": len(blocks)}
    numeric: dict[str, list[float]] = defaultdict(list)
    flags: Counter = Counter()
    winners: Counter = Counter()
    for b in blocks:
        for k, v in b.items():
            if isinstance(v, bool):
                flags[k] += int(v)
            elif isinstance(v, (int, float)):
                numeric[k].append(float(v))
        if "winner_attempt_index" in b:
            winners[b["winner_attempt_index"]] += 1
        # `n_distinct_products` is already in `numeric`: the loop above takes every int.
        # Appending it a second time here doubled every bucket of the histogram below
        # while leaving the mean unchanged, which is exactly the kind of error that reads
        # as plausible. The histogram is built from the same list, once.
    for k, vals in sorted(numeric.items()):
        out[f"mean_{k}"] = round(sum(vals) / len(vals), 4)
    for k, n in sorted(flags.items()):
        out[f"runs_{k}"] = n
        out[f"rate_{k}"] = round(n / len(blocks), 4)
    if winners:
        out["winner_attempt_counts"] = {str(k): v for k, v in sorted(winners.items())}
    if "n_distinct_products" in numeric:
        vals = numeric["n_distinct_products"]
        assert len(vals) == len(blocks), (
            f"distinct-product histogram would cover {len(vals)} values for "
            f"{len(blocks)} runs")
        out["distinct_product_counts"] = {
            str(n): sum(1 for v in vals if int(v) == n) for n in (1, 2, 3)}
        # Cross-tabulated against the outcome, because equal marginal counts do not
        # establish a run-by-run correspondence: "54 runs with one product" and "54 runs
        # scoring zero" can be the same 54 runs or merely the same number of them, and
        # only this table tells them apart.
        cross: dict[str, dict[str, int]] = {}
        for r in rows:
            d = r["diagnostics"]
            if "n_distinct_products" not in d:
                continue
            key = str(int(d["n_distinct_products"]))
            cell = cross.setdefault(key, {"satisfaction_zero": 0, "satisfaction_positive": 0})
            cell["satisfaction_zero" if r["satisfaction"] == 0.0
                 else "satisfaction_positive"] += 1
        out["distinct_products_by_outcome"] = dict(sorted(cross.items()))
    return out


def _paired(diag: Sequence[dict], c1: Sequence[dict], cond: str) -> dict[str, Any]:
    """Instance-paired comparison against the completed C1 arm. Counts, no test."""
    base = {r["instance_id"]: r for r in c1}
    rows = [r for r in diag if r["condition"] == cond]
    both = won = lost = tied = 0
    sat_delta = 0.0
    for r in rows:
        b = base.get(r["instance_id"])
        if b is None:
            continue
        both += 1
        sat_delta += r["satisfaction"] - b["satisfaction"]
        if r["nonempty_validated"] and not b["nonempty_validated"]:
            won += 1
        elif b["nonempty_validated"] and not r["nonempty_validated"]:
            lost += 1
        else:
            tied += 1
    return {
        "n_paired": both,
        "recovered_where_c1_failed": won,
        "lost_where_c1_succeeded": lost,
        "same_outcome": tied,
        "mean_satisfaction_delta_vs_c1": round(sat_delta / both, 4) if both else None,
        "definition": "delta = S(this condition) - S(c1_react) on the identical instance",
        "note": "counts only; no hypothesis test is performed by this diagnostic",
    }


def analyse(diag: Sequence[dict[str, Any]], c1_all: Sequence[dict[str, Any]],
            conditions: Sequence[str] = DEFAULT_DIAGNOSTIC_CONDITIONS) -> dict[str, Any]:
    c1 = [r for r in c1_all if r["cap"] == DIAGNOSTIC_CAP]
    by_cond: dict[str, Any] = {}
    for cond in (BASELINE_CONDITION, *conditions):
        rows = c1 if cond == BASELINE_CONDITION else [r for r in diag
                                                      if r["condition"] == cond]
        entry: dict[str, Any] = {
            "pooled": _block(rows),
            "by_band": {b: _block([r for r in rows if r["band"] == b]) for b in BANDS},
        }
        if cond == "c2_verify_revise":
            entry["stage_reach"] = _c2_stage_reach(rows)
        if cond == "c3_mas":
            entry["role_split"] = _c3_split(rows)
        if cond in ("c4_planner_critic", "c5_best_of_3"):
            entry["condition_diagnostics"] = _condition_diagnostics(rows)
        if cond != BASELINE_CONDITION:
            entry["paired_vs_c1"] = _paired(diag, c1, cond)
        by_cond[cond] = entry

    clears = {
        cond: all(by_cond[cond]["by_band"][b]["at_least_reference_rate"] for b in BANDS)
        for cond in conditions
    }
    any_clears = any(clears.values())
    return {
        "by_condition": by_cond,
        "reference_rate": REFERENCE_RATE,
        "reaches_reference_in_every_band": clears,
        "reading": {
            "verdict": ("NOT_A_UNIVERSAL_TASK_WIDE_FLOOR" if any_clears
                        else "N8_REMAINS_PROBLEMATIC"),
            "statement": (
                "at least one structured condition reaches the reference rate in every "
                "band, so the n = 8 floor is not universal across architectures"
                if any_clears else
                "no structured condition reaches the reference rate in every band, so "
                "n = 8 remains problematic across the condition set and instance size "
                "and design should be reconsidered"),
            "does_not_repair_the_c1_gate": (
                "The C1 budget gate remains failed. It read C1 only, by design, so this "
                "diagnostic is not evidence about it in either direction; no primary cap "
                "is selected here and the main experiment does not start on this result."),
            "status": "development diagnostic, descriptive; no hypothesis test, no "
                      "threshold tuned from these outcomes",
        },
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, au = doc["analysis"], doc["audit"]
    conds = tuple(a["by_condition"])

    add("# Post-gate architecture diagnostic — n = 8, cap 64 000\n")
    add("**Development diagnostic, descriptive.** It cannot repair the C1 budget gate: "
        "that gate reads C1 only by design, so a C2 or C3 result is not evidence about it "
        "in either direction. No cap is selected here.\n")
    add(f"Runs: {au['n_diagnostic_runs']} diagnostic + {au['n_c1_baseline_runs']} C1 "
        f"baseline, model `{au['model']}`, subset `{au['subset_hash'][:12]}`, paired on "
        "the identical instance. Completeness audit passed.\n")
    if not au["single_provenance_commit"]:
        add(f"> **Mixed provenance:** the diagnostic runs carry "
            f"{len(au['diagnostic_git_commits'])} different git commits.\n")

    add("\n## Validator-approved non-empty plan rate\n")
    add("| condition | Low | Medium | High | pooled |")
    add("|---|---|---|---|---|")
    for c in conds:
        cells = []
        for b in BANDS:
            blk = a["by_condition"][c]["by_band"][b]
            mark = "✓" if blk["at_least_reference_rate"] else "✗"
            cells.append(f"{blk['nonempty_validated']}/{blk['n']} = "
                         f"{blk['nonempty_validated_rate']:.2f} {mark}")
        p = a["by_condition"][c]["pooled"]
        add(f"| {c} | " + " | ".join(cells)
            + f" | {p['nonempty_validated']}/{p['n']} = "
              f"{p['nonempty_validated_rate']:.2f} |")
    add(f"\n`✓` marks a band at or above the reference rate of "
        f"{a['reference_rate']:.0%}, reused from the registered gate. It is a reference "
        "point here, not a decision rule.\n")

    add("\n## Reading, fixed before the outcomes\n")
    add(f"**{a['reading']['verdict']}** — {a['reading']['statement']}.\n")
    add(f"{a['reading']['does_not_repair_the_c1_gate']}\n")

    add("\n## Satisfaction, cost and behaviour (pooled)\n")
    add("| condition | mean sat | zeros | mean tokens | cap used | calls | "
        "proposal validity |")
    add("|---|---|---|---|---|---|---|")
    for c in conds:
        p = a["by_condition"][c]["pooled"]
        pv = p["proposal_validity_rate"]
        add(f"| {c} | {p['mean_satisfaction']:.3f} | {p['n_satisfaction_zero']}/{p['n']} "
            f"| {p['mean_tokens']:.0f} | {p['mean_cap_utilisation']:.2f} | "
            f"{p['mean_calls']:.1f} | " + (f"{pv:.3f}" if pv is not None else "—") + " |")

    add("\n## Paired against the completed C1 arm\n")
    add("| condition | recovered where C1 failed | lost where C1 succeeded | same | "
        "mean Δ satisfaction |")
    add("|---|---|---|---|---|")
    for c in conds:
        if c == BASELINE_CONDITION:
            continue
        pr = a["by_condition"][c]["paired_vs_c1"]
        add(f"| {c} | {pr['recovered_where_c1_failed']} | "
            f"{pr['lost_where_c1_succeeded']} | {pr['same_outcome']} | "
            f"{pr['mean_satisfaction_delta_vs_c1']:+.3f} |")
    add("\nCounts only. No hypothesis test is performed by this diagnostic.\n")

    if "c2_verify_revise" in conds:
        sr = a["by_condition"]["c2_verify_revise"]["stage_reach"]
        add("\n## C2 stage reach\n")
        add("| stage | runs reaching it once | runs reaching it twice | mean calls |")
        add("|---|---|---|---|")
        for role in C2_STAGE_ROLES:
            add(f"| {role} | {sr[f'runs_with_{role}_at_least_1']}/{sr['n']} | "
                f"{sr[f'runs_with_{role}_at_least_2']}/{sr['n']} | "
                f"{sr[f'mean_{role}_calls']} |")
        add(f"\nRuns completing both full cycles: "
            f"{sr['runs_reaching_full_two_cycles']}/{sr['n']}. C2 is the only structured "
            "condition whose later stages have no reserved budget, so a stage that was "
            "never reached is a fidelity fact about the run, not a model choice.\n")

    if "c3_mas" in conds:
        rs = a["by_condition"]["c3_mas"]["role_split"]
        add("\n## C3 by role\n")
        add("| role | proposal attempts | valid | validity rate |")
        add("|---|---|---|---|")
        add(f"| workers | {rs['worker_proposal_attempts']} | "
            f"{rs['worker_proposal_valid']} | "
            + (f"{rs['worker_validity_rate']:.3f}"
               if rs["worker_validity_rate"] is not None else "—") + " |")
        add(f"| critic | {rs['critic_proposal_attempts']} | "
            f"{rs['critic_proposal_valid']} | "
            + (f"{rs['critic_validity_rate']:.3f}"
               if rs["critic_validity_rate"] is not None else "—") + " |")
        add(f"\nThe critic ran in {rs['runs_where_critic_ran']}/{rs['n']} runs "
            f"({rs['critic_reach_rate']:.2f}). {rs['note']}.\n")

    for c in conds:
        cd = (a["by_condition"].get(c) or {}).get("condition_diagnostics")
        if cd:
            add(f"\n## {c} — condition diagnostics\n")
            add("Log-only quantities the condition records about its own structure. A "
                "degenerate condition shows up here as a number rather than being "
                "inferred from the rate.\n")
            add("| quantity | value |")
            add("|---|---|")
            for k, v in cd.items():
                if k == "distinct_products_by_outcome":
                    continue
                add(f"| `{k}` | {v} |")
            cross = cd.get("distinct_products_by_outcome")
            if cross:
                add("\nDistinct final products against the outcome. Equal marginal "
                    "counts would not establish a run-by-run correspondence; this table "
                    "does.\n")
                add("| distinct products | satisfaction = 0 | satisfaction > 0 |")
                add("|---|---|---|")
                for k, cell in cross.items():
                    add(f"| {k} | {cell['satisfaction_zero']} | "
                        f"{cell['satisfaction_positive']} |")

    add("\n## Why proposals were rejected\n")
    add("| condition | " + " | ".join(
        sorted({r for c in conds
                for r in a["by_condition"][c]["pooled"]["rejection_reasons"]})) + " |")
    keys = sorted({r for c in conds
                   for r in a["by_condition"][c]["pooled"]["rejection_reasons"]})
    add("|---" * (len(keys) + 1) + "|")
    for c in conds:
        rr = a["by_condition"][c]["pooled"]["rejection_reasons"]
        add(f"| {c} | " + " | ".join(str(rr.get(k, 0)) for k in keys) + " |")
    add("\nMulti-label over rejected proposals, so a row can exceed the number of "
        "rejections.\n")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag-runs", type=Path,
                    default=Path("results/logs/budget_dev_arch_diag"))
    ap.add_argument("--c1-runs", type=Path, default=Path("results/logs/budget_dev"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/budget_dev_arch_diag"))
    ap.add_argument("--conditions",
                    default=",".join(DEFAULT_DIAGNOSTIC_CONDITIONS),
                    help="comma-separated; the default reproduces the frozen C2/C3 run")
    args = ap.parse_args(argv)

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conditions if c not in KNOWN_DIAGNOSTIC_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown diagnostic condition(s): {unknown}; expected among "
                         f"{list(KNOWN_DIAGNOSTIC_CONDITIONS)}")

    # Only the declared conditions are read, so adding a later condition to the same
    # directory cannot change what an earlier invocation reports.
    diag = [r for r in load_runs(args.diag_runs) if r["condition"] in conditions]
    c1 = load_runs(args.c1_runs)
    audit_report = audit(diag, c1, conditions)
    analysis = analyse(diag, c1, conditions)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "post-gate development diagnostic; descriptive, not confirmatory",
        "cannot_repair_the_c1_gate": True,
        "registered": "THESIS_DECISIONS.md section 5, amendment of 2026-08-11",
        "question": ("on the same n = 8 development instances at 64 000, do C2 and C3 "
                     "recover validator-approved non-empty plans substantially more "
                     "often than plain C1?"),
        "reported_quantity": "validator-approved AND at least one meeting; a generic "
                             "feasibility rate is ~1.0 by construction in this harness",
        "audit": audit_report,
        "analysis": analysis,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    print(f"audit passed: {audit_report['n_diagnostic_runs']} diagnostic runs + "
          f"{audit_report['n_c1_baseline_runs']} C1 baseline, paired by instance")
    print()
    print(f"{'condition':<20}" + "".join(f"{b:>16}" for b in BANDS) + f"{'pooled':>16}")
    for cond in (BASELINE_CONDITION, *conditions):
        cells = []
        for b in BANDS:
            blk = analysis["by_condition"][cond]["by_band"][b]
            cells.append(f"{blk['nonempty_validated']:>2}/{blk['n']:<2} "
                         f"{blk['nonempty_validated_rate']:.2f} "
                         + ("ok" if blk["at_least_reference_rate"] else "--"))
        p = analysis["by_condition"][cond]["pooled"]
        print(f"{cond:<20}" + "".join(f"{c:>16}" for c in cells)
              + f"{p['nonempty_validated']:>6}/{p['n']:<3} {p['nonempty_validated_rate']:.2f}")
    print()
    print(f"READING : {analysis['reading']['verdict']}")
    print(f"          {analysis['reading']['statement']}")
    print(f"NOTE    : {analysis['reading']['does_not_repair_the_c1_gate']}")
    print(f"written : {args.out}/summary.json, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
