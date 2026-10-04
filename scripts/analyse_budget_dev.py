# scripts/analyse_budget_dev.py
"""Budget calibration: apply the registered primary-cap rule to the budget-dev runs.

    python -m scripts.analyse_budget_dev --runs results/logs/budget_dev \
        --out results/analysis/budget_dev

This script **applies a rule; it does not choose one**. The rule was registered in
THESIS_DECISIONS section 5 before the development set existed and again, with the ladder
named explicitly, before a single run was launched:

> **Primary cap** = the smallest cap in the frozen ladder at which C1 produces a non-empty
> validated plan in at least 50% of budget-dev runs **in each of the three frozen density
> regimes separately**.

Separately, not pooled. A pooled rate can pass on 90% at Low and 0% at High, which would
select a budget at which the High regime sits on the floor and the crossover the experiment
is built to detect is unreachable by construction.

> **Failure rule.** If no cap reaches 50% in all three regimes, no primary cap is selected,
> budget calibration is declared failed, and the main experiment does not start.

Falling back to the top rung because nothing qualified is not available, and this script
will not print one. A NO_CAP verdict is a result, not an error to be worked around.

> **Secondary cap** = the immediately preceding rung, or the next rung above if the primary
> is the lowest. Mechanical, so no rung is chosen after seeing which looks more interesting.

*Success, defined structurally rather than through the score* (section 5): a run counts if
its final plan is validator-approved **and** contains at least one meeting. Both conditions
are applied as written. Final-plan feasibility is close to 100% by construction, so the
operative condition is almost always non-emptiness -- which is exactly why the registered
definition names both rather than relying on either alone.

*What is not decided here.* Satisfaction, the optimality gap, the proposal taxonomy and the
termination mix are reported because they describe what the budget bought, and because the
next step -- choosing `N` from real spread and real tie structure -- reads them. **None of
them enters the cap decision.** The cap rule sees one binary per run and nothing else.

The frozen subset is pinned by hash. A different selection cannot be analysed by this
script by accident: the numbers would look identical and mean something else.
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

SCHEMA_VERSION = "budget_dev_analysis/1.0"

EXPECTED_SUBSET_HASH = (
    "911e4864d6fb8ea2a78ff465f8e55a286d3c48afc1b8403d5e63f231bf96fd91")
EXPECTED_BINNING_HASH = (
    "8e671cd13dbb262b638048eff58241226f09d5804373534789915bc302ba0dff")

FROZEN_LADDER = (8000, 16000, 32000, 64000)
CONDITION = "c1_react"
N_INSTANCES_PER_BAND = 20
SUCCESS_THRESHOLD = 0.50            # registered, section 5

# The runner's level labels are a compatibility alias for the frozen density bands.
BAND_OF_LEVEL = {"easy": "low", "medium": "medium", "hard": "high"}
BANDS = ("low", "medium", "high")


class AuditError(RuntimeError):
    """The run set is not the one the rule was registered against."""


def _plan_size(plan: Any) -> int:
    if not isinstance(plan, dict):
        return 0
    return max((len(v) for v in plan.values() if isinstance(v, list)), default=0)


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        score, toks = rr["score"], rr["tokens"]
        md = doc.get("pilot_sweep") or {}
        instance = doc.get("instance") or {}
        gp = instance.get("generator_params") or {}
        meetings = _plan_size(rr.get("final_plan"))
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
            "complexity_metric": instance.get("complexity_metric"),
            "tightness": gp.get("tightness"),
            "travel_structure": gp.get("travel_structure"),
            "optimum": score["solver_optimum"],
            "achieved": score["n_valid_meetings"],
            "satisfaction": score["satisfaction"],
            "feasible": bool(score.get("valid")),
            "final_plan_meetings": meetings,
            # The registered definition, both halves, applied as written.
            "counts_for_cap_rule": bool(score.get("valid")) and meetings >= 1,
            "tokens_total": toks["total"],
            "cap_utilisation": round(toks["total"] / rr["cap"], 4),
            "budget_exhausted": bool(toks.get("budget_exhausted")),
            "termination": agent.get("termination"),
            "n_steps": agent.get("n_steps"),
            "proposal_attempts": len(agent.get("proposals") or []),
            "proposal_valid": sum(1 for p in (agent.get("proposals") or []) if p["valid"]),
        })
    return rows


def audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prove the set is complete and is the frozen selection, before any rate is read.

    Same principle as the formal pilot's completeness audit: an incomplete or mixed set
    still yields a perfectly readable rate, and a cap chosen from it would mean nothing.
    """
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    expected = N_INSTANCES_PER_BAND * len(BANDS) * len(FROZEN_LADDER)
    check(len(rows) == expected, f"expected {expected} runs, found {len(rows)}")

    ids = {r["run_id"] for r in rows}
    check(len(ids) == len(rows), f"{len(rows) - len(ids)} duplicate run_id(s)")

    conditions = {r["condition"] for r in rows}
    check(conditions <= {CONDITION}, f"conditions beyond {CONDITION}: {sorted(conditions)}")

    caps = {r["cap"] for r in rows}
    check(caps == set(FROZEN_LADDER),
          f"caps {sorted(caps)} are not the frozen ladder {list(FROZEN_LADDER)}")

    subset_hashes = {r["subset_hash"] for r in rows}
    check(subset_hashes == {EXPECTED_SUBSET_HASH},
          f"runs carry subset hash(es) {sorted(h[:12] for h in subset_hashes if h)}, "
          f"expected {EXPECTED_SUBSET_HASH[:12]}")
    binning_hashes = {r["binning_hash"] for r in rows}
    check(binning_hashes == {EXPECTED_BINNING_HASH},
          f"runs carry binning hash(es) {sorted(h[:12] for h in binning_hashes if h)}, "
          f"expected {EXPECTED_BINNING_HASH[:12]}")

    models = {r["model"] for r in rows}
    check(len(models) == 1, f"more than one model in the set: {sorted(models)}")

    by_band_instances = defaultdict(set)
    for r in rows:
        by_band_instances[r["band"]].add(r["instance_id"])
    for band in BANDS:
        n = len(by_band_instances.get(band, ()))
        check(n == N_INSTANCES_PER_BAND,
              f"band {band}: {n} distinct instances, expected {N_INSTANCES_PER_BAND}")

    cells = Counter((r["instance_id"], r["cap"]) for r in rows)
    repeated = [k for k, v in cells.items() if v > 1]
    check(not repeated, f"{len(repeated)} (instance, cap) cell(s) run more than once")
    holes = [(i, c) for i in {r["instance_id"] for r in rows} for c in FROZEN_LADDER
             if (i, c) not in cells]
    check(not holes, f"{len(holes)} missing (instance, cap) cell(s): {holes[:5]}")

    seeds = {r["seed"] for r in rows}
    stray = sorted(s for s in seeds if not 30000 <= s <= 39999)
    check(not stray, f"{len(stray)} run(s) outside the development seed range: {stray[:8]}")

    if problems:
        raise AuditError("budget-dev run set failed its completeness audit:\n  - "
                         + "\n  - ".join(problems))
    return {
        "n_runs": len(rows),
        "n_expected": expected,
        "distinct_run_ids": len(ids),
        "condition": sorted(conditions)[0],
        "caps": sorted(caps),
        "model": sorted(models)[0],
        "instances_per_band": {b: len(by_band_instances[b]) for b in BANDS},
        "subset_hash": EXPECTED_SUBSET_HASH,
        "binning_hash": EXPECTED_BINNING_HASH,
        "seed_range_observed": [min(seeds), max(seeds)],
        "all_checks_passed": True,
    }


def success_rates(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """The one quantity the rule reads: per cap, per band, the share of qualifying runs."""
    out: dict[int, dict[str, Any]] = {}
    for cap in FROZEN_LADDER:
        per_band: dict[str, Any] = {}
        for band in BANDS:
            sel = [r for r in rows if r["cap"] == cap and r["band"] == band]
            hits = sum(r["counts_for_cap_rule"] for r in sel)
            per_band[band] = {
                "n": len(sel),
                "qualifying": hits,
                "rate": round(hits / len(sel), 4) if sel else None,
                "meets_threshold": bool(sel) and hits / len(sel) >= SUCCESS_THRESHOLD,
            }
        out[cap] = {
            "per_band": per_band,
            "all_bands_meet_threshold": all(per_band[b]["meets_threshold"] for b in BANDS),
            "worst_band": min(BANDS, key=lambda b: per_band[b]["rate"] or 0.0),
        }
    return out


def select_caps(rates: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The registered rule, applied without discretion."""
    qualifying = [c for c in FROZEN_LADDER if rates[c]["all_bands_meet_threshold"]]
    if not qualifying:
        return {
            "verdict": "NO_CAP",
            "primary_cap": None,
            "secondary_cap": None,
            "rule": ("smallest rung of the frozen ladder with at least "
                     f"{SUCCESS_THRESHOLD:.0%} qualifying runs in EACH band"),
            "consequence": ("budget calibration is declared failed and the main experiment "
                            "does not start; falling back to the top rung because nothing "
                            "qualified is not available"),
        }
    primary = min(qualifying)
    i = FROZEN_LADDER.index(primary)
    secondary = FROZEN_LADDER[i - 1] if i > 0 else FROZEN_LADDER[1]
    return {
        "verdict": "CAP_SELECTED",
        "primary_cap": primary,
        "secondary_cap": secondary,
        "secondary_rule": ("the immediately preceding rung, or the next above if the "
                           "primary is the lowest"),
        "qualifying_caps": qualifying,
        "rule": ("smallest rung of the frozen ladder with at least "
                 f"{SUCCESS_THRESHOLD:.0%} qualifying runs in EACH band"),
    }


def describe(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Non-binding description of what the budget bought. Never enters the cap decision."""
    def block(sel: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not sel:
            return {}
        n = len(sel)
        attempts = sum(r["proposal_attempts"] for r in sel)
        return {
            "n": n,
            "mean_satisfaction": round(sum(r["satisfaction"] for r in sel) / n, 4),
            "n_satisfaction_zero": sum(1 for r in sel if r["satisfaction"] == 0.0),
            "n_satisfaction_one": sum(1 for r in sel if r["satisfaction"] == 1.0),
            "mean_tokens": round(sum(r["tokens_total"] for r in sel) / n, 1),
            "mean_cap_utilisation": round(sum(r["cap_utilisation"] for r in sel) / n, 4),
            "n_budget_exhausted": sum(1 for r in sel if r["budget_exhausted"]),
            "nonempty_plan_rate": round(
                sum(1 for r in sel if r["final_plan_meetings"] >= 1) / n, 4),
            "final_plan_feasible_rate": round(sum(1 for r in sel if r["feasible"]) / n, 4),
            "proposal_attempts": attempts,
            "proposal_validity_rate": (round(sum(r["proposal_valid"] for r in sel) / attempts, 4)
                                       if attempts else None),
            "termination": dict(Counter(r["termination"] for r in sel)),
        }

    out: dict[str, Any] = {"by_cap": {}, "by_cap_and_band": {}}
    for cap in FROZEN_LADDER:
        out["by_cap"][str(cap)] = block([r for r in rows if r["cap"] == cap])
        for band in BANDS:
            out["by_cap_and_band"][f"{cap}|{band}"] = block(
                [r for r in rows if r["cap"] == cap and r["band"] == band])
    return out


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    rates, decision = doc["success_rates"], doc["decision"]

    add("# Budget calibration — the registered primary-cap rule applied\n")
    add("The rule was fixed in THESIS_DECISIONS section 5 before this development set "
        "existed, and the ladder was named explicitly before any run was launched. This "
        "document applies it; it does not choose it.\n")
    add(f"Run set: **{doc['audit']['n_runs']}** runs, condition `{doc['audit']['condition']}`, "
        f"model `{doc['audit']['model']}`, subset `{doc['audit']['subset_hash'][:12]}`, "
        f"seeds {doc['audit']['seed_range_observed'][0]}–"
        f"{doc['audit']['seed_range_observed'][1]}. Completeness audit passed.\n")
    add("**Success**, structurally defined: the final plan is validator-approved *and* "
        "contains at least one meeting. Nothing else enters the decision — not "
        "satisfaction, not the optimality gap, not the proposal taxonomy.\n")

    add("\n## The quantity the rule reads\n")
    add("| cap | Low | Medium | High | all bands ≥ 50% |")
    add("|---|---|---|---|---|")
    for cap in FROZEN_LADDER:
        r = rates[str(cap)] if str(cap) in rates else rates[cap]
        cells = []
        for band in BANDS:
            b = r["per_band"][band]
            mark = "✓" if b["meets_threshold"] else "✗"
            cells.append(f"{b['qualifying']}/{b['n']} = {b['rate']:.2f} {mark}")
        add(f"| {cap} | " + " | ".join(cells) + " | "
            + ("**yes**" if r["all_bands_meet_threshold"] else "no") + " |")

    add("\n## Verdict\n")
    if decision["verdict"] == "NO_CAP":
        add("### NO CAP SELECTED\n")
        add("No rung of the frozen ladder reaches 50% qualifying runs in all three density "
            "bands.\n")
        add(f"**{decision['consequence']}**\n")
        add("The bands, the estimand and the inference rule are unaffected: what failed is "
            "the budget at which this single-agent baseline can be run at all, which is a "
            "finding about the condition and the instance size, not about the design.\n")
    else:
        add(f"### Primary cap: **{decision['primary_cap']}** — "
            f"secondary cap: **{decision['secondary_cap']}**\n")
        add(f"Qualifying rungs: {decision['qualifying_caps']}. The smallest was taken, as "
            "registered. The secondary is "
            f"{decision['secondary_rule']} — mechanical, not chosen.\n")

    add("\n## What the budget bought — non-binding\n")
    add("Reported because the next step reads it, and because it describes the runs. None "
        "of it entered the decision above.\n")
    add("| cap | band | mean sat | zeros | non-empty plan | mean tokens | cap used | "
        "proposal validity |")
    add("|---|---|---|---|---|---|---|---|")
    for cap in FROZEN_LADDER:
        for band in BANDS:
            d = doc["description"]["by_cap_and_band"].get(f"{cap}|{band}") or {}
            if not d:
                continue
            add(f"| {cap} | {band} | {d['mean_satisfaction']:.3f} | "
                f"{d['n_satisfaction_zero']}/{d['n']} | {d['nonempty_plan_rate']:.2f} | "
                f"{d['mean_tokens']:.0f} | {d['mean_cap_utilisation']:.2f} | "
                + (f"{d['proposal_validity_rate']:.3f}"
                   if d["proposal_validity_rate"] is not None else "—") + " |")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/logs/budget_dev"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/budget_dev"))
    args = ap.parse_args(argv)

    rows = load_runs(args.runs)
    audit_report = audit(rows)
    rates = success_rates(rows)
    decision = select_caps(rates)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "apply the registered primary-cap rule; it decides nothing new",
        "registered_rule": {
            "threshold": SUCCESS_THRESHOLD,
            "per_band": "each band separately, never pooled",
            "success": "final plan validator-approved AND at least one meeting",
            "ladder": list(FROZEN_LADDER),
            "failure": "no cap selected, calibration failed, main experiment does not start",
        },
        "audit": audit_report,
        "success_rates": {str(c): v for c, v in rates.items()},
        "decision": decision,
        "description": describe(rows),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    print(f"audit passed: {audit_report['n_runs']} runs, "
          f"{audit_report['instances_per_band']} instances per band")
    print()
    print(f"{'cap':>7}  " + "  ".join(f"{b:>16}" for b in BANDS) + "   all bands")
    for cap in FROZEN_LADDER:
        r = rates[cap]
        cells = []
        for band in BANDS:
            b = r["per_band"][band]
            cells.append(f"{b['qualifying']:>2}/{b['n']:<2} {b['rate']:.2f} "
                         + ("ok " if b["meets_threshold"] else "-- "))
        print(f"{cap:>7}  " + "  ".join(f"{c:>16}" for c in cells)
              + ("   YES" if r["all_bands_meet_threshold"] else "   no"))
    print()
    if decision["verdict"] == "NO_CAP":
        print("VERDICT: NO CAP SELECTED")
        print(f"  {decision['consequence']}")
    else:
        print(f"VERDICT: primary cap {decision['primary_cap']}, "
              f"secondary cap {decision['secondary_cap']}")
    print(f"written: {args.out}/summary.json, report.md")
    return 0 if decision["verdict"] == "CAP_SELECTED" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
