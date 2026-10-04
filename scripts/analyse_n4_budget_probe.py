# scripts/analyse_n4_budget_probe.py
"""The n = 4 follow-up: 32 000 against 64 000 on the same frozen twelve. CPU only.

    python -m scripts.analyse_n4_budget_probe \
        --runs-32k results/logs/lowcx_calib \
        --runs-64k results/logs/lowcx_n4_64k_probe \
        --subset results/manifests/subset__lowcx_n4__74e57136280d.json \
        --expected results/manifests/expected_runs__lowcx_n4_64k_probe.json \
        --out results/analysis/lowcx_n4_64k_probe

Written and committed **while the 24 runs were still executing**, so its tables cannot have
been shaped by their outcome. The shape it produces is the one fixed in the pre-registration
of 2026-08-17 in THESIS_DECISIONS section 3.

**Exploratory development follow-up. Descriptive.** Twelve paired instances support
description and ordering, not inference. No p-value, no threshold, no gate, no cap selection,
and no numerical criterion may be introduced here that the pre-registration did not fix.

THE INFORMATIVENESS COUNT, AND WHY IT IS NOT DECORATION. Per call the agent is granted
`min(remaining - reserve - input, max_model_len - input)`, and the 32 768 window clamps every
call at both caps, so a larger cap buys **more calls, not longer ones**. A run that reproduces
its token total exactly executed the same trajectory twice and carries **no information**
about what the extra budget bought; averaging over twelve as though all twelve were informative
overstates the evidence. That error was caught once already on the 128 000 probe at `n = 8`,
where only 4 of 12 C1 runs diverged.

*Two candidate criteria, and the weaker one is correct.* Exceeding the lower cap is
sufficient evidence that a run was constrained by it, but **not necessary**, and taking it
as the definition undercounts a condition that subdivides its budget. C1 spends one
undivided working pool and diverges only when it would have run out. C3 splits the cap: its
worker quota is `3/8 * (cap - reserve)`, 11 904 at 32 000 against 23 904 at 64 000, so a C3
worker behaves differently long before any total approaches 32 000. The primary count is
therefore `runs_with_changed_execution` -- a differing token total is direct evidence that
the cap altered the run -- with the exceed-the-cap count reported beside it.

THE INTERPRETATION RULE IS QUOTED, NOT INVENTED. The pre-registration named two outcomes and
what each licenses before any of these runs existed. This script reports which one obtained
and reproduces the registered wording verbatim, so that the reading cannot drift after the
numbers are seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_expected_runs import audit, load_expected  # noqa: E402

SCHEMA_VERSION = "n4_budget_probe/1.0"
BASELINE_CAP = 32_000
PROBE_CAP = 64_000
CONDITIONS = ("c1_react", "c3_mas")
FIXED_OPTIMUM = 3
WINDOW = 32_768

# Fixed by the pre-registration, before any of these runs existed. Reproduced verbatim so
# that the reading cannot be rewritten once the numbers are known.
REGISTERED_READINGS = {
    "c1_at_or_above_c3": (
        "A candidate budget-dependent low-complexity regime has been observed. This is NOT "
        "a demonstrated statistical crossover -- twelve development instances cannot "
        "establish one -- and it may not be described as such. What it licenses is that "
        "such a regime is taken into account when the held-out design is fixed."),
    "c1_below_c3": (
        "C3 keeps the descriptive ordering even after the cap is doubled on the lowest "
        "frozen structural region. The lower-complexity calibration is then closed, and no "
        "easier development pool is built in response."),
}


class AnalysisRefused(RuntimeError):
    """The inputs do not support the comparison. Never worked around."""


def _plan_size(plan: Any) -> int:
    return len(plan.get("meetings") or []) if isinstance(plan, dict) else 0


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #
def load_instances(subset: Path) -> dict[str, dict[str, Any]]:
    doc = json.loads(subset.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for rows in doc["instances"].values():
        for e in rows:
            if e["optimum"] != FIXED_OPTIMUM:
                raise AnalysisRefused(
                    f"{e['instance_id']}: optimum {e['optimum']}, but this pool is matched "
                    f"at {FIXED_OPTIMUM}")
            out[e["instance_id"]] = {
                "seed": e["seed"], "level": e["level"],
                "conflict_pairs": e["complexity_metric"],
                "n_people": e["cell"]["n_people"],
                "travel_structure": e["cell"]["travel_structure"],
            }
    return out


def load_records(run_dir: Path, instances: dict[str, Any], cap: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        if (rr["instance_id"] not in instances or rr["condition"] not in CONDITIONS
                or rr["cap"] != cap):
            continue
        score, toks = rr["score"], rr["tokens"]
        md = doc.get("pilot_sweep") or {}
        props = agent.get("proposals") or []
        meetings = _plan_size(rr.get("final_plan"))
        rows.append({
            "run_id": rr["run_id"], "condition": rr["condition"],
            "instance_id": rr["instance_id"], "cap": rr["cap"], "model": rr["model"],
            "satisfaction": score["satisfaction"],
            "nonempty_validated": bool(score.get("valid")) and meetings >= 1,
            "n_meetings": meetings,
            "tokens_total": toks["total"],
            "cap_utilisation": round(toks["total"] / rr["cap"], 4),
            "over_baseline_cap": toks["total"] > BASELINE_CAP,
            "n_calls": len(rr.get("calls") or []),
            "termination": agent.get("termination"),
            "proposal_attempts": len(props),
            "proposal_valid": sum(1 for p in props if p["valid"]),
            "binning_hash": md.get("binning_hash"),
            "git_commit": md.get("git_commit"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Cells and contrasts.
# --------------------------------------------------------------------------- #
def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    k = len(rows)
    if not k:
        return {"n": 0}
    attempts = sum(r["proposal_attempts"] for r in rows)
    return {
        "n": k,
        "mean_satisfaction": round(sum(r["satisfaction"] for r in rows) / k, 4),
        "nonempty_validated": sum(1 for r in rows if r["nonempty_validated"]),
        "nonempty_validated_rate": round(
            sum(1 for r in rows if r["nonempty_validated"]) / k, 4),
        "mean_tokens": round(sum(r["tokens_total"] for r in rows) / k),
        "mean_cap_utilisation": round(sum(r["cap_utilisation"] for r in rows) / k, 4),
        "mean_calls": round(sum(r["n_calls"] for r in rows) / k, 1),
        "runs_over_32000_tokens": sum(1 for r in rows if r["over_baseline_cap"]),
        "runs_at_or_under_32000_tokens": sum(1 for r in rows if not r["over_baseline_cap"]),
        "termination": dict(sorted(Counter(r["termination"] for r in rows).items())),
        "proposal_attempts": attempts,
        "proposal_validity": (round(sum(r["proposal_valid"] for r in rows) / attempts, 4)
                              if attempts else None),
    }


def architecture_contrast(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C1 minus C3 on the identical instance at one cap. Counts and means only."""
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_instance[r["instance_id"]][r["condition"]] = r
    deltas, c1, c3, tied = [], 0, 0, 0
    for pair in by_instance.values():
        if len(pair) != len(CONDITIONS):
            continue
        d = pair["c1_react"]["satisfaction"] - pair["c3_mas"]["satisfaction"]
        deltas.append(d)
        c1, c3, tied = (c1 + (d > 0), c3 + (d < 0), tied + (d == 0))
    return {
        "n_pairs": len(deltas),
        "mean_delta_satisfaction": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "instances_c1_ahead": c1, "instances_c3_ahead": c3, "instances_tied": tied,
        "sign_convention": ("delta = satisfaction(C1) - satisfaction(C3); negative means "
                            "the hierarchy is ahead"),
    }


def budget_change(base: Sequence[dict[str, Any]], probe: Sequence[dict[str, Any]]
                  ) -> dict[str, Any]:
    """32 000 -> 64 000 for one condition, paired within instance.

    TWO DIFFERENT QUESTIONS, AND THE WEAKER TEST IS THE RIGHT ONE.

    `runs_exceeding_baseline_cap` counts runs that could not have finished under the lower
    cap. That is sufficient evidence of a constraint but **not necessary**, and using it
    alone undercounts badly for C3. C1 spends one undivided working budget, so its execution
    changes only when it would have run out; C3 subdivides the cap and its worker quota is
    `3/8 * (cap - reserve)` -- 11 904 at 32 000 against 23 904 at 64 000 -- so a C3 worker
    behaves differently long before any total approaches 32 000.

    `runs_with_changed_execution` is therefore the primary count: a differing token total is
    direct evidence that the cap altered the run. Its complement, `identical_token_totals`,
    is the set that provably reproduced and carries no information about the extra budget.
    """
    b = {r["instance_id"]: r for r in base}
    p = {r["instance_id"]: r for r in probe}
    shared = sorted(set(b) & set(p))
    improved = worsened = unchanged = gained = lost = informative = 0
    deltas, tok_deltas = [], []
    identical_tokens = 0
    for i in shared:
        d = p[i]["satisfaction"] - b[i]["satisfaction"]
        deltas.append(d)
        tok_deltas.append(p[i]["tokens_total"] - b[i]["tokens_total"])
        improved += d > 0
        worsened += d < 0
        unchanged += d == 0
        gained += p[i]["nonempty_validated"] and not b[i]["nonempty_validated"]
        lost += b[i]["nonempty_validated"] and not p[i]["nonempty_validated"]
        informative += p[i]["over_baseline_cap"]
        identical_tokens += p[i]["tokens_total"] == b[i]["tokens_total"]
    return {
        "n_pairs": len(shared),
        "mean_delta_satisfaction": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "improved": improved, "worsened": worsened, "unchanged": unchanged,
        "gained_a_validated_plan": gained, "lost_a_validated_plan": lost,
        "mean_delta_tokens": round(sum(tok_deltas) / len(tok_deltas)) if tok_deltas else None,
        "runs_with_changed_execution": len(shared) - identical_tokens,
        "identical_token_totals": identical_tokens,
        "runs_exceeding_baseline_cap": informative,
        "definition": (
            f"delta = satisfaction({PROBE_CAP}) - satisfaction({BASELINE_CAP}). "
            "`runs_with_changed_execution` is the primary count: a differing token total is "
            "direct evidence the cap altered the run. `runs_exceeding_baseline_cap` is "
            "sufficient but not necessary evidence of a constraint, and undercounts for a "
            "condition that subdivides its cap"),
    }


def registered_reading(c1_probe: float, c3_probe: float) -> dict[str, str]:
    key = "c1_at_or_above_c3" if c1_probe >= c3_probe else "c1_below_c3"
    return {"case": key, "licenses": REGISTERED_READINGS[key],
            "source": "pre-registration of 2026-08-17, THESIS_DECISIONS section 3"}


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def build(base: Sequence[dict[str, Any]], probe: Sequence[dict[str, Any]],
          instances: dict[str, Any], audit_report: dict[str, Any]) -> dict[str, Any]:
    cells = {f"{c}@{cap}": summarise([r for r in rows if r["condition"] == c])
             for cap, rows in ((BASELINE_CAP, base), (PROBE_CAP, probe))
             for c in CONDITIONS}
    c1_probe = cells[f"c1_react@{PROBE_CAP}"]["mean_satisfaction"]
    c3_probe = cells[f"c3_mas@{PROBE_CAP}"]["mean_satisfaction"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "standing": ("exploratory development follow-up; descriptive only. No hypothesis "
                     "test, no threshold, no gate, no cap selection."),
        "n_people": 4,
        "fixed_optimum": FIXED_OPTIMUM,
        "caps": [BASELINE_CAP, PROBE_CAP],
        "context_window": WINDOW,
        "n_instances": len(instances),
        "completeness_audit": {k: audit_report[k] for k in
                               ("n_expected", "n_verified", "n_missing", "n_incompatible",
                                "n_unexpected_files", "all_checks_passed")},
        "provenance": {
            "models": sorted({r["model"] for r in list(base) + list(probe)}),
            "binning_hashes": sorted({(r["binning_hash"] or "")[:12]
                                      for r in list(base) + list(probe)}),
            "git_commits": sorted({(r["git_commit"] or "")[:12]
                                   for r in list(base) + list(probe)}),
        },
        "cells": cells,
        "architecture_contrast": {
            str(BASELINE_CAP): architecture_contrast(base),
            str(PROBE_CAP): architecture_contrast(probe),
        },
        "budget_change": {
            c: budget_change([r for r in base if r["condition"] == c],
                             [r for r in probe if r["condition"] == c])
            for c in CONDITIONS
        },
        "registered_reading": registered_reading(c1_probe, c3_probe),
        "selects_a_cap": False,
        "introduces_a_threshold": False,
    }


def render(s: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    b, p = str(BASELINE_CAP), str(PROBE_CAP)
    add(f"# `n = 4` budget follow-up — {BASELINE_CAP} against {PROBE_CAP}")
    add("")
    add(f"**{s['standing']}** The same frozen twelve instances at both caps, all at oracle "
        f"optimum {s['fixed_optimum']}. Twelve paired instances: description and ordering, "
        "not inference.")
    add("")
    add(f"Completeness audit of the {PROBE_CAP} arm: "
        f"{s['completeness_audit']['n_verified']}/{s['completeness_audit']['n_expected']} "
        f"verified, passed = {s['completeness_audit']['all_checks_passed']}.")
    add("")

    add("## The four cells")
    add("")
    add(f"| metric | C1 @{BASELINE_CAP} | C3 @{BASELINE_CAP} | C1 @{PROBE_CAP} | "
        f"C3 @{PROBE_CAP} |")
    add("|---|---|---|---|---|")
    order = [f"c1_react@{b}", f"c3_mas@{b}", f"c1_react@{p}", f"c3_mas@{p}"]
    for label, key, fmt in (
        ("mean satisfaction", "mean_satisfaction", "{:.3f}"),
        ("validated non-empty", "nonempty_validated", "{}/12"),
        ("mean tokens", "mean_tokens", "{}"),
        ("mean calls", "mean_calls", "{}"),
        ("cap used", "mean_cap_utilisation", "{:.2f}"),
        ("proposal validity", "proposal_validity", "{}"),
    ):
        add(f"| {label} | " + " | ".join(fmt.format(s["cells"][k][key]) for k in order) + " |")
    add("| termination | " + " | ".join(str(s["cells"][k]["termination"]) for k in order)
        + " |")
    add("")

    add("## Architecture contrast at each cap")
    add("")
    add("Sign: `delta = C1 - C3`, so a **negative** delta means the hierarchy is ahead.")
    add("")
    add("| cap | Δ (C1−C3) | C1 ahead | C3 ahead | tied | pairs |")
    add("|---|---|---|---|---|---|")
    for cap in (b, p):
        a = s["architecture_contrast"][cap]
        add(f"| {cap} | {a['mean_delta_satisfaction']:+.3f} | {a['instances_c1_ahead']} | "
            f"{a['instances_c3_ahead']} | {a['instances_tied']} | {a['n_pairs']} |")
    add("")

    add(f"## Was the extra budget used, and by how many runs?")
    add("")
    add(f"The {WINDOW}-token window clamps every call at both caps, so a larger cap buys "
        f"**more calls, not longer ones**. A run that reproduces its token total exactly "
        f"ran the same trajectory at both caps and is **not informative** about the extra "
        "budget; a run whose total changed was altered by the cap and is.")
    add("")
    add("| condition | execution changed | reproduced exactly | exceeded 32 000 | "
        "mean Δ tokens |")
    add("|---|---|---|---|---|")
    for c in CONDITIONS:
        g = s["budget_change"][c]
        add(f"| {c} | {g['runs_with_changed_execution']}/{g['n_pairs']} | "
            f"{g['identical_token_totals']} | {g['runs_exceeding_baseline_cap']} | "
            f"{g['mean_delta_tokens']:+} |")
    add("")
    add("*Exceeding 32 000 is sufficient evidence of a constraint but not necessary.* C1 "
        "spends one undivided working budget and diverges only when it would have run out. "
        "C3 subdivides the cap: its worker quota is `3/8 * (cap - 256)`, 11 904 against "
        "23 904, so a worker behaves differently long before any total approaches 32 000. "
        "The changed-execution column is the one to read.")
    add("")

    add(f"## Paired change from {BASELINE_CAP} to {PROBE_CAP}, per condition")
    add("")
    add("| condition | improved | worsened | unchanged | mean Δ satisfaction | "
        "gained a validated plan | lost one | execution changed |")
    add("|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        g = s["budget_change"][c]
        add(f"| {c} | {g['improved']} | {g['worsened']} | {g['unchanged']} | "
            f"{g['mean_delta_satisfaction']:+.3f} | {g['gained_a_validated_plan']} | "
            f"{g['lost_a_validated_plan']} | "
            f"{g['runs_with_changed_execution']}/{g['n_pairs']} |")
    add("")
    add("Counts and means only; no hypothesis test is performed. Read every mean against "
        "the changed-execution count in the last column: the runs that reproduced exactly "
        "contribute a zero to the mean without being evidence either way.")
    add("")

    r = s["registered_reading"]
    add("## The reading, fixed before these runs existed")
    add("")
    add(f"Case obtained: **`{r['case']}`**. Registered consequence, quoted from the "
        f"{r['source']}:")
    add("")
    add(f"> {r['licenses']}")
    add("")
    add("---")
    add("")
    add("*No threshold is introduced, no cap is selected and no gate is applied. Twelve "
        "development instances support description and ordering only.*")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-32k", type=Path, required=True)
    ap.add_argument("--runs-64k", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--expected", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/lowcx_n4_64k_probe"))
    args = ap.parse_args(argv)

    audit_report = audit(load_expected(args.expected), args.runs_64k)
    if not audit_report["all_checks_passed"]:
        raise AnalysisRefused(
            f"completeness audit failed: {audit_report['n_missing']} missing, "
            f"{audit_report['n_incompatible']} incompatible, "
            f"{audit_report['n_unexpected_files']} unexpected. Analysis may not begin.")

    instances = load_instances(args.subset)
    base = load_records(args.runs_32k, instances, BASELINE_CAP)
    probe = load_records(args.runs_64k, instances, PROBE_CAP)
    want = len(instances) * len(CONDITIONS)
    for label, rows in ((f"{BASELINE_CAP}", base), (f"{PROBE_CAP}", probe)):
        if len(rows) != want:
            raise AnalysisRefused(f"cap {label}: {len(rows)} records, expected {want}")

    summary = build(base, probe, instances, audit_report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    print(f"audit passed: {len(probe)} runs at {PROBE_CAP}, "
          f"{len(base)} at {BASELINE_CAP}, {len(instances)} instances\n")
    print(f"{'cap':>7} {'C1':>8} {'C3':>8} {'delta':>9}   (negative = hierarchy ahead)")
    for cap in (BASELINE_CAP, PROBE_CAP):
        c1 = summary["cells"][f"c1_react@{cap}"]["mean_satisfaction"]
        c3 = summary["cells"][f"c3_mas@{cap}"]["mean_satisfaction"]
        d = summary["architecture_contrast"][str(cap)]["mean_delta_satisfaction"]
        print(f"{cap:>7} {c1:>8.3f} {c3:>8.3f} {d:>+9.3f}")
    for c in CONDITIONS:
        g = summary["budget_change"][c]
        print(f"  {c:<12} execution changed: "
              f"{g['runs_with_changed_execution']}/{g['n_pairs']}  "
              f"(over {BASELINE_CAP}: {g['runs_exceeding_baseline_cap']})")
    print(f"\nregistered case: {summary['registered_reading']['case']}")
    print("exploratory and descriptive: no threshold, no cap selection, no gate")
    print(f"written: {args.out}/summary.json, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
