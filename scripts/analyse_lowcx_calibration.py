# scripts/analyse_lowcx_calibration.py
"""C1 vs C3 on the frozen lower-complexity pool. CPU only, descriptive.

    python -m scripts.analyse_lowcx_calibration \
        --runs results/logs/lowcx_calib \
        --subset results/manifests/subset__lowcx_n4__74e57136280d.json \
        --subset results/manifests/subset__lowcx_n5__6bd65b0db5bf.json \
        --subset results/manifests/subset__lowcx_n6__f3efd0cd0b73.json \
        --expected results/manifests/expected_runs__lowcx_n4.json \
        --expected results/manifests/expected_runs__lowcx_n5.json \
        --expected results/manifests/expected_runs__lowcx_n6.json \
        --out results/analysis/lowcx_calib

**Exploratory and descriptive.** Twelve instances per size support description and ordering,
not inference. No hypothesis test is performed, **no threshold is introduced, no budget cap
is selected, and no gate is applied**. Whether any of this enters the held-out design is a
separate later decision.

WHAT IS COMPARED. C1 and C3 on the identical instance at the identical cap, paired within
instance. The sign follows the exposé: `delta = satisfaction(C1) - satisfaction(C3)`, so a
**negative** delta means the hierarchy is ahead.

WHAT THE OPTIMUM BUYS HERE. Every instance in this pool has oracle optimum 3, so
`satisfaction = achieved / 3` throughout and the metric takes the same four values
{0, 1/3, 2/3, 1} at every size. A difference between sizes therefore cannot come from the
denominator, which is the whole reason the pool was matched that way.

WHY `n` IS NOT A COMPLEXITY AXIS, RESTATED WHERE THE NUMBERS ARE READ. `n` is a generator
parameter. Smaller `n` is used because `H = alpha_reachable - O <= n - O` bounds the
higher-order interaction the pairwise density metric cannot see. `n` and the attainable `H`
are tied by that inequality and their separate effects are **not identified** here, so no
table below may be read as "n = 4 is the easy level".

THE ALTERNATIVE THIS REPORT MUST LET THE READER SEE. C3 divides its working budget: each
worker receives `floor(3/8 * (cap - reserve))`, which is 2904 tokens at cap 8000. That is
small enough that a poor C3 result at the bottom cap could be budget starvation rather than
anything about architecture or complexity. Cap utilisation, termination reasons and call
counts are therefore reported beside satisfaction in every cell, because satisfaction alone
cannot separate the two.
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

SCHEMA_VERSION = "lowcx_calibration/1.0"
CONDITIONS = ("c1_react", "c3_mas")
CAPS = (8000, 16000, 32000)
FIXED_OPTIMUM = 3

# C3's split, from src/agents/multi_agent. Recomputed here only to report it beside the
# results; nothing is read from the agent code at analysis time.
FINALISATION_RESERVE = 256
WORKER_SHARE_NUM, WORKER_SHARE_DEN = 3, 8


class AnalysisRefused(RuntimeError):
    """The inputs do not support an analysis. Never worked around."""


def worker_quota(cap: int) -> int:
    return max(0, cap - FINALISATION_RESERVE) * WORKER_SHARE_NUM // WORKER_SHARE_DEN


def _plan_size(plan: Any) -> int:
    if isinstance(plan, dict):
        return len(plan.get("meetings") or [])
    return 0


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #
def load_instances(subsets: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """instance_id -> its frozen structural facts. Also fixes which 36 may be analysed."""
    out: dict[str, dict[str, Any]] = {}
    for p in subsets:
        doc = json.loads(p.read_text(encoding="utf-8"))
        n = doc["n_people"]
        for rows in doc["instances"].values():
            for e in rows:
                if e["optimum"] != FIXED_OPTIMUM:
                    raise AnalysisRefused(
                        f"{e['instance_id']}: optimum {e['optimum']}, but this pool is "
                        f"matched at {FIXED_OPTIMUM} and the report assumes it")
                out[e["instance_id"]] = {
                    "n_people": n, "seed": e["seed"], "level": e["level"],
                    "conflict_pairs": e["complexity_metric"], "optimum": e["optimum"],
                    "tightness": e["cell"]["tightness"],
                    "travel_structure": e["cell"]["travel_structure"],
                }
    return out


def load_records(runs_dir: Path, instances: dict[str, dict[str, Any]]
                 ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        if rr["instance_id"] not in instances or rr["condition"] not in CONDITIONS:
            continue
        score, toks = rr["score"], rr["tokens"]
        md = doc.get("pilot_sweep") or {}
        props = agent.get("proposals") or []
        meetings = _plan_size(rr.get("final_plan"))
        info = instances[rr["instance_id"]]
        rows.append({
            "run_id": rr["run_id"],
            "condition": rr["condition"],
            "instance_id": rr["instance_id"],
            "n_people": info["n_people"],
            "conflict_pairs": info["conflict_pairs"],
            "level": info["level"],
            "cap": rr["cap"],
            "model": rr["model"],
            "satisfaction": score["satisfaction"],
            "nonempty_validated": bool(score.get("valid")) and meetings >= 1,
            "n_meetings": meetings,
            "tokens_total": toks["total"],
            "cap_utilisation": round(toks["total"] / rr["cap"], 4),
            "n_calls": len(rr.get("calls") or []),
            "termination": agent.get("termination"),
            "proposal_attempts": len(props),
            "proposal_valid": sum(1 for p in props if p["valid"]),
            "binning_hash": md.get("binning_hash"),
            "git_commit": md.get("git_commit"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Cells.
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
        "mean_meetings": round(sum(r["n_meetings"] for r in rows) / k, 2),
        "termination": dict(sorted(Counter(r["termination"] for r in rows).items())),
        "proposal_attempts": attempts,
        "proposal_validity": (round(sum(r["proposal_valid"] for r in rows) / attempts, 4)
                              if attempts else None),
    }


def paired(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """C1 minus C3 on the identical instance AT THE IDENTICAL CAP. Counts and means only.

    The key is `(instance_id, cap)`, not `instance_id`. Keying on the instance alone is
    correct only while the rows come from a single cap; pool over caps and each instance
    appears three times, the later record overwrites the earlier one, and the function
    silently reports whichever cap happened to be read last while still looking like a
    complete pairing.
    """
    by_instance: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_instance[(r["instance_id"], r["cap"])][r["condition"]] = r
    deltas, c1_ahead, c3_ahead, tied = [], 0, 0, 0
    tok_deltas = []
    for pair in by_instance.values():
        if len(pair) != len(CONDITIONS):
            continue
        d = pair["c1_react"]["satisfaction"] - pair["c3_mas"]["satisfaction"]
        deltas.append(d)
        tok_deltas.append(pair["c1_react"]["tokens_total"]
                          - pair["c3_mas"]["tokens_total"])
        if d > 0:
            c1_ahead += 1
        elif d < 0:
            c3_ahead += 1
        else:
            tied += 1
    return {
        "n_pairs": len(deltas),
        "mean_delta_satisfaction": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "instances_c1_ahead": c1_ahead,
        "instances_c3_ahead": c3_ahead,
        "instances_tied": tied,
        "mean_delta_tokens": round(sum(tok_deltas) / len(tok_deltas)) if tok_deltas else None,
        "sign_convention": ("delta = satisfaction(C1) - satisfaction(C3); negative means "
                            "the hierarchy is ahead"),
    }


def build(rows: Sequence[dict[str, Any]], instances: dict[str, dict[str, Any]],
          audit_report: dict[str, Any]) -> dict[str, Any]:
    sizes = sorted({i["n_people"] for i in instances.values()})
    cells: dict[str, Any] = {}
    for n in sizes:
        for cap in CAPS:
            for c in CONDITIONS:
                sel = [r for r in rows
                       if r["n_people"] == n and r["cap"] == cap and r["condition"] == c]
                cells[f"n{n}|{cap}|{c}"] = summarise(sel)

    pooled_by_cap = {
        str(cap): {c: summarise([r for r in rows if r["cap"] == cap
                                 and r["condition"] == c]) for c in CONDITIONS}
        for cap in CAPS
    }
    pooled_by_size = {
        str(n): {c: summarise([r for r in rows if r["n_people"] == n
                               and r["condition"] == c]) for c in CONDITIONS}
        for n in sizes
    }
    contrasts = {
        f"n{n}|{cap}": paired([r for r in rows if r["n_people"] == n and r["cap"] == cap])
        for n in sizes for cap in CAPS
    }
    contrasts_by_cap = {str(cap): paired([r for r in rows if r["cap"] == cap])
                        for cap in CAPS}
    contrasts_by_size = {str(n): paired([r for r in rows if r["n_people"] == n])
                         for n in sizes}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "standing": ("exploratory development calibration; descriptive only. No hypothesis "
                     "test, no threshold, no cap selection, no gate."),
        "primary_metric": "satisfaction = achieved / oracle optimum, optimum 3 throughout",
        "secondary_diagnostic": ("validated non-empty rate; never described as performance "
                                 "or accuracy"),
        "sign_convention": "delta = satisfaction(C1) - satisfaction(C3)",
        "n_runs": len(rows),
        "n_instances": len(instances),
        "sizes": sizes,
        "caps": list(CAPS),
        "c3_worker_quota_by_cap": {str(cap): worker_quota(cap) for cap in CAPS},
        "completeness_audit": {k: audit_report[k] for k in
                               ("n_expected", "n_verified", "n_missing", "n_incompatible",
                                "n_unexpected_files", "all_checks_passed")},
        "provenance": {
            "models": sorted({r["model"] for r in rows}),
            "binning_hashes": sorted({(r["binning_hash"] or "")[:12] for r in rows}),
            "git_commits": sorted({(r["git_commit"] or "")[:12] for r in rows}),
        },
        "cells": cells,
        "pooled_by_cap": pooled_by_cap,
        "pooled_by_size": pooled_by_size,
        "contrasts": contrasts,
        "contrasts_by_cap": contrasts_by_cap,
        "contrasts_by_size": contrasts_by_size,
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def render(s: dict[str, Any]) -> str:
    L: list[str] = []
    add = L.append
    sizes, caps = s["sizes"], s["caps"]

    add("# Lower-complexity calibration — C1 vs C3")
    add("")
    add(f"**{s['standing']}** Twelve instances per size at each cap: description and "
        "ordering, not inference.")
    add("")
    add(f"{s['n_runs']} runs over {s['n_instances']} frozen instances, "
        f"model {', '.join(s['provenance']['models'])}. Completeness audit: "
        f"{s['completeness_audit']['n_verified']}/{s['completeness_audit']['n_expected']} "
        f"verified, passed = {s['completeness_audit']['all_checks_passed']}.")
    add("")
    add("Every instance has oracle optimum 3, so satisfaction takes the same four values "
        "{0, 1/3, 2/3, 1} at every size and no difference below can come from the "
        "denominator.")
    add("")

    add("## Satisfaction — the primary metric")
    add("")
    add("Sign: `delta = C1 - C3`, so a **negative** delta means the hierarchy is ahead.")
    add("")
    head = "| n | cap | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |"
    add(head)
    add("|---|---|---|---|---|---|---|---|")
    for n in sizes:
        for cap in caps:
            c1 = s["cells"][f"n{n}|{cap}|c1_react"]
            c3 = s["cells"][f"n{n}|{cap}|c3_mas"]
            p = s["contrasts"][f"n{n}|{cap}"]
            add(f"| {n} | {cap} | {c1['mean_satisfaction']:.3f} | "
                f"{c3['mean_satisfaction']:.3f} | {p['mean_delta_satisfaction']:+.3f} | "
                f"{p['instances_c1_ahead']} | {p['instances_c3_ahead']} | "
                f"{p['instances_tied']} |")
    add("")

    add("### Pooled over sizes, by cap")
    add("")
    add("| cap | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |")
    add("|---|---|---|---|---|---|---|")
    for cap in caps:
        c1, c3 = s["pooled_by_cap"][str(cap)]["c1_react"], s["pooled_by_cap"][str(cap)]["c3_mas"]
        p = s["contrasts_by_cap"][str(cap)]
        add(f"| {cap} | {c1['mean_satisfaction']:.3f} | {c3['mean_satisfaction']:.3f} | "
            f"{p['mean_delta_satisfaction']:+.3f} | {p['instances_c1_ahead']} | "
            f"{p['instances_c3_ahead']} | {p['instances_tied']} |")
    add("")

    add("### Pooled over caps, by size")
    add("")
    add("*`n` is a generator parameter, not a complexity level. It is varied because "
        "`H <= n - O` bounds the higher-order interaction; `n` and the attainable `H` are "
        "tied by that inequality and their separate effects are not identified here.*")
    add("")
    add("| n | C1 | C3 | Δ (C1−C3) | C1 ahead | C3 ahead | tied |")
    add("|---|---|---|---|---|---|---|")
    for n in sizes:
        c1, c3 = s["pooled_by_size"][str(n)]["c1_react"], s["pooled_by_size"][str(n)]["c3_mas"]
        p = s["contrasts_by_size"][str(n)]
        add(f"| {n} | {c1['mean_satisfaction']:.3f} | {c3['mean_satisfaction']:.3f} | "
            f"{p['mean_delta_satisfaction']:+.3f} | {p['instances_c1_ahead']} | "
            f"{p['instances_c3_ahead']} | {p['instances_tied']} |")
    add("")

    add("## Budget — can a poor result be starvation rather than difficulty?")
    add("")
    add("C3 splits its working budget; each worker gets `floor(3/8 * (cap - "
        f"{FINALISATION_RESERVE}))`:")
    add("")
    add("| cap | C3 worker quota | two workers |")
    add("|---|---|---|")
    for cap in caps:
        add(f"| {cap} | {worker_quota(cap)} | {2 * worker_quota(cap)} |")
    add("")
    add("| n | cap | condition | mean tokens | cap used | calls | termination |")
    add("|---|---|---|---|---|---|---|")
    for n in sizes:
        for cap in caps:
            for c in CONDITIONS:
                b = s["cells"][f"n{n}|{cap}|{c}"]
                add(f"| {n} | {cap} | {c} | {b['mean_tokens']} | "
                    f"{b['mean_cap_utilisation']:.2f} | {b['mean_calls']} | "
                    f"{b['termination']} |")
    add("")

    add("## Validated non-empty rate — secondary, diagnostic")
    add("")
    add("A structural precondition for scoring at all. **Never** described as performance "
        "or accuracy.")
    add("")
    add("| n | cap | C1 | C3 |")
    add("|---|---|---|---|")
    for n in sizes:
        for cap in caps:
            c1 = s["cells"][f"n{n}|{cap}|c1_react"]
            c3 = s["cells"][f"n{n}|{cap}|c3_mas"]
            add(f"| {n} | {cap} | {c1['nonempty_validated']}/{c1['n']} | "
                f"{c3['nonempty_validated']}/{c3['n']} |")
    add("")

    add("---")
    add("")
    add("*No threshold is introduced, no cap is selected and no gate is applied. Whether "
        "any of this enters the held-out design is a separate dated decision.*")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--subset", action="append", type=Path, required=True)
    ap.add_argument("--expected", action="append", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/analysis/lowcx_calib"))
    args = ap.parse_args(argv)

    # The completeness audit is a precondition, not a report section: an analysis over a
    # partial sweep would still produce readable means.
    audit_report = audit(load_expected(args.expected), args.runs)
    if not audit_report["all_checks_passed"]:
        raise AnalysisRefused(
            f"completeness audit failed: {audit_report['n_missing']} missing, "
            f"{audit_report['n_incompatible']} incompatible, "
            f"{audit_report['n_unexpected_files']} unexpected. Analysis may not begin.")

    instances = load_instances(args.subset)
    rows = load_records(args.runs, instances)
    expected_runs = len(instances) * len(CAPS) * len(CONDITIONS)
    if len(rows) != expected_runs:
        raise AnalysisRefused(f"{len(rows)} records for {len(instances)} instances; "
                              f"expected {expected_runs}")

    summary = build(rows, instances, audit_report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = render(summary)
    (args.out / "report.md").write_text(report, encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    print(f"audit passed: {len(rows)} runs over {len(instances)} instances\n")
    print(f"{'cap':>7} {'C1':>8} {'C3':>8} {'delta':>9}   (negative = hierarchy ahead)")
    for cap in CAPS:
        c1 = summary["pooled_by_cap"][str(cap)]["c1_react"]["mean_satisfaction"]
        c3 = summary["pooled_by_cap"][str(cap)]["c3_mas"]["mean_satisfaction"]
        d = summary["contrasts_by_cap"][str(cap)]["mean_delta_satisfaction"]
        print(f"{cap:>7} {c1:>8.3f} {c3:>8.3f} {d:>+9.3f}")
    print("\nexploratory and descriptive: no threshold, no cap selection, no gate")
    print(f"written: {args.out}/summary.json, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
