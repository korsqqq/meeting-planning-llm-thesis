# scripts/analyse_budget_probe.py
"""Paired 64 000 -> 128 000 budget probe. Description, not inference.

    python -m scripts.analyse_budget_probe \
        --runs results/logs/budget_dev \
        --runs results/logs/budget_dev_arch_diag \
        --runs results/logs/budget_probe_128k \
        --conditions c1_react,c3_mas,c5_best_of_3 \
        --out results/analysis/budget_probe_128k

Registered in THESIS_DECISIONS section 5, amendments of 2026-08-14 and 2026-08-15, before
any 128 000 outcome existed. **Exploratory, descriptive, and not a gate.** It selects no
cap, introduces no threshold and performs no hypothesis test: twelve instances support
description, not inference. Whether 128 000 joins the held-out design is a separate later
decision that this script does not make.

The question: does doubling the equal total workflow budget materially change how the
conditions behave? Every comparison is **paired within instance** — the same twelve frozen
task objects at both caps — so the 64 000 arm is a measured baseline rather than a chosen
one.

*Primary metric: satisfaction.* The validated non-empty rate is reported alongside as a
secondary, diagnostic quantity and is **never** called performance or accuracy.

*The quantity that can answer the question on its own.* If runs at 128 000 do not actually
spend more than 64 000, the extra budget is unused and nothing else needs interpreting. Both
directions are counted: runs that exceeded 64 000, and runs that finished under it with
128 000 available.

Fail-loud: a probe arm that is short, duplicated, drawn from another subset or run at the
wrong cap would still produce readable numbers, so each of those raises instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "budget_probe/1.0"

# The frozen probe selection. Pinned so another subset cannot be analysed by accident.
PROBE_SUBSET = ("results/manifests/subset__budget_probe_128k__76f3bc0f0654.json")
EXPECTED_BINNING_HASH = (
    "8e671cd13dbb262b638048eff58241226f09d5804373534789915bc302ba0dff")

BASELINE_CAP = 64_000
PROBE_CAP = 128_000
N_PER_BAND = 4
BANDS = ("low", "medium", "high")
BAND_OF_LEVEL = {"easy": "low", "medium": "medium", "hard": "high"}
DEFAULT_CONDITIONS = ("c1_react", "c3_mas", "c5_best_of_3")


class ProbeAuditError(RuntimeError):
    """The run set is not the one the probe was registered against."""


def _plan_size(plan: Any) -> int:
    if not isinstance(plan, dict):
        return 0
    return max((len(v) for v in plan.values() if isinstance(v, list)), default=0)


def load_probe_instances(path: Path) -> dict[str, str]:
    """`instance_id -> band`, straight from the frozen probe subset."""
    subset = json.loads(path.read_text(encoding="utf-8"))
    return {e["instance_id"]: BAND_OF_LEVEL[lv]
            for lv, rows in subset["instances"].items() for e in rows}


def load_runs(dirs: Sequence[Path], instances: dict[str, str],
              conditions: Sequence[str]) -> list[dict[str, Any]]:
    """Every result for the probe instances at either cap, from any of the directories."""
    rows: list[dict[str, Any]] = []
    for run_dir in dirs:
        for path in sorted(run_dir.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            rr, agent = doc["run_result"], doc["agent"]
            if rr["instance_id"] not in instances:
                continue
            if rr["condition"] not in conditions or rr["cap"] not in (BASELINE_CAP,
                                                                     PROBE_CAP):
                continue
            score, toks = rr["score"], rr["tokens"]
            md = doc.get("pilot_sweep") or {}
            calls = rr.get("calls") or []
            props = agent.get("proposals") or []
            meetings = _plan_size(rr.get("final_plan"))
            rows.append({
                "run_id": rr["run_id"],
                "condition": rr["condition"],
                "instance_id": rr["instance_id"],
                "band": instances[rr["instance_id"]],
                "cap": rr["cap"],
                "model": rr["model"],
                "seed": rr["seed"],
                "binning_hash": md.get("binning_hash"),
                "git_commit": md.get("git_commit"),
                "satisfaction": score["satisfaction"],
                "nonempty_validated": bool(score.get("valid")) and meetings >= 1,
                "tokens_total": toks["total"],
                "cap_utilisation": round(toks["total"] / rr["cap"], 4),
                "over_baseline_cap": toks["total"] > BASELINE_CAP,
                "n_calls": len(calls),
                "termination": agent.get("termination"),
                "proposal_attempts": len(props),
                "proposal_valid": sum(1 for p in props if p["valid"]),
                "source_dir": run_dir.name,
            })
    return rows


def audit(rows: Sequence[dict[str, Any]], instances: dict[str, str],
          conditions: Sequence[str]) -> dict[str, Any]:
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    n = len(instances)
    check(n == N_PER_BAND * len(BANDS), f"probe subset holds {n} instances, expected 12")
    check(len(rows) == n * len(conditions) * 2,
          f"expected {n * len(conditions) * 2} runs for {list(conditions)} at both caps, "
          f"found {len(rows)}")

    ids = {r["run_id"] for r in rows}
    check(len(ids) == len(rows), f"{len(rows) - len(ids)} duplicate run_id(s)")

    for cond in conditions:
        for cap in (BASELINE_CAP, PROBE_CAP):
            cell = [r for r in rows if r["condition"] == cond and r["cap"] == cap]
            check(len(cell) == n, f"{cond} at {cap}: {len(cell)} runs, expected {n}")
            covered = {r["instance_id"] for r in cell}
            missing = sorted(set(instances) - covered)
            check(not missing, f"{cond} at {cap}: missing {len(missing)} instance(s): "
                               f"{missing[:4]}")

    binnings = {r["binning_hash"] for r in rows}
    check(binnings == {EXPECTED_BINNING_HASH},
          f"runs carry binning hash(es) {sorted(h[:12] for h in binnings if h)}, "
          f"expected {EXPECTED_BINNING_HASH[:12]}")
    models = {r["model"] for r in rows}
    check(len(models) == 1, f"more than one model: {sorted(models)}")
    seeds = {r["seed"] for r in rows}
    stray = sorted(s for s in seeds if not 30000 <= s <= 39999)
    check(not stray, f"{len(stray)} run(s) outside the development seed range: {stray[:8]}")

    if problems:
        raise ProbeAuditError("budget probe failed its completeness audit:\n  - "
                              + "\n  - ".join(problems))

    per_commit = {c: sorted({r["git_commit"] for r in rows
                             if r["condition"] == c and r["cap"] == PROBE_CAP
                             and r["git_commit"]})
                  for c in conditions}
    return {
        "n_instances": n,
        "n_runs": len(rows),
        "conditions": list(conditions),
        "caps": [BASELINE_CAP, PROBE_CAP],
        "model": sorted(models)[0],
        "instances_per_band": {b: sum(1 for v in instances.values() if v == b)
                               for b in BANDS},
        "probe_subset": PROBE_SUBSET,
        "binning_hash": EXPECTED_BINNING_HASH,
        "paired_on": "identical instance_id at both caps",
        "probe_git_commit_per_condition": per_commit,
        "all_checks_passed": True,
    }


def _block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    k = len(rows)
    attempts = sum(r["proposal_attempts"] for r in rows)
    return {
        "n": k,
        "mean_satisfaction": round(sum(r["satisfaction"] for r in rows) / k, 4),
        "nonempty_validated": sum(1 for r in rows if r["nonempty_validated"]),
        "nonempty_validated_rate": round(
            sum(1 for r in rows if r["nonempty_validated"]) / k, 4),
        "mean_tokens": round(sum(r["tokens_total"] for r in rows) / k, 1),
        "mean_cap_utilisation": round(sum(r["cap_utilisation"] for r in rows) / k, 4),
        "mean_calls": round(sum(r["n_calls"] for r in rows) / k, 2),
        "termination": dict(sorted(Counter(r["termination"] for r in rows).items())),
        "proposal_attempts": attempts,
        "proposal_validity_rate": (round(sum(r["proposal_valid"] for r in rows) / attempts, 4)
                                   if attempts else None),
        # Both directions of the question the probe can answer on its own.
        "runs_over_64000_tokens": sum(1 for r in rows if r["over_baseline_cap"]),
        "runs_under_64000_tokens": sum(1 for r in rows if not r["over_baseline_cap"]),
    }


def _paired(base: Sequence[dict], probe: Sequence[dict]) -> dict[str, Any]:
    """Instance-paired change from 64 000 to 128 000. Counts and means, no test."""
    by_id = {r["instance_id"]: r for r in base}
    deltas, improved, worsened, unchanged = [], 0, 0, 0
    gained, lost = 0, 0
    for r in probe:
        b = by_id.get(r["instance_id"])
        if b is None:
            continue
        d = r["satisfaction"] - b["satisfaction"]
        deltas.append(d)
        if d > 0:
            improved += 1
        elif d < 0:
            worsened += 1
        else:
            unchanged += 1
        if r["nonempty_validated"] and not b["nonempty_validated"]:
            gained += 1
        elif b["nonempty_validated"] and not r["nonempty_validated"]:
            lost += 1
    return {
        "n_paired": len(deltas),
        "mean_delta_satisfaction": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "improved": improved, "worsened": worsened, "unchanged": unchanged,
        "gained_a_validated_plan": gained, "lost_a_validated_plan": lost,
        "definition": "delta = satisfaction(128000) - satisfaction(64000) on the "
                      "identical instance",
        "note": "counts and means only; no hypothesis test is performed",
    }


def analyse(rows: Sequence[dict[str, Any]],
            conditions: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_condition": {}}
    for cond in conditions:
        entry: dict[str, Any] = {"by_cap": {}, "by_band": {}, "paired": {}}
        for cap in (BASELINE_CAP, PROBE_CAP):
            sel = [r for r in rows if r["condition"] == cond and r["cap"] == cap]
            entry["by_cap"][str(cap)] = _block(sel)
            for band in BANDS:
                entry["by_band"].setdefault(band, {})[str(cap)] = _block(
                    [r for r in sel if r["band"] == band])
        base = [r for r in rows if r["condition"] == cond and r["cap"] == BASELINE_CAP]
        probe = [r for r in rows if r["condition"] == cond and r["cap"] == PROBE_CAP]
        entry["paired"]["pooled"] = _paired(base, probe)
        for band in BANDS:
            entry["paired"][band] = _paired(
                [r for r in base if r["band"] == band],
                [r for r in probe if r["band"] == band])
        out["by_condition"][cond] = entry
    out["reading"] = {
        "status": "exploratory development probe; descriptive, not a hypothesis test",
        "primary_metric": "satisfaction",
        "secondary": "validated non-empty rate, diagnostic only, never called performance "
                     "or accuracy",
        "no_gate": "no threshold is introduced and no cap is selected; whether 128000 "
                   "enters the held-out design is a separate later decision",
        "n_caveat": f"{N_PER_BAND} instances per band supports description and ordering, "
                    "not inference",
    }
    return out


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    a, au = doc["analysis"], doc["audit"]
    conds = au["conditions"]

    add("# Budget probe — 64 000 vs 128 000, paired within instance\n")
    add("**Exploratory development probe. Descriptive, not a hypothesis test, not a gate.** "
        "No threshold is introduced and no cap is selected here.\n")
    add(f"{au['n_runs']} runs over {au['n_instances']} frozen instances "
        f"({N_PER_BAND} per band), model `{au['model']}`, paired on the identical instance "
        "at both caps. Completeness audit passed.\n")

    add("\n## Satisfaction — the primary metric\n")
    header = "| band |" + "".join(
        f" {c} 64k | {c} 128k | Δ {c} |" for c in conds)
    add(header)
    add("|---" * (1 + 3 * len(conds)) + "|")
    for band in (*BANDS, "pooled"):
        cells = []
        for c in conds:
            e = a["by_condition"][c]
            if band == "pooled":
                b64 = e["by_cap"][str(BASELINE_CAP)]
                b128 = e["by_cap"][str(PROBE_CAP)]
                pr = e["paired"]["pooled"]
            else:
                b64 = e["by_band"][band][str(BASELINE_CAP)]
                b128 = e["by_band"][band][str(PROBE_CAP)]
                pr = e["paired"][band]
            cells.append(f" {b64['mean_satisfaction']:.3f} | {b128['mean_satisfaction']:.3f} "
                         f"| {pr['mean_delta_satisfaction']:+.3f} |")
        add(f"| {band} |" + "".join(cells))

    add("\n## Was the extra budget used at all?\n")
    add("| condition | mean tokens 64k | mean tokens 128k | cap used 128k | "
        "runs over 64 000 | runs under 64 000 |")
    add("|---|---|---|---|---|---|")
    for c in conds:
        b64 = a["by_condition"][c]["by_cap"][str(BASELINE_CAP)]
        b128 = a["by_condition"][c]["by_cap"][str(PROBE_CAP)]
        add(f"| {c} | {b64['mean_tokens']:.0f} | {b128['mean_tokens']:.0f} | "
            f"{b128['mean_cap_utilisation']:.2f} | "
            f"{b128['runs_over_64000_tokens']}/{b128['n']} | "
            f"{b128['runs_under_64000_tokens']}/{b128['n']} |")
    add("\nIf runs at 128 000 do not spend more than 64 000, the extra budget is unused and "
        "nothing else needs interpreting.\n")

    add("\n## Paired change, pooled\n")
    add("| condition | improved | worsened | unchanged | mean Δ satisfaction | "
        "gained a validated plan | lost one |")
    add("|---|---|---|---|---|---|---|")
    for c in conds:
        pr = a["by_condition"][c]["paired"]["pooled"]
        add(f"| {c} | {pr['improved']} | {pr['worsened']} | {pr['unchanged']} | "
            f"{pr['mean_delta_satisfaction']:+.3f} | {pr['gained_a_validated_plan']} | "
            f"{pr['lost_a_validated_plan']} |")
    add("\nCounts and means only; no hypothesis test is performed.\n")

    add("\n## Validated non-empty rate — secondary, diagnostic\n")
    add("A structural precondition for scoring at all, reported beside the primary metric "
        "and **never called performance or accuracy**.\n")
    add("| condition | 64k | 128k |")
    add("|---|---|---|")
    for c in conds:
        b64 = a["by_condition"][c]["by_cap"][str(BASELINE_CAP)]
        b128 = a["by_condition"][c]["by_cap"][str(PROBE_CAP)]
        add(f"| {c} | {b64['nonempty_validated']}/{b64['n']} = "
            f"{b64['nonempty_validated_rate']:.2f} | {b128['nonempty_validated']}/"
            f"{b128['n']} = {b128['nonempty_validated_rate']:.2f} |")

    add("\n## Cost and behaviour\n")
    add("| condition | cap | calls | proposal validity | termination |")
    add("|---|---|---|---|---|")
    for c in conds:
        for cap in (BASELINE_CAP, PROBE_CAP):
            b = a["by_condition"][c]["by_cap"][str(cap)]
            pv = b["proposal_validity_rate"]
            add(f"| {c} | {cap} | {b['mean_calls']:.1f} | "
                + (f"{pv:.3f}" if pv is not None else "—")
                + f" | {b['termination']} |")

    add(f"\n---\n\n*{a['reading']['n_caveat']}. {a['reading']['no_gate'].capitalize()}.*\n")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, action="append", required=True,
                    help="a directory of result documents; repeatable")
    ap.add_argument("--subset", type=Path, default=Path(PROBE_SUBSET))
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/budget_probe_128k"))
    args = ap.parse_args(argv)

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    instances = load_probe_instances(args.subset)
    rows = load_runs(args.runs, instances, conditions)
    audit_report = audit(rows, instances, conditions)
    analysis = analyse(rows, conditions)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory development probe; descriptive, not confirmatory",
        "selects_a_cap": False,
        "registered": "THESIS_DECISIONS.md section 5, amendments of 2026-08-14 and "
                      "2026-08-15",
        "audit": audit_report,
        "analysis": analysis,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    write_report(doc, args.out / "report.md")

    print(f"audit passed: {audit_report['n_runs']} runs over "
          f"{audit_report['n_instances']} instances, paired at both caps")
    print()
    print(f"{'condition':<18}{'sat 64k':>10}{'sat 128k':>10}{'delta':>9}"
          f"{'tok 64k':>10}{'tok 128k':>10}{'>64k':>7}")
    for c in conditions:
        e = analysis["by_condition"][c]
        b64, b128 = e["by_cap"][str(BASELINE_CAP)], e["by_cap"][str(PROBE_CAP)]
        pr = e["paired"]["pooled"]
        print(f"{c:<18}{b64['mean_satisfaction']:>10.3f}{b128['mean_satisfaction']:>10.3f}"
              f"{pr['mean_delta_satisfaction']:>+9.3f}{b64['mean_tokens']:>10.0f}"
              f"{b128['mean_tokens']:>10.0f}"
              f"{b128['runs_over_64000_tokens']:>4}/{b128['n']:<2}")
    print()
    print("exploratory and descriptive: no cap is selected and no threshold is introduced")
    print(f"written: {args.out}/summary.json, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
