# scripts/analyse_c1_calibration.py
"""Analysis of the C1 calibration sweep. CPU only -- reads JSON, runs nothing.

    python -m scripts.analyse_c1_calibration \
        --runs results/calibration/c1_20260805 \
        --formal-subset results/manifests/subset__pilot__29d836cb5614.json \
        --out results/analysis/c1_calibration

The runs analysed here come from a subset that is NOT held-out: one of its thirty
instances, `uniform-n4-t20-o20-s10000`, was executed and inspected during an earlier cap
calibration, before the manifest was frozen. They are therefore calibration evidence --
useful for choosing a budget ladder and for describing mechanisms -- and not
pre-registered evidence for comparing C1 against C2/C3. Every table is repeated with that
instance removed, so the reader can see for themselves that it carries nothing.

Rates follow the locked definitions (THESIS_DECISIONS section 5): the denominator is ALL
proposal attempts, malformed included. A cap where nothing was proposed has an empty
denominator, and the rates are reported as undefined rather than as zero -- reporting 0.0
would claim the agent proposed and failed.

Levels here are the OLD binning (medium <= 5). The formal manifest cuts its own
boundaries over its own pool (medium <= 6), so categorical levels are NOT comparable
across the two; nothing in this file mixes them.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPOSED_INSTANCE = "uniform-n4-t20-o20-s10000"
_LEVELS = ("easy", "medium", "hard")
EXPECTED_RUNS = 120
EXPECTED_INSTANCES = 30
EXPECTED_CAPS = {8000, 16000, 32000, 64000}
EXPECTED_CONDITION = "c1_react"
EXPECTED_MODEL = "Qwen/Qwen3-32B-AWQ"
EXPECTED_MAX_MODEL_LEN = 32768
# This analyser describes the CALIBRATION sweep specifically: it subtracts an exposed
# instance and labels its output accordingly. Pointed at the formal runs it would pass
# every uniformity check and then mislabel them, so the set is pinned by identity.
EXPECTED_SUBSET_HASH = (
    "d29b547529713e54dffbf3bc902f0007cbd3590d4a121b929fcea1b053a8bcd8"
)


class CalibrationIntegrityError(RuntimeError):
    """The run set is not the sweep it claims to be. Analysis must not proceed.

    Everything downstream is an average over these files; a set that is silently short,
    duplicated, or mixed across models or manifests would produce numbers that look
    perfectly reasonable and mean nothing. So this fails loudly rather than warning.
    """


# --------------------------------------------------------------------------- #
# Small statistics helpers (no scipy in the harness environment).
# --------------------------------------------------------------------------- #
def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation. None when it is not defined (n < 3 or no variation)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def _mean(v: Sequence[float]) -> float | None:
    return round(statistics.fmean(v), 4) if v else None


def _median(v: Sequence[float]) -> float | None:
    return round(statistics.median(v), 4) if v else None


def _rate(hits: int, total: int) -> float | None:
    """Undefined (None) on an empty denominator -- never silently 0.0."""
    return round(hits / total, 4) if total else None


# --------------------------------------------------------------------------- #
# Per-run extraction.
# --------------------------------------------------------------------------- #
def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(run_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        calls = rr.get("calls", [])
        props = agent.get("proposals", [])

        # Cumulative internal cost up to and including the call a proposal came from.
        # `step` is 1-based over the reasoning calls, which are the leading entries of
        # `calls`, so the cumulative cost through proposal k is the sum over calls[:k].
        cum = []
        running = 0
        for c in calls:
            running += c["input_tokens"] + c["thinking_tokens"] + c["answer_tokens"]
            cum.append(running)

        def cost_through(step: int) -> int | None:
            idx = step - 1
            return cum[idx] if 0 <= idx < len(cum) else None

        first_valid = next((i for i, p in enumerate(props) if p["valid"]), None)
        first_accept = next(
            (i for i, p in enumerate(props) if p["accepted_into_best_plan"]), None
        )
        instance = doc.get("instance", {})
        gp = instance.get("generator_params", {})
        md = doc.get("pilot_sweep", {})
        audit = doc.get("usage_audit", {}) or {}
        totals = audit.get("totals", {}) or {}

        runs.append({
            "instance_id": rr["instance_id"],
            "cap": rr["cap"],
            # Provenance and audit fields, used by the integrity check rather than by
            # any statistic; kept on the row so the check needs no second read.
            "run_id": rr["run_id"],
            "condition": rr["condition"],
            "model": rr["model"],
            "subset_hash": md.get("subset_hash"),
            "binning_hash": md.get("binning_hash"),
            "max_model_len": md.get("max_model_len"),
            "has_endpoint_usage": audit.get("has_endpoint_usage"),
            "internal_total_tokens": totals.get("internal_total_tokens"),
            "endpoint_total_tokens": totals.get("endpoint_total_tokens"),
            "think_leak": (doc.get("transcript_sanity") or {}).get("think_leak"),
            "level_old": rr["level"],
            "n_people": gp.get("n_people"),
            "travel_structure": gp.get("travel_structure"),
            "tightness": gp.get("tightness"),
            "overlap": gp.get("overlap"),
            "complexity_metric": instance.get("complexity_metric"),
            "optimum": rr["score"]["solver_optimum"],
            "achieved": rr["score"]["n_valid_meetings"],
            "abs_gap": rr["score"]["solver_optimum"] - rr["score"]["n_valid_meetings"],
            "satisfaction": rr["score"]["satisfaction"],
            "tokens_total": rr["tokens"]["total"],
            "termination": agent.get("termination"),
            "n_calls": len(calls),
            "attempts": len(props),
            "parsed": sum(p["parsed"] for p in props),
            "valid": sum(p["valid"] for p in props),
            "accepted": sum(p["accepted_into_best_plan"] for p in props),
            "any_valid": any(p["valid"] for p in props),
            "any_accepted": any(p["accepted_into_best_plan"] for p in props),
            "first_valid_index": first_valid,
            "first_accepted_index": first_accept,
            "tokens_to_first_valid": (
                cost_through(props[first_valid]["step"]) if first_valid is not None else None
            ),
            "tokens_to_first_accepted": (
                cost_through(props[first_accept]["step"]) if first_accept is not None else None
            ),
            "reasons": [r for p in props if not p["valid"] for r in p["reasons"]],
            "ctx_limited_calls": sum(bool(c.get("context_limited")) for c in calls),
            "length_calls": sum(c.get("finish_reason") == "length" for c in calls),
            "ctx_and_length": sum(
                bool(c.get("context_limited")) and c.get("finish_reason") == "length"
                for c in calls),
            "ctx_and_stop": sum(
                bool(c.get("context_limited")) and c.get("finish_reason") == "stop"
                for c in calls),
            "nolimit_and_length": sum(
                (not c.get("context_limited")) and c.get("finish_reason") == "length"
                for c in calls),
        })
    return runs


# --------------------------------------------------------------------------- #
# Integrity: prove the set is the sweep before averaging anything over it.
# --------------------------------------------------------------------------- #
def integrity_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fail loudly unless the run set is complete, uniform and internally consistent."""
    problems: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    check(len(rows) == EXPECTED_RUNS, f"expected {EXPECTED_RUNS} runs, found {len(rows)}")

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_instance[r["instance_id"]].append(r)
    check(len(by_instance) == EXPECTED_INSTANCES,
          f"expected {EXPECTED_INSTANCES} distinct instances, found {len(by_instance)}")

    for iid, group in sorted(by_instance.items()):
        caps = {r["cap"] for r in group}
        check(caps == EXPECTED_CAPS, f"{iid}: caps {sorted(caps)}, expected {sorted(EXPECTED_CAPS)}")
        # The same instance must look identical at every cap: the cap changes the budget,
        # never the task or its ground truth.
        for field in ("optimum", "complexity_metric", "level_old", "n_people",
                      "travel_structure"):
            values = {r[field] for r in group}
            check(len(values) == 1, f"{iid}: {field} differs across caps: {sorted(map(str, values))}")

    dup = [k for k, v in Counter(r["run_id"] for r in rows).items() if v > 1]
    check(not dup, f"duplicate run ids: {dup}")

    for field, expected in (("condition", EXPECTED_CONDITION), ("model", EXPECTED_MODEL),
                            ("max_model_len", EXPECTED_MAX_MODEL_LEN)):
        seen = {r[field] for r in rows}
        check(seen == {expected}, f"{field} is {sorted(map(str, seen))}, expected {expected!r}")

    for field in ("subset_hash", "binning_hash"):
        seen = {r[field] for r in rows}
        check(len(seen) == 1 and None not in seen,
              f"{field} is not uniform across the set: {sorted(map(str, seen))}")

    seen_subset = {r["subset_hash"] for r in rows}
    check(seen_subset == {EXPECTED_SUBSET_HASH},
          f"subset hash is {sorted(map(str, seen_subset))}, but this analyser describes the "
          f"calibration sweep ({EXPECTED_SUBSET_HASH[:12]}). Formal-pilot runs need their "
          "own analysis: this one would subtract an exposed instance that is not in them "
          "and label the output calibration-exposed.")

    missing_usage = [r["run_id"] for r in rows if not r["has_endpoint_usage"]]
    check(not missing_usage, f"{len(missing_usage)} run(s) without endpoint usage")

    # The ledger-vs-endpoint identity is the whole basis for calling the budget exact.
    mismatched = [r["run_id"] for r in rows
                  if r["internal_total_tokens"] != r["endpoint_total_tokens"]]
    check(not mismatched, f"{len(mismatched)} run(s) where internal != endpoint tokens")

    leaks = [r["run_id"] for r in rows if r["think_leak"] is not False]
    check(not leaks, f"{len(leaks)} run(s) with think_leak not False")

    if problems:
        raise CalibrationIntegrityError(
            "calibration set failed the integrity audit:\n  - " + "\n  - ".join(problems))

    return {
        "n_runs": len(rows),
        "n_instances": len(by_instance),
        "caps": sorted(EXPECTED_CAPS),
        "condition": EXPECTED_CONDITION,
        "model": EXPECTED_MODEL,
        "max_model_len": EXPECTED_MAX_MODEL_LEN,
        "subset_hash": rows[0]["subset_hash"],
        "binning_hash": rows[0]["binning_hash"],
        "duplicate_run_ids": [],
        "runs_without_endpoint_usage": 0,
        "runs_with_token_mismatch": 0,
        "runs_with_think_leak": 0,
        "all_checks_passed": True,
    }


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_runs": 0}
    att = sum(r["attempts"] for r in rows)
    with_valid = [r for r in rows if r["any_valid"]]
    reasons: Counter = Counter(x for r in rows for x in r["reasons"])
    return {
        "n_runs": len(rows),
        "optimum_mean": _mean([r["optimum"] for r in rows]),
        "optimum_median": _median([r["optimum"] for r in rows]),
        "achieved_mean": _mean([r["achieved"] for r in rows]),
        "achieved_median": _median([r["achieved"] for r in rows]),
        "abs_gap_mean": _mean([r["abs_gap"] for r in rows]),
        "satisfaction_mean": _mean([r["satisfaction"] for r in rows]),
        "satisfaction_median": _median([r["satisfaction"] for r in rows]),
        "tokens_mean": _mean([r["tokens_total"] for r in rows]),
        "attempts_total": att,
        "attempts_per_run": _mean([r["attempts"] for r in rows]),
        "parse_rate": _rate(sum(r["parsed"] for r in rows), att),
        "validity_rate": _rate(sum(r["valid"] for r in rows), att),
        "acceptance_rate": _rate(sum(r["accepted"] for r in rows), att),
        "share_runs_any_valid": _rate(len(with_valid), len(rows)),
        # Conditional on success, and the count it is conditioned on is reported with
        # it -- a median over three runs is not the same statement as one over twenty.
        "n_runs_with_valid": len(with_valid),
        "median_tokens_to_first_valid": _median(
            [r["tokens_to_first_valid"] for r in with_valid
             if r["tokens_to_first_valid"] is not None]),
        "median_first_valid_index": _median(
            [r["first_valid_index"] for r in with_valid
             if r["first_valid_index"] is not None]),
        "rejection_reasons": dict(reasons),
        "termination": dict(Counter(r["termination"] for r in rows)),
        "ctx_limited_calls": sum(r["ctx_limited_calls"] for r in rows),
        "total_calls": sum(r["n_calls"] for r in rows),
        "length_calls": sum(r["length_calls"] for r in rows),
    }


def correlations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["complexity_metric"] is not None]
    if len(usable) < 3:
        return {"n": len(usable)}
    cx = [r["complexity_metric"] for r in usable]
    return {
        "n": len(usable),
        "spearman_complexity_vs_satisfaction": spearman(cx, [r["satisfaction"] for r in usable]),
        "spearman_complexity_vs_achieved": spearman(cx, [r["achieved"] for r in usable]),
        "spearman_complexity_vs_abs_gap": spearman(cx, [r["abs_gap"] for r in usable]),
        "spearman_complexity_vs_optimum": spearman(cx, [r["optimum"] for r in usable]),
    }


def paired(rows: Sequence[dict[str, Any]], lo: int, hi: int) -> dict[str, Any]:
    """32k vs 64k on the instances present at both caps."""
    by_cap = {c: {r["instance_id"]: r for r in rows if r["cap"] == c} for c in (lo, hi)}
    shared = sorted(set(by_cap[lo]) & set(by_cap[hi]))
    deltas = [(by_cap[hi][i]["satisfaction"] - by_cap[lo][i]["satisfaction"]) for i in shared]
    tok_hi = [by_cap[hi][i]["tokens_total"] for i in shared]
    q = statistics.quantiles(tok_hi, n=4) if len(tok_hi) >= 4 else []
    return {
        "n_paired": len(shared),
        "improved": sum(d > 0 for d in deltas),
        "tied": sum(d == 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "satisfaction_delta_mean": _mean(deltas),
        "satisfaction_delta_median": _median(deltas),
        "attempts_delta_mean": _mean(
            [by_cap[hi][i]["attempts"] - by_cap[lo][i]["attempts"] for i in shared]),
        "tokens_delta_mean": _mean(
            [by_cap[hi][i]["tokens_total"] - by_cap[lo][i]["tokens_total"] for i in shared]),
        "tokens_at_hi_quartiles": [round(x, 1) for x in q],
        "share_hi_over_32k_tokens": _rate(sum(t > 32000 for t in tok_hi), len(tok_hi)),
        "share_hi_over_48k_tokens": _rate(sum(t > 48000 for t in tok_hi), len(tok_hi)),
    }


def context_crosstab(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """context_limited is a reduced ALLOWANCE; finish_reason=length is actual truncation."""
    return {
        "limited_and_length": sum(r["ctx_and_length"] for r in rows),
        "limited_and_stop": sum(r["ctx_and_stop"] for r in rows),
        "not_limited_and_length": sum(r["nolimit_and_length"] for r in rows),
        "limited_calls": sum(r["ctx_limited_calls"] for r in rows),
        "total_calls": sum(r["n_calls"] for r in rows),
    }


def by_key(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[Any, list] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {str(k): aggregate(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def analyse(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    caps = sorted({r["cap"] for r in rows})
    out: dict[str, Any] = {"label": label, "n_runs": len(rows), "caps": caps, "by_cap": {}}
    for cap in caps:
        at_cap = [r for r in rows if r["cap"] == cap]
        block: dict[str, Any] = {
            "all": aggregate(at_cap),
            "by_level_old": {lv: aggregate([r for r in at_cap if r["level_old"] == lv])
                             for lv in _LEVELS},
            "by_n_people": by_key(at_cap, "n_people"),
            "by_travel_structure": by_key(at_cap, "travel_structure"),
            "context": context_crosstab(at_cap),
        }
        if cap in (32000, 64000):
            block["correlations"] = correlations(at_cap)
        out["by_cap"][str(cap)] = block
    out["paired_32k_64k"] = paired(rows, 32000, 64000)
    return out


# --------------------------------------------------------------------------- #
# Audit of the new formal subset (composition only -- no runs exist yet).
# --------------------------------------------------------------------------- #
def audit_subset(subset_path: Path, old_subset_path: Path | None) -> dict[str, Any]:
    s = json.loads(subset_path.read_text(encoding="utf-8"))
    per_level: dict[str, Any] = {}
    all_ids: list[str] = []
    for lvl, rows in s["instances"].items():
        all_ids += [r["instance_id"] for r in rows]
        per_level[lvl] = {
            "n": len(rows),
            "strata_n_people_travel": {
                f"{k[0]}-{k[1]}": v for k, v in sorted(Counter(
                    (r["cell"]["n_people"], r["cell"]["travel_structure"]) for r in rows
                ).items())},
            "n_people": dict(Counter(r["cell"]["n_people"] for r in rows)),
            "travel_structure": dict(Counter(r["cell"]["travel_structure"] for r in rows)),
            "tightness": dict(Counter(r["cell"]["tightness"] for r in rows)),
            "overlap": dict(Counter(r["cell"]["overlap"] for r in rows)),
            "seeds": sorted({r["seed"] for r in rows}),
            "complexity_min_median_max": [
                min(r["complexity_metric"] for r in rows),
                _median([r["complexity_metric"] for r in rows]),
                max(r["complexity_metric"] for r in rows)],
            "optimum_mean": _mean([r["optimum"] for r in rows]),
            "optimum_median": _median([r["optimum"] for r in rows]),
            "optimum_min": min(r["optimum"] for r in rows),
            "optimum_max": max(r["optimum"] for r in rows),
        }
    audit = {
        "subset_hash": s["content_hash"],
        "manifest_hash": s["manifest_hash"],
        "counts": {lvl: per_level[lvl]["n"] for lvl in per_level},
        "is_10_10_10": all(per_level[lvl]["n"] == 10 for lvl in _LEVELS)
                       and set(per_level) == set(_LEVELS),
        "duplicate_instance_ids": [k for k, v in Counter(all_ids).items() if v > 1],
        "per_level": per_level,
    }
    if old_subset_path and old_subset_path.exists():
        old = json.loads(old_subset_path.read_text(encoding="utf-8"))
        old_ids = {r["instance_id"] for v in old["instances"].values() for r in v}
        audit["overlap_with_superseded_subset"] = sorted(set(all_ids) & old_ids)
    return audit


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v}" if not isinstance(v, float) else f"{v:.3f}")


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    add("# C1 calibration sweep — analysis\n")
    add("These runs are **calibration evidence, not held-out pilot evidence**: one of the")
    add(f"thirty instances (`{EXPOSED_INSTANCE}`) had been run and inspected before the")
    add("manifest was frozen. They inform the budget ladder and describe mechanisms; they")
    add("are not a pre-registered basis for comparing C1 with C2/C3.\n")
    add("Levels below are the OLD binning (medium <= 5). The formal manifest cuts its own")
    add("boundaries (medium <= 6), so categorical levels are **not** comparable across the")
    add("two, and no table here mixes them.\n")

    for name in ("full", "without_exposed"):
        a = doc[name]
        add(f"\n## {a['label']} (n = {a['n_runs']} runs)\n")
        add("| cap | sat | optimum | achieved | abs gap | tokens | att/run | parse | validity | acceptance | runs w/ valid |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for cap in a["caps"]:
            g = a["by_cap"][str(cap)]["all"]
            add(f"| {cap} | {_fmt(g['satisfaction_mean'])} | {_fmt(g['optimum_mean'])} | "
                f"{_fmt(g['achieved_mean'])} | {_fmt(g['abs_gap_mean'])} | "
                f"{_fmt(g['tokens_mean'])} | {_fmt(g['attempts_per_run'])} | "
                f"{_fmt(g['parse_rate'])} | {_fmt(g['validity_rate'])} | "
                f"{_fmt(g['acceptance_rate'])} | {_fmt(g['share_runs_any_valid'])} |")
        p = a["paired_32k_64k"]
        add(f"\n**32k vs 64k, paired on {p['n_paired']} instances**: improved {p['improved']}, "
            f"tied {p['tied']}, worsened {p['worsened']}; mean Δsat "
            f"{_fmt(p['satisfaction_delta_mean'])}, median {_fmt(p['satisfaction_delta_median'])}; "
            f"mean Δattempts {_fmt(p['attempts_delta_mean'])}; mean Δtokens "
            f"{_fmt(p['tokens_delta_mean'])}; 64k token quartiles {p['tokens_at_hi_quartiles']}; "
            f"share above 32k tokens {_fmt(p['share_hi_over_32k_tokens'])}, above 48k "
            f"{_fmt(p['share_hi_over_48k_tokens'])}.\n")

    add("\n## Level tables (old binning), full set\n")
    for cap in doc["full"]["caps"]:
        add(f"\n### cap {cap}\n")
        add("| level | n | sat | optimum | achieved | abs gap | att | validity | runs w/ valid | termination |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for lv in _LEVELS:
            g = doc["full"]["by_cap"][str(cap)]["by_level_old"][lv]
            if not g.get("n_runs"):
                continue
            add(f"| {lv} | {g['n_runs']} | {_fmt(g['satisfaction_mean'])} | "
                f"{_fmt(g['optimum_mean'])} | {_fmt(g['achieved_mean'])} | "
                f"{_fmt(g['abs_gap_mean'])} | {g['attempts_total']} | "
                f"{_fmt(g['validity_rate'])} | {_fmt(g['share_runs_any_valid'])} | "
                f"{g['termination']} |")
        corr = doc["full"]["by_cap"][str(cap)].get("correlations")
        if corr and corr.get("n", 0) >= 3:
            add(f"\nSpearman over n = {corr['n']} runs (wide interval at this size, reported")
            add("as description, not as an effect size): complexity vs satisfaction "
                f"{_fmt(corr['spearman_complexity_vs_satisfaction'])}, vs achieved "
                f"{_fmt(corr['spearman_complexity_vs_achieved'])}, vs absolute gap "
                f"{_fmt(corr['spearman_complexity_vs_abs_gap'])}, vs oracle optimum "
                f"{_fmt(corr['spearman_complexity_vs_optimum'])}.\n")
        ctx = doc["full"]["by_cap"][str(cap)]["context"]
        add(f"\nContext cross-tab: limited+length {ctx['limited_and_length']}, "
            f"limited+stop {ctx['limited_and_stop']}, not-limited+length "
            f"{ctx['not_limited_and_length']}, limited calls {ctx['limited_calls']} of "
            f"{ctx['total_calls']}. A limited call had a reduced ALLOWANCE; only "
            "finish_reason=length is actual truncation.\n")

    aud = doc["formal_subset_audit"]
    add("\n## Formal subset audit (composition only — no runs exist for it yet)\n")
    add(f"- subset `{aud['subset_hash'][:12]}` from manifest `{aud['manifest_hash'][:12]}`")
    add(f"- counts {aud['counts']}, 10/10/10: {aud['is_10_10_10']}")
    add(f"- duplicate instance ids: {aud['duplicate_instance_ids'] or 'none'}")
    add(f"- overlap with the superseded subset: "
        f"{aud.get('overlap_with_superseded_subset') or 'none'}")
    add("\n| level | n | strata (n_people-travel) | seeds | complexity min/med/max | optimum mean/med/min/max |")
    add("|---|---|---|---|---|---|")
    for lv in _LEVELS:
        p = aud["per_level"][lv]
        add(f"| {lv} | {p['n']} | {p['strata_n_people_travel']} | {p['seeds']} | "
            f"{p['complexity_min_median_max']} | {p['optimum_mean']} / {p['optimum_median']} / "
            f"{p['optimum_min']} / {p['optimum_max']} |")
    add("\n**Coverage limitation, recorded not fixed.** The formal hard level contains no")
    add("n=4 instances: with the new pool's boundary (hard > 6) no four-person instance")
    add("qualifies as hard, so the round-robin cannot place one. Level and n_people are")
    add("therefore partially confounded at the top level. The manifest is NOT regenerated")
    add("to repair this -- doing so would mean choosing the selection after seeing its")
    add("composition.\n")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path("results/calibration/c1_20260805"))
    ap.add_argument("--formal-subset", type=Path,
                    default=Path("results/manifests/subset__pilot__29d836cb5614.json"))
    ap.add_argument("--superseded-subset", type=Path,
                    default=Path("results/manifests/subset__pilot__d29b54752971.json"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/c1_calibration"))
    args = ap.parse_args(argv)

    rows = load_runs(args.runs)
    if not rows:
        raise SystemExit(f"no run documents under {args.runs}")
    # The infrastructure smoke lives in its own directory and is never read here; the
    # audit below would reject it anyway (wrong window, wrong seed range, no provenance).
    audit = integrity_audit(rows)
    kept = [r for r in rows if r["instance_id"] != EXPOSED_INSTANCE]

    doc = {
        "schema_version": "c1_calibration_analysis/1.1",
        # posix form: str(Path) uses the platform separator, so a Windows rerun would
        # rewrite this field and the artifact would look changed without any data change.
        "runs_dir": args.runs.as_posix(),
        "integrity_audit": audit,
        "n_run_documents": len(rows),
        "exposed_instance": EXPOSED_INSTANCE,
        "n_runs_excluding_exposed": len(kept),
        "full": analyse(rows, "All runs"),
        "without_exposed": analyse(kept, f"Excluding {EXPOSED_INSTANCE}"),
        "formal_subset_audit": audit_subset(args.formal_subset, args.superseded_subset),
    }

    # Every artifact is written with explicit LF and no platform-dependent formatting, so
    # a rerun on another machine reproduces the same bytes -- which is what makes
    # `git status` a usable reproducibility check on these files.
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    fields = [k for k in rows[0] if k != "reasons"]
    with (args.out / "runs.csv").open("w", encoding="utf-8", newline="\n") as fh:
        # DictWriter defaults to CRLF on every platform; git then normalises it on one
        # machine and not the other, and the file appears to change without any data
        # changing.
        w = csv.DictWriter(fh, fieldnames=fields + ["reasons"], lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in fields}, "reasons": ";".join(r["reasons"])})
    write_report(doc, args.out / "report.md")

    print("integrity audit: passed — "
          f"{audit['n_runs']} runs, {audit['n_instances']} instances, "
          f"caps {audit['caps']}, subset {audit['subset_hash'][:12]}, "
          f"binning {audit['binning_hash'][:12]}, window {audit['max_model_len']}")
    print(f"runs: {len(rows)} ({len(kept)} excluding the exposed instance)")
    for cap in doc["full"]["caps"]:
        g = doc["full"]["by_cap"][str(cap)]["all"]
        print(f"cap {cap:>6}: sat {_fmt(g['satisfaction_mean'])} | opt {_fmt(g['optimum_mean'])} "
              f"| achieved {_fmt(g['achieved_mean'])} | att {g['attempts_total']:>3} "
              f"| validity {_fmt(g['validity_rate'])} "
              f"| runs w/ valid {_fmt(g['share_runs_any_valid'])}")
    print(f"written: {args.out}/summary.json, runs.csv, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
