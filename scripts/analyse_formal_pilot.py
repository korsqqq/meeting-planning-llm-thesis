# scripts/analyse_formal_pilot.py
"""Analysis of the pre-registered formal pilot. CPU only -- reads JSON, runs nothing.

    python -m scripts.analyse_formal_pilot \
        --runs results/formal_pilot \
        --out results/analysis/formal_pilot

This is a PILOT analysis, not a confirmatory test of the research question. The pilot
draws seeds 10005+ and carries ten instances per complexity level; the held-out main
experiment (seeds 100000+) is the confirmatory one and needs at least fifty per level and
cap. Everything here is design validation: it says whether the instrument measures what
the design assumed, and it is allowed to find that it does not.

Sign convention follows the expose. The primary contrast is

    delta = S_C1 - S_C3      (single agent minus multi-agent)

so **delta > 0 means the single agent scored higher and delta < 0 means the hierarchy
did**. H1 predicts delta > 0 on easy instances, H2 predicts delta < 0 on hard ones, and
the crossover the research question asks about is a sign change from + to - as complexity
rises. Contrasts are named `delta_<a>_minus_<b>` throughout so the direction is never
inferred from a label like "better".

The design is paired: every instance is solved by all three conditions at the same cap.
Both inferential procedures preserve that pairing. The bootstrap resamples instances --
that is, whole paired differences -- never the two conditions independently. The
interaction test permutes complexity labels across instances while each paired difference
stays intact, so it asks whether the difference depends on complexity rather than whether
the conditions differ at all.

Proposal validity is reported per role and is a DIAGNOSTIC, not the evidence for why one
condition wins. A C3 worker plans a decomposed sub-instance while C1 plans the whole
task, so worker and C1 proposals are not the same measurement; and a C3 critic proposal
has already passed the aggregator gate, so it is close to valid by construction. Pooling
the three would compare generation against generation-plus-filtered-selection. The causal
question these numbers cannot settle is what C4 exists to address.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_pilot_sweep import content_hash  # noqa: E402

_LEVELS = ("easy", "medium", "hard")
CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas")
EXPECTED_RUNS = 360
EXPECTED_INSTANCES = 30
EXPECTED_CAPS = {8000, 16000, 32000, 64000}
EXPECTED_MODEL = "Qwen/Qwen3-32B-AWQ"
EXPECTED_MAX_MODEL_LEN = 32768

# Full digests, as pre-registered in THESIS_DECISIONS section 3. Pinned in full rather
# than by prefix: a prefix is enough to tell two known manifests apart and not enough to
# detect a manifest that was edited after freezing.
EXPECTED_SUBSET_HASH = (
    "29d836cb56141f3a29ac3fa9670e662c123586e967f2d5ae1280925c2228edbd"
)
EXPECTED_BINNING_HASH = (
    "e6a365481dd4a485867c62377ac5bc6dc58b277113ccd207605a0258d5cba306"
)
EXPECTED_RUNS_MANIFEST_HASH = (
    "cc2e97997004e57e2b065005b5b016b067aeb3ef5fe51bf5a8f29b6b0c8a663a"
)
# Provenance has two commits and the report must not blur them. `bc968bb` is the code the
# pre-registration froze; `916434a` is what actually executed. The difference between them
# is additive `--write-expected-runs` plumbing that returns before the first run, so the
# execution path is unchanged -- checkable with
# `git diff bc968bb 916434a -- scripts/run_pilot_sweep.py`.
PREREGISTERED_CODE_COMMIT = "bc968bbb5d60043b52d64173115b8b261e0649bf"
EXECUTING_COMMIT = "916434ac8d27a4c889ee754c779d1dfeaf47e29a"

BOOTSTRAP_RESAMPLES = 10000
PERMUTATIONS = 10000
RESAMPLE_SEED = 20260807


class FormalPilotIntegrityError(RuntimeError):
    """The run set is not the pre-registered matrix. Analysis must not proceed."""


# --------------------------------------------------------------------------- #
# Statistics. No scipy in the harness environment; everything here is exact or
# resampling-based, with a fixed seed so the artifacts stay byte-reproducible.
# --------------------------------------------------------------------------- #
def _ranks(values: Sequence[float]) -> list[float]:
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
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else None


def sign_test(deltas: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided sign test over non-tied pairs. SECONDARY robustness check only."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    n = pos + neg
    if n == 0:
        return {"n_pairs": len(deltas), "positive": 0, "negative": 0, "ties": ties,
                "p_two_sided": None}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return {"n_pairs": len(deltas), "positive": pos, "negative": neg, "ties": ties,
            "p_two_sided": round(min(1.0, 2 * tail / (2 ** n)), 5)}


def paired_bootstrap_ci(
    deltas: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = RESAMPLE_SEED, alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile CI for the mean paired difference, resampling INSTANCES.

    The unit of resampling is the instance, so a draw takes the whole paired difference
    `d_i = S_a(i) - S_b(i)` with it. Resampling the two conditions independently would
    destroy the blocking the design was built on and would widen the interval by
    re-introducing between-instance variance that the pairing already removes.

    With ten instances per level and cap, and a majority of exact ties, these intervals
    are wide and usually contain zero. That is a property of the pilot's size, and it is
    the reason the pilot cannot serve as the confirmatory test.
    """
    n = len(deltas)
    if n < 2:
        return {"n": n, "mean": _mean(deltas), "ci_low": None, "ci_high": None,
                "excludes_zero": False, "resamples": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[max(0, int(math.floor((alpha / 2) * resamples)) - 1)]
    hi = means[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return {"n": n, "mean": _mean(deltas), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "excludes_zero": (lo > 0) or (hi < 0), "resamples": resamples}


def _kruskal_h(groups: Sequence[Sequence[float]]) -> float:
    """Kruskal-Wallis H over ranks pooled across groups.

    No tie correction: the permutation null below reshuffles the same values, so the tie
    pattern is common to the observed statistic and to every replicate. The correction
    exists to make H comparable to a chi-square reference distribution, which is not used
    here -- and with this many exact zeros that reference would be poor anyway.
    """
    flat = [x for g in groups for x in g]
    n = len(flat)
    if n < 2:
        return 0.0
    ranks = _ranks(flat)
    out, i, total = 0.0, 0, 0.0
    for g in groups:
        rs = ranks[i:i + len(g)]
        i += len(g)
        if rs:
            total += len(rs) * (statistics.fmean(rs) ** 2)
    out = 12.0 / (n * (n + 1)) * total - 3 * (n + 1)
    return out


def _jonckheere(groups: Sequence[Sequence[float]]) -> float:
    """Jonckheere-Terpstra statistic for an ORDERED alternative across groups."""
    jt = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for x in groups[i]:
                for y in groups[j]:
                    jt += 1.0 if y > x else (0.5 if y == x else 0.0)
    return jt


def interaction_tests(
    deltas_by_instance: Sequence[tuple[float, str, int]], *,
    permutations: int = PERMUTATIONS, seed: int = RESAMPLE_SEED,
) -> dict[str, Any]:
    """Does the paired difference depend on complexity? Blocked permutation, three levels.

    Input is one triple per instance: `(d_i, level_i, complexity_i)`. Every permutation
    shuffles the level labels across instances while each `d_i` stays whole, so the null
    is "the paired difference is unrelated to complexity" rather than "the conditions do
    not differ". That is the interaction the research question asks about, and neither the
    sign test nor a per-cap mean can address it.

    Three statistics, because they answer different questions:

    * **omnibus** (Kruskal-Wallis H over easy/medium/hard) -- do the three levels differ at
      all, in any pattern?
    * **trend** (Jonckheere-Terpstra, levels ordered easy < medium < hard) -- do they
      differ monotonically? Under the expose convention `delta = S_C1 - S_C3`, H1 and H2
      together predict delta FALLING from positive on easy to negative on hard, so the
      pre-registered alternative is JT **below** its null expectation. That one-sided
      lower-tail p is the pre-registered test. A p for the opposite, rising direction is
      also reported, and it is exploratory: it tests an alternative nobody registered, and
      quoting it as though it evaluated H1/H2 would be evaluating the hypothesis against
      its own contradiction. The two-sided p is reported as well and labelled as such.
    * **extreme contrast** mean(d | easy) - mean(d | hard), reported because it is the
      quantity the crossover claim is about and is readable in the metric's own units.
      Same treatment: the pre-registered alternative is a POSITIVE contrast.

    A large p here does NOT establish that the difference is independent of complexity.
    Ten instances per level gives very little power against anything but a large effect,
    so a null result is uninformative about the alternative rather than evidence for the
    null. Reported as a pilot diagnostic; the held-out main experiment is what can test it.
    """
    rows = [(d, lv) for d, lv, _ in deltas_by_instance if lv in _LEVELS]
    present = [lv for lv in _LEVELS if any(l == lv for _, l in rows)]
    if len(present) < 2:
        return {"n": len(rows), "omnibus": None, "trend": None, "extreme_contrast": None}

    def split(labels: Sequence[str]) -> list[list[float]]:
        by: dict[str, list[float]] = {lv: [] for lv in present}
        for (d, _), lb in zip(rows, labels):
            by[lb].append(d)
        return [by[lv] for lv in present]

    def extreme(groups: Sequence[Sequence[float]]) -> float:
        first, last = groups[0], groups[-1]
        if not first or not last:
            return 0.0
        return statistics.fmean(first) - statistics.fmean(last)

    labels = [lv for _, lv in rows]
    obs_groups = split(labels)
    obs_h = _kruskal_h(obs_groups)
    obs_jt = _jonckheere(obs_groups)
    obs_ex = extreme(obs_groups)
    sizes = [len(g) for g in obs_groups]
    n = sum(sizes)
    jt_null_mean = (n * n - sum(s * s for s in sizes)) / 4.0

    rng = random.Random(seed)
    shuffled = list(labels)
    hits_h = 0
    jt_le = jt_ge = jt_two = 0
    ex_ge = ex_le = ex_two = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        g = split(shuffled)
        if _kruskal_h(g) >= obs_h - 1e-12:
            hits_h += 1
        jt = _jonckheere(g)
        if jt <= obs_jt + 1e-12:
            jt_le += 1
        if jt >= obs_jt - 1e-12:
            jt_ge += 1
        if abs(jt - jt_null_mean) >= abs(obs_jt - jt_null_mean) - 1e-12:
            jt_two += 1
        ex = extreme(g)
        if ex >= obs_ex - 1e-12:
            ex_ge += 1
        if ex <= obs_ex + 1e-12:
            ex_le += 1
        if abs(ex) >= abs(obs_ex) - 1e-12:
            ex_two += 1

    def p(hits: int) -> float:
        return round((hits + 1) / (permutations + 1), 5)

    rising = obs_jt > jt_null_mean
    return {
        "n": n,
        "levels_present": present,
        "permutations": permutations,
        "omnibus": {
            "test": "Kruskal-Wallis H across easy/medium/hard, permutation null",
            "directional": False,
            "statistic": round(obs_h, 4),
            "p": p(hits_h),
        },
        "trend": {
            "test": "Jonckheere-Terpstra, levels ordered easy < medium < hard",
            "statistic": round(obs_jt, 2),
            "null_expectation": round(jt_null_mean, 2),
            "observed_direction": ("delta rises with complexity" if rising
                                   else "delta falls with complexity"
                                   if obs_jt < jt_null_mean else "flat"),
            "preregistered_direction": "delta falls with complexity (H1 delta > 0 on easy, "
                                       "H2 delta < 0 on hard)",
            # The registered alternative only. A falling delta means fewer concordant
            # increases, so it lives in the LOWER tail of the permutation distribution.
            "p_one_sided_preregistered_decreasing": p(jt_le),
            # The opposite alternative. Exploratory by definition: it is the direction the
            # data happens to show, not one anybody registered, and it must never be
            # quoted as a test of H1/H2.
            "p_one_sided_opposite_increasing_exploratory": p(jt_ge),
            "p_two_sided": p(jt_two),
        },
        "extreme_contrast": {
            "statistic_easy_minus_hard": round(obs_ex, 4),
            "preregistered_direction": "positive (delta larger on easy than on hard)",
            "p_one_sided_preregistered_positive": p(ex_ge),
            "p_one_sided_opposite_negative_exploratory": p(ex_le),
            "p_two_sided": p(ex_two),
        },
        # The continuous form §6 requires for the main analysis, over every instance.
        "spearman_complexity_vs_delta": spearman(
            [c for _, _, c in deltas_by_instance], [d for d, _, _ in deltas_by_instance]),
        "power_note": (
            "A large p does not establish independence from complexity: at ten instances "
            "per level this has power only against a large effect."
        ),
    }


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
def _plan_size(plan: Any) -> int:
    if not isinstance(plan, dict):
        return 0
    return max((len(v) for v in plan.values() if isinstance(v, list)), default=0)


def load_runs(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        rr, agent = doc["run_result"], doc["agent"]
        calls = rr.get("calls", [])
        props = agent.get("proposals", [])
        score = rr["score"]
        toks = rr["tokens"]

        call_roles = Counter(c.get("role") for c in calls)
        prop_roles = Counter(p.get("role") for p in props)
        valid_by_role: Counter = Counter()
        for p in props:
            if p["valid"]:
                valid_by_role[p.get("role")] += 1

        instance = doc.get("instance", {})
        gp = instance.get("generator_params", {})
        md = doc.get("pilot_sweep", {}) or {}
        audit = doc.get("usage_audit", {}) or {}
        totals = audit.get("totals", {}) or {}

        rows.append({
            "run_id": rr["run_id"],
            "condition": rr["condition"],
            "instance_id": rr["instance_id"],
            "cap": rr["cap"],
            "level": rr["level"],
            "model": rr["model"],
            "seed": rr["seed"],
            "subset_hash": md.get("subset_hash"),
            "binning_hash": md.get("binning_hash"),
            "git_commit": md.get("git_commit"),
            "max_model_len": md.get("max_model_len"),
            "base_url": md.get("base_url"),
            "has_endpoint_usage": audit.get("has_endpoint_usage"),
            "internal_total_tokens": totals.get("internal_total_tokens"),
            "endpoint_total_tokens": totals.get("endpoint_total_tokens"),
            "think_leak": (doc.get("transcript_sanity") or {}).get("think_leak"),
            "n_people": gp.get("n_people"),
            "travel_structure": gp.get("travel_structure"),
            "tightness": gp.get("tightness"),
            "overlap": gp.get("overlap"),
            "complexity_metric": instance.get("complexity_metric"),
            # Outcome, primary and the expose's secondaries.
            "optimum": score["solver_optimum"],
            "achieved": score["n_valid_meetings"],
            "abs_gap": score["solver_optimum"] - score["n_valid_meetings"],
            "satisfaction": score["satisfaction"],
            "feasible": bool(score.get("valid")),
            "optimality": bool(score.get("optimality")),
            "invalid_reasons": list(score.get("invalid_reasons") or []),
            "final_plan_meetings": _plan_size(rr.get("final_plan")),
            "nonempty_plan": _plan_size(rr.get("final_plan")) > 0,
            "tokens_total": toks["total"],
            "tokens_remaining": rr["cap"] - toks["total"],
            "cap_utilisation": round(toks["total"] / rr["cap"], 4),
            "budget_exhausted": bool(toks.get("budget_exhausted")),
            "latency_seconds": rr.get("latency_seconds"),
            "n_calls": len(calls),
            "n_steps": agent.get("n_steps"),
            "termination": agent.get("termination"),
            "finalization_mismatch": bool(agent.get("finalization_mismatch")),
            "empty_turns": agent.get("empty_turns"),
            # Call roles: the direct measurement of what a condition actually did.
            "calls_react_step": call_roles.get("react_step", 0),
            "calls_verify": call_roles.get("verify", 0),
            "calls_revise": call_roles.get("revise", 0),
            "calls_worker_a": call_roles.get("worker_a", 0),
            "calls_worker_b": call_roles.get("worker_b", 0),
            "calls_critic": call_roles.get("critic", 0),
            "calls_finalize": call_roles.get("finalize", 0),
            # Proposals, split by the role that produced them.
            "attempts": len(props),
            "attempts_react_step": prop_roles.get("react_step", 0),
            "attempts_revise": prop_roles.get("revise", 0),
            "attempts_worker_a": prop_roles.get("worker_a", 0),
            "attempts_worker_b": prop_roles.get("worker_b", 0),
            "attempts_critic": prop_roles.get("critic", 0),
            "valid_react_step": valid_by_role.get("react_step", 0),
            "valid_revise": valid_by_role.get("revise", 0),
            "valid_worker_a": valid_by_role.get("worker_a", 0),
            "valid_worker_b": valid_by_role.get("worker_b", 0),
            "valid_critic": valid_by_role.get("critic", 0),
            "parsed": sum(p["parsed"] for p in props),
            "valid": sum(p["valid"] for p in props),
            "accepted": sum(p["accepted_into_best_plan"] for p in props),
            "any_valid": any(p["valid"] for p in props),
            "reasons": [r for p in props if not p["valid"] for r in p["reasons"]],
            "ctx_limited_calls": sum(bool(c.get("context_limited")) for c in calls),
            "length_calls": sum(c.get("finish_reason") == "length" for c in calls),
        })
    return rows


# --------------------------------------------------------------------------- #
# Integrity gate.
# --------------------------------------------------------------------------- #
def integrity_audit(
    rows: Sequence[dict[str, Any]], expected_runs_manifest: Path | None = None,
) -> dict[str, Any]:
    problems: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    check(len(rows) == EXPECTED_RUNS, f"expected {EXPECTED_RUNS} runs, found {len(rows)}")

    seen_conditions = {r["condition"] for r in rows}
    check(seen_conditions == set(CONDITIONS),
          f"conditions {sorted(seen_conditions)}, expected {sorted(CONDITIONS)}")

    cells = Counter((r["condition"], r["cap"]) for r in rows)
    for cond in CONDITIONS:
        for cap in sorted(EXPECTED_CAPS):
            n = cells.get((cond, cap), 0)
            check(n == EXPECTED_INSTANCES,
                  f"cell {cond}/{cap} has {n} runs, expected {EXPECTED_INSTANCES}")

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_instance[r["instance_id"]].append(r)
    check(len(by_instance) == EXPECTED_INSTANCES,
          f"expected {EXPECTED_INSTANCES} distinct instances, found {len(by_instance)}")

    for iid, group in sorted(by_instance.items()):
        check(len(group) == len(CONDITIONS) * len(EXPECTED_CAPS),
              f"{iid}: {len(group)} runs, expected {len(CONDITIONS) * len(EXPECTED_CAPS)}")
        for field in ("optimum", "complexity_metric", "level", "n_people",
                      "travel_structure", "tightness", "overlap", "seed"):
            values = {r[field] for r in group}
            check(len(values) == 1,
                  f"{iid}: {field} differs across the matrix: {sorted(map(str, values))}")

    dup = [k for k, v in Counter(r["run_id"] for r in rows).items() if v > 1]
    check(not dup, f"duplicate run ids: {dup}")

    for field, expected in (("model", EXPECTED_MODEL),
                            ("max_model_len", EXPECTED_MAX_MODEL_LEN),
                            ("subset_hash", EXPECTED_SUBSET_HASH),
                            ("binning_hash", EXPECTED_BINNING_HASH)):
        seen = {r[field] for r in rows}
        check(seen == {expected},
              f"{field} is {sorted(map(str, seen))}, expected {expected!r}")

    commits = {r["git_commit"] for r in rows}
    check(len(commits) == 1 and None not in commits,
          f"the matrix spans more than one repository state: {sorted(map(str, commits))}")

    missing_usage = [r["run_id"] for r in rows if not r["has_endpoint_usage"]]
    check(not missing_usage, f"{len(missing_usage)} run(s) without endpoint usage")
    mismatched = [r["run_id"] for r in rows
                  if r["internal_total_tokens"] != r["endpoint_total_tokens"]]
    check(not mismatched, f"{len(mismatched)} run(s) where internal != endpoint tokens")
    leaks = [r["run_id"] for r in rows if r["think_leak"] is not False]
    check(not leaks, f"{len(leaks)} run(s) with think_leak not False")

    manifest_check: dict[str, Any] = {"checked": False}
    if expected_runs_manifest is not None and not expected_runs_manifest.exists():
        check(False, f"expected-runs manifest not found at {expected_runs_manifest}")
    elif expected_runs_manifest is not None:
        exp = json.loads(expected_runs_manifest.read_text(encoding="utf-8"))
        # Recompute rather than trust the stored digest: a manifest edited after freezing
        # would still carry a self-consistent-looking `content_hash` field.
        recomputed = content_hash(exp)
        check(recomputed == exp.get("content_hash"),
              f"expected-runs manifest is self-inconsistent: stored "
              f"{exp.get('content_hash')}, recomputed {recomputed}")
        check(recomputed == EXPECTED_RUNS_MANIFEST_HASH,
              f"expected-runs manifest hashes to {recomputed}, "
              f"pre-registered {EXPECTED_RUNS_MANIFEST_HASH}")
        expected_ids = {r["run_id"] for r in exp["runs"]}
        found_ids = {r["run_id"] for r in rows}
        missing, extra = sorted(expected_ids - found_ids), sorted(found_ids - expected_ids)
        check(not missing, f"{len(missing)} pre-registered run(s) absent: {missing[:5]}")
        check(not extra, f"{len(extra)} run(s) present but not pre-registered: {extra[:5]}")
        manifest_check = {"checked": True, "recomputed_hash": recomputed,
                          "n_expected": len(expected_ids), "missing": missing,
                          "extra": extra}

    if problems:
        raise FormalPilotIntegrityError(
            "formal pilot set failed the integrity audit:\n  - " + "\n  - ".join(problems))

    return {
        "n_runs": len(rows),
        "n_instances": len(by_instance),
        "conditions": sorted(seen_conditions),
        "caps": sorted(EXPECTED_CAPS),
        "model": EXPECTED_MODEL,
        "max_model_len": EXPECTED_MAX_MODEL_LEN,
        "subset_hash": EXPECTED_SUBSET_HASH,
        "binning_hash": EXPECTED_BINNING_HASH,
        "expected_runs_manifest": manifest_check,
        "provenance": {
            "preregistered_code_commit": PREREGISTERED_CODE_COMMIT,
            "executing_commit": rows[0]["git_commit"],
            "note": (
                "The pre-registration names the frozen behavioural code; the runs record "
                "the commit that executed. They differ by additive --write-expected-runs "
                "plumbing that returns before the first run, so the execution path is "
                "identical. Verify with: git diff "
                f"{PREREGISTERED_CODE_COMMIT[:7]} {EXECUTING_COMMIT[:7]} "
                "-- scripts/run_pilot_sweep.py"
            ),
        },
        # Informational, deliberately not a uniformity requirement: one GPU was released
        # mid-sweep and the remaining shard was replayed against the other endpoint.
        "base_urls": dict(Counter(r["base_url"] for r in rows)),
        "all_checks_passed": True,
    }


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_runs": 0}
    n = len(rows)
    att = sum(r["attempts"] for r in rows)
    tokens = sum(r["tokens_total"] for r in rows)
    lat = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None]
    return {
        "n_runs": n,
        # Primary.
        "satisfaction_mean": _mean([r["satisfaction"] for r in rows]),
        "satisfaction_median": _median([r["satisfaction"] for r in rows]),
        # Expose secondaries.
        "feasibility_rate": _rate(sum(r["feasible"] for r in rows), n),
        "optimality_rate": _rate(sum(r["optimality"] for r in rows), n),
        "optimum_mean": _mean([r["optimum"] for r in rows]),
        "achieved_mean": _mean([r["achieved"] for r in rows]),
        "abs_gap_mean": _mean([r["abs_gap"] for r in rows]),
        "share_nonempty_plan": _rate(sum(r["nonempty_plan"] for r in rows), n),
        "tokens_mean": _mean([r["tokens_total"] for r in rows]),
        "cap_utilisation_mean": _mean([r["cap_utilisation"] for r in rows]),
        "tokens_remaining_mean": _mean([r["tokens_remaining"] for r in rows]),
        "budget_exhausted_runs": sum(r["budget_exhausted"] for r in rows),
        "satisfaction_per_1k_tokens": (
            round(1000 * sum(r["satisfaction"] for r in rows) / tokens, 5) if tokens else None
        ),
        "calls_mean": _mean([r["n_calls"] for r in rows]),
        "calls_total": sum(r["n_calls"] for r in rows),
        "latency_mean_seconds": _mean(lat),
        "latency_median_seconds": _median(lat),
        # Proposal taxonomy, pooled and by role.
        "attempts_total": att,
        "attempts_per_run": _mean([r["attempts"] for r in rows]),
        "proposal_parse_rate": _rate(sum(r["parsed"] for r in rows), att),
        "proposal_validity_rate": _rate(sum(r["valid"] for r in rows), att),
        "proposal_acceptance_rate": _rate(sum(r["accepted"] for r in rows), att),
        "by_role": {
            role: {
                "attempts": sum(r[f"attempts_{role}"] for r in rows),
                "valid": sum(r[f"valid_{role}"] for r in rows),
                "validity_rate": _rate(sum(r[f"valid_{role}"] for r in rows),
                                       sum(r[f"attempts_{role}"] for r in rows)),
            }
            for role in ("react_step", "revise", "worker_a", "worker_b", "critic")
            if sum(r[f"attempts_{role}"] for r in rows)
        },
        "share_runs_any_valid": _rate(sum(r["any_valid"] for r in rows), n),
        "rejection_reasons": dict(Counter(x for r in rows for x in r["reasons"])),
        "final_invalid_reasons": dict(Counter(x for r in rows for x in r["invalid_reasons"])),
        # Termination. `aborted` is a grant that expired inside a reasoning block, so
        # nothing survived after the closing think tag; it is budget exhaustion reported
        # one level finer and is collapsed with `budget` rather than read as a distinct
        # outcome. Note this is NOT the same as `tokens.budget_exhausted`, which is the
        # ledger's own flag; both are reported so they can be compared.
        "termination": dict(Counter(r["termination"] for r in rows)),
        "budget_limited": sum(1 for r in rows if r["termination"] in ("aborted", "budget")),
        "finalization_mismatch": sum(r["finalization_mismatch"] for r in rows),
        "ctx_limited_calls": sum(r["ctx_limited_calls"] for r in rows),
        "length_calls": sum(r["length_calls"] for r in rows),
    }


def stage_exposure(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What each condition actually executed, counted from call roles.

    Direct measurement, replacing the earlier token-identity proxy: identical token totals
    are consistent with a treatment never running but do not demonstrate it, and the call
    log records the answer outright.
    """
    n = len(rows)
    if not n:
        return {}
    out: dict[str, Any] = {"n_runs": n}
    for role in ("react_step", "verify", "revise", "worker_a", "worker_b", "critic"):
        total = sum(r[f"calls_{role}"] for r in rows)
        if not total:
            continue
        reached = sum(1 for r in rows if r[f"calls_{role}"] > 0)
        out[role] = {
            "calls_total": total,
            "calls_per_run": _mean([r[f"calls_{role}"] for r in rows]),
            "runs_reached": reached,
            "share_runs_reached": _rate(reached, n),
        }
    return out


def by_key(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[Any, list] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {str(k): aggregate(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


# --------------------------------------------------------------------------- #
# Paired contrasts.
# --------------------------------------------------------------------------- #
def _index(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(r["condition"], r["instance_id"], r["cap"]): r for r in rows}


def paired_rows(
    rows: Sequence[dict[str, Any]], a: str, b: str, cap: int,
) -> list[tuple[str, float, str, int, dict[str, Any], dict[str, Any]]]:
    """`(instance, S_a - S_b, level, complexity, row_a, row_b)` for one cap."""
    idx = _index(rows)
    out = []
    for iid in sorted({r["instance_id"] for r in rows}):
        ra, rb = idx.get((a, iid, cap)), idx.get((b, iid, cap))
        if ra is None or rb is None:
            continue
        out.append((iid, ra["satisfaction"] - rb["satisfaction"], ra["level"],
                    ra["complexity_metric"], ra, rb))
    return out


def contrast(rows: Sequence[dict[str, Any]], a: str, b: str, cap: int) -> dict[str, Any]:
    pairs = paired_rows(rows, a, b, cap)
    deltas = [d for _, d, _, _, _, _ in pairs]
    out: dict[str, Any] = {
        "minuend": a, "subtrahend": b, "cap": cap,
        "definition": f"delta = S_{a} - S_{b}; delta < 0 means {b} scored higher",
        "n_paired": len(pairs),
        "delta_mean": _mean(deltas),
        "delta_median": _median(deltas),
        "bootstrap": paired_bootstrap_ci(deltas),
        "tokens_delta_mean": _mean(
            [ra["tokens_total"] - rb["tokens_total"] for _, _, _, _, ra, rb in pairs]),
        "sign_test_secondary": sign_test(deltas),
        "interaction": interaction_tests(
            [(d, lv, cx) for _, d, lv, cx, _, _ in pairs if cx is not None]),
        "by_level": {},
    }
    for lv in _LEVELS:
        d_lv = [d for _, d, level, _, _, _ in pairs if level == lv]
        if not d_lv:
            continue
        out["by_level"][lv] = {
            "n_paired": len(d_lv),
            "delta_mean": _mean(d_lv),
            "bootstrap": paired_bootstrap_ci(d_lv),
            "sign_test_secondary": sign_test(d_lv),
        }
    return out


def crossover_summary(contrasts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The expose's crossover criterion, applied to one contrast across caps.

    A sign change counts only where the bootstrap interval supports it, and the change is
    called robust only if the same level shows it in at least two caps.
    """
    out: dict[str, Any] = {}
    for lv in _LEVELS:
        per_cap = {}
        for cap in sorted(EXPECTED_CAPS):
            b = contrasts[str(cap)]["by_level"].get(lv)
            if not b:
                continue
            m = b["delta_mean"]
            per_cap[str(cap)] = {
                "delta_mean": m,
                "sign": "0" if m == 0 else ("+" if m > 0 else "-"),
                "ci": [b["bootstrap"]["ci_low"], b["bootstrap"]["ci_high"]],
                "ci_excludes_zero": b["bootstrap"]["excludes_zero"],
            }
        supported = [c for c, v in per_cap.items() if v["ci_excludes_zero"]]
        out[lv] = {"per_cap": per_cap, "caps_with_ci_excluding_zero": supported}
    signs = {lv: {c: v["sign"] for c, v in out[lv]["per_cap"].items()} for lv in out}
    out["hypothesised_pattern"] = (
        "H1: delta > 0 on easy (single agent ahead); H2: delta < 0 on hard; crossover is "
        "+ on easy turning - on hard within a cap."
    )
    out["observed_signs"] = signs
    return out


def analyse(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    caps = sorted(EXPECTED_CAPS)
    out: dict[str, Any] = {"levels": list(_LEVELS), "by_condition": {}, "contrasts": {}}
    for cond in CONDITIONS:
        block: dict[str, Any] = {
            "overall": aggregate([r for r in rows if r["condition"] == cond]),
            "by_cap": {},
        }
        for cap in caps:
            at = [r for r in rows if r["condition"] == cond and r["cap"] == cap]
            block["by_cap"][str(cap)] = {
                "all": aggregate(at),
                "stage_exposure": stage_exposure(at),
                "by_level": {lv: aggregate([r for r in at if r["level"] == lv])
                             for lv in _LEVELS},
                "by_n_people": by_key(at, "n_people"),
                "by_travel_structure": by_key(at, "travel_structure"),
            }
        out["by_condition"][cond] = block

    # Named so the direction is explicit. The first is the expose's delta.
    for a, b in (("c1_react", "c3_mas"),
                 ("c1_react", "c2_verify_revise"),
                 ("c2_verify_revise", "c3_mas")):
        key = f"delta_{a}_minus_{b}"
        out["contrasts"][key] = {str(cap): contrast(rows, a, b, cap) for cap in caps}
    out["crossover_primary"] = crossover_summary(
        out["contrasts"]["delta_c1_react_minus_c3_mas"])
    return out


# --------------------------------------------------------------------------- #
# Report.
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else f"{v}")


def _p(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if v >= 0.0001 else "<0.0001")


def _ci(b: dict[str, Any]) -> str:
    if b.get("ci_low") is None:
        return "n/a"
    return f"[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]"


def write_report(doc: dict[str, Any], path: Path) -> None:
    L: list[str] = []
    add = L.append
    a = doc["analysis"]
    aud = doc["integrity_audit"]
    caps = sorted(EXPECTED_CAPS)

    add("# Formal pilot — pre-registered design-validation analysis\n")
    add("**This is a pilot, not a confirmatory test of the research question.** The pilot "
        "carries ten instances per complexity level on seeds 10005+; the held-out main "
        "experiment (seeds 100000+) is the confirmatory one and requires at least fifty "
        "instances per level and cap. Everything below is design validation, and it is "
        "allowed to conclude that the instrument does not measure what the design "
        "assumed.\n")
    add(f"Pre-registered matrix (THESIS_DECISIONS §3): {len(CONDITIONS)} conditions × "
        f"{len(EXPECTED_CAPS)} caps × {EXPECTED_INSTANCES} instances = {EXPECTED_RUNS} "
        f"runs, one run per cell. Subset `{aud['subset_hash']}`, binning "
        f"`{aud['binning_hash']}`, model {aud['model']}, window {aud['max_model_len']}, "
        "prefix caching disabled.\n")
    prov = aud["provenance"]
    add(f"**Provenance.** Frozen behavioural code: `{prov['preregistered_code_commit'][:7]}`. "
        f"Executing commit recorded in every run: `{str(prov['executing_commit'])[:7]}`. "
        "The inspected difference between them is additive pre-registration plumbing "
        "(`--write-expected-runs`) that returns before the first run; the execution path "
        "is unchanged.\n")
    add("**Sign convention (expose).** `delta = S_C1 − S_C3`, single agent minus "
        "multi-agent. A **positive** delta means the single agent scored higher; a "
        "**negative** delta means the hierarchy did. H1 expects delta > 0 on easy "
        "instances, H2 expects delta < 0 on hard ones, and the crossover the research "
        "question asks about is a + to − sign change as complexity rises.\n")
    add("**Inference preserves the pairing.** Every instance is solved by all three "
        "conditions at the same cap. The bootstrap resamples instances, carrying whole "
        "paired differences; the interaction test permutes complexity labels while each "
        "paired difference stays intact. With ten instances per level and cap, intervals "
        "are wide and the interaction test is a diagnostic rather than a hypothesis "
        "test.\n")

    add("\n## Primary outcome and the expose's secondary metrics\n")
    add("| condition | cap | sat | achieved | optimum | optimality | feasible | non-empty plan | tokens | cap used | exhausted | sat/1k | calls | latency s |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cap in caps:
            g = a["by_condition"][cond]["by_cap"][str(cap)]["all"]
            add(f"| {cond} | {cap} | {_fmt(g['satisfaction_mean'])} | "
                f"{_fmt(g['achieved_mean'])} | {_fmt(g['optimum_mean'])} | "
                f"{_fmt(g['optimality_rate'])} | {_fmt(g['feasibility_rate'])} | "
                f"{_fmt(g['share_nonempty_plan'])} | {_fmt(g['tokens_mean'])} | "
                f"{_fmt(g['cap_utilisation_mean'])} | {g['budget_exhausted_runs']} | "
                f"{_fmt(g['satisfaction_per_1k_tokens'])} | {_fmt(g['calls_mean'])} | "
                f"{_fmt(g['latency_mean_seconds'])} |")
    add("\n**Feasibility is 1.000 everywhere and carries no information about the agent.** "
        "The harness finalises whatever plan survives validation, and an empty plan is "
        "vacuously feasible, so the rate is 1 by construction (§5). The informative "
        "counterpart is the share of runs holding a non-empty plan, in the column beside "
        "it. `final_invalid_reasons` is empty in every cell for the same reason; the "
        "rejection taxonomy that does carry information is over proposals, further "
        "below.\n")
    add("**Latency is wall-clock on a shared machine** with thermal throttling active "
        "throughout and one GPU released partway through the sweep, after which the "
        "remaining work ran alone. It is reported because the expose asks for it, and it "
        "must not be read as a clean efficiency comparison between conditions.\n")

    add("\n## Paired contrasts\n")
    for key, blocks in a["contrasts"].items():
        pretty = key.replace("delta_", "").replace("_minus_", " − ")
        add(f"\n### delta = S({pretty.split(' − ')[0]}) − S({pretty.split(' − ')[1]})\n")
        add(f"{blocks[str(caps[0])]['definition']}.\n")
        add("| cap | n | mean delta | median | 95% CI (paired bootstrap) | CI excl. 0 | mean Δtokens | sign test p |")
        add("|---|---|---|---|---|---|---|---|")
        for cap in caps:
            c = blocks[str(cap)]
            add(f"| {cap} | {c['n_paired']} | {_fmt(c['delta_mean'])} | "
                f"{_fmt(c['delta_median'])} | {_ci(c['bootstrap'])} | "
                f"{'yes' if c['bootstrap']['excludes_zero'] else 'no'} | "
                f"{_fmt(c['tokens_delta_mean'])} | "
                f"{_p(c['sign_test_secondary']['p_two_sided'])} |")
        add("\n| cap | level | n | mean delta | 95% CI | CI excl. 0 |")
        add("|---|---|---|---|---|---|")
        for cap in caps:
            for lv in _LEVELS:
                b = blocks[str(cap)]["by_level"].get(lv)
                if not b:
                    continue
                add(f"| {cap} | {lv} | {b['n_paired']} | {_fmt(b['delta_mean'])} | "
                    f"{_ci(b['bootstrap'])} | "
                    f"{'yes' if b['bootstrap']['excludes_zero'] else 'no'} |")
        add("\nInteraction with complexity — all three levels, each paired difference kept "
            "intact and only the level labels permuted:\n")
        add("| cap | omnibus H | omnibus p | pre-registered decreasing-trend p | observed direction | two-sided trend p | easy − hard | pre-reg. positive p | ρ(complexity, delta) |")
        add("|---|---|---|---|---|---|---|---|---|")
        for cap in caps:
            it = blocks[str(cap)]["interaction"]
            om, tr, ex = it.get("omnibus"), it.get("trend"), it.get("extreme_contrast")
            if not om:
                add(f"| {cap} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            add(f"| {cap} | {_fmt(om['statistic'])} | {_p(om['p'])} | "
                f"{_p(tr['p_one_sided_preregistered_decreasing'])} | "
                f"{tr['observed_direction']} | {_p(tr['p_two_sided'])} | "
                f"{_fmt(ex['statistic_easy_minus_hard'])} | "
                f"{_p(ex['p_one_sided_preregistered_positive'])} | "
                f"{_fmt(it.get('spearman_complexity_vs_delta'))} |")
        add("\n**The pre-registered alternative is delta FALLING with complexity** — H1 "
            "puts the single agent ahead on easy, H2 puts the hierarchy ahead on hard, so "
            "the crossover is a `+ → −` change. The one-sided columns above test that "
            "alternative and only that one. Where the observed direction is the opposite, "
            "the pre-registered p is necessarily large, and the p for the rising direction "
            "is recorded in `summary.json` as exploratory: it tests an alternative nobody "
            "registered, and quoting it as evidence about H1/H2 would be evaluating the "
            "hypothesis against its own contradiction.\n")
        add("A large p here does **not** establish that the difference is independent of "
            "complexity. With ten instances per level these tests have power only against "
            "a large effect, so a null result leaves the question open rather than "
            "answering it. Where the point estimates trend against the pre-registered "
            "crossover, the defensible statement is that the estimates lean that way and "
            "the pilot provides insufficient evidence to establish any trend.\n")

    add("\n## Crossover check on the primary contrast\n")
    cs = a["crossover_primary"]
    add(f"{cs['hypothesised_pattern']}\n")
    add("| level | " + " | ".join(f"{c} sign" for c in caps) + " | caps whose CI excludes 0 |")
    add("|---|" + "---|" * (len(caps) + 1))
    for lv in _LEVELS:
        row = cs[lv]
        signs = " | ".join(row["per_cap"].get(str(c), {}).get("sign", "—") for c in caps)
        add(f"| {lv} | {signs} | {row['caps_with_ci_excluding_zero'] or 'none'} |")

    add("\n## Pilot diagnostic: the low-complexity sanity check fails\n")
    add("The expose specifies that on the easiest level the single agent should be at "
        "least on a par with the hierarchy, and that a violation points at the "
        "measurement design, the experimental setup or the evaluation protocol rather "
        "than at a finding. The pilot violates it: on easy instances the hierarchy is "
        "ahead at both working caps, and the margin there is at least as large as on "
        "hard.\n")
    add("This diagnostic does **not** rest on the interaction tests. It is a statement "
        "about the easy level on its own, where the paired interval excludes zero. The "
        "interaction tests are underpowered at ten instances per level and neither "
        "establish nor rule out a complexity dependence; the sanity check fails "
        "regardless of how that question is eventually resolved.\n")
    add("Two properties of the level assignment bear on this and are recorded in §3 "
        "rather than repaired, because repairing them after seeing the scores would mean "
        "choosing the selection from its results:\n")
    add("| level | mean oracle optimum | C1 satisfaction at 64k | C1 meetings at 64k | C3 − C1 delta at 64k |")
    add("|---|---|---|---|---|")
    for lv in _LEVELS:
        g = a["by_condition"]["c1_react"]["by_cap"]["64000"]["by_level"][lv]
        d = a["contrasts"]["delta_c1_react_minus_c3_mas"]["64000"]["by_level"].get(lv, {})
        add(f"| {lv} | {_fmt(g['optimum_mean'])} | {_fmt(g['satisfaction_mean'])} | "
            f"{_fmt(g['achieved_mean'])} | {_fmt(d.get('delta_mean'))} |")
    add("\nThe baseline's own satisfaction does not fall as the level rises, which is the "
        "first sign that the axis is not ordering difficulty as intended.\n")
    add("\nThe oracle optimum falls as structural conflict rises, so the denominator of "
        "the primary metric shrinks with the level it is meant to index: one meeting is "
        "worth far more on hard than on easy. The hard level additionally contains no "
        "four-person instances and is nearly determined by `tightness`. Whether the "
        "complexity axis creates the rising-difficulty ordering the research question "
        "assumes is therefore open, and it is the decision the main experiment must "
        "settle before it runs.\n")

    add("\n## Proposal taxonomy by role — diagnostic only\n")
    add("These rates describe internal behaviour and are **not** the evidence for why one "
        "condition outscores another. A C1 proposal addresses the whole task; a C3 worker "
        "proposal addresses a decomposed sub-instance; a C3 critic proposal has already "
        "passed the aggregator gate and is close to valid by construction. Pooling them "
        "would compare generation against generation-plus-filtered-selection, and even "
        "split by role the worker and C1 numbers are not the same measurement. "
        "Establishing what causes the difference is what C4 is for.\n")
    add("| condition | cap | role | attempts | validity |")
    add("|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cap in caps:
            g = a["by_condition"][cond]["by_cap"][str(cap)]["all"]
            for role, v in g.get("by_role", {}).items():
                add(f"| {cond} | {cap} | {role} | {v['attempts']} | "
                    f"{_fmt(v['validity_rate'])} |")
    add("\n| condition | cap | pooled attempts | pooled validity | acceptance | rejection reasons |")
    add("|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cap in caps:
            g = a["by_condition"][cond]["by_cap"][str(cap)]["all"]
            add(f"| {cond} | {cap} | {g['attempts_total']} | "
                f"{_fmt(g['proposal_validity_rate'])} | "
                f"{_fmt(g['proposal_acceptance_rate'])} | "
                f"{g['rejection_reasons'] or '{}'} |")

    add("\n## What each condition actually executed\n")
    add("Counted from call roles, which record the answer directly. An earlier version of "
        "this analysis inferred C2 exposure from identical token totals; identical totals "
        "are consistent with a stage never running but do not demonstrate it.\n")
    add("| condition | cap | role | calls | calls/run | runs reached | share |")
    add("|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cap in caps:
            se = a["by_condition"][cond]["by_cap"][str(cap)]["stage_exposure"]
            for role, v in se.items():
                if role == "n_runs":
                    continue
                add(f"| {cond} | {cap} | {role} | {v['calls_total']} | "
                    f"{_fmt(v['calls_per_run'])} | {v['runs_reached']} | "
                    f"{_fmt(v['share_runs_reached'])} |")

    add("\n## Termination and budget\n")
    add("`aborted` denotes a grant that expired inside a reasoning block, leaving nothing "
        "after the closing think tag; the loop then finalises with the plan it already "
        "holds, and across the whole matrix a run holds a non-empty final plan exactly "
        "when it holds a non-empty best plan. It is collapsed with `budget` as "
        "budget-limited. That is a property of the loop and is **not** the same as the "
        "ledger's `budget_exhausted` flag, reported beside it: a condition can stop "
        "without exhausting its grant, and the two columns are given separately so no "
        "mechanism is read off one of them alone.\n")
    add("| condition | cap | termination | budget-limited | ledger exhausted | tokens left | finalisation mismatch |")
    add("|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        for cap in caps:
            g = a["by_condition"][cond]["by_cap"][str(cap)]["all"]
            add(f"| {cond} | {cap} | {g['termination']} | {g['budget_limited']} | "
                f"{g['budget_exhausted_runs']} | "
                f"{_fmt(g['tokens_remaining_mean'])} | {g['finalization_mismatch']} |")

    add("\n## Level tables\n")
    for cond in CONDITIONS:
        add(f"\n### {cond}\n")
        add("| cap | level | n | sat | achieved | optimum | optimality | attempts | runs w/ valid | budget-limited |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for cap in caps:
            for lv in _LEVELS:
                g = a["by_condition"][cond]["by_cap"][str(cap)]["by_level"][lv]
                if not g.get("n_runs"):
                    continue
                add(f"| {cap} | {lv} | {g['n_runs']} | {_fmt(g['satisfaction_mean'])} | "
                    f"{_fmt(g['achieved_mean'])} | {_fmt(g['optimum_mean'])} | "
                    f"{_fmt(g['optimality_rate'])} | {g['attempts_total']} | "
                    f"{_fmt(g['share_runs_any_valid'])} | {g['budget_limited']} |")

    add("\n## Limitations\n")
    add("- **Pilot, not confirmatory.** Ten instances per level and cap. The bootstrap "
        "intervals are correspondingly wide, and the interaction analysis is a diagnostic. "
        "No claim here is a test of the research question.")
    add("- **A non-significant interaction is not a finding of no interaction.** The "
        "omnibus and trend tests above have power only against a large effect at this "
        "sample size. Where they return a large p, the correct reading is that the pilot "
        "does not resolve whether the condition difference depends on complexity, not "
        "that it does not.")
    add("- One run per cell. Determinism removes sampling variance, so a cell is a point "
        "estimate of a deterministic system rather than a mean over repetitions.")
    add("- The hard level contains no four-person instances and is nearly determined by "
        "`tightness`, so level, instance size and that generator knob are partially "
        "confounded. Recorded rather than repaired.")
    add("- Oracle optimum falls with level, so the primary metric's denominator shrinks "
        "along the axis it indexes. Absolute meetings are reported beside satisfaction "
        "for that reason.")
    add("- Categorical levels come from this pool's own boundaries and are not comparable "
        "with the calibration sweep's levels.")
    add("- One GPU was released partway through, and the remaining shard was replayed "
        "against the other endpoint. Both served the same model revision under the same "
        "flags and cross-endpoint determinism was verified separately; the endpoint "
        "distribution is recorded in the audit block rather than treated as uniform.")
    add("- Associations between instance properties and outcomes are descriptive: the "
        "generator varies several properties together, so none is manipulated alone.\n")

    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("results/formal_pilot"))
    ap.add_argument("--expected-runs", type=Path,
                    default=Path("results/manifests/expected_runs__formal_pilot.json"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/formal_pilot"))
    args = ap.parse_args(argv)

    rows = load_runs(args.runs)
    if not rows:
        raise SystemExit(f"no run documents under {args.runs}")
    audit = integrity_audit(rows, args.expected_runs)

    doc = {
        "schema_version": "formal_pilot_analysis/2.0",
        # posix form: str(Path) uses the platform separator, so a Windows rerun would
        # rewrite this field and the artifact would look changed without any data change.
        "runs_dir": args.runs.as_posix(),
        "status": "pilot design-validation analysis, not a confirmatory test",
        "delta_convention": "delta = S(minuend) - S(subtrahend); expose primary is C1 - C3",
        "resampling": {"bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                       "permutations": PERMUTATIONS, "seed": RESAMPLE_SEED},
        "integrity_audit": audit,
        "analysis": analyse(rows),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    scalar = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with (args.out / "runs.csv").open("w", encoding="utf-8", newline="\n") as fh:
        # DictWriter defaults to CRLF on every platform; git then normalises it on one
        # machine and not the other, and the file appears to change without any data
        # changing.
        w = csv.DictWriter(fh, fieldnames=scalar + ["reasons", "invalid_reasons"],
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in scalar},
                        "reasons": ";".join(r["reasons"]),
                        "invalid_reasons": ";".join(r["invalid_reasons"])})
    write_report(doc, args.out / "report.md")

    print("integrity audit: passed — "
          f"{audit['n_runs']} runs, {audit['n_instances']} instances, "
          f"subset {audit['subset_hash'][:12]}, "
          f"executed {str(audit['provenance']['executing_commit'])[:7]} "
          f"(frozen {audit['provenance']['preregistered_code_commit'][:7]})")
    an = doc["analysis"]
    for cond in CONDITIONS:
        line = " ".join(
            f"{cap // 1000}k {_fmt(an['by_condition'][cond]['by_cap'][str(cap)]['all']['satisfaction_mean'])}"
            for cap in sorted(EXPECTED_CAPS))
        print(f"{cond:<18} sat: {line}")
    print("delta = S_C1 - S_C3 (negative = hierarchy ahead)")
    for cap in sorted(EXPECTED_CAPS):
        c = an["contrasts"]["delta_c1_react_minus_c3_mas"][str(cap)]
        it = c["interaction"]
        om, tr = it.get("omnibus"), it.get("trend")
        print(f"  cap {cap:>6}: delta {_fmt(c['delta_mean'])} CI {_ci(c['bootstrap'])}"
              f" | omnibus p {_p(om['p']) if om else 'n/a'}"
              f" | prereg decreasing-trend p "
              f"{_p(tr['p_one_sided_preregistered_decreasing']) if tr else 'n/a'}"
              f" | observed {tr['observed_direction'] if tr else 'n/a'}")
    print("the one-sided p tests the PRE-REGISTERED falling trend only; a large p reflects "
          "the pilot's size and does not establish independence from complexity")
    print(f"written: {args.out}/summary.json, runs.csv, report.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
