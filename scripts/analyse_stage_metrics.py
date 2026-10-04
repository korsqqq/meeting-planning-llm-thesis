# scripts/analyse_stage_metrics.py
"""Descriptive stage tables over the held-out extraction. CPU, local, no cluster.

    .venv-harness/bin/python -m scripts.analyse_stage_metrics \
        --stages results/exports/heldout_stages/stage_metrics.csv \
        --proposals results/exports/heldout_stages/proposal_diagnostics.csv \
        --out results/analysis/heldout_stages

WHAT THIS IS. Counts and means over the committed extraction. **No hypothesis test,
no p-value, no correction, no interval and no significance claim.** Nothing here
establishes a cause. Where a stage is followed by a better plan this is called the
**observed gain of the stage**, never its effect: the runs were not assigned to a
counterfactual and no comparison identifies what would have happened without it.

EVERYTHING IS COUNTED IN DISTINCT RUNS. The extraction writes one row per stage, and a
stage can repeat within a run -- C2 enters `revise` up to twice, so 677 revise rows
carrying a proposal come from only 354 distinct runs. Every count below goes through a
set of `run_id`, and the tables say `runs` where they mean runs.

TWO DENOMINATORS, BOTH REPORTED, NEITHER CORRECTING THE OTHER. The share of critic runs
that improved and the share among runs where improvement was structurally possible
answer different questions: the first describes the stage as it was actually invoked,
the second isolates the cases where it had anything to work with. Assessing the
architecture needs both, and neither is a repaired version of the other.

STRATA. Block A is reported by conflict-density band and Block B by task size, with a
pooled row kept alongside. The pooled row is an overview; the complexity dependence the
thesis asks about lives in the strata, and pooling can hide it.

MISSING IS NOT ZERO. A stage that ran but recorded no quality is an unknown and stops
the calculation rather than being counted as no improvement. A stage that never ran
contributes a real zero to the all-runs mean, and only there.

ON `evidence_travel_coverage`. In these data coverage falls as pool size rises, and
pool size bounds headroom, so a cross-tabulation of improvement against coverage alone
reproduces the headroom axis inverted. That is an observation about this sample, not a
mathematical identity; coverage is therefore reported inside the headroom strata.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "heldout_stage_tables/2.0"
CAPS = ("16000", "32000", "64000", "128000")
CAP_LABEL = {"16000": "16k", "32000": "32k", "64000": "64k", "128000": "128k"}
CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic",
              "c5_best_of_3")
CRITIC_CONDITIONS = ("c3_mas", "c4_planner_critic")
BANDS = ("low", "medium", "high")
SIZES = ("4", "5", "6")
POOLED = "all"
UNKNOWN = ""


class AnalysisRefused(RuntimeError):
    """An input is missing where the calculation needs it. Never silently defaulted."""


def strata(block: str) -> list[tuple[str, Callable[[dict[str, str]], bool]]]:
    """Pooled first, then the axis that matters for the block."""
    if block == "A":
        return [(POOLED, lambda m: True)] + [
            (b, lambda m, b=b: m["band"] == b) for b in BANDS]
    return [(POOLED, lambda m: True)] + [
        (f"n={n}", lambda m, n=n: m["n_people"] == n) for n in SIZES]


class Runs:
    """The extraction, indexed by run. Every count is over these keys, never rows."""

    def __init__(self, stage_rows: Sequence[dict[str, str]],
                 proposal_rows: Sequence[dict[str, str]]):
        self.stages: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in stage_rows:
            self.stages[row["run_id"]].setdefault(row["stage_role"], row)
        self.meta = {run: next(iter(by_role.values()))
                     for run, by_role in self.stages.items()}
        self.with_proposal = {row["run_id"] for row in proposal_rows}

    def select(self, condition: str, cap: str, block: str,
               keep: Callable[[dict[str, str]], bool]) -> list[str]:
        return [run for run, m in self.meta.items()
                if m["condition"] == condition and m["cap"] == cap
                and m["block"] == block and keep(m)]

    def stage(self, run: str, role: str) -> dict[str, str] | None:
        return self.stages[run].get(role)


def _int(value: str | None) -> int | None:
    if value in (None, UNKNOWN):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    if value in (None, UNKNOWN):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> Any:
    values = list(values)
    return round(statistics.fmean(values), 3) if values else UNKNOWN


def required(stage: dict[str, str], field: str, run: str) -> float:
    """Read a field a ran stage must carry, or refuse.

    The earlier version of this file wrote `_int(x) or 0`, which turned an unknown into
    a measured zero the moment one appeared. There are none today; the guard is what
    keeps that true when the extraction changes.
    """
    value = _float(stage[field])
    if value is None:
        raise AnalysisRefused(
            f"{run}: stage {stage['stage_role']!r} ran but recorded no {field!r}. "
            "An unknown is not zero, and this calculation will not assume one.")
    return value


# --------------------------------------------------------------------------- #
# 1. Runs that never proposed.
# --------------------------------------------------------------------------- #
def table_no_proposal(runs: Runs) -> list[dict[str, Any]]:
    out = []
    for block in ("A", "B"):
        for stratum, keep in strata(block):
            for condition in CONDITIONS:
                for cap in CAPS:
                    selected = runs.select(condition, cap, block, keep)
                    if not selected:
                        continue
                    silent = [r for r in selected if r not in runs.with_proposal]
                    out.append({
                        "block": block, "stratum": stratum, "condition": condition,
                        "cap": cap, "runs": len(selected),
                        "runs_without_any_proposal": len(silent),
                        "share": round(len(silent) / len(selected), 3),
                    })
    return out


# --------------------------------------------------------------------------- #
# 2. What became of the critic.
# --------------------------------------------------------------------------- #
def critic_fate(runs: Runs, run: str, condition: str) -> str:
    stage = runs.stage(run, "critic")
    if stage is None:
        return "not_reached"
    if (_int(stage["n_proposals"]) or 0) == 0:
        return "reached_no_proposal"
    if condition == "c4_planner_critic" and stage["critic_improved"] != UNKNOWN:
        improved = stage["critic_improved"] == "1"          # recorded by the harness
    else:
        improved = required(stage, "meetings_added", run) > 0   # reconstructed
    return "improved" if improved else "proposed_not_improved"


def table_critic_outcomes(runs: Runs) -> list[dict[str, Any]]:
    out = []
    for block in ("A", "B"):
        for stratum, keep in strata(block):
            for condition in CRITIC_CONDITIONS:
                for cap in CAPS:
                    selected = runs.select(condition, cap, block, keep)
                    if not selected:
                        continue
                    fates = [critic_fate(runs, run, condition) for run in selected]
                    counts = {f: fates.count(f) for f in
                              ("not_reached", "reached_no_proposal",
                               "proposed_not_improved", "improved")}
                    ran = len(selected) - counts["not_reached"]
                    out.append({
                        "block": block, "stratum": stratum, "condition": condition,
                        "cap": cap, "runs": len(selected), **counts,
                        "improved_share_of_runs":
                            round(counts["improved"] / len(selected), 3),
                        "improved_share_of_reached":
                            (round(counts["improved"] / ran, 3) if ran else UNKNOWN),
                        "source": ("recorded" if condition == "c4_planner_critic"
                                   else "reconstructed"),
                    })
    return out


# --------------------------------------------------------------------------- #
# 3. The observed gain of the stage, and its price.
# --------------------------------------------------------------------------- #
def table_critic_gain(runs: Runs) -> list[dict[str, Any]]:
    out = []
    for block in ("A", "B"):
        for stratum, keep in strata(block):
            for condition in CRITIC_CONDITIONS:
                for cap in CAPS:
                    selected = runs.select(condition, cap, block, keep)
                    if not selected:
                        continue
                    ran = [(run, runs.stage(run, "critic")) for run in selected]
                    ran = [(run, s) for run, s in ran if s is not None]
                    added = [required(s, "meetings_added", run) for run, s in ran]
                    # Satisfaction is a ratio to a per-instance optimum, so the mean of
                    # per-run differences is the quantity, not a difference of means.
                    sat_delta = [required(s, "sat_delta", run) for run, s in ran]
                    n = len(selected)
                    out.append({
                        "block": block, "stratum": stratum, "condition": condition,
                        "cap": cap, "runs": n, "runs_critic_ran": len(ran),
                        "mean_meetings_before": _mean(
                            required(s, "best_meetings_before", run) for run, s in ran),
                        "mean_meetings_after": _mean(
                            required(s, "best_meetings_after", run) for run, s in ran),
                        # Both denominators, neither correcting the other.
                        "mean_added_when_critic_ran": _mean(added),
                        "mean_added_over_all_runs": round(sum(added) / n, 3),
                        "mean_sat_gain_when_critic_ran": _mean(sat_delta),
                        "mean_sat_gain_over_all_runs": round(sum(sat_delta) / n, 4),
                        "mean_critic_tokens": _mean(
                            required(s, "stage_tokens", run) for run, s in ran),
                        "mean_critic_latency_s": _mean(
                            required(s, "latency_seconds", run) for run, s in ran),
                        "source": ("recorded" if condition == "c4_planner_critic"
                                   else "reconstructed"),
                    })
    return out


# --------------------------------------------------------------------------- #
# 4. What room the critic had, for both architectures.
# --------------------------------------------------------------------------- #
def opportunity_row(runs: Runs, run: str, condition: str) -> dict[str, Any] | None:
    """The room a run left its critic, from whichever stage records it.

    C4 records `pool_size` and `fallback_size` in its diagnostics. C3 records neither,
    and both are reconstructed onto its synthesised `aggregate` stage. `headroom` is
    `pool_size - fallback_size`: candidates the critic could add on top of the fallback.

    A positive headroom means further candidates existed, **not** that a longer feasible
    plan existed: the extra people still have to fit the travel and window constraints
    together with the ones already scheduled. `room_to_optimum` says how much the oracle
    proves is attainable on the instance as a whole, which the pool may not contain
    either. Neither quantity is a claim that an improvement was available.
    """
    source = "critic" if condition == "c4_planner_critic" else "aggregate"
    stage = runs.stage(run, source)
    if stage is None:
        return None
    pool, fallback = _int(stage["pool_size"]), _int(stage["fallback_size"])
    optimum = _int(runs.meta[run]["optimum"])
    if pool is None or fallback is None or optimum is None:
        return None
    return {"pool_size": pool, "fallback_size": fallback,
            "headroom": pool - fallback, "room_to_optimum": optimum - fallback,
            "fallback_at_optimum": fallback >= optimum,
            "coverage": _float(stage["evidence_travel_coverage"])}


def table_opportunity(runs: Runs) -> list[dict[str, Any]]:
    out = []
    for block in ("A", "B"):
        for stratum, keep in strata(block):
            for condition in CRITIC_CONDITIONS:
                for cap in CAPS:
                    selected = [r for r in runs.select(condition, cap, block, keep)
                                if runs.stage(r, "critic") is not None]
                    if not selected:
                        continue
                    facts = {r: opportunity_row(runs, r, condition) for r in selected}
                    facts = {r: f for r, f in facts.items() if f is not None}
                    if not facts:
                        continue
                    for label, keep_stratum in (
                            ("headroom = 0", lambda f: f["headroom"] == 0),
                            ("headroom >= 1", lambda f: f["headroom"] >= 1)):
                        group = {r: f for r, f in facts.items() if keep_stratum(f)}
                        if not group:
                            continue
                        improved = sum(
                            1 for r in group
                            if critic_fate(runs, r, condition) == "improved")
                        out.append({
                            "block": block, "stratum": stratum,
                            "condition": condition, "cap": cap,
                            "headroom_stratum": label,
                            "runs_critic_ran": len(group), "improved": improved,
                            "improved_share": round(improved / len(group), 3),
                            "mean_pool_size": _mean(f["pool_size"]
                                                    for f in group.values()),
                            "mean_fallback_size": _mean(f["fallback_size"]
                                                        for f in group.values()),
                            "mean_headroom": _mean(f["headroom"]
                                                   for f in group.values()),
                            "mean_room_to_optimum": _mean(f["room_to_optimum"]
                                                          for f in group.values()),
                            "share_fallback_at_optimum": round(sum(
                                1 for f in group.values()
                                if f["fallback_at_optimum"]) / len(group), 3),
                            "mean_travel_coverage": _mean(
                                f["coverage"] for f in group.values()
                                if f["coverage"] is not None),
                            "source": ("recorded"
                                       if condition == "c4_planner_critic"
                                       else "reconstructed"),
                        })
    return out


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[str]:
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return out


def render(tables: dict[str, list[dict[str, Any]]]) -> str:
    L: list[str] = []
    add = L.append
    add("# Held-out stage tables — descriptive")
    add("")
    add("Counts and means over the committed stage extraction. **No hypothesis test, "
        "p-value, correction, interval or significance claim.** Where a stage is "
        "followed by a better plan this is the **observed gain of the stage**, never "
        "its effect: nothing here identifies what would have happened without it.")
    add("")
    add("Every count is over distinct runs. Block A is broken out by conflict-density "
        "band and Block B by task size; the `all` row pools them and is an overview "
        "only.")
    add("")

    add("## 1. Runs that never produced a proposal")
    add("")
    add("The outcome that dominates the low caps. Reported as an outcome, not as the "
        "established cause of the low scores.")
    add("")
    for block in ("A", "B"):
        for stratum, _ in strata(block):
            rows = [r for r in tables["no_proposal"]
                    if r["block"] == block and r["stratum"] == stratum]
            if not rows:
                continue
            add(f"**Block {block} — {stratum}**")
            add("")
            by_condition = defaultdict(dict)
            for r in rows:
                by_condition[r["condition"]][r["cap"]] = r
            add("| condition | " + " | ".join(CAP_LABEL[c] for c in CAPS) + " |")
            add("|---|" + "---|" * len(CAPS))
            for condition in CONDITIONS:
                cells = [(f"{r['runs_without_any_proposal']}/{r['runs']}"
                          if (r := by_condition[condition].get(cap)) else "—")
                         for cap in CAPS]
                add(f"| {condition} | " + " | ".join(cells) + " |")
            add("")

    add("## 2. What became of the critic")
    add("")
    add("Four mutually exclusive fates per run, summing to the runs in the cell. C4 "
        "reads the harness's own `critic_improved`; C3 records no such field and is "
        "reconstructed, which the `source` column states.")
    add("")
    add("Two shares are given. `improved_share_of_runs` describes the architecture as "
        "it ran; `improved_share_of_reached` describes the stage where it was actually "
        "invoked. They answer different questions and neither replaces the other.")
    add("")
    for block in ("A", "B"):
        rows = [r for r in tables["critic_outcomes"] if r["block"] == block]
        add(f"**Block {block}**")
        add("")
        for line in md_table(rows, ["stratum", "condition", "cap", "runs",
                                    "not_reached", "reached_no_proposal",
                                    "proposed_not_improved", "improved",
                                    "improved_share_of_runs",
                                    "improved_share_of_reached", "source"]):
            add(line)
        add("")

    add("## 3. The observed gain of the critic stage, and its price")
    add("")
    add("`mean_sat_gain_*` averages per-run differences in satisfaction rather than "
        "differencing two means: satisfaction is a ratio to a per-instance optimum and "
        "the optima differ across instances. As in table 2, the two denominators are "
        "both reported.")
    add("")
    for block in ("A", "B"):
        rows = [r for r in tables["critic_gain"] if r["block"] == block]
        add(f"**Block {block}**")
        add("")
        for line in md_table(rows, ["stratum", "condition", "cap", "runs",
                                    "runs_critic_ran", "mean_meetings_before",
                                    "mean_meetings_after",
                                    "mean_added_when_critic_ran",
                                    "mean_added_over_all_runs",
                                    "mean_sat_gain_when_critic_ran",
                                    "mean_sat_gain_over_all_runs",
                                    "mean_critic_tokens", "mean_critic_latency_s",
                                    "source"]):
            add(line)
        add("")

    add("## 4. What room the critic had")
    add("")
    add("`headroom = pool_size − fallback_size`: candidates the critic could add on top "
        "of the fallback. `room_to_optimum = optimum − fallback_size`: how much more "
        "the oracle proves attainable on the instance.")
    add("")
    add("**A positive headroom is not an available improvement.** Extra candidates still "
        "have to fit the travel and window constraints alongside the people already "
        "scheduled, and the pool need not contain a set that reaches the optimum. The "
        "table therefore says what room existed, not what was achievable.")
    add("")
    add("Where headroom is zero the critic cannot improve at all, so those runs are "
        "separated rather than averaged in. `mean_travel_coverage` sits inside the "
        "strata because in these data coverage falls as pool size rises — an "
        "observation about this sample, not an identity — so a cross-tabulation "
        "against coverage alone would reproduce the headroom axis inverted.")
    add("")
    for block in ("A", "B"):
        rows = [r for r in tables["opportunity"] if r["block"] == block]
        add(f"**Block {block}**")
        add("")
        for line in md_table(rows, ["stratum", "condition", "cap",
                                    "headroom_stratum", "runs_critic_ran", "improved",
                                    "improved_share", "mean_pool_size",
                                    "mean_fallback_size", "mean_headroom",
                                    "mean_room_to_optimum",
                                    "share_fallback_at_optimum",
                                    "mean_travel_coverage", "source"]):
            add(line)
        add("")

    add("---")
    add("")
    add("*C2's per-cycle detail is limited by 26 stage rows whose proposals could not "
        "be assigned to a cycle unambiguously; those carry no quality and are not "
        "counted as zero. C2's end-to-end quality is unaffected. Nothing in these "
        "tables shows why one architecture spends its budget better than another.*")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", type=Path,
                    default=Path("results/exports/heldout_stages/stage_metrics.csv"))
    ap.add_argument("--proposals", type=Path,
                    default=Path("results/exports/heldout_stages/"
                                 "proposal_diagnostics.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/analysis/heldout_stages"))
    args = ap.parse_args(argv)

    with args.stages.open(encoding="utf-8", newline="") as fh:
        stage_rows = list(csv.DictReader(fh))
    with args.proposals.open(encoding="utf-8", newline="") as fh:
        proposal_rows = list(csv.DictReader(fh))
    runs = Runs(stage_rows, proposal_rows)

    tables = {
        "no_proposal": table_no_proposal(runs),
        "critic_outcomes": table_critic_outcomes(runs),
        "critic_gain": table_critic_gain(runs),
        "opportunity": table_opportunity(runs),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.out / f"{name}.csv", rows)
    (args.out / "report.md").write_text(render(tables), encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover
        pass
    print(f"{len(runs.meta)} runs, {len(stage_rows)} stage rows, "
          f"{len(proposal_rows)} proposal rows")
    for name, rows in tables.items():
        print(f"  {name:<16} {len(rows)} rows")
    print("\ndescriptive only: no test, no p-value, no interval")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
