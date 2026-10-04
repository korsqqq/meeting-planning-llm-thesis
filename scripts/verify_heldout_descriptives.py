# scripts/verify_heldout_descriptives.py
"""Independent recomputation of the descriptive statistics, as a gate before LaTeX. CPU only.

    python -m scripts.verify_heldout_descriptives

Uses only the Python standard library (csv, json, statistics) -- no numpy, no pandas -- and
none of the code of `analyse_heldout_factorial`. It recomputes N, mean, median, SD,
variance, min, max, Q1, Q3 and IQR from `results/exports/heldout/runs.csv` and compares
them with `results/analysis/heldout_factorial/describe/descriptives.csv`.

Quartiles use `statistics.quantiles(method="inclusive")`, which is the linear interpolation
numpy uses by default. Bootstrap intervals are not recomputed (they are random by
construction); only their ordering around the mean is checked.

Exit code 0 when every compared value agrees within 1e-9, 1 otherwise. Mismatches are
written to `describe/verification_mismatches.csv`.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_CSV = REPO / "results/exports/heldout/runs.csv"
DESCRIBE_CSV = REPO / "results/analysis/heldout_factorial/describe/descriptives.csv"
TOL = 1e-9

ARCH_OF = {"c1_react": "C1", "c2_verify_revise": "C2", "c3_mas": "C3",
           "c4_planner_critic": "C4", "c5_best_of_3": "C5"}
LEVEL_OF = {"low": "Low", "medium": "Medium", "high": "High"}
FIELDS = ("n", "mean", "median", "sd", "variance", "min", "max", "q1", "q3", "iqr")


def stats_of(values: list[float]) -> dict[str, float]:
    """The ten descriptive fields, standard library only."""
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    sd = statistics.stdev(values)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "sd": sd, "variance": statistics.variance(values), "min": min(values), "max": max(values),
            "q1": q1, "q3": q3, "iqr": q3 - q1}


def load() -> list[dict[str, object]]:
    rows = []
    with RUNS_CSV.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            details = json.loads(r["details_json"])
            valid, meetings = int(r["valid"]), int(r["n_meetings"])
            rows.append({
                "block": r["block"], "arch": ARCH_OF[r["condition"]], "cap": int(r["cap"]),
                "task": r["instance_id"], "complexity": LEVEL_OF.get(r["band"], ""),
                "size": int(r["n_people"]),
                "satisfaction": float(r["satisfaction"]),
                "valid_nonempty": 1.0 if (valid == 1 and meetings > 0) else 0.0,
                "optimal": 1.0 if details["score"]["optimality"] else 0.0,
            })
    return rows


def expected_cells(rows: list[dict[str, object]], outcome: str) -> dict[tuple[str, ...], list[float]]:
    """Cell key (block, scope, architecture, cap, complexity, size) -> values."""
    cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    task_means: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for r in rows:
        v = float(r[outcome])  # type: ignore[arg-type]
        a, cap, blk = str(r["arch"]), str(r["cap"]), str(r["block"])
        cells[(blk, "architecture_cap", a, cap, "", "")].append(v)
        if blk == "A":
            cells[("A", "architecture_cap_complexity", a, cap, str(r["complexity"]), "")].append(v)
            task_means[("marginal_architecture", a, "", "", str(r["task"]))].append(v)
            task_means[("marginal_cap", "", cap, "", str(r["task"]))].append(v)
            task_means[("marginal_complexity", "", "", str(r["complexity"]), str(r["task"]))].append(v)
        else:
            cells[("B", "architecture_cap_size", a, cap, "", str(r["size"]))].append(v)
    for (scope, a, cap, level, _task), vals in task_means.items():
        cells[("A", scope, a, cap, level, "")].append(statistics.fmean(vals))
    return cells


def main() -> int:
    if not DESCRIBE_CSV.exists():
        print(f"missing {DESCRIBE_CSV}; run the describe step first", file=sys.stderr)
        return 1
    rows = load()
    with DESCRIBE_CSV.open(encoding="utf-8", newline="") as fh:
        described = list(csv.DictReader(fh))
    mismatches, compared, unmatched = [], 0, 0
    expected = {o: expected_cells(rows, o) for o in ("satisfaction", "valid_nonempty", "optimal")}
    seen: set[tuple[str, ...]] = set()
    for d in described:
        key = (d["block"], d["scope"], d["architecture"], d["cap"], d["complexity"], d["size"])
        vals = expected[d["outcome"]].get(key)
        if vals is None:
            unmatched += 1
            mismatches.append({"outcome": d["outcome"], "cell": "|".join(key), "field": "cell",
                               "expected": "absent", "found": "present"})
            continue
        seen.add((d["outcome"], *key))
        ref = stats_of(vals)
        for f in FIELDS:
            compared += 1
            if abs(float(d[f]) - float(ref[f])) > TOL:
                mismatches.append({"outcome": d["outcome"], "cell": "|".join(key), "field": f,
                                   "expected": ref[f], "found": d[f]})
        if not float(d["ci_low"]) <= float(d["mean"]) + TOL or not float(d["mean"]) <= float(d["ci_high"]) + TOL:
            mismatches.append({"outcome": d["outcome"], "cell": "|".join(key), "field": "ci_order",
                               "expected": "ci_low <= mean <= ci_high", "found": f"{d['ci_low']},{d['ci_high']}"})
    for outcome, cells in expected.items():
        for key in cells:
            if (outcome, *key) not in seen:
                mismatches.append({"outcome": outcome, "cell": "|".join(key), "field": "cell",
                                   "expected": "present", "found": "absent"})
    out = DESCRIBE_CSV.parent / "verification_mismatches.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["outcome", "cell", "field", "expected", "found"])
        w.writeheader()
        w.writerows(mismatches)
    print(f"compared {compared} values in {len(described)} cells; {len(mismatches)} mismatch(es); "
          f"{unmatched} unmatched cell(s) -> {out}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
