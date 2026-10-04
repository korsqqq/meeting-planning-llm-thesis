# scripts/extract_c5_attempts.py
"""C5 best-of-3 diagnostics from the stored run documents. CPU only, no new agent runs.

    python -m scripts.extract_c5_attempts

For each of the 792 C5 runs it reads `agent.diagnostics.attempt_products` (index, seed,
plan, n_meetings per attempt) and `winner_attempt_index`.

Why n_meetings / optimum is the satisfaction of an attempt. Each product is that attempt's
`best_plan_so_far`, which the in-loop hidden validator gate has already accepted
(`src/agents/single_agent/best_of_3.py`, module docstring and `select_best`). The script
does not re-validate; it checks consistency instead and flags every run where

  * the winner index differs from the selection rule recomputed here (most meetings,
    lowest attempt index on ties), or
  * the winner's meetings differ from `run_result.score.n_valid_meetings`.

Per run (`c5_attempts_runs.csv`): attempt satisfactions, winner satisfaction,
winner - first attempt, winner - mean of the three attempts, whether selection improved on
the first attempt / on the mean, number of non-empty and distinct attempts.
Per cell (`c5_attempts_summary.csv`): Block A by cap and complexity level and pooled, Block B
by cap and size and pooled. Descriptive only: no test, no p-value.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "results/logs/heldout"
RUNS_CSV = REPO / "results/exports/heldout/runs.csv"
OUT = REPO / "results/analysis/heldout_factorial/c5_attempts"
LEVEL_OF = {"low": "Low", "medium": "Medium", "high": "High"}
N_ATTEMPTS = 3


def run_row(doc: dict, meta: dict[str, str]) -> dict[str, object]:
    """One C5 run -> attempt-level diagnostics and consistency flags."""
    diag = doc["agent"]["diagnostics"]
    products = sorted(diag["attempt_products"], key=lambda p: p["attempt_index"])
    if len(products) != N_ATTEMPTS:
        raise ValueError(f"{doc['run_id']}: {len(products)} attempt products")
    optimum = int(doc["oracle"]["optimum"])
    meetings = [int(p["n_meetings"]) for p in products]
    sat = [m / optimum for m in meetings]
    winner = int(diag["winner_attempt_index"])
    rule_winner = min(range(N_ATTEMPTS), key=lambda i: (-meetings[i], i))
    score = doc["run_result"]["score"]
    s_win = sat[winner]
    mean3 = statistics.fmean(sat)
    return {
        "run_id": doc["run_id"], "instance_id": doc["instance"]["instance_id"],
        "block": meta["block"], "cap": int(doc["cap"]),
        "complexity": LEVEL_OF.get(meta["band"], ""), "size": int(meta["n_people"]),
        "optimum": optimum,
        **{f"satisfaction_attempt_{i}": sat[i] for i in range(N_ATTEMPTS)},
        "winner_attempt_index": winner, "satisfaction_winner": s_win,
        "satisfaction_final_score": float(score["satisfaction"]),
        "winner_minus_first": s_win - sat[0], "winner_minus_mean3": s_win - mean3,
        "improved_over_first": int(s_win > sat[0]), "improved_over_mean3": int(s_win > mean3 + 1e-12),
        "n_nonempty_attempts": sum(m > 0 for m in meetings),
        "n_distinct_products": int(diag["n_distinct_products"]),
        "flag_winner_rule_mismatch": int(winner != rule_winner),
        "flag_score_mismatch": int(meetings[winner] != int(score["n_valid_meetings"])),
    }


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    def mean(key: str) -> float:
        return statistics.fmean(float(r[key]) for r in rows)  # type: ignore[arg-type]
    return {
        "n_runs": len(rows),
        "mean_satisfaction_attempt_0": mean("satisfaction_attempt_0"),
        "mean_satisfaction_mean3": statistics.fmean(
            statistics.fmean(float(r[f"satisfaction_attempt_{i}"]) for i in range(N_ATTEMPTS)) for r in rows),  # type: ignore[arg-type]
        "mean_satisfaction_winner": mean("satisfaction_winner"),
        "mean_winner_minus_first": mean("winner_minus_first"),
        "mean_winner_minus_mean3": mean("winner_minus_mean3"),
        "share_improved_over_first": mean("improved_over_first"),
        "share_improved_over_mean3": mean("improved_over_mean3"),
        "share_winner_not_attempt_0": statistics.fmean(int(r["winner_attempt_index"] != 0) for r in rows),
        "share_no_nonempty_attempt": statistics.fmean(int(r["n_nonempty_attempts"] == 0) for r in rows),
        "n_flag_winner_rule_mismatch": sum(int(r["flag_winner_rule_mismatch"]) for r in rows),  # type: ignore[arg-type]
        "n_flag_score_mismatch": sum(int(r["flag_score_mismatch"]) for r in rows),  # type: ignore[arg-type]
    }


def main() -> int:
    with RUNS_CSV.open(encoding="utf-8", newline="") as fh:
        meta = {r["run_id"]: r for r in csv.DictReader(fh) if r["condition"] == "c5_best_of_3"}
    rows = [run_row(json.loads(p.read_text(encoding="utf-8")), meta[p.stem])
            for p in sorted(RUNS_DIR.glob("c5_best_of_3__*.json"))]
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        blk, cap = str(r["block"]), str(r["cap"])
        groups[(blk, cap, "", "")].append(r)
        if blk == "A":
            groups[(blk, cap, str(r["complexity"]), "")].append(r)
        else:
            groups[(blk, cap, "", str(r["size"]))].append(r)
    summary = [{"block": b, "cap": c, "complexity": lv, "size": sz, **summarise(g)}
               for (b, c, lv, sz), g in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2], kv[0][3]))]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in (("c5_attempts_runs.csv", rows), ("c5_attempts_summary.csv", summary)):
        with (OUT / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    flags = sum(int(r["flag_winner_rule_mismatch"]) + int(r["flag_score_mismatch"]) for r in rows)  # type: ignore[arg-type]
    print(f"{len(rows)} C5 runs; {flags} consistency flag(s); wrote {OUT}")
    if len(rows) != 792:
        print("expected 792 C5 runs", file=sys.stderr)
        return 1
    return 0 if flags == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
