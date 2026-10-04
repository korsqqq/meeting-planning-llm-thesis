# scripts/hpc/size_heldout_campaign.py
"""Size the held-out Slurm array from measurements already in the repository.

    uv run python scripts/hpc/size_heldout_campaign.py

Runs NO experiment. Reads the frozen development run documents for their recorded
`latency_seconds`, converts to H200 time with the measured speed-up, and prices
the 3960-run held-out campaign under the frozen execution path: one vLLM per
shard, sequential runner invocations, one per (subset, condition) pair.

The earlier sizing (24 shards / %6 / 20h) was for the superseded 2760-run design
and is not reused. Nothing here changes the experiment -- only how it is split
across jobs.

WHERE THE INPUTS COME FROM. `scripts/hpc/README.md` §7 records the calibration
job 1891293: H200 is 2.15x the A6000 on total-token throughput, model load ~1600 s,
per-invocation harness startup ~350 s. The development runs were produced on the
A6000 machine, so their latencies are divided by that factor.

MISSING CELLS. The development sweeps do not cover every (block, cap, condition).
Two fills are reported side by side rather than one blended guess:

  mean          per-block measured value where it exists; otherwise the closest
                measured analogue, named in the printed table
  conservative  every cell takes the per-cap MAXIMUM over all measured conditions
                AND the n=8 block, which is the most expensive block. Block B is
                priced as if it were n=8.

Walltime is chosen from the conservative figure, because the cost of
underestimating is a mass timeout across the whole array.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

# Measured on the cluster, recorded in scripts/hpc/README.md §7 (job 1891293).
H200_SPEEDUP = 2.15
MODEL_LOAD_S = 1600.0
INVOCATION_STARTUP_S = 350.0
# Four frozen subsets (n=8 bands, plus n=4/5/6) x five conditions. The runner takes one
# subset and one condition per invocation, so a shard pays the ~350 s process startup
# twenty times, not five. Merging them would change the execution path, which is frozen.
SUBSETS = 4
RUNNERS_PER_SHARD = SUBSETS * 5

CAPS = (16000, 32000, 64000, 128000)
CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")

# The frozen held-out design.
BLOCK_A_INSTANCES = 150          # n = 8, three D bands x 50
BLOCK_B_INSTANCES = 48           # n = 4, 5, 6 x 16
TOTAL_INSTANCES = BLOCK_A_INSTANCES + BLOCK_B_INSTANCES
TOTAL_RUNS = TOTAL_INSTANCES * len(CAPS) * len(CONDITIONS)


def measured() -> dict[tuple[str, int, str], float]:
    """(block, cap, condition) -> mean latency in seconds on the measurement machine."""
    acc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for path in glob.glob("results/dev_runs/*/*.json"):
        rr = json.loads(Path(path).read_text(encoding="utf-8"))["run_result"]
        match = re.search(r"-n(\d+)-", rr["instance_id"])
        if not match:
            continue
        block = "A" if int(match.group(1)) == 8 else "B"
        acc[(block, rr["cap"], rr["condition"])].append(rr["latency_seconds"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def build_tables(m: dict[tuple[str, int, str], float]):
    """Return (mean_table, conservative_table, provenance) in H200 seconds."""
    mean: dict[tuple[str, int, str], float] = {}
    cons: dict[tuple[str, int, str], float] = {}
    prov: dict[tuple[str, int, str], str] = {}

    # Per-cap maximum over everything measured at that cap -- the conservative fill.
    cap_max = {
        cap: max((v for (_, c, _), v in m.items() if c == cap), default=0.0) for cap in CAPS
    }
    # Condition ratios to C1 at 64k, where all five conditions are measured on block A.
    base = m.get(("A", 64000, "c1_react"))
    ratio = {
        cond: (m[("A", 64000, cond)] / base) if ("A", 64000, cond) in m and base else 1.0
        for cond in CONDITIONS
    }

    for block in ("A", "B"):
        for cap in CAPS:
            for cond in CONDITIONS:
                key = (block, cap, cond)
                if key in m:
                    mean[key], prov[key] = m[key] / H200_SPEEDUP, "measured"
                elif ("A", cap, cond) in m:
                    # Block B unmeasured: price it as block A, which is dearer.
                    mean[key] = m[("A", cap, cond)] / H200_SPEEDUP
                    prov[key] = "block A at same cap"
                elif cap == 128000:
                    # C1 does not grow into the 128k budget while the structured
                    # conditions do, so the C1-ratio model under-predicts here.
                    # Use the measured structured analogue at 128k instead.
                    analogue = m.get(("A", 128000, "c3_mas"))
                    mean[key] = analogue / H200_SPEEDUP
                    prov[key] = "c3_mas at 128k (structured analogue)"
                elif ("A", cap, "c1_react") in m:
                    mean[key] = m[("A", cap, "c1_react")] * ratio[cond] / H200_SPEEDUP
                    prov[key] = f"c1 at cap x {ratio[cond]:.3f} (64k ratio)"
                else:
                    raise SystemExit(f"no basis to price {key}; refusing to invent one")
                cons[key] = cap_max[cap] / H200_SPEEDUP

    return mean, cons, prov


def campaign_seconds(table: dict[tuple[str, int, str], float]) -> float:
    total = 0.0
    for cap in CAPS:
        for cond in CONDITIONS:
            total += BLOCK_A_INSTANCES * table[("A", cap, cond)]
            total += BLOCK_B_INSTANCES * table[("B", cap, cond)]
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--shards", default="12,16,18,20,24,30,36")
    args = ap.parse_args()

    m = measured()
    mean_t, cons_t, prov = build_tables(m)

    print(f"held-out campaign: {TOTAL_INSTANCES} instances x {len(CAPS)} caps x "
          f"{len(CONDITIONS)} conditions = {TOTAL_RUNS} runs")
    print(f"H200 speed-up {H200_SPEEDUP}x, model load {MODEL_LOAD_S:.0f}s, "
          f"invocation startup {INVOCATION_STARTUP_S:.0f}s x {RUNNERS_PER_SHARD}\n")

    print("per-run H200 seconds, block A (n=8) -- 'mean' basis")
    print(f"  {'cap':>7} " + " ".join(f"{c.split('_')[0].upper():>7}" for c in CONDITIONS))
    for cap in CAPS:
        row = " ".join(f"{mean_t[('A', cap, c)]:>7.0f}" for c in CONDITIONS)
        print(f"  {cap:>7} {row}")
    print("\n  extrapolated cells (block A):")
    for cap in CAPS:
        for cond in CONDITIONS:
            if prov[("A", cap, cond)] != "measured":
                print(f"    {cap:>6} {cond:<20} <- {prov[('A', cap, cond)]}")

    mean_s = campaign_seconds(mean_t)
    cons_s = campaign_seconds(cons_t)
    print(f"\ncompute-only campaign cost")
    print(f"  mean basis         {mean_s / 3600:8.1f} GPU-hours "
          f"({mean_s / TOTAL_RUNS:.0f} s/run)")
    print(f"  conservative basis {cons_s / 3600:8.1f} GPU-hours "
          f"({cons_s / TOTAL_RUNS:.0f} s/run)")

    overhead = MODEL_LOAD_S + INVOCATION_STARTUP_S * RUNNERS_PER_SHARD
    print(f"\nfixed overhead per shard: {overhead:.0f}s "
          f"({overhead / 3600:.2f}h) = model load + {RUNNERS_PER_SHARD} invocations")

    print(f"\nsizing at concurrency %{args.concurrency}")
    print(f"  {'shards':>6} {'runs/sh':>8} {'mean_h':>7} {'cons_h':>7} {'ovh_h':>6} "
          f"{'waves':>6} {'campaign_h':>11} {'walltime':>9} {'margin':>7}")
    for shards in (int(s) for s in args.shards.split(",")):
        if TOTAL_RUNS % shards:
            note = f"{TOTAL_RUNS / shards:.1f}"
        else:
            note = f"{TOTAL_RUNS // shards}"
        mean_h = (mean_s / shards + overhead) / 3600
        cons_h = (cons_s / shards + overhead) / 3600
        waves = -(-shards // args.concurrency)
        campaign_h = waves * cons_h
        walltime = max(2, int(cons_h) + 1)
        while walltime / cons_h < 1.35:
            walltime += 1
        print(f"  {shards:>6} {note:>8} {mean_h:>7.1f} {cons_h:>7.1f} "
              f"{overhead / 3600:>6.2f} {waves:>6} {campaign_h:>11.1f} "
              f"{walltime:>8}h {walltime / cons_h:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
