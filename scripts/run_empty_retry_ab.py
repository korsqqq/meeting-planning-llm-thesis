# scripts/run_empty_retry_ab.py
"""A/B on the section-2 empty-post-think rule. NOT a thesis number.

    python -m scripts.run_empty_retry_ab --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-32B-AWQ

Arm A (locked, current default): the FIRST empty post-think turn finalises the run
immediately.
Arm B (experimental): the first empty turn gets exactly one retry -- the same single
retry a non-empty formatting slip already gets; a second CONSECUTIVE empty turn
finalises as before.

The question this answers: how much of a run's budget the locked rule discards. On
Qwen3-32B-AWQ at cap 32000 one empty turn ended a run with 13 427 tokens (41% of the
budget) unspent, and that run scored 0 where its siblings scored 1.00. Whether the
rule should change is a section-2 decision; this script only measures the cost.

Design, so the two arms are comparable:
  * the SAME instance object is passed to both arms (one deterministic generation per
    seed, not one per arm);
  * every other parameter is identical -- cap, condition, model, sampling, the derived
    step cap, the finalisation reserve;
  * results are collected in memory and written to ONE document, so the two arms
    cannot overwrite each other's `<run_id>.json` (the run id has no arm in it).

Residual non-determinism at fixed seed (batching, FP reduction order) is a disclosed
confound: a difference of one run between arms is noise, not an effect.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core import DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402
from src.data import generate_instance                # noqa: E402
from src.schemas import TravelStructure               # noqa: E402

BANNER = "EMPTY-TURN A/B — NOT PILOT, NOT EXPERIMENT"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--condition", default="c1_react",
                        choices=["c1_react", "c2_verify_revise", "c3_mas"])
    parser.add_argument("--caps", default="16000,32000",
                        help="comma-separated token caps")
    parser.add_argument("--seeds", type=int, default=5,
                        help="instances per cap; seeds are 0..n-1 and identical in both arms")
    parser.add_argument("--n-people", type=int, default=4)
    parser.add_argument("--tightness", type=float, default=0.2)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--output", type=Path,
                        default=Path("results/logs/ab_empty_retry"))
    return parser


def _row(run, cap: int) -> dict:
    rr = run.run_result
    return {
        "satisfaction": rr.score.satisfaction,
        "valid": rr.score.valid,
        "tokens_used": rr.tokens.total,
        "unused_budget": cap - rr.tokens.total,
        "unused_share": round((cap - rr.tokens.total) / cap, 4),
        "empty_turns": run.agent.empty_turns,
        "termination": run.agent.termination,
        "n_steps": run.agent.n_steps,
        "budget_exhausted": rr.tokens.budget_exhausted,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    caps = [int(c) for c in args.caps.split(",") if c.strip()]

    from src.core import LLMClient, QwenTokenizer
    from src.harness import run_single_instance

    client = LLMClient(
        tokenizer=QwenTokenizer(), model=args.model, base_url=args.base_url
    )

    print("=" * 72)
    print(BANNER)
    print("A = locked rule (first empty post-think finalises)")
    print("B = one retry, second consecutive empty finalises")
    print("=" * 72)
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"condition: {args.condition}")
    print(f"caps     : {caps}   seeds: 0..{args.seeds - 1}")
    print()

    records: list[dict] = []
    for cap in caps:
        for seed in range(args.seeds):
            instance = generate_instance(
                n_people=args.n_people,
                tightness=args.tightness,
                overlap=args.overlap,
                travel_structure=TravelStructure.UNIFORM,
                seed=seed,
            )
            arms: dict[str, dict] = {}
            for arm, retry in (("A", False), ("B", True)):
                try:
                    run = run_single_instance(
                        instance=instance,
                        client=client,
                        condition=args.condition,
                        cap=cap,
                        model_label=args.model,
                        retry_on_empty=retry,
                        output_dir=None,
                        notes=f"empty-turn A/B arm {arm} ({BANNER})",
                    )
                except ValueError as exc:
                    print(f"[cap {cap} seed {seed}] skipped: {exc}")
                    arms = {}
                    break
                arms[arm] = _row(run, cap)

            if not arms:
                continue
            a, b = arms["A"], arms["B"]
            records.append({"cap": cap, "seed": seed, "A": a, "B": b})
            print(
                f"[cap {cap:>5} seed {seed}] "
                f"A sat {a['satisfaction']:.2f} used {a['tokens_used']:>6} "
                f"unused {a['unused_budget']:>6} empty {a['empty_turns']} "
                f"end {a['termination']:<12} | "
                f"B sat {b['satisfaction']:.2f} used {b['tokens_used']:>6} "
                f"unused {b['unused_budget']:>6} empty {b['empty_turns']} "
                f"end {b['termination']}"
            )

    if not records:
        print("no comparable runs — nothing to report")
        return 1

    print()
    print("=" * 72)
    print("per cap (means over seeds; 'rescued' = A scored 0 and B scored above 0)")
    print("=" * 72)
    summary = []
    for cap in caps:
        rows = [r for r in records if r["cap"] == cap]
        if not rows:
            continue
        block = {
            "cap": cap,
            "n": len(rows),
            "satisfaction_A": statistics.fmean(r["A"]["satisfaction"] for r in rows),
            "satisfaction_B": statistics.fmean(r["B"]["satisfaction"] for r in rows),
            "unused_budget_A": statistics.fmean(r["A"]["unused_budget"] for r in rows),
            "unused_budget_B": statistics.fmean(r["B"]["unused_budget"] for r in rows),
            "empty_turns_A": sum(r["A"]["empty_turns"] for r in rows),
            "empty_turns_B": sum(r["B"]["empty_turns"] for r in rows),
            "aborted_A": sum(r["A"]["termination"] == "aborted" for r in rows),
            "aborted_B": sum(r["B"]["termination"] == "aborted" for r in rows),
            # The headline: runs the retry turned from nothing into something, and --
            # reported with equal weight -- runs it made worse.
            "rescued": sum(
                r["A"]["satisfaction"] == 0 and r["B"]["satisfaction"] > 0 for r in rows
            ),
            "regressed": sum(r["B"]["satisfaction"] < r["A"]["satisfaction"] for r in rows),
        }
        summary.append(block)
        print(
            f"cap {cap:>5} (n={block['n']}): "
            f"sat A {block['satisfaction_A']:.3f} -> B {block['satisfaction_B']:.3f} | "
            f"unused A {block['unused_budget_A']:.0f} -> B {block['unused_budget_B']:.0f} | "
            f"empty turns A {block['empty_turns_A']} B {block['empty_turns_B']} | "
            f"aborted A {block['aborted_A']} B {block['aborted_B']} | "
            f"rescued {block['rescued']} regressed {block['regressed']}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "-")
    path = args.output / f"ab_empty_retry__{args.condition}__{tag}__{stamp}.json"
    document = {
        "schema_version": "ab_empty_retry/1.0",
        "banner": BANNER,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": args.base_url,
        "model": args.model,
        "condition": args.condition,
        "caps": caps,
        "seeds": list(range(args.seeds)),
        "generator": {
            "n_people": args.n_people,
            "tightness": args.tightness,
            "overlap": args.overlap,
            "travel_structure": TravelStructure.UNIFORM.value,
        },
        "runs": records,
        "summary": summary,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print()
    print(f"written: {path}")
    print("This is an A/B on a locked rule, not a model-quality measurement.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
