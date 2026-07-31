# scripts/run_vertical_slice.py
"""Offline vertical-slice smoke: generator -> oracle -> C1 -> scorer -> JSON.

    python -m scripts.run_vertical_slice
    python scripts/run_vertical_slice.py --condition c1_react --cap 4000 --seed 0

This runs ONE deterministic generated instance through one condition with the
OFFLINE SCRIPTED CLIENT (src/harness/smoke_client.py) and writes one JSON log
under --output-dir. No model is involved: the run proves the plumbing, not model
behaviour, and does NOT close the live Qwen/vLLM checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_vertical_slice.py` from the repo root as well as -m.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import generate_instance                      # noqa: E402
from src.harness import OfflineSmokeClient, run_single_instance  # noqa: E402
from src.schemas import TravelStructure                     # noqa: E402

MODEL_LABEL = "offline-smoke-client"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        default="c1_react",
        choices=["c1_react", "c2_verify_revise"],
        help="condition to run (C3/C4 are intentionally not part of this step)",
    )
    parser.add_argument("--cap", type=int, default=4000, help="token budget cap")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    parser.add_argument("--n-people", type=int, default=4)
    parser.add_argument("--tightness", type=float, default=0.2)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument(
        "--structure",
        default="uniform",
        choices=[s.value for s in TravelStructure],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/logs/vertical_slice"),
        help="directory for the JSON log (git-ignored by default)",
    )
    parser.add_argument("--max-steps", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print("=" * 72)
    print("OFFLINE SCRIPTED-CLIENT SMOKE -- no live model.")
    print("This validates the harness plumbing only; it does NOT close the")
    print("live Qwen/vLLM checkpoint and is not evidence about model behaviour.")
    print("=" * 72)

    instance = generate_instance(
        n_people=args.n_people,
        tightness=args.tightness,
        overlap=args.overlap,
        travel_structure=TravelStructure(args.structure),
        seed=args.seed,
    )
    run = run_single_instance(
        instance=instance,
        client=OfflineSmokeClient(),
        condition=args.condition,
        cap=args.cap,
        model_label=MODEL_LABEL,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        notes=(
            "offline scripted-client smoke (plumbing only); sampling config is the "
            "nominal LLMClient profile, no model was called"
        ),
    )

    rr = run.run_result
    n_meetings = len(rr.final_plan.meetings) if rr.final_plan is not None else 0
    print(f"instance        : {rr.instance_id} (level={rr.level.value}, "
          f"conflicts={run.instance.complexity_metric})")
    print(f"oracle          : optimum={run.oracle.optimum} status={run.oracle.status} "
          f"({run.oracle_latency_seconds:.2f}s)")
    print(f"agent           : {n_meetings} meeting(s) emitted, "
          f"{rr.tokens.n_calls} call(s), {rr.tokens.total} pseudo-tokens of cap {rr.cap}")
    print(f"score           : valid={rr.score.valid} "
          f"satisfaction={rr.score.satisfaction:.2f} optimality={rr.score.optimality}")
    print(f"json written to : {run.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
