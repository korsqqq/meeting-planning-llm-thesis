# scripts/run_live_smoke.py
"""LIVE SMOKE ONLY -- NOT PILOT, NOT EXPERIMENT.

    python -m scripts.run_live_smoke --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-8B-AWQ --cap 2000 --n-runs 2

Thin wrapper over the existing harness: runs 1-3 easy (0-conflict) generated
instances through one condition (C1, C2 or C3) against a REAL OpenAI-compatible endpoint with the
real `LLMClient`, and writes the usual JSON documents including the
`usage_audit` (internal ledger vs endpoint-reported usage) and
`transcript_sanity` (think-leak / oracle-vocabulary) blocks.

What this is for -- the infrastructure questions that only a real endpoint can
answer: did vLLM return the expected thinking/content split, did the section-2
policy behave on real output, does the internal ledger match endpoint usage, and
is endpoint usage (or its absence) recorded honestly in the JSON. Run it against
whichever model is currently the target before any GPU sweep.

What this is NOT: a pilot, cap calibration, or any source of thesis numbers.
No result printed or logged by this script is a model-quality measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/run_live_smoke.py` from the repo root as well as -m.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core import DEFAULT_BASE_URL, DEFAULT_MODEL  # noqa: E402
from src.data import generate_instance                # noqa: E402
from src.schemas import TravelStructure               # noqa: E402

BANNER = "LIVE SMOKE ONLY — NOT PILOT, NOT EXPERIMENT"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        default="c1_react",
        choices=["c1_react", "c2_verify_revise", "c3_mas"],
        help="condition to run (C4 is intentionally not part of this step)",
    )
    parser.add_argument("--cap", type=int, default=2000, help="token budget cap")
    parser.add_argument("--seed", type=int, default=0,
                        help="first generator seed; run i uses seed + i")
    parser.add_argument("--n-runs", type=int, default=1, choices=[1, 2, 3],
                        help="how many instances to run (smoke = 1-3, no more)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="served model name, e.g. Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/logs/live_smoke"))
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="deployment context window; every request is clamped to "
                             "min(budget grant, window - input). Lower it deliberately "
                             "to exercise the guard against a live endpoint.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="step cap; default derives it from --cap so the budget, "
                             "not the counter, is what ends the run")
    parser.add_argument("--n-people", type=int, default=4)
    parser.add_argument("--tightness", type=float, default=0.2)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument(
        "--require-endpoint-usage",
        action="store_true",
        help="exit non-zero if any run comes back without endpoint-reported usage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_model_len <= 0:
        raise SystemExit("--max-model-len must be a positive number of tokens")

    print("=" * 72)
    print(BANNER)
    print("Real endpoint, but smoke-scale only: results are for the manual")
    print("checkpoint (thinking split, section-2 behaviour, ledger vs usage),")
    print("never for thesis numbers.")
    print("=" * 72)
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"context  : max_model_len {args.max_model_len}")

    # Imported here so `import scripts.run_live_smoke` never builds a client or
    # downloads tokenizer files (tests import this module offline).
    from src.core import LLMClient, QwenTokenizer
    from src.harness import run_single_instance

    client = LLMClient(
        tokenizer=QwenTokenizer(), model=args.model, base_url=args.base_url,
        max_model_len=args.max_model_len,
    )

    missing_usage = 0
    for i in range(args.n_runs):
        seed = args.seed + i
        instance = generate_instance(
            n_people=args.n_people,
            tightness=args.tightness,
            overlap=args.overlap,
            travel_structure=TravelStructure.UNIFORM,
            seed=seed,
        )
        try:
            run = run_single_instance(
                instance=instance,
                client=client,
                condition=args.condition,
                cap=args.cap,
                model_label=args.model,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                notes=f"live smoke ({BANNER}); endpoint={args.base_url}; "
                      f"max_model_len={args.max_model_len}",
            )
        except ValueError as exc:
            # The harness refuses conflicted instances without pilot binning --
            # the smoke stays on easy/0-conflict instances by design.
            print(f"[seed {seed}] skipped: {exc}")
            continue

        rr = run.run_result
        totals = run.usage_audit["totals"]
        endpoint_total = totals["endpoint_total_tokens"]
        delta = totals["delta_total_tokens"]
        if not run.usage_audit["has_endpoint_usage"]:
            missing_usage += 1
        print(f"[seed {seed}] run_id           : {rr.run_id}")
        print(f"[seed {seed}] score            : valid={rr.score.valid} "
              f"satisfaction={rr.score.satisfaction:.2f}")
        print(f"[seed {seed}] internal tokens  : {totals['internal_total_tokens']} "
              f"(cap {rr.cap}, exhausted={rr.tokens.budget_exhausted})")
        print(f"[seed {seed}] endpoint tokens  : "
              f"{endpoint_total if endpoint_total is not None else 'MISSING'}")
        print(f"[seed {seed}] delta            : "
              f"{delta if delta is not None else 'n/a'}")
        print(f"[seed {seed}] think leak       : {run.sanity['think_leak']}")
        print(f"[seed {seed}] json             : {run.json_path}")

    if args.require_endpoint_usage and missing_usage:
        print(f"FAIL: {missing_usage} run(s) without endpoint-reported usage "
              "(--require-endpoint-usage set)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
