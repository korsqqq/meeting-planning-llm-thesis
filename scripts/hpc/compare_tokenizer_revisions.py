# scripts/hpc/compare_tokenizer_revisions.py
"""Gate: does pinning the counting tokenizer change what the harness counts?

`run_pilot_sweep.py` builds `QwenTokenizer()` with no revision, so every run so
far has counted input tokens with whatever `Qwen/Qwen3-8B-AWQ` was current on the
Hub. THESIS_DECISIONS section 2 requires the revision to be pinned. Pinning is
only safe if the pinned revision renders **byte-identical** chat-template strings
and **identical** token-id sequences to what is being used now -- otherwise the
budget accounting shifts and the held-out run stops being comparable to the pilot.

This compares the two revisions over the same five representative prompt shapes
the parity test uses. They are imported from `tests/test_tokenizer.py` rather than
copied, so this gate cannot drift away from the documented suite.

Run it BEFORE changing anything, with network access:

    python scripts/hpc/compare_tokenizer_revisions.py \\
        --pinned 4da05a8edb55c6046cce958586c33b61da07bb79

Exit codes: 0 identical (safe to pin), 7 a difference was found (STOP), 1 error.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.core.tokenizer import DEFAULT_TOKENIZER, QwenTokenizer  # noqa: E402

EXIT_DIFFERENT = 7


def load_parity_cases() -> dict:
    """Import the five representative shapes straight out of the test module."""
    path = REPO_ROOT / "tests" / "test_tokenizer.py"
    spec = importlib.util.spec_from_file_location("_parity_src", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._PARITY_CASES


def resolve_latest(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception as exc:                          # pragma: no cover - network probe
        print(f"  (could not resolve the current Hub head: {exc})")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_TOKENIZER)
    parser.add_argument("--pinned", required=True, help="the revision we intend to freeze")
    parser.add_argument("--baseline", default=None,
                        help="revision to compare against (default: current Hub head, "
                             "which is what the unpinned runner resolves to today)")
    args = parser.parse_args()

    cases = load_parity_cases()
    latest_sha = resolve_latest(args.repo)

    print("=" * 70)
    print("COUNTING-TOKENIZER PINNING GATE")
    print(f"  repo             : {args.repo}")
    print(f"  pinned (target)  : {args.pinned}")
    print(f"  baseline         : {args.baseline or 'current Hub head (unpinned behaviour)'}")
    print(f"  Hub head resolves: {latest_sha or 'unknown'}")
    print("=" * 70)

    if args.baseline is None and latest_sha == args.pinned:
        print("\n  NOTE: the Hub head IS the pinned revision, so today's unpinned runner")
        print("        already resolves to it. The comparison below is then a tautology;")
        print("        it still runs, and still has to pass.")

    baseline_tok = QwenTokenizer(args.repo, revision=args.baseline)
    pinned_tok = QwenTokenizer(args.repo, revision=args.pinned)

    differences = []
    for name in sorted(cases):
        spec = cases[name]
        kwargs = {
            "tools": spec.get("tools"),
            "enable_thinking": spec.get("enable_thinking", True),
        }
        rendered_b = baseline_tok.render_input(spec["messages"], **kwargs)
        rendered_p = pinned_tok.render_input(spec["messages"], **kwargs)
        ids_b = baseline_tok.encode_input(spec["messages"], **kwargs)
        ids_p = pinned_tok.encode_input(spec["messages"], **kwargs)

        same_text = rendered_b == rendered_p
        same_ids = ids_b == ids_p
        flag = "OK  " if (same_text and same_ids) else "DIFF"
        print(f"  [{flag}] {name:<24} ids {len(ids_b):>5} vs {len(ids_p):>5}"
              f"   text {'same' if same_text else 'DIFFERENT'}"
              f"   ids {'same' if same_ids else 'DIFFERENT'}")

        if not (same_text and same_ids):
            entry = {"case": name, "text_same": same_text, "ids_same": same_ids}
            if not same_text:
                for i, (cb, cp) in enumerate(zip(rendered_b, rendered_p)):
                    if cb != cp:
                        entry["first_text_diff_at"] = i
                        entry["baseline_excerpt"] = rendered_b[max(0, i - 40):i + 40]
                        entry["pinned_excerpt"] = rendered_p[max(0, i - 40):i + 40]
                        break
                else:
                    entry["first_text_diff_at"] = min(len(rendered_b), len(rendered_p))
                    entry["note"] = "one rendering is a prefix of the other"
            if not same_ids:
                entry["baseline_len"] = len(ids_b)
                entry["pinned_len"] = len(ids_p)
            differences.append(entry)

    print("=" * 70)
    if differences:
        print(f"  RESULT: {len(differences)} case(s) DIFFER -- do NOT pin, stop and report")
        for entry in differences:
            print(f"    - {entry}")
        return EXIT_DIFFERENT

    print("  RESULT: identical on every case -- pinning changes no counted token")
    print(f"  Record this revision as the frozen counting tokenizer: {args.pinned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
