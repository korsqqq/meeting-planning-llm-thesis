# scripts/check_endpoint.py
"""Pre-flight probe of a served endpoint: does it support what the harness needs?

    python -m scripts.check_endpoint --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-32B-AWQ

This is NOT a smoke run of the agents (that is scripts/run_live_smoke.py) and not
a source of any thesis number. It answers one question per probe, so a failure
points at the exact feature that is missing instead of at "the run crashed":

  A  reachable        -- /v1/models answers and serves the expected model id
  B  completions      -- a raw /v1/completions call returns text plus usage
  C  extra_body       -- the endpoint accepts the exact extra params the client
                         sends (top_k / min_p / think / return_token_ids /
                         add_special_tokens); on rejection each one is retried
                         individually to name the culprit
  D  thinking split   -- with the Qwen3 chat template and enable_thinking=True the
                         raw completion actually contains a `</think>` delimiter
                         (the whole budget split depends on it)
  E  token ids        -- `return_token_ids` puts generated token ids on the choice
                         (exact thinking/answer counting; otherwise the client
                         silently falls back to re-tokenising text)
  F  guided json      -- constrained decoding for the terminal emit node; if
                         `guided_json` is rejected, `structured_outputs` is tried
                         and the working parameter name is reported
  G  input parity     -- locally counted input tokens == endpoint prompt_tokens
                         (the ledger-vs-usage identity, at single-call scale)
  H  seed stability   -- two identical requests return identical text (reported,
                         never enforced: batching non-determinism is a known
                         confound, section 7)

Exit code 0 only if every REQUIRED probe passed (A-G). H is informational.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.react_core import ANSWER_GUIDED_SCHEMA  # noqa: E402
from src.core.llm_client import MIN_P, SEED, TEMPERATURE, TOP_K, TOP_P  # noqa: E402

# The exact extra params LLMClient.complete sends on every call.
CLIENT_EXTRA_BODY = {
    "top_k": TOP_K,
    "min_p": MIN_P,
    "think": True,
    "return_token_ids": True,
    "add_special_tokens": False,
}

THINK_PROMPT = [
    {
        "role": "user",
        "content": (
            "You visit two people. Alice is free from minute 0 to 60 at loc_a, "
            "Bob from minute 30 to 90 at loc_b, travel a->b takes 15 minutes. "
            "State one feasible visiting order in a single short sentence."
        ),
    }
]


class Probes:
    """Collects pass/fail per probe and prints one summary table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool | None, str]] = []

    def record(self, key: str, name: str, ok: bool | None, detail: str = "") -> None:
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
        print(f"[{mark}] {key}  {name}" + (f" — {detail}" if detail else ""))
        self.rows.append((key, name, ok, detail))

    @property
    def failed(self) -> list[str]:
        return [f"{k} {n}" for k, n, ok, _ in self.rows if ok is False]


def build_parser() -> argparse.ArgumentParser:
    from src.core import DEFAULT_BASE_URL, DEFAULT_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None,
                        help="HF repo for token counting (default: the served model id)")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="exact HF commit of the tokenizer files (pin before the pilot)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="generation cap for the probe calls (thinking needs room)")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from openai import OpenAI

    from src.core import QwenTokenizer

    print("=" * 72)
    print("ENDPOINT PRE-FLIGHT — capability probe, not an experiment")
    print("=" * 72)
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")

    probes = Probes()
    client = OpenAI(base_url=args.base_url, api_key="preflight", timeout=args.timeout)

    # --- A: reachable, serving the expected id ------------------------------ #
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as exc:
        probes.record("A", "endpoint reachable", False, f"{type(exc).__name__}: {exc}")
        print("\nnothing else can be probed until the server answers.")
        return 1
    if args.model in served:
        probes.record("A", "endpoint reachable", True, f"serves {args.model}")
    else:
        probes.record("A", "endpoint reachable", False,
                      f"served ids are {served}, expected {args.model!r} "
                      "(start the server with --served-model-name equal to the repo id)")
        return 1

    # --- tokenizer (local counting side of the parity check) ---------------- #
    tok_repo = args.tokenizer or args.model
    try:
        tokenizer = QwenTokenizer(tok_repo, revision=args.tokenizer_revision)
    except Exception as exc:
        probes.record("-", "tokenizer load", False, f"{tok_repo}: {exc}")
        return 1
    print(f"tokenizer: {tok_repo} (revision {args.tokenizer_revision or 'latest'})")
    print()

    # --- B: plain completion ------------------------------------------------ #
    try:
        plain = client.completions.create(
            model=args.model, prompt="2 + 2 =", max_tokens=8, temperature=0.0, n=1
        )
        text = plain.choices[0].text or ""
        usage = getattr(plain, "usage", None)
        probes.record("B", "plain completion", True,
                      f"{len(text)} chars, usage={'yes' if usage else 'MISSING'}")
    except Exception as exc:
        probes.record("B", "plain completion", False, f"{type(exc).__name__}: {exc}")
        return 1

    # --- C: the client's exact extra_body ----------------------------------- #
    rendered = tokenizer.render_input(THINK_PROMPT, enable_thinking=True)
    local_input_tokens = tokenizer.count_text(rendered)

    def call(extra: dict, *, max_tokens: int | None = None, prompt: str | None = None):
        return client.completions.create(
            model=args.model,
            prompt=prompt if prompt is not None else rendered,
            max_tokens=max_tokens or args.max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            n=1,
            extra_body=extra,
        )

    completion = None
    try:
        completion = call(dict(CLIENT_EXTRA_BODY))
        probes.record("C", "client extra_body accepted", True,
                      ", ".join(CLIENT_EXTRA_BODY))
    except Exception as exc:
        # Narrow it down: which single parameter does this vLLM version reject?
        rejected = []
        for key, value in CLIENT_EXTRA_BODY.items():
            try:
                call({key: value}, max_tokens=8)
            except Exception:
                rejected.append(key)
        probes.record("C", "client extra_body accepted", False,
                      f"{type(exc).__name__}; rejected individually: "
                      f"{rejected or 'none — the combination is the problem'}")

    if completion is None:  # retry without extras so D/E/G still produce evidence
        try:
            completion = call({})
        except Exception as exc:
            probes.record("D", "thinking delimiter", False, f"no usable call: {exc}")
            return 1

    choice = completion.choices[0]
    raw = choice.text or ""

    # --- D: thinking delimiter present -------------------------------------- #
    if "</think>" in raw:
        head = raw.split("</think>")[0]
        probes.record("D", "thinking delimiter", True,
                      f"</think> present, {len(head)} chars of reasoning before it")
    elif raw.strip():
        probes.record("D", "thinking delimiter", False,
                      f"no </think> in {len(raw)} chars — either thinking is off, or the "
                      "cap truncated the think block (retry with a larger --max-tokens)")
    else:
        probes.record("D", "thinking delimiter", False,
                      "empty completion — the endpoint hides reasoning output "
                      "(this is the known Ollama behaviour; on vLLM it must not happen)")

    # --- E: generated token ids --------------------------------------------- #
    token_ids = getattr(choice, "token_ids", None)
    usage = getattr(completion, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    if isinstance(token_ids, (list, tuple)) and token_ids and all(
        isinstance(t, int) for t in token_ids
    ):
        agree = (completion_tokens is None) or (len(token_ids) == completion_tokens)
        probes.record("E", "return_token_ids", agree,
                      f"{len(token_ids)} ids vs usage.completion_tokens="
                      f"{completion_tokens}")
    else:
        probes.record("E", "return_token_ids", False,
                      "no generated token ids on the choice — the client would fall "
                      "back to re-tokenising text (works, but the exact split is lost)")

    # --- F: constrained JSON for the finalize node -------------------------- #
    finalize_prompt = tokenizer.render_input(
        [{"role": "user", "content":
          'Output the final plan as JSON: {"meetings": [{"person_id": "p0", '
          '"start_time": 10}]}'}],
        enable_thinking=False,
    )
    guided_ok = False
    last = "not attempted"
    for param, payload in (
        ("guided_json", {"guided_json": ANSWER_GUIDED_SCHEMA}),
        ("structured_outputs", {"structured_outputs": {"json": ANSWER_GUIDED_SCHEMA}}),
    ):
        try:
            out = call({**payload, "think": False}, max_tokens=128, prompt=finalize_prompt)
            parsed = json.loads(out.choices[0].text or "")
            assert "meetings" in parsed
            probes.record("F", "constrained JSON", True,
                          f"{param} accepted, parsed {len(parsed['meetings'])} meeting(s)")
            if param != "guided_json":
                print("      NOTE: this vLLM version wants 'structured_outputs'. "
                      "src/core/llm_client.py sends 'guided_json' — update it before "
                      "the pilot, or pin a version that accepts guided_json.")
            guided_ok = True
            break
        except Exception as exc:
            last = f"{param}: {type(exc).__name__}: {exc}"
    if not guided_ok:
        probes.record("F", "constrained JSON", False, last)

    # --- G: local input count vs endpoint prompt_tokens --------------------- #
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    if prompt_tokens is None:
        probes.record("G", "input token parity", False,
                      "endpoint reported no prompt_tokens — the ledger cannot be "
                      "cross-checked against the server")
    else:
        delta = local_input_tokens - prompt_tokens
        probes.record("G", "input token parity", delta == 0,
                      f"local={local_input_tokens} endpoint={prompt_tokens} delta={delta}"
                      + ("" if delta == 0 else
                         "  (a constant offset usually means add_special_tokens was "
                         "not honoured — the prompt got re-templated server-side)"))

    # --- H: seed stability (informational) ---------------------------------- #
    try:
        again = call(dict(CLIENT_EXTRA_BODY))
        identical = (again.choices[0].text or "") == raw
        probes.record("H", "seed stability", None,
                      "identical output on repeat" if identical else
                      "output differs on repeat — expected under batching, record it "
                      "as the section-7 confound")
    except Exception as exc:
        probes.record("H", "seed stability", None, f"not probed: {exc}")

    print()
    if probes.failed:
        print(f"PRE-FLIGHT FAILED: {len(probes.failed)} required probe(s) — "
              + "; ".join(probes.failed))
        return 1
    print("PRE-FLIGHT PASSED — the endpoint supports everything the harness needs.")
    print("Next: pytest, then scripts/run_live_smoke.py against this endpoint "
          "(infrastructure check only — its scores are not quality evidence).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
