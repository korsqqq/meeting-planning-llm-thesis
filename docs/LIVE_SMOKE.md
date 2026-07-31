# Live smoke: purpose, how to run, how to judge

**LIVE SMOKE ONLY — NOT PILOT, NOT EXPERIMENT.** The live smoke exists to close
one manual checkpoint: confirm on a *real* Qwen/vLLM endpoint that the plumbing
behaves the way the offline tests assume. It produces no thesis numbers, no cap
calibration, no level binning. 1–3 easy (0-conflict) instances, nothing more.

## What it answers

1. Does vLLM return the expected `<think>…</think>` / answer split through
   `/v1/completions` (self-parsed by `split_thinking`)?
2. Does the Section-2 policy behave on real output — in particular, does empty
   post-think content route straight to finalise?
3. Does the internal ledger (Qwen tokenizer counts) approximately match the
   endpoint-reported usage?
4. Is endpoint usage stored in the JSON for manual inspection — or, if missing
   or incompatible, is that recorded honestly (`has_endpoint_usage: false`)?

## How to run

```bash
python -m scripts.run_live_smoke \
    --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B-FP8 \
    --condition c1_react --cap 2000 --seed 0 --n-runs 2
```

Useful flags: `--require-endpoint-usage` (exit non-zero if the endpoint reports
no usage), `--n-runs {1,2,3}`, `--output-dir results/logs/live_smoke` (default,
git-ignored). The script refuses nothing about the endpoint — it just runs the
normal harness (`run_single_instance`) with the real `LLMClient` and prints a
per-run summary.

## What to inspect in the JSON

Each run writes `results/logs/live_smoke/<run_id>.json`:

* `usage_audit.totals` — internal input/thinking/answer/total vs endpoint
  prompt/completion/total, plus `delta_total_tokens` and
  `relative_delta_total`. `usage_audit.per_call` gives the same per call with
  the call's role.
* `usage_audit.has_endpoint_usage` — `false` means the endpoint reported
  nothing; the fields stay null (never faked).
* `transcript_sanity` — `think_leak` must be `false`; `oracle_term_hits` should
  be empty (a hit is a *read-this-by-hand* flag, not automatically a failure —
  a model may innocently say "score" in its own prose).
* `run_result.calls[*]` — roles and per-call internal counts; check that an
  empty post-think call is immediately followed by `finalize`, never a retry.
* `agent.transcript` — read it. This is checkpoint reading, not optional.

## Manual pass/fail checklist

**PASS** if all of:

* the endpoint returns completions and the runs finish;
* no `<think>`/`</think>` text appears anywhere in the transcript
  (`transcript_sanity.think_leak == false`);
* if an empty post-think output occurred, the very next call is `finalize`
  (no retry, no verify/revise after an aborted draft in C2);
* the JSON contains endpoint usage, or honestly marks it missing;
* internal vs endpoint totals are close enough to understand — the internal
  count includes the `<think>`/`</think>` tag tokens and inter-part whitespace
  (split at the last `</think>`), so `delta_total_tokens` should be ~0: exactly
  0 for the output half when vLLM returns generated token ids
  (`return_token_ids`), small residuals only where text segments had to be
  re-tokenised; note any observed delta for the Limitations chapter.

**FAIL / investigate** if any of:

* `<think>` appears in the transcript;
* an Action was executed that only existed inside thinking text
  (`raw_text` parsed as an action — must be impossible by construction);
* C2 runs verify/revise after an aborted (empty) draft;
* the code claims endpoint usage that the endpoint did not send;
* the ledger differs wildly (large or *negative* delta) with no explanation;
* solver/oracle/scorer vocabulary appears in the agent's prompt or
  observations (`transcript_sanity.oracle_term_hits` pointing at a harness-side
  message, not model prose).

## What the live smoke does NOT close

Cap calibration ([PILOT]), level binning ([PILOT]), C3/C4, statistics, and the
main-run `guided_regex` Action constraint (not yet implemented in `LLMClient`).
