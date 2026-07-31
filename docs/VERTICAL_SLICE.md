# Offline harness vertical slice

One reproducible run through the whole stack, with no model and no endpoint:

```
generator / handcrafted Instance
        -> CP-SAT oracle          (optimum + binding-conflict metric; harness-side only)
        -> C1 ReAct  (or C2)      (LangGraph loop, budget ledger, section-2 finalisation)
        -> hidden validator/scorer (independent of the agent's self-report)
        -> RunResult              (canonical Layer-0 record)
        -> JSON document          (results/logs/vertical_slice/<run_id>.json)
```

## Run it

```bash
python -m scripts.run_vertical_slice                      # C1, cap 4000, seed 0
python -m scripts.run_vertical_slice --condition c2_verify_revise
python -m scripts.run_vertical_slice --seed 3 --cap 2000 --n-people 5
```

The CLI generates one deterministic instance, runs the **offline scripted client**
(`src/harness/smoke_client.py`) and writes one JSON log. The banner says it loudly:
this validates plumbing, not model behaviour. It does **not** close the live
Qwen/vLLM checkpoint (thinking-mode dynamics, ledger vs `completion.usage`,
guided decoding are all untested until a real endpoint run).

## Entry point

`src.harness.run_single_instance(instance=..., client=..., condition=..., cap=...,
model_label=..., [level=], [output_dir=])` returns a `HarnessRun`:
`run_result` (validated `RunResult`), the annotated instance, the `OracleSolution`,
oracle latency, the raw agent artifacts (transcript, best plan, calls) and the JSON
path. Conditions: `c1_react`, `c2_verify_revise`; `c3_mas` / `c4_planner_critic`
raise `NotImplementedError` on purpose (later work, same dispatch).

## Invariants encoded here

* **Oracle isolation.** The solver runs before the agent and nothing derived from
  it (optimum, plan, status, metric, level, score) enters prompts, tools,
  observations, or the transcript. A test scans everything the client is shown for
  oracle/scorer vocabulary.
* **Honest levels.** `level` is derived only via the pre-registered anchor
  "easy = 0 binding conflicts". A conflicted instance without an explicit
  pilot-binned `level` is refused, not guessed.
* **Honest labels.** `model_label` names what produced the tokens
  (`offline-smoke-client` for the CLI); scripted runs are marked in `notes`.

## JSON document shape (`harness_slice/1.0`)

Top-level keys: `schema_version`, `run_id`, `timestamp_utc`, `condition`, `cap`,
`model_label`, `finalization_reserve`, `seeds` (generator / solver / inference —
all three logged separately), `provenance` (git commit + branch), `instance`,
`oracle`, `agent` (transcript + step count), `run_result` (the full canonical
record: tokens, calls, score, plans, sampling). Reruns are byte-stable except
timestamps and latencies; the filename is `<run_id>.json`, so a rerun of the same
cell overwrites idempotently.
