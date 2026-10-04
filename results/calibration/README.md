# C1 calibration artifacts

**Status: calibration / exploratory. NOT formal pilot evidence.**

These runs inform the budget ladder and describe mechanisms. They are **not** a
pre-registered, held-out basis for comparing C1 against C2/C3, and no architectural
comparison may rest on them.

## Why they are not the formal pilot

The subset they were run on contains one instance that had already been executed and its
results inspected **before** the manifest was frozen:

    uniform-n4-t20-o20-s10000

It comes from the cell `n_people=4, tightness=0.2, overlap=0.2, uniform`, which was used
in an earlier cap-calibration pass over seeds 10000–10004. The freezing commit
(`ba55687`) claimed the pilot range had been "read exactly once, with no prior
inspection"; that claim was wrong, and this directory exists so the error stays visible
rather than being quietly overwritten.

The exposure is narrow — 1 of 30 subset instances, 1 of 48 cells, 5 of 240 pool
instances — and every table in the analysis is repeated with that instance removed
(`results/analysis/c1_calibration/`), which shows it carries nothing on its own. The
reclassification is nevertheless of the whole set: a subset that contains an exposed
instance is no longer a pre-registered selection, regardless of how little that instance
moves the averages.

The formal pilot is a different, disjoint selection scanned from seed 10005 onward
(`THESIS_DECISIONS.md` §3). No instance appears in both.

## Contents

| Path | What it is |
|---|---|
| `c1_20260805/` | 120 run documents: 30 instances × caps 8000/16000/32000/64000, C1 on `Qwen/Qwen3-32B-AWQ`, prefix caching off, context window 32768 |
| `infra_smoke_context/` | One infrastructure smoke on **development** seed 0 with the window lowered to 4096, used to verify the context guard against a live endpoint. Not calibration data; excluded from every analysis. |

Raw artifacts were archived in commit `bbe5a7b`. The analysis that reads them
(`scripts/analyse_c1_calibration.py`) refuses to run unless the set passes a fail-loud
integrity audit: 120 documents, 30 instances, exactly the four caps per instance, no
duplicate run ids, one condition, one model, one window, one subset and one binning
manifest, endpoint usage present everywhere, internal tokens equal to endpoint tokens in
every run, and no think leak.

## What these files contain, and what a public release must not

The run documents include `agent.transcript`: the full visible ReAct conversation, with
the agent's `Thought:` rationale, its tool calls and the observations returned to it.

They do **not** contain the model's hidden `<think>` reasoning. That block is split off
and counted for the budget, never stored and never re-fed into the conversation
(`THESIS_DECISIONS.md` §2). Only its token count survives.

This repository is a **private draft**, and the raw transcripts stay here. A public
release must not copy these JSON files as they are. Publish either

  * the derived artifacts only — `runs.csv`, `summary.json`, `report.md` and checksums; or
  * sanitised JSON with message contents removed, each carrying the SHA-256 of the
    private original so the derivation can be checked without republishing the text.
