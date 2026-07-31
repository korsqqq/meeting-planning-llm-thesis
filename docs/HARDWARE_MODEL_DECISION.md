# Hardware and model-serving decision

**Date:** 2026-07-31
**Status:** LOCKED before pilot
**Scope:** Pilot, main run, and robustness run on the TU Berlin workstation with 2 × NVIDIA RTX A6000 (48 GB each)

## Decision

The experiment will use the official AWQ 4-bit Qwen3 checkpoints instead of the previously selected FP8 checkpoints:

- **Main model:** `Qwen/Qwen3-32B-AWQ`
- **Pilot and robustness model:** `Qwen/Qwen3-8B-AWQ`
- **Serving backend:** vLLM, with one model served on one GPU (`tensor_parallel_size=1`) whenever possible

This file is an explicit pre-pilot amendment to `THESIS_DECISIONS.md`. Wherever the older documentation says `Qwen3-32B-FP8`, `Qwen3-8B-FP8`, “FP8 parity”, or “pilot on FP8”, it must be read as the AWQ choices above. Historical progress-log entries are not retroactively rewritten; they describe the decision that existed at that time.

## Reason

The available RTX A6000 cards are Ampere-generation GPUs. Rather than make the experiment depend on an FP8 compatibility path that must first be validated on this hardware, the project selects a deployment format that is directly supported on Ampere by vLLM and has substantially lower memory demand.

A 32B model in 4-bit form fits comfortably within one 48 GB GPU together with runtime overhead and KV cache for the planned context lengths. This keeps the serving topology simple and avoids splitting one model across both GPUs unless a live smoke test shows a concrete need.

## Methodological consequence

This change is not a change to the research question, conditions, task, token-budget rule, scoring, prompts, or analysis. It changes only the numerical weight format used to serve the same Qwen3 model family.

The 8B and 32B runs use the same official AWQ quantisation family. Therefore, the model-size robustness comparison does not introduce a deliberate FP8-versus-INT4 mismatch.

The thesis must report the following accurately:

1. Both evaluated model sizes were served from official Qwen3 AWQ checkpoints.
2. The choice was made before the pilot because of the available 2 × RTX A6000 hardware.
3. Results apply to the tested AWQ deployments and must not be presented as results for unquantised BF16 models.
4. Quantisation may affect absolute model quality and is therefore listed as a limitation, but it is held constant across the compared agent architectures within each model size.

## Serving layout

Preferred layout:

- GPU 0: `Qwen/Qwen3-32B-AWQ` for the main experiment
- GPU 1: `Qwen/Qwen3-8B-AWQ` for setup, pilot, and robustness work

The two models do not have to run simultaneously. Because the workstation is shared, only the GPU capacity actually needed for the current job should be occupied.

Default serving rule:

- start with `tensor_parallel_size=1`;
- use the exact same pinned vLLM version for pilot and main runs;
- pin the exact Hugging Face revision for each checkpoint;
- record GPU model, driver, CUDA, vLLM, checkpoint revision, quantisation method, context length, and launch command in every experiment environment record;
- do not silently fall back to another checkpoint or numerical format.

## Required pre-flight checks

The change removes FP8 compatibility as a decision gate, but it does not remove the ordinary live checks. Before the pilot:

- both AWQ checkpoints must start successfully in vLLM;
- the endpoint must return valid completions;
- tokenizer and chat-template parity across 8B and 32B must be tested empirically;
- local ledger counts must be compared with endpoint usage on representative calls;
- thinking extraction and schema-constrained finalisation must pass the live smoke test;
- peak GPU memory and usable `max_model_len` must be recorded.

If `Qwen3-32B-AWQ` unexpectedly fails on one A6000, the first response is to reduce concurrency, `max_model_len`, or KV-cache utilisation. Tensor parallelism across both A6000 cards is a fallback, not the default. A different weight format requires a new written amendment before any pilot data are used.

## Documentation replacements

For current planning, apply these replacements:

| Old wording | Current wording |
|---|---|
| Qwen3-32B-FP8 | Qwen3-32B-AWQ |
| Qwen3-8B-FP8 | Qwen3-8B-AWQ |
| FP8 parity | AWQ quantisation parity |
| pilot on FP8 | pilot on Qwen3-8B-AWQ |
| main sweep on Qwen3-32B-FP8 | main sweep on Qwen3-32B-AWQ |
| final 48 GB+ GPU plan | one RTX A6000 per served AWQ model by default |

## Unchanged decisions

All other locked decisions remain unchanged, including:

- thinking mode and sampling settings;
- exact equal-token-budget accounting;
- finalisation reserve and structured output;
- C1, C2, C3, and any pre-declared optional conditions;
- generated benchmark, solver, hidden validator, and data splits;
- pilot-derived complexity boundaries and token caps;
- held-out main test and statistical analysis plan.
