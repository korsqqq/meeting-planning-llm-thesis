#!/usr/bin/env bash
# scripts/serve_vllm.sh
#
# Start ONE vLLM OpenAI-compatible server for ONE model on ONE GPU (TP=1).
#
#   ./scripts/serve_vllm.sh Qwen/Qwen3-8B-AWQ   1 8001    # pilot / robustness
#   ./scripts/serve_vllm.sh Qwen/Qwen3-32B-AWQ  0 8000    # main run
#
# Arguments: <model> [gpu_index] [port]
#
# Why one model per GPU instead of tensor_parallel_size=2 (THESIS_DECISIONS.md
# section 1): both AWQ INT4 checkpoints fit comfortably on a single 48 GB A6000,
# TP=2 adds NCCL synchronisation (PCIe without NVLink) and one more source of
# run-to-run non-determinism, and separate servers let the 8B pilot and the 32B
# main run proceed independently.
#
# Run it inside tmux (the process stays in the foreground and logs to stdout):
#   tmux new -s vllm32 './scripts/serve_vllm.sh Qwen/Qwen3-32B-AWQ 0 8000 2>&1 | tee logs/vllm_32b.log'

set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <model> [gpu_index] [port]}"
GPU="${2:-0}"
PORT="${3:-8000}"

# 32k native context covers the 16k budget cap plus scaffold (no YaRN needed).
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
# Batching width. Lower = less batch-composition variance between runs (the
# residual-non-determinism confound of section 7), higher = more throughput.
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
# Inference seed. Must equal src.core.llm_client.SEED for the run log to be honest.
SERVER_SEED="${VLLM_SEED:-42}"
# Exact HF commit of the checkpoint. Empty = "latest on the Hub", acceptable for
# the first bring-up ONLY; the pilot and the main run must pin it.
REVISION="${VLLM_REVISION:-}"
# Escape hatch for version-specific flags, e.g. prefix-caching control. Check
# `vllm serve --help` on the pinned version before adding anything here.
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

echo "=================================================================="
echo "model        : ${MODEL}"
echo "revision     : ${REVISION:-<latest on Hub — pin before the pilot>}"
echo "gpu          : ${GPU} (tensor_parallel_size=1)"
echo "port         : ${PORT}"
echo "max-model-len: ${MAX_MODEL_LEN}"
echo "max-num-seqs : ${MAX_NUM_SEQS}"
echo "seed         : ${SERVER_SEED}"
echo "extra args   : ${EXTRA_ARGS:-<none>}"
echo "=================================================================="
echo "Quantisation is auto-detected from the checkpoint's config.json (AWQ INT4)."
echo "On Ampere (A6000, SM 8.6) vLLM selects the awq_marlin kernel — confirm that"
echo "in the startup log; it is the evidence that the served path is the intended one."
echo

REV_ARG=()
if [[ -n "${REVISION}" ]]; then
  REV_ARG=(--revision "${REVISION}" --tokenizer-revision "${REVISION}")
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
CUDA_VISIBLE_DEVICES="${GPU}" exec vllm serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --seed "${SERVER_SEED}" \
  --dtype auto \
  "${REV_ARG[@]}" \
  ${EXTRA_ARGS}
