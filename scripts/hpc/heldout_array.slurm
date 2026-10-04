#!/bin/bash
# scripts/hpc/heldout_array.slurm
#
# THE HELD-OUT RUN. Do not submit until every gate in scripts/hpc/README.md
# section 5.5 is closed and the dry-run job has been inspected.
#
#   sbatch scripts/hpc/heldout_array.slurm
#     sbatch scripts/hpc/heldout_array.slurm
#
# Frozen scheduling (2026-08-26, from measured calibration -- job 1891293):
#
#   logical shards : 24          --num-shards 24
#   concurrency    : %6          4 waves, ~2.3 days wall-clock
#   walltime       : 20:00:00    estimate 13.9 h, margin 1.44x
#   execution      : sequential, one request at a time
#   runners/shard  : 5, one per condition, five separate processes
#
# The five conditions are deliberately NOT merged into one Python process. The
# saving (~30 min per shard) does not justify changing the execution path before
# the held-out run.
#
# One vLLM server per shard serves all five runners: startup is ~25 minutes and
# paying it five times would cost more than the runner startups it replaces.
#
# Resume: all shards share one output directory. The runner verifies each result
# on disk against the manifest and skips only what is genuinely complete, so a
# shard killed at its walltime can be resubmitted and will continue. Shards never
# collide -- the split is disjoint, proven by scripts/hpc/verify_shard_plan.py.

#SBATCH --job-name=mas-heldout
#SBATCH --partition=h200_short
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-23%6
#SBATCH --output=logs/hpc/heldout-%A_%a.out
#SBATCH --error=logs/hpc/heldout-%A_%a.out

set -euo pipefail
JOB_START_EPOCH=$(date +%s)

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

# The four frozen held-out manifests, in a fixed order. Named here rather than
# passed in: these are the manifests the design froze, and a campaign that can be
# pointed at other data by an environment variable is one typo from running the
# wrong experiment. Override only to rerun a subset deliberately.
SUBSETS="${SUBSETS:-results/manifests/subset__bands_held_out_n8__278ce0f2e8fe.json results/manifests/subset__heldout_n4__54f3f48d6232.json results/manifests/subset__heldout_n5__4b91dee07c34.json results/manifests/subset__heldout_n6__44418ababc22.json}"
for sub in ${SUBSETS}; do
  [ -s "$sub" ] || { echo "!!! manifest missing: $sub"; exit 6; }
done
NUM_SHARDS="${NUM_SHARDS:-24}"
CAPS="${CAPS:-16000,32000,64000,128000}"
OUT_DIR="${OUT_DIR:-results/logs/heldout}"
CONDITIONS="${CONDITIONS:-c1_react c2_verify_revise c3_mas c4_planner_critic c5_best_of_3}"
SHARD="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array}"

for venv in .venv-vllm .venv-harness; do
  [ -x "$venv/bin/python" ] || { echo "!!! ${venv} missing -- run scripts/hpc/bootstrap.sh first"; exit 4; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "!!! nvidia-smi not found -- this is not a GPU node"; exit 4; }

export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
[ -d "$HF_HOME" ] || export HF_HOME="$HOME/hf-cache"
export MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
export MODEL_REVISION="${MODEL_REVISION:-0499c3ac83fdef8810b907a23894ba91e95eddd8}"

SNAPSHOT_DIR="${HF_HOME}/hub/models--${MODEL//\//--}/snapshots/${MODEL_REVISION}"
[ -s "${SNAPSHOT_DIR}/config.json" ] || {
  echo "!!! pinned checkpoint not in the cache vLLM reads: ${SNAPSHOT_DIR}"
  echo "!!! re-run scripts/hpc/bootstrap.sh"; exit 6; }

TOKENIZER_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-8B-AWQ"
[ -d "$TOKENIZER_DIR" ] || {
  echo "!!! pinned counting tokenizer not cached: ${TOKENIZER_DIR}"
  echo "!!! HF_HUB_OFFLINE=1 would fail -- re-run scripts/hpc/bootstrap.sh"; exit 6; }

echo "=================================================================="
echo "HELD-OUT RUN -- shard ${SHARD} of ${NUM_SHARDS}"
echo "array job    : ${SLURM_ARRAY_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}"
echo "node         : $(hostname)   partition ${SLURM_JOB_PARTITION}"
echo "subsets      : $(echo ${SUBSETS} | tr ' ' '
' | xargs -n1 basename | tr '
' ' ')"
echo "caps         : ${CAPS}"
echo "out          : ${OUT_DIR}"
echo "commit       : $(git rev-parse HEAD)"
echo "started      : $(date -Is)"
echo "=================================================================="

# ------------------------------------------------------- seed-range guard
# --expect held_out is the direction that matters here: it refuses to run the
# held-out campaign against development instances, which would produce 3,960
# perfectly healthy-looking runs of the wrong thing.
echo "--- seed-range guard (expect held_out), every manifest"
for sub in ${SUBSETS}; do
  python3 scripts/hpc/guard_seed_range.py "${sub}" --expect held_out
done

# ------------------------------------------------------- queue wait
SLURM_LINE="$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null || true)"
SUBMIT_TIME="$(echo "$SLURM_LINE" | grep -o 'SubmitTime=[^ ]*' | head -1 | cut -d= -f2- || true)"
START_TIME="$(echo "$SLURM_LINE" | grep -o 'StartTime=[^ ]*' | head -1 | cut -d= -f2- || true)"
QUEUE_WAIT_S=""
if [ -n "$SUBMIT_TIME" ] && [ -n "$START_TIME" ]; then
  S=$(date -d "$SUBMIT_TIME" +%s 2>/dev/null || echo "")
  T=$(date -d "$START_TIME"  +%s 2>/dev/null || echo "")
  [ -n "$S" ] && [ -n "$T" ] && QUEUE_WAIT_S=$(( T - S ))
fi
echo "--- queue wait: ${QUEUE_WAIT_S:-unknown} s"

# ------------------------------------------------------- hardware gate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
COMPUTE_CAP="$(.venv-vllm/bin/python -c \
  'import torch;print("%d.%d"%torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else print("none")' \
  2>/dev/null || echo "unknown")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "--- gpu: ${GPU_NAME} | compute capability ${COMPUTE_CAP}"
case "$COMPUTE_CAP" in
  6.*|7.*|none|unknown)
    echo "!!! compute capability ${COMPUTE_CAP} cannot serve AWQ INT4"; exit 3 ;;
esac
# The registered hardware policy (README section 0.1) is one GPU class for the
# whole primary held-out run. A shard that lands on something else is refused
# rather than quietly mixing classes inside the experiment.
REQUIRE_GPU="${REQUIRE_GPU:-H200}"
case "$GPU_NAME" in
  *"$REQUIRE_GPU"*) echo "--- ${REQUIRE_GPU} confirmed" ;;
  *) echo "!!! this shard got ${GPU_NAME}, not ${REQUIRE_GPU}."
     echo "!!! GPU classes must not be mixed inside the primary held-out run."
     echo "!!! Override deliberately with REQUIRE_GPU=... if the policy changed."; exit 3 ;;
esac

# ------------------------------------------------------- start vLLM once
PORT=$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
)
VLLM_LOG="logs/hpc/vllm-${SLURM_ARRAY_JOB_ID}_${SHARD}.log"
export VLLM_LAUNCH_CMD="HF_HOME=${HF_HOME} HF_HUB_OFFLINE=1 VLLM_REVISION=${MODEL_REVISION} VLLM_EXTRA_ARGS=--no-enable-prefix-caching ./scripts/serve_vllm.sh ${MODEL} 0 ${PORT}"
echo "--- launch: ${VLLM_LAUNCH_CMD}"

(
  # shellcheck disable=SC1091
  source .venv-vllm/bin/activate
  export HF_HOME
  export HF_HUB_OFFLINE=1
  VLLM_REVISION="${MODEL_REVISION}" \
  VLLM_EXTRA_ARGS=--no-enable-prefix-caching \
    ./scripts/serve_vllm.sh "${MODEL}" 0 "${PORT}"
) > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'echo "--- stopping vLLM (pid ${VLLM_PID})"; kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

echo "--- waiting for vLLM to become healthy (up to 60 min)"
LOAD_START=$(date +%s)
DEADLINE=$(( SECONDS + 3600 ))
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "!!! vLLM exited before becoming healthy:"; tail -40 "${VLLM_LOG}"; exit 1
  fi
  if (( SECONDS > DEADLINE )); then
    echo "!!! vLLM did not become healthy within 60 min"; tail -40 "${VLLM_LOG}"; exit 1
  fi
  sleep 10
done
MODEL_LOAD_S=$(( $(date +%s) - LOAD_START ))
echo "--- vLLM healthy after ${MODEL_LOAD_S}s"

PROV_JSON="logs/hpc/provenance-${SLURM_ARRAY_JOB_ID}_${SHARD}.json"
bash scripts/hpc/provenance.sh "${PROV_JSON}" "${VLLM_LOG}"

# ------------------------------------------------------- the five runners
# Sequential, one condition per process, one request at a time. The batch
# composition regime is unchanged from the frozen pilot: max_num_seqs is served
# as configured and nothing here issues concurrent requests.
# shellcheck disable=SC1091
source .venv-harness/bin/activate
export HF_HUB_OFFLINE=1          # the pinned tokenizer is cached; no Hub at run time
mkdir -p "${OUT_DIR}"

SUMMARY="logs/hpc/heldout-${SLURM_ARRAY_JOB_ID}_${SHARD}-summary.json"
summarise() {
  python scripts/hpc/summarise_runs.py \
    --runs-dir "${OUT_DIR}" --out "${SUMMARY}" \
    --label "held-out shard ${SHARD}/${NUM_SHARDS}, array ${SLURM_ARRAY_JOB_ID}" \
    --extra "shard=${SHARD}" --extra "num_shards=${NUM_SHARDS}" \
    --extra "gpu=${GPU_NAME}" --extra "queue_wait_seconds=${QUEUE_WAIT_S:-}" \
    --extra "model_load_seconds=${MODEL_LOAD_S}" \
    --extra "job_elapsed_seconds=$(( $(date +%s) - JOB_START_EPOCH ))" || true
}

# --- BEGIN RUN LOOP (extracted verbatim and executed by
# --- tests/test_heldout_array_wiring.py; keep the markers, and keep everything
# --- between them runnable in isolation against stubbed commands)
i=0
TOTAL_CELLS=$(( $(echo ${SUBSETS} | wc -w) * $(echo ${CONDITIONS} | wc -w) ))
for sub in ${SUBSETS}; do
  for cond in ${CONDITIONS}; do
    i=$(( i + 1 ))
    echo
    echo "=== [${i}/${TOTAL_CELLS}] ${cond} on $(basename "${sub}") -- shard ${SHARD}/${NUM_SHARDS} -- $(date -Is)"
    CELL_START=$(date +%s)
    python scripts/run_pilot_sweep.py \
      --subset "${sub}" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${MODEL}" \
      --condition "${cond}" \
      --caps "${CAPS}" \
      --shard-index "${SHARD}" \
      --num-shards "${NUM_SHARDS}" \
      --out "${OUT_DIR}"
    echo "=== [${i}/${TOTAL_CELLS}] wall $(( $(date +%s) - CELL_START ))s"
    summarise
  done
done
# --- END RUN LOOP

echo
echo "================= SHARD ${SHARD} COMPLETE ================="
echo "  queue wait  : ${QUEUE_WAIT_S:-unknown} s"
echo "  model load  : ${MODEL_LOAD_S} s"
echo "  job elapsed : $(( $(date +%s) - JOB_START_EPOCH )) s"
echo "  summary     : ${SUMMARY}"
echo "  run docs    : ${OUT_DIR}"
echo "=========================================================="
