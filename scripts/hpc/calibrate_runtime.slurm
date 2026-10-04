#!/bin/bash
# scripts/hpc/calibrate_runtime.slurm
#
# Runtime calibration for scheduling the held-out sweep. NOT a thesis result.
#
#   mkdir -p logs/hpc && sbatch scripts/hpc/calibrate_runtime.slurm
#
# What it measures: how long a run of each expensive shape actually takes on
# H200, so that shard size, array concurrency and walltime for the ~2,760
# held-out runs are chosen from measurement rather than from the A6000 numbers.
#
# What it runs: nine runs on TWO already-used, non-held-out n = 8 development
# instances from the frozen budget-dev set (seeds 30000-30063).
#
#   hard band, first instance in manifest order : C1/C3/C5 x 64k and 128k = 6
#   low  band, first instance in manifest order : C1/C3/C5 x 128k        = 3
#
# The instances are taken by position in the frozen manifest, never by outcome.
# Both already have frozen runs elsewhere; this job writes to its own directory
# and never touches them.
#
# One vLLM server serves all nine runs: startup costs ~27 minutes on a cold
# node, and paying it nine times would swamp the thing being measured.
#
# The summary is rewritten after every run, so a job killed at its walltime
# still leaves the timings it had already collected.

#SBATCH --job-name=mas-calib
#SBATCH --partition=h200_short
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hpc/calib-%j.out
#SBATCH --error=logs/hpc/calib-%j.out

set -euo pipefail
JOB_START_EPOCH=$(date +%s)

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

for venv in .venv-vllm .venv-harness; do
  [ -x "$venv/bin/python" ] || { echo "!!! ${venv} missing -- run scripts/hpc/bootstrap.sh first"; exit 4; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "!!! nvidia-smi not found -- this is not a GPU node"; exit 4; }

export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
[ -d "$HF_HOME" ] || export HF_HOME="$HOME/hf-cache"

export MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
export MODEL_REVISION="${MODEL_REVISION:-0499c3ac83fdef8810b907a23894ba91e95eddd8}"

SNAPSHOT_DIR="${HF_HOME}/hub/models--${MODEL//\//--}/snapshots/${MODEL_REVISION}"
if [ ! -s "${SNAPSHOT_DIR}/config.json" ]; then
  echo "!!! pinned checkpoint is not in the cache vLLM reads:"
  echo "!!!   ${SNAPSHOT_DIR}"
  echo "!!! re-run scripts/hpc/bootstrap.sh"
  exit 6
fi

# The frozen n = 8 development set. Seeds 30000-30063: development, never
# held-out. The guard below proves it rather than trusting the filename.
SUBSET="${SUBSET:-results/manifests/subset__bands_budget_dev__911e4864d6fb.json}"
EXPECT_SUBSET_HASH="${EXPECT_SUBSET_HASH:-911e4864d6fb}"
OUT_DIR="results/logs/hpc_calib/${SLURM_JOB_ID}"
SUMMARY="logs/hpc/calib-${SLURM_JOB_ID}-summary.json"

case "$SUBSET$OUT_DIR" in
  *heldout*|*held_out*|*held-out*)
    echo "!!! refusing to run: this job must never touch held-out data"; exit 2 ;;
esac

echo "=================================================================="
echo "RUNTIME CALIBRATION -- timing only, NOT a thesis outcome"
echo "job          : ${SLURM_JOB_ID} on $(hostname)"
echo "partition    : ${SLURM_JOB_PARTITION}"
echo "subset       : ${SUBSET}  (n = 8 development set -- NOT held-out)"
echo "out          : ${OUT_DIR}"
echo "started      : $(date -Is)"
echo "=================================================================="

echo "--- held-out guard"
python3 scripts/hpc/guard_seed_range.py "${SUBSET}" --expect development \
  --expect-content-hash "${EXPECT_SUBSET_HASH}"

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
    echo "!!! compute capability ${COMPUTE_CAP} cannot serve AWQ INT4 -- request an H200 or A100"; exit 3 ;;
esac

# ------------------------------------------------------- instance choice
# First instance of each band in manifest order. Positional, deterministic, and
# fixed before any timing exists -- it cannot be a choice made from results.
read -r HARD_ID LOW_ID <<<"$(SUBSET="$SUBSET" python3 -c '
import json, os
d = json.load(open(os.environ["SUBSET"], encoding="utf-8"))
inst = d["instances"]
print(inst["hard"][0]["instance_id"], inst["easy"][0]["instance_id"])
')"
echo "--- hard-band instance : ${HARD_ID}"
echo "--- low-band  instance : ${LOW_ID}"

# ------------------------------------------------------- start vLLM once
PORT=$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
)
VLLM_LOG="logs/hpc/vllm-${SLURM_JOB_ID}.log"
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
    echo "!!! vLLM exited before becoming healthy -- last 40 log lines:"; tail -40 "${VLLM_LOG}"; exit 1
  fi
  if (( SECONDS > DEADLINE )); then
    echo "!!! vLLM did not become healthy within 60 min"; tail -40 "${VLLM_LOG}"; exit 1
  fi
  sleep 10
done
MODEL_LOAD_S=$(( $(date +%s) - LOAD_START ))
echo "--- vLLM healthy after ${MODEL_LOAD_S}s"

PROV_JSON="logs/hpc/provenance-${SLURM_JOB_ID}.json"
bash scripts/hpc/provenance.sh "${PROV_JSON}" "${VLLM_LOG}"

# ------------------------------------------------------- decode throughput
probe_tok_s() {
  local start end response
  start=$(date +%s.%N)
  response=$(curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Count from one to two hundred in words.\",\"max_tokens\":512,\"temperature\":0.6,\"top_p\":0.95,\"seed\":42}")
  end=$(date +%s.%N)
  RESPONSE="$response" START="$start" END="$end" python3 -c '
import json, os
elapsed = float(os.environ["END"]) - float(os.environ["START"])
try:
    n = json.loads(os.environ["RESPONSE"])["usage"]["completion_tokens"]
except Exception:
    print(""); raise SystemExit(0)
print(f"{n/elapsed:.1f}")
' || echo ""
}
echo "--- decode throughput: 1 warmup (discarded) + 2 timed"
WARMUP_TOK_S=$(probe_tok_s); echo "    warmup   : ${WARMUP_TOK_S:-unavailable} tok/s (discarded)"
TOK_1=$(probe_tok_s);        echo "    timed #1 : ${TOK_1:-unavailable} tok/s"
TOK_2=$(probe_tok_s);        echo "    timed #2 : ${TOK_2:-unavailable} tok/s"

# ------------------------------------------------------- the nine runs
# shellcheck disable=SC1091
source .venv-harness/bin/activate
mkdir -p "${OUT_DIR}"

summarise() {
  python scripts/hpc/summarise_runs.py \
    --runs-dir "${OUT_DIR}" --out "${SUMMARY}" \
    --label "runtime calibration, job ${SLURM_JOB_ID}, ${GPU_NAME}" \
    --extra "gpu=${GPU_NAME}" \
    --extra "compute_capability=${COMPUTE_CAP}" \
    --extra "queue_wait_seconds=${QUEUE_WAIT_S:-}" \
    --extra "model_load_seconds=${MODEL_LOAD_S}" \
    --extra "decode_tok_s_timed=${TOK_1:-},${TOK_2:-}" \
    --extra "decode_tok_s_warmup_discarded=${WARMUP_TOK_S:-}" \
    --extra "job_elapsed_seconds=$(( $(date +%s) - JOB_START_EPOCH ))"
}

CELLS=(
  "${HARD_ID}|64000|c1_react"
  "${HARD_ID}|64000|c3_mas"
  "${HARD_ID}|64000|c5_best_of_3"
  "${HARD_ID}|128000|c1_react"
  "${HARD_ID}|128000|c3_mas"
  "${HARD_ID}|128000|c5_best_of_3"
  "${LOW_ID}|128000|c1_react"
  "${LOW_ID}|128000|c3_mas"
  "${LOW_ID}|128000|c5_best_of_3"
)

i=0
for cell in "${CELLS[@]}"; do
  i=$(( i + 1 ))
  IFS='|' read -r inst cap cond <<<"$cell"
  echo
  echo "=== [${i}/${#CELLS[@]}] ${cond} @ ${cap} on ${inst} -- $(date -Is)"
  CELL_START=$(date +%s)
  python scripts/run_pilot_sweep.py \
    --subset "${SUBSET}" --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL}" --condition "${cond}" --caps "${cap}" \
    --instance-ids "${inst}" --out "${OUT_DIR}"
  echo "=== [${i}/${#CELLS[@]}] wall $(( $(date +%s) - CELL_START ))s"
  # Rewritten every time: a walltime kill still leaves what was measured.
  summarise
done

echo
echo "================= CALIBRATION COMPLETE ================="
echo "  queue wait   : ${QUEUE_WAIT_S:-unknown} s"
echo "  model load   : ${MODEL_LOAD_S} s"
echo "  job elapsed  : $(( $(date +%s) - JOB_START_EPOCH )) s"
echo "  summary      : ${SUMMARY}"
echo "  run docs     : ${OUT_DIR}"
echo "  standing     : timing only -- NOT a thesis outcome"
echo "======================================================="
