#!/bin/bash
# scripts/hpc/smoke.slurm
#
# First real batch job: bring vLLM up INSIDE the allocation, prove the harness
# can talk to it, and measure what the held-out planning decision needs.
#
#   mkdir -p logs/hpc && sbatch scripts/hpc/smoke.slurm            # H200 (default)
#   sbatch -p gpu --gres=gpu:a100:1 scripts/hpc/smoke.slurm        # A100 instead
#
# Produces NO thesis numbers and touches NO held-out data. It runs a handful of
# already-used pilot instances into a separate directory, purely to time them.
#
# Four measurements are reported separately, because they are four different
# planning quantities and averaging them together would hide the one that bites:
#
#   1. queue_wait_seconds        how long Slurm made us wait before starting
#   2. job_elapsed_seconds       total wall-clock inside the allocation
#   3. per_run_latency_seconds   end-to-end seconds per sweep run
#   4. vllm_decode_tokens_per_s  raw generation speed of this GPU
#
# (1) determines how many array tasks are worth submitting; (2) determines the
# --time to request; (3) is what turns 2,760 runs into GPU-hours; (4) is the
# hardware comparison against the A6000 box.
#
# NOTE: Slurm does not create the --output directory. `mkdir -p logs/hpc` first.

#SBATCH --job-name=mas-smoke
#SBATCH --partition=h200_short
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hpc/smoke-%j.out
#SBATCH --error=logs/hpc/smoke-%j.out

set -euo pipefail
JOB_START_EPOCH=$(date +%s)

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

# Preflight, so a missing prerequisite reports itself instead of surfacing later
# as a confusing "compute capability unknown".
for venv in .venv-vllm .venv-harness; do
  [ -x "$venv/bin/python" ] || { echo "!!! ${venv} missing -- run scripts/hpc/bootstrap.sh first"; exit 4; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "!!! nvidia-smi not found -- this is not a GPU node"; exit 4; }

export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
[ -d "$HF_HOME" ] || export HF_HOME="$HOME/hf-cache"

export MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
export MODEL_REVISION="${MODEL_REVISION:-0499c3ac83fdef8810b907a23894ba91e95eddd8}"

# The checkpoint must already sit in the cache vLLM actually reads. With only
# HF_HOME set, huggingface_hub resolves HF_HUB_CACHE to $HF_HOME/hub -- NOT
# $HF_HOME. Checking it here converts a silent 9-minute Hub round-trip (and, with
# HF_HUB_OFFLINE=1, an outright failure after the model load has started) into an
# immediate, actionable error.
SNAPSHOT_DIR="${HF_HOME}/hub/models--${MODEL//\//--}/snapshots/${MODEL_REVISION}"
if [ ! -s "${SNAPSHOT_DIR}/config.json" ]; then
  echo "!!! pinned checkpoint is not in the cache vLLM reads:"
  echo "!!!   ${SNAPSHOT_DIR}"
  echo "!!! re-run scripts/hpc/bootstrap.sh -- it now writes to \$HF_HOME/hub"
  exit 6
fi

# Development instances only. The held-out pool is not generated and the
# reserved seed range (100000+) is untouched; nothing here may read or write it.
SUBSET="${SUBSET:-results/manifests/subset__pilot__29d836cb5614.json}"
EXPECT_SUBSET_HASH="${EXPECT_SUBSET_HASH:-29d836cb5614}"
SMOKE_N_RUNS="${SMOKE_N_RUNS:-2}"
SMOKE_CAP="${SMOKE_CAP:-32000}"
SMOKE_CONDITION="${SMOKE_CONDITION:-c1_react}"

# One directory per job. The runner's strict resume is a feature for the real
# sweep and a trap here: a second smoke would find the first smoke's results
# already on disk, skip the runs, and report a latency it never measured.
# The final experiment does the opposite -- one directory plus resume, so an
# interrupted sweep continues instead of restarting.
OUT_DIR="results/logs/hpc_smoke/${SLURM_JOB_ID}"

echo "=================================================================="
echo "job          : ${SLURM_JOB_ID} on $(hostname)"
echo "partition    : ${SLURM_JOB_PARTITION}"
echo "HF_HOME      : ${HF_HOME}"
echo "subset       : ${SUBSET}   (development pilot set -- NOT held-out)"
echo "started      : $(date -Is)"
echo "=================================================================="

# ------------------------------------------------- held-out content guard
# Filename checks are not a safeguard -- a held-out manifest called test.json
# would pass one. This reads the manifest and rejects ANY instance whose seed
# is at or above the locked held-out floor of 100000.
echo "--- held-out guard"
python3 scripts/hpc/guard_seed_range.py "${SUBSET}" --expect development \
  --expect-content-hash "${EXPECT_SUBSET_HASH}"

# ------------------------------------------------------- 1. queue wait
SLURM_LINE="$(scontrol show job "${SLURM_JOB_ID}" -o 2>/dev/null || true)"
SUBMIT_TIME="$(echo "$SLURM_LINE" | grep -o 'SubmitTime=[^ ]*' | head -1 | cut -d= -f2- || true)"
START_TIME="$(echo "$SLURM_LINE" | grep -o 'StartTime=[^ ]*' | head -1 | cut -d= -f2- || true)"
QUEUE_WAIT_S=""
if [ -n "$SUBMIT_TIME" ] && [ -n "$START_TIME" ]; then
  S=$(date -d "$SUBMIT_TIME" +%s 2>/dev/null || echo "")
  T=$(date -d "$START_TIME"  +%s 2>/dev/null || echo "")
  [ -n "$S" ] && [ -n "$T" ] && QUEUE_WAIT_S=$(( T - S ))
fi
echo "--- [metric 1] queue wait: ${QUEUE_WAIT_S:-unknown} s  (submitted ${SUBMIT_TIME:-?}, started ${START_TIME:-?})"

# ------------------------------------------ 2. hardware gate, before any work
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
COMPUTE_CAP="$(.venv-vllm/bin/python -c \
  'import torch;print("%d.%d"%torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else print("none")' \
  2>/dev/null || echo "unknown")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
echo "--- gpu: ${GPU_NAME} | compute capability ${COMPUTE_CAP}"

# awq_marlin needs SM 8.0+. On a P100 (6.0) or V100 (7.0) the AWQ path does not
# exist, and the 32B checkpoint does not fit in 16 GB anyway. Fail here rather
# than after a 20 GB model load.
case "$COMPUTE_CAP" in
  6.*|7.*|none|unknown)
    echo "!!! compute capability ${COMPUTE_CAP} cannot serve AWQ INT4 -- request an H200 or A100"
    echo "!!! hint: --gres=gpu:1 hands out P100s on this cluster; name the type explicitly"
    exit 3 ;;
esac
case "$GPU_NAME" in
  *H200*) echo "--- H200 confirmed: this is the hardware the primary held-out run is meant to use" ;;
  *)      echo "--- NOTE: this is ${GPU_NAME}, not an H200. Fine for a smoke test." ;;
esac

# ---------------------------------------------------------------- port
# Ask the kernel for a free port instead of hardcoding 8000: several array
# tasks can land on the same node, and a fixed port would collide.
PORT=$(python3 - <<'PY'
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0))
print(s.getsockname()[1]); s.close()
PY
)
echo "--- vLLM port: ${PORT}"

# ------------------------------------------------------------- start vLLM
# Backgrounded inside the allocation. The trap guarantees teardown even if the
# harness crashes or the job hits its walltime; otherwise a stray process would
# hold the GPU until Slurm reaps it.
VLLM_LOG="logs/hpc/vllm-${SLURM_JOB_ID}.log"

# Recorded verbatim in the provenance file. "vllm 0.19.1" alone does not say
# which flags produced the numbers; prefix caching in particular changes them.
# HF_HUB_OFFLINE=1: the checkpoint is pinned and already cached (verified above),
# so every Hub round-trip at startup is pure latency. The first smoke job spent
# ~9 minutes there. Offline also turns "cache miss" into an immediate error
# instead of a silent re-download.
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

# ------------------------------------------------------------ wait for it
# Poll /health, but check liveness every iteration. Without the kill -0 guard a
# dead server leaves this loop spinning until the walltime ends, and the log
# shows nothing but silence.
# 60 minutes, not 30. Job 1887852 needed ~15 min just to import torch and
# initialise CUDA from BeeGFS on a cold node, before vLLM printed its first line,
# and was killed mid-load at the 30-minute mark. Startup cost here is filesystem
# and import latency, not model size, so the ceiling has to allow for a cold node.
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
echo "--- vLLM healthy after ${MODEL_LOAD_S}s (model load + warmup)"

# ------------------------------------------------------------ provenance
# Full environment record for this job: git, model revision, nvidia-smi, driver,
# compute capability, torch, torch CUDA, vLLM, and the quantisation lines quoted
# verbatim from the startup log.
PROV_JSON="logs/hpc/provenance-${SLURM_JOB_ID}.json"
bash scripts/hpc/provenance.sh "${PROV_JSON}" "${VLLM_LOG}"

# --------------------------------------------------- 4. decode throughput
# The FIRST request a fresh vLLM server sees is the least representative one:
# with CompilationMode.VLLM_COMPILE and lazy CUDA-graph capture it pays inductor
# compilation on top of decoding. Job 1890125 timed exactly that request and
# reported 18.9 tok/s, which is not a decode rate. One identical request is now
# issued and discarded, then two are timed.
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

echo "--- [metric 4] decode throughput: 1 warmup (discarded) + 2 timed, 512 new tokens each"
WARMUP_TOK_S=$(probe_tok_s)
echo "    warmup   : ${WARMUP_TOK_S:-unavailable} tok/s  (discarded)"
TOK_PER_S_1=$(probe_tok_s)
echo "    timed #1 : ${TOK_PER_S_1:-unavailable} tok/s"
TOK_PER_S_2=$(probe_tok_s)
echo "    timed #2 : ${TOK_PER_S_2:-unavailable} tok/s"

# ------------------------------------------------------------ harness
# --dry-run touches no GPU: it proves the manifest resolves, the subset hash
# matches and the output layout is what the analyser will later expect.
# shellcheck disable=SC1091
source .venv-harness/bin/activate
echo "--- harness dry run"
python scripts/run_pilot_sweep.py \
  --subset "${SUBSET}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${MODEL}" --out "${OUT_DIR}" --dry-run

# ------------------------------------------------ 3. per-run latency
# Take the first N instance ids from the frozen development manifest, in
# manifest order, so the choice is deterministic and not a judgement call.
INSTANCE_IDS=$(SUBSET="$SUBSET" N="$SMOKE_N_RUNS" python3 -c '
import json, os
d = json.load(open(os.environ["SUBSET"]))
ids = [i["instance_id"] for lvl in d["instances"].values() for i in lvl]
print(",".join(ids[: int(os.environ["N"])]))
')
echo "--- [metric 3] timing ${SMOKE_N_RUNS} real runs: ${SMOKE_CONDITION} @ cap ${SMOKE_CAP}"
echo "    instances: ${INSTANCE_IDS}"
python scripts/run_pilot_sweep.py \
  --subset "${SUBSET}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${MODEL}" --condition "${SMOKE_CONDITION}" \
  --caps "${SMOKE_CAP}" --instance-ids "${INSTANCE_IDS}" --out "${OUT_DIR}"

# ------------------------------------------------------------- summary
JOB_ELAPSED_S=$(( $(date +%s) - JOB_START_EPOCH ))
SUMMARY="logs/hpc/smoke-${SLURM_JOB_ID}-summary.json"

OUT_DIR="$OUT_DIR" PROV_JSON="$PROV_JSON" SUMMARY="$SUMMARY" \
QUEUE_WAIT_S="${QUEUE_WAIT_S:-}" JOB_ELAPSED_S="$JOB_ELAPSED_S" \
MODEL_LOAD_S="$MODEL_LOAD_S" \
TOK_WARMUP="${WARMUP_TOK_S:-}" TOK_1="${TOK_PER_S_1:-}" TOK_2="${TOK_PER_S_2:-}" \
python3 - <<'PY'
import json, os, glob

def maybe_num(k, cast=float):
    v = os.environ.get(k, "")
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None

runs = []
for path in sorted(glob.glob(os.path.join(os.environ["OUT_DIR"], "*.json"))):
    try:
        doc = json.load(open(path, encoding="utf-8"))
        rr = doc["run_result"]
        tokens = rr.get("tokens") or {}
        runs.append({
            "instance_id": rr.get("instance_id"),
            "condition": rr.get("condition"),
            "cap": rr.get("cap"),
            "latency_seconds": rr.get("latency_seconds"),
            "tokens_total": tokens.get("total"),
            # rr["calls"] is the full per-call trace; only its length belongs here
            "n_calls": tokens.get("n_calls") or len(rr.get("calls") or []),
            "satisfaction": (rr.get("score") or {}).get("satisfaction"),
        })
    except Exception as exc:
        runs.append({"file": os.path.basename(path), "parse_error": str(exc)})

lat = [r["latency_seconds"] for r in runs if isinstance(r.get("latency_seconds"), (int, float))]

try:
    provenance = json.load(open(os.environ["PROV_JSON"], encoding="utf-8"))
except Exception:
    provenance = None

timed = [v for v in (maybe_num("TOK_1"), maybe_num("TOK_2")) if v is not None]

summary = {
    "schema_version": "hpc_smoke_summary/1.1",
    "standing": "smoke test on development instances; no thesis numbers, no held-out data",
    "metrics": {
        "queue_wait_seconds": maybe_num("QUEUE_WAIT_S", int),
        "job_elapsed_seconds": maybe_num("JOB_ELAPSED_S", int),
        "model_load_seconds": maybe_num("MODEL_LOAD_S", int),
        "vllm_decode_tokens_per_s": {
            # The warmup value is kept for the record, never averaged in: it is
            # the compilation-contaminated first request.
            "warmup_discarded": maybe_num("TOK_WARMUP"),
            "values": timed,
            "mean": (sum(timed) / len(timed)) if timed else None,
        },
        "per_run_latency_seconds": {
            "values": lat,
            "n": len(lat),
            "mean": (sum(lat) / len(lat)) if lat else None,
            "max": max(lat) if lat else None,
        },
    },
    "runs": runs,
    "provenance": provenance,
}
with open(os.environ["SUMMARY"], "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)

m = summary["metrics"]
print()
print("================= SMOKE SUMMARY =================")
print(f"  1. queue wait          : {m['queue_wait_seconds']} s")
print(f"  2. job elapsed         : {m['job_elapsed_seconds']} s  (of which model load {m['model_load_seconds']} s)")
print(f"  3. per-run latency     : n={m['per_run_latency_seconds']['n']} "
      f"mean={m['per_run_latency_seconds']['mean']} max={m['per_run_latency_seconds']['max']}")
d = m["vllm_decode_tokens_per_s"]
print(f"  4. vLLM decode         : {d['values']} -> mean {d['mean']} tok/s "
      f"(warmup {d['warmup_discarded']} discarded)")
print(f"  written to             : {os.environ['SUMMARY']}")
print("================================================")
PY

echo
echo "smoke complete at $(date -Is)"
