#!/usr/bin/env bash
# scripts/hpc/provenance.sh
#
# Capture the complete environment record for ONE GPU job, as JSON.
#
#   bash scripts/hpc/provenance.sh <out.json> [vllm_startup_log]
#
# Every GPU job must call this. The run documents written by the harness carry
# the git commit and the subset hash, but they say nothing about the machine
# that produced them: on a cluster the same code can land on four different GPU
# architectures, and "which kernel actually ran" is not recoverable after the
# fact. That is what this file is for.
#
# Recorded, per the held-out provenance requirement:
#   git commit + branch + dirty flag   model repo + pinned revision
#   nvidia-smi output                  driver version
#   GPU name + compute capability      torch version + torch CUDA version
#   vLLM version                       quantisation method / backend actually selected
#   Slurm job id, partition, node, submit/start times, queue wait
#
# The quantisation field is quoted verbatim from the vLLM startup log rather
# than asserted, so a fallback kernel shows up as itself instead of being
# silently reported as the intended one.

set -euo pipefail

OUT="${1:?usage: provenance.sh <out.json> [vllm_startup_log]}"
VLLM_LOG="${2:-}"
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VLLM_PY="${VLLM_PY:-$PROJECT_ROOT/.venv-vllm/bin/python}"

mkdir -p "$(dirname "$OUT")"

# ---------------------------------------------------------------- git
# Requires .git to be present on the cluster; the transfer must not exclude it.
GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo "NO_GIT")"
GIT_BRANCH="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "NO_GIT")"
# `|| true` matters: under `set -o pipefail` a git failure outside a repository
# would abort the whole provenance capture instead of recording NO_GIT.
GIT_DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | head -c 1 || true)"
[ -n "$GIT_DIRTY" ] && GIT_DIRTY="true" || GIT_DIRTY="false"

# ---------------------------------------------------------------- slurm
SLURM_LINE="$(scontrol show job "${SLURM_JOB_ID:-0}" -o 2>/dev/null || true)"
extract() { echo "$SLURM_LINE" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2- || true; }
SUBMIT_TIME="$(extract SubmitTime)"
START_TIME="$(extract StartTime)"

QUEUE_WAIT_S=""
if [ -n "$SUBMIT_TIME" ] && [ -n "$START_TIME" ]; then
  S=$(date -d "$SUBMIT_TIME" +%s 2>/dev/null || echo "")
  T=$(date -d "$START_TIME" +%s 2>/dev/null || echo "")
  [ -n "$S" ] && [ -n "$T" ] && QUEUE_WAIT_S=$(( T - S ))
fi

# ---------------------------------------------------------------- gpu
NVIDIA_SMI="$(nvidia-smi 2>&1 || echo "nvidia-smi unavailable")"
GPU_CSV="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "")"

# ------------------------------------------------- quantisation, verbatim
QUANT_LINES=""
STARTUP_HEADER=""
if [ -n "$VLLM_LOG" ] && [ -f "$VLLM_LOG" ]; then
  QUANT_LINES="$(grep -i -E "awq|marlin|quantization|quant_method|kernel" "$VLLM_LOG" | head -12 || true)"
  # serve_vllm.sh echoes its resolved configuration before exec'ing vllm.
  STARTUP_HEADER="$(head -20 "$VLLM_LOG" || true)"
fi

# ------------------------------------------------- dependency set identity
# Two installations can both report "vllm 0.19.1 / torch 2.10" and still differ
# in every transitive dependency. The lock-file hash is what makes
# "software environment = this exact frozen dependency set" a checkable claim.
lockhash() {
  [ -f "$1" ] || { echo ""; return; }
  sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" 2>/dev/null | cut -d' ' -f1 || echo ""
}
HARNESS_LOCK_SHA="$(lockhash "$PROJECT_ROOT/requirements/harness.lock.txt")"
SERVING_LOCK_SHA="$(lockhash "$PROJECT_ROOT/requirements/serving.lock.txt")"

# ---------------------------------------------------------------- assemble
OUT="$OUT" \
GIT_COMMIT="$GIT_COMMIT" GIT_BRANCH="$GIT_BRANCH" GIT_DIRTY="$GIT_DIRTY" \
MODEL="${MODEL:-}" MODEL_REVISION="${MODEL_REVISION:-}" \
NVIDIA_SMI="$NVIDIA_SMI" GPU_CSV="$GPU_CSV" \
SUBMIT_TIME="$SUBMIT_TIME" START_TIME="$START_TIME" QUEUE_WAIT_S="$QUEUE_WAIT_S" \
QUANT_LINES="$QUANT_LINES" STARTUP_HEADER="$STARTUP_HEADER" \
HARNESS_LOCK_SHA="$HARNESS_LOCK_SHA" SERVING_LOCK_SHA="$SERVING_LOCK_SHA" \
"$VLLM_PY" - <<'PY'
import json, os, datetime

def env(k, default=""):
    v = os.environ.get(k, default)
    return v if v != "" else None

gpu_name = gpu_mem = driver = None
csv = env("GPU_CSV")
if csv:
    parts = [p.strip() for p in csv.splitlines()[0].split(",")]
    if len(parts) >= 3:
        gpu_name, gpu_mem, driver = parts[0], parts[1], parts[2]

torch_version = torch_cuda = compute_cap = vllm_version = None
try:
    import torch
    torch_version = torch.__version__
    torch_cuda = torch.version.cuda
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        compute_cap = f"{major}.{minor}"
        gpu_name = gpu_name or torch.cuda.get_device_name(0)
except Exception as exc:                      # pragma: no cover - environment probe
    torch_version = f"IMPORT FAILED: {exc}"
try:
    import vllm
    vllm_version = vllm.__version__
except Exception as exc:                      # pragma: no cover - environment probe
    vllm_version = f"IMPORT FAILED: {exc}"

qwait = env("QUEUE_WAIT_S")

record = {
    "schema_version": "hpc_provenance/1.0",
    "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git": {
        "commit": env("GIT_COMMIT"),
        "branch": env("GIT_BRANCH"),
        "dirty": os.environ.get("GIT_DIRTY") == "true",
    },
    "model": {
        "repo": env("MODEL"),
        "revision": env("MODEL_REVISION"),
    },
    "slurm": {
        "job_id": env("SLURM_JOB_ID"),
        "job_name": env("SLURM_JOB_NAME"),
        "partition": env("SLURM_JOB_PARTITION"),
        "node": env("SLURMD_NODENAME"),
        "array_job_id": env("SLURM_ARRAY_JOB_ID"),
        "array_task_id": env("SLURM_ARRAY_TASK_ID"),
        "submit_time": env("SUBMIT_TIME"),
        "start_time": env("START_TIME"),
        "queue_wait_seconds": int(qwait) if qwait and qwait.lstrip("-").isdigit() else None,
    },
    "gpu": {
        "name": gpu_name,
        "memory_total": gpu_mem,
        "driver_version": driver,
        "compute_capability": compute_cap,
        # Which physical device Slurm actually handed this job. Without these
        # two, "it ran on an H200" is an assumption about node homogeneity.
        "cuda_visible_devices": env("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": env("SLURM_JOB_GPUS"),
        "slurm_step_gpus": env("SLURM_STEP_GPUS"),
        "nvidia_smi": env("NVIDIA_SMI"),
    },
    "software": {
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "vllm": vllm_version,
        # Version strings do not pin transitive dependencies; these hashes do.
        "harness_lock_sha256": env("HARNESS_LOCK_SHA"),
        "serving_lock_sha256": env("SERVING_LOCK_SHA"),
    },
    "launch": {
        # The exact invocation, and the configuration serve_vllm.sh resolved
        # from it and echoed before exec'ing vllm.
        "vllm_command": env("VLLM_LAUNCH_CMD"),
        "vllm_startup_header": (env("STARTUP_HEADER") or "").splitlines(),
    },
    # Verbatim quotes from the vLLM startup log. NOT an assertion that
    # awq_marlin was used -- read the lines and see.
    "quantization_log_lines": (env("QUANT_LINES") or "").splitlines(),
}

out = os.environ["OUT"]
with open(out, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2, ensure_ascii=False)
print(f"provenance written to {out}")
print(f"  gpu    : {gpu_name} (compute {compute_cap}, driver {driver})")
print(f"  torch  : {torch_version} / cuda {torch_cuda}")
print(f"  vllm   : {vllm_version}")
print(f"  git    : {record['git']['commit']} on {record['git']['branch']} dirty={record['git']['dirty']}")
print(f"  locks  : harness {str(record['software']['harness_lock_sha256'])[:12]} "
      f"serving {str(record['software']['serving_lock_sha256'])[:12]}")
print(f"  devices: CUDA_VISIBLE_DEVICES={record['gpu']['cuda_visible_devices']} "
      f"SLURM_JOB_GPUS={record['gpu']['slurm_job_gpus']}")
print(f"  queue  : waited {record['slurm']['queue_wait_seconds']} s")
PY
