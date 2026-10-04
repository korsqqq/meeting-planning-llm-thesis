#!/bin/bash
# scripts/hpc/measure_startup.slurm
#
# How much of the ~347 s per-invocation overhead is Hub traffic, and how much is
# just importing Python off BeeGFS?
#
#   mkdir -p logs/hpc && sbatch scripts/hpc/measure_startup.slurm
#
# CPU only: it never contacts a GPU or an endpoint. Two invocation shapes are
# timed, each with HF_HUB_OFFLINE unset and then set:
#
#   --dry-run   returns before `from src.core import ...`, so it measures the
#               floor: interpreter start, top-level imports, manifest load and
#               oracle re-verification. No tokenizer is built.
#   probe       the same command with one instance, pointed at a dead endpoint,
#               so it DOES import src.core and build the pinned tokenizer, then
#               fails on connection. The delta over --dry-run is the cost of the
#               transformers import plus tokenizer resolution.
#
# The offline pair only works once bootstrap.sh has cached the pinned counting
# tokenizer; before that it is expected to fail, and that failure is itself the
# answer to "is the tokenizer in the cache".

#SBATCH --job-name=mas-startup
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/hpc/startup-%j.out
#SBATCH --error=logs/hpc/startup-%j.out

set -uo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
[ -d "$HF_HOME" ] || export HF_HOME="$HOME/hf-cache"

SUBSET="results/manifests/subset__bands_budget_dev__911e4864d6fb.json"
INSTANCE="random-n8-t80-o100-s30000"
DEAD_URL="http://127.0.0.1:1/v1"     # nothing listens on port 1

# shellcheck disable=SC1091
source .venv-harness/bin/activate

echo "=================================================================="
echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "HF_HOME : ${HF_HOME}"
echo "commit  : $(git rev-parse --short HEAD)"
echo "=================================================================="

# Is the pinned counting tokenizer actually in the cache?
TOK_DIR="${HF_HOME}/hub/models--Qwen--Qwen3-8B-AWQ"
if [ -d "$TOK_DIR" ]; then
  echo "counting tokenizer cache : present ($(du -sh "$TOK_DIR" 2>/dev/null | cut -f1))"
else
  echo "counting tokenizer cache : ABSENT -- offline runs below will fail; re-run bootstrap.sh"
fi
echo

# Nanosecond epochs and integer arithmetic: `bc` is not installed on the compute
# nodes, and job 1892585 silently reported every timing as 0.0 s because of it.
time_it() {
  local label="$1"; shift
  local log start end ms rc note
  log=$(mktemp)
  start=$(date +%s%N)
  "$@" >"$log" 2>&1
  rc=$?
  end=$(date +%s%N)
  ms=$(( (end - start) / 1000000 ))

  # A non-zero exit is expected for the probe (dead endpoint). Distinguish that
  # from a tokenizer that could not load, which would look identical otherwise:
  # the runner prints the pinned tokenizer line just before it builds the client.
  note="-"
  if grep -q "counting tokenizer:" "$log"; then
    note="tokenizer OK"
  elif [ "$rc" -ne 0 ]; then
    note="FAILED BEFORE the tokenizer -- see below"
  fi
  if grep -qiE "connection|refused|APIConnection" "$log"; then
    note="${note}, endpoint unreachable (expected)"
  fi
  printf "  %-34s %4d.%03ds   exit %d   %s\n" \
    "$label" $((ms / 1000)) $((ms % 1000)) "$rc" "$note"
  if [ "$note" = "FAILED BEFORE the tokenizer -- see below" ]; then
    sed 's/^/        /' "$log" | tail -8
  fi
  rm -f "$log"
}

DRY=(python scripts/run_pilot_sweep.py --subset "$SUBSET" --condition c1_react
     --caps 64000 --instance-ids "$INSTANCE" --out results/logs/_startup_probe --dry-run)
PROBE=(python scripts/run_pilot_sweep.py --subset "$SUBSET" --condition c1_react
       --caps 64000 --instance-ids "$INSTANCE" --out results/logs/_startup_probe
       --base-url "$DEAD_URL" --request-timeout 5)

echo "--- HF_HUB_OFFLINE unset (current behaviour)"
unset HF_HUB_OFFLINE
time_it "dry-run (no src.core import)" "${DRY[@]}"
time_it "dry-run, second time (warm)"  "${DRY[@]}"
time_it "full import + tokenizer"      "${PROBE[@]}"

echo
echo "--- HF_HUB_OFFLINE=1"
export HF_HUB_OFFLINE=1
time_it "dry-run (no src.core import)" "${DRY[@]}"
time_it "full import + tokenizer"      "${PROBE[@]}"
time_it "full import + tokenizer (warm)" "${PROBE[@]}"

# The probe cannot write a result -- it never reaches an endpoint -- but remove the
# directory anyway so nothing stray is left under results/logs/.
rm -rf results/logs/_startup_probe

echo
echo "=================================================================="
echo "Read it as: dry-run is the import/verify floor; the gap to the full"
echo "invocation is transformers + tokenizer; the gap between the offline"
echo "and online pairs is Hub traffic that HF_HUB_OFFLINE=1 removes."
echo "=================================================================="
