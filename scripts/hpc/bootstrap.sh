#!/usr/bin/env bash
# scripts/hpc/bootstrap.sh
#
# One-time environment bootstrap on the TU Berlin HPC cluster.
#
# Run this ON THE FRONTEND (frontend02), never inside a Slurm job:
# it only downloads and installs, and the frontend is the node that is
# allowed to do that. Compute nodes do have internet here, but making
# every array task re-resolve wheels would be both slow and a source of
# version drift.
#
#   ssh korsun_a@gateway.hpc.tu-berlin.de
#   cd ~/thesis-mas-planning
#   bash scripts/hpc/bootstrap.sh
#
# Idempotent: safe to re-run. Existing venvs are reused, an already
# downloaded checkpoint is not fetched again.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
# Model weights are reproducible data, so they belong on scratch rather than in
# the mirrored home directory (see the cluster intro, "Network Storage").
# Falls back to $HOME when /scratch is not writable for this account.
HF_CACHE_DEFAULT="/scratch/$USER/hf-cache"
if ! mkdir -p "$HF_CACHE_DEFAULT" 2>/dev/null; then
  HF_CACHE_DEFAULT="$HOME/hf-cache"
fi
HF_CACHE="${HF_CACHE:-$HF_CACHE_DEFAULT}"

# Exact checkpoint commit, from SETUP.md section 3. The revision, not the repo
# name, is the reproducible identifier: chat templates change on the Hub
# independently of the weights.
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
MODEL_REVISION="${MODEL_REVISION:-0499c3ac83fdef8810b907a23894ba91e95eddd8}"

# The BUDGET-COUNTING tokenizer, which is a different repo from the served model.
# THESIS_DECISIONS section 2: counting uses Qwen/Qwen3-8B-AWQ's tokenizer -- AWQ
# repos ship their own tokenizer files and revisions, and 8B/32B parity is asserted
# by test, not assumed. Only tokenizer files are fetched; the 8B weights are neither
# served nor needed.
TOKENIZER_REPO="${TOKENIZER_REPO:-Qwen/Qwen3-8B-AWQ}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-4da05a8edb55c6046cce958586c33b61da07bb79}"

echo "=================================================================="
echo "project root : ${PROJECT_ROOT}"
echo "HF cache     : ${HF_CACHE}"
echo "model        : ${MODEL}@${MODEL_REVISION}"
echo "=================================================================="

cd "$PROJECT_ROOT"
mkdir -p logs/hpc "$HF_CACHE"

# ---------------------------------------------------------------- 1. uv
# uv brings its own Python 3.12, so the cluster's `module load python`
# version does not matter. src/schemas/ needs >= 3.11 for enum.StrEnum.
if ! command -v uv >/dev/null 2>&1; then
  echo "--- installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# shellcheck disable=SC1091
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "--- installing Python 3.12"
uv python install 3.12

# ------------------------------------------------- 2. harness environment
# CPU-only: generator, oracle, budget ledger, agents, scorer, tests.
# Kept separate from vLLM so a serving upgrade can never change the
# tokenizer used for budget accounting.
if [ ! -d .venv-harness ]; then
  echo "--- creating .venv-harness"
  uv venv --python 3.12 .venv-harness
fi
echo "--- installing harness requirements (locked)"
VIRTUAL_ENV="$PROJECT_ROOT/.venv-harness" uv pip install -r requirements/harness.lock.txt
VIRTUAL_ENV="$PROJECT_ROOT/.venv-harness" uv run python -c \
  "import sys, pydantic, ortools, langgraph; print('harness python', sys.version.split()[0])"

# ------------------------------------------------- 3. serving environment
# vLLM pins its own torch/transformers. Installed from the LOCK file so the
# held-out run uses byte-identical versions to the frozen record.
if [ ! -d .venv-vllm ]; then
  echo "--- creating .venv-vllm"
  uv venv --python 3.12 .venv-vllm
fi
echo "--- installing serving requirements (locked) -- this pulls several GB"
VIRTUAL_ENV="$PROJECT_ROOT/.venv-vllm" uv pip install -r requirements/serving.lock.txt
VIRTUAL_ENV="$PROJECT_ROOT/.venv-vllm" uv run python -c \
  "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# ------------------------------------------------------ 4. checkpoint
echo "--- downloading ${MODEL} at pinned revision into ${HF_CACHE}"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
VIRTUAL_ENV="$PROJECT_ROOT/.venv-vllm" uv run python - <<PY
from huggingface_hub import constants, snapshot_download

# cache_dir is deliberately NOT passed. Passing HF_HOME here would override the
# default and write to \$HF_HOME/models--.../, one level ABOVE where every reader
# looks: with only HF_HOME set, huggingface_hub resolves HF_HUB_CACHE to
# \$HF_HOME/hub. vLLM would then miss the cache and go back to the Hub -- which is
# what cost the first smoke job ~9 minutes, and what would make HF_HUB_OFFLINE=1
# fail outright.
path = snapshot_download("${MODEL}", revision="${MODEL_REVISION}")
print("HF_HUB_CACHE  :", constants.HF_HUB_CACHE)
print("checkpoint at :", path)
PY

# ------------------------------------------ 5. counting tokenizer (8B, no weights)
echo "--- caching counting tokenizer ${TOKENIZER_REPO}@${TOKENIZER_REVISION} (files only)"
VIRTUAL_ENV="$PROJECT_ROOT/.venv-vllm" uv run python - <<PY
import os
from huggingface_hub import snapshot_download

# allow_patterns keeps this to tokenizer/config files. The 8B safetensors are ~6 GB
# and are never served -- the harness only needs to render the chat template and
# count ids. model.safetensors.index.json matches *.json and is a few KB.
path = snapshot_download(
    "${TOKENIZER_REPO}",
    revision="${TOKENIZER_REVISION}",
    allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"],
)
total = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(path) for f in files
)
print("tokenizer at  :", path)
print(f"fetched       : {total / 1024 / 1024:.1f} MiB (no weights)")
weights = [f for _, _, fs in os.walk(path) for f in fs if f.endswith((".safetensors", ".bin"))]
print("weight files  :", weights or "none -- correct")
PY

echo
echo "=================================================================="
echo "bootstrap complete."
echo
echo "Put this in ~/.bashrc so every job inherits the same cache:"
echo "    export HF_HOME=${HF_CACHE}"
echo
echo "Next steps, in this order:"
echo "    1. sbatch scripts/hpc/tests.slurm    # offline suite on a CPU node"
echo "    2. sbatch scripts/hpc/smoke.slurm    # only once (1) is green"
echo
echo "The test suite deliberately does NOT run here: the frontend is a shared"
echo "login node, and the suite is a few hundred CP-SAT solves."
echo "=================================================================="
