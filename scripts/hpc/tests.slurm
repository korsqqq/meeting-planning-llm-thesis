#!/bin/bash
# scripts/hpc/tests.slurm
#
# The offline test suite, as a CPU batch job.
#
#   mkdir -p logs/hpc && sbatch scripts/hpc/tests.slurm
#
# Why not on the frontend: the frontend is a shared login node for the whole
# university. It is for editing, transferring and submitting -- not for running
# a few hundred CP-SAT solves. The suite is CPU-only by construction (generator,
# oracle, budget ledger, agents against a fake client, scorer), so it does not
# belong in a GPU allocation either: that would hold a scarce GPU idle.
#
# This job must be green before any GPU time is spent.

#SBATCH --job-name=mas-tests
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/hpc/tests-%j.out
#SBATCH --error=logs/hpc/tests-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

echo "=================================================================="
echo "job       : ${SLURM_JOB_ID} on $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION}"
echo "commit    : $(git rev-parse HEAD 2>/dev/null || echo NO_GIT)"
echo "branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo NO_GIT)"
echo "started   : $(date -Is)"
echo "=================================================================="

# shellcheck disable=SC1091
source .venv-harness/bin/activate
python -c "import sys, pydantic, ortools, langgraph; print('python', sys.version.split()[0])"

# THESIS_DECISIONS section 2: before the pilot the parity evidence is MANDATORY.
# Without this flag a tokenizer that cannot load makes the parity test SKIP, and
# "all green" would silently mean the evidence never ran -- exactly what happened
# in job 1887836 (823 passed, 1 skipped). With it, a load failure FAILS.
export REQUIRE_TOKENIZER_PARITY="${REQUIRE_TOKENIZER_PARITY:-1}"
echo "REQUIRE_TOKENIZER_PARITY=${REQUIRE_TOKENIZER_PARITY}"

# The parity suite first and verbosely, so the archived evidence names the cases.
echo
echo "--- tokenizer parity evidence"
python -m pytest tests/test_tokenizer.py -v

echo
echo "--- full offline suite"
python -m pytest -q --durations=10

echo
echo "=================================================================="
echo "tests finished at $(date -Is)"
echo "elapsed: ${SECONDS}s"
echo "=================================================================="
