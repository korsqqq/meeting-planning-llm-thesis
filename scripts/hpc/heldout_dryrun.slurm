#!/bin/bash
# scripts/hpc/heldout_dryrun.slurm
#
# Everything that must be green before the held-out array is submitted, in one
# CPU job. It never contacts an endpoint and never writes a run document.
#
#   sbatch scripts/hpc/heldout_dryrun.slurm
#
# Five checks, in the order a failure is cheapest to find:
#
#   1. design audit -- 198 instances, the frozen O histogram, 3960 cells, and
#      the shard split over all four manifests at once
#   2. seed-range guard, --expect held_out, on every manifest
#   3. shard plan per manifest, the shipped check in its own right
#   4. expected-runs manifests -- the completion criterion, before the run
#   5. end-to-end --dry-run on shard 0 -- manifest verification and ordering
#
# (1) and (3) are what a per-shard dry run cannot do: each invocation sees only
# its own slice, so a dropped or duplicated cell looks healthy in all 24 logs and
# surfaces only at the audit, after the campaign. (1) additionally spans
# manifests, which (3) cannot: it is the union that must be complete and disjoint.
#
# Deliberately CPU-only: nothing here needs a GPU, and holding an H200 to check
# arithmetic would be waste.

#SBATCH --job-name=mas-heldout-dry
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/hpc/heldout-dryrun-%j.out
#SBATCH --error=logs/hpc/heldout-dryrun-%j.out

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/thesis-mas-planning}"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
[ -d "$HF_HOME" ] || export HF_HOME="$HOME/hf-cache"

SUBSETS="${SUBSETS:-results/manifests/subset__bands_held_out_n8__278ce0f2e8fe.json results/manifests/subset__heldout_n4__54f3f48d6232.json results/manifests/subset__heldout_n5__4b91dee07c34.json results/manifests/subset__heldout_n6__44418ababc22.json}"
NUM_SHARDS="${NUM_SHARDS:-24}"
CAPS="${CAPS:-16000,32000,64000,128000}"
CONDITIONS_CSV="${CONDITIONS_CSV:-c1_react,c2_verify_revise,c3_mas,c4_planner_critic,c5_best_of_3}"

# shellcheck disable=SC1091
source .venv-harness/bin/activate

echo "=================================================================="
echo "HELD-OUT PRE-FLIGHT -- no endpoint contacted, no run document written"
echo "job     : ${SLURM_JOB_ID} on $(hostname)"
echo "subsets : $(echo ${SUBSETS} | wc -w) manifests"
echo "shards  : ${NUM_SHARDS}   caps ${CAPS}"
echo "commit  : $(git rev-parse HEAD)"
echo "=================================================================="

echo
echo "### 1. design audit (all four manifests together)"
python scripts/hpc/audit_heldout_design.py --num-shards "${NUM_SHARDS}"

echo
echo "### 2. seed-range guard (expect held_out), every manifest"
for sub in ${SUBSETS}; do
  python scripts/hpc/guard_seed_range.py "${sub}" --expect held_out
done

echo
echo "### 3. shard plan, per manifest"
for sub in ${SUBSETS}; do
  echo "--- $(basename "${sub}")"
  python scripts/hpc/verify_shard_plan.py     --subset "${sub}" --caps "${CAPS}"     --num-shards "${NUM_SHARDS}" --conditions "${CONDITIONS_CSV}"
done

echo
echo "### 4. expected-runs manifests (the completion criterion)"
TOTAL_EXPECTED=0
for sub in ${SUBSETS}; do
  tag="$(basename "${sub}" .json | sed -e 's/^subset__//' -e 's/__[0-9a-f]*$//')"
  out="results/manifests/expected_runs__${tag}.json"
  python scripts/run_pilot_sweep.py     --subset "${sub}" --caps "${CAPS}"     --conditions "${CONDITIONS_CSV}"     --write-expected-runs "${out}"
  n="$(python -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(len(d.get('expected_runs') or d.get('runs') or []))" "${out}")"
  echo "  ${out}: ${n}"
  TOTAL_EXPECTED=$(( TOTAL_EXPECTED + n ))
done
echo "total expected run ids: ${TOTAL_EXPECTED}"
[ "${TOTAL_EXPECTED}" -eq 3960 ] || { echo "!!! expected 3960, got ${TOTAL_EXPECTED}"; exit 9; }

echo
echo "### 5. end-to-end dry run, shard 0, every manifest"
for sub in ${SUBSETS}; do
  echo "--- $(basename "${sub}")"
  python scripts/run_pilot_sweep.py     --subset "${sub}" --caps "${CAPS}"     --condition c1_react     --shard-index 0 --num-shards "${NUM_SHARDS}"     --out results/logs/heldout --dry-run
done

echo
echo "=================================================================="
echo "PRE-FLIGHT COMPLETE -- review the five sections above before GO."
echo "After the campaign, close the loop with:"
echo "  python scripts/audit_expected_runs.py \\"
for sub in ${SUBSETS}; do
  tag="$(basename "${sub}" .json | sed -e 's/^subset__//' -e 's/__[0-9a-f]*$//')"
  echo "      --expected results/manifests/expected_runs__${tag}.json \\"
done
echo "      --runs results/logs/heldout \\"
echo "      --out results/analysis/heldout/completeness_audit.json"
echo "  (all four manifests in ONE call -- they share an output directory)"
echo "=================================================================="
