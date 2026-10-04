# tests/test_heldout_array_wiring.py
"""Execution wiring of the held-out array job.

WHY THIS EXISTS. Array 1894448 died on its first shard with

    heldout_array.slurm: line 206: SUBSET: unbound variable

The manifest list had been renamed `SUBSET` -> `SUBSETS` and turned into four
paths, but the runner invocation still passed `--subset "${SUBSET}"`. Nothing
caught it: `bash -n` checks syntax and not unbound variables, and the pre-flight
job exercises `heldout_dryrun.slurm`, which is a different file. The defect was
therefore only reachable by submitting the real campaign to a GPU.

So these tests do what neither of those did: they take the run loop out of the
script and RUN it, under `set -u`, with the runner stubbed, then check what it
scheduled. An undefined variable aborts the loop; a loop over the wrong dimension
shows up as the wrong set of (subset, condition) pairs.

The runner is stubbed with a shell FUNCTION rather than a file on PATH. A stub
executable does not survive Windows, where `chmod` is a no-op and the real
interpreter wins the lookup; a function shadows the name on every platform.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARRAY = REPO / "scripts" / "hpc" / "heldout_array.slurm"
DRYRUN = REPO / "scripts" / "hpc" / "heldout_dryrun.slurm"

CONDITIONS = ("c1_react", "c2_verify_revise", "c3_mas", "c4_planner_critic", "c5_best_of_3")
EXPECTED_MANIFESTS = 4
EXPECTED_CELLS = EXPECTED_MANIFESTS * len(CONDITIONS)

# Provided by Slurm, the environment, or the shell -- never assigned in the script.
EXTERNAL = {
    "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID", "SLURM_JOB_GPUS",
    "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_SUBMIT_DIR", "SLURM_NTASKS",
    "CUDA_VISIBLE_DEVICES", "HOME", "USER", "PATH", "HF_HOME", "TMPDIR", "PROJECT_ROOT",
    "PWD", "RANDOM", "LINENO", "BASH_SOURCE",
}


def read_array() -> str:
    return ARRAY.read_text(encoding="utf-8")


def extract_run_loop() -> str:
    """The block between the markers, verbatim."""
    text = read_array()
    return text[text.index("# --- BEGIN RUN LOOP"):text.index("# --- END RUN LOOP")]


def subsets_of(text: str) -> list[str]:
    line = next(l for l in text.splitlines() if l.startswith("SUBSETS="))
    return line.split(":-", 1)[1].rsplit("}", 1)[0].split()


def declared_subsets() -> list[str]:
    return subsets_of(read_array())


def run_loop(loop: str, record: Path | None) -> subprocess.CompletedProcess:
    """Execute a run-loop body under `set -u` with the runner stubbed out."""
    sink = f'printf "%s\\n" "$*" >> "{str(record).replace(chr(92), "/")}"' if record else ":"
    script = "\n".join([
        "set -euo pipefail",
        f"python() {{ {sink}; }}",
        f'SUBSETS="{" ".join(declared_subsets())}"',
        f'CONDITIONS="{" ".join(CONDITIONS)}"',
        "SHARD=0",
        "NUM_SHARDS=24",
        "PORT=8000",
        'MODEL="Qwen/Qwen3-32B-AWQ"',
        'CAPS="16000,32000,64000,128000"',
        'OUT_DIR="results/logs/heldout"',
        "summarise() { :; }",
        loop,
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=REPO)


# --------------------------------------------------------------------------- #
# Static: the exact defect, named
# --------------------------------------------------------------------------- #
def test_no_singular_subset_variable_survives():
    """`${SUBSET}` is the stale name that took down array 1894448."""
    text = read_array()
    assert "${SUBSET}" not in text
    assert "$SUBSET " not in text and '$SUBSET"' not in text


def test_every_referenced_variable_is_defined_or_external():
    """Catch an unbound variable statically, the way `bash -n` does not."""
    import re

    text = read_array()
    assigned = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", text, re.MULTILINE))
    assigned |= set(re.findall(r"^\s*(?:export|local)\s+([A-Za-z_][A-Za-z0-9_]*)=",
                               text, re.MULTILINE))
    assigned |= set(re.findall(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in", text, re.MULTILINE))
    referenced = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:#%\-+}]", text))
    undefined = sorted(referenced - assigned - EXTERNAL)
    assert not undefined, f"referenced but never assigned: {undefined}"


def test_the_loop_iterates_subsets_and_conditions():
    loop = extract_run_loop()
    assert "for sub in ${SUBSETS}" in loop
    assert "for cond in ${CONDITIONS}" in loop
    assert '--subset "${sub}"' in loop
    assert '--condition "${cond}"' in loop


def test_four_manifests_are_declared_and_exist():
    subsets = declared_subsets()
    assert len(subsets) == EXPECTED_MANIFESTS, subsets
    assert len(set(subsets)) == EXPECTED_MANIFESTS, "a manifest is listed twice"
    for s in subsets:
        assert (REPO / s).is_file(), f"declared manifest missing: {s}"
    assert sum("bands_held_out_n8" in s for s in subsets) == 1
    for n in (4, 5, 6):
        assert sum(f"heldout_n{n}__" in s for s in subsets) == 1


# --------------------------------------------------------------------------- #
# Dynamic: run the loop and see what it schedules
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_running_the_loop_schedules_every_manifest_and_condition(tmp_path):
    """The test that would have failed on the real defect.

    With `--subset "${SUBSET}"` and `set -u`, bash aborts on the first iteration.
    """
    record = tmp_path / "invocations.txt"
    record.touch()
    proc = run_loop(extract_run_loop(), record)
    assert proc.returncode == 0, f"loop failed:\n{proc.stderr}"

    invocations = [l for l in record.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(invocations) == EXPECTED_CELLS, (
        f"{len(invocations)} runner invocations, expected {EXPECTED_CELLS} "
        f"({EXPECTED_MANIFESTS} manifests x {len(CONDITIONS)} conditions)")

    pairs = set()
    for line in invocations:
        parts = line.split()
        assert parts[0] == "scripts/run_pilot_sweep.py", parts[0]
        assert parts[parts.index("--num-shards") + 1] == "24"
        assert parts[parts.index("--shard-index") + 1] == "0"
        assert parts[parts.index("--caps") + 1] == "16000,32000,64000,128000"
        pairs.add((parts[parts.index("--subset") + 1],
                   parts[parts.index("--condition") + 1]))

    assert pairs == {(s, c) for s in declared_subsets() for c in CONDITIONS}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_undefined_subset_variable_would_fail_the_loop():
    """Guard the guard.

    Re-introduces the exact defect in a copy of the loop and asserts it aborts, so a
    future change that made the check above vacuous is itself caught.
    """
    broken = extract_run_loop().replace('--subset "${sub}"', '--subset "${SUBSET}"')
    proc = run_loop(broken, None)
    assert proc.returncode != 0
    assert "SUBSET" in proc.stderr and "unbound" in proc.stderr.lower()


# --------------------------------------------------------------------------- #
# The dry-run job must certify what the array actually runs
# --------------------------------------------------------------------------- #
def test_dryrun_and_array_agree_on_the_manifest_list():
    dry = DRYRUN.read_text(encoding="utf-8")
    assert subsets_of(dry) == declared_subsets(), (
        "the pre-flight would certify a different set of manifests from the one the "
        "array runs")
    assert "${SUBSET}" not in dry
