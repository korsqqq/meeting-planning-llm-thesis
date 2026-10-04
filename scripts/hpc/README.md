# Running this project on the TU Berlin HPC cluster

Operational runbook for `gateway.hpc.tu-berlin.de`. Written for account
`korsun_a`, Slurm account `qu`.

The rule that shapes everything below: **the run machine used to be a box you
owned, and the cluster is not.** You do not get a long-lived vLLM server to
point `$VLLM_URL` at. You get a node for N hours, and everything — serving and
harness — has to happen inside that window.

---

## 0. Standing constraints

These bind every script in this directory.

| Constraint | Consequence |
|---|---|
| **Seed range is enforced by content** | `guard_seed_range.py` reads the manifest and checks every seed against the locked partition, in **both** directions. `--expect development` keeps held-out data out of development jobs; `--expect held_out` keeps development data out of the held-out array — the more dangerous mistake, because it would produce 2,760 healthy-looking runs of the wrong thing. A manifest renamed `test.json` fails either way. |
| **Single GPU class inside the primary held-out run** | See §0.1. Never mix GPU classes within one held-out experiment. |
| **Every GPU job records its provenance** | git commit, model revision, exact vLLM launch command, `nvidia-smi`, driver, compute capability, `CUDA_VISIBLE_DEVICES`, `SLURM_JOB_GPUS`, torch, torch CUDA, vLLM, lock-file SHA-256, and the quantisation backend actually selected. Written as JSON by `provenance.sh`. |
| **`.git` travels with the code** | The commit hash is part of the experimental record and `provenance.sh` reads it on the cluster. Prefer transferring by git (§4). |
| **Tests run in Slurm, not on the frontend** | The frontend is a shared login node. `tests.slurm` runs the suite on a CPU node. |

### 0.1 Hardware policy for the primary held-out run

Registered **before** the first held-out run, so that a queue problem cannot
turn into an improvised hardware decision afterwards:

> **Preferred primary hardware: H200.**
> If H200 proves both operational (smoke passes) and practically schedulable
> (queue waits observed during smoke/queue testing are workable), then the
> entire primary held-out experiment runs on H200.
>
> Otherwise, **before any held-out run is executed**, one A100 configuration is
> chosen as the fallback, and the entire primary held-out experiment runs
> there instead.
>
> **GPU classes are never mixed inside the primary held-out experiment.**
> Once the first real held-out run has executed, the hardware is frozen and
> does not change for the remainder of the experiment.

"Operational" and "schedulable" are separate tests. H200 may serve AWQ
perfectly and still be the wrong choice at a 20-hour queue wait; that is what
`queue_wait_seconds` is measured for. Deciding this in advance means the answer
does not depend on how impatient we are on the day.

---

## 1. The mental model, and how it differs from the A6000 box

| | A6000 box | HPC cluster |
|---|---|---|
| vLLM | `tmux` session, lives for weeks | started **inside each job**, dies with it |
| harness reaches it via | `$VLLM_URL` over the network | `http://127.0.0.1:$PORT` inside the job |
| you start work by | `ssh` and just running things | writing a job script and `sbatch`-ing it |
| results | already on the machine | on BeeGFS; you pull them with `rsync` |
| parallelism | 2 GPUs you own | job array over many GPUs you queue for |
| failure mode to fear | OOM | **walltime expiry** — killed, no warning |

Almost no Python changes. `run_pilot_sweep.py` already takes `--base-url`,
`--shard-index` and `--num-shards`; `llm_client.py` already reads `VLLM_URL`.
What is new is the wrapper around them.

---

## 2. Hardware: which GPUs are usable

`--gres=gpu:1` is a **trap** on this cluster — it will usually hand you a
Tesla P100, which cannot run this project at all. Always name the type.

| GPU | VRAM | Compute cap | Usable? |
|---|---|---|---|
| H200 | 141 GB | 9.0 | ✅ target for the primary held-out run |
| A100 80 GB | 80 GB | 8.0 | ✅ |
| A100 40 GB | 40 GB | 8.0 | ✅ |
| RTX PRO 6000 (Blackwell) | 96 GB | 12.0 | ⚠️ untested with the pinned vLLM |
| V100s | 32 GB | 7.0 | ❌ awq_marlin needs 8.0+ |
| P100 | 16 GB | 6.0 | ❌ too small **and** too old |

Qwen3-32B-AWQ weighs ~20 GB at INT4, so 16 GB is a hard stop regardless of
kernels. On the P100 nodes the GRES is confusingly named `tesla`, not `p100`.

`smoke.slurm` refuses to proceed on compute capability below 8.0, before the
20 GB model load rather than after it.

### Partitions open to account `qu`

| Partition | GPU | Count | Walltime |
|---|---|---|---|
| `h200_short` | H200 | 12 | 4 days |
| `scioi_a100nv` | A100 80 GB NVLink | 4 | 7 days |
| `gpu` | A100 40 GB | 4 | 7 days |
| `gpu_short` | A100 40 GB | 3 | 1 day |
| `scioi_gpu` | A100 (80 GB + 40 GB) | 2 | 7 days |
| `hri_gpu` | Blackwell | 4 | 2 days |

Everything prefixed `ex_` is reserved for the group that owns the hardware.

---

## 3. Two configuration traps

**The default partition is `TestAndBuild`** (5 h limit, no usable GPU). Forget
`-p` and your job goes somewhere useless.

**`gpu` has `DefaultTime=00:01:00`.** Forget `--time` and your job is killed
after one minute. Always set it explicitly.

Also: `TRESBillingWeights=CPU=0.1,Mem=0.0128G`. GPUs are not billed against
fair-share here, CPU and memory are — so do not request 200 GB of RAM "to be
safe", it costs queue priority for nothing.

---

## 4. First-time setup

**Preferred: transfer by git.** Then `git rev-parse HEAD` on the cluster has an
unambiguous meaning, and the provenance commit hash is verifiable against a
remote rather than being whatever happened to be in a working copy.

From your laptop:

```bash
git push -u origin main
```

On the cluster (first time):

```bash
git clone <repo-url> ~/thesis-mas-planning && cd ~/thesis-mas-planning && git switch main
```

On the cluster (afterwards):

```bash
cd ~/thesis-mas-planning && git fetch && git switch main && git pull
```

**Fallback: `rsync`,** if GitHub authentication is not yet set up on the
cluster. Note that `rsync` does not exist in `cmd.exe` — this needs Git Bash or
WSL. Do not exclude `.git`:

```bash
rsync -avz --exclude '.venv*' --exclude 'results/logs' ./ korsun_a@gateway.hpc.tu-berlin.de:~/thesis-mas-planning/
```

Then on the frontend, once:

```bash
cd ~/thesis-mas-planning && mkdir -p logs/hpc && bash scripts/hpc/bootstrap.sh
```

`bootstrap.sh` installs `uv`, creates both virtual environments from the
**lock** files, and downloads the pinned checkpoint into
`/scratch/$USER/hf-cache`. It is idempotent. It deliberately does **not** run
the test suite — that is the next step.

---

## 5. The order of jobs

### 5.1 Offline tests, on a CPU node

```bash
sbatch scripts/hpc/tests.slurm
```

The suite is CPU-only by construction, so it belongs neither on the frontend
nor in a GPU allocation. It must be green before any GPU time is spent.

### 5.2 Smoke, on a GPU node

```bash
sbatch scripts/hpc/smoke.slurm
```

Watch it:

```bash
squeue --me
```

```bash
tail -f logs/hpc/smoke-<jobid>.out
```

It stops at the first failure, in this order:

1. **held-out guard** — every instance in the manifest has seed < 100000
2. compute capability ≥ 8.0, checked before the model is loaded
3. the pinned vLLM starts on this GPU architecture
4. `awq_marlin` appears in the startup log — evidence the served path is the
   intended one and not a slower fallback
5. the harness reaches vLLM on localhost
6. a small number of real runs complete

Each smoke job writes to its **own** directory, `results/logs/hpc_smoke/<jobid>/`.
The runner's strict resume is correct for the real sweep and a trap here: a
second smoke sharing one directory would find the first smoke's results already
on disk, skip the runs, and report a latency it never measured.

### 5.3 What smoke reports

Four measurements, kept **separate** because they are four different planning
quantities:

| Metric | Why it matters |
|---|---|
| `queue_wait_seconds` | how many array tasks are worth submitting at this fair-share |
| `job_elapsed_seconds` (incl. `model_load_seconds`) | what `--time` to request, and the per-task startup overhead a shard must amortise |
| `per_run_latency_seconds` | converts 2,760 runs into GPU-hours — the number the Block A / Block B decision waits on |
| `vllm_decode_tokens_per_s` | hardware comparison against the A6000 box |

Written to `logs/hpc/smoke-<jobid>-summary.json`, together with the full
provenance record.

Only `per_run_latency_seconds` is a scientific quantity — it is the time the
system spent on the task. `queue_wait_seconds` is a property of the cluster's
scheduler, `model_load_seconds` of the serving stack, and
`vllm_decode_tokens_per_s` of the GPU. None of the three may be folded into a
reported C1/C3 latency.

### 5.4 Go / no-go gate before writing the array job

| Check | Requirement |
|---|---|
| GPU | H200 (or the registered A100 fallback — §0.1) |
| compute capability | 9.0 (A100 fallback: 8.0) |
| vLLM | `0.19.1` |
| torch / torch CUDA | the frozen versions |
| model revision | the pinned revision |
| model loads | yes |
| quantisation backend | AWQ Marlin, or an equivalent explained in advance |
| manifest | pilot only, guard passed |
| instances with seed ≥ 100000 | **0** |
| run JSON written | yes |
| validator / scorer ran | yes |
| token accounting present | yes |
| job exit code | 0 |
| per-run latency | recorded |
| model-load time | recorded |
| queue wait | recorded |

Every row green before the array job is written.

---

## 6. Provenance: what every GPU job records

`provenance.sh` writes `logs/hpc/provenance-<jobid>.json`:

- **git** — commit, branch, dirty flag
- **model** — repo and pinned revision
- **slurm** — job id, partition, node, submit/start times, queue wait
- **gpu** — name, memory, driver version, compute capability, full `nvidia-smi`
- **software** — torch, torch CUDA, vLLM versions
- **quantisation** — the matching lines from the vLLM startup log, **quoted
  verbatim** rather than asserted, so a fallback kernel shows up as itself

The run documents the harness writes already carry the git commit and subset
hash, but they say nothing about the machine. On a cluster the same code can
land on four different GPU architectures, and "which kernel actually ran" is
not recoverable after the fact.

---

## 7. The held-out array

Scheduling was **recomputed on 2026-08-27** for the 3,960-run design by
`scripts/hpc/size_heldout_campaign.py`. The 2026-08-26 freeze (24 / %6 / 20 h) was
priced for 2,760 runs and is superseded.

| | Value | Where it came from |
|---|---|---|
| logical shards | **24** | 792 items / 24 = 33 each, exactly |
| concurrency | **%6** | ~106 h wall-clock excluding queue |
| walltime | **36:00:00** | conservative estimate 26.5 h, margin 1.36x |
| execution | sequential | batch-composition regime unchanged from the pilot |
| runners per shard | **20**, one per (subset, condition) | four manifests x five conditions |
| vLLM per shard | 1 | startup ~25 min, paid once |

Inputs are the development runs' own recorded `latency_seconds`, converted with the
measured H200 speed-up of 2.15x (job 1891293), plus model load ~1,600 s and
per-invocation startup ~350 s. Compute-only cost is **447 GPU-hours** on the mean
basis and **578** on the conservative one; the walltime is chosen from the
conservative figure, because the cost of underestimating it is a mass timeout
across the whole array.

Concurrency, not shard count, is what sets the campaign length: at `%6` the total
is ~106 h whatever the split, because the work is fixed. More shards only buy
finer resume granularity and a shorter walltime, at the price of paying the
per-shard startup more often.

**Four manifests, not one.** Block A (`n = 8`) and the three Block B sizes are
separate subsets, each with its own binning document beside it. The runner takes
one subset and one condition per invocation, so the array loops over all twenty
pairs. Merging them would need band boundaries spanning four values of `n`; the
saving does not justify it. Each subset is sharded by the same `index % num_shards`,
so the union stays complete and disjoint -- proven over all four at once by
`scripts/hpc/audit_heldout_design.py`.

**Pool size.** Block A needs 150 instances and the frozen selector allows one
instance per seed across the whole manifest, so it needs 150 distinct seeds. The
`n = 8` pool is therefore **400 seeds** (`100000-100399`), fixed before selection:
2.7x the requirement, and about 4x margin on the scarcest cell. The earlier
100-seed pools were sized for 20 per band and cannot supply this design.

### 7.1 Pre-flight, then the array

```bash
SUBSET=results/manifests/subset__heldout__<hash>.json sbatch scripts/hpc/heldout_dryrun.slurm
```

CPU-only, contacts no endpoint, writes no run document. Four checks: seed-range
guard, shard plan, expected-runs manifest, end-to-end dry run on shard 0.

The shard-plan check is the one a per-shard dry run cannot do. Each invocation
sees only its own slice, so a dropped or duplicated cell would look healthy in
all 24 logs and surface only at the completeness audit, after the campaign.

```bash
SUBSET=results/manifests/subset__heldout__<hash>.json sbatch scripts/hpc/heldout_array.slurm
```

### 7.2 Gates that must be closed before submitting

| # | Gate | Evidence |
|---|---|---|
| 1 | tokenizer revision comparison | job 1891411 — five shapes identical |
| 2 | pinned tokenizer in sweep, recorded per run | `pilot_sweep/1.1` metadata |
| 3 | harness `HF_HUB_OFFLINE=1` | set in `heldout_array.slurm` |
| 4 | strict tokenizer test, 0 skipped | job 1892479 — `10 passed` |
| 5 | full pytest | job 1892479 — `823 passed, 1 skipped` |
| 6 | dry-run / expected-run audit / seed guard | `heldout_dryrun.slurm`, all four manifests |
| 7 | design audit -- 198 instances, O histogram, 3960 cells, shard split | `audit_heldout_design.py`, passed 2026-08-27 |
| 8 | selector determinism | both selectors re-run from the committed tree; all eight manifests identical except `timestamp_utc`, and every `content_hash` matches |

**One fragility worth knowing before rebuilding a manifest.** `histogram_ladder`
materialises every composition of `per_band` over the distinct optima present in
the pool. The 400-seed `n = 8` pool carries seven distinct optima, so that is
about 32.5 million dictionaries -- roughly 25 minutes and enough memory that a
rebuild died with `MemoryError` when other work was competing for RAM. It is a
resource limit, not a defect in the selection: the run that completes is
deterministic. The manifests are committed and the cluster never rebuilds them,
so this touches nobody who is only running the campaign.

### 7.3 After the campaign

```bash
python scripts/audit_expected_runs.py     --expected results/manifests/expected_runs__bands_held_out_n8.json     --expected results/manifests/expected_runs__heldout_n4.json     --expected results/manifests/expected_runs__heldout_n5.json     --expected results/manifests/expected_runs__heldout_n6.json     --runs results/logs/heldout     --out results/analysis/heldout/completeness_audit.json

**Pass all four manifests in one call.** `--expected` is repeatable, and the four
shards share one output directory, so auditing one manifest at a time reports the
other 3,640 run documents as "unexpected files" and fails a campaign that is in
fact complete.
```

Analysis may begin only once every expected id has a verified result. All shards
share one output directory; the runner verifies each result against the manifest
and resumes only what is genuinely incomplete, so a shard killed at its walltime
can simply be resubmitted.

---

## 8. Getting results back

Nothing is sent to you. Results stay on BeeGFS; pull them:

```bash
rsync -avz korsun_a@gateway.hpc.tu-berlin.de:~/thesis-mas-planning/results/logs/ ./results/logs/
```

Add `--mail-type=END,FAIL --mail-user=...` to a job script for an email when it
finishes — that is a notification, not the data.

---

## 9. To record in THESIS_DECISIONS.md before the held-out run

Both are dated amendments, written **before** the first official LLM call:

1. **Hardware.** The pilots ran on A6000 (SM 8.6). The primary held-out run
   uses a single GPU class, chosen by the policy in §0.1 — H200 preferred, one
   named A100 configuration as the registered fallback, decided **before** the
   first held-out run and frozen after it. Within the held-out run everything
   is homogeneous, so the C1–C5 comparisons hold; the pilot-to-held-out
   comparison crosses GPU architectures and that has to be on record.
2. **Serving version.** If `vllm==0.19.1` installs and serves correctly on the
   cluster, the runtime is unchanged from the pilots and only the verification
   needs recording. If it does not, the replacement version is a dated
   amendment, because chat templating and sampling behaviour may differ.

---

## 10. Slurm commands worth memorising

| Command | Use |
|---|---|
| `squeue --me` | what am I running / waiting for |
| `squeue --me --start` | estimated start time of pending jobs |
| `scancel <jobid>` | cancel; `scancel -n <name>` by name |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode` | **why did it die** |
| `sinfo -p h200_short -o "%P %D %F %G"` | what is free right now |
| `sshare -U` | your fair-share standing |
| `scontrol show job <jobid>` | full detail on one running job |

`sacct` is the one to reach for first when a job disappears: it distinguishes
`TIMEOUT` (walltime too short) from `OUT_OF_MEMORY` from a real `FAILED`.
