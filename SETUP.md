# Environment Setup (run machine)

Operational procedure for the machine that produces thesis numbers. Every value
that must be frozen for reproducibility has a slot in
[§7 Frozen record](#7-frozen-record); fill it in as the steps are completed and
commit the result.

Rule that overrides convenience: **no number from the local dev box (RTX 2080
Super) enters the thesis.** The dev box runs smoke tests and offline tests only.

---

## 1. Target machine

| Resource | Value |
|---|---|
| GPU | 2 × NVIDIA RTX A6000, 49 140 MiB each |
| CPU | AMD Threadripper PRO 7955WX, 16 cores / 32 threads |
| RAM | 502 GiB |
| Disk | ~2.8 TB total, ~377 GB free |
| OS | Ubuntu 22.04.4 LTS |
| Driver / CUDA | 550.144.03 / CUDA 12.4 |
| System Python | 3.10.12 — **not usable for the harness** (see below) |

Two consequences drive the whole setup:

1. **The system Python is too old.** `src/schemas/` uses `enum.StrEnum`, which
   requires Python ≥ 3.11. The harness runs on a project-local Python 3.12.
2. **A6000 is Ampere (SM 8.6), which has no native FP8 tensor cores.** The
   design therefore serves the official **AWQ INT4** checkpoints instead of the
   FP8 ones (decision recorded in `THESIS_DECISIONS.md` §1). AWQ is natively
   supported on Ampere through the `awq_marlin` kernel, and quantisation parity
   between the 8B and the 32B model is preserved because both come from the same
   official AWQ family.

Model placement: one model per GPU, `tensor_parallel_size=1`.

| GPU | Model | Port | Role |
|---|---|---|---|
| 0 | `Qwen/Qwen3-32B-AWQ` | 8000 | main run |
| 1 | `Qwen/Qwen3-32B-AWQ` | 8002 | second shard of the same sweep (§6) |

`Qwen/Qwen3-8B-AWQ` was the planned pilot and robustness model on GPU 1. It was
**excluded** on 2026-08-04 (`THESIS_DECISIONS.md` §1): C1 never leaves satisfaction 0
across 4k–32k because the model never issues `Action: propose`. Its checkpoint and
pre-flight archive are kept for the record; no run time is spent on it.

Weights are roughly 20 GB (32B) and 6 GB (8B) at INT4, so each server has ample
KV-cache headroom on a 48 GB card and the two runs never contend for memory.

---

## 2. Base tooling and the two environments

The serving stack and the harness are installed in **separate virtual
environments**. vLLM pins its own torch/transformers versions; the harness must
not inherit those pins, and a broken vLLM upgrade must not be able to change the
tokenizer used for budget accounting.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
```

Harness environment (CPU-only: generator, oracle, budget, agents, scorer, tests):

```bash
cd ~/thesis-mas-planning
uv venv --python 3.12 .venv-harness
source .venv-harness/bin/activate
uv pip install -r requirements/harness.txt
uv pip freeze > requirements/harness.lock.txt
python -c "import sys, pydantic, ortools; print(sys.version)"
deactivate
```

Serving environment (GPU):

```bash
uv venv --python 3.12 .venv-vllm
source .venv-vllm/bin/activate
uv pip install -r requirements/serving.txt
python -c "import vllm; print(vllm.__version__)"
uv pip freeze > requirements/serving.lock.txt
```

Then **pin the exact vLLM release**: write `vllm==<printed version>` into
`requirements/serving.txt` and record it in §7. A floor such as `>=0.8.5` is not
acceptable — chat templating, sampling behaviour and the response format must not
drift between the pilot and the main run (`THESIS_DECISIONS.md` §1).

The default vLLM wheels are built against CUDA 12.x and run on driver 550
through CUDA minor-version compatibility. If the import fails with a CUDA
version error, install the build matching CUDA 12.4 rather than upgrading the
driver.

---

## 3. Checkpoints and pinned revisions

Keep the Hugging Face cache on the large disk and record the exact commit of
each repository. Chat templates can change on the Hub independently of the
weights, so the revision — not just the repository name — is the reproducible
identifier.

```bash
export HF_HOME=/path/to/large/disk/hf            # add to ~/.bashrc
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download Qwen/Qwen3-32B-AWQ      # newer hub versions: `hf download`
huggingface-cli download Qwen/Qwen3-8B-AWQ
```

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo in ("Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-8B-AWQ"):
    print(repo, api.model_info(repo).sha)
PY
```

Record both hashes in §7. From the pilot onward every server start and every
tokenizer load passes them explicitly (`VLLM_REVISION`,
`--tokenizer-revision`).

---

## 4. Starting the servers

`scripts/serve_vllm.sh <model> [gpu] [port]` starts one server in the
foreground; run each inside its own tmux session so it survives a disconnect.

Two things must be inside the tmux command string, not before it: the venv activation
and `HF_HOME`. A tmux server that is already running does not inherit them from the
client's shell, and the failure looks like `vllm: command not found` or a surprise
re-download.

```bash
mkdir -p logs

tmux new -d -s vllm32 'cd ~/thesis-mas-planning && source .venv-vllm/bin/activate \
  && export HF_HOME=$HOME/hf-cache \
  && VLLM_REVISION=0499c3ac83fdef8810b907a23894ba91e95eddd8 \
     VLLM_EXTRA_ARGS=--no-enable-prefix-caching \
     ./scripts/serve_vllm.sh Qwen/Qwen3-32B-AWQ 0 8000 2>&1 | tee logs/vllm_32b.log'

# Second shard of the same model on the other GPU (§6): identical flags, different
# GPU and port. Verified to produce identical output on the same cell.
tmux new -d -s vllm32b 'cd ~/thesis-mas-planning && source .venv-vllm/bin/activate \
  && export HF_HOME=$HOME/hf-cache \
  && VLLM_REVISION=0499c3ac83fdef8810b907a23894ba91e95eddd8 \
     VLLM_EXTRA_ARGS=--no-enable-prefix-caching \
     ./scripts/serve_vllm.sh Qwen/Qwen3-32B-AWQ 1 8002 2>&1 | tee logs/vllm_32b_gpu1.log'
```

`VLLM_EXTRA_ARGS=--no-enable-prefix-caching` is not optional — see §6. Confirm it took
effect with `grep -o "enable_prefix_caching=[A-Za-z]*" logs/vllm_32b.log | head -1`,
which must print `enable_prefix_caching=False`.

`Qwen/Qwen3-8B-AWQ` is **not** started: it was excluded on the pre-pilot feasibility
check (`THESIS_DECISIONS.md` §1).

Defaults set by the script, all overridable by environment variable:
`--max-model-len 32768` (native context, covers the 16k cap plus scaffold),
`--max-num-seqs 16`, `--gpu-memory-utilization 0.90`, `--seed 42`,
`--tensor-parallel-size 1`.

**Confirm the quantisation path actually taken** — this is the evidence that the
Ampere/AWQ assumption held:

```bash
grep -iE "awq|marlin|quantization" logs/vllm_32b.log | head
nvidia-smi --query-gpu=index,memory.used --format=csv
```

Expected: the startup log names an AWQ (Marlin) kernel, and each GPU shows the
weights plus the pre-allocated KV cache. If vLLM refuses the checkpoint, do not
work around it silently — that is a design-level event and goes to
`THESIS_DECISIONS.md` §1 with the fallback that was chosen.

Client-side environment for the harness:

```bash
export VLLM_URL=http://localhost:8000/v1        # GPU 0; the second shard is :8002
export VLLM_MODEL=Qwen/Qwen3-32B-AWQ
```

---

## 5. Pre-flight gates

Run in this order, from the harness environment, and archive the output of each.
Later steps are meaningless if an earlier one failed.

**5.1 Capability probe** — does the endpoint support what the harness needs
(extra sampling params, the `</think>` split on `/v1/completions`, generated
token ids, constrained JSON, and local-vs-endpoint input-token parity)?

```bash
python -m scripts.check_endpoint --base-url $VLLM_URL --model $VLLM_MODEL
```

Every required probe must pass. Probe G (input token parity, delta 0) is the
single-call form of the ledger-versus-usage identity the budget accounting rests
on; probe F reports whether this vLLM version wants `guided_json` or
`structured_outputs`, which decides whether `src/core/llm_client.py` needs an
adjustment before the pilot.

**5.2 Tokenizer parity** — a cross-model check, and therefore **not a mandatory
gate at present**. It exists so that an equal-budget comparison across two model
sizes counts tokens the same way on both. The 8B arm was dropped on a pre-pilot
feasibility check and no second model has been chosen, so there is currently no
second tokenizer for this to constrain. Run it as a gate again once a second
model is selected.

```bash
REQUIRE_TOKENIZER_PARITY=1 pytest tests/test_tokenizer.py -v
```

When it is a gate, the archived output must read `... passed, 0 skipped`; a skip
means the parity evidence never ran.

**5.3 Full offline suite** — nothing about the server may have changed harness
behaviour.

```bash
pytest -q
```

**5.4 Live smoke** — a **development-only infrastructure check** against whatever
model is currently the target. It answers questions no offline test can: does the
endpoint return the expected thinking/content split, does the ledger match
endpoint-reported usage call by call, is the absence of usage recorded honestly.

**No score this produces is quality evidence**, at any cap. It is not a pilot,
not a calibration, and nothing it prints belongs in the thesis.

```bash
python -m scripts.run_live_smoke --base-url $VLLM_URL --model $VLLM_MODEL \
    --condition c1_react --n-runs 2 --require-endpoint-usage
```

Choose `--cap` to suit the check; there is no fixed protocol value, and a cap low
enough to floor the agent still exercises the accounting path. Repeat for
`c2_verify_revise` and `c3_mas`, then **read two or three transcripts by hand** —
this checkpoint has caught real bugs twice, and no automated gate replaces it.

This is a smoke test. It produces no thesis number, no calibration value and no
statement about model quality.

---

## 6. Operational notes

- **Long runs:** always inside tmux; the sweep runner is resumable, but a lost
  server means a lost hour of generation.
- **CPU work in parallel with GPU work:** instance generation and CP-SAT solving
  are CPU-only. Parallelise across instances (16 physical cores); each individual
  solve stays single-worker so the optimum and the returned plan remain
  deterministic.
- **Batching and determinism:** `--max-num-seqs` trades throughput against
  batch-composition variance. Residual non-determinism at fixed seed is a
  disclosed confound (`THESIS_DECISIONS.md` §7), not something this setup claims
  to eliminate.
- **Prefix caching: DECIDED 2026-08-04 — OFF, and it must stay off.** Start every
  server with `VLLM_EXTRA_ARGS=--no-enable-prefix-caching`. vLLM 0.19.1 enables it by
  default, and with it on the same cell repeated five times returned satisfaction
  0.00/0.75/1.00/1.00/1.00; off, the five repeats were byte-identical. It does not
  affect token accounting, but it was the whole of the run-to-run variance, so it
  decides whether `n=1` per cell means anything. The price is throughput: one 32k-cap
  C1 run takes 7 min 47 s, and the pilot's time budget must be built from that number.
  Never change the setting between the pilot and the main run — results from the two
  regimes are not comparable.
- **Sharding across the two GPUs.** Two identical servers (same version, revision and
  flags) produce identical output on the same cell — verified 2026-08-04 on GPU 0
  (port 8000) and GPU 1 (port 8002). Splitting the sweep across both halves the wall
  time without adding a confound. Check the second GPU is actually free first: a
  leftover server there will make the new one fail engine initialisation on memory,
  and the traceback names the engine, not the cause.

---

## 7. Frozen record

Fill in on the run machine and commit. Everything here is required by the
reproducibility appendix.

| Item | Value |
|---|---|
| Date frozen | 2026-08-03 |
| Host / GPU | 2 × RTX A6000 48 GB, driver 550.144.03, CUDA 12.4 |
| OS | Ubuntu 22.04.4 LTS |
| Python (harness) | 3.12.13 (uv-managed, `.venv-harness`) |
| Python (serving) | 3.12.13 (uv-managed, `.venv-vllm`) |
| vLLM version | `0.19.1` — **not** the latest release; see the note below |
| torch version | 2.10.0+cu128 |
| transformers version | 5.14.1 (identical in both environments) |
| ortools version | 9.15.6755 |
| `Qwen/Qwen3-32B-AWQ` revision | `0499c3ac83fdef8810b907a23894ba91e95eddd8` |
| `Qwen/Qwen3-8B-AWQ` revision | `4da05a8edb55c6046cce958586c33b61da07bb79` |
| Serving flags | `--tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.90 --seed 42 --no-enable-prefix-caching`; chunked prefill left at the vLLM default (ON) |
| Prefix caching | **OFF** (decided 2026-08-04, `THESIS_DECISIONS.md` §1). ON it was the sole source of run-to-run variance: one cell repeated five times gave satisfaction 0.00/0.75/1.00/1.00/1.00; OFF the same five repeats were identical (22 220 tokens each). Cost: one 32k-cap C1 run takes 7 min 47 s |
| Cross-server determinism | verified 2026-08-04 — the same cell on a second identical server on GPU 1 (port 8002) reproduced 22 220 tokens and satisfaction 0.00 exactly, so the sweep may be sharded across both A6000s |
| Quantisation kernel observed | `awq_marlin` — `quantization=awq_marlin`, `Using MarlinLinearKernel for AWQMarlinLinearMethod` |
| KV cache observed | 32B **with** prefix caching: 94 320 tokens (23.03 GiB); 8B with prefix caching: 260 544 tokens (35.78 GiB). **32B as actually served for the formal pilot, prefix caching off: 98 928 tokens on both GPUs.** Source: `results/evidence/serving/vllm_config_extract.txt` |
| Constrained-decoding parameter | `structured_outputs` (`{"json": <schema>}`); 0.19.1 rejects `guided_json` |
| Pre-flight passed | 32B: 2026-08-02, `logs/preflight/preflight_32b_20260802.txt`; 8B: 2026-08-03, `logs/preflight/preflight_8b_20260803.txt` |
| Tokenizer parity strict run | 2026-08-02, `10 passed, 0 skipped` |
| Offline suite | 2026-08-04, `223 passed` on Python 3.12.13 |
| Live smoke passed | 2026-08-04, `results/logs/live_smoke/*.json`; C1 / 32B / cap 16000 → satisfaction 1.00 at 14 595 tokens, ledger-vs-usage delta 0 on every call |
| Context-window guard verified live | 2026-08-05 — see the note below |

**Context-window guard, live verification (2026-08-05). NOT PILOT / NOT QUALITY
EVIDENCE.** The guard clamps every request to `min(budget grant, max_model_len -
input_tokens)`; the served window is 32768, but a run at cap 64000 would otherwise ask
for ~63.5k generation tokens and be rejected outright. To exercise it against the real
endpoint rather than only in unit tests, one C1 run was made on **development seed 0**
(never part of the pilot or the main experiment) at cap 8000 with the window
**deliberately lowered to 4096**, model `Qwen/Qwen3-32B-AWQ`.

Result: the first four calls were context-limited, with the requested grant reduced
exactly to `4096 - input_tokens` (7519 → 3871, 6079 → 3840, 5269 → 3781, 4147 → 3722);
from the fifth call the budget grant was already smaller than the free window, so no
clamping applied — the `min` of the two limits, not a fixed ceiling. The endpoint
returned **no HTTP 400**, internal and endpoint token totals agreed exactly (7508 /
7508, delta 0 — the ledger books realised tokens, so clamping cannot desynchronise it),
every `finish_reason` was `stop`, and the transcript scan was clean with no think leak.

The run's `valid=True, satisfaction=0.00` must NOT be read as a solved instance: an
empty final plan is valid by construction and contains zero meetings. This is
infrastructure verification only and is not an experimental result. The smoke JSON is
not committed.

Not covered live: the `ContextWindowError` path, where the prompt alone fills the
window. The input never exceeded 553 tokens here, so that branch remains covered by
unit tests only.

**vLLM version — why it is not the latest.** Releases from 0.20.0 onward depend on
torch 2.11, built against CUDA 13, which cannot initialise on driver 550.144.03
(CUDA 12.4): the engine fails in `torch._C._cuda_init()` before reading any weights.
0.19.1 is the newest release still built against CUDA 12. This is an environment
constraint, not a quality-motivated choice (`THESIS_DECISIONS.md` §1). Its one
harness-visible consequence is the constrained-decoding parameter name in the row
above.

**Pre-flight note.** Probe D (`</think>` delimiter) fails at the script's default
`--max-tokens 256`: a Qwen3 think block does not fit. Use `--max-tokens 2048` for the
32B and `8192` for the 8B — on the same probe prompt the 32B spent 413 reasoning
tokens and the 8B 2 315, which is also the first measurement showing that a budget
ladder calibrated on one model does not transfer to the other.
