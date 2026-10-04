# Multi-Agent vs. Single-Agent LLMs Under Equal Token Budgets

Bachelor thesis, TU Berlin, Quality and Usability Lab (Faculty IV), by Andrii Korsun.
First examiner: Prof. Sebastian Möller. Second examiner: Prof. Dr. Axel Küpper.
Supervisor: Dr.-Ing. Stefan Hillmann.

This repository contains the code, the prompts, the frozen design record and the derived
results of the thesis. The version referenced in the thesis is the tag `thesis-final`.
It does not contain the thesis text.

## Research question

Under an **equal token budget**, is there a complexity threshold at which a hierarchical
multi-agent system (MAS) starts to outperform a single ReAct agent on meeting planning, and
is the gain worth the coordination overhead? All outcomes are treated as valid results: a
crossover exists, the single agent always wins, or the hierarchy always wins.

## Design

**Task.** Meeting planning is generated as an orienteering problem with time windows: a
deterministic generator controls time windows, travel structure (uniform, clustered, line,
random) and conflict density. In the main block, complexity is the number of *pairwise
conflicts* of an instance (two people who can each be met alone but not both, in either
order), verified after solving, not the number of people.

**Ground truth.** An OR-Tools CP-SAT oracle computes the optimum for every instance. It is
never exposed to the agents. A hidden validator, independent of the solver, checks every
plan; an invalid plan scores 0.

**Primary metric.** Satisfaction rate = achieved reward of the agent plan / oracle optimum.

**Architectures (conditions).**

| ID | Architecture |
|----|--------------|
| C1 | ReAct, single agent, single context (baseline) |
| C2 | ReAct plus a fixed verify/revise loop in the same context (at most two revisions) |
| C3 | Hierarchical MAS: supervisor, two workers, aggregator, critic |
| C4 | Planner with a fresh-context critic (the C3 critic without task decomposition) |
| C5 | Validated best-of-3 sampling of the single agent (matched-budget control) |

All architectures run as fixed LangGraph workflows with the same three read-only tools
(`list_people`, `get_availability`, `get_travel_time`) and the same inference parameters.

**Token budget.** Input, thinking and output tokens all count, as do inter-agent messages in
C3. The budget is enforced by the harness, not by the model: tokens are counted with the
Qwen tokenizer before each call, and a fixed slice is reserved for a finalisation node. A
valid best-so-far plan is always held in state, so every run ends with a valid, possibly
empty, answer. Budget caps in the held-out study: 16k, 32k, 64k and 128k tokens.

**Model and serving.** `Qwen/Qwen3-32B-AWQ` (INT4 AWQ, thinking mode on), self-hosted with
vLLM 0.19.1 behind an OpenAI-compatible endpoint. Sampling is fixed: temperature 0.6,
top_p 0.95, top_k 20, min_p 0, seed 42. No external API is called. Qwen3-8B-AWQ was excluded
after a pre-pilot feasibility check (it never issued a plan proposal) and the cross-model
robustness question remains open; see `THESIS_DECISIONS.md` §1 and §7.

## Held-out study

- 198 instances, 5 architectures, 4 budget caps: 3,960 runs, all present, none missing.
- **Block A:** n = 8 people, pairwise-conflict groups Low / Medium / High, 50 instances each,
  analysed as Architecture × Budget × Complexity.
- **Block B:** n = 4, 5, 6 people, 16 instances each, size reported descriptively.
- Run on the TU Berlin HPC cluster, one NVIDIA H200 per job.

Block A mean satisfaction, pooled over complexity levels (150 instances per cell):

| Architecture | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| C1 | 0.003 | 0.177 | 0.330 | 0.368 |
| C2 | 0.003 | 0.193 | 0.447 | 0.527 |
| C3 | 0.011 | 0.136 | 0.643 | 0.822 |
| C4 | 0.007 | 0.116 | 0.589 | 0.639 |
| C5 | 0.003 | 0.009 | 0.088 | 0.531 |

These are descriptive means. The inferential analysis of the thesis is the factorial
analysis over all four caps (task-level permutation tests, Holm correction within each
family, 40 paired architecture comparisons). Architecture, Budget and Architecture × Budget
are significant; the three-way interaction Architecture × Budget × Complexity is not
established (Holm p = 0.073). A complexity-dependent change in architecture ranking was not
established under the present operationalisation of task complexity. The specific test
procedures were fixed after the runs and before the tests were run.

`results/analysis/heldout/` holds an earlier single-cap analysis that is superseded and not
used in the thesis.

The final plan is valid by construction, so plan validity is not reported as a metric; the
informative secondary outcomes are the valid non-empty rate and the optimality rate. Full
numbers, amendments and limitations are in `THESIS_DECISIONS.md` and in
`results/analysis/heldout_factorial/` and `results/analysis/heldout_full_matrix/`.

The earlier formal pilot (360 runs, C1–C3, 8k–64k caps) is kept as design history in
`results/analysis/formal_pilot/`. It is not a result of the thesis.

## Where to find what the thesis refers to

| Thesis | Repository |
|---|---|
| Prompts of all roles | `src/agents/` (`react_core.py`, `single_agent/`, `multi_agent/prompts.py`) |
| Run table, 3,960 runs | `results/exports/heldout/runs.csv` |
| Stage table | `results/exports/heldout_stages/stage_metrics.csv` |
| Factorial analysis | `scripts/analyse_heldout_factorial.py`, outputs in `results/analysis/heldout_factorial/` |
| Stage diagnostics | `scripts/analyse_stage_metrics.py`, outputs in `results/analysis/heldout_stages/` |
| Design record | `THESIS_DECISIONS.md` |

The run documents record commit `f1f1ff7` of the development repository. The source code
here is the same as in that commit, apart from comments and line endings.

## Repository layout

```
src/         Implementation
  schemas/     Pydantic v2 models
  data/        Instance generator and complexity annotation
  oracle/      CP-SAT solver, brute-force check, validator
  agents/      C1–C5 as LangGraph workflows, shared tools
  core/        Tokenizer, token budget ledger, vLLM client
  harness/     Runner and result writer
  evaluation/  Scorer
scripts/     Manifests, sweep runner, analyses, HPC job scripts (scripts/hpc/)
tests/       Offline test suite (fake client, no GPU)
results/     Manifests, derived analyses, exported run table, calibration records
logs/        Frozen pre-flight records
thesis/      Literature notes and bibliography
requirements/ Harness and serving environments, with lock files
```

## Reproducing the analysis

Two separate environments are used (see `SETUP.md`). The harness environment is CPU only and
runs the generator, oracle, agents with a fake client, and the analyses:

```bash
uv venv --python 3.12 .venv-harness
uv pip install -r requirements/harness.txt
pytest -q
```

The factorial analysis reads `results/exports/heldout/runs.csv` (3,960 rows) and writes to
`results/analysis/heldout_factorial/`. It is run step by step; each step is documented in
the header of `scripts/analyse_heldout_factorial.py`, for example:

```bash
uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial audit
uv run --with numpy --with pandas python -m scripts.analyse_heldout_factorial omnibus --block A
```

Re-running the experiment itself needs a GPU server with the serving environment and the
pinned checkpoint revisions from `SETUP.md`.

**Not included.** The 3,960 raw run documents (`results/logs/`) and their tarball are too
large to version. Their SHA-256 is recorded in `heldout_results_3960.sha256`, and the run
table in `results/exports/heldout/` is derived from them.

## Where the design lives

`THESIS_DECISIONS.md` is the canonical record of methodology, locked decisions, amendments,
results and limitations. `SETUP.md` covers the environment, serving configuration and the
frozen version record.

## Citation

Korsun, A. (2026). *Multi-Agent vs. Single-Agent LLMs Under Equal Token Budgets*.
Bachelor thesis, Technische Universität Berlin.

## License

MIT, see `LICENSE`.
