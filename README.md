# Single-Agent vs Multi-Agent LLM Planning Under Equal Token Budget

Bachelor thesis, TU Berlin (Faculty IV). Supervisor: Prof. Stefan Hillmann.

## Research question

Under an **equal token budget**, is there a complexity threshold at which a
hierarchical multi-agent system starts to outperform a single ReAct agent on
meeting planning, and is the gain worth the coordination overhead?

All outcomes are treated as valid results: a crossover exists, the single agent
always wins, or the hierarchy always wins.

## Experimental framework

Three conditions receive the **same budget cap** on the **same instance**, so
comparisons are paired within instance:

1. **C1** — ReAct, single agent, single context.
2. **C2** — ReAct plus a fixed verify/revise loop in the same context.
3. **C3** — hierarchical: supervisor, two workers, aggregator, critic.

A fourth condition (planner plus fresh-context critic) is optional and not
implemented.

The budget is enforced by the harness rather than by the model. Input, output and
thinking tokens all count; inter-agent messages in C3 count against the same
shared budget. A current best plan is always held in state, so a valid answer
exists when the budget runs out.

## Benchmark and ground truth

The **primary benchmark is generated**: a deterministic meeting-planning
generator with controlled time windows, travel structure and conflict density.
NATURAL PLAN is a **planned external-validation slice**, not the benchmark the
main results rest on, and it is not yet part of the pipeline.

Ground truth comes from an **OR-Tools CP-SAT oracle** that is never exposed to
the agents, and a hidden validator independent of the solver. The primary metric
is the satisfaction rate — achieved reward over the oracle optimum — with no
partial credit for an invalid plan.

## Models

`Qwen/Qwen3-32B-AWQ` (INT4 AWQ), self-hosted with vLLM behind an
OpenAI-compatible endpoint, is the model that completed the formal pilot.

`Qwen/Qwen3-8B-AWQ` was **excluded on a pre-pilot feasibility check** — it never
issues a plan proposal at any budget, so it cannot support a robustness claim.
The cross-model robustness question is therefore **open**: whether a different
second model returns, or the size axis is dropped and reported as a limitation,
is not yet decided.

## Repository layout

```
src/        Implementation (schemas, data, oracle, agents, core, harness, evaluation)
scripts/    Manifest building, sweep runner, analysers, endpoint and smoke tools
tests/      Offline test suite
results/    Manifests, run documents, derived analysis artifacts
logs/       Frozen pre-flight records
thesis/     LaTeX source, literature notes, references
```

## Where the current state lives

- **`THESIS_DECISIONS.md` is the canonical source** for methodology, locked
  decisions, results and open items. If anything else disagrees with it,
  including this file, it is out of date.
- **`SETUP.md`** covers the environment, serving configuration and
  reproducibility record.
