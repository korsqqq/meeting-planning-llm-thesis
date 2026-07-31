# Thesis Execution Checklist

**Project:** *When Does a Multi-Agent LLM System Beat a Single Agent? Testing
for a Complexity Threshold on Meeting Planning under an Equal Token Budget*
**Programme:** B.Sc. Computer Science, TU Berlin
**Supervisor:** Dr.-Ing. Stefan Hillmann

**Research question:** Under an equal token budget, is there a level of task
complexity at which a hierarchical multi-agent LLM system starts to outperform
a single ReAct agent on meeting planning, and if so, is the gain worth the
added coordination overhead?

This checklist operationalises the approved exposé and adds concrete
implementation decisions that must be documented and frozen before the main
experiment. It defines what must be completed, in which order, and how each
step is verified. Progress marks are intentionally blank so completion can be
tracked against it.

**Legend:** tasks marked **▲** encode implementation decisions that go beyond
the exposé (e.g. FP8 serving, exact sampling parameters, geographic split,
quantile binning). Each of these must be justified in writing and frozen in
`THESIS_DECISIONS.md` before the pilot, so that exposé commitments and added
engineering choices stay clearly separated.

---

## Phase 0 — Foundations & environment

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 0.1 | Lock all design decisions in a single source-of-truth document (models, sampling parameters, budget rules, metrics, conditions), separating exposé commitments from added implementation decisions | THESIS_DECISIONS.md; every later artefact must cite it, conflicts resolved in favour of the doc | ☐ |
| 0.2 | ▲ Define data schemas (Instance, Plan, TokenUsage, Score, run log) with Pydantic v2 | `src/schemas/`; JSON round-trip tests pass | ☐ |
| 0.3 | Set up environments: local dev GPU (smoke tests only), CPU-only pipeline, and the final 48 GB+ GPU plan | SETUP.md; pinned Python + package versions; explicit rule "no thesis numbers from the dev machine" | ☐ |
| 0.4 | ▲ Pin the serving stack: exact vLLM version/image, Qwen3-32B-FP8 + Qwen3-8B-FP8 checkpoints, tokenizer revision | Versions recorded in the run log; 8B↔32B tokenizer parity verified by an automated test, not assumed | ☐ |

## Phase 1 — Task generator, complexity control & data splits

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 1.1 | Implement a deterministic meeting-planning instance generator (people, time windows, travel times; knobs for tightness/overlap/structure) | Same seed + args → byte-identical instance; unit tests | ☐ |
| 1.2 | Operationalise the complexity metric as binding conflict pairs, defined as a property of the **instance**, not of any particular optimal route: a pair {a, b} is conflicting iff each is individually reachable but **neither ordering** (a→b nor b→a) is feasible. The solver certifies infeasibility of both orderings; the metric must be invariant to which of several optimal solutions CP-SAT returns | Precise written definition covering: what counts as a pair, when it is conflicting, both orderings, invariance to the returned optimal plan; property test: re-solving with a different search order leaves the metric unchanged | ☐ |
| 1.3 | ▲ Keep complexity as a **continuous** value; derive frozen bins only for presentation tables (quantile-based on pilot data, not a priori); use 5–7 levels instead of 3 if the pilot yields enough data | Continuous metric stored per instance; binning function + test; boundaries computed on the pilot set only | ☐ |
| 1.4 | Verify the knob→complexity mapping covers the full easy→hard range | Sweep plot of conflicts vs. knobs; sampling plan for the pilot grid | ☐ |
| 1.5 | Create explicit data splits: **development**, **pilot calibration**, **held-out main test**, and **external validation (NATURAL PLAN subset)** | No duplicate or near-duplicate instances across splits; main-test data never used for any calibration decision | ☐ |

## Phase 2 — Oracle (ground truth, never shown to agents)

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 2.1 | Implement the CP-SAT solver for the planning task (optimum value + one optimal plan), fully deterministic | Fixed seed, single worker; identical output across repeated runs | ☐ |
| 2.2 | Implement an independent brute-force solver for small n | Separate code path, no shared logic with CP-SAT | ☐ |
| 2.3 | Cross-check CP-SAT vs. brute-force on a large batch of small instances | 100% agreement on the optimum; any mismatch = stop and fix | ☐ |
| 2.4 | Implement the plan validator (checks an agent plan against the instance; typed invalid reasons) | Validator test suite; invalid-reason taxonomy documented for later error analysis | ☐ |
| 2.5 | Establish the solver tractability ceiling (max n with a proven optimum in bounded time) | Documented instance-size limit; the solver raises instead of logging unproven optima | ☐ |

## Phase 3 — Budget infrastructure (the equal-budget invariant)

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 3.1 | Tokenizer wrapper: count input via the exact chat template; count generated output, logging the thinking/answer split where the backend reliably separates them | Local token counts produced with the pinned tokenizer and exact chat template match the serving endpoint's usage accounting on representative live requests | ☐ |
| 3.2 | Budget ledger shared per instance: all tokens count (input, generated, repeated context, tool schemas, inter-agent messages). Hard-budget identity: `total = all input tokens + all generated tokens`; the thinking/answer split is additional logging, not the budget basis | Ledger identity enforced; unit tests | ☐ |
| 3.3 | Budget guard: pre-call check, per-call `max_new_tokens` cap, fixed finalisation reserve, forced routing to finalise | Property: no sequence of calls can dip into the reserve | ☐ |
| 3.4 | ▲ LLM client against the OpenAI-compatible endpoint with fixed sampling (temperature 0.6, top-p 0.95, top-k 20, seed 42, thinking ON) | One code path for dev and main run; live smoke test against the endpoint | ☐ |
| 3.5 | ▲ Finalisation node: one uniform **LLM finalisation call** (thinking OFF, guided/schema-constrained JSON decoding bound to the answer schema) that serialises `best_plan_so_far` from state — never the trajectory. Identical across all conditions, counted in the shared ledger, covered by the fixed reserve | Empty plan → honest score 0, never a crash; single decode policy logged; reserve provably sufficient for the short serialisation prompt + JSON | ☐ |

## Phase 4 — Agent conditions (LangGraph, fixed workflows)

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 4.1 | Shared tool set for all conditions: `list_people`, `get_availability`, `get_travel_time` (pure functions of the instance) | Identical tool docs/text across C1–C4; tests | ☐ |
| 4.2 | **C1** ReAct baseline: Thought→Action→Observation loop, budget guard on every step, `best_plan_so_far` tracked | Offline tests + live vertical slice generator→solver→C1→score | ☐ |
| 4.3 | ▲ **C2** ReAct + verify/revise: fixed Draft→Verify→Revise×2 structure on top of the shared C1 core | C1 behaviour, prompts, tools, workflow transitions and configuration remain unchanged after adding C2; regression tests and configuration hashes confirm equivalence | ☐ |
| 4.4 | ▲ **C3** hierarchical MAS: supervisor with a **fixed** splitting strategy (geographic, k=2), two workers, aggregator, critic; one shared ledger, inter-agent messages count toward the budget | Budget symmetry documented; empty-partition rule fixed in advance (no hidden retries) | ☐ |
| 4.5 | **C4** planner + fresh-context critic (optional, cut first if scope is tight) | Explicit go/no-go decision recorded | ☐ |
| 4.6 | Pre-register interpretation boundaries (e.g. what C2's reflection can and cannot fix; what the C3 aggregator may not do) | Written in THESIS_DECISIONS.md before the pilot | ☐ |
| 4.7 | ▲ Validate and justify the geographic split before the pilot: behaviour on uniform / clustered / line travel structures (no accidental advantage on clustered instances); empty-partition handling; aggregator merges only, never re-plans; critic cannot alter the split | All rules written and frozen before the pilot; split behaviour demonstrated on instances from each travel structure | ☐ |

## Phase 5 — Evaluation harness & controls

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 5.1 | Scorer: `S = achieved / oracle optimum`; independent revalidation; any violation → S = 0; optimum-0 rule | The scorer never trusts agent self-reports; boundary check `achieved ≤ optimum` | ☐ |
| 5.2 | Runner: execute condition × level × cap × instance cells with paired instances and a full structured JSON log per instance | One log schema; all seeds (generator/solver/inference) logged | ☐ |
| 5.3 | Secondary metrics pipeline: feasibility rate, optimality gap, invalid breakdown, number of calls, latency, satisfaction per 1k tokens | Computed at aggregation, not inside the scorer | ☐ |
| 5.4 | Implement the **full-information control**: same instances and scoring, but all facts supplied directly in the prompt (no tool calls), to separate planning ability from information gathering. Every architecture receives the identical fact set, the identical answer contract, and the identical shared token cap; the enlarged prompt input is counted in the ledger like any other input | Same scorer and budget accounting; results stored separately from the primary benchmark | ☐ |
| 5.5 | Re-encode the **NATURAL PLAN meeting-planning subset** into the Instance schema for external validation | Conversion rules documented; every converted instance manually verified; oracle scoring runs on converted instances | ☐ |

## Phase 6 — Pilot (Qwen3-8B FP8, real GPU)

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 6.1 | Generate and solve 200+ instances on the **pilot calibration split**; measure the complexity distribution | Frozen pilot instance set; distribution report | ☐ |
| 6.2 | Confirm budget caps 2k/4k/8k/16k are informative (2k not degenerate under thinking mode; headroom at 16k) | Adjust caps if needed; decision recorded before the main run | ☐ |
| 6.3 | Run mandatory pre-flight checks: tokenizer parity test in strict mode, pinned versions, live smoke of every condition | Archived test output as evidence | ☐ |
| 6.4 | Pilot run of all conditions on a subset; manual inspection checkpoint (read real trajectories, classify invalid plans) | Pilot report; go/no-go + any pre-declared ablation triggers | ☐ |
| 6.5 | **Mandatory** stochastic variance probe on a balanced subset: 10–20 instances covering easy/medium/hard, at least two caps, C1 and C3, 5 inference seeds each | Variance across inference seeds quantified; assessment of how much a single fixed seed can move the crossover | ☐ |
| 6.6 | Freeze pilot-derived complexity boundaries and cap values; apply them **unchanged** to the held-out main-test set | Boundaries and caps stored in the frozen configuration; no recalibration on main-test data | ☐ |

## Phase 7 — Main run

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 7.1 | Freeze everything: code tag, instance sets, caps, boundaries, prompts, versions | Git tag + frozen config; no changes after this point | ☐ |
| 7.2 | Full **Qwen3-32B** grid on the held-out main-test set: condition × complexity level × budget cap, with **at least 50 held-out instances per complexity level and budget cap**, one seeded run per cell | Complete paired result matrix; complete JSON logs; failures re-run only per a pre-declared rule | ☐ |
| 7.3 | Selected **Qwen3-8B robustness sweep** (not the full grid): a pre-declared subset of cells to check whether the qualitative pattern holds at the smaller scale | Selection rule documented **before** inspecting robustness results | ☐ |
| 7.4 | Run the full-information control and the NATURAL PLAN subset (selected cells) | Results stored separately from the primary benchmark | ☐ |
| 7.5 | Integrity checks after the run: budget never exceeded, every instance has a scored result, oracle agreement spot-checks | Automated audit script over the logs | ☐ |

## Phase 8 — Analysis & statistics

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 8.1 | Aggregate satisfaction per bin (mean + median robustness check); keep the continuous complexity value alongside the binned view | Result tables per condition × level × cap; per-instance table with continuous complexity | ☐ |
| 8.2 | Crossover detection — primary: paired mean difference Δ = S_C1 − S_C3 with paired bootstrap 95% CIs, where the bootstrap **resamples paired instances with replacement** (never individual runs or aggregated means). A crossover is claimed only if all three criteria hold: (1) the sign of Δ flips from + to − as complexity rises; (2) the CI excludes zero both before and after the flip; (3) the pattern repeats on at least two budget caps. Architecture × complexity interaction tested **also on continuous complexity**. Cohen's d reported as a supplementary effect size only | Analysis script, fully reproducible from the logs | ☐ |
| 8.3 | Secondary analyses: efficiency (satisfaction per 1k tokens), invalid-error taxonomy, per-cap stability, feasibility difference, 8B vs. 32B robustness, full-information control vs. tool-based results | Figures + tables for the thesis | ☐ |
| 8.4 | Write up confounds and limitations (vLLM non-determinism, ceiling/floor effects, single split strategy, latency vs. tokens) | Limitations section draft | ☐ |
| 8.5 | Stochastic robustness analysis from the repeated-run subset (Phase 6.5 design, re-run on the main model if budget allows). The **instance remains the statistical unit**: multiple inference seeds are repeated measures of the same instance, never treated as independent observations | Variance across inference seeds reported; stated impact on the crossover conclusion | ☐ |

## Phase 9 — Thesis writing

| # | Task | Deliverable / How to verify | Done |
|---|------|------------------------------|------|
| 9.1 | Related work: ReAct, multi-agent planning, LLM planning benchmarks, token-budget studies | Literature notes → Related Work chapter | ☐ |
| 9.2 | Method chapter: task, complexity metric, oracle, budget accounting, conditions — written so a reader can reproduce the experiment; exposé commitments vs. added implementation decisions clearly separated | Cross-checked against THESIS_DECISIONS.md | ☐ |
| 9.3 | Results + Discussion: answer the RQ for all three possible outcomes (crossover exists / single agent always wins / MAS always wins) | All outcomes treated as valid results, per the exposé | ☐ |
| 9.4 | Reproducibility appendix: seeds, versions, run configs, hardware | Everything needed to re-run from the frozen tag | ☐ |
| 9.5 | Final review pass with the supervisor; proofread; submit | — | ☐ |

---

## Ordering rationale

- **Oracle before agents (Phase 2 before Phase 4):** without ground truth,
  neither the complexity metric nor the scoring can be validated; the
  CP-SAT↔brute-force cross-check is the only protection against a silently
  wrong optimum.
- **Budget infrastructure before conditions (Phase 3 before Phase 4):** the
  equal-budget invariant is the core of the research question; if token
  accounting is wrong, the C1 vs. C3 comparison is meaningless regardless of
  agent quality.
- **Splits before calibration (Phase 1.5 before Phase 6):** every calibration
  decision (boundaries, caps) is made on the pilot split and applied unchanged
  to the held-out main-test set, so nothing is tuned on the data that produces
  the headline numbers.
- **Pilot as a separate gate (Phase 6):** complexity boundaries, final budget
  caps, the variance probe and strict pre-flight checks are completed *before*
  the main run, so nothing is tuned post hoc.
