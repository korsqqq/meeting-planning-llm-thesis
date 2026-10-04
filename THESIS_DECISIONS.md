# Thesis Design Decisions

**Project:** When Does a Multi-Agent LLM System Beat a Single Agent? — meeting
planning under an equal token budget (B.Sc., TU Berlin, supervisor Prof. Hillmann)
**Version:** v1 · **Date:** 2026-06-23 · **Status:** locked (pilot-dependent values flagged)

This file is the single source of truth for finalised design decisions. It
supersedes ad-hoc notes. Values that are deliberately deferred to the pilot are
marked **[PILOT]**. Items still open are in §6. Confounds to disclose are in §7 (including
the pre-specified limitations L1–L5 drawn from the MAS literature). §8 records the
literature-derived design audit behind them.

---

## 1. Model

**Weights (final run, AWQ INT4, one model per 48 GB GPU):**
- Main: `Qwen/Qwen3-32B-AWQ`
- Robustness / pilot: `Qwen/Qwen3-8B-AWQ`
- Quantisation parity is guaranteed by construction: both are official AWQ INT4
  checkpoints from the same family, so the 8B/32B comparison is not confounded by
  differing quantisation.
- Dev/debug only: Qwen3-8B Q4 on RTX 2080 Super. No number from the dev machine
  enters the thesis.

**REVISED 2026-07-31 — FP8 → AWQ INT4 (hardware-forced).** The original decision
named `Qwen/Qwen3-32B-FP8` and `Qwen/Qwen3-8B-FP8`. The confirmed run machine has
2 × RTX A6000, which is Ampere (SM 8.6) and has no native FP8 tensor cores: an FP8
checkpoint could only be served through a dequantising weight-only path, i.e. FP8
storage with fp16 arithmetic, which is not the numerical regime the FP8 decision
assumed. The official AWQ INT4 checkpoints are natively supported on Ampere
(`awq_marlin`), keep the 8B↔32B quantisation parity that the original decision was
made for, and leave a 48 GB card with ample KV-cache headroom at
`tensor_parallel_size=1`. The interpretive cost — results describe INT4-quantised
Qwen3, and INT4 is a stronger compression than FP8 — is recorded in §7.

**EXCLUDED 2026-08-04 — `Qwen/Qwen3-8B-AWQ` fails the pre-pilot feasibility check.**
C1 on the 8B stays at satisfaction 0 across the whole budget range tested (4k, 8k,
16k, 32k; three seeds each, easy instances with zero binding conflicts). The cause is
behavioural, not budgetary, and the transcripts settle it: across every run the model
issued **zero** `Action: propose` and no malformed ones either — no tool error, every
action a well-formed query — so this is not a parser or format mismatch. It explores
the travel matrix until the budget guard stops it (token totals sit at `cap - 156` in
almost every run, the signature of a loop that always runs to the guard) and never
commits to a plan. More budget buys more exploration, not a plan.

Consequences, all deliberate:
  * The 8B is **not used for cap calibration**: a model on the floor everywhere cannot
    locate the transition the ladder is supposed to bracket. Calibration runs on the
    32B.
  * No further GPU time is spent on it.
  * **This is a scope reduction, not a completed objective.** "Robustness across two
    model sizes" is *withdrawn* pending a decision on a different second model — it is
    NOT to be reported as satisfied, and no robustness claim may rest on the 8B runs.
    Whether a second model returns, and which, is an open item (§6).
  * The finding itself is reportable and belongs in the results, not in a footnote:
    under this scaffold a 8B-class model does not move from exploration to commitment
    at any budget tested. One pilot probe is consistent with the hierarchy mattering
    here — C3 on the 8B scored 0.25 on one of three seeds at cap 16000, against twelve
    consecutive zeros for C1 — but one seed of three is a hint, not an effect, and no
    claim rests on it.

**Reasoning mode:**
- thinking **ON** (`enable_thinking=True`); the model emits a `<think>...</think>`
  block followed by the final answer.
- Thinking tokens **count toward the budget** (they are generated and capped by
  `max_new_tokens`). This is symmetric across all conditions and mirrors the
  thinking-token control of Tran & Kiela (2026).

**Sampling (mandatory for thinking mode, per Qwen):**
- `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`
- Greedy decoding is forbidden (Qwen warns it causes degradation and endless
  repetition in thinking mode).
- `presence_penalty=0` by default; raise toward 1.5 only if endless repetition
  appears (document if used — higher values risk language mixing).

**Reproducibility:**
- Fixed seed, `n=1` run per cell (condition × level × cap × instance).
- **Prefix caching OFF (`--no-enable-prefix-caching`), decided 2026-08-04 and frozen
  for the pilot and the main run.** This is what makes `n=1` defensible, and it was
  measured, not assumed. With vLLM's default prefix caching ON, repeating one cell
  (C1, 32B, cap 32000, instance seed 2) five times with everything else identical
  gave satisfaction 0.00 / 0.75 / 1.00 / 1.00 / 1.00 — mean 0.75, sd 0.43, the full
  range of the metric. At that spread a single run of a cell near the decision
  boundary can report anything, and detecting a 0.2 difference between conditions
  would have needed on the order of 70 runs per cell, which makes the grid
  uncomputable. With prefix caching OFF the same five repeats were **identical**
  (22 220 tokens, satisfaction 0.00 each). The residual non-determinism was therefore
  the cache, not floating-point reduction order: requests were issued one at a time,
  so batch composition never varied, but each run met a different cache state.
- Determinism also holds **across servers**: the same cell run on a second identical
  server on the other GPU (same version, revision and flags) reproduced 22 220 tokens
  and satisfaction 0.00 exactly. The sweep may therefore be sharded across both A6000s
  without introducing a confound; both servers must carry identical flags, and the
  fact that two were used is recorded in `SETUP.md` §7.
- Two consequences, both deliberate. Determinism costs throughput: every ReAct step
  re-encodes the whole prefix, and one 32k-cap C1 run takes 7 min 47 s (~47 tok/s
  effective), which is what the pilot's time budget must be built from. And a
  deterministic run fixes ONE trajectory rather than the mean of the cached-on
  distribution — on the cell above the fixed trajectory scores 0.00 where the
  cached-on mean was 0.75. That is not a worse measurement, it is a reproducible one;
  variation across instances, not across repeats of one instance, is what carries the
  statistics.
- **Every number measured before 2026-08-04 came from the cache-ON regime and does not
  transfer.** The cap-ladder calibration of 2026-08-04 is superseded and must be
  re-taken.
- A variance probe is no longer needed to quantify run-to-run spread (it is zero by
  construction); it is retained only as a check that the setting is still in force.

**Serving:**
- Hardware (confirmed 2026-07-31): 2 × RTX A6000 (49 140 MiB each), Threadripper
  PRO 7955WX (16 cores / 32 threads), 502 GiB RAM, Ubuntu 22.04.4, driver
  550.144.03 / CUDA 12.4. Setup procedure and the frozen version record:
  `SETUP.md`.
- **One model per GPU, `tensor_parallel_size=1`** (32B on GPU 0, 8B on GPU 1)
  rather than TP=2 for a single model. Both INT4 checkpoints fit on one card, so
  TP=2 would only add cross-GPU synchronisation (PCIe, no NVLink assumed) and a
  further source of run-to-run non-determinism on top of the batching effect
  already disclosed in §7. Separate servers also let the 8B pilot and the 32B main
  run proceed independently.
- vLLM, OpenAI-compatible endpoint (`$VLLM_URL`). The vLLM version is pinned to
  ONE exact release (or Docker image digest), frozen when the pilot environment is
  set up and recorded in the run log — not a floor such as `>= 0.8.5` — so that
  chat templating, sampling behaviour and response format cannot drift between
  pilot and main run.

**PINNED 2026-08-03 — `vllm==0.19.1`, and it is not the latest release.** Every
release from 0.20.0 onward depends on torch 2.11, which is built against CUDA 13 and
refuses to initialise on the run machine's driver 550.144.03 (CUDA 12.4): the engine
dies in `torch._C._cuda_init()` before any weights are read. Minor-version
compatibility holds only inside a CUDA major version, so this is not something the
harness can work around; updating the driver needs root and a reboot of a shared
machine. 0.19.1 (torch 2.10.0+cu128) is the newest release still built against CUDA 12.
This is an **environment constraint, not a quality-motivated choice**, and it is
reported as such.

The version choice is visible to the harness in exactly one place: constrained
decoding is requested as `structured_outputs` (`{"json": <schema>}`), not
`guided_json`, which 0.19.1 rejects. Only the wire payload in
`src/core/llm_client.py` changed; the `guided_json` keyword argument that
`react_core` and every test fake use is unchanged. Pre-flight probe F is what
decides this, and it must be re-run if the pin ever moves (`SETUP.md` §5.1).
Everything else the harness depends on was verified present on 0.19.1:
`return_token_ids` for the exact thinking/answer split, the `</think>` delimiter on
raw completions, and input-token parity with the endpoint (delta 0).
- **Reasoning extraction:** the main run uses the raw `/v1/completions` endpoint,
  NOT vLLM's server-side `--reasoning-parser` (which applies to chat-style
  endpoints only). The harness renders and counts the exact Qwen3 chat-template
  prompt itself, sends it as a raw completion, and splits the generated output at
  the LAST `</think>`: tokens up to and including the delimiter are thinking
  tokens, the rest are answer tokens (the opening `<think>` is not required — the
  official Qwen3 example also splits on the closing delimiter only). If no
  `</think>` is generated before a length stop, the entire output counts as
  thinking and the post-think content is empty (-> finalise immediately, §2).
- `max-model-len ~32k` (native) — comfortably covers the 16k cap plus scaffold;
  YaRN not needed.
- Token accounting uses the Qwen3 tokenizer (not an API counter).

---

## 2. Token budget & accounting

**What counts (per instance, summed over all calls):**
`input + generated (thinking + answer) + repeated context + tool schemas +
inter-agent messages (MAS)`.

- Repeated context is counted every call (the model is stateless; full history is
  re-sent each time). This is a real architectural cost and is included on purpose.
- Thinking tokens are included in the budget; they are additionally logged
  **separately** so `satisfaction / 1k tokens` can be computed both ways
  (full call cost vs. planning-only).

**On budget exhaustion:**
- The harness force-stops the agent and returns `best_plan_so_far` (the best valid
  intermediate plan recorded so far). An empty `best_plan_so_far` scores 0 — an
  honest zero.

**Mechanics:**
- A single budget ledger per instance, shared by all agents (identical for C1 with
  many calls and C3 with several agents).
- A fixed part of the budget is reserved for finalisation so a plan is always
  returned.

**Counting & ledger (implemented Layer 4, src/core/):**
- Counted with the Qwen3 tokenizer (`src/core/tokenizer.py`), split per call:
  - **INPUT** before the call = `apply_chat_template(history + prompt + tool schemas,
    add_generation_prompt=True, enable_thinking=True)` — the exact rendered string
    sent to vLLM, including role tags and special tokens (raw text undercounts).
    Requires `jinja2` for the chat template.
  - **OUTPUT** after the call = the generated output split at the LAST `</think>`:
    tokens up to and including the delimiter -> thinking_tokens, tokens after ->
    answer_tokens. The delimiter tags and inter-part whitespace are part of
    thinking_tokens — they were generated, so they are budget. The split is done on
    the generated token ids when vLLM returns them (`return_token_ids`, exact);
    otherwise the raw text segments are counted with the same Qwen3 tokenizer.
  - Identity enforced by the ledger / `TokenUsage`: `total = input + thinking + answer`.
  - Counting tokenizer: `Qwen/Qwen3-8B-AWQ` (`DEFAULT_TOKENIZER`,
    `src/core/tokenizer.py`) — the tokenizer of the served checkpoint itself, not the
    base repo (the AWQ repos are separate HF repos with their own tokenizer files and
    revisions). AWQ-8B/AWQ-32B parity is **verified empirically, not assumed**: an
    automated test asserts identical rendered chat-template strings AND identical
    token-id sequences on five representative prompt shapes currently used by the
    harness (plain system+user, tool schemas, ReAct Thought/Action/Observation
    history, the thinking-OFF finalise prompt, non-ASCII content) —
    `tests/test_tokenizer.py::test_tokenizer_parity_across_qwen3_sizes`.
  - What parity does and does not buy (corrected 2026-08-03). It means one counting
    tokenizer is valid for both models: a token counted for the 8B is the same token
    for the 32B, so the budgets being compared are the same unit. It does **not** mean
    a calibrated cap transfers between model sizes — that was the earlier claim and it
    is wrong. On the identical pre-flight prompt the 8B spent 2 315 reasoning tokens
    against the 32B's 413, and at cap 16000 on the same easy instance the 32B reached
    satisfaction 1.00 while the 8B scored 0. The same budget therefore buys a very
    different number of steps per model, and the cap ladder is calibrated **per
    model**. **[PILOT]**
  - The parity test skips when a tokenizer cannot be downloaded (offline dev is
    fine). Before the pilot it is MANDATORY: run the suite with
    `REQUIRE_TOKENIZER_PARITY=1` (a load failure then FAILS instead of skipping)
    and archive the `... passed, 0 skipped` output as the parity evidence. **[PILOT]**
  - The tokenizer `revision` (exact HF commit, per checkpoint) is pinned before the
    pilot via `QwenTokenizer(revision=...)` and recorded in the run log — chat
    templates on the Hub can change independently of the weights. **[PILOT]**
- `BudgetLedger` (`src/core/budget.py`) is framework-agnostic (no LangGraph yet) and
  maps onto the `TokenUsage` schema. Guard arithmetic: `max_new_tokens(input,
  finalizing)` keeps the finalisation reserve for ordinary calls; `should_finalize()`
  routes to the final node once reasoning room is gone; `record_call()` refuses an
  overspend (the guard must cap `max_new_tokens` first).

**Guard limits fixed after the first live runs (recorded 2026-08-03, before the pilot):**

The first live runs on the A6000 machine showed that the guard as specified above does
not by itself make the budget the binding constraint. Three limits are fixed here. All
three live in code shared by every condition, so none of them can favour one condition
over another, and all three are **[PILOT]** — re-checked against measured step costs
before the main run. Live-smoke results recorded on 2026-08-02/03 predate these limits
and are superseded: they describe a different system.

- **Per-call ceiling: proposed, TRIED, and currently OFF (`max_call_tokens = None`).**
  The motivation stands: `max_new_tokens` grants a single call the entire remainder, so
  one thinking block that never closes destroys the budget at once — on Qwen3-8B-AWQ at
  cap 16000 one call spent 5083 tokens and returned nothing after `</think>`, 32% of the
  run lost to a single step. Enabling a 2000-token ceiling made this **worse**, and the
  reason is structural rather than numeric: truncating a think block leaves nothing
  after `</think>`, which the empty-post-think rule below reads as "the agent produced
  nothing" and routes straight to finalisation. Measured on the same instance and cap,
  both models then ended on their ninth step with the budget still half unspent, and C1
  on Qwen3-32B-AWQ fell from satisfaction 1.00 to 0.00 at cap 16000. A ceiling is
  therefore not admissible on its own: it requires a way to distinguish "the harness cut
  this call short" from "the agent had nothing to say", and that distinction touches a
  locked rule. Recorded here so the attempt is not repeated blindly. **[PILOT]** —
  decide the ceiling and the distinction together, or accept the unbounded grant and
  report the wasted-budget share in the error analysis.

- **Step cap derived from the budget: `max_steps = max(1, cap // 250)` unless passed
  explicitly** (`steps_for_cap`, `src/harness/runner.py`). The research question is
  about behaviour under an equal TOKEN budget, so the step counter must never bind
  first; otherwise the independent variable is silently the number of steps. The
  previous fixed `max_steps = 12` was already at the edge: C1 on Qwen3-32B-AWQ solved
  the EASIEST instance (n = 4, zero binding conflicts) in 11 steps, so any harder
  instance would have been cut by the counter rather than by tokens. The divisor is a
  conservative lower bound on the cost of one ReAct step (input + thinking + answer);
  the cheapest step observed live cost ~430 tokens. The counter stays what it is meant
  to be — a runaway guard behind the budget and the `aborted` path, not an experimental
  knob.

- **Prompt instruction to propose early: TRIED and REVERTED.** The observation that
  motivated it stands: both models spent ten steps on tool calls and never issued
  `Action: propose`, so low-cap runs score 0 on an empty plan that finalisation emits
  faithfully — an honest zero that measures nothing. The attempted wording told the
  agent not to gather every fact before planning, to propose as soon as a route is
  workable, and to propose again on improvement. Measured on the same instance and cap,
  it made C1 on Qwen3-32B-AWQ *worse*: satisfaction fell from 1.00 to 0.00 at cap 16000,
  the agent stopped terminating on its own, and it exhausted the budget (15726 of 16000)
  while still never proposing. The plausible reading is that "keep exploring and propose
  again" rewards more exploration rather than earlier commitment. The prompt is back to
  its previous wording.

  Two lessons are recorded rather than the wording: a prompt change is a change to the
  measured system and needs a multi-seed A/B, not a single confirming run; and the
  stronger alternative — disclosing the remaining budget — remains deliberately NOT
  adopted, since it changes the agent's information position and would weaken
  comparability with the reported literature. **[PILOT]** — if the propose-never
  behaviour is to be addressed at all, it is addressed with evidence, and the
  no-intervention option (report the floor as a property of the model-budget pair) is
  a legitimate outcome.

**Unified finalisation policy (LOCKED, one invariant + its consequences):**

*Invariant.* finalise ALWAYS runs with >= `reserve` (256 **[PILOT]**) tokens, however the
loop ended. No reasoning node can dip into the reserve. If this holds mathematically, an
"empty answer" stops being a bug and becomes expected behaviour.

*What finalise IS.* finalise is the **stateless serialisation of `best_plan_so_far`**, NOT
a continuation of reasoning. Its input is a short serialisation prompt + the structured
`best_plan_so_far` object from state -- **NOT the whole trajectory**. Output: guided-JSON
`AnswerContract`, thinking OFF. That is why 256 suffices (short prompt + JSON <= 131). The
original MALFORMED bug was finalise re-encoding the full ~3900-token trajectory to emit a
131-token plan -- the reserve could never cover that. `best_plan_so_far` is held in state
separately (a harness requirement); empty -> finalise honestly emits a null plan -> score
0 ("the agent found nothing within budget", not a crash).

*Serialisation is FAITHFUL, and that is enforced in code (fixed 2026-07-26).* Guided decoding
constrains the emit's SHAPE, not its VALUES, so "serialisation" was previously an intention
rather than a guarantee: the implementation accepted any parseable emit, meaning a model that
altered a start time, dropped or added a person, or reordered the route would have had its
altered plan scored. The emit is now accepted **only when it equals `best_plan_so_far`
exactly**; otherwise the structural plan is used. Consequences, all deliberate:
  * the scored semantic output is always `best_plan_so_far`, by construction, in every
    condition -- so the finalisation node can no longer influence any score;
  * divergences are not repaired and not retried, only recorded as
    `finalization_mismatch` per run (a pilot integrity diagnostic on the terminal emit, §7);
  * because `best_plan_so_far` only ever accepts validator-approved plans, the scored plan is
    now always valid or empty. The consequence for the invalid-breakdown secondary metric is
    recorded as an open item in §6.

*Mechanics that keep the reserve untouchable* (per reasoning call -- agent / verify /
revise / worker / supervisor / critic):
1. Count `input_k` via `apply_chat_template` (already done for `count_input`).
2. Guard BEFORE the call: if `remaining - input_k <= reserve` -> do not call, go to
   finalise. This catches input GROWTH (re-encoding) -- the thing that burned the budget.
3. Per-call output cap: `max_tokens = max(0, remaining - reserve - input_k)`. A call cannot
   physically touch the reserve even if it spends its whole cap inside `<think>`.
   (`BudgetLedger.should_finalize` / `max_new_tokens` already implement exactly this.)

*Empty answer is expected, not an error.* Under a tight cap, Qwen3-32B with thinking ON can
spend its (already-capped) `max_tokens` inside `<think>` and be cut off before emitting the
Action/plan -> empty post-think content. This is a real vLLM failure mode, not Ollama-only
(Ollama just surfaced it earlier). Rule: **empty post-think content -> finalise immediately**
(a retry has the same budget -> same outcome, and wastes another `input_k`). One retry is
kept only for a NON-empty format failure (the model produced something unparseable).

*The terminal emit* is pinned to **guided / JSON-schema decoding** bound to `AnswerContract`
(vLLM `guided_json`; Ollama `format`). Schema-constrained output makes the non-thinking
sampling profile near-irrelevant (no second sampling regime to justify) and kills format
failures; a single decode policy is logged.

*Symmetry across C1-C4 (else confound).* Identical on every condition: the `reserve` value;
the guard formula; the per-call cap formula; the empty->finalise rule; the one-retry cap on
non-empty format failure; and the definition of finalise as serialisation of
`best_plan_so_far`. Only the SET of nodes differs, never the budget discipline.

*Park-flag for C3 -- RESOLVED (locked in §4 C3-6):* in the MAS, an empty worker is not the
same as an empty single agent. Decision: the empty->finalise rule is scoped to the
ENCLOSING loop -- an empty worker ends only its own sub-loop and the sibling worker still
runs (the next scheduled stage of the fixed workflow on different input, not a retry of
the failed one). The hidden-retry boundary is enforced at the critic instead: an empty
candidate pool (no worker produced any valid sub-plan) skips the critic and routes
straight to finalise, because a critic over an empty pool would be a fresh solo planning
attempt -- exactly the extra effective retry (hidden budget bonus) this flag forbids.

**Caps:** 2k / 4k / 8k / 16k tokens. **[PILOT]** Final cap values are confirmed after
the pilot on the AWQ INT4 checkpoints; 2k may be lowered if not tight enough, the top
raised if no crossover appears by 16k. The ladder stays geometric (doubling). The first
live runs suggested the ladder moves **up**, not down, and that it is calibrated
separately for each model, but those runs came from the cache-ON regime (§1) and are
superseded — see the deterministic calibration below.

**Deterministic cap calibration — INTERIM, easy `n=4` only (2026-08-04).** C1 on
Qwen3-32B-AWQ, prefix caching off, pilot seeds 10000–10004 (the first five qualifying
instances in ascending seed order, §4 C3-8), five runs per cap. **The ladder is not
fixed here**: this is one complexity level, and medium and hard must be calibrated
before any cap decision is locked.

| cap | proposal attempts | parse | validity | acceptance | termination |
|---|---|---|---|---|---|
| 4 000 | 0 | n/a | n/a | n/a | budget 3, aborted 2 |
| 8 000 | 0 | n/a | n/a | n/a | budget 1, aborted 4 |
| 16 000 | 2 | 1.00 | 0.00 | 0.00 | aborted 4, budget 1 |
| 32 000 | 7 | 1.00 | 0.29 | 0.29 | agent_finish 5 |

- **4k and 8k produced zero proposal attempts.** In this slice those rungs therefore
  measure mostly *failure to reach a proposal at all*, not plan quality. A zero there
  must not be read as "the agent planned badly", and the three proposal rates are
  undefined (empty denominator) rather than zero.
- **16k is a transition zone.** Proposals appear (2 attempts over 5 runs) but
  empty-post-think termination still dominates: 4 of 5 runs end `aborted`.
- **32k is the first stable working regime.** All five runs end via `agent_finish` —
  none aborted, none out of budget — and proposal validity is 0.29.
- **64k is not justified for easy `n=4`**: at 32k the runs stop with 5–8k of the budget
  unspent, so extra budget would not be consumed. This is **not** yet a reason to drop
  the rung for medium/hard, where the runs may well use it.
- **The only validator rejection reason observed in this slice is `travel_infeasible`**
  (7 of 7 rejections).
- `proposal parse rate` = 1.00, and `acceptance` equals `validity`. Both are properties
  of THIS easy slice, not general claims: no malformed proposal occurred, and the agent
  proposed complete routes only, so the strictly-longer conjunct never bound.

**C1 cap calibration on a calibration-exposed subset (2026-08-05).** 120 runs, C1 on
`Qwen/Qwen3-32B-AWQ`, prefix caching off, context window 32768, 30 instances × four caps.
The subset contained one previously exposed instance (§3), so this is **calibration
evidence for choosing the ladder, not a formal architectural comparison**; nothing about
C1 versus C2/C3 may rest on it. Artifacts: `results/calibration/c1_20260805/`, analysis
in `results/analysis/c1_calibration/`, which refuses to run unless the set passes a
fail-loud integrity audit.

| cap | satisfaction | tokens used | attempts | parse | validity | acceptance | runs with ≥1 valid |
|---|---|---|---|---|---|---|---|
| 8 000 | 0.011 | 7 660 | 1 | 1.00 | 1.00 | 1.00 | 0.03 |
| 16 000 | 0.011 | 15 781 | 9 | 1.00 | 0.11 | 0.11 | 0.03 |
| 32 000 | 0.300 | 29 660 | 30 | 1.00 | 0.33 | 0.33 | 0.33 |
| 64 000 | 0.502 | 40 631 | 51 | 1.00 | 0.33 | 0.33 | 0.53 |

- **64k is informative and not saturated.** Satisfaction rises 0.300 → 0.502 while actual
  consumption rises only to 40 631 of 64 000. Paired on the same 30 instances, 32k → 64k
  **improved 7, tied 23, worsened 0**; mean Δ +0.20, median 0. In this deterministic
  paired calibration, no instance had lower satisfaction at 64k than at 32k. The rung
  stays.
- **`context_limited` at 64k means a reduced ALLOWANCE, not truncation.** 408 of 482 calls
  had their grant clamped by the window, but only **2** of those ended with
  `finish_reason=length`; 406 stopped on their own. At 8k and 16k the 18 and 26 length
  stops are *not* context-limited at all -- there the binding constraint is the budget
  grant. The two mechanisms are cleanly separable in the log, which is what the field was
  added for.
- **The gain is more runs reaching a valid plan, not better aim.** Per-attempt validity is
  flat at 0.33 across both upper rungs, while the share of runs producing at least one
  valid proposal rises 0.33 → 0.53. Attempts per run rise only 1.0 → 1.7. And the median
  index of the first valid proposal is **0** at both caps: when a run succeeds, it
  succeeds on its first proposal. So the extra budget does not buy repeated tries that
  eventually land -- it buys more runs that get far enough to propose at all.
- **Sensitivity.** Removing the exposed instance moves satisfaction 0.300 → 0.276 (32k)
  and 0.502 → 0.484 (64k), and the paired counts from 7/23/0 to 7/22/0. Excluding the
  exposed instance changes the point estimates slightly but does not alter any
  qualitative conclusion.

*On the mechanism behind the low-cap aborts — stated carefully.* What is measured is
that the visible post-think output was empty and the section-2 rule therefore routed
the run to finalisation. At the time of the runs above the log did **not** distinguish a
generation cut short by the grant (a length stop) from a model that closed `</think>`
and then emitted nothing: the token counts are identical in both cases. A shrinking
per-call grant at low caps is therefore a *plausible* mechanism for those runs and is
**not** asserted as fact — for the 2026-08-04 slice the honest statement is "ended on an
empty visible output", nothing stronger.

`finish_reason` is now recorded per call on `CallRecord` (optional, `None` for offline
scripted clients, which have no endpoint to report one). It is diagnostic: nothing
routes on it and the empty-post-think rule is unchanged. From the pilot onward the
distinction can be reported directly, and the low-cap aborts can be attributed rather
than guessed at. The runs above predate the field and cannot be re-labelled
retroactively — they would have to be re-taken to carry it.

---

## 3. Generator & solver (ground truth)

**Complexity metric:**
- A combined measure of: number of people + time-window tightness + travel
  structure.
- Measured **after** the instance is solved, not assumed from generator inputs.
- Operationalised as the count of binding pairwise conflict pairs in the conflict
  graph (solver-verified).
- **Binding pairwise conflict (precise definition, implemented Layer 1):** an edge
  `{a, b}` is added iff both `a` and `b` are individually reachable (each can be met
  alone in a feasible schedule) **and** no ordering — neither `a → b` nor `b → a` —
  meets both. A pair where one endpoint is individually infeasible is **not** a
  conflict (that is an isolated infeasibility / a generator artefact, not task
  difficulty). The metric is the number of such edges.

**Feasibility model (precise, implemented Layer 1 — solver / brute-force / validator
share it identically):**
- A tour starts at `start_location` at `start_time`; each person carries prize 1
  (orienteering: the route need not return to start nor visit everyone).
- A meeting for person `p` at start `s` is feasible iff
  `window_start[p] <= s`, `s + meeting_duration <= window_end[p]`, and
  `s + meeting_duration <= end_of_day`.
- Travel from the previous stop (the depot for the first meeting) takes
  `travel_times[prev][p]` minutes; the next meeting cannot start before arrival.
- **`waiting_allowed`** (confirmed semantics): `True` → `s >= arrival` (the traveller
  may wait at a location for the window to open). `False` → `s == arrival` (no idle
  time; the schedule is forced by the route). An illegal idle insertion under
  `waiting_allowed = False` is reported as `TRAVEL_INFEASIBLE` (no dedicated reason
  code exists; the validator buckets it there).

**Level boundaries (easy / medium / hard):** **[PILOT]**
1. Generate 200+ instances. 2. Solve all with CP-SAT. 3. Measure the complexity
distribution. 4. Cut bins by quantiles. Boundaries are not set manually a priori.
- **Binning method (implemented Layer 2, confirm during pilot):** the conflict
  distribution is strongly zero-inflated (~half a broad sweep has 0 conflicts), so a
  plain tertile collapses (both cuts land on 0, no medium bin). Method used: easy =
  the 0-conflict mass (`b1 = 0`, matching the "easy = 0 conflicts" anchor), then the
  POSITIVE conflicts are split at their median into medium and hard. Still derived
  from pilot data, not a priori. `level_boundaries()` returns `(0, median_positive)`.

**Ground truth (optimal plan):**
- Main: OR-Tools CP-SAT for all instances.
- Cross-check: brute-force for n ≤ 6. CP-SAT and brute-force must agree on n ≤ 6; a
  mismatch means a bug in one of them — stop and fix.
- For n > 6: CP-SAT only (brute-force is combinatorially infeasible).
- The solver is never shown to the agents — scoring and complexity measurement only.

**Oracle implementation (Layer 1, decided during Layer 2):**
- **Determinism:** CP-SAT runs single-worker (`num_search_workers = 1`, fixed seed).
  A multi-worker portfolio is faster but returns different optimal *solutions* across
  runs (only the objective value is stable), which would make the canonical optimal
  plan irreproducible. One worker gives an identical optimum *and* plan every run.
- **Speed:** the model carries three valid cuts — a global time-budget bound, a 0/1
  knapsack cut on the visit count, and pairwise-conflict cuts (the metric edges,
  added only when the matrix is metric) — so the proof closes fast despite one worker.
- **Tractability ceiling:** proving the visit-count optimum blows up past ~10 people
  on clustered, medium-window instances. At **n ≤ 9** every config closes in ≤ ~8 s;
  the solver raises (rather than logging an unproven optimum) past a 60 s budget.
  Since complexity here is conflict-driven, not headcount-driven, instance sizes stay
  at n ≤ 9 (`generator.RECOMMENDED_MAX_N`).

**Generator (implemented Layer 2):**
- Parameters: `n_people, tightness, overlap, travel_structure, seed`.
- Fixed seeds → reproducible instances (same args + seed → byte-identical instance).
- Instances are paired across conditions (C1, C2, C3 see the same instances).
- Knob → instance mapping: `tightness` sets window length (0 = whole-day windows,
  1 = windows that just hold one meeting); `overlap` sets how tightly window starts
  cluster in time (0 = spread across the day, 1 = all centred); `travel_structure`
  lays out locations (UNIFORM/CLUSTERED/LINE are metric from coordinates, RANDOM is a
  non-metric matrix). Day defaults: start 0, end 480 min, meeting 30 min.
- Knob → complexity is non-linear: binding conflicts only emerge at high tightness
  (≈ 0.8–1.0) and high overlap. The pilot must therefore sample tightness/overlap
  densely in that upper region to populate the full easy→hard range (a uniform
  `{0, 0.5, 1}` grid yields only `{0, many}` and no medium bin).

**Pilot manifest configuration — LOCKED 2026-08-04.** Frozen before any pilot seed was
read, on the evidence of a development-range preflight (below). Implemented in
`scripts/build_pilot_manifest.py`; nothing here may be re-tuned in response to pilot
results, since that would make the selection chosen rather than pre-registered.

*Grid.*
- `n_people = [4, 6, 8]` — 4 is where every measurement so far was taken, 8 sits below
  the oracle ceiling of 9 without risking the 60 s solve limit, 6 spans the middle.
- `tightness = [0.2, 0.7, 0.85, 1.0]` — dense at the top, because conflicts appear only
  there; 0.2 anchors easy.
- `overlap = [0.2, 0.8]` — low and high clustering; intermediate values duplicate the
  effect.
- `travel_structure = [uniform, clustered]`. `line` is excluded as qualitatively close to
  `clustered`, and `random` because its non-metric matrix (no triangle inequality) would
  undermine reading the metric as geometric conflict. Both stay available for robustness
  analysis OUTSIDE the pilot.
- **48 cells**, `N = 5` accepted instances per cell, **240 accepted instances** — above
  the 200+ the binning method assumes.

*Selection.*
- Seed range `10000–19999` (the pilot range, §4 C3-8).
- Each cell is scanned INDEPENDENTLY, seeds ascending, taking the first N that qualify.
- Qualifying = the instance generates AND CP-SAT **proves** the optimum AND `optimum > 0`.
  A rejected seed is consumed, never swapped for a later one; both rejection causes
  (`not_proven`, a solver limitation; `optimum_zero`, a property of the instance) are
  recorded per seed and rated per cell.
- Level boundaries are cut ONCE over the complete accepted pool, never per cell.
- The manifest carries a SHA-256 over a canonical payload with wall-clock timings
  stripped, so a rerun reproduces it exactly.

*LLM subset.* The 240-instance pool is a CPU artefact; the conditions are run on a
subset of **10 easy + 10 medium + 10 hard = 30**, selected by the implemented
round-robin rule over (n_people × travel_structure) strata, ascending seed within a
stratum. Round-robin rather than a plain ascending-seed cut, which would drain whichever
cell sorts first and lose the grid's variety exactly where the expensive runs happen.

*Technical failure criteria* (the only grounds for changing anything after the freeze):
a cell starved of qualifying seeds; a cell where the solver cannot prove the majority of
seeds considered; a write error; a content hash that does not reproduce. A high
`optimum_zero` rate is explicitly **not** one of them — it is reported, not acted on.

**Development-range preflight (seeds 0–9999, full grid, real CP-SAT, 2026-08-04).** Run
on the development range precisely so the pilot range stayed unread. Result: seeds 0–4
sufficed in **all 48 cells**, `not_proven = 0` and `optimum_zero = 0` everywhere
including the heaviest cells (`n=8`, clustered, tightness 1.0); levels came out easy 123
/ medium 64 / hard 53 at boundaries easy ≤ 0 and medium ≤ 5; wall time 3.4 s for 240
instances; the content hash reproduced byte for byte across reruns; and the stage-2
subset returned 10/10/10 with all three `n_people` values and both travel structures
represented in every level.

Those boundaries and level counts are **evidence that the grid is fit for purpose and
nothing more**. They come from different instances and do **not** transfer to the pilot:
the pilot cuts its own boundaries over its own accepted pool, and its rejection rates may
well be non-zero.

**CORRECTED 2026-08-05 — seeds 10000–10004 were calibration-exposed; the formal scan
starts at 10005.** The manifest below was generated on the claim that the pilot range had
been "read exactly once, with no prior inspection". That claim was wrong. Seeds
10000–10004 of the cell `n_people=4, tightness=0.2, overlap=0.2, uniform` had already
been run through C1 and their results inspected during an earlier cap-calibration pass,
before the freeze.

What follows from that, and what deliberately does not:
- The reserved pilot block stays `10000–19999`. Seeds `10000–10004` are **excluded from
  formal pilot analysis** in every cell, not only the exposed one: a per-cell exclusion
  would need a judgement about how much exposure counts, and the mechanical rule is the
  one that can be checked.
- The reason for the exclusion is **prior exposure, known in advance and independent of
  any LLM outcome**. Nothing was dropped because of how it scored; that distinction is
  what separates this from cherry-picking.
- The grid, the qualifying rule, the first-N-in-ascending-order selection and the subset
  algorithm are **unchanged**. Only the starting seed moves.
- The exposure is narrow — 1 of 30 subset instances, 1 of 48 cells, 5 of 240 pool
  instances — but the reclassification applies to the whole set: a subset containing an
  exposed instance is no longer a pre-registered selection.

**Superseded manifest (calibration-exposed audit artifact, kept, not deleted).**
`binning__pilot__67d1c874261f.json` and `subset__pilot__d29b54752971.json` remain in the
repository as the audit trail of the error and as the provenance of the C1 calibration
sweep that ran on them (§2). They are not used for any formal analysis.

**FORMAL PILOT MANIFEST — generated 2026-08-05, seeds 10005–19999.**

| | |
|---|---|
| Binning manifest | `results/manifests/binning__pilot__e6a365481dd4.json` |
| Content hash | `e6a365481dd4a485867c62377ac5bc6dc58b277113ccd207605a0258d5cba306` |
| Rerun (`--dry-run`) | reproduced the same hash |
| Seeds consumed | 10005–10009; 240 considered, 240 accepted |
| `not_proven` / `optimum_zero` | 0 / 0; no technical failure criterion triggered |
| Boundaries | easy ≤ 0, medium ≤ 6 |
| Level counts | easy 128, medium 63, hard 49 |
| LLM subset | `results/manifests/subset__pilot__29d836cb5614.json` |
| Content hash | `29d836cb56141f3a29ac3fa9670e662c123586e967f2d5ae1280925c2228edbd` |
| Composition | 10 easy / 10 medium / 10 hard, no duplicates, **no overlap** with the superseded subset |

**FORMAL PILOT RUN MATRIX — PRE-REGISTERED 2026-08-06, before the first run.** Written
down in full before any token was spent, so that what is reported afterwards can be
checked against what was planned rather than against what happened to finish.

| | |
|---|---|
| Code | commit `bc968bb` (`bc968bbb5d60043b52d64173115b8b261e0649bf`) |
| Subset | `subset__pilot__29d836cb5614.json`, hash `29d836cb56141f3a29ac3fa9670e662c123586e967f2d5ae1280925c2228edbd` |
| Binning manifest | `binning__pilot__e6a365481dd4.json`, hash `e6a365481dd4a485867c62377ac5bc6dc58b277113ccd207605a0258d5cba306` |
| Expected runs | `results/manifests/expected_runs__formal_pilot.json`, hash `cc2e97997004e57e2b065005b5b016b067aeb3ef5fe51bf5a8f29b6b0c8a663a` |
| Matrix | 3 conditions × 4 caps × 30 instances = **360 runs** |
| Conditions | `c1_react`, `c2_verify_revise`, `c3_mas` |
| Caps | 8 000 / 16 000 / 32 000 / 64 000 |
| Output | `results/logs/formal_pilot/`, one document per (condition × cap × instance) |

*Serving configuration, identical on both endpoints.* vLLM `0.19.1` (torch 2.10.0+cu128),
`Qwen/Qwen3-32B-AWQ` at revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`,
`--tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 16
--gpu-memory-utilization 0.90 --seed 42 --no-enable-prefix-caching`, awq_marlin kernel.
GPU 0 serves port 8000 (shard 0), GPU 1 serves port 8002 (shard 1). Client side:
`--max-model-len 32768`, `--request-timeout 1800`. Prefix caching **off** is what makes
`n = 1` per cell defensible (§1); the two servers were verified to produce bit-identical
output on the same cell, which is what allows sharding across them at all.

*Completion criterion, and the rule that follows from it.* Analysis of the formal pilot
begins **only after all 360 expected runs exist and pass the manifest audit**: every run
id in `expected_runs__formal_pilot.json` present, each document verified against the
subset and binning hashes, and no partial condition. A condition that is only half
finished is not analysed, not compared, and not previewed -- looking at half of C2 beside
a complete C1 would turn an interruption into a selection effect. The sweep is resumable
precisely so that an interruption costs time and nothing else.

*Coverage limitations of the formal subset, recorded rather than repaired.* Repairing any
of these would mean choosing the selection after seeing its composition, which is exactly
what the pre-registration forbids.
- **The hard level contains no `n_people=4` instances** (strata: 6-clustered ×3,
  6-uniform ×3, 8-clustered ×2, 8-uniform ×2). With this pool's boundary (hard > 6) no
  four-person instance qualifies as hard, so the round-robin has no such stratum to draw
  from. Level and `n_people` are therefore partially confounded at the top level.
- **Level is nearly determined by `tightness`**: easy is entirely 0.2, hard entirely 1.0,
  medium a mix of 0.85 and 1.0. The categorical level and that generator knob cannot be
  separated within this design.
- **Oracle optimum falls with level**: mean 5.5 (easy), 3.3 (medium), 2.5 (hard). Since
  satisfaction is `achieved / optimum`, a single meeting is worth 0.40 on hard and 0.17
  on easy. See the construct-validity item in §6.

**Superseded manifest — generated 2026-08-04, seeds 10000–19999.**

| | |
|---|---|
| Binning manifest | `results/manifests/binning__pilot__67d1c874261f.json` |
| Binning content hash | `67d1c874261f59eb3c4a040113562e7693514235a20375a5341adebec72ce302` |
| Rerun (`--dry-run`) | reproduced the same hash exactly |
| Seeds considered / accepted | 240 / 240 |
| `not_proven` / `optimum_zero` | 0 / 0 |
| Technical failure criteria | none triggered; no incomplete cell |
| Boundaries | easy ≤ 0, medium ≤ 5 |
| Level counts | easy 123, medium 62, hard 55 |
| LLM subset | `results/manifests/subset__pilot__d29b54752971.json` |
| Subset content hash | `d29b547529713e54dffbf3bc902f0007cbd3590d4a121b929fcea1b053a8bcd8` |
| Subset composition | 10 easy / 10 medium / 10 hard |
| Subset representation | every level carries `n_people` 4, 6 and 8 and both `uniform` and `clustered` |

Every cell closed on seeds 10000–10004, so the range was consumed at its minimum -- which
is also how the exposed seeds entered every cell. The measured complexity ran from 0 (123
instances) up to 28, the complete conflict graph at n = 8.

*On seed reuse across cells.* The subset draws mostly seed 10000, and for the hard level
seeds 10000–10002. That is a property of the deterministic round-robin, not a defect: a
stratum spans eight cells (four `tightness` values × two `overlap` values), and the
tie-break `(seed, cell_key)` therefore exhausts every cell's seed 10000 before reaching
10001. **The same seed in different cells is not the same instance.** Instance identity is
the full generator parameter set together with the seed, and the RNG path itself diverges
before the people are drawn: the travel matrix is built first, consuming a number of draws
that depends on `n_people` and on the travel structure, so the window draws that follow
start from a different RNG state. Variety across `n_people`, travel structure, `tightness`
and `overlap` — the dimensions the design cares about — is complete in every level.
Recorded here so it is not later mistaken for an error. The subset algorithm is not
revisited after the LLM runs begin.

**FORMAL PILOT RESULTS — recorded 2026-08-07, after the completeness audit passed.**
*Partly superseded the same day: the analysis that produced the numbers below carried four
methodological defects, and the block that follows this one states what replaced them. The
original is kept rather than edited into shape, because the correction is part of the
record.*
All 360 pre-registered runs exist; the set matches `expected_runs__formal_pilot.json`
exactly, carries one subset hash, one binning hash and one repository state (`916434ac`),
and no run is duplicated. Analyser `scripts/analyse_formal_pilot.py`; artifacts in
`results/analysis/formal_pilot/`, raw documents in `results/formal_pilot/`. Rerunning the
analyser reproduces all three artifacts byte for byte.

| condition | cap | satisfaction | tokens | sat/1k | proposal validity | runs with ≥1 valid |
|---|---|---|---|---|---|---|
| C1 | 8 000 | 0.000 | 7 674 | 0.000 | — | 0.00 |
| C1 | 16 000 | 0.067 | 15 551 | 0.004 | 0.400 | 0.07 |
| C1 | 32 000 | 0.250 | 28 549 | 0.009 | 0.348 | 0.27 |
| C1 | 64 000 | 0.436 | 38 934 | 0.011 | 0.372 | 0.47 |
| C2 | 8 000 | 0.000 | 7 674 | 0.000 | — | 0.00 |
| C2 | 16 000 | 0.067 | 15 747 | 0.004 | 0.400 | 0.07 |
| C2 | 32 000 | 0.306 | 30 767 | 0.010 | 0.441 | 0.33 |
| C2 | 64 000 | 0.556 | 49 213 | 0.011 | 0.484 | 0.60 |
| C3 | 8 000 | 0.000 | 5 755 | 0.000 | — | 0.00 |
| C3 | 16 000 | 0.000 | 11 702 | 0.000 | 0.000 | 0.00 |
| C3 | 32 000 | 0.489 | 24 453 | 0.020 | 0.900 | 0.63 |
| C3 | 64 000 | 0.789 | 34 420 | 0.023 | 0.843 | 0.97 |

*Paired comparisons* (same instance, same cap; sign test over non-tied pairs):

| contrast | cap | mean Δ | better / worse / tied | p |
|---|---|---|---|---|
| C2 − C1 | 32 000 | +0.056 | 2 / 0 / 28 | 0.500 |
| C2 − C1 | 64 000 | +0.119 | 5 / 1 / 24 | 0.219 |
| C3 − C1 | 16 000 | −0.067 | 0 / 2 / 28 | 0.500 |
| C3 − C1 | 32 000 | +0.239 | 12 / 2 / 16 | 0.013 |
| C3 − C1 | 64 000 | +0.353 | 17 / 3 / 10 | 0.0026 |
| C3 − C2 | 32 000 | +0.183 | 11 / 4 / 15 | 0.118 |
| C3 − C2 | 64 000 | +0.233 | 15 / 3 / 12 | 0.0075 |

**DECIDED — multiple comparisons, and an admission about when the decision was made.**
The pre-registration fixed the matrix, the completion criterion and the metrics, but it
did **not** name a primary contrast and did not fix a correction. That gap is recorded
rather than repaired: designating C3 − C1 as primary now, after the results are visible,
would be choosing the test after seeing its outcome. The family is therefore taken as all
twelve cells of the table above (three contrasts × four caps), each reported with its raw
p, and Bonferroni is applied at 0.05/12 = 0.0042 as the confirmatory threshold. **Only
C3 − C1 at 64 000 tokens (p = 0.0026) survives it.** C3 − C2 at 64 000 (0.0075) and
C3 − C1 at 32 000 (0.013) are reported as suggestive and not as confirmed. The per-level
breakdowns are exploratory throughout. The next pre-registration must name its primary
contrast in advance.

**Mechanism, predicted before the run and confirmed by it.** A worker reasons over a
sub-instance, so the chain of travel constraints it must respect is shorter; the
prediction was that proposal validity would rise and that the `travel_infeasible` share
specifically would fall. At 64 000 tokens: C1 makes 43 attempts at validity 0.372 with 23
`travel_infeasible` (0.53 per attempt); C3 makes 83 attempts at validity 0.843 with 11
(0.13 per attempt). The MAS advantage is not that it proposes more, but that what it
proposes is feasible.

**The threshold in these data is on the budget, not on complexity.** Below 32 000 tokens
C3 is not merely level with C1 but behind it (0.000 against 0.067 at 16 000), and it does
not consume its grant: 5 755 of 8 000, 11 702 of 16 000, with 18 of 30 runs at the lowest
cap ending on the agent's own `finish` without a single proposal. The fixed worker quota
(3/8·W each) leaves each worker below the budget it needs to produce anything, so the
decomposition costs without paying. Above 32 000 the ordering inverts and C3 leads on
satisfaction, on proposal validity, and on tokens spent at once: at 64 000 it reaches
0.789 while spending 34 420 tokens against C1's 38 934, roughly doubling satisfaction per
thousand tokens.

**The directional hypothesis about complexity is not supported.** The research question
asks whether there is a complexity threshold above which the hierarchy wins. The measured
association runs the other way: Spearman(complexity, Δ satisfaction) for C3 − C1 is −0.179
at 32 000 and −0.219 at 64 000, and by level at 64 000 the paired advantage is +0.460
(easy), +0.457 (medium), +0.142 (hard). It attenuates on the absolute scale as well —
mean additional meetings +2.6, +1.6, +0.4 — so it is not an artifact of dividing by a
falling optimum. The recorded confounds still apply: the hard level has mean optimum 2.5,
which makes satisfaction coarse there and caps how large any absolute gap could be, it
contains no `n_people=4` instances, and level is nearly determined by `tightness`. The
direction is nevertheless consistent across both scales and both working caps, and is
reported as the pilot's finding rather than set aside.

**C2 buys accuracy with tokens and not with efficiency.** At 64 000 it lifts satisfaction
from 0.436 to 0.556 while spending 49 213 tokens against 38 934, a 26% increase, leaving
satisfaction per thousand tokens unchanged at 0.011; no contrast reaches significance. On
the lower caps the comparison does not test the treatment at all: C1 and C2 spend an
identical number of tokens in 30 of 30 instances at 8 000, 28 of 30 at 16 000 and 21 of 30
at 32 000, falling to 4 of 30 at 64 000, because the draft phase exhausts the budget
before verify/revise can run. "C2 did not help" is only a statement about the top cap.

**POST-PILOT METHODOLOGICAL CORRECTION — 2026-08-07, after external review of the first
analysis.** No run was repeated and no raw document was altered: every number below comes
from re-analysing the same 360 files. `scripts/analyse_formal_pilot.py` was rewritten and
the artifacts regenerated. Four defects in the analysis above were identified in review;
each is stated with what replaced it.

1. **The sign of Δ was inverted relative to the expose.** The expose defines
   `Δ = S_SA − S_MAS`, so Δ > 0 means the single agent leads and the crossover is a `+ → −`
   change as complexity rises. The first analysis reported `treatment − baseline` and
   labelled it Δ, which would have inverted H1 and H2 wherever the text quoted it.
   Contrasts are now named `delta_<a>_minus_<b>` so the direction cannot be inferred from
   a word like "better".
2. **The inference the expose specifies was missing.** Paired bootstrap confidence
   intervals per cap and per level, and a test of the `system × complexity` interaction,
   were promised in the expose and absent. Both are now computed, with the pairing
   preserved: the bootstrap resamples instances so a draw carries a whole paired
   difference, and the interaction permutes level labels while every difference stays
   intact. The sign test and Spearman are retained as secondary.
3. **C3's proposal validity was inflated by pooling the critic with the workers.** Of C3's
   134 proposal attempts, 48 carry `role = critic`, and the critic emits only candidates
   that already passed the aggregator gate — at cap 64 000 its validity is 29 of 29.
   Pooling them reported 0.843 where the workers alone reach 0.759. The rates are now
   split by role.
4. **C2's treatment exposure was inferred from a proxy that was wrong.** The first
   analysis argued from identical token totals that verify/revise had not run. Identical
   totals are consistent with that and do not demonstrate it, and the call log answers it
   outright: verify runs in 0/30 runs at 8 000, 2/30 at 16 000, 9/30 at 32 000 and **25/30
   at 64 000**. So the treatment does run at the top cap and still produces no
   interval-supported gain, which is a stronger statement than the proxy could support.

*Corrected primary contrast, `Δ = S_C1 − S_C3` (negative = the hierarchy scored higher):*

| cap | Δ | 95% CI (paired bootstrap) | CI excludes 0 | omnibus p | pre-registered decreasing-trend p | observed direction |
|---|---|---|---|---|---|---|
| 8 000 | 0.000 | [0.000, 0.000] | no | 1.000 | 1.000 | flat |
| 16 000 | +0.067 | [+0.000, +0.167] | no | 0.309 | 0.665 | flat |
| 32 000 | −0.239 | [−0.403, −0.075] | **yes** | 0.478 | 0.835 | Δ rises with complexity |
| 64 000 | −0.353 | [−0.526, −0.173] | **yes** | 0.297 | 0.935 | Δ rises with complexity |

Per level, the interval excludes zero on easy at 64 000 (−0.460, [−0.767, −0.133]) and on
medium at both working caps; **it excludes zero on hard at no cap**.

*On the direction of the trend test.* H1 and H2 together register a **falling** Δ — positive
on easy, negative on hard. The Jonckheere-Terpstra column above is the one-sided
lower-tail p for exactly that alternative, and it is far from significance because the
point estimates move the other way. The two-sided p (0.346 at 32 000, 0.136 at 64 000) and
the one-sided p for the opposite, rising direction (0.171 and 0.068) are carried in
`summary.json` and are **exploratory**: they concern an alternative nobody registered, and
quoting either as a test of H1/H2 would evaluate the hypothesis against its own
contradiction.

**REVISED — what may and may not be claimed about complexity.** The earlier block states
that the directional hypothesis "is not supported" and that the association "runs the
other way". That was stronger than the evidence. The interaction is tested three ways —
Kruskal-Wallis omnibus across all three levels, Jonckheere-Terpstra for a monotone
ordering, and the easy-versus-hard contrast — and none reaches significance at any cap.
The point estimates lean against H1/H2, but **at ten instances per level these tests have
power only against a large effect, so a null result leaves the question open rather than
answering it.** The defensible statement is that the pilot does not resolve whether the
condition difference depends on complexity. A non-significant interaction is not a finding
of no interaction, and the thesis text must not read as though it were.

**The low-complexity sanity check fails, and that conclusion does not depend on the
interaction tests.** The expose specifies that on the easiest level the single agent
should be at least on a par with the hierarchy, and that a violation points at the
measurement design, the setup or the evaluation protocol rather than at a finding. On easy
instances at cap 64 000 the hierarchy leads by 0.460 with an interval that excludes zero.
That is a statement about the easy level alone and stands however the interaction question
is eventually resolved. It is the pilot's principal diagnostic result and the reason the
complexity operationalisation must be settled before the held-out main experiment runs.

**Secondary metrics the expose asks for, now extracted from the same documents.**
Feasibility is 1.000 in all 360 runs, which is a property of the harness rather than a
result: the finaliser emits whatever survives validation and an empty plan is vacuously
feasible. The informative counterpart is the share of runs holding a non-empty plan.
Optimality rate, which the first analysis reported only pooled, is strongly cap-dependent:
C1 0.000 / 0.067 / 0.233 / 0.367 and C3 0.000 / 0.000 / 0.300 / 0.533 across the four caps,
so the architectures differ more in partial plan quality than in how often they land
exactly on the optimum. Latency is recorded because the expose asks for it and is **not**
usable as an efficiency comparison: it is wall-clock on a shared machine with thermal
throttling active throughout and one GPU released partway through the sweep.

**C3 stage reach, measured rather than inferred.** Workers run in 30/30 runs at every cap,
but produce zero proposals at 8 000 and one at 16 000, so the aggregator has nothing to
select from and the critic never runs (0/30 at both caps, 19/30 at 32 000, 29/30 at
64 000). The ledger's own `budget_exhausted` flag is False in all 120 C3 runs. Taken
together these describe a condition that stops early without spending its grant; naming
the worker quota as the cause remains an interpretation, and §6 records it as a decision
still to be taken rather than as an established mechanism.

**Provenance, stated precisely.** The pre-registration names `bc968bb` as the frozen code;
every run records `916434a` as the executing commit. The difference is additive
`--write-expected-runs` plumbing that returns before the first run, so the execution path
is identical — verifiable with
`git diff bc968bb 916434a -- scripts/run_pilot_sweep.py`. Both commits are now carried in
the analysis artifacts and in the report rather than collapsed into one line.

**REVISED COMPLEXITY AXIS FOR THE MAIN EXPERIMENT — DECIDED 2026-08-08, before any
candidate instance is generated.** This closes the gating item opened by the pilot: the
easy/medium/hard axis used there does not order planning difficulty (see the post-pilot
correction above and the construct-validity item in §6). The revision is recorded in full
before the structural calibration runs, so that what is reported afterwards can be checked
against what was planned.

### The decision in one line

Task size stops being the complexity axis. The main experiment fixes `n_people = 8` and
varies **normalised conflict density** alone, with the oracle optimum used as a matching
variable rather than as part of the definition.

### Primary structural axis

    D = (binding conflicting pairs) / C(n, 2)          with n = 8, so C(n,2) = 28

A binding conflicting pair is a pair of people each individually reachable, for which no
ordering admits both; the definition is already implemented as `conflict_graph` in
`src/oracle/solver.py`, and pairs with an individually unreachable endpoint are excluded
there as isolated infeasibility rather than conflict. Normalising by the number of possible
pairs removes the mechanical growth of the raw count with `n` — the defect that made the
pilot's raw-count axis uninterpretable across sizes. Levels are Low / Medium / High `D`,
cut on the empirical distribution of the candidate pool rather than on thresholds chosen in
advance.

*A caveat this denominator carries.* `C(8,2) = 28` is constant, so at fixed `n` the
normalisation is a rescaling of the raw count. But the conflict graph is built only over
individually reachable people, so an instance where three of eight are unreachable can
carry at most `C(5,2) = 10` edges and can never reach a high band. The collector therefore
records `individually_reachable_count` and **both** denominators — `D` over `C(8,2)` as
defined above, and `D_reachable` over `C(reachable, 2)` as a diagnostic — so the amendment
can see whether the two disagree before the bands are fixed.

### Why `n` is fixed, and why at 8

Holding `n` constant removes several confounds at once rather than modelling them: the
number of possible pairs is identical across levels, prompt length does not grow along the
complexity axis, the two C3 workers always split the same number of people, and the
question "does a fixed cap mean less reasoning budget at larger n" does not arise. The
research question defines complexity through interacting constraints; `n_people` is a
generator parameter, not the construct.

`n = 8` rather than 9 for two reasons. It is inside the oracle's proven-tractable range —
`RECOMMENDED_MAX_N = 9`, and the single-worker CP-SAT proof closes in about 8 s at worst
there (§ generator/solver comments) — so **the oracle is not modified and the determinism
record from §1 stands unchanged**. And an even number splits symmetrically for two
workers, where 9 would hand one worker an extra person under any near-equal split.

`n = 10` and `n = 12`, proposed and rejected: past about ten people the visit-count proof
blows up on clustered medium-window instances, so the matrix would have required either a
much larger time limit, a multi-worker solver (which changes the frozen oracle
configuration and the determinism story), or discarding unproven instances — and that last
option selects on solver difficulty, which is precisely the confound being removed.
Instance size returns as a **secondary robustness analysis** at `n = 6` and possibly `n = 9`
inside the proven range, on a smaller sample. No claim in the thesis depends on its outcome.

### Matching variable: the oracle optimum

The pilot's central defect was that the optimum fell with the nominal level (5.5 / 3.3 /
2.5), so the denominator of the primary metric shrank along the axis it was supposed to
index, and a highly constrained instance became arithmetically easy. With `n` fixed there
is no need for a ratio: the **absolute oracle optimum `O` is matched across the D bands**.
The target is bands that share a comparable `O` distribution — ideally the same modal value
— so that size, denominator and reward scale are held constant and only the conflict
structure differs. `O` is a matching and admissibility variable. It is **not** part of the
definition of complexity.

### Structural diagnostics recorded for every candidate

Computed on CPU from the generator, the solver and the conflict graph. At `n = 8` every one
of these is exhaustive over 256 subsets, so no approximation is involved.

| Field | Meaning |
|---|---|
| `conflicting_pairs`, `conflict_density_D` | the axis itself |
| `solver_optimum` (`O`), `proven_optimal` | ground truth and whether it is proven |
| `alpha_reachable` | independence number of the conflict graph **induced on the individually reachable people only** — an unreachable person has no edges and would otherwise inflate it |
| `higher_order_gap_H` = `alpha_reachable − O` | interaction the pairwise metric cannot see: a triple jointly infeasible with no infeasible pair contributes to `H` and nothing to `D` |
| `max_degree`, `edge_share_top1`, `edge_share_top2` | conflict concentration — whether the conflicts sit on one or two people |
| `largest_component` | whether the conflict structure is connected or fragmented |
| `n_max_independent_sets` | how many distinct maximum-size pairwise-compatible sets exist, i.e. how non-obvious the choice is |
| `triangle_inequality_violations` | see below |

`H` is a **diagnostic, not part of the complexity definition**. Folding it into a composite
score after seeing the pilot would rebuild the same problem one level up.

*On the sign of `H`.* The inequality `O ≤ alpha_reachable` follows from feasible sets being
closed under deletion, and that holds only if travel times satisfy the triangle inequality:
dropping an intermediate meeting sends the route directly from the previous stop to the
next, which is slower when `travel(A,C) > travel(A,B) + travel(B,C)`. The generator has four
travel structures, and its own specification divides them: `UNIFORM`, `CLUSTERED` and `LINE`
are position-derived, while **`RANDOM` is an arbitrary asymmetric matrix with no triangle
inequality** by construction.

Measured on a first calibration sample (288 instances at `n = 8`, 504 ordered triples each),
that division holds but is not absolute: `RANDOM` violates on about 67 triples per instance,
while the position-derived structures still violate on roughly 1–4. They are metric in
continuous coordinates but their travel times are **rounded to integer minutes**, and
rounding breaks the inequality on rare triples. So `H ≥ 0` is *expected* rather than
guaranteed even there. In that sample `H ≥ 0` held everywhere, with `H` ranging up to 4 —
higher-order interaction is common and invisible to `D`, which is exactly why it is
recorded.

The violation rate is therefore measured and reported per structure. A negative `H` is
**not** clipped or corrected: it records that feasibility is not monotone under deletion,
which is itself a source of planning difficulty, and whether `RANDOM` belongs in the final
bands is part of the diversity decision below.

*On conflict concentration.* A star-shaped conflict graph centred on one person has high
`D` and a trivial solution — decline to meet that person. It is not caught by `H`, which is
near zero there, but by high `D` together with high `alpha_reachable`. Conflict
concentration is therefore a **pre-registered structural diagnostic**: no final `D` band may
consist predominantly of hub- or star-like instances. The exact threshold is deliberately
**not** fixed here — any number chosen before seeing the distribution would be arbitrary. It
is fixed during the structural calibration, from generator properties only, and before any
LLM run.

### Generator-parameter diversity, as a checkable rule

The pilot's level was nearly determined by `tightness`. To prevent the same confound
returning measured rather than assumed:

> For every final `D` band, at least two distinct `tightness` values and at least two
> distinct travel structures must be represented, and no single `tightness` value may
> account for more than 60 % of the band.

Requiring all four travel structures in every band is deliberately **not** imposed — it would
make the sampling artificial.

### Selection is structural only

The held-out set is selected from the generator, CP-SAT, the brute-force checker where
applicable, the validator and descriptive statistics. **No LLM is run at any point in
dataset construction.** Generating many instances, running C1, and keeping those where C1
degrades smoothly would tune the complexity axis to observed model behaviour and make the
whole design circular. This rule is absolute.

### Separation of seed ranges, and the freeze between them

Structural calibration and held-out sampling are separated. Without that separation the
procedure would read: inspect the structural properties of the future held-out pool,
choose the bands to fit them, then call the set held out. That is not leakage of LLM
results and would not invalidate anything, but the stages are cleanly separable, so they
are separated.

> **Numeric Low/Medium/High `D` bands, the conflict-concentration criterion, the
> optimum-matching rule and every admissibility threshold are determined exclusively on a
> dedicated structural-calibration seed range. They are then frozen in a dated amendment to
> this section before any instance from the held-out main seed range is selected or
> inspected. If the frozen rules fail to populate the held-out matrix, the pre-registered
> fallback is triggered; the bands are not re-tuned on the held-out pool.**

This is what makes the fallback a real fallback rather than an invitation to calibrate
again, and it is the reason no numeric `D` boundary and no concentration threshold appear
above: they cannot honestly be written before the calibration, and they must not be
writable after the held-out pool has been seen.

| Range | Purpose | Status |
|---|---|---|
| 0 – 9 999 | development and smoke | consumed |
| 10 005 – 19 999 | formal pilot | consumed, frozen |
| 20 000 – 29 999 | structural calibration for the revised axis | allocated 2026-08-08 |
| 30 000 – 39 999 | budget-cap dev sample for the main experiment | allocated 2026-08-08 |
| 100 000 – 199 999 | held-out main experiment | untouched, not to be inspected before the freeze |

### Phases, in order

1. **Structural calibration.** A large CPU-only candidate pool at `n = 8` across the full
   range of `tightness`, `overlap`, travel structure and seeds, on a seed range distinct
   from both the pilot's and the held-out main's.
2. **Diagnostics.** Distributions of `D`, `O`, `H`, concentration and the generator
   parameters; `D` against `O`; `D` against each knob; cell counts for every planned band.
3. **Admission criteria fixed**, in writing, before the held-out set is touched: final `D`
   bands, the `O` matching rule, the concentration threshold, the diversity rule above,
   exclusions, and instances per band.
4. **Budget-cap calibration on a separate dev sample.** The pilot calibrated the ladder over
   `n = 4…8`; the working rungs were 32k and 64k. Caps for the main experiment are chosen on
   a dev sample from its own seed range — never on the held-out instances, which would burn
   them.
5. **Freeze the held-out manifest** with fixed hashes and seeds, reproducibly.
6. **Pre-register the main experiment**, and this time name in advance what the pilot's
   pre-registration omitted: the **primary contrast** and the **multiplicity policy**,
   alongside the primary test, the caps, the complexity definition, the model configuration
   and the manifest.
7. **Run.** Only then.

### Scale

Three `D` levels × 50 held-out instances × 3 conditions × 2 caps = **900 runs**, against
3 600 for the rejected `n × D` matrix. At the pilot's observed throughput this is roughly
four to five days on two A6000s. It also puts **50 paired C1/C3 observations per level and
cap** behind the interaction analysis, against the ten that left the pilot unable to resolve
it in either direction.

"At least 50 instances per band" means **50 unique held-out instances per band, each solved
by all three conditions at both caps** — the same 50 Low-`D` instances go to C1, C2 and C3,
and to each cap, because the analysis is paired within instance. The main set is therefore
150 instances in total, not 150 per condition. Instances drawn for the structural
calibration or for the budget-cap dev sample are **not** among these 150 and never enter the
main analysis.

### Fallback, recorded before the calibration that would trigger it

If the candidate pool cannot deliver well-separated `D` bands, comparable optimum
distributions and generator-parameter diversity **at the same time** — the likely failure
being that high `D` mechanically destroys the optimum — then the bands are **not** adjusted
until the picture looks right. Instead:

> `D` and oracle attainability are analysed as **separate structural dimensions**, rather
> than forcing a one-dimensional Low/Medium/High complexity scale that does not exist in
> the instance space.

### The sanity check is retained, and its status is bounded

The low-complexity check stays: on the easiest band the single agent should be at least on
a par with the hierarchy. But its meaning changes now that the specific confounds it
exposed have been removed.

> **The revised low-complexity sanity check is diagnostic, not a requirement that C1 must
> win. Once the revised axis has passed its pre-registered structural admission criteria, an
> observed C3 advantage at Low `D` is retained as an experimental result rather than used to
> redesign the axis again.**

Without that bound the design admits an endless loop: C3 wins on easy, the easy level is
redefined, C3 wins again, and the axis is quietly tuned until the hypothesis survives. If a
C3 advantage at Low `D` persists on an axis that passed its admission criteria, exactly two
readings remain — an unknown design problem, or H1 simply being false for this task and
this model — and separating them is the job of the architecture audit and the
full-information control, not of another complexity axis.

**AMENDMENT — COMPLEXITY BANDS FROZEN, 2026-08-09.** The structural calibration this
section required is complete and the three bands are fixed. The fallback recorded above is
**not** activated: the registered design turned out to be reachable.

### How this was reached, including the branch that failed

Recorded in order, because the failed branch is part of the evidence and not an
embarrassment to be tidied away.

1. A stricter design than anything registered was proposed: fix the stratum to
   `O = 4` and `H = 0`, with bands at 1–4 / 7–10 / 13–16.
2. It was checked and returned **NOT ADMISSIBLE** (`b2c3d22`). Two of three bands held 9
   and 24 candidates against a required 50, dominant tightness reached 0.67 and 0.79
   against the 60% ceiling, and an identical composition across bands had capacity 7.
3. Review established that this proposal was stricter than the registered requirement.
   Section 3 asked for separated `D` levels, **comparable** `O` distributions and generator
   diversity, and registered `H` as a **diagnostic**. Exact matching on `(O, H)` was
   invented during calibration. Its failure therefore rejected that proposal and said
   nothing about whether the registered design was reachable — a distinction the first
   reading of the result missed.
4. The registered phrase "comparable `O` distribution" was then formalised **before** any
   search ran: it means an identical empirical `O` histogram, and every requirement must
   hold jointly on the selected sample rather than separately on raw bands.
5. An exhaustive deterministic search over all 593 775 band triples was specified and
   committed before it was run (`c393124`), then made fast without changing what it
   decides (`67bb7de`, held to a brute-force reference by test).
6. It returned **FEASIBLE** (`858aee4`), reproduced byte for byte from the committed state.
7. The design below was chosen by the fixed tie-break — largest minimum gap, then largest
   joint capacity, then lexicographic order — not by judgement.

### The frozen design

| Level | conflict pairs `k` | `D = k/28` |
|---|---|---|
| **Low pairwise conflict density** | 0–1 | 0.000 – 0.036 |
| **Medium pairwise conflict density** | 7 | 0.250 |
| **High pairwise conflict density** | 13–16 | 0.464 – 0.571 |

`n_people = 8` throughout. Minimum gap of five unused conflict counts between neighbouring
bands. Joint matched capacity 65 per band against a target of 50, with an **identical
oracle-optimum histogram** in all three: 41 instances at `O = 3` and 24 at `O = 4`.

*Provenance hashes (SHA-256).* Byte-identical reproduction from the committed state was
verified before this amendment was written; these are the long-term identifiers of what the
decision was taken against.

| Artifact | SHA-256 |
|---|---|
| `results/calibration/structural/candidates.csv` | `01adfc30cdf6c2742292e199aebfb354dcffbf160660b308844575def9560735` |
| `results/calibration/structural/pool_meta.json` | `0d79d2f28cd6a00bea66cc4548c1c4a7bc784c3d22918d725b9215c36156f0e6` |
| `results/analysis/band_search/summary.json` | `1a51736f3518fae490e15d239cbf2f0d9633375f195cdf2aea56abfbd4b34f8e` |
| `results/analysis/band_search/report.md` | `f365625b845651b70b92e8989033b6013eccfbff7a95301ffab44c8e3a815f26` |

The levels are named **pairwise conflict density**, never "task complexity" unqualified.
That wording is binding on the thesis text for the reason in the next paragraph.

### Binding requirements, frozen

`n = 8`; the three bands above; an identical empirical `O` histogram across bands; at least
two tightness values per band; no tightness above 60% of a band; at least two travel
structures per band. **Boundaries may not be changed after any held-out instance has been
seen.** If the held-out candidate pool cannot populate this design, the response is the
fallback recorded above, not an adjustment of the bands.

### `H` is systematically unbalanced, and that is a recorded limitation

> The selected bands isolate large differences in pairwise binding-conflict density while
> permitting exactly matched oracle-optimum distributions. They do **not** isolate all forms
> of structural interaction. In particular the pre-registered diagnostic
> `H = alpha_reachable − O` varies systematically across the selected bands: in the low band
> it spreads to 5 with 1 263 of 5 572 candidates at `H ≥ 3`, while in the high band 284 of
> 456 sit at `H = 0`. The primary manipulation is therefore interpreted specifically as
> **pairwise conflict density**, not as an exhaustive scalar measure of task complexity. `H`
> is retained as a pre-specified diagnostic and sensitivity variable and is **not** used to
> redesign the bands.

What this forbids, stated now so it cannot be rationalised later: if the main experiment
returns an unexpected result, excluding high-`H` instances, rebalancing the low band on `H`,
or re-running the search are all **post-hoc redesign** and are not available. The only
permitted response is the sensitivity analysis declared in advance — how much the
architecture effect moves once `H` is accounted for — together with an honest statement of
limited identifiability if `D` and `H` prove too closely tied.

### `overlap`

Reported, never binding, and excluded from the tie-break by construction. Its distribution
differs across the selected bands. This is described as a **residual generator-parameter
imbalance and a possible alternative structural explanation**, not as a confound: `overlap`
is one of the mechanisms through which `D` arises, so some difference is expected. It would
become a confound only if it affected the architectures beyond what `D` already captures,
and nothing here establishes that.

### What this amendment does NOT freeze

The calibration proved that a design of this shape **exists**. It did not prove that the
untouched seed range 100 000+ will populate it. Still open, and all of it must be fixed
before any seed in that range is inspected:

- **The exact `O` histogram of the 50 instances per band.** Calibration gives a common
  capacity of 41 at `O = 3` and 24 at `O = 4`; scaling that to 50 by largest remainder
  would give 32 and 18, but **that number is not adopted here**. Choosing it after seeing
  the held-out pool would be a free parameter, so a deterministic rule must be fixed first.
- The held-out sampling algorithm, in full and machine-checkable form: strata, tie-breaks,
  allowable imbalance, minimum per stratum.
- A power analysis for 50 per band under this design, rather than a number carried over.
- The statistical pre-registration: whether one cap or both are primary; the primary
  architecture contrast, almost certainly C1 versus C3; how `Architecture × D-level` is
  tested; what formally counts as a **crossover** rather than a monotone trend; the
  multiplicity policy; whether C2 is primary or secondary; and the pre-declared `H`
  sensitivity analysis.

No held-out LLM run may start before those are recorded.

**AMENDMENT — LOWER-COMPLEXITY EXTENSION FROZEN, 2026-08-16.** Thirty-six development
instances at `n = 4, 5, 6`, twelve per size, every one at oracle optimum 3. Selected from
three structural pools, verified, and frozen **before any agent has been run on any of
them**. This is an *extension*: the frozen `n = 8` design of 2026-08-09 is untouched, no
`n = 8` instance is re-selected, and no new `n = 8` subsample is created.

### Why the extension exists

The development results put the frozen Low band within reach of no single-agent condition
at `n = 8`, so the registered scale had no region in which a single trajectory is expected
to do well. Reading the pool showed why, and it is not what the band's name suggests. The
Low band is Low in **pairwise conflict density** — 0 or 1 binding pair out of 28 — but its
selected instances carry the largest higher-order gap anywhere in the pool: for `k <= 1`
and `O in {3,4}` at `n = 8`, `H = alpha_reachable - O` is 3 in 108 candidates, 4 in 361 and
5 in 22, a mean of **3.83**, with all eight people individually reachable and a mean of 0.0
unreachable. The floor of the registered scale is therefore easy for the metric that
defines it and hard for the interaction that metric cannot see. `check_joint_support.py`
had already recorded the direction of this effect; what is new is that the optimum matching
selects for its extreme.

Lowering the conflict density further is not possible: `k = 0` is the floor and the Low band
already sits on it. The extension moves the quantity that actually varies there.

### Why `O = 3`, everywhere

`satisfaction = achieved / O`. One optimum at every size makes the denominator **identical
rather than comparable**, so satisfaction takes the same four values {0, 1/3, 2/3, 1}
throughout and a difference between sizes cannot arise from the metric's granularity. `O = 3`
also has large support at every size (1836 / 1662 / 1647 candidates at `n = 4 / 5 / 6`) and
is already present in the frozen `n = 8` design, where it accounts for exactly 12 of the 20
instances in each band.

`O = 4` was measured and rejected, on the pool and not on taste. At `n = 4` that cell is
`O = n`: the best possible plan meets everyone, 5123 of its 5126 candidates have zero
conflicting pairs, 5123 have zero higher-order gap, and it is 53.4% of the whole `n = 4`
pool. A histogram over `{3, 4}` scales to 6 and 6 at twelve per size, so half of the `n = 4`
sample would have sat in a cell degenerate on three dimensions at once. `O = 3` is therefore
not "the best optimum" but the cleanest common matching value.

### What smaller `n` is used for, and what it is not

**`n` remains a generator parameter and does not become a complexity metric.** The design
does not read "n = 4 is Low, n = 6 is High", and it must not be described that way. What
smaller `n` does is arithmetic:

    H = alpha_reachable - O  <=  n - O

so at a fixed optimum a smaller size **bounds the higher-order interaction that the pairwise
density metric cannot see**: the ceiling is 1, 2 and 3 at `n = 4, 5, 6` against 5 observed at
`n = 8`. That is the whole mechanism, and it is a ceiling rather than a guarantee.

*Stated as a limitation, not worked around.* `n` and the attainable `H` are tied together by
that inequality. The separate causal effect of task size and of higher-order interaction is
**not identified** by this design, and no analysis of these 36 instances may claim to
separate them.

*And the sample means are not monotone.* Over the selected twelve, mean `H` is 0.92 at
`n = 4`, **1.58 at `n = 5` and 1.42 at `n = 6`** — `n = 5` above `n = 6`. Only the ceiling is
ordered by arithmetic; the mean is a property of twelve instances and is reported as
measured rather than presented as a ladder. All three sit far below the frozen `n = 8` Low
band's 3.83, which is the comparison the extension exists to make.

### The first selection was rejected, before any agent ran

The first run of the selector returned **11 clustered and 1 line at every size**. It was
discarded and its manifests deleted while still untracked. The cause is the selector, not
the data: one instance per seed, a canonical order that sorts the travel-structure name
alphabetically, `clustered` first in that order, and an `O = 3` clustered candidate available
for all 100 seeds — so the cheapest choice for every seed was the same topology. The pools
themselves are balanced; at `n = 4, O = 3` they hold clustered 370, line 458, random 484 and
uniform 524. At `n = 8` the band restriction removed most seeds' in-band clustered candidate,
which is why the effect never appeared there.

This rejection used **no agent information of any kind**. It is a structural defect visible
in the manifest alone, and it was found and fixed before the first LLM call on any of these
instances.

### The travel-structure rule, and why it is an equal split

The frozen `n = 8` composition was read first, to continue it if it offered a template. Its
`O = 3` entries do not: by band they are clustered 5 / 3 / 9, line 2 / 7 / 1, random 4 / 2 / 2
and uniform 1 / 0 / 0, so `uniform` is absent from two bands of three and the pooled shape
17 / 10 / 8 / 1 is itself dominated by one topology. With no natural composition to carry
forward, the pre-agreed fallback applies:

> **Exactly 3 clustered, 3 line, 3 random and 3 uniform at every size.** Identical across
> sizes, so that a change in `n` is not accompanied by a change in travel topology.

A 60% share ceiling was considered and rejected as insufficient: it still permits 7 of one
topology against 5 of another. The exact histogram supersedes the older "at least two travel
structures" rule and is strictly stronger; that older rule stays in force, unchanged, in the
`n = 8` selector, which this work does not touch.

### Everything else is the `n = 8` procedure, unchanged

At least two tightness values per size; no tightness value above 60% of a size; one selected
instance per seed; proven optima only; the earliest admissible selection under the canonical
order seed → travel structure → tightness → overlap, chosen by CP-SAT with one worker and
`random_seed = 0`. No agent, plan, score, token count or termination reason enters the
selector, and a regression test enforces that against its source.

Level labels are re-derived from the conflict count through boundaries that are the frozen
`n = 8` cut-points expressed as densities and read back at each size — `easy` is `D <= 1/28`,
`medium` is `D <= 7/28` — which returns exactly 1 and 7 at `n = 8`. **A level is therefore a
statement about conflict density and never about size**, and a dense `n = 5` instance can and
does outrank a sparse `n = 6` one.

### The frozen sample

| | `n = 4` | `n = 5` | `n = 6` |
|---|---|---|---|
| instances | 12 | 12 | 12 |
| optimum | all 3 | all 3 | all 3 |
| distinct seeds | 12 | 12 | 12 |
| seed range | 40000–40011 | 50000–50011 | 60000–60011 |
| `k` min / max / mean | 0 / 1 / 0.08 | 0 / 4 / 0.67 | 0 / 8 / 2.83 |
| `D` min / max | 0.000 / 0.167 | 0.000 / 0.400 | 0.000 / 0.533 |
| `H` histogram | {0: 1, 1: 11} | {0: 2, 1: 1, 2: 9} | {0: 3, 1: 2, 2: 6, 3: 1} |
| `H` min / max / mean | 0 / 1 / 0.92 | 0 / 2 / 1.58 | 0 / 3 / 1.42 |
| tightness | 0.6 ×6, 0.8 ×6 | 0.6 ×3, 0.8 ×7, 0.9 ×2 | 0.8 ×7, 0.9 ×5 |
| travel structure | 3 / 3 / 3 / 3 | 3 / 3 / 3 / 3 | 3 / 3 / 3 / 3 |
| all individually reachable | 12 / 12 | 12 / 12 | 12 / 12 |
| levels easy / medium / hard | 11 / 1 / 0 | 9 / 1 / 2 | 1 / 7 / 4 |

*A thin conflict axis at `n = 4`, recorded because it constrains what may be read from that
size.* Eleven of its twelve instances have `k = 0`, so `n = 4` contributes essentially
nothing to the density axis. It is a low-interaction sample, which is its purpose, and it is
not evidence about conflict density.

### Provenance

Structural pools collected at commit `e5b6342`, 9600 candidates each on the same knob grid
the `n = 8` pools used — 100 seeds × 6 tightness × 4 overlap × 4 travel structures — one
reserved seed range per size, every pool passing the collector's completeness audit with
zero unproven optima.

| artifact | SHA-256 |
|---|---|
| `subset__lowcx_n4__74e57136280d.json` | `74e57136280d6271991bbbca93ad5188adb2dead7a2e98d10df8b3f92aeacfe4` |
| `binning__lowcx_n4__9c553e394e79.json` | `9c553e394e79838f349dbb7b03e024ee01ca28bcb5e8daf3463e1a08bcf95a43` |
| `results/calibration/lowcx_n4/candidates.csv` | `1601348e620de74b76253b1b1306bfd2ed8e2b4abb6b706f472a76a07c0ee6d6` |
| `subset__lowcx_n5__6bd65b0db5bf.json` | `6bd65b0db5bfdd9555874a3a36d83348438a35798ebb0ead1d182fe005d16ac9` |
| `binning__lowcx_n5__65d7ebc329f2.json` | `65d7ebc329f2c7bb87645663523c8d55dc1b44b43d5e8f5e0c984ac267d97028` |
| `results/calibration/lowcx_n5/candidates.csv` | `206913b78a62653b2e09218ae7aa2a84eba47111c8907bb7155e35a42be54ad0` |
| `subset__lowcx_n6__f3efd0cd0b73.json` | `f3efd0cd0b7310ca70c4dac27b7e3e1a2f419ed894cb2b82a7138bcb5caa9931` |
| `binning__lowcx_n6__eb01223da4a0.json` | `eb01223da4a02fa5fcc1c47f4ac29f6dd5b357c5bb36d84855142f10937027b2` |
| `results/calibration/lowcx_n6/candidates.csv` | `06d28d880bbcd405242cc0bb6c3571fb8ccbc4828ce7ff780e608b8dd72a5d0c` |

The selector and its 39 tests were committed at `0eae288`, **before** this selection was
run; the version that produced the rejected 11-clustered sample was committed at `d47829b`,
also before it ran. Every manifest re-derives every requirement from its own documents and
passed that check at write time.

*A naming correction, recorded because the table above was written before it.* The selector
first named each binning document after the **subset's** hash rather than its own, which
left it undiscoverable: the sweep runner locates a binning document by globbing for the hash
the subset points at. The three files were renamed to carry their own hashes, as the `n = 8`
pair already does. This is a rename and nothing else -- the diff is zero insertions and zero
deletions, every content hash reproduces, and **no instance, seed, optimum or manifest
content changed**. The selector was corrected and two regression tests added: one that every
document is named after its own hash, one that performs the runner's own discovery glob.

### Freeze

The 36 instances are fixed. No instance may be added, removed or exchanged after any agent
has been run on them; the optimum may not be changed; and a future LLM result is not grounds
for re-selecting. Held-out seeds 100 000+ remain untouched and were not read at any point.
No condition has been run on any of these instances as of this amendment.

**AMENDMENT — LOWER-COMPLEXITY CALIBRATION, RESULT, 2026-08-17.** 216 runs: the frozen 36
instances at `n = 4, 5, 6`, C1 and C3, caps 8 000 / 16 000 / 32 000. Completeness audit
passed — 216 of 216 verified, nothing missing, nothing incompatible, no stray file. The
frozen 36 are unchanged and no instance was added, removed or exchanged.

*Standing.* **Exploratory development calibration, descriptive only.** No hypothesis test,
no threshold, no cap selection, no gate. Twelve instances per cell support description and
ordering, not inference. Held-out seeds 100 000+ remain untouched.

### The question, and the answer

The calibration asked whether the extended lower range contains a region in which the single
agent performs at least as well as the hierarchy at a small budget. **No such region was
found on the frozen lower-complexity development set at `n = 4–6` across the caps examined,
8 000 / 16 000 / 32 000.** The sign follows the exposé — `delta = satisfaction(C1) -
satisfaction(C3)`, negative means the hierarchy is ahead — and the delta is `<= 0` in **every
one of the nine cells** and at every size and cap when pooled.

*The scope of that statement is the set and the caps that were run, and no wider.* `n = 4` at
64 000 was not tested, and `n = 4` at 32 000 is the first cell in which C1 becomes
substantially functional at all, so whether a budget-dependent low-complexity regime exists
is **not** settled by this calibration. It is the subject of the registered follow-up below.

| cap | C1 | C3 | delta | C1 ahead | C3 ahead | tied |
|---|---|---|---|---|---|---|
| 8 000 | 0.000 | 0.000 | +0.000 | 0 | 0 | 36 |
| 16 000 | 0.028 | 0.130 | −0.102 | 1 | 13 | 22 |
| 32 000 | 0.259 | 0.509 | −0.250 | 5 | 18 | 13 |

| `n` | C1 | C3 | delta | C1 ahead | C3 ahead | tied |
|---|---|---|---|---|---|---|
| 4 | 0.194 | 0.306 | −0.111 | 4 | 12 | 20 |
| 5 | 0.093 | 0.241 | −0.148 | 2 | 12 | 22 |
| 6 | 0.000 | 0.093 | −0.093 | 0 | 7 | 29 |

### The extension did what it was built to do — and it was not enough

C1 improves monotonically as the size falls: 0.000 at `n = 6`, 0.093 at `n = 5`, 0.194 at
`n = 4`, against 0.000 in every cell of the frozen `n = 8` Low band's neighbourhood at these
caps. Tasks in this range are reachable for a single agent in a way the registered Low band's
were not, which is exactly the property the extension was constructed to obtain.

**C3 improves alongside it** — 0.093, 0.241, 0.306 over the same sizes — so the ordering
never changes. The gap itself is not ordered (−0.093, −0.148, −0.111 at `n = 6, 5, 4`), so no
trend in the gap is claimed either.

*A negative result within its scope, not an inconclusive one.* The lower edge at which a
single trajectory was expected to be competitive was looked for on a purpose-built axis, at
three budgets, on 36 instances matched at one optimum, and was not found **there**. That is a
statement about this development set at these caps; it is not a statement that no such regime
exists at any budget.

### Cap 8 000 gave no discriminative signal between C1 and C3

The zero at the bottom cap is not a tie in the ordinary sense: **both architectures scored
satisfaction 0 on all 36 paired instances**, with no validated non-empty plan in any of the
72 runs — 0 of 12 in each of the six cells. The cap is not uninformative. It locates a
**floor common to both architectures** at this instance size, and that is real information;
what it does not do is separate them.

The two failures are not the same failure:

| | mean tokens | cap used | termination |
|---|---|---|---|
| C1 at 8 000 | ~7 750 | **0.97** | aborted 26, budget 10 |
| C3 at 8 000 | ~5 740 | **0.72** | aborted 13, agent_finish 23 |

C1 exhausts the cap. C3 does **not**: each worker receives `floor(3/8 * (8000 - 256)) =
2904` tokens, finishes inside them, and returns no product anyway. **The C3 result at
8 000 is consistent with quota starvation rather than with anything about task difficulty,
and this design cannot separate the two** — attempt depth and architecture vary together at
that cap, exactly as they did in the C5 diagnostic. The alternative was recorded before the
sweep was launched, not after seeing the number.

### C1 fails by aborting, not by running out of budget

Across all 108 C1 runs the dominant termination is `aborted`, not `budget`: 26 of 36 at
8 000, 24 at 16 000, 20 at 32 000. At `n = 6`, cap 32 000, it is 9 of 12 while cap
utilisation is 0.97. C1 reached a validated non-empty plan in **12 of 108 runs**; C3 in 41 of
108. Both figures are the secondary diagnostic and neither is described as performance.

That C1 aborts rather than exhausts is recorded as an observation. No cause is asserted for
it here: this calibration varied size and budget, not anything that would isolate why a
trajectory terminates that way.

### What this does and does not license

The low-complexity sanity check of §3 has now returned a hierarchy advantage on a **second**
axis: first on the registered Low `D` band at `n = 8`, and now on an axis built specifically
to reach further down by bounding higher-order interaction at a matched optimum. The
2026-08-09 amendment fixed in advance what may follow from that, and it binds here:

> an observed C3 advantage at Low `D` is retained as an experimental result rather than used
> to redesign the axis again.

So the axis is **not** redesigned a third time. Exactly two readings remain open — an unknown
problem in the design, or H1 being false for this task, this model and this budget range —
and separating them is the job of the architecture audit and the full-information control,
not of another complexity axis. No new development pool is built in response to this result.

*And what stays open.* Nothing here selects a cap, and the failed C1 budget gate remains
failed. Cap 8 000 is not shown to be too small in general; it is shown to place both
architectures on a common floor on this pool. The **budget** direction is not exhausted
either: `n = 4` at 64 000 is untested, and the follow-up registered below runs it on the same
frozen instances. Whether the held-out design keeps three conditions, which caps it uses, and
how `Architecture × complexity` is tested are separate decisions, still unmade.

### One analyser defect, found and corrected before any conclusion was drawn

The first run of the report gave `delta +0.000` with twelve ties for every size, while the
C1 and C3 means in the same rows plainly differed. The pairing keyed on `instance_id` alone,
so pooling over caps let one instance's three records overwrite one another and the survivor
decided the contrast; filenames sort `cap16000 < cap32000 < cap8000`, so the survivor was
cap 8 000, where every run scored zero. The key is now `(instance_id, cap)` and a regression
test requires one instance at three caps to count as three paired observations. The per-cell
and by-cap tables were never affected, since a single-cap filter already made the instance
unique. Recorded because the corrected by-size numbers are the ones above.

### Provenance

Model `Qwen/Qwen3-32B-AWQ`, checkpoint revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`,
vLLM with `enable_prefix_caching=False` confirmed in the server log, KV cache 98 928 tokens —
identical to the formal pilot — `awq_marlin` kernel, one model on one GPU, `max_model_len`
32 768, `--max-num-seqs 16`, server seed 42. Results in `results/logs/lowcx_calib`, kept
separate from every earlier development and probe directory; the audit confirms no other
sweep wrote there. Expected-runs manifests `expected_runs__lowcx_n4/n5/n6.json`
(`9a03d5709386`, `14ed104b967a`, `e23b9dc6f753`), 72 runs each. Analysis in
`results/analysis/lowcx_calib`. Selector and analyser were committed before they were run.

**PRE-REGISTRATION — 64 000 FOLLOW-UP AT `n = 4`, 2026-08-17.** Written and committed
**before the first LLM call of this probe**, so that what is reported afterwards can be
checked against what was planned.

### What is run

The already-frozen twelve `n = 4` instances — `subset__lowcx_n4__74e57136280d.json`, all at
oracle optimum 3 — at cap **64 000**, conditions **C1 and C3 only**. Twelve instances × two
conditions × one cap = **24 runs**.

Not a new dataset and not a new axis: this is the **budget** direction continued on the same
frozen sample. No instance is selected, generated, added, removed or exchanged. `n = 5` and
`n = 6` are not run at this cap, and C2, C4 and C5 are not run at all.

### Why

> After the frozen 8 000 / 16 000 / 32 000 lower-complexity calibration, no tested cell had
> mean C1 satisfaction equal to or above C3. However, `n = 4` at 32 000 was the first cell in
> which C1 became substantially functional — 0.500 satisfaction against 0.722 for C3. A final
> development follow-up therefore reuses exactly the already-frozen twelve `n = 4` instances
> at cap 64 000 for C1 and C3 only. No selection or architecture is changed. This checks
> whether the low-complexity hierarchy advantage observed at 32 000 is budget-dependent. **No
> new lower-complexity pool will be constructed in response to its result.**

The specific opening this addresses: at 32 000 C1 spent a mean of 25 581 tokens of 32 000 and
did not terminate cleanly everywhere — 4 of 12 runs aborted and 1 hit the budget. A cap that
still binds on some runs cannot rule out that the ordering at that cap is partly a budget
effect.

### Held fixed, and not to be touched for this probe

The frozen `n = 4` subset and its selection; the generator; the CP-SAT solver; the hidden
validator; every prompt; the C1 and C3 implementations; token accounting; the finalisation
node; the model `Qwen/Qwen3-32B-AWQ`; its checkpoint revision
`0499c3ac83fdef8810b907a23894ba91e95eddd8`; the sampling configuration; `max_model_len`
32 768; prefix caching **off**; and the seed policy.

Results go to `results/logs/lowcx_n4_64k_probe`, kept separate from
`results/logs/lowcx_calib`. The 216 documents already there stay byte-for-byte unchanged.

### Standing, and the interpretation rule — fixed now, not after the numbers

Exploratory development follow-up. **Descriptive.** Twelve paired instances support
description and ordering, not inference. No p-value, no threshold, no gate and no cap
selection follows from it, and no new numerical criterion may be introduced afterwards.

Two outcomes are possible and both are accepted as they come:

- **mean C1 at 64 000 `>=` mean C3 at 64 000.** A candidate budget-dependent low-complexity
  regime has been observed. This is **not** a demonstrated statistical crossover — twelve
  development instances cannot establish one — and it may not be described as such. What it
  licenses is that such a regime is taken into account when the held-out design is fixed.
- **mean C1 at 64 000 `<` mean C3 at 64 000.** C3 keeps the descriptive ordering even after
  the cap is doubled on the lowest frozen structural region. The lower-complexity calibration
  is then closed, and no easier development pool is built in response.

Whether a crossover will or will not appear is deliberately **not** predicted here.

### The paired comparison to be reported

Against the same twelve instances at 32 000, already on disk and not re-run: mean
satisfaction, validated non-empty rate, mean tokens and mean calls for each condition at each
cap; instances where C1 leads, where C3 leads and where they tie at 64 000; paired
`delta = satisfaction(C1) - satisfaction(C3)`; improved / worsened / unchanged from 32 000 to
64 000 separately for each condition; validated non-empty plans gained and lost; termination
reasons; proposal validity; and the counts of runs spending more than 32 000 tokens and
32 000 or fewer.

*Why that last pair of counts matters.* Per call the agent is granted
`min(remaining − reserve − input, max_model_len − input)`, and the 32 768 window clamps every
call at both caps. A larger cap therefore buys **more calls, not longer ones**, and a run
that spends 32 000 or fewer tokens at the higher cap was never constrained by the lower one.
Such a run reproduces its own trajectory and carries **no information** about the effect of
the extra budget. The informative subset is the runs that exceed 32 000, and the report must
state its size rather than averaging over twelve as though all twelve were informative — the
error the 128 000 probe already caught once at `n = 8`, where only 4 of 12 C1 runs diverged.

### Expected runs, frozen before the sweep

`results/manifests/expected_runs__lowcx_n4_64k_probe.json` — 24 run ids, 12 instances, cap
64 000 only, conditions `c1_react` and `c3_mas` only, pointing at subset
`74e57136280d…` and its binning document `9c553e394e79…`. Analysis may begin only once every
id in it has a verified result on disk.

**AMENDMENT — 64 000 FOLLOW-UP AT `n = 4`, RESULT, 2026-08-17.** The 24 registered runs are
complete and audited: 24 of 24 verified, nothing missing, nothing incompatible, no stray
file. The frozen twelve `n = 4` instances are unchanged and nothing was re-selected.

*Standing.* Exploratory development follow-up, descriptive. Twelve paired instances support
description and ordering, not inference. No threshold, no gate, no cap selection.

### The registered case that obtained

| | C1 @32 000 | C3 @32 000 | C1 @64 000 | C3 @64 000 |
|---|---|---|---|---|
| mean satisfaction | 0.500 | 0.722 | **0.583** | **0.917** |
| validated non-empty | 7/12 | 12/12 | 8/12 | 12/12 |
| mean tokens | 25 581 | 22 546 | 29 158 | 26 708 |
| cap used | 0.80 | 0.70 | **0.46** | **0.42** |
| mean calls | 13.0 | 16.5 | 13.8 | 17.2 |
| proposal validity | 0.571 | 0.968 | 0.739 | 0.917 |
| termination | aborted 4, finish 7, budget 1 | finish 12 | **finish 11, budget 1** | finish 12 |

`delta = satisfaction(C1) - satisfaction(C3)` moves from **−0.222 to −0.333**: doubling the
cap widened the gap rather than closing it. At 64 000, C1 leads on 1 instance, C3 on 6, and 5
are tied.

The pre-registration of 2026-08-17 named this outcome `c1_below_c3` and fixed what follows
before any of these runs existed:

> C3 keeps the descriptive ordering even after the cap is doubled on the lowest frozen
> structural region. The lower-complexity calibration is then closed, and no easier
> development pool is built in response.

**That is the reading taken.** The lower-complexity calibration is closed.

### Neither architecture exhausted the 64 000 total cap

Cap utilisation is 0.46 for C1 and 0.42 for C3, and C1's runs finish voluntarily — 11 of 12
`agent_finish`, one on budget. Both conditions leave more than half the total budget unspent.

**That does not make the cap behaviourally irrelevant, and it may not be reported as if it
did.** C3's internal worker quota is itself derived from the total cap — `3/8 * (cap -
reserve)`, 11 904 at 32 000 against 23 904 at 64 000 — so C3 can spend only 26 708 tokens at
the higher cap and still have executed differently because each worker was permitted more
in advance. Low utilisation of the total is therefore not evidence that the extra budget was
inert.

The budget's effect is consequently assessed by **paired divergence in execution**, and
exceeding 32 000 total tokens is reported only as a supporting diagnostic. On that measure
5 of 12 C1 runs and 6 of 12 C3 runs were altered by the higher cap, which is what the
following section reports.

*What the follow-up does establish.* **The single agent's poor performance at 32 000 cannot
be attributed solely to its total token cap.** At 64 000 the same twelve instances leave C1
with more than half its budget unspent while its satisfaction moves only 0.500 → 0.583.

The cap did account for part of what was happening, and that part is not dismissed here: the
extra budget removed every abort, improved one instance and added one validated plan. What it
did not do is close the quality gap. So the claim is about how much the total cap explains,
not that it explains nothing — and it remains a statement about **C1's total budget**, not
about the cap's role in either architecture's behaviour, which the worker quota above already
shows cannot be read off total utilisation.

### C1 stopped aborting, and did not start planning better

At 32 000 C1 aborted on 4 of 12 runs; at 64 000 on **none**. The abort behaviour was
budget-related and the extra cap removed it. Satisfaction moved 0.500 → 0.583 and validated
non-empty plans 7/12 → 8/12 — one instance gained, none lost, ten unchanged.

*Two distinct deficits, and only one of them is a budget deficit.* C1 now terminates cleanly
and still produces the same plans. Proposal validity rose 0.571 → 0.739, so its proposals
improved while its final scores barely moved. No cause is asserted for the remaining gap:
this follow-up varied the budget and nothing else.

C3 was already at 12/12 validated non-empty at both caps; its gain is in plan quality,
0.722 → 0.917, with 4 instances improved and none worsened.

### How many runs actually carry information, and a correction to how that was counted

| condition | execution changed | reproduced exactly | exceeded 32 000 |
|---|---|---|---|
| C1 | 5/12 | 7 | 5 |
| C3 | 6/12 | 6 | 0 |

*The correction.* The analyser first counted only runs whose total exceeded 32 000 and so
reported **0 of 12 informative runs for C3** — while C3's mean spend had risen from 22 546 to
26 708 and only 6 of 12 totals reproduced. Both cannot be true, and the criterion was the
part that was wrong. Exceeding the lower cap is *sufficient* evidence that a run was
constrained by it but **not necessary**, and it undercounts a condition that subdivides its
budget: C1 spends one undivided working pool and diverges only when it would have run out,
while C3's worker quota is `3/8 * (cap - reserve)` — 11 904 against 23 904 — so a C3 worker
behaves differently long before any total approaches 32 000. The primary count is now a
changed token total, which is direct evidence that the cap altered the run. Caught and fixed
before any claim rested on it; a regression test now requires a run that changed without
crossing the lower cap to be counted.

With the corrected count, roughly half the runs of each condition were altered by the extra
budget — 5 of 12 and 6 of 12 — so neither mean above rests on twelve independent looks at the
effect of the cap, and both are reported with that count beside them.

### What the three looks together now say, and what they do not

The low-complexity sanity check of §3 has now returned a hierarchy advantage three times: on
the registered Low `D` band at `n = 8`; on the extended axis at `n = 4–6` across 8 000 /
16 000 / 32 000; and here at `n = 4` with the cap doubled to 64 000, where neither condition
exhausted its total budget. The permitted readings are the two fixed on 2026-08-09 and they
are unchanged — an unknown problem in the design, or H1 being false for this task, this model
and this budget range. Separating them is the job of the architecture audit and the
full-information control (C3-FI), not of another complexity axis or another development pool.

### The result of the lower-complexity calibration, stated once

> **Within the tested model, task family and budget range, no low-complexity regime was
> observed in which the single ReAct agent outperformed the hierarchical MAS on mean
> satisfaction. Reducing structural interaction made C1 viable, but C3 improved alongside it
> and retained the higher mean satisfaction.**

Both halves of that sentence are load-bearing. The extension was not wasted work: at `n = 6`
C1 scored 0.000 at every `n = 6` cap that was tested — 8 000, 16 000 and 32 000, the size
having no 64 000 arm — and at `n = 4` with 64 000 it reaches 0.583 with 8 of 12 validated
plans and no aborts, a single agent that functions where the registered axis gave nothing. And the ordering never turned: `+0.000` at the common 8 000 floor, then −0.102 and
−0.250 pooled at 16 000 and 32 000, then −0.333 at `n = 4` and 64 000, the widest of them.

**The calibration is closed here, and the following are not available as responses to this
result:** an `n = 3` pool; any other easier development pool; a change to the matched
optimum; a 128 000 probe at `n = 4`; and a fourth complexity axis. Continuing to build easier
development sets after this sequence would not be calibration — it would be a search for the
point at which C1 is obliged to win, and the 2026-08-09 amendment forbids exactly that.

*What is still open, and is not touched by this result.* The failed C1 budget gate remains
failed. No cap is selected. The held-out design — which conditions, which caps, how
`Architecture × complexity` is tested, and the multiplicity policy — is unmade, and this
follow-up neither fixes nor constrains it beyond what is written above.

### Provenance

Model `Qwen/Qwen3-32B-AWQ`, revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`,
`enable_prefix_caching=False` confirmed in the server log, KV cache 98 928 tokens,
`awq_marlin` kernel, `max_model_len` 32 768, one model on one GPU, server seed 42 — identical
to the formal pilot and to the 8 000 / 16 000 / 32 000 calibration. Runs in
`results/logs/lowcx_n4_64k_probe`, kept separate from `results/logs/lowcx_calib`, whose 216
documents are unchanged. Expected runs `expected_runs__lowcx_n4_64k_probe.json`, content hash
`1f4e17fe767d`. Analysis in `results/analysis/lowcx_n4_64k_probe`. The analyser was committed
at `f70b06a` **while the 24 runs were still executing**, so its tables could not have been
shaped by them.

---

## 4. Conditions (C1–C5)

**C1 — ReAct baseline:**
- One agent, one context; loop Thought → Tool call → Observation.
- May stop early when it emits a final plan; no mandated verification stage.
- The budget guard forces a stop only when the budget is exhausted.

**C2 — ReAct + verify/revise:**
- One agent, one context; fixed structure
  Draft → Verify → Revise → Verify → Revise → Stop (**max 2 revisions**).
- Stops after 2 revisions or on budget exhaustion, whichever comes first.
- Isolates the effect of mandated verification vs. C1.

**C3 — hierarchical MAS:**
- Five roles: supervisor, worker A, worker B, aggregator, critic. The LLM agents
  are the two workers and the critic; the supervisor and the aggregator are
  deterministic zero-token nodes (C3-2 / C3-3 below).
- **Fixed** splitting strategy: the supervisor always splits people into 2
  geographic clusters (deterministic, not prompt-dependent; exact algorithm C3-1).
- All agents share one budget ledger; inter-agent messages count toward the budget.
- Two workers are used to keep coordination overhead minimal while preserving a
  meaningful decomposition; more workers exhaust the budget at the tighter caps.
- Isolates the effect of decomposition vs. C2.

**C4 — planner + fresh-context critic (SPECIFIED 2026-08-12; design E):**
- One planner on the **full** `n`-person instance, one fresh-context LLM critic, **no
  decomposition**. The critic is an LLM, **not** CP-SAT — the solver is never shown to
  the agents.
- *Why the literal "C3 with one worker" is rejected.* With one worker the candidate pool
  is exactly the people of that worker's best plan, which is also the fallback, so the
  gate's strictly-longer conjunct is **unsatisfiable by construction**: no subset of the
  pool can contain more people than the pool. C4 would be C1 plus a wasted call. In C3
  the pool is the *union* of two sub-plans while the fallback is the *better single* one,
  and that gap is what the critic exists to exploit.
- *Candidate pool (design E).* People appearing in at least one **parsed** planner
  proposal **and** whose availability the planner actually acquired through
  `get_availability`. Validity never filters pool membership: a pool built from
  validator-approved proposals only would encode the hidden verdict in the critic's input.
- *Evidence rule.* A deterministic harness-side cache holds exactly what the planner's
  tool calls returned — ids from `list_people`, a person's location and window only after
  `get_availability`, one ordered travel entry per `get_travel_time`. **A proposal
  discloses nothing.** Naming a person the planner never queried must not hand the critic
  that person's facts, because C1 and C2 have to buy every task fact through the tools.
- *Critic input.* Task rules, the pool people with only their cached facts, only the
  cached travel entries, required-but-unqueried pairs shown explicitly as `unknown`, and
  the deterministic draft. No tools, no optimum, no validator verdict, no full-instance
  dump, no planner transcript.
- *Gate, draft, fallback and failure semantics are C3's, unchanged*: subset of the pool
  ∧ full-instance validator ∧ strictly longer than the fallback; the draft is the raw
  union of the planner's proposals in one fixed order; the fallback is the planner's
  `best_plan_so_far`; every failure mode is a no-op with no retry; one finalisation.
- *Budget.* `W = cap − reserve`, planner quota `floor(3/4 · W)`, the critic takes what
  remains under the ordinary global guard. This matches C3's maximum pre-critic search
  allowance (two workers at `3/8 W` each) and its structural guarantee of `W/4` to the
  critic.

**Interpretation rules for C4, binding on the thesis text (2026-08-12).** Both correct
wording used in earlier discussion and are recorded so the looser phrasings cannot
reappear.

1. C3's cross-cluster travel entries are **coordination information required to reconnect
   decomposed subproblems**, not a "subsidy" and not automatically an experimental defect.
   Partitioning an instance removes those entries from the workers' sub-matrices, so no
   worker *could* have acquired them; supplying them is part of C3's coordination
   interface. The quantity is still measured and reported — how many entries the critic
   receives that no worker queried — but it is described neutrally.
2. **C2 → C4 is not a one-factor contrast.** The two also differ in the number and
   structure of reflection turns, in search-budget allocation, and in same-context
   iterative revision versus a single fresh critic call. The permitted claim is narrower:
   *C4 assesses whether a fresh independent critic provides additional value beyond
   same-context self-verification, without task decomposition.*
3. **C4 → C3 is not a pure decomposition effect.** The permitted claim is: *C4 vs C3 helps
   assess the contribution of decomposition plus its coordination interface after both
   architectures contain a fresh independent critic.*

**C5 — Validated Best-of-3 single-agent sampling (LOCKED 2026-07-26; added after the
literature audit, §8 / §7 L5):**
- **Why it exists.** Parmar et al. (2025) found plain Best-of-N the strongest PlanGEN
  variant on NATURAL PLAN. Without it, "C3 does not beat a single agent" would only ever be
  established against a *single-trajectory* agent, never against the cheapest competing use
  of the same tokens. C5 makes the sampling alternative an explicit, matched-budget column.
- **Structure.** Three independent C1 search trajectories share the instance-level budget;
  deterministic selection over their products; then ONE finalisation. Deliberately NOT
  described as "three full C1 runs at cap/3": a full C1 run includes its own finalisation,
  and here the inner trajectories have none.
- **Budget (mirrors C3-5, not a new discipline).** One instance-level reserve, exactly as
  everywhere else: `W = cap − reserve`, and each trajectory gets `quota_i = floor(W / 3)`.
  Quotas are **non-transferable**: unused quota does NOT pass to the next attempt, because
  that would make the result depend on attempt order (the same reason worker quotas do not
  transfer in C3). **Any remainder of the integer division (0–2 tokens) is left unused and is
  not assigned to any attempt** — handing it to the first or last attempt would reintroduce a
  small order dependence for no benefit. Trajectories do not finalise, so the reserve is
  charged once, not three times — charging three reserves would hand C5 a structural handicap
  at the 2k cap that has nothing to do with sampling.
- **Sampling diversity, and the seed split.** All decoding parameters stay byte-identical to
  every other condition (temperature 0.6 / top_p 0.95 / top_k 20 / min_p 0, thinking ON) so
  C5 gains no hidden decoding advantage. Only the **attempt seed varies**, as a pre-specified
  sequence `seed_i = BASE_SEED + attempt_index` (42 / 43 / 44), computed in one place and
  logged per call. **The finalisation call is explicitly excluded from that sequence:** it
  always uses a single fixed `FINALIZATION_SEED`, identical for C1–C5, so the terminal emit
  never inherits the winning attempt's seed. (After the faithful-serialisation fix the
  finalisation seed can no longer change the scored plan at all, but keeping it constant
  across conditions removes one needless source of variation.) Rationale for varying the
  attempt seed at all, stated carefully: vLLM's online serving does not *guarantee*
  bit-identical output for a repeated request (scheduling/batching affect reduction order —
  see the §7 confound), but a fixed request seed makes repeated attempts **likely identical or
  insufficiently diverse**, which would make Best-of-3 degenerate. Residual engine
  non-determinism must NEVER be relied on as the source of diversity; it is a confound, not
  a mechanism.
- **Selection (deterministic, zero tokens, oracle-free).** Each trajectory yields its own
  `best_plan_so_far` — a structured object already filtered by the in-loop validator gate.
  The selector compares those THREE OBJECTS, not three finalised texts: (1) discard
  candidates the full-instance validator rejects; (2) among valid ones take the most
  meetings; (3) break ties by the fixed rule (lowest attempt index); (4) if none is valid,
  take attempt 0 — the score is 0 either way. The solver optimum is NOT consulted at any
  point.
- **Why selection must precede finalisation.** Even with faithful serialisation enforced
  (§7: an emit is accepted only when it equals `best_plan_so_far`), finalising three attempts
  and then choosing among their outputs would spend three terminal emits instead of one and
  would make the selection depend on which emits happened to round-trip cleanly. Selecting
  among the structured plans first and finalising once keeps C5's terminal behaviour and its
  reserve usage identical to C1/C2/C3, and makes selection a pure function of the search
  results.
- **Logging.** `attempt_index` and the actual seed are recorded per call; the CallRecord role
  is `bon_attempt_{i}` for search calls and `finalize` for the single terminal emit, so the
  call log distinguishes trajectories without changing the shared core.
- **What C5 is not.** Not self-consistency (no answer voting), not a MAS (no inter-agent
  messages, no coordination), and not a verification condition (no critique step). It varies
  exactly one factor against C1: how the same budget is spent — one long trajectory versus
  three shorter independent ones with harness-side selection.

**All conditions share:** the same tools (`list_people`, `get_availability`,
`get_travel_time`), the same ledger and guard, the same paired instances, the same
Qwen3 tokenizer, the same decoding profile (§1) and the same instance-level finalisation
reserve. C5 is the only condition whose seed varies across internal attempts, and it varies
nothing else (§4 C5).

**Tool-calling & transport (LOCKED, implemented Layer 4 `src/core/llm_client.py`):**
- Tools are invoked via **ReAct text trajectory** (the model emits a `Thought / Action /
  Observation` text; LangGraph parses the `Action` line and runs the tool), **not** the
  endpoint's native function-calling. Reason: it preserves the exact equal-budget
  invariant -- we render the prompt ourselves (`render_input`) and send the raw string to
  **`/v1/completions`**, so the input count equals exactly what the model sees (no
  double-templating), and it is identical on vLLM and local Ollama. It also makes C1
  literally the ReAct baseline and extends uniformly to the C3/C4 agents. Tool docs live
  as text in the system prompt, so they are counted by `count_input` like any other input.
- Format-failure risk (a malformed `Action`) is handled by a robust parser now; on the
  vLLM main run the `Action` line is additionally constrained with `guided_regex` so the
  tool name/args are always valid -- removing format failure as a confound.
- The reasoning/answer split is done by **self-parsing at the last `</think>`** in the
  completion (on generated token ids when the endpoint returns them), not via vLLM's
  chat-mode `--reasoning-parser` (which we do not use, because we use `/v1/completions`
  for exact counting). Robust to a missing opening `<think>` tag, matching the official
  Qwen3 parsing recipe. One code path on every endpoint.
- Local debug runs against Ollama (`qwen3:8b`); the main run swaps `$VLLM_URL` to the
  vLLM endpoint with no code change. Ollama does not surface `<think>` via raw
  completions -- a debug-only limitation; the vLLM run produces it.

**C2 verify/revise (LOCKED, implemented `src/agents/single_agent/react_verify_revise.py`):**
- **Fixed structure, budget is the only early exit.** The cycle always runs
  Draft -> Verify -> Revise -> Verify -> Revise (`MAX_REVISIONS = 2`); there is no
  "plan looks good -> stop" branch. Such a branch would be prompt-dependent control flow
  whose trigger depends on the model and the condition, making token spend endogenous to
  behaviour. The verification overhead is therefore deterministic and identical on every
  instance -- a fixed cost, not a per-instance confound.
- **Expected, pre-specified before the pilot (a result, not a defect):** the verify/revise scaffold has a
  structural minimum token cost, so at low complexity x tight cap (e.g. easy x 2k) C2 can
  score BELOW C1 -- the scaffold consumes budget that C1 would have spent on search, or
  not spent at all (finishing early). Under equal budget this is not a confound; it is the
  threshold story itself ("verification costs budget that could have bought more search").
  `satisfaction/1k` and raw `satisfaction` will diverge here by design. Recorded in
  advance so it is not read as a post-hoc excuse.
- **Reflection-only revise (scoping decision + its boundary).** Verify and Revise make
  NO tool calls; they reason over the context the Draft already gathered, in the same
  single conversation. Rationale: under equal budget, tools-in-revise would not give C2
  *more* search (same cap) -- it would only spend the budget differently, while blurring
  the mechanism. Reflection-only yields a clean ladder C1 (act) -> C2 (act + reflect) and
  a one-sentence delta: "C2 - C1 = the effect of mandated reflection over a fixed
  context." The boundary this draws MUST condition the interpretation:
    1. revise operates only over Draft-gathered context;
    2. so it can fix reasoning errors over that context, but NOT missing-information
       errors -- a window violation caused by the Draft never calling the needed
       `get_availability` is unreachable, the data is not in context;
    3. therefore "C2 ~= C1" is interpretable only together with the invalid-error
       taxonomy (the `invalid breakdown` secondary metric, hand-classified at manual
       checkpoint #3): if most invalids are missing-information, reflection-only C2 cannot
       touch them by construction, and "verification did not help" then means something
       narrower than it sounds. This makes the invalid breakdown load-bearing, not
       secondary.
- **Scored output = best harness-validated candidate.** finalize serialises
  `best_plan_so_far` (the best VALID intermediate plan), never the last-emitted text;
  `best_plan_so_far` updates only to a strictly longer VALID plan, so a revise that
  proposes a worse or invalid plan cannot regress the score. Validity is never revealed to
  the agent (hidden bookkeeping). Same finalisation node and reserve as every other
  condition (section 2).
- **Reuse boundary.** C2 imports the shared `react_core` unchanged and adds only its two
  nodes + routers; the C1 path is byte-identical (`git diff` empty on
  `react_core.py`/`react.py`, full suite green). The verify critique reaches revise
  through the single shared conversation (revise sends the full history, which contains the
  critique turn; the revise prompt opens "Based on your review"), not a side channel -- so
  "revise ignores the critique" is structurally impossible, not merely unverified.

**C3 hierarchical MAS (design LOCKED in full, C3-1..C3-8):**

This block closes every open C3 design question and is frozen BEFORE any pilot or
main-run data exists. After this point the C3 architecture does not change in response
to results: the only value a pilot may still calibrate is the worker share `s`
(C3-5, [PILOT], frozen before the main run exactly like the reserve and the caps), and
the only pre-declared exception is the pilot-gated ablation slice C3-8. Main-run results
can trigger nothing here; a main-run-only anomaly goes to Limitations, not to a re-run.

Revision note (2026-07-16, pre-pilot, no runs affected): C3-3 and C3-4 were revised
after an internal design review found two violations of the condition-symmetry
invariants -- the simulate-and-drop merge was uncounted central planning, and the
critic's full-instance dump was privileged information relative to the tool-gated
C1/C2. The rejected variants and the reasons are recorded inside those blocks; the
freeze holds because no experiment had run.

- **C3-1 -- Topology and deterministic split.** Fixed workflow: supervisor -> worker A ->
  worker B -> aggregator -> critic -> finalise. Workers run SEQUENTIALLY (A then B; a
  determinism device only -- C3-5 makes the order budget-irrelevant) in ISOLATED
  conversations: neither worker sees the other's transcript, the full people list, or any
  cross-cluster data. No worker re-entry, no second critic turn, no dynamic graph changes.
  The split is computed from the travel matrix (the Instance carries no coordinates),
  people only (depot excluded), k = 2:
    1. symmetrise distances: d(x, y) = (travel[x][y] + travel[y][x]) / 2 over the
       people's locations;
    2. seeds = the pair of people whose locations maximise d; ties by the
       lexicographically smallest sorted (person_id, person_id) pair;
    3. labels: cluster A is seeded by the smaller person_id of the winning pair;
    4. assignment: every other person joins the seed nearer by d; distance ties -> A;
    5. n = 1: the single person forms cluster A and cluster B is empty. Empty-cluster
       rule: an empty cluster's worker is SKIPPED outright (zero calls, zero tokens),
       its sub-plan is the empty plan, and quotas never reallocate between workers (C3-5).
  A sub-instance is the parent instance restricted to one cluster: its people;
  locations = depot + their locations (parent order preserved); the travel matrix
  restricted to those locations; every scalar field (day bounds, depot, meeting duration,
  waiting_allowed, seed, generator_params, tie_break) copied verbatim. The partition is
  hard: each person occurs in exactly one sub-instance. Restriction changes no constraint
  on the remaining people, so a plan valid in a sub-instance is valid in the full
  instance -- the fallback (C3-3) relies on this and a test asserts it. Workers are the C1
  loop, byte-identical (same react_core nodes, same system-prompt template built by the
  same function, same three tools bound to the sub-instance, same §2 rules including the
  hidden best-plan bookkeeping, checked against the sub-instance). A worker's product is
  its best_plan_so_far (possibly empty); workers do NOT serialise -- the instance-level
  finalise runs once, at the end, identical to C1/C2 (C3-7).
- **Supervisor = deterministic control node, no LLM call (C3-2).** The supervisor is a
  deterministic coordination node, not an LLM agent: it applies the fixed geographic split
  (k=2), emits the two sub-instances, and enforces the empty-cluster rule. It issues no
  model calls. The hierarchy is preserved (a coordinator dispatches to workers), but
  coordination carries no decision content under a fixed split, so an LLM call there would
  add only overhead and MAST FC1/FC2 failure surface. The real LLM nodes of C3 are the two
  workers and the critic.
- **Aggregator = deterministic, zero-token, and NON-PLANNING: fallback + draft +
  candidate pool (C3-3; revised 2026-07-16 after design review).** The first locked
  version was a simulate-and-drop merge: three candidate orders, earliest-feasible
  re-timing, drop-on-infeasible, argmax by kept meetings, |merged| >= max(|A|, |B|)
  guaranteed by simulation. The review rejected it: that is a handcrafted planning
  heuristic performing uncounted central repair that C1/C2 never receive -- their
  best_plan is kept only when the MODEL itself emitted fully valid times -- so a C3
  gain could have been attributed to the mechanical machinery (re-timing, order
  search) rather than to decomposition, contradicting this block's own "zero
  reasoning" claim. The aggregator is therefore stripped to three moves that contain
  NO planning decision:
    1. **Fallback.** The better single worker sub-plan (ties -> worker A) enters the
       SAME hidden gate as any propose (full-instance validator + strictly longer)
       and becomes best_plan_so_far. Selection between two finished artifacts, not
       construction: C3's floor is the best half, max(|A|, |B|), by selection.
    2. **Draft.** The union of the two sub-plans in ONE fixed order -- sorted by
       worker-assigned start_time, ties by person_id -- with times UNCHANGED, nothing
       dropped, no simulation, no argmax. The draft may be INVALID on the full
       instance, deliberately: it is raw material handed to the critic, never a
       scoring candidate, and it never enters the gate. Every repair decision
       (re-order, re-time, drop) belongs to the critic's LLM call, where it is paid
       for in counted tokens.
    3. **Candidate pool.** The set of person_ids appearing in at least one sub-plan,
       packaged for the critic together with their windows/locations and the travel
       sub-matrix over depot + candidate locations (C3-4). This is the information
       the workers' tool calls already surfaced, plus exactly the cross-candidate
       travel entries that coordination needs; it is counted input like any message.
    * **No oracle access (hard invariant, all four conditions).** CP-SAT lives only
      harness-side as scorer/validator, never inside the agent system. The aggregator
      runs no solver and no heuristic over the union; it makes no feasibility
      judgement at all -- its only validator use is the shared hidden gate on the
      fallback, identical to the C1/C2 bookkeeping.
    * **Why deterministic and non-planning.** It keeps the definition of "best plan"
      byte-identical to C1/C2 (harness-side validation; no LLM re-rank and no
      harness-side construction), isolates the decomposition effect (all task
      reasoning happens in LLM nodes: search in the workers, coordination in the
      critic), and spends zero tokens. C3 therefore measures "split + LLM
      coordination of discovered candidates vs one agent", over a mechanical floor of
      the best single sub-plan.
- **C3-4 -- Critic = pool-restricted coordinator: placement, contract, NO revision
  round (revised 2026-07-16 after design review).** The first locked version handed the
  critic a FULL-instance dump (all people, all windows, the whole travel matrix) and
  invited it to look for people to add. The review rejected it: C1/C2 must buy every
  task fact through the three tools, so a free full dump turns C3 into a partially
  full-information condition; and a full-information critic adopted only when strictly
  longer is a central solo planner, not verification. The critic is now restricted to
  the workers' candidates:
    * **Placement (unchanged).** After the aggregator, before finalise, with a FRESH
      context: the workers' transcripts are not passed (context isolation is the
      decomposition mechanism, and re-encoding two full trajectories would spend the
      shared budget on transport). The critic runs iff the candidate pool is non-empty
      (C3-6) and the §2 guard admits its input. Exactly one call, thinking ON, no tools.
    * **Input = the candidate pool, NOT the instance.** A fixed system prompt (reviewer
      role + the same task rules the workers see) and one user turn holding (i) the pool
      people only -- person_id, location, window, in instance order; a person appears
      iff some worker put them in its sub-plan; (ii) the travel matrix restricted to
      depot + candidate locations; (iii) the draft, explicitly marked as possibly
      rule-breaking; (iv) the instruction. People no worker discovered stay invisible to
      the critic. No optimum, no validator verdicts, no complexity metric, no worker
      transcripts, no oracle vocabulary. This dump is C3's only inter-agent message
      besides the sub-instances, and it is counted input like any other.
    * **Powers.** Re-order, re-time, drop, and stitch across the two sub-plans -- full
      coordination authority over the candidates. It may NOT introduce a person_id
      outside the pool.
    * **Output contract.** The visible answer must contain exactly one line
      `Action: propose[p0@t0, p1@t1, ...]` carrying the FULL plan in visiting order.
      Parsed by the same parser as C1/C2.
    * **Hidden gate (three conjuncts, all enforced harness-side).** Proposed ids are a
      SUBSET of the pool AND the full-instance validator passes AND the plan is strictly
      longer than best_plan_so_far (the fallback). The subset conjunct makes an
      out-of-pool proposal structurally worthless, not merely prompt-forbidden; the
      strictly-longer conjunct means the critic is adopted exactly when it realises a
      genuine coordination gain over the best single worker. The critic is never told
      the outcome (no validity leakage, as everywhere).
    * **Failure semantics (mirrors C2's reflection turns, not the draft loop).** Empty
      post-think -> straight to finalise (§2, no retry). Non-empty but unparseable -> a
      no-op: the fallback stands and the workflow proceeds to finalise (C2's revise has
      exactly this semantics -- a wasted turn, not a retried one). A gate-rejected
      proposal is a no-op. Every critic outcome therefore leaves best_plan_so_far >= the
      fallback.
    * **No revision round -- one critic call, final.** No critique->revise pair, no worker
      re-entry, no second critic turn. A separate critique-then-propose pair would re-send
      the same (pool dump + draft) input twice with no new information in between (the
      critic has no tools); with thinking ON the deliberation already happens inside the
      think block, so the pair buys a second input re-encode, not a second insight --
      pure fixed overhead exactly at the caps where C3 is squeezed hardest. Re-opening
      workers on critique would re-encode both worker contexts (the most expensive
      operation in C3) and add MAST-style inter-agent misalignment surface. The ladder
      stays clean: C2 = mandated reflection with revision cycles inside one shared
      context; C3 = decomposition + one fresh-context coordination pass over the
      workers' candidates; C4 (if run) reuses this same pool-restricted critic design on
      an undecomposed planner, isolating decomposition itself.
    * **Interpretation sentence (pre-specified before the pilot).** Workers independently generate
      candidate sub-plans; the critic coordinates ONLY the candidates the workers
      discovered. As C2's reflection-only revise cannot reach missing-information
      errors, the pool-restricted critic cannot reach missing-CANDIDATE errors (a person
      never proposed by any worker is unreachable by construction), and a failed critic
      leaves C3 at the best single half. Both constraints condition the reading of
      "C3 ~= C2" exactly as the invalid taxonomy conditions "C2 ~= C1".
- **C3-5 -- Budget symmetry under the single shared cap (mechanism LOCKED; share value
  [PILOT]).** One shared ledger per instance (§2, unchanged: same reserve, same guard
  formula, same per-call cap formula). On top, C3 adds an intra-condition allocation so
  that execution order confers no budget advantage:
    * Working budget W = cap - reserve. Each worker gets a quota Q = floor(W * s),
      s = 3/8 by default ([PILOT]: confirmed or re-set once on the pilot, then frozen
      for the main run).
    * Per worker call: max_new_tokens = max(0, min(remaining - reserve, quota_remaining)
      - input) -- the global §2 cap and the quota cap, whichever is tighter. The call is
      booked to BOTH the global ledger and the worker's quota (input + thinking + answer,
      same identity). A worker whose next call no longer fits ends its loop (C3-6).
    * Quotas never transfer between workers: worker B's room is invariant to everything
      worker A does, and vice versa -- symmetry by construction, which is what makes the
      fixed A-then-B execution order harmless.
    * Unused worker budget flows only DOWNSTREAM: whatever the workers leave unspent is
      simply global `remaining` when the critic runs. The critic has NO quota -- it runs
      under the plain global guard, exactly like C2's verify/revise. With s = 3/8 the two
      workers can jointly book at most 3/4 W, so the critic is structurally guaranteed at
      least W/4 of room; worker leftovers only add to it.
    * Quota exhaustion is an internal allocation boundary, not budget exhaustion:
      TokenUsage.budget_exhausted stays a global-ledger fact (a worker stopped by its
      quota while global room remains does not set it).
    * Pre-specified expectation recorded before the pilot (mirror of C2's scaffold-cost note): at the 2k cap the
      critic's input (the candidate-pool dump, which scales with pool size) will often
      exceed its guaranteed room -> the critic is guard-skipped and C3 degenerates to the
      best single sub-plan. This is the §2 discipline working, not a defect: the
      structural minimum cost of the MAS apparatus, symmetric with C2's verify/revise
      being skipped at the same caps. Recorded in advance so it is not read as a post-hoc
      excuse.
- **C3-6 -- Empty-worker policy (resolves the §2 park-flag).** The §2 empty rule
  generalises per scope: empty post-think ends the ENCLOSING loop immediately with its
  best_plan_so_far. For C1/C2 the enclosing loop is the whole instance; for a C3 worker
  it is that worker's sub-loop only.
    * A worker that aborts (empty post-think, or the second format failure after the one
      permitted retry) or runs out of room (quota or global) ends ITS loop; its product is
      its best_plan_so_far, possibly empty. The sibling worker still runs: it is the next
      scheduled stage of the fixed workflow, on different input, under its own untouched
      quota -- not a retry of the failed one. Aborting the whole workflow on one worker's
      empty output would make C3 strictly more fragile than C1/C2 for no methodological
      reason.
    * The aggregator always runs (zero tokens) over whatever exists; an empty sub-plan
      contributes nothing to the fallback, the draft, or the pool.
    * The hidden-retry boundary sits at the critic: if the candidate pool is EMPTY
      (equivalently, both workers produced nothing valid), the critic is SKIPPED and the
      workflow routes straight to finalise. Reviewing "nothing" is not coordination -- a
      critic over an empty pool would be a fresh single-agent planning attempt, i.e. the
      extra effective retry the park-flag forbids (C1's empty run goes straight to
      finalise; C3's must too).
    * Consequently every empty output has one predictable consequence: the stage's scope
      ends, the plan floor (the fallback) survives, and no LLM stage is ever re-run.
- **C3-7 -- Roles, logging, and finalisation identity.** CallRecord roles: `worker_a`,
  `worker_b`, `critic`, `finalize`. The supervisor and the aggregator never appear in the
  call log -- their absence is the recorded evidence of zero-token coordination. The
  instance-level finalise is byte-identical to C1/C2: the same react_core node, the same
  reserve, thinking OFF + guided JSON, serialising best_plan_so_far and never a
  trajectory. n_steps = worker A steps + worker B steps + (1 if the critic ran).
- **Mirror of the C2 boundary (pre-specified; recorded before the pilot).** As
  reflection-only revise cannot fix missing-information errors in C2, the pool-restricted
  critic cannot fix missing-candidate errors, and a failed critic leaves C3 at the best
  single half. Risk: baseline C3 may be so weak that C3 wins nowhere, which would conflate
  "decomposition does not help" (an interesting result, supporting H0 / Tran) with "the
  candidate restriction and the single critic call are too tight to realise coordination's
  benefit" (a design artefact). Resolution, fixed in advance:
    1. Baseline C3 = the pool-restricted critic over the non-planning aggregator, for
       interpretive cleanliness and the preserved scoring invariant.
    2. The fallback rule above is the locked floor; no mechanically constructed plan ever
       enters the score.
    3. The disambiguation is the pre-declared, pilot-gated ablation slice C3-8. Baseline
       C3 stays pool-restricted. One pre-announced slice, not scope creep -- exactly as
       the invalid taxonomy makes "C2 ~= C1" interpretable.
- **C3-8 -- Full-information-critic ablation: operational, pilot-gated trigger (LOCKED;
  re-cast 2026-07-16 from the earlier LLM-aggregator slice after the C3-3/C3-4
  revision -- the baseline coordinator is now itself an LLM, so the open question is no
  longer "is the stitch too weak?" but "is the candidate restriction too tight?").**
    * **When evaluated (re-cast 2026-08-04).** Once, on a pre-registered pilot slice,
      before the main run. The original wording named the `Qwen3-8B-AWQ` pilot sweep;
      that model was excluded on the pre-pilot feasibility check (§1), so the pilot no
      longer runs on a different model from the main run. The separation the trigger
      depends on is therefore **instance-level, not model-level** — otherwise the
      ablation decision would be taken on the same data the thesis later reports.
      Main-run results can NEVER trigger it; a main-run-only C3 collapse is reported in
      Limitations, not chased with a new architecture.
    * **Seed ranges (LOCKED 2026-08-04).** Generator seeds are partitioned once, and no
      instance appears in two ranges:
        - `0–9999` — development, live smoke and ad-hoc calibration. **Never** enter
          the pilot or the main run. Everything measured during the 2026-08-02/04
          bring-up sits here, which is why none of it can leak into a reported result.
        - `10000–19999` — pilot: complexity binning, cap calibration, and the C3-8
          trigger slice.
        - `100000–199999` — held-out main experiment.
      Within each generator-parameter cell, take the first N qualifying instances in
      **ascending seed order**, with N fixed in advance. No manual selection, and no
      replacing an instance after seeing its result — that is what makes the slice
      pre-registered rather than chosen.
      If a second model is reinstated (§6), the split may be revisited **before the
      first pilot run**. Once the pilot has started, the rule and the ranges are frozen.
    * **Trigger.** In the pilot bins, mean satisfaction of C3 is <= that of BOTH C1 and
      C2 in EVERY (level x cap) cell -- C3 is nominally best nowhere, including the
      predicted medium-complexity cells. If any cell has C3 nominally on top, the
      ablation does not run at all.
    * **The slice (C3-FI), fixed in advance.** Identical to C3 except ONE component: the
      critic receives the FULL instance data (every person, window, and the whole travel
      matrix) and the subset conjunct of the hidden gate is lifted; guard mechanics,
      output contract, failure semantics, fallback and everything else stay byte-
      identical, so information scope is the single varied factor. Scope: medium and
      hard levels x the 4k and 8k caps, same paired instances, pilot model. Results go
      to the Discussion as a secondary analysis; the primary C1-C3 comparison always
      uses baseline C3.
    * **Interpretation, fixed in advance.** C3-FI >> C3 on the slice => the candidate
      restriction was the binding constraint (the baseline reading is qualified
      accordingly, and C3-FI is reported as a full-information upper reference, not a
      like-for-like condition -- it reintroduces exactly the privilege the review
      removed, which is why it lives in an ablation, never in the baseline);
      C3-FI ~= C3 => the weakness is decomposition itself (supports the H0 / Tran
      reading).

---

## 5. Evaluation & statistics

**Hypotheses (canonical definitions — transcribed from the exposé 2026-07-26).**
This block is the single authority for the labels `H0` / `H1` / `H2`. It was added because
the labels were previously defined only in the exposé PDF, and the paper notes in
`thesis/literature/` drifted into at least two mutually incompatible numbering schemes
(documented in §8). Any note or chapter using a different scheme is stale and must be
aligned to this block, not the reverse.

- **H1** — the single agent is better at LOW task complexity.
- **H2** — the MAS is better at HIGH task complexity.
- **H0** — no crossover: the ordering does not reverse anywhere in the studied range.

These definitions are aligned with the approved exposé and are the canonical hypothesis
labels for the repository.

Consequences for wording: "supports H1" means "supports single-agent superiority in the
easy bins", not "supports a MAS advantage". Because the labels have been used both ways in
older notes, prefer spelling out the substance (e.g. "an expected MAS advantage at high
complexity") wherever ambiguity could survive; §7 L1–L5 deliberately does exactly that.

**Satisfaction rate (per instance):**
- `S = valid meetings / solver optimum`.
- Any hard-constraint violation makes the whole plan invalid → `S = 0`.
- Instances with optimum = 0 are excluded from the primary analysis and reported
  separately.

**Scorer (implemented Layer 5, `src/evaluation/scorer.py`, CPU-only):**
- **Same objective both sides.** `S = achieved_objective / solver_optimum`, where the
  achieved objective is computed with the SAME expression as the solver's objective.
  The OPTW uses unit prizes (every met person = 1), so the objective is the met-people
  count and numerator and denominator are both that count. If prizes ever become
  non-uniform, the solver AND the scorer switch to the weighted sum together — a raw
  count would then mis-credit.
- **Independent revalidation.** The scorer ignores the agent's self-report: it runs the
  hidden `validator` on the raw plan; any hard violation → whole plan invalid → `S = 0`;
  only a fully valid plan earns its objective.
- **optimum == 0 rule:** a valid plan (necessarily empty) scores `S = 1.0` (it reached
  the zero optimum, avoiding 0/0); these instances are still excluded from the primary
  analysis at aggregation.
- **Boundary check:** a valid plan cannot beat the proven optimum; `achieved > optimum`
  raises (solver/validator disagreement → stop and fix).
- **`satisfaction / 1k` is NOT computed here.** The scorer stays pure correctness; the
  efficiency metric joins `Score` with `TokenUsage` only at aggregation (`RunResult`),
  so no `TokenUsage` enters the scorer.

**Aggregation (per bin = condition × level × cap):**
- Primary: `mean(S)` over the instances in the bin.
- Robustness check: `median(S)` — flag in Discussion if it diverges from the mean
  (signals a skewed distribution).

**Secondary metrics** (naming fixed 2026-08-04, see §6): `proposal parse rate`,
`proposal validity rate` (malformed counts as invalid), `proposal acceptance rate` —
all three over ALL proposal attempts — plus the invalid breakdown by error type
(multi-label, counted over rejected proposals), optimality rate, number of calls,
latency, and satisfaction per 1000 tokens.

The old `feasibility rate` label is retired. Measured on `final_plan` it is trivially
100% by construction and is reported as a property of the design, never as a finding.

**Crossover detection:**
- `Δ = S_C1 − S_C3` for each (level × cap).
- Bootstrap 95% CI by resampling instances with replacement.
- A crossover counts only if:
  1. the sign of Δ flips from + to − as complexity rises,
  2. the CI excludes zero both before and after the flip,
  3. the pattern repeats on ≥ 2 cap values.
- Plus an interaction effect (system × complexity level) as a formal test.
- Plus Cohen's d reported alongside each CI (effect size).

**AMENDMENT — PRIMARY INFERENCE FOR THE MAIN EXPERIMENT, 2026-08-09.** The block above was
written for the old level binning and for a design with several caps. The bands are now
frozen (§3, amendment of the same date) and the budget policy has changed, so the primary
decision rule is restated here. Recorded **before** any budget-dev or held-out run.

### Conditions and their standing

| | Role |
|---|---|
| **C1 vs C3** | **Primary contrast.** Single ReAct against the hierarchy |
| C2 | Secondary architectural contrast |
| **C5 — Validated Best-of-3** | **Secondary strong single-agent baseline.** LOCKED 2026-07-26 (§4 C5); the specification stands and its implementation is still a separate commit. It is carried into the main matrix if implemented in time, and its absence must be reported as a missing sampling baseline rather than passed over |
| C4 | Optional and mechanistic; the first thing cut if resources run short |

### Primary estimand

    Δ_l = mean over instances of ( S_C1 − S_C3 )

computed separately for `l ∈ {Low, Medium, High}` pairwise-conflict-density regimes, on the
same instances, at the primary cap. Paired within instance.

### Primary inference: an intersection–union test

    H0_Low  : Δ_Low  ≤ 0        vs   H1_Low  : Δ_Low  > 0
    H0_High : Δ_High ≥ 0        vs   H1_High : Δ_High < 0

> **A crossover is established if and only if the 95% one-sided LOWER confidence bound on
> `Δ_Low` lies above zero AND the 95% one-sided UPPER confidence bound on `Δ_High` lies
> below zero.**

The one-sided bounds are the decision rule; two-sided 95% intervals are shown in every
table and figure but do not decide anything. Using a two-sided interval as the decision
would test each component at 0.025 and make the procedure more conservative than intended.

No multiplicity correction is applied to the conjunction. In an intersection–union test the
null is rejected only when **both** components reject, so the level of the whole procedure
is at most the level of each component — this is a property of the construction, not a
concession.

**What the test does and does not claim.** It establishes a **sign reversal across the
tested range of pairwise conflict density**. It does **not** locate the threshold. The
permitted primary claim is "a crossover across the tested range is established", never
"the crossover occurs at Medium".

`Medium` is descriptive and is not part of the primary criterion; adding a third condition
would only tighten an already conservative test. It localises afterwards: `Δ_Medium > 0`
places the transition above Medium, `Δ_Medium < 0` places it at or below, and an interval
covering zero marks Medium itself as the likely transition region.

*Bootstrap.* Percentile bootstrap over instances, 10 000 resamples, fixed seed, matching the
procedure already used for the pilot. The one-sided lower bound is the 5th percentile of the
resampled mean; the one-sided upper bound is the 95th.

### Budget policy

> **[SUPERSEDED 2026-09-14]** The single-primary-cap policy below no longer governs the
> analysis. See §5, amendment 2026-09-14: all four caps have equal status. Kept for history.

One **primary cap**, selected mechanically from the frozen ladder by the rule below, plus
one **secondary cap** for a resource-regime analysis. Everything at the secondary cap is
secondary.

> **Primary cap** = the smallest cap in the frozen ladder at which C1 produces a non-empty
> validated plan in **at least 50% of budget-dev runs in each of the three frozen density
> regimes separately**.

Separately, not pooled: a pooled rate can pass on 90% at Low and 0% at High, choosing a
budget at which the High regime is on the floor and `Δ_High < 0` is unreachable by
construction.

The rule reads C1 only. Requiring C3 to work would let the treatment architecture choose
the budget, and C3's coordination overhead is part of what the experiment measures.

*Definition of success for this rule*, fixed structurally rather than through the score:
a run counts if its final plan is validator-approved **and** contains at least one meeting.

> **Failure rule.** If no cap in the frozen ladder reaches 50% in all three regimes, **no
> primary cap is selected, budget calibration is declared failed, and the main experiment
> does not start.** Falling back to the top rung because nothing qualified is not available.

> **Secondary cap** = the immediately preceding rung of the frozen ladder, or the next rung
> above if the primary is the lowest. Mechanical, so no rung is chosen after seeing which
> looks more interesting.

### Still open, and blocking

The number of instances per band. 50 is inherited from the previous design and **is not
adopted here**: the primary inference is now a conjunction, whose power is governed by the
weaker of the two components, so it must be established by simulation against this rule
rather than carried over. The `O` allocation rule and the held-out sampling algorithm follow
from that number and are fixed after it.

**AMENDMENT — the Type-I validation rule was wrong, 2026-08-10.** Recorded before the
corrected check runs, and separately from the power result it wrongly condemned.

*What was specified.* Power simulation v2 (`8d4c51a`) validated the percentile bootstrap
against two boundary nulls across 108 nuisance configurations each, and declared the
procedure anticonservative when the **worst** cell's Monte-Carlo interval cleared
`alpha = 0.05` entirely.

*Why that is wrong.* The rule reads a maximum over 108 draws as if it were a single
estimate. At 2000 replicates one cell has a standard error of 0.0049 under a true rate of
0.05, so the maximum of 108 independent cells is expected near `0.05 + 2.7 x 0.0049 =
0.063` when nothing at all is wrong. The observed worst cells, 0.0645 and 0.0660, sit just
above that. The escalation rule could not rescue it: those intervals do not straddle 0.05,
so the cells counted as decided and were never re-run. **A threshold applied to the maximum
of a large grid is crossed by construction.** The NO-GO it produced (`e8bef01`) is recorded
and is not treated as evidence about the bootstrap.

*What replaces it.* Every one of the 108 configurations is tested for both boundary nulls,
each cell one-sided against `H0: rate <= 0.05` with an exact binomial tail at 10 000
replicates, and the family of 216 tests is corrected by **Holm**. The procedure is called
anticonservative only if some cell survives Holm at the family level. Cells are not chosen
after seeing the results: the whole grid is tested, and picking "the two worst" for a
high-precision re-run — which is what the first reading of the failure suggested — would
have reintroduced the same selection effect one level down.

*What this does not touch.* `N = 95` is not recalculated. It passed on power across all 108
binding configurations under a rule that had no such defect, and the Type-I question is
about the inferential procedure, not the sample size. **If the corrected validation still
fails, the response is to change the statistical test, never to raise `N`**: an
anticonservative interval procedure stays anticonservative at any sample size.

**AMENDMENT — the synthetic power model stops being binding, 2026-08-10.** Recorded before
any new LLM outcome is generated.

**Neither `N = 35` nor `N = 95` is adopted.** Both simulations, their artifacts and their
audit trail are kept in full — `results/analysis/power_iut/` and `power_iut_v2/`, with the
commits that produced them — and are reclassified as **exploratory**. Nothing is deleted.

*Why.* The two runs differed by a factor of nearly three, and the whole of that difference
came from assumptions nobody has measured: how much instance heterogeneity there is on the
logit scale, how often an agent produces no plan at all, and how these interact. The second
model was more honest than the first precisely because it added a failure mode the first
lacked — which is the point: the answer moved that much because the inputs were invented,
not observed. A sample size is only as defensible as the variance estimate under it, and
sizing a real experiment on a distribution whose parameters were chosen by hand inverts the
proper order.

*What replaces it.* The sample size will be derived from **real development runs**. The
order is: freeze the budget-dev procedure, run it on the development seed range with the
real model, select the primary cap by the rule already registered above, then run C1 and C3
on a development set drawn from the frozen density bands and observe the actual
distribution of satisfaction — the real zeros, the real failures, the real spread and the
real tie rate. `N` is then chosen conservatively from those observations and frozen before
the held-out range is opened.

*What stays binding.* The bands, the estimand, the intersection-union rule, the one-sided
bootstrap bounds, the primary-cap rule and its failure rule, and the standing of C2 and C5.
Only the route to `N` changes.

*What is untouched.* Held-out seeds 100 000+ remain unopened. Development runs use
30 000–39 999 and nothing else.

*The corrected Type-I check finished after this decision and is recorded here.* It had
already been launched, so it was allowed to complete rather than being discarded. Verdict:
**ANTICONSERVATIVE** — one cell of 216 survives Holm, rejecting at 0.0589 against a nominal
0.05 (base mean 0.35, `σ = 0`, `λ = 0`, calibration-mix optima; 10 000 replicates, so about
four standard errors above nominal). The mean rate across the whole family is 0.0517 and 22
cells exceed 0.055.

Like everything else from the synthetic model this is **exploratory and binds nothing**. It
is kept because the question it raises is not about the invented parameters: the offending
corner is `σ = 0, λ = 0`, the configuration with the *fewest* distinct outcome values, and
the real data will also be coarse and tie-heavy — satisfaction is `achieved / O` with `O`
restricted to 3 and 4, so per-instance differences take few values. A percentile bootstrap
run on a small, heavily tied discrete sample can miss its nominal level, and that would
transfer. The already-registered response stands and is not weakened by the model's
retirement: **if this reproduces on real development data, the statistical test changes and
`N` does not.** The real development runs are what decide it.

**AMENDMENT — THE BUDGET-DEV PROCEDURE, 2026-08-10.** Written and committed before any
budget-dev run existed, so nothing below can have been shaped by its outcome.

*The frozen ladder, stated explicitly.* The primary-cap rule above refers to "the frozen
ladder" without naming its rungs anywhere in this document. The rungs are **8 000 / 16 000 /
32 000 / 64 000**, the ladder the formal pilot ran. This closes a gap in the earlier text
rather than making a new choice.

*All four rungs are run.* Ascending with an early stop at the first qualifying rung would
be cheaper, but the secondary cap is defined as an adjacent rung and would have to be
measured anyway, and the budget-response curve is a result in its own right.

*The development set.* Sixty instances — twenty per frozen density band — selected by
`scripts/build_band_manifest.py` from a structural pool over seeds 30 000–30 099. C1 only.
240 runs in total.

*What twenty per band can and cannot support.* This is a development calibration set, not a
confirmatory sample. It is sized to tell a rate clearly above one half from one clearly
below, and it is **not** an estimate of where the 50% threshold lies. No interval from it
enters any inferential claim.

*Selection rules fixed here, before any development run.* The 2026-08-09 amendment froze the
bands, `n = 8`, the identical-`O` requirement and the diversity requirements. Three further
rules were needed to turn those into a specific sample and are fixed now:

1. *The shape of the shared `O` histogram* is the frozen design's own composition — 41 at
   `O = 3` and 24 at `O = 4` out of 65 — scaled to the requested band size by
   largest-remainder rounding; at twenty per band, 13 and 7. Keeping the composition constant
   is what allows a rate measured on the development set to be read as a rate on the main
   set. If it is unattainable the selector walks a fixed ladder of alternatives ordered by
   distance from it and records which was used.
2. *One instance per seed across the whole manifest.* The primary analysis resamples
   instances as independent units, and two instances built from one seed share a random
   stream.
3. *The canonical order* — seed, travel structure, tightness, overlap — with the selection
   being the earliest admissible one under it, so the sample is a fact about the pool rather
   than a solver artefact.

The same selector produces the held-out main set later; these rules are therefore fixed for
it too. The held-out range is not reachable from the selector at all — adding it is a change
to this document, not a command-line argument.

*Sequence.* Budget-dev fixes the cap. Only then do C1 and C3 development runs on a
development set establish the real spread, the real zero rate and the real tie rate, from
which `N` is chosen conservatively and frozen. Held-out opens after that and not before.

**RESULT — BUDGET CALIBRATION FAILED, 2026-08-10.** Recorded before the cause was
investigated, so the record cannot be shaped by what the diagnosis turns out to say.

240 C1 runs over the 60-instance development set, four rungs, zero errors. The completeness
audit passed: 240 runs, one condition, one model, 20 distinct instances per band, no
duplicated or missing cell, every seed inside 30 000–39 999, both manifest hashes matching
the frozen selection. Analyser `scripts/analyse_budget_dev.py`, committed at `f7fc931`
before it was pointed at the data; artifacts in `results/analysis/budget_dev/`.

Share of runs whose final plan is validator-approved and holds at least one meeting:

| cap | Low | Medium | High |
|---|---|---|---|
| 8 000 | 0.00 | 0.00 | 0.00 |
| 16 000 | 0.00 | 0.00 | 0.05 |
| 32 000 | 0.00 | 0.10 | 0.05 |
| 64 000 | 0.45 | 0.45 | 0.25 |

**No rung reaches 50% in any band, let alone in all three. By the registered rule no primary
cap is selected, budget calibration is declared failed, and the main experiment does not
start.** The top rung is the nearest miss and is not available: taking it would be the
fallback the rule explicitly forecloses.

*The rule did what it was written to do.* It caught C1 sitting on the floor before the main
experiment spent a token. Had the experiment run at 64 000, every `Δ = S_C1 − S_C3` would
have been measured against a comparator that produces nothing in most runs, and a crossover
result would have been an artifact of that floor rather than a property of the
architectures.

*What this does not touch.* The bands, the estimand, the intersection–union rule and the
one-sided bounds are unaffected. What failed is the budget at which this single-agent
baseline can operate at `n = 8` — a finding about the condition and the instance size.

*What is forbidden in response*, stated now so it cannot be rationalised later: lowering the
threshold, pooling the bands, dropping the High band, redefining success, or selecting
64 000 anyway. Any change to the ladder made **after** this result is a post-hoc change and
must be recorded as one, with the consequence that the budget choice specifically is no
longer pre-registered.

**DIAGNOSIS — the budget is not the binding constraint at the top rung, 2026-08-10.**
Non-binding description, read after the verdict above was recorded and unable to change it.

At 64 000 the qualifying rate is flat across termination modes: 0.44 among runs the budget
guard cut short, 0.37 among runs that finished on their own, 0.33 among runs that exhausted
the budget. Budget-limited runs succeed **no less often** than voluntary ones. Failures at
that rung leave 30% of the budget unspent and make 1.43 proposal attempts each; only 5 of 37
never proposed at all. Proposal validity is 0.247, and the dominant rejection reason is
`travel_infeasible`, followed by `window_violation`.

At 32 000 the picture is the opposite — failures consume 94% of the cap and 28 of 57 never
propose — so the character of the constraint changes between the two rungs. Below 32 000 the
budget binds; at 64 000 it does not, and a further rung would buy additional invalid
proposals rather than plans. **Extending the ladder is therefore contraindicated by the
measurement, independently of being post-hoc.** At 8 000 there are zero proposal attempts in
60 runs at 96% cap utilisation: at `n = 8` the lower rungs measure whether the instance can
be read, not whether it can be planned.

*A defect in the rule itself, recorded because executing it is what exposed it.* The formal
pilot already showed C1 reaching a valid plan in 0.47 of runs at cap 64 000 — on the easier
mixed `n = 4, 6, 8` subset, and that number is in section 3. A 50% requirement at pure
`n = 8` therefore demanded more of the baseline than it had demonstrated on simpler
instances at the moment the threshold was registered. The threshold is not revised in
response; it is recorded that it was set above the available evidence.

**AMENDMENT — POST-GATE ARCHITECTURE DIAGNOSTIC AT `n = 8`, 2026-08-11.** Written and
committed before any C2 or C3 outcome on this instance set exists.

*Standing of this run, stated first because it is the thing most easily misread.* This is a
**development diagnostic**, not a confirmatory test and not a second attempt at the budget
gate. **It cannot repair, replace, reopen or retroactively pass the C1 budget gate recorded
above.** That gate read C1 only, by design, so that the treatment architecture could not
choose the budget; a C2 or C3 result is therefore not evidence about it in either direction.
Whatever this diagnostic returns, no primary cap is selected by it and the main experiment
does not start on its authority.

*The question, fixed in advance.* On the same `n = 8` development instances at cap 64 000, do
the structured conditions C2 and C3 recover validator-approved non-empty plans substantially
more often than plain C1 does?

*Why it is worth 120 runs.* The gate established that C1 sits near the floor at `n = 8`. It
did not establish whether that floor is a property of the **task at this size** or of the
**plain single-trajectory architecture**. Those two readings lead to different next steps —
the first says the instance size must be reconsidered, the second says the baseline is the
weak component — and nothing already collected separates them. C3 at `n = 8` gives each of
its two workers four people, a size at which the formal pilot shows the agent working; if the
floor is architectural, C3 should clear it and C1 should not.

*The matrix, frozen.* The identical 60 instances of
`results/manifests/subset__bands_budget_dev__911e4864d6fb.json` — 20 Low, 20 Medium, 20 High
at `n = 8` — cap **64 000 only**, conditions **C2 and C3**, 60 runs each, 120 in total. No new
seeds, no held-out data, prefix caching off, `max_model_len` 32768, and the same model,
sampling profile and serving flags as the completed C1 budget-dev. Outcomes are written to a
**separate directory**, `results/logs/budget_dev_arch_diag/`, and never mixed with the C1
budget-dev outputs.

*The descriptive interpretation, frozen before the outcomes.*

1. Report the validator-approved **non-empty** plan rate, pooled and separately for Low,
   Medium and High. Non-empty specifically: an empty plan is vacuously feasible in this
   harness, so a generic feasibility rate would be uninformative by construction.
2. Compare against the already-completed C1 result at the same cap on the same instances,
   paired by instance.
3. **If at least one structured condition reaches at least 50% non-empty validated plans in
   each of the three bands**, record that `n = 8` is **not a universal task-wide floor under
   all architectures**. This is a statement about architectures at `n = 8` and nothing more.
4. **If neither does**, record that `n = 8` remains problematic across the condition set and
   that instance size and design should be reconsidered.
5. Either way this is description, not inference. No hypothesis test is introduced, no
   threshold is tuned from the outcomes, and the 50% figure is reused from the registered
   gate rather than chosen anew — it is a reference point here, not a decision rule.

*Analyser, committed before the outcomes.* `scripts/analyse_arch_diagnostic.py`, fail-loud:
it verifies the exact expected run set, refuses duplicates, missing cells and foreign
manifest hashes, requires exactly 60 C2 and 60 C3 at cap 64 000, and pairs C1/C2/C3 on the
identical instance. It reports by band and pooled: non-empty validated rate, satisfaction,
realised tokens, calls, termination mix, proposal validity and the invalid-reason taxonomy;
for C2 the verify and revise stage reach; for C3 proposal validity split by worker versus
critic, and how often the critic ran at all.

*Provenance rule for the run machine.* The sweep records the git HEAD in each result. While
these two runs are in flight the server working tree must not be pulled, checked out or moved
in any way; development continues elsewhere. Mixed provenance inside one diagnostic would be
unrecoverable after the fact.

**RESULT — THE `n = 8` FLOOR IS NOT UNIVERSAL ACROSS ARCHITECTURES, 2026-08-13.**

120 runs, C2 and C3 sequentially on one endpoint at cap 64 000 over the identical frozen 60
instances, zero errors. Both conditions carry the same git HEAD `4e52673`, so the provenance
is single. The completeness audit passed: exactly 60 per condition, no duplicate run id, no
missing cell, both manifest hashes matching the frozen selection, one model, every seed
inside the development range, and all three conditions paired on the identical instance.
Analyser `scripts/analyse_arch_diagnostic.py`, committed at `4e52673` before any of these
outcomes existed; artifacts in `results/analysis/budget_dev_arch_diag/`.

Share of runs whose final plan is validator-approved and holds at least one meeting:

| condition | Low | Medium | High | pooled |
|---|---|---|---|---|
| C1 ReAct | 9/20 = 0.45 | 9/20 = 0.45 | 5/20 = 0.25 | 23/60 = 0.38 |
| C2 verify/revise | 14/20 = 0.70 | 10/20 = 0.50 | 9/20 = 0.45 | 33/60 = 0.55 |
| C3 hierarchical MAS | **16/20 = 0.80** | **14/20 = 0.70** | **14/20 = 0.70** | 44/60 = 0.73 |

Mean satisfaction over the same runs: C1 0.300, C2 0.418, C3 0.643. Proposal validity: C1
0.247, C2 0.375, C3 0.691.

**Reading, as fixed in advance.** At least one structured condition — C3 — reaches the 50%
reference rate in **every** band, so **`n = 8` is not a universal task-wide floor under all
architectures**. C2 clears Low and Medium and misses High at 0.45.

*What this does not do.* **The C1 budget gate remains failed.** It read C1 only, by design,
so that the treatment architecture could not choose the budget; a C2 or C3 result is not
evidence about it in either direction. No primary cap is selected here, and the main
experiment does not start on this result. The 50% figure is the reference point reused from
the gate, not a decision rule in this diagnostic.

*What it changes about the diagnosis.* The failed gate established that C1 sits near the
floor at `n = 8` but could not say whether the cause was the task at this size or the plain
single-trajectory architecture. It was the architecture: on the same instances, at the same
budget, with the same model and serving configuration, the hierarchy reaches a valid
non-empty plan in 70–80% of runs where the single agent reaches 25–45%. The earlier
recommendation to abandon `n = 8` is therefore **not supported by this evidence** and is
withdrawn as a default; whether to keep `n = 8` is decided together with the budget question
and not by this diagnostic alone.

*The shape of the difference, recorded because it is what the research question is about.*
C1 and C2 both fall away at High density (0.25 and 0.45) while C3 stays nearly flat across
the three bands (0.80 / 0.70 / 0.70). This is the first observation in the project of an
architecture difference that grows with pairwise conflict density rather than sitting at a
constant offset. It is **descriptive**: 20 instances per band on a development set, one cap,
no hypothesis test, and the development seeds are consumed for exploration by construction.
It is not evidence of a crossover, which requires the held-out design and the registered
intersection–union test.

*What it does not resolve, and what C4 is now for.* C2 and C3 differ in several ways at once
— decomposition, the number of search agents, the locus of verification, the pool
construction — so this result cannot attribute the difference to decomposition. **C4 is the
condition that separates them**: a fresh independent critic without decomposition. If C4
clears High as C3 does, the fresh critic is doing the work; if C4 falls away as C2 does,
decomposition is. C4 and C5 are implemented and untested as of this entry.

**EXTENSION — C4 JOINS THE SAME DIAGNOSTIC, 2026-08-13.** Written and committed before any
C4 outcome on this instance set exists.

*Why C4 next and not C5.* The C2 → C3 step changes decomposition, the number of search
agents, the locus of verification and the pool construction at once, so the result above
cannot attribute anything to decomposition. C4 splits that step in two: a fresh independent
critic **without** decomposition. The architectural path becomes C1 → C2 → C4 → C3, and the
two contrasts are the ones already written into §4 — C2 → C4 asks whether a fresh
independent critic adds value beyond same-context self-verification without decomposition;
C4 → C3 asks what decomposition together with its coordination interface adds once both
architectures already contain a fresh critic. C5 follows afterwards as the compute-allocation
control: whether several independent search trajectories, rather than decomposition, explain
what C3 gains.

*Matrix.* The identical 60 instances, cap 64 000, condition `c4_planner_critic`, 60 runs,
same model and serving configuration, written to the same
`results/logs/budget_dev_arch_diag/`. Frozen expected-runs manifest
`expected_runs__budget_dev_arch_diag_c4_64k.json`, content hash `9c0fa44384bd`.

*The analyser is extended, not rewritten.* Its condition set becomes a parameter whose
**default is the frozen pair**, and it reads only the conditions the caller declares, so the
recorded C2/C3 reading stays reproducible byte for byte from the same directory even after
C4 files land there. A test asserts exactly that. Provenance is now reported **per
condition**: conditions run days apart necessarily carry different commits and that is not a
defect, whereas a single condition split across commits is, and only the latter is flagged.

*What is fixed in advance for C4, and what is not.* Reported: the validated non-empty rate
per band and pooled, paired against C1 and against the other conditions, plus C4's own
log-only diagnostics — pool size, fallback size, headroom, critic reach, critic improvement
and evidence coverage. **No threshold decides anything for C4.** The 50% figure remains the
reference point inherited from the gate. No hypothesis test is introduced, and the two
permitted claims are the ones already recorded in §4 and may not be strengthened.

*The degeneracy check is part of the result.* If C4's mean headroom is near zero, or the
critic is rarely reached, then C4 had little opportunity to improve anything and a null
result says nothing about fresh critics. That must be read off the diagnostics before the
rate is interpreted, not after.

**RESULT — C4, AND THE DECOMPOSITION HYPOTHESIS IS NOT SUPPORTED, 2026-08-13.**

60 C4 runs on the identical instances at cap 64 000, zero errors, 180 diagnostic runs in
total. The audit passed. Provenance differs between conditions — C2/C3 at `4e52673`, C4 at
`f27b137` — which is expected for runs days apart; the per-condition check is clean and no
condition is split across commits. Artifacts in `results/analysis/budget_dev_arch_diag_c4/`.

*Degeneracy check first, as registered.* The critic was reached in **49 of 60** runs (0.817),
mean headroom was **2.3** people, and the critic's plan beat the fallback in **32 of 60**
runs (0.533). **C4 is not degenerate**: the gate was satisfiable and was in fact satisfied
often. Design E therefore works — and it works under its own information restriction, since
mean evidence travel coverage was **0.277**: the critic knew 3.3 of the roughly 13.8 ordered
pairs it needed and received the other 10.5 as `unknown`. It improved half the runs anyway,
without any privileged fact.

*Which quantity is which, stated before the table.* The rate below is the
**validator-approved non-empty plan rate** — a feasibility-like precondition, and the
quantity this diagnostic was registered on. It is **not** the thesis's primary metric. The
primary metric is **satisfaction**, and on it **C3 leads**: 0.643 against C4's 0.489. No
condition may be called "best" on the strength of the rate alone, and C4 is not the leading
condition on performance.

| condition | validated non-empty rate (Low / Medium / High) | pooled rate | mean satisfaction | mean tokens |
|---|---|---|---|---|
| C1 | 0.45 / 0.45 / 0.25 | 0.38 | 0.300 | 48 673 |
| C2 | 0.70 / 0.50 / 0.45 | 0.55 | 0.418 | 55 141 |
| C3 | 0.80 / 0.70 / 0.70 | 0.73 | **0.643** | 45 274 |
| C4 | 0.80 / 0.70 / 0.75 | 0.75 | 0.489 | 45 160 |

**The decomposition hypothesis is not supported on the rate.** It held that C3 gains because splitting
reduces the sub-problem to four people, a size at which the agent works. C4 puts one planner
on all eight, decomposes nothing, and matches C3 on the validated non-empty rate in every
band. **On that quantity, and only on it**, the C2 → C4 step captures essentially the whole
gap and the C4 → C3 step adds nothing measurable. Producing a valid plan at all appears to
be the work of the **fresh independent critic** rather than of decomposition.

*But decomposition is not doing nothing — it is doing something else.* The headline rate
hides a sharp split:

| | C3 | C4 |
|---|---|---|
| validated non-empty | 44/60 | 45/60 |
| mean satisfaction | **0.643** | 0.489 |
| mean satisfaction among runs that produced a plan | **0.877** | 0.652 |
| paired vs C1, recovered / lost | +28 / −7 | +26 / −4 |
| mean Δ satisfaction vs C1 | **+0.343** | +0.189 |

Conditional on producing a valid plan at all, C3 reaches 88% of the oracle optimum and C4
65%. The division of labour this suggests is a **candidate explanation**, not a
scientific conclusion, and must be written as one:

> *Candidate explanation.* The fresh critic may govern whether a valid plan is produced at
> all, while decomposition may govern how good that plan is.

It is not licensed as a causal claim. C3 and C4 differ in more than decomposition — the
number of search agents, the pool rule and the critic's information regime — so neither half
of the sentence is isolated by this design, and the sample is 20 instances per band at one
cap. Writing it as "fresh critic causes feasibility, decomposition causes quality" would
assert two causal attributions the data cannot carry.

A mechanism is available and consistent with the logged structure: two workers each search a
four-person sub-problem thoroughly, so the union of their products names more people than one
planner finds among eight — C4's mean pool size is only 2.87. The C3 critic then coordinates
a richer pool. C4 is meanwhile the less destructive of the two, losing 4 C1-successful
instances against C3's 7, which fits a condition that never breaks a pairing by splitting it.

*Limits, which are severe enough to name before the finding.* Twenty instances per band on a
development set, one cap, no hypothesis test. The C4−C3 difference at High is 15 runs against
14 — one instance — and no claim that C4 ≥ C3 is licensed. What the data support more
robustly is the size and consistency of the C2 → C4 step, which holds in all three bands.
C4 and C3 also differ in more than decomposition: the number of search agents, the pool rule,
and the critic's information regime. The permitted wordings in §4 stand and may not be
strengthened.

**EXTENSION — C5 COMPLETES THE DIAGNOSTIC SET, 2026-08-13.** Written and committed before any
C5 outcome on this instance set exists.

*The question C5 answers, and no other.* C3 and C4 both spend the budget on more than one
piece of work. C5 spends it on **three independent search trajectories with no coordination,
no decomposition and no critic** — the cheapest competing use of the same tokens. It is the
compute-allocation control: is part of what C3 and C4 gain simply the effect of searching
more than once?

*Matrix.* The identical 60 instances, cap 64 000, condition `c5_best_of_3`, 60 runs, same
model and serving configuration, into the same `results/logs/budget_dev_arch_diag/`. Frozen
expected-runs manifest `expected_runs__budget_dev_arch_diag_c5_64k.json`, content hash
`3e43bf255212`. With it the diagnostic set is complete at 300 runs: C1 → C2 → C4 → C3 as the
architectural path, and C1 → C5 as the repeated-search control.

*Degeneracy check, to be read before the rates exactly as for C4.* C5's log-only diagnostics
carry `n_distinct_products` and `winner_attempt_index`. If most runs produce **one** distinct
product, the three trajectories were not diverse and C5 collapsed to C1 on a third of the
budget; a low result then says nothing about sampling. If the winner is almost always attempt
0, the later attempts bought nothing. Either pattern must be reported before the rate is
interpreted.

*Readings fixed in advance, on the primary metric.* Compared on **satisfaction**, with the
validated non-empty rate reported alongside:

1. **C5 close to C3** — part of C3's advantage may come from multiple search trajectories
   rather than from decomposition or coordination.
2. **C5 close to C4** — decomposition and coordination become the more plausible account of
   C3's higher plan quality.
3. **C5 above C3** — distributing the budget across independent attempts would be more
   effective than architectural complexity, and the framing of the whole comparison changes.

No threshold decides among these, no hypothesis test is introduced, and "close to" is not
given a numeric cut: with 20 instances per band at one cap the differences are described, not
tested. Whichever pattern appears, the C1 budget gate stays failed and the main experiment
does not start on this result.

*After C5.* A single C1–C5 diagnostic report is assembled and the interpretation frozen.
Only then are the failed budget gate and the held-out design taken up.

**RESULT — C5 IS STARVED BY ITS OWN BUDGET SPLIT, 2026-08-13.**

60 C5 runs on the identical instances at cap 64 000, zero errors. The diagnostic set is
complete at 300 runs and the audit passed. Artifacts in
`results/analysis/budget_dev_arch_diag_c5/`.

*Degeneracy check first, and it fires.* **C5 showed very low final-product diversity: 54 of
60 runs produced only one distinct final product across the three attempts, 6 produced two,
and none produced three.** The winner was attempt 0 in **56 of 60**, so the later two
attempts won only four times between them.

*Stated as what was measured.* The diagnostic compares the three attempts' **final
products**, not their internal trajectories: three attempts could differ in tool calls and
reasoning and still converge on the same product, and nothing here rules that out. The claim
is about product diversity, which is the precondition Best-of-3 needs to be Best-of-3 at all,
not about trajectory identity.

| condition | validated non-empty rate | mean satisfaction | mean tokens | cap used |
|---|---|---|---|---|
| C1 | 0.38 | 0.300 | 48 673 | 0.76 |
| C2 | 0.55 | 0.418 | 55 141 | 0.86 |
| C3 | 0.73 | **0.643** | 45 274 | 0.71 |
| C4 | 0.75 | 0.489 | 45 160 | 0.71 |
| **C5** | **0.10** | **0.079** | **62 905** | **0.98** |

C5 is the most expensive and the weakest condition of the five, by a wide margin, and it is
the only one that is **worse than C1**: paired on the identical instance it recovers 2 runs
and destroys **19**, mean Δ satisfaction −0.221. It made 37 proposal attempts across 60 runs
against C1's 93, at validity 0.162.

*Consistent with budget starvation, which is not yet established as the mechanism.* Each
attempt receives only `floor(W / 3) = 21 248` working tokens, substantially below the
**48 673** a full C1 trajectory uses on average, and C5 consumes 98% of the total cap while
producing very little final-product diversity and 37 proposal attempts across 60 runs against
C1's 93. Those facts fit starvation. What would establish it is attempt-level evidence —
whether individual attempts reach a proposal at all and where they stop — and that has not
been extracted. Until it is, the sentence stays "consistent with", not "because of".

*The cross-tabulation, and what it can and cannot show.* The joint table separates
perfectly, with nothing off the diagonal:

| distinct final products | satisfaction = 0 | satisfaction > 0 |
|---|---|---|
| 1 | **54** | 0 |
| 2 | 0 | **6** |

The run-by-run correspondence is therefore established: the 54 zero-satisfaction runs are
exactly the 54 single-product runs, not merely the same number of them.

**Two of the three cells are forced by construction and carry no information.** An attempt's
product is its `best_plan_so_far`, which the in-loop gate keeps validator-approved, so it is
either empty or valid. If all three attempts end empty there is exactly one distinct product
and the score is 0 necessarily; if two distinct products exist then at most one can be empty,
so the winner is non-empty and the score is positive necessarily. The cells (1, sat = 0) and
(2, sat > 0) could not have come out otherwise.

The informative cell is the empty one: **(1, sat > 0) = 0** — in no run did all three attempts
converge on the same *non-empty* plan. With the 54 in the first cell this establishes that
every single-product run was a run in which all three attempts produced nothing.

*Which limits what the diversity metric can be used to say.* Low product diversity and
failure are not independent measurements here: a run that produces nothing cannot exhibit
diversity, because "no plan" is one product. The metric therefore cannot separate "the three
attempts were similar" from "the three attempts all failed", and the table must not be read
as showing that low diversity *caused* the failures.

*What Best-of-3's selection actually changed.* The winner was attempt 0 in 56 of 60 runs, and
exactly 6 runs had more than one distinct product. So the selection among attempts altered
the outcome in **at most 4 of 60 runs** — the four in which a later attempt won.

*What this licenses, and what it does not.* The registered question was whether part of C3's
advantage is simply the effect of searching more than once. **The answer is no**: at a
matched budget, splitting into three independent trajectories does not approach C3, C4, C2 or
even C1. None of the three pre-registered patterns applies — C5 landed far below all of them,
a fourth outcome the readings did not anticipate, and it is recorded as such rather than
forced into one.

**This is not evidence that best-of-N sampling is generally ineffective.** The frozen
statement, which the thesis text may use and may not strengthen:

> At `n = 8` and a 64 000 equal total token cap, allocating the budget across three
> independent C1-style attempts performed substantially worse than one longer trajectory and
> produced very little final-product diversity. Repeated independent search under this budget
> therefore does not explain the observed C3 advantage. This does not imply that best-of-N
> sampling is generally ineffective: the per-attempt budget in this diagnostic was
> substantially smaller than the typical token usage of a full C1 trajectory.

Note also what the design varies: C5 changes **both the number of attempts and the depth of
each**, so it answers the practical question — what is better to do with the same 64 000, one
long trajectory or three short independent ones — and not the question of whether sampling
helps when each sample is adequately funded. On the practical question the development answer
is unambiguous: one long trajectory is much better.

*A defect found in the analyser while reading this, and fixed.* The condition-diagnostics
summariser added `n_distinct_products` to its value list twice — once in the generic numeric
loop, once in an explicit branch — so the histogram reported 108/12 over 120 values for 60
runs while the mean stayed correct at 1.1. The corrected histogram is 54/6/0. A doubled
count with a correct mean is precisely the kind of error that reads as plausible; the
histogram is now built from the same list once, an assertion refuses a length that does not
match the number of runs, and a test pins it.

**AMENDMENT — EXPLORATORY 128 000 BUDGET PROBE, 2026-08-14.** Written and committed before
any 128 000 run exists.

*Standing.* An **exploratory development probe**. Not a hypothesis test, not a gate, not a
main result, and **not a cap in the frozen ladder**. It cannot select a primary cap and it
cannot reopen the failed C1 budget gate. Its only job is to decide whether 128 000 deserves
to enter the held-out design at all.

*The question.* The 64 000 diagnostic showed a large gap between C1 and C3. One alternative
account remains open: that 64 000 is simply not enough for a single ReAct trajectory, and
that at 128 000 C1 would improve, close the gap, or overtake C3. The probe measures that
directly, paired within instance: **C1 at 64 000 → 128 000** and **C3 at 64 000 → 128 000**.

*Matrix.* 12 instances, cap 128 000, conditions C1 and C3 only, **24 new runs**. The
64 000 arm already exists and is read from the completed development logs; no run is
repeated. C2, C4 and C5 are not run at 128 000.

*The 12 instances, and the rule that chose them, fixed before the first 128 000 run.* Per
band, the first two instances with oracle optimum 3 and the first two with optimum 4, in the
canonical order of the frozen budget-dev subset. Position and structural facts only.

> **Forbidden as selection inputs:** C1 or C3 satisfaction, feasibility, termination, token
> usage, or any other agent outcome from the completed 64 000 runs. Choosing the probe set
> by how the conditions already did on it would turn the 64 000 arm of a paired comparison
> into a chosen baseline instead of a measured one.

*Why the optimum is balanced 2:2 rather than left to fall where it may.* Satisfaction is
`achieved / O`, so one meeting is worth 0.333 at `O = 3` and 0.25 at `O = 4`. A band that came
out all `O = 3` beside one that came out mostly `O = 4` would measure the same improvement on
different scales and the per-band rows would not be comparable. It is also the discipline the
complexity bands were frozen under in §3, kept here rather than loosened.

| band | instances |
|---|---|
| Low | `random-n8-t80-o100-s30000`, `random-n8-t80-o80-s30002`, `clustered-n8-t60-o100-s30001`, `clustered-n8-t60-o100-s30004` |
| Medium | `line-n8-t90-o80-s30039`, `line-n8-t80-o100-s30040`, `uniform-n8-t90-o50-s30036`, `clustered-n8-t90-o20-s30041` |
| High | `random-n8-t90-o80-s30015`, `clustered-n8-t90-o80-s30017`, `uniform-n8-t100-o50-s30024`, `uniform-n8-t90-o50-s30027` |

*Provenance.* Frozen probe subset
`results/manifests/subset__budget_probe_128k__76f3bc0f0654.json`, content hash
`76f3bc0f06547a7d0fa1593c62bd207c0c027d94324030bf10b35a198eea6100`, cut from parent subset
`911e4864d6fb` and pointing at the same binning manifest `8e671cd13dbb`, so every band label
is re-derived from the same frozen boundaries. Every entry is **copied verbatim** from the
parent — same ids, seeds, optima and complexity metrics — so the probe runs on literally the
same task objects the 64 000 arm ran on, which is what makes the comparison paired. Expected
runs `expected_runs__budget_probe_128k.json`, content hash `a7b274f876b8`, 24 runs. Outputs
go to a **new** directory, `results/logs/budget_probe_128k/`; the completed C1–C5 development
logs are not written to.

*One note on that hash, recorded so it is never read as tampering.* The expected-runs writer
was fixed after this file was frozen — it had been ignoring `--instance-ids`, `--levels` and
the shard split — and the fix adds a `filters` block to the document. Regenerating the probe
manifest with the fixed writer produces the **identical 24 run ids and counts** but the hash
`a72650a67866`, because the new field enters the hashed payload. The frozen file is **not**
regenerated: `a7b274f876b8` remains the identifier of the document this probe was registered
against. The probe itself never depended on the defect — the runner was handed a
12-instance subset directly, so no filter was involved.

*And the fix does not touch the probe's runtime.* It changes only how an offline manifest is
written; the execution path of C1 and C3 is untouched. The C1 and C3 arms of the probe are
nevertheless both run on the same runtime commit, so their scientific provenance is
identical regardless.

*128 000 is a workflow budget, not a context window.* Verified against the implementation
rather than assumed: the guard grants `remaining − reserve − input` and the client
independently clamps each call to `max_model_len − input`, leaving the unspent grant in the
budget. `max_model_len` stays **32 768**. A larger cap therefore buys **more calls, not
longer ones**, and in C3 the whole system shares the 128 000 — each worker's quota is
`floor(W · 3/8) = 47 904`, not 128 000 apiece.

*Context headroom, measured.* Across 1 664 calls at 64 000 the largest single prompt was
**5 440 tokens** against the 32 768 window, with the 99th percentile at 2 727. A prompt would
have to grow roughly sixfold to exhaust the window; doubling the budget is expected to
roughly double it. `ContextWindowError` is raised before any request and is **not caught
anywhere**, so exhaustion would stop the sweep loudly rather than silently — recoverable,
because writes are atomic and resume is strict.

*What is expected, recorded so it cannot be claimed afterwards as a prediction that came
true.* At 64 000 C1 already leaves about a quarter of its budget unspent, with mean cap
utilisation 0.76 and 38 of 60 runs ending voluntarily rather than on budget. On that evidence
the most likely outcome is that little changes. **No outcome is treated as desirable in
advance**, and a result that closes or reverses the gap would be the more informative one.

*Reading.* The primary metric is **satisfaction**. The validated non-empty rate is secondary
and diagnostic and is not to be called performance or accuracy. **No numerical gate and no
threshold is introduced**: twelve instances support description, not inference. The probe
reports the paired change per band and pooled, together with how many runs at 128 000
actually spent more than 64 000 tokens — and how many finished under 64 000 with 128 000
available, which answers the question on its own if the extra budget goes unused.

*What stays untouched.* C1 and C3 architectures, prompts, tools, hidden validator, answer
contract, model, tokenizer, budget accounting, finalisation, generation settings, complexity
bands, `n = 8`, existing development results, server configuration and the prefix-caching
policy. The only change is the total workflow cap and the twelve-instance subset.

**EXTENSION — C5 JOINS THE 128 000 PROBE, 2026-08-15.** Written and committed before any
C5 run at 128 000 exists. It extends the amendment above, which registered C1 and C3 only.

*Why, and why the reason is not a reaction to what C1 showed.* The C5 result at 64 000 was
recorded as **consistent with budget starvation but not established as caused by it**, and
the record named what would settle it: whether an attempt can reach a proposal when it is
adequately funded. That limitation was written on 2026-08-13, before a 128 000 run existed.
Doubling the cap is the direct test.

| | attempt quota | share of a full C1 trajectory (48 673 mean) |
|---|---|---|
| cap 64 000 | 21 248 | **44%** |
| cap 128 000 | **42 581** | **87%** |

The regime changes qualitatively. C1's own runs at 128 000 consumed between roughly 34 000
and 69 000 tokens, so an attempt quota of 42 581 sits **inside** that range: at this cap an
individual attempt can plausibly finish, which at 64 000 it could not.

*What it can settle.* If C5 improves sharply, starvation was the cause and the recorded
caveat becomes a demonstrated mechanism. If it stays on the floor, the cause lies elsewhere
and the statement "this says nothing about best-of-N in general" has to be replaced with
something more definite. Separately, the final-product diversity metric becomes interpretable
for the first time: at 64 000 a single distinct product could simply mean three attempts that
produced nothing, whereas at 87% funding a convergence to one product would be an observation
about sampling rather than an artefact of starvation.

*Matrix.* The identical 12 frozen instances, cap 128 000, condition `c5_best_of_3`, 12 runs,
into the same `results/logs/budget_probe_128k/`. The probe becomes 36 runs in total —
12 C1, 12 C3, 12 C5 — each paired against its own completed 64 000 arm. Frozen expected-runs
manifest `expected_runs__budget_probe_128k_c5.json`, content hash `473f87ea5cc4`. The
original 24-run manifest is untouched.

*Provenance.* All three arms run on the same runtime commit `fb597ad`; this extension is
committed before the C5 run but is **not** pulled into the run machine, because `c5_best_of_3`
already exists there and the document is not needed to execute it. Record predates the run
and the runtime stays identical across arms.

*Standing is unchanged.* Still exploratory, still descriptive, still not a gate. No threshold
is introduced, satisfaction remains the primary metric, and the validated non-empty rate stays
secondary and diagnostic. Held-out is not generated, seeds 100 000+ are untouched, and whether
128 000 joins the held-out design remains a separate later decision.

*And what is not decided by it.* Held-out is not generated, seeds 100 000+ are not touched,
no new cap enters the ladder, and the failed C1 budget gate remains failed. Whether 128 000
joins the held-out design is a separate dated decision taken after these 24 runs.

**FROZEN — THE C1–C5 DEVELOPMENT DIAGNOSTIC IS CLOSED, 2026-08-13.** 300 runs over the
identical 60 frozen instances at `n = 8`, cap 64 000, one model, one serving configuration.
No further architectural development experiment is run.

*Status of everything below, stated once and binding on the thesis text.* This is
**development, descriptive evidence**. It is not a hypothesis test, not a confirmatory
result, and not the primary result of the thesis. The development seeds are consumed for
exploration by construction. Twenty instances per band at one cap support description and
ordering, not inference.

| condition | validated non-empty rate | mean satisfaction | mean tokens | cap used |
|---|---|---|---|---|
| C1 ReAct | 0.38 | 0.300 | 48 673 | 0.76 |
| C2 verify/revise | 0.55 | 0.418 | 55 141 | 0.86 |
| C3 hierarchical MAS | 0.73 | **0.643** | 45 274 | 0.71 |
| C4 planner + fresh critic | 0.75 | 0.489 | 45 160 | 0.71 |
| C5 Best-of-3 | 0.10 | 0.079 | 62 905 | 0.98 |

**Observation 1 — the `n = 8` floor is not task-wide.** C1 reaches a validator-approved
non-empty plan in 0.45 / 0.45 / 0.25 of runs by band; C3 and C4 reach 0.70–0.80 on the same
instances at the same budget with the same model. The failed budget gate identified a floor
belonging to the plain single-trajectory architecture, not to the task at this instance size.

**Observation 2 — a fresh independent critic is strongly associated with reaching a valid
plan, but C3 leads on the primary metric.** C2 → C4 moves the validated non-empty rate from
0.55 to 0.75 and clears the reference rate in every band without any decomposition. On
**satisfaction**, which is the primary metric, the order is C3 0.643 above C4 0.489: C4 is
not the leading condition on performance and may not be presented as one.

**Observation 3 — the split between reaching a plan and reaching a good one is a candidate
explanation, not a causal conclusion.** Conditional on producing a plan, C3 reaches 88% of
the oracle optimum against C4's 65%; C4's mean pool is 2.87 people. The candidate
explanation is that a fresh critic may govern whether a valid plan appears while
decomposition and parallel search may govern how good it is. C3 and C4 differ in the number
of search agents, the pool rule and the critic's information regime as well as in
decomposition, so neither half is isolated by this design.

**Observation 4 — repeated independent search does not explain the C3 advantage at this
budget.** Under the same 64 000 cap, three independent C1-style attempts scored 0.079 against
C3's 0.643 while consuming 98% of the budget, and the selection among attempts changed the
outcome in at most 4 of 60 runs. The simple alternative account — that C3 is ahead merely
because it searches more than once — is not supported. This says nothing about best-of-N in
general: each attempt received 21 248 working tokens against a 48 673 mean for a full C1
trajectory, so the diagnostic varies attempt count and attempt depth together.

*What this does not settle, and what comes next.* The C1 budget gate remains failed and no
primary cap is selected. Nothing here licenses a crossover claim, which needs the held-out
design and the registered intersection–union test. The next work is the failed gate and the
held-out design, not another development run.

### Two findings in the same run that qualify how the conditions may be read

**C3 does not dominate C1 on the binary outcome.** Paired on the identical instance, C3
recovers 28 runs on which C1 produced no validated non-empty plan and **turns 7
C1-successful instances into failures**, with 25 unchanged; mean Δ satisfaction +0.343. C2
recovers 10 and turns none into a failure; mean Δ satisfaction +0.118.

*Read this table for exactly what it counts.* It counts transitions between validated
non-empty plan and no such plan. It does **not** establish that C2's satisfaction is at least
C1's on every instance, and no claim that C2 is strictly better than C1 is licensed by it.
The permitted wording is: *C2 recovered 10 instances on which C1 failed and did not turn any
C1-successful instance into a failure*, with the mean improvement reported separately.

On C3 the asymmetry is the substantive point: decomposition has a cost as well as a benefit,
and the cost is not hypothetical — on about one instance in nine the split appears to
separate people a single agent had managed to schedule together. The candidate pool cannot recover such a case, because a person neither
worker put in its sub-plan never reaches the critic. This must be reported alongside the
headline rate, not folded into it.

**C2's mandated structure was frequently never administered.** Of 60 runs the verify stage
was reached at all in 35 and twice in 27; revise in 32 and twice in 24. Only **24 of 60 runs
completed both full verify/revise cycles**, and in **25 runs the explicit verify stage was
never reached**. Cap utilisation is 0.86, the highest of the three, so the draft phase is
consuming the budget the later stages needed.

*Stated as the measured fact and no further.* Those 25 runs are **not** described as "C1 with
a different label": C2's draft phase runs under its own graph and its own routing, so calling
them C1 would assert an execution-path identity this diagnostic did not check. What was
measured is that explicit verification was not reached.

The cause is structural and worth stating plainly: **C2 is the only structured condition
whose later stages have no reserved budget.** C3 guarantees its critic at least `W/4` by
quota, C4 does the same, and C5 splits `W` into three protected thirds; C2's verify and
revise get whatever its draft phase happens to leave. At `n = 8` that is often nothing.

*This is not repaired.* C2 is locked (§4) and was run this way in the formal pilot, so
changing it now would break comparability with everything already collected. It is recorded
as a **treatment-fidelity limitation**: any C1 → C2 contrast at this instance size measures
"mandated verification, administered in roughly half of runs", not "mandated verification".
The stage-reach counts must be reported with any such contrast, and the analyser already
computes them.

### The 128 000 budget probe — result

**FROZEN — THE 128 000 DEVELOPMENT PROBE IS COMPLETE, 2026-08-16.** 36 runs at cap 128 000
over the 12 frozen probe instances (4 per band, `n = 8`), conditions C1 / C3 / C5, paired
against the existing 64 000 runs on the identical instances. Subset manifest
`subset__budget_probe_128k__76f3bc0f0654`, selected before any 128 000 run existed and
selected without reading any agent outcome. The paired completeness audit passed: 72 runs,
12 instances, both caps present for every condition.

*Standing, restated because it binds the wording.* Exploratory and descriptive. Not a
hypothesis test, not a gate, not a cap selection. Four instances per band support description
and ordering, not inference. No threshold is introduced; whether 128 000 enters the held-out
design remains a separate later decision. Held-out is not generated and seeds 100 000+ are
untouched.

| condition | sat 64k | sat 128k | Δ | tokens 64k | tokens 128k | cap used 128k | runs above 64 000 |
|---|---|---|---|---|---|---|---|
| C1 ReAct | 0.278 | 0.278 | +0.000 | 43 594 | 46 967 | 0.37 | 4/12 |
| C3 hierarchical MAS | 0.688 | 0.778 | +0.090 | 42 975 | 62 168 | 0.49 | 5/12 |
| C5 Best-of-3 | 0.083 | 0.479 | +0.396 | 62 363 | 112 088 | 0.88 | 12/12 |

**Result 1 — doubling the cap does not move C1, and the probe is able to detect it if it
did.** C1's mean satisfaction is identical at both caps, no run terminated on budget at
128 000 (11 of 12 `agent_finish`, 1 aborted), and mean spend rose only to 0.37 of the cap.
The simple alternative account of the failed budget gate — that 64 000 is merely too small
for a single ReAct trajectory at `n = 8` — is **not supported**.

This null carries weight only because the same twelve instances, the same analyser and the
same serving configuration returned +0.396 for C5 and +0.090 for C3. The instrument
demonstrated that it registers a budget effect where one exists, and registered none for C1.

**Result 2 — C1 has four informative instances here, not twelve, and this was predicted
before it was checked.** Per call the agent is granted
`min(remaining − reserve − input, max_model_len − input)`; at both caps the 32 768 window
clamps every call, so a larger cap buys more calls and not longer ones, and the two runs
execute an identical configuration until the 64 000 one exhausts its budget. Eight of twelve C1
runs never reached that point and reproduced their token totals and call counts exactly. The
prediction of which four would diverge was made in advance and matched 4/4 and 8/8.

On all four that did diverge the extra budget bought a voluntary finish and nothing else:
`s30004` 63 268 → 68 800 tokens, budget → agent_finish, satisfaction 1.00 → 1.00; `s30040`
63 855 → 83 586, aborted → agent_finish, 0.67 → 0.67; `s30000` 63 888 → 73 615, aborted →
agent_finish, 0.67 → 0.67; `s30036` 63 844 → 69 337, budget → agent_finish, 0.00 → 0.00. C1
converted extra budget into termination quality, not into plan quality. Any wording built on
C1's null must carry the count four, not twelve.

**Result 3 — C3 rises modestly and uses roughly half the larger cap.** +0.090 pooled, 3
instances improved and none worsened, validated non-empty 10/12 → 11/12, every run
`agent_finish`. Ten of twelve instances are informative for C3 against four for C1, so the
two nulls are not equally powered and the C1 and C3 columns must not be read as a like-for-
like comparison of sensitivity.

**Result 4 — the C5 collapse at 64 000 was an artefact of dividing the budget, not a property
of repeated sampling.** The caveat recorded on 2026-08-13 — that the diagnostic varied
attempt count and attempt depth together, and said nothing about best-of-N in general — is
confirmed by direct measurement. At `W/3` each attempt received 44% of a full C1 trajectory
and C5 scored 0.083; at 128 000 it receives 87%, and C5 reaches 0.479 with validated
non-empty 1/12 → 7/12, six instances improved and none worsened, at 0.88 of the cap.

The selection mechanism was inert at the smaller cap and active at the larger one. At 64 000
only 6 of 60 runs produced more than one distinct final product and the winner was attempt 0
in 56 of 60 — which is mostly the tie-break firing on equal or empty attempts, not evidence
that the first attempt was better. At 128 000, 7 of 12 runs produced two or three distinct
products and 7 of 12 winners were a later attempt, and the rule promotes a later index only
on a strictly larger meeting count. Selection therefore changed the outcome in 7 of 12 runs.

*What this does not decompose.* Attempt depth and a working selection stage improved
together, so +0.396 may not be attributed to choosing among three attempts or to attempts
finally reaching a proposal. This probe does not separate them, and Observation 4 of the
closed C1–C5 diagnostic is qualified accordingly rather than withdrawn: at 64 000 repeated
independent search did not explain the C3 advantage, and that statement stays bounded to
64 000.

*Ordering, for completeness.* C5 at 128 000 (0.479) remains below C3 at 64 000 (0.688), and
the C3 − C1 gap widens from 0.410 to 0.500 when the cap is doubled. Both are descriptive
statements about twelve instances.

**Result 5 — a determinism check that came free.** Where the two caps cannot diverge — a run
that finishes before 64 000 is exhausted is executing an identical configuration at both caps
— the harness reproduced token totals and call counts **exactly**: 8 of 12 C1 runs and 2 of
12 C3 runs, ten in total, across sweeps executed days apart, on different commits, after a
vLLM restart. Prefix caching off, seed 42 and a fixed serving configuration reproduce to the
token at this scale. This is a property of the harness, recorded because it was observed in
passing, and it is not a claim about the model's behaviour in general.

*Provenance.* C1 and C3 ran at commit `fb597ad`, C5 at `1b6fc92`; the difference is confined
to offline manifests and analysers and touches no agent or serving code. The analyser reports
the commit per condition rather than assuming one.

*What is still not decided.* The failed C1 budget gate remains failed and no primary cap is
selected. No new cap enters the ladder. Whether 128 000 joins the held-out design is a
separate dated decision. The next work is the failed gate and the held-out design.

---

**AMENDMENT — HELD-OUT DATASET DESIGN FROZEN, 2026-08-26.** Recorded before any seed
in the held-out range has been generated, inspected or read. Nothing below was chosen
from held-out data, because no held-out data exists yet.

*Seed sub-ranges.* The locked held-out range `100000-199999` is partitioned once, one
reserved block per pool, never shared. The reason is the one already recorded for the
lower-complexity ranges: a pool is identified by its seeds alone in every downstream
artifact, so two pools drawing on one block would make `instance_id` collisions possible
and would stop the seed from identifying which pool a candidate came from.

| block | range |
|---|---|
| `held_out_n8` | `100000-109999` |
| `held_out_n4` | `110000-119999` |
| `held_out_n5` | `120000-129999` |
| `held_out_n6` | `130000-139999` |
| unused reserve | `140000-199999` |

*Block A, `n = 8`.* **50 instances per D band**, which is the number the 2026-08-09 design
registered, not the 30 that the superseded scheduling estimate assumed. The shared
oracle-optimum histogram is fixed **now, before the pool exists**:

    O = 3 : 32     O = 4 : 18     total : 50

That is the largest-remainder scaling of the calibrated capacity 41/24 to 50. The
paragraph above this amendment named 32/18 and explicitly declined to adopt it, on the
grounds that choosing it after seeing the held-out pool would be a free parameter. Fixing
it here, before the pool is generated, is the condition that paragraph asked for.

*Block B, `n = 4, 5, 6`.* **16 instances per size, every one at `O = 3`**, under the
selector already used for the frozen lower-complexity extension. `n` remains a generator
parameter and does not become a complexity metric.

*Resulting matrix.*

    Block A : 3 bands x 50            = 150 instances
    Block B : 3 sizes x 16            =  48 instances
    total                             = 198 instances
    198 x 5 conditions x 4 caps       = 3960 runs

*Consequence for scheduling.* The 24-shard / %6 / 20 h freeze of 2026-08-26 was computed
for 2760 runs and is **superseded**. The replacement is computed after the expected-run
matrix is built, not before.

*Selector determinism, corrected here.* `scripts/build_band_manifest.py` accepted a
`FEASIBLE` result alongside `OPTIMAL` under a 120 s limit. Its objective is "earliest
admissible selection in canonical order", so a run that hit the limit would have returned
some admissible selection rather than the earliest one, and a loaded node would have
produced a different manifest from an idle one with identical code, pool and seed. A
timeout is now a loud failure. Strata, tie-breaks and diversity were audited at the same
time and contain no discretionary choice: pools are canonically sorted, the histogram
scaling breaks ties on remainder then optimum value, candidate histograms are ordered by
distance then lexicographically, and the CP-SAT model runs single-threaded at a fixed
seed with a rank objective.

*Still open, and still blocking the held-out RUN rather than this manifest.*

1. The power analysis for 50 per band. It may **not** come from the synthetic
   intersection-union simulator: the amendment of 2026-08-10 deregistered that route, and
   `results/analysis/power_iut_v2/STATUS.md` records that neither `N = 35` nor `N = 95` is
   adopted. The registered route is empirical -- spread, zero rate and tie rate taken from
   the real development runs, which now exist.
2. Whether the anticonservatism found by the corrected Type-I validation reproduces on
   real development data. The registered response if it does is that the test changes and
   `N` does not.
3. The statistical pre-registration in full: primary cap(s), the primary architecture
   contrast, how `Architecture x D-level` is tested, what counts as a crossover rather
   than a monotone trend, the multiplicity policy, whether C2 is primary or secondary, and
   the pre-declared `H` sensitivity analysis.

### CLOSED 2026-08-27 — all three, before any held-out seed was inspected

1. **Closed.** The empirical route was taken, not the deregistered simulator:
   `results/analysis/empirical_power/` estimates spread, zero rate and tie rate from the
   real development runs. `N = 50` per `n = 8` D-band stands as the main-design target.
   A further crossover-specific analysis exists in `results/analysis/interaction_power/`;
   its `{50, 60, 65}` selection rule was **withdrawn as over-engineering for this scope**,
   not applied. No further power analysis is to be run.
2. **Closed, and it reproduced.** The percentile bootstrap exceeded nominal 0.05 in six of
   seven cells on real paired differences, worst where ties were heaviest. The registered
   response was followed exactly: the test changed and `N` did not. The bootstrap is
   excluded as primary; the sign-flip permutation test on the mean replaces it.
3. **Closed** by `results/analysis/interaction_power/PRE_ANALYSIS_PLAN.md` and its two
   amendments of 2026-08-27. Primary cap 64000, the other three run and reported as
   pre-declared secondary. **C1-vs-C3 is *not* the primary contrast** — that framing was
   withdrawn. The confirmatory family is five architecture-motivated contrasts
   (C1→C2, C2→C4, C4→C3, C1→C5, C5→C3), each with an interaction test and a crossover
   IUT, Holm across the ten p-values; all ten pairwise contrasts are reported as the
   complete comparison matrix. C2, C4 and C5 are **not** secondary. Medium band
   descriptive. Estimand fixed as the mean paired difference; sign test is sensitivity.
   Crossover stays confirmatory with its limitation pre-declared: failure to establish
   crossover is **not** evidence that no crossover exists.
   **[SUPERSEDED 2026-09-14 as the primary analysis]** — this cap-64 000 family was run
   (`b84959f`) and stays unchanged as a registered analysis reported in an appendix; the
   primary analysis is the all-budget factorial analysis of the 2026-09-14 amendment.

These gates no longer block the RUN. The RUN itself still requires an explicit GO.

**AMENDMENT — THE MAIN DESCRIPTIVE ANALYSIS USES THE FULL EXPERIMENTAL MATRIX,
2026-09-02.**

*Status, stated first because it governs how everything below may be read.* This is **not a
pre-registration**. It is recorded **after** the frozen confirmatory analysis at cap 64 000
had been run and read. That analysis returned no Holm rejection among its ten p-values and
established no crossover (`results/analysis/heldout/report.md`, commit `b84959f`). Nothing
here may be presented as having been decided before that result was known.

This amendment introduces **no hypothesis, no p-value, no multiple-comparison family and no
decision about statistical significance.** It fixes the scope of the descriptive
presentation, and nothing else.

*History, recorded because the sequence is what makes the scope defensible.* The approved
exposé treated the budget ladder as an axis of the experiment and specified analysis across
it, not at a single operating point. The scope was narrowed to one primary cap later — the
budget policy of 2026-08-09 and the pre-analysis plan of 2026-08-27 — under an expectation
of limited compute, so that a confirmatory family would remain affordable. The move to the
TU Berlin HPC cluster removed that constraint, and the experiment was in fact executed in
full: 198 instances × 5 conditions × 4 caps = 3960 runs, every one verified
(`results/analysis/heldout/completeness_audit.json`). The narrowing was therefore a resource
decision that the resources no longer require. The decision recorded here, taken after the
results existed, is to present the full matrix that was actually run.

*What is decided.*

1. The main final descriptive analysis uses the full matrix: **198 instances × 5 conditions
   × 4 budget caps**.
2. All four caps — 16 000, 32 000, 64 000 and 128 000 — carry the **same status** and appear
   on equal footing in every main table and figure. None of them is the primary cap for
   presentation.
3. All five architectures — C1, C2, C3, C4 and C5 — carry the **same status**. C2, C4 and C5
   are not intermediate or secondary conditions, and C1 and C3 are not the two principal
   ones.
4. For each `condition × cap`, results are reported separately for the **Low, Medium and
   High** density bands of Block A.
5. Block B (`n = 4, 5, 6`) is analysed **separately, as a change of task size**. `n` remains
   a generator parameter and is never presented as a complexity band.
6. The reported metrics are **satisfaction, validated non-empty rate, optimality rate,
   realised tokens, model calls, latency and efficiency** (satisfaction per 1 000 tokens).
7. Where pairwise differences are shown, the **complete matrix of all ten pairs** is shown.
   The five contrasts of amendment B2 keep their registered orientation and are not given
   greater prominence in the descriptive presentation.
8. The analysis at this stage is **descriptive**: means, distributions, intervals, tables and
   plots. No new system of hypotheses is created.
9. The frozen confirmatory analysis at cap 64 000 is **preserved unchanged and is not
   overwritten**. It remains a separate registered analysis, part of the audit trail, and is
   reported as such.
10. The all-budget outputs are written to **their own directory**,
    `results/analysis/heldout_full_matrix/`, separate from the frozen confirmatory outputs in
    `results/analysis/heldout/`.

*What this does not do.* It does not withdraw, weaken or reinterpret the confirmatory
analysis: that analysis stands with its registered scope and its registered null result, and
the fact that it found no crossover at cap 64 000 is reported as it stands. It does not
promote any descriptive ordering to a finding. An ordering visible in a descriptive table
describes this sample; establishing it would require a test, and no test is added here.

> **[SUPERSEDED IN PART 2026-09-14]** Items 8 ("no new system of hypotheses") and the
> sentence "no test is added here" are replaced by the amendment of 2026-09-14 below.
> Items 1–7, 9 and 10 stand.

**AMENDMENT — THE PRIMARY ANALYSIS COVERS ALL BUDGETS AND COMPLEXITY LEVELS, 2026-09-14.**

*Decision.* The experimental design is `Architecture × Computational Budget × Task
Complexity`, with the four budget caps 16 000, 32 000, 64 000 and 128 000, as in the approved
exposé and as executed (198 instances × 5 architectures × 4 caps = 3960 runs). The primary
analysis evaluates performance across all three dimensions. Restricting the primary analysis
to one cap (budget policy of 2026-08-09; pre-analysis plan of 2026-08-27, item 3 of the
2026-08-27 closure above) is **superseded**: in the author's judgement it did not match the
three-factor design of the study. Cap 64 000 is **not** a primary or privileged budget.

*What replaces it.*

1. Block A (`n = 8`): primary factorial analysis `Architecture × Budget × Complexity`
   (Low / Medium / High complexity levels), all seven terms, task-level permutation tests.
2. Block B (`n = 4, 5, 6`): inferential analysis `Architecture × Budget`; task size is
   analysed descriptively and is never a complexity level.
3. Holm families, never pooled with each other:
   `blockA_satisfaction_primary` (7 terms, all caps — the factorial analysis);
   `blockA_satisfaction_no16k_sensitivity` (7 terms, 32k/64k/128k — sensitivity only, checks
   whether the conclusion rests on cap 16 000; it never replaces the primary result);
   `blockA_valid_nonempty_secondary` (7 terms — secondary outcome);
   `blockB_satisfaction` (Architecture, Budget, Architecture × Budget);
   `blockA_satisfaction_pairwise` (10 pairs × 4 caps = 40 tests, pooled over complexity
   levels). No pairwise family for the valid non-empty solution rate and no no-16k pairwise
   family. Any later localising follow-up is labelled exploratory and added only by decision.
4. Primary p-values: task-level sign-flip / label permutation, 99 999 permutations, fixed
   seed, `p = (b + 1) / (B + 1)`. A linear mixed model with a task random intercept is a
   supporting check only. Sensitivity: the same analysis without cap 16 000.
5. Effect description: mean paired difference `Δ_{X−Y} = S_X − S_Y` with the order always
   written, share of tasks better / equal / worse, paired Cohen's d as secondary. Mean
   intervals: task bootstrap, 10 000 resamples, fixed seed.
6. Secondary outcome: valid non-empty solution rate, permutation tests and rates only.
7. Code and outputs: `scripts/analyse_heldout_factorial.py` →
   `results/analysis/heldout_factorial/`. No held-out result, manifest or frozen output is
   modified.

*Standing, stated honestly.* The three factors are part of the design and the research
question. The **specific statistical procedures** listed above — the permutation scheme for
the factorial terms, the Holm families, the mixed-model check and the follow-up comparisons —
were specified **after all runs were complete, and after the descriptive results and the
registered cap-64 000 analysis were available**. They are not pre-registered and must not be
presented as such. The registered cap-64 000 analysis (`results/analysis/heldout/`, commit
`b84959f`) is kept unchanged and is reported in an appendix as a pre-specified analysis that
did not detect the expected crossover; it does not structure the main Results. The earlier
decisions remain in this file and in git history.

---

## 6. Open items (still to decide)

- **OPEN — the second model.** `Qwen/Qwen3-8B-AWQ` was excluded on 2026-08-04 (§1):
  C1 stays at satisfaction 0 across 4k–32k because the model never proposes, so it can
  neither calibrate caps nor support a robustness claim. "Robustness across two model
  sizes" is therefore **withdrawn, not satisfied**. Decide whether a second model
  returns and which — a mid-size checkpoint that does commit to plans (so the
  comparison is about size, not about a floor), or dropping the size axis and stating
  that the study covers one model, with the 8B result reported as a finding about
  small-model behaviour rather than as a robustness arm.
- **OPEN — construct validity of the structural-complexity axis.** On the C1 calibration
  subset the categorical levels show **no stable monotone relation to satisfaction**: at
  cap 64000 the means are easy 0.471, medium 0.600, hard 0.433, and at 32000 they are
  0.400 / 0.267 / 0.233. This is a preliminary descriptive result over **10 instances per
  level per cap**, on a calibration-exposed subset, and it is not sufficient to declare
  the axis broken.

  It must not be read as "the metric measures difficulty for the solver rather than for
  the agent". `complexity_metric` measures **structural pairwise incompatibility** -- the
  number of person pairs each reachable alone but not in either order -- and says nothing
  about CP-SAT's effort.

  The likelier reading is an interaction with the normalisation. Satisfaction is
  `achieved / oracle optimum`, and the optimum itself falls as conflicts rise: measured on
  these runs, Spearman(complexity, optimum) = **−0.778**, with mean optimum 5.50 / 3.60 /
  2.00 across easy / medium / hard. In **absolute meetings** the ordering is the expected
  one -- at 64k the means are 2.10 / 2.00 / 0.90, and Spearman(complexity, achieved) is
  −0.20 -- while Spearman(complexity, satisfaction) is −0.00. So the agent does do worse
  on more conflicted instances; the ratio hides it because the denominator shrinks faster
  than the numerator.

  Before anything is concluded about the axis, the comparison across levels must be made
  on all of: oracle optimum, absolute meetings achieved, satisfaction, optimality gap,
  proposal validity, and the share of runs with at least one valid proposal. The primary
  analysis should use the **continuous** complexity metric and control for `n_people`,
  travel structure, `tightness`, `overlap` and oracle optimum -- in the formal subset,
  level is nearly determined by `tightness` and the hard level has no `n_people=4` at all
  (§3), so the categorical contrast is confounded by construction. Categorical levels are
  kept as pre-registered strata; they are **not** declared a validated scale of LLM
  difficulty.

  **Formal pilot, 2026-08-07, corrected the same day; the item stays open and is now the
  gating decision for the main experiment.** A first reading of these runs claimed the
  axis orders the condition difference monotonically in the direction opposite to the
  research question. It does not support that claim. With `Δ = S_C1 − S_C3`, the level
  means at cap 64 000 are −0.460 (easy), −0.457 (medium), −0.142 (hard), and the interval
  excludes zero on easy and medium but on hard at no cap. Tested three ways — Kruskal-
  Wallis omnibus, Jonckheere-Terpstra for a monotone ordering, and the easy-versus-hard
  contrast — nothing supports the registered alternative: the one-sided p for the
  **falling** trend H1 and H2 predict is 0.835 at 32 000 and 0.935 at 64 000, because the
  estimates move the other way. The opposite, rising direction is nearer conventional
  thresholds (one-sided 0.171 and 0.068) but is **exploratory**, since no one registered
  it. Ten instances per level gives power only against a large effect, so **the point
  estimates trend against the pre-registered crossover while the pilot provides
  insufficient evidence to establish any trend**, and neither "the hierarchy helps least
  where the instance is most conflicted" nor "there is no complexity threshold" may be
  written as a result.

  What is established is narrower and more damaging to the design: the hierarchy leads on
  the *easiest* level with interval support, which is the condition the expose names as a
  sign that the measurement, the setup or the evaluation protocol is at fault. Combined
  with the falling optimum (5.5 / 3.3 / 2.5, so one meeting is worth 0.40 on hard against
  0.17 on easy), the absent `n_people=4` stratum at the hard level, and level being nearly
  determined by `tightness`, the axis cannot be assumed to order difficulty. Deciding its
  operationalisation — a density such as incompatible pairs over `C(n,2)`, or stratifying
  complexity within `n_people`, or balancing optimum across levels — must happen and be
  recorded **before** the held-out main experiment runs, and the choice must be justified
  on the pilot's diagnostics rather than on which option produces the expected pattern.

  **DECIDED 2026-08-08 — see §3, "REVISED COMPLEXITY AXIS FOR THE MAIN EXPERIMENT".** The
  axis becomes normalised conflict density at a fixed `n_people = 8`, with the oracle
  optimum as a matching variable and higher-order interaction recorded as a diagnostic.
  Instance size moves to a secondary robustness analysis inside the oracle's proven range.
  The diagnosis above stands as the evidence that motivated the revision; what remains open
  is only whether the candidate pool can satisfy the admission criteria, which the
  structural calibration answers and whose fallback is recorded in advance.
- **OPEN — travel structure is the strongest association measured so far, and only an
  association.** On the calibration subset, uniform instances score far above clustered
  ones (at 64k: 0.648 vs 0.356 satisfaction, 0.44 vs 0.23 proposal validity). Re-running
  two clustered instances on the second GPU reproduced them bit for bit, which establishes
  that the split is not an artefact of the shard/GPU assignment -- the shards happened to
  align exactly with travel structure. It establishes nothing about causation: uniform and
  clustered cells also differ in optimum, complexity and instance composition, and the
  repeat run is **not** an additional sample. The set remains 120 runs, not 240. A causal
  claim needs matched parameters and seeds, or a model controlling `n_people`, tightness,
  overlap, complexity and optimum.
- **Mechanism-based pre-pilot expectation for C3 (recorded 2026-08-04, NOT a new
  primary hypothesis).** This arose *after* looking at the first C1 calibration runs,
  so it is recorded as an expectation with a stated mechanism, not as an independent
  hypothesis the study was designed around — the RQ and the primary metric are
  unchanged. The observation: C1's failures on the 32B are almost entirely
  `travel_infeasible`, i.e. the agent gets the chain of travel constraints wrong across
  a four-meeting route. A C3 worker plans a two-person sub-instance, so it has to
  satisfy fewer coupled travel constraints in one plan. If that is what drives the
  failures, worker proposals should show a higher validity rate than C1 proposals at
  the same cap, and the effect should grow with instance size. The proposal taxonomy
  measures exactly this, so the expectation is falsifiable on data the pilot already
  collects. It must be reported as post-hoc-motivated whatever the outcome, and a
  confirmation is not evidence for the mechanism unless the validity rate moves in the
  predicted direction *and* `travel_infeasible` is the reason that shrinks.

  **Measured on the formal pilot (2026-08-07), and DOWNGRADED to a diagnostic on the same
  day.** The first reading claimed confirmation from a pooled validity of 0.843 for C3
  against 0.372 for C1 at cap 64 000. That figure pooled the critic with the workers.
  Split by role, C3 at 64 000 is worker_a 0.815 (27 attempts), worker_b 0.704 (27) and
  critic **1.000 (29)** — the critic emits only what already passed the aggregator gate,
  so it is valid close to by construction. Workers alone reach 0.759. The
  `travel_infeasible` share per attempt still falls sharply, and C3 still spends fewer
  tokens than C1 at the same cap.

  Even corrected, this is **not** evidence for the mechanism, for a reason that has
  nothing to do with the critic: a C1 proposal addresses the whole task and a C3 worker
  proposal addresses a decomposed sub-instance, so the two are not the same measurement
  and the comparison is not apples to apples in either direction. Proposal validity is
  therefore reported per role as a description of internal behaviour and is not used in
  the text as the reason C3 outscores C1. Establishing what causes the difference is what
  C4 is for; until it exists, the mechanism stays an interpretation.
- **CLOSED 2026-08-04 — the empty-post-think rule stays as locked.** The retry
  alternative was implemented behind `retry_on_empty` (default OFF) and measured A/B on
  the 32B, both arms over the same instances, caps 16000 and 32000, five seeds each
  (`scripts/run_empty_retry_ab.py`). Result: no evidence for a change, on two counts.
  Where the retry actually fired — both arms recording an empty turn — it changed
  nothing: same score (0.00) and the same 15 844 tokens, only the termination label
  moved from `aborted` to `budget`. The reason is that the empty turns in that sample
  all landed on the last affordable call (`unused` 156 tokens), so the retry had
  nothing left to spend. And the one apparent rescue (cap 32000, 0.00 → 0.75) came from
  a seed where NEITHER arm recorded an empty turn, i.e. where the two arms run
  byte-identical code — it was cache-driven run-to-run variance, since that A/B
  predates the prefix-caching decision above. The motivating observation (one empty turn
  discarding 41% of a budget) did not reproduce: the same cell later ran to 31 579
  tokens with no empty turn at all. The switch is kept in the code, off, so the measurement
  can be repeated in the deterministic regime if the question returns.
- **[PILOT]** Final budget cap values (§2) and easy/medium/hard boundaries (§3).
- **[PILOT]** C3 worker share `s` (§4 C3-5): default 3/8 of the working budget per
  worker; confirmed (or re-set once) on the pilot, then frozen for the main run. The
  mechanism (equal quotas, downstream-only leftovers, critic under the plain guard) is
  locked and not pilot-dependent.

  **Evidence from the formal pilot, decision still to be taken.** At 8 000 and 16 000
  tokens C3 scores zero and does not even spend its grant (5 755 and 11 702 tokens), with
  18 of 30 runs at the lowest cap ending on the agent's own `finish` without a single
  proposal: 3/8 of 8 000 is 3 000 tokens per worker, which is below what one worker needs
  to reach any proposal at all. The quota is therefore the proximate cause of C3's floor
  on the low rungs, and raising `s` would trade critic and aggregation budget for worker
  budget. Whether to re-set it once before the main run, or to keep 3/8 and report the
  low-cap floor as a property of the architecture under a tight budget, is a design
  decision and not a bug fix — and re-setting it invalidates the pilot's C3 numbers as a
  baseline for the main run, so the choice must be made explicitly and recorded here.
- **Tight-cap × thinking interaction:** with thinking ON, the 2k (possibly 4k) cap
  may be spent entirely on the `<think>` block before any plan is emitted, flooring
  scores via an empty `best_plan_so_far`. The pilot must confirm 2k is not
  degenerate (not all cells = 0). Methodologically valid if it is — but must be
  observed, not assumed.
- **`THESIS_SETUP_v3.md`** is referenced by the project notes for the log schema but was
  not located. If it exists, align the run-result schema to it; otherwise it is the
  first Layer-0 file to write.
- **OPEN — where the invalid-plan taxonomy is actually measured.** Surfaced by the
  faithful-serialisation fix (§2), which made an existing property complete: `best_plan_so_far`
  only ever accepts validator-approved plans, and the scored `final_plan` now always equals it,
  so **the scored plan is always valid or empty**. Therefore, measured on `final_plan`:
  `Score.valid` is always True, `invalid_reasons` is always empty, `MALFORMED` is unreachable,
  and feasibility rate is 100% by construction. That is a problem for §5's declared secondary
  metric "invalid breakdown by error type", which §4 C2 calls *load-bearing* for interpreting
  a "C2 ≈ C1" result. Before the fix the metric was only ever populated by finalisation drift —
  i.e. by a bug, not by agent planning errors — so it was measuring the wrong thing either way.
  Decide before the pilot which of these the thesis means, and record it here:
    1. measure the taxonomy over the agent's **rejected proposals** (every `propose` the hidden
       gate refused, labelled by `validate`) — this is what "what kinds of mistakes does the
       agent make" actually requires, and the data is already in the transcript, but it needs a
       small extraction layer and a decision about counting (per proposal, or per run);
    2. keep it on `final_plan` and report honestly that it is trivially empty by construction,
       dropping the "load-bearing" framing in C2's interpretation;
    3. score the agent's last emitted plan instead of `best_plan_so_far` — **rejected**: it
       would re-open exactly the drift the fix closed and make the score depend on finalisation.
  Recommended: option 1. Whatever is chosen, `feasibility rate` in §5 must be redefined
  accordingly (it is not informative on `final_plan`).

  **CLOSED 2026-08-04 — option 1 adopted, and it is implemented.** The first C1
  calibration runs settled it: on the 32B the agent does reach the proposal stage and
  the zeros are *rejected* plans, not absent ones. Across the deterministic 16k/32k
  runs, 12 proposals produced 9 `travel_infeasible`, 2 unparseable and 1 valid — one
  failure mode carries essentially the whole signal, and none of it is visible on
  `final_plan`. Every propose is now logged **at the moment it is handled**, where the
  parse result and the `validate()` verdict are both in hand (`proposal_record` in
  `react_core`, mirrored by C2's revise and C3's critic and workers). It is never
  reconstructed by re-parsing a stored transcript, which would depend on transcript
  formatting and could drift from what the harness actually did.

  Each row (`proposal/1.0`, in the JSON document under `agent.proposals`) carries:
  condition, role, step, the raw proposal text as written, parsed / unparsed, the
  parsed plan, `n_meetings`, `valid`, `reasons` (multi-label — a plan can break several
  hard constraints at once and collapsing them would bias the breakdown toward
  whichever check runs first), and `accepted_into_best_plan`. C3's critic row adds
  `in_pool`, since the subset conjunct is the critic's own gate rather than a
  feasibility verdict and the two refusals must stay apart.

  Counting is per proposal, with the run recoverable by grouping — a run that proposes
  three infeasible plans is not the same evidence as three runs that each propose one.
  Diagnostics only: the score, the hidden gate and everything the agent sees are
  unchanged, and nothing in any loop routes on a row.

  **Naming (fixed 2026-08-04).** The old label `feasibility rate` is retired: it is
  ambiguous about what is in the denominator and about whether an unparseable attempt
  counts. Three separate rates replace it, all over **all proposal attempts** (every
  `Action: propose`, parsed or not), reported per condition × level × cap:
    * `proposal parse rate` = parsed proposals / all proposal attempts;
    * `proposal validity rate` = validator-valid proposals / all proposal attempts —
      **malformed counts as invalid**, so this rate is never inflated by dropping
      unparseable attempts from the denominator;
    * `proposal acceptance rate` = `accepted_into_best_plan` / all proposal attempts.
  Validity and acceptance differ exactly by the strictly-longer conjunct, so reporting
  both separates "the agent cannot produce a feasible plan" from "the agent produced a
  feasible plan that was not an improvement".

  The reason taxonomy is counted over **rejected proposals**, per proposal, with the
  run recoverable by grouping — a run that proposes three infeasible plans is not the
  same evidence as three runs proposing one each.

  **Validity of `final_plan` is trivially 100% by construction** (`best_plan_so_far`
  only ever accepts validator-approved plans and the scored plan equals it). It is
  recorded here as a property of the design and is **not** reported as a substantive
  secondary metric; a 100% figure there says nothing about the agent.
- **RESOLVED 2026-07-26 — sampling baseline: GO.** The literature audit (§8, Parmar et al.
  2025) showed the condition set had no sampling baseline. Decision: add **C5 — Validated
  Best-of-3 single-agent sampling**, specified in §4 C5. Cost accepted: one additional sweep
  column. Implementation is a separate commit after this specification (it needs a new
  `Condition` member, runner dispatch, per-attempt seed/quota handling and tests — the
  existing gate covers only the selection logic).

---

## 7. Confounds to report (Limitations)

- vLLM residual non-determinism even at fixed seed (FP arithmetic / batching).
- **Quantisation regime.** All reported numbers come from AWQ INT4 checkpoints
  (§1, hardware-forced by an Ampere run machine). INT4 compresses more
  aggressively than the originally planned FP8, so absolute satisfaction levels
  may sit below what the same models would reach in FP8 or bf16. The comparison
  across conditions is unaffected — every condition is served by the same
  checkpoint under the same budget — but a floor effect at the hard × 2k cells
  would be partly attributable to quantisation and must be interpreted with that
  in mind.
- The crossover point is not universal — it depends on task, model, prompts and
  implementation; the contribution is the method for finding it, not a fixed number.
- Possible ceiling/floor effects at extreme bins: where C1 is near 100% or near 0%,
  a crossover cannot surface from saturation alone — interpret those bins with care.
- The MAS uses one fixed splitting strategy; other strategies could differ
  (future work).
- The C2 <-> C3 contrast is system-level, not a minimal pair: C3's verification stage is
  one fresh-context critic call, C2's is two in-context revision cycles. "Decomposition"
  in this design includes its coordination apparatus (fixed split, mechanical merge,
  fresh-context review) and is reported as such.
- KV-cache reuse means latency and token-count tell different stories; report both
  and do not equate them.
- **Finalisation mismatch rate is a pilot diagnostic, not a scoring risk (fixed 2026-07-26).**
  The audit found that the code violated the §2 invariant: `finalize_step` used
  `final = emitted if emitted is not None else best`, so ANY parseable emit became the scored
  artefact even if the model had changed a start time, dropped or added a person, or reordered
  the route — guided JSON constrains the *shape* of the emit, not its *values*. Calling this a
  symmetric confound was not sufficient: symmetry does not guarantee absence of bias, because
  conditions produce plans of different length and structure and could therefore drift at
  different rates. **The invariant is now enforced in code** (`emitted` is accepted only when
  it equals `best_plan_so_far` exactly; otherwise the structural `best` is used), so the
  scored semantic output is always `best_plan_so_far` by construction. What remains is a
  measurement: `finalization_mismatch` is logged per run, and the pilot reports its rate as an
  integrity check on the terminal emit. It cannot move a score. This is also why C5 selects
  among structured plans BEFORE finalising (§4 C5).

### Pre-specified limitations from the MAS literature (recorded 2026-07-26, before the pilot)

Recorded BEFORE any run, after a systematic audit of the paper notes in
`thesis/literature/` (§8 records the audit itself). Several of these **weaken the a priori
expectation of a high-complexity MAS advantage and make a no-crossover result theoretically
plausible**; they are stated here so that such a result cannot be read as a post-hoc excuse,
and a positive result cannot be read as having ignored the closest prior work. Hypothesis
labels are deliberately avoided in this block: the canonical definitions live in §5, and the
older paper notes use incompatible numbering (§8).

- **L1 — The task is predominantly sequential, the one structural regime where the closest
  controlled benchmark reports no MAS advantage.** Ke et al. (2026) decompose MAS gains
  along five axes and find MAS ahead on Breadth, Horizon, Parallel and Robustness but *not*
  on **Depth** (the longest chain of dependencies), which their verified summary calls "an
  explicit counterexample"; they attribute it to coordination overhead on strictly
  interdependent steps. Meeting planning is strongly coupled and predominantly sequential —
  the route is a dependency chain in which every kept meeting shifts all later ones — while
  still containing limited decomposable components (per-person fact gathering can proceed
  independently; checking distinct parts of a plan can be split). Consequence, stated
  carefully: Ke et al. give an *a priori reason to expect limited MAS gains on this task*,
  NOT a direct prediction that C3 must underperform C1. A no-crossover outcome is therefore
  theoretically plausible and non-surprising rather than pre-judged. The contribution stands
  either way: it is the *measurement* — an equal-budget threshold method on a formally
  verifiable planning task.
- **L2 — The published mechanism behind "MAS gain grows with complexity" is absent from this
  architecture.** Tang et al. (2025) model the gain as `r·[1−(1−s(w))^N]^d`, where `N` is the
  number of agents attacking the SAME step (debate: several propose, critique, an aggregator
  reconciles) — the gain comes from **redundancy**. C3 has none: two workers take disjoint
  halves, each exactly once, and the critic runs once. Their own limitations state the study
  covers debate-style MAS, "not hierarchical decomposition". Correct use in the thesis: Tang
  et al. show that task complexity can *moderate* the advantage of redundant, debate-style
  MAS; their model does not predict the behaviour of the non-redundant hierarchical C3
  condition used here, and in particular says nothing about it under an equal total token
  budget. Any Introduction sentence implying otherwise must be rewritten.
- **L3 — The design deliberately partitions information, which is NOT MAST's FM-2.4 but
  carries an analogous risk.** Cemri et al. (2025) define FM-2.4 *Information Withholding*
  as an agent possessing relevant information and failing to pass it on, and report it among
  the modes most strongly associated with failed traces. C3 does something different: it
  **partitions** information by design — a worker never receives the other cluster at all,
  so it cannot withhold what it never had, and the critic sees only the candidate pool. The
  distinction matters for correctness of the failure-mode label; the *risk* is nevertheless
  analogous, because relevant cross-partition dependencies may never be recovered during
  aggregation and the critic cannot repair what it never sees. That quantity is exactly what
  the C3-FI ablation (§4 C3-8) measures by lifting the restriction as the single varied
  factor. Report L3 and C3-FI together; never report one without the other.
- **L4 — KV-cache reuse bears on physical cost and latency, not automatically on the logical
  token ledger.** Xu et al. (2026) show that a *homogeneous* workflow (one base model, agents
  differing only by prompt and position) can often be executed by a single agent over
  multi-turn dialogue, where KV-cache reuse reduces repeated prefill computation, latency and
  serving cost. C3 is homogeneous in that sense. Under this thesis's ledger, however,
  repeated context is counted every call **by design** (§2), so KV-cache reuse does not
  automatically reduce logical token usage; whether a single-agent execution of the same
  workflow would consume more or fewer *logical* tokens depends on the exact prompt and state
  representation it uses. Correct scoping: the C1↔C3 comparison of *logical tokens* remains
  valid as specified, while any claim about compute cost or latency overhead of decomposition
  must be qualified as implementation-dependent — consistent with the existing confound that
  latency and token counts tell different stories.
- **L5 — Sampling baseline: substantially mitigated, with a scope limit that remains.**
  Parmar et al. (2025) found plain Best-of-N to be the strongest PlanGEN variant on NATURAL
  PLAN (Meeting 43.80 EM), i.e. the cheapest competing use of the same budget may beat
  elaborate coordination. Adding **C5 — Validated Best-of-3 single-agent sampling** (§4 C5,
  GO decision 2026-07-26) removes the original gap: the comparison is no longer made only
  against single-trajectory agents. The residual limitation, stated explicitly: *L5 is
  substantially mitigated by the fixed C5 Best-of-3 baseline. The thesis does not optimise
  over `N` and therefore does not claim that Best-of-3 is the strongest possible sampling
  strategy.* Not covered: Best-of-2, Best-of-5, alternative budget splits, and the dependence
  of the result on `N` — a sampling-strategy sweep is future work, not a claim of this thesis.

### Literature claims this design deliberately does NOT implement

- **LLM-Modulo backprompting (Kambhampati et al. 2024) is out of scope by design, not by
  budget.** Their performance mechanism is a loop in which an EXTERNAL sound verifier
  repeatedly backprompts the LLM (Blocksworld ~35% → 82% over 15 rounds; ~6× on
  TravelPlanner). This thesis adopts their *soundness* argument — validity is decided by a
  formal checker, never by the model — but deliberately excludes the loop. The binding
  reason is the design invariant, stated precisely in §8: **no formal-verifier feedback
  reaches the model and no agent has access to the solver optimum.** Exposing verifier
  diagnostics to a planner would change the experimental condition and confound the
  comparison of *LLM architectures* with the comparison of *verifier-assisted repair*.
  Equal-budget control alone would NOT prohibit such a system — one could run LLM-Modulo
  under the same cap with fewer rounds; the budget merely bounds how many repair rounds fit.
  Consequence: C3 and C4 are NOT LLM-Modulo instantiations and must never be described as
  such; the thesis reports no evidence about LLM-Modulo-style repair. Constructing such a
  condition would require lifting the no-oracle invariant and is left to future work.

---

## 8. Literature-derived design audit (MAS)

Performed 2026-07-26 against the notes in `thesis/literature/` (both the Russian originals
and `english_verified/`), pre-pilot, on the implemented C3 (post-review revision, §4
C3-1..C3-8). Purpose: check the design against the failure modes the literature says MAS
systems actually die of, and record which defences are *structural* (impossible by
construction) rather than merely *prompted*. Structural defences are the ones that survive
a model that ignores its instructions.

### MAST failure-mode audit (Cemri et al. 2025 — 14 modes, 3 categories)

Only the modes reachable in this task are listed; the paper's category shares are in
brackets. Two statuses are deliberately kept apart, because conflating them would make the
planned trace-level failure analysis incoherent:

- **prevented** — the harness makes the failure impossible to occur at all;
- **occurrable but non-corrupting** — the model can still commit the failure inside its
  trace, but it cannot corrupt the recorded score. MAST classifies *behaviour in the trace*,
  so such a mode must still be counted as present when observed during error analysis.

| MAST mode | Status in C3 | Mechanism |
|---|---|---|
| FM-1.3 Step Repetition [FC1, 15.7% — most frequent overall] | Bounded (cannot run away) | Every worker loop is capped twice: `max_steps` and its own token quota (§4 C3-5). Repetition can occur; it cannot become unbounded, and it cannot starve the sibling or the critic. |
| **FM-1.5 Unaware of Termination** [FC1, one of two modes most associated with failure] | Workflow-level **prevented**; agent-level tendency remains observable | No agent decides when the run stops: termination is harness-owned (§2 budget guard, `max_steps`, exactly one critic call, no revision round). A model may still show no awareness that a good stopping point was reached — that tendency is visible in the trace and is a legitimate error-analysis label. |
| FM-1.2 Context loss / dropped state | Prevented | `best_plan_so_far` is harness state, not conversation state; it survives empty output, format failure, abort and budget exhaustion. |
| FM-2.4 Information Withholding [FC2] | **Not applicable as defined** — see §7 L3 | Workers cannot withhold what they never received; the design *partitions* information instead. The analogous risk (unrecovered cross-partition dependencies) is real and is measured by C3-FI, but it must not be labelled FM-2.4. |
| FM-2.6 Reasoning-Action Mismatch [FC2, 13.2%] | Constrained at syntax/tool-contract level; **occurrable at the semantic level**, non-corrupting | The parser and tool contract exclude malformed or non-existent actions. A model can still reason correctly about a conflict and then emit a well-formed but wrong slot — that is precisely FM-2.6 and it can occur. It cannot corrupt the score: the hidden gate judges the emitted plan, and the fallback stands. |
| FM-2.3 Task drift / role confusion | Bounded structurally | Roles are graph positions, not prompt requests: a worker cannot see or act on the other cluster (restricted sub-instance + tools bound to it), and the critic cannot introduce a person outside the pool (subset conjunct). |
| FM-3.1 Premature termination [FC3] | Occurrable; non-corrupting | An early `finish` cannot lose work: the score comes from `best_plan_so_far`, not from the last message. |
| FM-3.2 Missing / Incomplete Verification [FC3] | **Occurrable**; deterministically detected post-hoc | A critic may check only some windows, skip travel feasibility, or miss a duplicate. Nothing prevents the behaviour; the hidden validator simply refuses the resulting plan, so the failure appears in the trace but not in the score. |
| FM-3.3 Incorrect Verification [FC3, 9.1%] | **Occurrable**; non-corrupting | A critic may assert that a plan is fine when it violates travel time. Validity is never delegated to the model — `validate()` (independent of the solver) is the sole authority — so the wrong verdict cannot be recorded as a valid plan. Aligns with Kambhampati's finding that LLM self-verification can be worse than none. |

Modes out of scope for this task: FM-2.2 (Failure to Request Clarification) — the
instance is fully specified and there is no user to ask; FM-1.1 (disobeying task
specification) is constrained by the answer contract + guided decoding.

**Consequence for error analysis.** Because FM-2.6, FM-3.1, FM-3.2 and FM-3.3 are
occurrable, the planned MAST-based trace labelling (see the note on Cemri et al.) can and
should record them when observed. A clean satisfaction score is NOT evidence that these
modes did not occur; it is evidence that they did not propagate to the score.

### Precise statement of the no-oracle invariant

The shorthand "the solver is never inside the agent system" (§4 C3-3) is accurate about
CP-SAT but too coarse to describe the hidden gate, because `is_valid` **is** called inside
the workflow (`react_core.agent_step`, `react_verify_revise.revise_step`,
`multi_agent/hierarchical.aggregate_node`, `multi_agent/critic.critic_step`). The exact
invariant, which the code satisfies and the AST test in `tests/test_mas_split_merge.py`
enforces, is:

> **No formal-verifier feedback reaches the model, and no agent has access to the solver
> optimum.**

Operationally, three distinct things must not be confused:

| Component | Where it runs | What it may influence | What the model learns |
|---|---|---|---|
| CP-SAT (`src/oracle/solver.py`) | Harness only, before the agent | Ground truth: the metric denominator, the complexity metric | Nothing — never imported by any agent module |
| `validate` / `is_valid` (`src/oracle/validator.py`) | **Inside the workflow**, as an in-loop non-communicating deterministic constraint validator and candidate-selection gate | Which proposal becomes `best_plan_so_far`, hence the scored artefact | Nothing — the verdict is never returned to the model, and no diagnostics are backprompted |
| `score_plan` (`src/evaluation/scorer.py`) | Harness only, after the agent | The recorded score (independent revalidation) | Nothing |

So the correct description of the validator is neither "post-hoc" (it runs in-loop and
selects) nor merely a "silent filter": it is an **in-loop non-communicating deterministic
constraint validator
and candidate-selection gate**. Use that phrasing in the thesis.

### Alignment summary per paper

| Source | What it demands / warns | This design |
|---|---|---|
| Cemri 2025 (MAST) | Structural redesign beats prompt patches; termination and verification are where MAS dies | Termination is workflow-level prevented and verification is never delegated to the model (table above). Information partitioning is retained deliberately and measured by C3-FI, but is NOT labelled FM-2.4 (L3). |
| Kambhampati 2024 (LLM-Modulo) | LLMs cannot self-verify; soundness must come from an external checker | Adopted: `validate` decides validity, the critic's opinion decides nothing; a bad critic can fail to help but cannot lower the score (gate + fallback floor). Backprompting loop deliberately NOT adopted — reason is the no-oracle design, not the budget (§7). |
| Tran 2026 (equal budget, DPI) | MAS ≤ SAS on information grounds *under their assumptions*; MAS helps where the single agent's *effective* context use degrades | Accepted as the null. The mechanism conjectured here is budget-induced degradation (C1 must buy every fact with tokens it also needs for search), NOT context corruption as in Tran; labelled an analogy, never a proven equivalence. Budget control here is stricter than theirs (self-hosted tokenizer vs unreliable API counters). |
| Ke 2026 (MAS-Orchestra) | Gains are axis-dependent; Depth is an explicit counterexample; MAS helps most at the *edge* of sub-agent competence | Disclosed as L1 — an a priori reason to expect limited gains on a predominantly sequential task, not a prediction that C3 must lose. Reinforced by the strong-model caveat: Qwen3-32B is a capable sub-agent, which their result says shrinks any MAS gain. |
| Tang 2025 (complexity) | Complexity can moderate the advantage of redundant debate-style MAS | Cited only in that scoped form (L2); the redundancy mechanism their model depends on is absent from C3, and their model says nothing about equal-budget hierarchical decomposition. |
| Xu 2026 (OneFlow) | Homogeneous workflows are often single-agent-executable; KV-cache reuse cuts repeated prefill, latency and serving cost | Disclosed as L4, scoped: under this ledger repeated context is counted by design, so KV-cache reuse does not automatically change logical token usage; the qualification applies to compute/latency claims. |
| Parmar 2025 (PlanGEN) | Best-of-N was the strongest variant; complexity-stratified evaluation matters | Stratification adopted (level × cap bins). Sampling baseline adopted as **C5 — Validated Best-of-3** (§4 C5) after the audit; interpretive limit recorded as L5. |
| Amonkar 2025 (CSP) | LLM-as-solver ≥ LLM-as-formalizer on Meeting Planning; both degrade with constraint count | Supports ReAct-as-baseline and the solver-as-oracle-only choice; their Qwen3-32B Meeting-Planning degradation curve is the external reference point for our complexity axis. |

### What this audit changed, and what it did not

**Unchanged: the C3 architecture and all code.** Every C3 finding was either a defence
already present by construction or a disclosure item (L1–L4). C3 stays exactly as locked in
§4 C3-1..C3-8 and implemented in `src/agents/multi_agent/`; the audit produced no edit to
`src/`.

**Changed: one addition to the condition set.** The absence of a sampling baseline (L5) was
a genuine gap rather than a wording problem, so **C5 — Validated Best-of-3** was specified
(§4 C5) and its go/no-go resolved in §6. Its implementation is a separate commit.

**Also produced by the audit (documentation only):**
- the canonical hypothesis definitions in §5, added because the labels existed only in the
  exposé and had drifted into incompatible schemes across the paper notes;
- the precise statement of the no-oracle invariant (above), replacing a shorthand that was
  accurate about CP-SAT but too coarse about the in-loop validator gate;
- the finalisation-drift confound in §7, found while checking why C5 must select before
  finalising;
- corrections to `thesis/literature/kambhampati2024_llmmodulo.md` and its verified English
  edition, which described C4 as an LLM-Modulo instantiation with CP-SAT as an in-loop hard
  critic — a direct contradiction of the invariant.

**Two known gaps left open on purpose:**
1. The hypothesis numbering in the older paper notes is not yet rewritten note-by-note; §5
   is now the authority and §7 avoids bare labels, so nothing downstream depends on the stale
   ones. Aligning every note is a separate documentation pass.
2. The §5 hypothesis wording is marked **[VERIFY]** because the exposé PDF is git-ignored and
   was not readable during the audit.

Re-run this audit if the condition set changes again.
