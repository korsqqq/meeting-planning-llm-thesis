# Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks

<!-- h-labels-banner -->
> ⚠️ **On the H0/H1/H2 labels in this note (added 2026-07-26).** The canonical hypothesis
> definitions now live ONLY in `THESIS_DECISIONS.md` §5. These notes predate that block and
> use at least two mutually incompatible numbering schemes (in some, `H2` means "a threshold
> exists"; in others, "gains justify the overhead"), and none is guaranteed to match the
> exposé. **The substance in parentheses is authoritative, not the number.** Re-check any
> label against §5 and restate it substantively before it enters a thesis chapter. A
> note-by-note renumbering is a separate pass (see §8, "Two known gaps left open on purpose").

## 1. Metadata

| Field | Value |
|---|---|
| **Authors** | Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Saldyt, Anil Murthy |
| **Affiliation** | School of Computing and AI, Arizona State University |
| **Year** | 2024 |
| **Venue** | ICML 2024, PMLR 235, pages 22895–22907 |
| **Type** | Position paper |
| **arXiv ID** | 2402.01817v3 |
| **BibTeX key** | `kambhampati2024llmmodulo` |
| **Qwen used?** | No |

## 2. One-sentence summary

The paper argues that autoregressive LLMs are unreliable autonomous planners and self-verifiers, but can contribute effectively as approximate knowledge sources and candidate generators inside an LLM-Modulo system that uses external, model-based critics for sound verification.

## 3. Key arguments

1. **LLMs should not be equated with sound planners.** The position paper treats them as powerful approximate knowledge sources whose fluent generation does not itself provide correctness guarantees.

2. **Autonomous planning performance is weak and brittle.** Earlier PlanBench experiments cited by the paper report about 12% average valid-plan success for GPT-4 across the evaluated domains. Newer models improve considerably on ordinary Blocksworld in some settings, but performance collapses on obfuscated Mystery Blocksworld.

3. **Self-verification by the same LLM is unreliable.** In graph-coloring experiments, iterative self-critique does not provide the guarantees of a formal checker and can reject correct candidates or preserve incorrect ones.

4. **LLM-Modulo follows a generate-test-critique loop.** An LLM proposes candidates; external hard critics check formal properties; a controller converts diagnostic feedback into a new prompt; optional soft critics evaluate less formal preferences.

5. **Correctness guarantees come from the external verifier.** The LLM helps with candidate generation, interpretation, and model acquisition, but system soundness depends on trusted critics such as VAL or a constraint solver.

## 4. Methodology and evidence

### Autonomous planning evidence
- PlanBench domains include Blocksworld, Logistics, and obfuscated Mystery Blocksworld.
- Plans are checked automatically with the VAL validator for executability and goal achievement.
- The paper discusses results for GPT-family models and newer systems including GPT-4o, Claude 3 Opus, Llama 3 70B, and Gemini Pro.
- Mystery Blocksworld replaces meaningful predicate and object names with arbitrary tokens to reduce reliance on memorized world knowledge.

### Self-verification evidence
- Task: graph coloring, formulated as a CSP.
- Conditions compare direct generation, iterative LLM-based checking, and feedback from a correct external verifier.
- The external verifier is the condition that can provide reliable correctness feedback.

### LLM-Modulo case studies

| Domain | Benchmark | External critic | Reported outcome |
|---|---|---|---|
| Classical planning | Blocksworld | VAL | Up to 82% after 15 feedback rounds in the cited setup |
| Classical planning | Logistics | VAL | Approximately 70% in the cited setup |
| Travel planning | TravelPlanner | Hard and commonsense critics | Roughly sixfold improvement over the original very low baseline |

### Architecture

```text
Problem specification → LLM candidate generation → Plan blackboard
                                                 ↓
                                      Bank of critics
                                  ↙ agreement   disagreement ↘
                         Valid solution       Meta-controller
                                                ↓
                                           Back-prompt LLM
```

Possible LLM roles include candidate generation, format conversion, clarification of incomplete specifications, and helping domain experts build or refine critic models.

## 5. Results and conclusions

### Autonomous mode

| Domain / condition | Best reported model | Accuracy |
|---|---|---:|
| Ordinary Blocksworld, zero-shot | Claude 3 Opus | 59.3% |
| Ordinary Blocksworld, one-shot | Claude 3 Opus | 48.17% |
| Mystery Blocksworld, zero-shot | All tested models | approximately 0–0.16% |
| Mystery Blocksworld, one-shot | All tested models | approximately 0.4–4.3% |

The sharp ordinary-to-obfuscated drop supports the authors' argument that success on familiar-looking domains may rely heavily on learned associations rather than robust domain-independent planning.

### External-verifier loop
- Blocksworld improves from roughly 35% in the referenced baseline condition to 82% after repeated VAL feedback.
- TravelPlanner improves from a sub-1% baseline to a few percent, a large relative improvement but still a low absolute success rate.
- Mystery Blocksworld remains difficult because a verifier cannot help if the LLM rarely generates a promising candidate to repair.

### Design principles
- LLM correctness is not the same as system correctness.
- Soundness comes from model-based critics.
- Completeness still depends on whether candidate generation explores useful plans.
- Hard critics check formal constraints; soft critics can score style or preference.
- Additional iterations improve reliability only when the feedback is informative and candidate generation can respond to it.

## 6. Direct relevance to my thesis

### What this thesis takes from LLM-Modulo — and what it deliberately does not

> **Corrected 2026-07-26 (audit, THESIS_DECISIONS §8).** An earlier version of this table
> mapped the in-loop "hard critic" onto OR-Tools CP-SAT and the meta-controller onto the
> LangGraph loop, i.e. it described C4 as an LLM-Modulo instantiation. That contradicts the
> locked invariant (THESIS_DECISIONS §4: "The critic is an LLM with fresh context, **not**
> CP-SAT — the solver is never shown to the agents") and the project-wide no-oracle
> invariant. Neither C3 nor C4 is an LLM-Modulo instantiation.

Adopted: the **soundness principle** — correctness is decided by an external formal checker
rather than by the model itself (their graph-colouring result shows LLM self-verification can
be worse than none). In this thesis the hidden `validator` is that checker; its verdict is
never revealed to any agent, and the hidden gate makes a poor critic harmless (it can fail to
help but cannot lower the score).

Not adopted: the **back-prompting loop**, which is where their measured gain comes from
(Blocksworld ~35% → 82% over 15 rounds; roughly 6× on TravelPlanner). The binding reason is
the design invariant, not the budget: **no formal-verifier feedback reaches the model, and no
agent has access to the solver optimum** (THESIS_DECISIONS §8). Exposing verifier diagnostics
to a planner would change the experimental condition and confound the comparison of *LLM
architectures* with the comparison of *verifier-assisted repair*. Equal-budget control alone
would not prohibit such a system — one could run it under the same cap with fewer rounds; the
budget merely bounds how many repair rounds fit. The thesis therefore reports no evidence
about LLM-Modulo-style repair.

| LLM-Modulo component | Status in this thesis |
|---|---|
| LLM candidate generator | Present — Qwen3 workers (C3) / planner (C4) |
| Hard model-based critic **inside the loop** | **Absent by invariant.** CP-SAT is harness-side only: optimum for the metric denominator and independent scoring |
| Sound external checker **outside the loop** | Present — hidden `validator` as a silent gate, verdict never disclosed |
| Meta-controller and back-prompting | **Absent.** Exactly one critic call, no revision round (§4 C3-4) |
| Optional soft critic | Present — the LLM critic, but its proposal must pass the formal gate rather than replace it |
| Problem specification | Parameterized meeting-scheduling constraints |

### Relationship to the hypotheses
- **H1:** External verification may become more valuable in constraint-rich tasks, as illustrated by Blocksworld and TravelPlanner. The paper does not itself establish a complexity crossover under equal budgets.
- **H0 for internal self-critique:** The graph-coloring discussion warns that an LLM verifying itself may fail to improve and can sometimes hurt.
- **Ceiling condition:** At extreme difficulty, even a perfect verifier cannot compensate for a generator that never proposes repairable candidates.

### Methodological ideas to reuse
1. Separate hard, formally checkable critics from soft LLM critics.
2. Treat the LLM as a candidate generator rather than the source of correctness guarantees.
3. Return precise violation feedback through a controller rather than merely asking the LLM to “try again”.
4. Report both relative improvement and absolute final success, especially when baselines are near zero.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Motivate hybrid planning systems with trusted verification. |
| **Related Work — LLM Planning** | Evidence of brittle autonomous planning and self-verification. |
| **Related Work — Hybrid Systems** | Contrast case: an LLM coupled to an in-loop verifier. State explicitly that this thesis adopts the soundness principle but not the back-prompting loop. |
| **Methodology** | Justification that validity is decided by an external formal checker (`validator`), not by the model. Do NOT cite CP-SAT as an in-loop hard critic — the solver is harness-side only. |
| **Discussion** | Explain why verifier loops may plateau when candidate generation collapses. |

## 8. Important differences

1. The paper focuses on classical PDDL planning, graph coloring, and TravelPlanner; my task is meeting scheduling with controlled interacting constraints.
2. VAL is a domain-specific plan validator; OR-Tools CP-SAT is used in my design as a constraint solver and evaluation oracle.
3. The paper permits many feedback rounds and does not match token budgets against single-agent alternatives.
4. LLM-Modulo is primarily a generator-plus-verifier architecture, not a hierarchical supervisor/worker MAS.
5. The paper does not estimate a continuous complexity threshold.
6. Its principal outcome is binary validity; my primary metric gives partial credit relative to the solver optimum.

## 9. Short quotations

> "results in the autonomous mode are pretty bleak"

> "acting without the ability to plan is surely a recipe for unpleasant consequences"

> "automated critics in the loop significantly improves the performance"

## 10. Verification notes
- ICML 2024, PMLR volume 235, author list, and position-paper status are confirmed by the proceedings record.
- The original summary overgeneralized the **~12%** figure. That figure refers to the average success of the best GPT-4 system in earlier PlanBench work cited by the position paper; it is not the maximum score in every table of the 2024 paper. Ordinary Blocksworld results for Claude 3 Opus are much higher, while Mystery Blocksworld remains near zero.
- “LLMs cannot plan” is the authors' position and framing. The empirical evidence supports severe unreliability and lack of guarantees, but the thesis should present the statement as the paper's argument rather than an uncontested theorem.
- The reported sixfold TravelPlanner improvement is large in relative terms but remains only a few percentage points in absolute final success.
- CP-SAT is an example relevant to my implementation; the paper's concrete classical-planning verifier is VAL.

## 11. BibTeX

```bibtex
@inproceedings{kambhampati2024llmmodulo,
  title     = {Position: {LLMs} Can't Plan, But Can Help Planning in {LLM-Modulo} Frameworks},
  author    = {Kambhampati, Subbarao and Valmeekam, Karthik and Guan, Lin and
               Verma, Mudit and Stechly, Kaya and Bhambri, Siddhant and
               Saldyt, Lucas and Murthy, Anil},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {22895--22907},
  year      = {2024},
  publisher = {PMLR}
}
```
