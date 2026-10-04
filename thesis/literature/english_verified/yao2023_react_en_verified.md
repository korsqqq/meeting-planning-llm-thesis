# ReAct: Synergizing Reasoning and Acting in Language Models

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
| **Authors** | Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao |
| **Affiliations** | Princeton University; Google Research / Google Brain |
| **Year** | 2023 |
| **Venue** | ICLR 2023 |
| **arXiv ID** | 2210.03629v3 |
| **BibTeX key** | `yao2023react` |
| **Main models** | PaLM-540B; GPT-3 results in the appendix; PaLM-8B and PaLM-62B for fine-tuning experiments |
| **Qwen used?** | No |

## 2. One-sentence summary

ReAct is a single-agent prompting paradigm in which an LLM alternates language-based reasoning traces with task-specific actions and observations, allowing it to plan, gather external evidence, track progress, and revise its behavior in one interpretable trajectory.

## 3. Key arguments

1. **Reasoning and acting are complementary.** Reasoning traces support decomposition, progress tracking, error recovery, and action selection; environment actions provide information that is unavailable from the model's internal parameters alone.

2. **External interaction can reduce unsupported factual reasoning.** On knowledge-intensive tasks, ReAct grounds answers through Wikipedia search and lookup actions, reducing some hallucinated reasoning found in pure chain-of-thought trajectories.

3. **Action-only agents are weak on multi-step tasks.** Without explicit reasoning, agents struggle to maintain goals, track state, and choose useful actions.

4. **ReAct is a strong few-shot single-agent baseline.** On ALFWorld and WebShop, prompted ReAct agents outperform several learned imitation or reinforcement-learning baselines despite receiving only a small number of demonstrations.

5. **ReAct is not universally best.** On HotpotQA, pure ReAct is weaker than CoT self-consistency, while hybrid fallback strategies perform best. Its value depends on the task and on the quality of external observations.

## 4. Methodology

### 4.1 Core idea
The agent accumulates a trajectory containing observations, reasoning traces, actions, and results. Reasoning traces do not directly change the environment; they update the linguistic context used to choose the next action.

```text
Thought: ...
Action: ...
Observation: ...
Thought: ...
Action: ...
Observation: ...
Finish: ...
```

### 4.2 Tasks

| Task family | Benchmark | Capability tested |
|---|---|---|
| Knowledge-intensive reasoning | HotpotQA | Multi-hop question answering |
| Fact verification | FEVER | Evidence-based verification using Wikipedia |
| Interactive decision-making | ALFWorld | Text-based household tasks |
| Web navigation and shopping | WebShop | Searching, filtering, selecting options, and purchasing |

### 4.3 Action space
For HotpotQA and FEVER, the paper uses a compact Wikipedia interface:

| Action | Purpose |
|---|---|
| `search[entity]` | Retrieve the beginning of a page or related entities |
| `lookup[string]` | Find the next sentence containing a target string |
| `finish[answer]` | Return the final answer |

ALFWorld and WebShop expose task-specific navigation, object manipulation, search, option-selection, and completion actions.

### 4.4 Models

| Role | Model |
|---|---|
| Main prompting experiments | PaLM-540B |
| Additional appendix experiments | GPT-3 |
| Fine-tuning | PaLM-8B and PaLM-62B |

### 4.5 Baselines

| Method | Description |
|---|---|
| Standard | Direct answer without explicit reasoning or actions |
| CoT | Reasoning-only prompting |
| CoT-SC | Self-consistency over multiple CoT samples |
| Act | Action-only trajectory |
| ReAct | Interleaved reasoning and actions |
| ReAct → CoT-SC | Use CoT-SC when ReAct does not finish |
| CoT-SC → ReAct | Use ReAct when CoT-SC confidence is insufficient |
| BUTLER | Learned imitation baseline for ALFWorld |
| IL / IL+RL | Learned WebShop baselines |

### 4.6 Metrics

| Benchmark | Metric |
|---|---|
| HotpotQA | Exact Match |
| FEVER | Accuracy |
| ALFWorld | Success rate |
| WebShop | Reward/score and success rate |
| Manual analysis | Hallucination, reasoning, search-result, and false-positive error categories |

### 4.7 Experimental details
- Six manually written ReAct exemplars for HotpotQA.
- Three manually written ReAct exemplars for FEVER.
- CoT-SC uses 21 sampled reasoning trajectories at temperature 0.7.
- ALFWorld evaluates 134 unseen games and several prompt permutations.
- WebShop evaluates 500 test instructions.
- Fine-tuning experiments use approximately 3,000 successful trajectories generated by ReAct or baseline methods.

## 5. Results and conclusions

### 5.1 HotpotQA and FEVER

| Method | HotpotQA EM | FEVER accuracy |
|---|---:|---:|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | 64.6 |
| ReAct → CoT-SC | 35.1 | 62.0 |

ReAct improves over action-only prompting and performs strongly on FEVER, but pure ReAct does not beat CoT-SC on HotpotQA. The highest scores come from hybrid methods that combine internal parametric knowledge with external evidence gathering.

### 5.2 Error analysis
In a manual analysis of 200 HotpotQA trajectories:
- CoT failures more often involve hallucinated facts or reasoning.
- ReAct trajectories are more grounded but can fail because search results are unhelpful, reasoning is incorrect, or the agent repeats actions.

| Error category | ReAct | CoT |
|---|---:|---:|
| False positive among successful trajectories | 6% | 14% |
| Hallucination as a failure mode | 0% | 56% |
| Reasoning error as a failure mode | 47% | 16% |
| Search-result error | 23% | Not applicable |

These percentages refer to the paper's manually categorized subset and should not be generalized beyond that analysis.

### 5.3 ALFWorld

| Method | Overall success rate |
|---|---:|
| Act, best of six prompt permutations | 45% |
| ReAct, average | 57% |
| ReAct, best of six | 71% |
| BUTLER, best of eight | 37% |

Reasoning traces help the agent decompose goals, remember subgoals, and infer likely object locations.

### 5.4 WebShop

| Method | Score | Success rate |
|---|---:|---:|
| Act | 62.3 | 30.1% |
| ReAct | 66.6 | 40.0% |
| IL | 59.9 | 29.1% |
| IL+RL | 62.4 | 28.7% |
| Human expert | 82.1 | 59.6% |

ReAct improves absolute success by roughly ten percentage points over the strongest listed non-ReAct learned baseline, while remaining well below expert human performance.

### 5.5 Overall conclusion
ReAct demonstrates that a strong single-agent system can combine:
- explicit reasoning traces;
- external tools and actions;
- observation feedback;
- an interpretable trajectory;
- hybrid fallback to reasoning-only methods when appropriate.

## 6. Direct relevance to my thesis

### Role as the primary single-agent baseline
ReAct is the canonical reference for my strong single-agent condition. A fair MAS comparison should not use only direct prompting or a weak action-only agent; it should compare against an agent that can reason, call tools, inspect observations, and revise its next step.

### Relationship to the hypotheses

| Hypothesis | Relevance |
|---|---|
| **H1: MAS eventually beats SAS** | Not directly tested; the paper contains no MAS comparison. |
| **H2: Gains justify overhead after a threshold** | Not tested; there is no equal-token or architecture-cost analysis. |
| **H0: A strong SAS remains competitive** | Indirect support: careful single-agent design substantially improves over weak baselines. |

### Methodological ideas to reuse
1. **ReAct loop:** `Reasoning → Action → Observation → ... → Final answer`.
2. **Ablation logic:** Compare reasoning-only, action-only, combined, and critic-assisted conditions.
3. **Trajectory-level analysis:** Inspect where an agent forms an incorrect assumption, calls an unnecessary tool, misses a constraint, or repeats an action.
4. **Internal versus external knowledge:** Separate LLM planning from feedback supplied by a validator or CP-SAT-based checker.
5. **Strong-baseline requirement:** MAS claims are meaningful only when the single-agent comparator is itself competently designed.

### Suggested thesis formulation
> ReAct motivates the single-agent baseline used in this thesis. By interleaving reasoning traces with environment or tool actions, it produces stronger and more interpretable agents than reasoning-only or action-only prompting. The hierarchical MAS is therefore evaluated against a ReAct-style agent rather than a weak direct-prompting baseline.

## 7. Where to cite it in the thesis

| Section | Use |
|---|---|
| **Introduction** | Establish reasoning-plus-acting as a foundational LLM-agent design. |
| **Related Work** | Describe the canonical single-agent tool-use paradigm. |
| **Methodology** | Justify the Thought/Action/Observation loop and final-answer protocol. |
| **Baselines** | Explain why direct CoT is not a sufficiently strong comparator. |
| **Discussion** | Emphasize that MAS must improve on a strong SAS enough to justify overhead. |
| **Error Analysis** | Motivate inspection of complete trajectories rather than final answers alone. |

## 8. Important differences from my thesis

| ReAct paper | My thesis |
|---|---|
| Proposes a single-agent prompting paradigm | Compares single ReAct with hierarchical MAS and critic-assisted variants |
| No MAS architecture | Explicit supervisor/worker/aggregator/critic conditions |
| No equal token budget | Matched total token budgets |
| No systematic cost analysis | Tokens, calls, latency, and efficiency are measured |
| No solver optimum | OR-Tools CP-SAT provides the reference optimum |
| EM, accuracy, and task success | Meeting-satisfaction rate relative to the optimum |
| QA, fact verification, household interaction, and shopping | Constraint-based meeting planning |
| Complexity is not the primary independent variable | Interacting-constraint complexity is systematically varied |
| PaLM and GPT-3 models | Self-hosted Qwen3 32B and Qwen3 8B |

## 9. Short quotations
- "synergize reasoning and acting"
- "ReAct outperforms Act consistently"
- "human aligned and controllable"

## 10. Verification notes
- The metadata, task suite, model family, principal result tables, and interpretation of ReAct as interleaved reasoning and acting match the ICLR paper.
- The original note is largely accurate. The main qualification is that claims of lower hallucination come from a limited manually inspected HotpotQA subset and should not be generalized to every task.
- ReAct does **not** uniformly beat CoT: it underperforms CoT-SC on HotpotQA, and hybrid strategies are strongest there. This caveat is retained because it matters for presenting ReAct as a strong but not universally optimal baseline.

## 11. BibTeX

```bibtex
@inproceedings{yao2023react,
  title     = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author    = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=WE_vluYUL-X},
  note      = {arXiv:2210.03629}
}
```
