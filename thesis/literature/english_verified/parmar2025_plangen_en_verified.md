# PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving

<!-- h-labels-banner -->
> ⚠️ **On the H0/H1/H2 labels in this note (added 2026-07-26).** The canonical hypothesis
> definitions now live ONLY in `THESIS_DECISIONS.md` §5. These notes predate that block and
> use at least two mutually incompatible numbering schemes (in some, `H2` means "a threshold
> exists"; in others, "gains justify the overhead"), and none is guaranteed to match the
> exposé. **The substance in parentheses is authoritative, not the number.** Re-check any
> label against §5 and restate it substantively before it enters a thesis chapter. A
> note-by-note renumbering is a separate pass (see §8, "Two known gaps left open on purpose").

## Document status
This English edition translates and updates the original note. The source note was based on arXiv v1 from 22 February 2025. The work was subsequently published in the main proceedings of EMNLP 2025, so the publication metadata and BibTeX have been updated accordingly.

## 1. Metadata

| Field | Value |
|---|---|
| **Authors** | Mihir Parmar, Xin Liu, Palash Goyal, Yanfei Chen, Long Le, Swaroop Mishra, Hossein Mobahi, Jindong Gu, Zifeng Wang, Hootan Nakhost, Chitta Baral, Chen-Yu Lee, Tomas Pfister, Hamid Palangi |
| **Affiliations in the arXiv version** | Google; Arizona State University |
| **Year** | 2025 |
| **Venue** | Proceedings of EMNLP 2025, main conference |
| **Pages** | 20640–20666 |
| **DOI** | 10.18653/v1/2025.emnlp-main.1042 |
| **arXiv ID** | 2502.16111v1 |
| **BibTeX key** | `parmar-etal-2025-plangen` |
| **Qwen used?** | No. The note reports Gemini 1.5 Pro as the main model and Gemini 2.0 Flash/GPT-4o in additional model-agnostic studies. |

## 2. One-sentence summary

PlanGEN combines a constraint agent, verification agent, and selection agent to improve planning and reasoning through constraint-guided evaluation and instance-adaptive selection among Best-of-N, Tree-of-Thought, and REBASE.

## 3. Key arguments

1. Existing inference-time planning methods often verify only the overall answer and do not explicitly check instance-specific constraints.
2. A fixed inference algorithm may be inappropriate because instances within the same benchmark can have very different complexity.
3. The constraint agent extracts requirements; the verification agent scores candidates against those requirements; the selection agent chooses an inference-time algorithm.
4. Constraint-guided iterative verification improves results across NATURAL PLAN, OlympiadBench, DocFinQA, and GPQA.
5. The work is relevant to architecture adaptation, but it does not provide a strict equal-token SAS-versus-MAS comparison against a strong ReAct baseline.

## 4. Methodology

### 4.1 Benchmarks

| Benchmark | Capability | Evaluation set described in the note |
|---|---|---:|
| NATURAL PLAN | Calendar, meeting, and trip planning | 1,000 / 1,000 / 1,600 |
| GPQA Diamond | Graduate-level scientific reasoning | 198 |
| OlympiadBench | Text-only mathematical and physics reasoning | 674 math; 236 physics |
| DocFinQA | Financial-document reasoning | 922 |

NATURAL PLAN is particularly relevant because it contains the same broad meeting-planning family as my thesis.

### 4.2 Models
- Main experiments: Gemini 1.5 Pro.
- Additional model-agnostic case studies: Gemini 2.0 Flash and GPT-4o.
- Qwen is not part of the reported model stack.

### 4.3 Baselines
- Zero-shot chain-of-thought.
- A vanilla multi-agent feedback baseline.
- Model-specific reference results on selected benchmarks.

The study does not include a fully tool-enabled ReAct baseline under the same total token cap.

### 4.4 PlanGEN agents

| Agent | Function |
|---|---|
| **Constraint agent** | Extracts instance-specific constraints from the task description. |
| **Verification agent** | Evaluates candidates against extracted constraints and returns a reward from `-100` to `100`. |
| **Selection agent** | Uses modified UCB and LLM-guided priors to choose an inference algorithm. |

### 4.5 Variants
1. **Best of N:** Generate several complete candidates and select the highest-reward plan.
2. **Tree-of-Thought:** Expand and score intermediate reasoning or planning nodes.
3. **REBASE:** Use reward-guided tree search and pruning.
4. **Mixture of Algorithms:** Dynamically select among Best of N, ToT, and REBASE based on the instance.

### 4.6 Metrics

| Benchmark | Metric |
|---|---|
| NATURAL PLAN | Exact Match |
| OlympiadBench | Micro-average accuracy |
| GPQA | Accuracy |
| DocFinQA | Accuracy and F1 |

My meeting-satisfaction metric differs by providing partial credit relative to a formal optimum.

### 4.7 Reported setup
- Agent temperature: `0` for deterministic stages.
- Best of N: five samples at temperature `0.7`.
- Tree-of-Thought: three children per root node at temperature `0.7`.
- REBASE: initial width `10`, with width reduced after expansions; temperature `0.7`.
- Verification reward: `-100` to `100`; examples treat scores around `95` or higher as very high quality.

## 5. Results and conclusions

### 5.1 Aggregate improvements

| Benchmark | Improvement over the strongest baseline reported by the paper |
|---|---:|
| NATURAL PLAN | approximately +8% |
| OlympiadBench | approximately +4% |
| DocFinQA | approximately +7% |
| GPQA | approximately +1% |

### 5.2 NATURAL PLAN results
The note reports the following strongest PlanGEN scores:

| Task | Best reported score |
|---|---:|
| Calendar Scheduling | 60.70 EM |
| Meeting Planning | 43.80 EM |
| Trip Planning | 41.63 EM |

Best of N is particularly strong on NATURAL PLAN, which is methodologically important: a complex multi-agent control structure should be compared not only with direct prompting but also with strong sampling-and-verification baselines.

### 5.3 Complexity-dependent selection
- Calendar Scheduling: different methods dominate at different complexity ranges, with adaptive selection becoming more useful as complexity rises.
- Meeting Planning: Best of N is strong on simple and intermediate cases, while the Mixture of Algorithms is reported as more competitive on complex cases.
- Trip Planning: Best of N and adaptive selection outperform other methods overall, but all methods remain weak on the most difficult instances.

These results motivate complexity-aware evaluation, but they do not by themselves establish a clean causal token-controlled crossover between SAS and MAS.

### 5.4 Verification agent
Higher verification rewards correlate with a higher probability of a correct outcome in the paper's DocFinQA and GPQA analyses. This supports verification as a useful quality-control signal, while leaving open whether an LLM reward model is sufficiently reliable for formally checkable scheduling constraints.

### 5.5 Overhead
PlanGEN measures overhead largely through the number of LLM calls. ToT and REBASE can require calls for node generation, reward evaluation, and completion checks. Call count is informative but is not equivalent to a matched total token budget because prompt length, repeated context, and output length can differ greatly.

### 5.6 Limitations
- Selection uses predefined heuristics and may not generalize optimally across domains.
- Computational overhead remains a deployment concern.
- Future adaptive strategies may use reinforcement learning or meta-learning.
- The framework could be extended to multilingual and multimodal problems.

## 6. Direct relevance to my thesis

### Suggested Related Work formulation
> Parmar et al. introduce PlanGEN, a multi-agent framework that extracts constraints, verifies generated plans, and dynamically selects inference-time algorithms according to instance complexity. Their findings support complexity-aware verification, but the study does not compare a strong ReAct agent with hierarchical MAS under a matched token budget.

### Relationship to the hypotheses

| Hypothesis | Relevance |
|---|---|
| **H1: MAS wins after a complexity threshold** | Partial support: adaptive algorithm selection becomes more useful on some complex planning subsets. The architecture comparison is not token-controlled. |
| **H2: Gains justify overhead** | Indirect support only: performance and LLM-call counts are discussed, but tokens and latency are not normalized under equal budgets. |
| **H0: No stable MAS advantage after control** | Not ruled out. Best of N is often the strongest NATURAL PLAN variant, showing that additional agent roles are not automatically superior to sampling plus selection. |

### Methodological ideas to reuse
1. Extract explicit constraints before evaluating a candidate.
2. Separate generation and verification stages.
3. Report performance by complexity level rather than only aggregate average.
4. Include component ablations: single agent, verification-assisted single agent, planner/critic, and hierarchical MAS.
5. Plot quality against tokens, calls, and latency.
6. Test whether conclusions survive a smaller Qwen3 model.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Motivate constraint-aware planning and instance-level complexity. |
| **Related Work** | Closely related multi-agent planning framework with verification and adaptive search. |
| **Methodology** | Inspiration for separate constraint extraction and verification stages. |
| **Discussion** | Contrast its gains with my equal-budget strong-ReAct comparison. |

## 8. Important differences

1. PlanGEN does not equalize total token budgets across conditions.
2. Its principal single-agent baseline is not a fully comparable ReAct agent with the same tools and budget.
3. PlanGEN's roles are constraint, verification, and selection agents; my hierarchical architecture uses supervisor, workers, aggregator, and critic.
4. PlanGEN evaluates EM, accuracy, and F1; my primary metric is partial meeting satisfaction relative to a CP-SAT optimum.
5. Its verification signal is LLM-based; mine can use a formal hidden validator.
6. Its complexity often follows number of people, cities, or days; mine targets the number and interaction structure of constraints.
7. Its models are Gemini and GPT variants; mine are self-hosted Qwen3 models.
8. LLM call count does not capture repeated context, tool schemas, messages, total tokens, or wall-clock latency.
9. Best of N is a strong competitor and should be treated as an important baseline rather than merely a PlanGEN subcomponent.

## 9. Short quotations
- "PlanGEN consists of three specialized agents"
- "constraint-guided iterative verification improves inference-time algorithms"
- "all algorithms exhibit poor performance"

## 10. Verification notes
- The original note's publication status is outdated. PlanGEN was published in the **main proceedings of EMNLP 2025**, pages 20640–20666, with DOI `10.18653/v1/2025.emnlp-main.1042`.
- The arXiv title page confirms the affiliations **Google** and **Arizona State University**.
- The title, author list, three-agent design, algorithms, and abstract-level improvement claims match the final publication record.
- Claims of a “crossover” have been qualified: the paper reports complexity-dependent method rankings, but does not run the strict equal-budget causal comparison planned in my thesis.
- The BibTeX below uses the final ACL Anthology record rather than the obsolete `@misc` preprint entry.

## 11. BibTeX

```bibtex
@inproceedings{parmar-etal-2025-plangen,
    title = "{P}lan{GEN}: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving",
    author = "Parmar, Mihir and Liu, Xin and Goyal, Palash and Chen, Yanfei and Le, Long and Mishra, Swaroop and Mobahi, Hossein and Gu, Jindong and Wang, Zifeng and Nakhost, Hootan and Baral, Chitta and Lee, Chen-Yu and Pfister, Tomas and Palangi, Hamid",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    pages = "20640--20666",
    doi = "10.18653/v1/2025.emnlp-main.1042",
    url = "https://aclanthology.org/2025.emnlp-main.1042/"
}
```
