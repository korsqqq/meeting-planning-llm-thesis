# MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks

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
| **Authors** | Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Jiayu Wang, Semih Yavuz, Caiming Xiong, Shafiq Joty |
| **Affiliations** | Salesforce Research; MIT; University of Wisconsin–Madison |
| **Year** | 2026 |
| **Venue** | ICML 2026, Proceedings of Machine Learning Research, volume 306 |
| **arXiv ID** | 2601.14652v5, 21 May 2026 |
| **BibTeX key** | `ke2026masorchestra` |
| **Qwen used?** | Yes. Qwen2.5-7B-Instruct is the orchestrator in the controlled experiments. |

## 2. One-sentence summary

MAS-Orchestra treats multi-agent orchestration as a holistic function-calling reinforcement-learning problem and introduces MASBENCH, a controlled benchmark with five structural axes, showing that MAS benefits depend on task topology and sub-agent capability rather than holding universally.

## 3. Key arguments

1. **MAS gains are structure-dependent.** The paper studies Depth, Horizon, Breadth, Parallelism, and Robustness. MAS helps most on structures requiring aggregation, independent parallel work, intermediate state passing, or adversarial cross-checking; strictly sequential depth often favors or matches SAS.

2. **MAS is most useful near the edge of sub-agent competence.** When the available sub-agent is already very strong, coordination cost and error propagation can eliminate the benefit of orchestration.

3. **Robustness is the clearest MAS advantage.** Under adversarial perturbations, cross-verification and task separation make the multi-agent system substantially more resilient than the single-agent alternative.

4. **Holistic orchestration is more efficient than incremental workflow search.** MAS-Orchestra constructs the overall collaboration structure directly and reports more than an order-of-magnitude reduction in calls relative to AFlow in the cited AIME24 comparison.

5. **Instruction-tuned orchestrators delegate more effectively than reasoning-language-model orchestrators in the tested setup.** The latter often attempt to solve the task themselves instead of routing work to stronger specialists.

## 4. Methodology

### MAS-Orchestra framework
- Orchestration is represented as function calling with two primitives: `create_agent` and `create_flow`.
- **Degree of MAS (DoM):** low DoM permits at most one sub-agent; high DoM permits unrestricted multi-agent construction.
- **Holistic orchestration:** The workflow structure is generated as a complete design rather than grown one agent at a time.
- Training uses Group Relative Policy Optimization (GRPO).

### MASBENCH

| Axis | Definition | Coordination demand |
|---|---|---|
| **Depth** | Length of the longest dependency chain | Sequential reasoning |
| **Horizon** | Intermediate subtasks whose outputs must be passed onward | State tracking and intermediate verification |
| **Breadth** | Maximum in-degree or fan-in | Aggregation of several inputs |
| **Parallel** | Independent subtask components or fan-out | Parallel decomposition |
| **Robustness** | Subtasks affected by adversarial perturbations | Cross-verification and correction |

Axis values range from 2 to 12. The benchmark is generated from iGSM-style mathematical task graphs. Reported train/test sizes are: Depth 3,993/1,195; Horizon 2,174/567; Breadth 2,000/676; Parallel 1,807/567; Robustness 3,000/600.

### Models and agents
- **Orchestrator:** Qwen2.5-7B-Instruct
- **Sub-agents:** GPT-OSS-120B at different reasoning-effort settings
- **RLM ablations:** GPT-OSS-20B and DeepSeek-R1-Distill-Qwen-7B
- Available agent types include CoTAgent, self-consistency, DebateAgent, ReflexionAgent, and SearchAgent.

### Public benchmarks
AIME24, AIME25, GPQA, HotpotQA, and BrowseComp+.

### Evaluation
The paper reports Avg@8 accuracy. AIME uses answer matching; other benchmarks use Llama-3.3-70B-Instruct as an evaluator in the reported setup.

## 5. Results and conclusions

### Controlled MASBENCH findings

| Axis | Does MAS outperform SAS? | Interpretation |
|---|---|---|
| Depth | Generally no | A single uninterrupted sequential chain avoids coordination overhead. |
| Horizon | Yes in the weaker-agent setting | Explicit intermediate state passing can help. |
| Breadth | Yes | Fan-in aggregation benefits from specialization. |
| Parallel | Yes | Independent components can be delegated effectively. |
| Robustness | Strongly yes | Cross-checking protects against adversarial failures. |

When the sub-agent is strengthened, the multi-agent advantage shrinks or disappears on most axes because the potential gain no longer compensates for orchestration cost.

### Public-benchmark results

| Benchmark | MAS-Orchestra | Comparison reported in the note | Difference |
|---|---:|---:|---:|
| AIME24 | **66.25%** | DebateAgent 62.08% | +4.17 pp |
| AIME25 | **61.25%** | DebateAgent 57.50% | +3.75 pp |
| GPQA | 65.21% | AFlow 65.43% | approximately equal |
| HotpotQA | **49.00%** | DeepResearchAgent 46.44% | +2.56 pp |
| BrowseComp+ | **11.00%** | DeepResearchAgent 8.56% | +2.44 pp |

On AIME24, the paper reports 51 LLM calls for MAS-Orchestra versus 1,288 for AFlow in the cited comparison, while also obtaining higher accuracy.

## 6. Direct relevance to my thesis

### Role in the thesis
This is one of the most directly relevant controlled studies of **when** multi-agent orchestration helps. It provides a structural vocabulary for discussing task-dependent MAS benefits.

### Relationship to the hypotheses

| Hypothesis | Relevance |
|---|---|
| **H1: MAS wins at high complexity** | Supported for breadth, horizon, parallelism, and robustness in the weaker-agent regime, but not for pure sequential depth. |
| **H2: A threshold exists** | Supports capability-dependent and structure-dependent crossovers, although it does not test my exact equal-token scheduling setting. |
| **H0: No difference** | Remains plausible for depth-dominated tasks and for settings with a sufficiently strong sub-agent. |

### Key implication
The claim that MAS is most effective near the edge of sub-agent competence is important for interpreting my Qwen3 32B experiments: a strong base model may reduce or eliminate a multi-agent advantage even when decomposition is well designed.

### Methodological ideas to reuse
1. Use the five-axis vocabulary to describe whether my scheduling instances contain sequential depth, aggregation breadth, independent parallel components, or robustness demands.
2. Use Degree of MAS as a conceptual way to distinguish single-agent, critic-assisted, and hierarchical conditions.
3. Compare my verify-revise architecture with the paper's robustness and cross-verification mechanisms.
4. Treat Qwen2.5-7B orchestration results as a useful but not identical reference for my Qwen3-8B robustness study.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Evidence that MAS effectiveness depends on task structure. |
| **Related Work** | Controlled SAS/MAS evaluation and the MASBENCH axes. |
| **Methodology** | Vocabulary for positioning interacting scheduling constraints. |
| **Discussion** | Capability boundary, coordination overhead, and depth-dominated caveats. |

## 8. Important differences

| Aspect | MAS-Orchestra | My thesis |
|---|---|---|
| Domain | Mathematical task graphs, QA, and search | Meeting planning and scheduling |
| Task type | Reasoning over synthetic graph structures | Temporal/resource CSP planning |
| Complexity | Five structural axes | Number and interaction pattern of constraints |
| Optimization | GRPO-trained orchestrator | Fixed inference-time prompting architectures |
| Token control | Cost is measured, but total token budgets are not strictly matched | Equal token budget is central |
| Ground truth | String matching or LLM-as-judge | OR-Tools CP-SAT oracle |
| Metric | Binary Avg@8 accuracy | Partial-credit meeting-satisfaction rate |
| Models | Qwen2.5-7B orchestrator with much larger sub-agents | Qwen3 32B and Qwen3 8B in controlled fixed architectures |

## 9. Short quotations

> "MAS are most effective at the edge of sub-agent competence"

> "not all problems benefit from MAS"

> "SAS performance collapses to near-zero accuracy in this adversarial setting"

## 10. Verification notes
- The current arXiv record confirms the author list, v5 date, ICML 2026 status, and the paper's five controlled axes.
- The wording “the first controlled benchmark” has been qualified as the paper's positioning rather than treated as an independently proven priority claim.
- The original hypothesis table overstated H2 by describing a specific low–middle–high nonlinearity as universal. The verified version states the safer result: crossover depends jointly on task structure and sub-agent capability.
- The results should not be read as “MAS wins on every high-complexity task”; pure depth and strong-agent conditions are explicit counterexamples.

## 11. BibTeX

```bibtex
@inproceedings{ke2026masorchestra,
  title     = {{MAS-Orchestra}: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks},
  author    = {Ke, Zixuan and Ming, Yifei and Xu, Austin and Chin, Ryan and
               Nguyen, Xuan-Phi and Jwalapuram, Prathyusha and Wang, Jiayu and
               Yavuz, Semih and Xiong, Caiming and Joty, Shafiq},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR}
}
```
