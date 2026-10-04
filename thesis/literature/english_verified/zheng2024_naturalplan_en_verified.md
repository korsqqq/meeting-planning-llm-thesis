# NATURAL PLAN: Benchmarking LLMs on Natural Language Planning

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
| **Authors** | Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou |
| **Affiliation** | Google DeepMind |
| **Year** | 2024 |
| **Venue** | arXiv preprint in the cited version |
| **arXiv ID** | 2406.04520v1, submitted 6 June 2024 |
| **BibTeX key** | `zheng2024naturalplan` |
| **Qwen used?** | No. The evaluated models are GPT-3.5, GPT-4, GPT-4o, Gemini 1.5 Flash, and Gemini 1.5 Pro. |

## 2. One-sentence summary

NATURAL PLAN benchmarks three natural-language planning tasks—Trip Planning, Meeting Planning, and Calendar Scheduling—and shows that even leading models remain below 50% exact match while performance drops sharply as the number of planning entities and constraints increases.

## 3. Key arguments

1. **Natural-language planning remains difficult for LLMs.** Even when all relevant outputs from Flights, Maps, or Calendar tools are already included in the prompt, the best reported scores are 34.8% for Trip Planning and 48.9% for Calendar Scheduling.

2. **Complexity is a major failure factor.** Performance falls sharply as the number of cities, people, days, and interacting restrictions grows. In Meeting Planning, all tested models fall below 10% for eight or more people; in Trip Planning, all models are below 5% for ten cities.

3. **Prompt-based self-correction can hurt.** The self-correction ablation reduces performance across the evaluated models, with particularly large losses for some stronger systems.

4. **Long-context in-context learning is promising but model-dependent.** Gemini 1.5 Pro continues to benefit from very large numbers of demonstrations and reaches 39.9% on Trip Planning with 800 examples, whereas GPT-4 and Gemini 1.5 Flash stop benefiting or degrade after much smaller demonstration sets.

5. **Easy-to-hard transfer is more useful than hard-to-easy transfer.** Simpler demonstrations can provide strategies that generalize to harder instances, while very complex exemplars are not necessarily the most effective teaching examples.

## 4. Methodology

### Tasks

| Task | Number of examples | Main complexity variable |
|---|---:|---|
| Trip Planning | 1,600 | Number of cities, `N ∈ [3, 10]` |
| Meeting Planning | 1,000 | Number of people, `N ∈ [1, 10]` |
| Calendar Scheduling | 1,000 | Number of participants, `N ∈ [2, 7]`, or number of days, `N ∈ [1, 5]` |

The instances are synthetically constructed from realistic information derived from Google Flights, Google Maps, and Google Calendar. Relevant tool outputs are supplied directly in the model context, so the benchmark primarily evaluates planning rather than live tool execution.

### Models
- GPT-3.5 (`gpt-3.5-turbo-0125`)
- GPT-4 (`gpt-4-turbo-2024-04-09`)
- GPT-4o (`gpt-4o-2024-05-13`)
- Gemini 1.5 Flash
- Gemini 1.5 Pro

### Metric
**Exact Match (EM)** is binary: the generated plan is counted as correct only when all required output fields match the reference solution. No partial credit is awarded.

### Experimental setup
- Main condition: 5-shot prompting.
- Ablations: prompt-based self-correction, easy-to-hard versus hard-to-easy demonstration transfer, and long-context in-context planning with up to 800 demonstrations.

## 5. Results and conclusions

### Main 5-shot results

| Task | GPT-3.5 | GPT-4 | GPT-4o | Gemini 1.5 Flash | Gemini 1.5 Pro |
|---|---:|---:|---:|---:|---:|
| Trip Planning | 7.3% | 31.1% | 3.7% | 25.6% | **34.8%** |
| Meeting Planning | 19.1% | **47.0%** | 45.2% | 23.9% | 39.1% |
| Calendar Scheduling | 19.9% | 41.2% | 43.7% | 34.3% | **48.9%** |

### Main findings
- **Complexity cliff:** All models fall below 5% on ten-city Trip Planning and below 10% on Meeting Planning with at least eight people.
- **Self-correction backfires:** The tested prompt-based correction procedure causes a substantial performance drop rather than a reliable improvement.
- **GPT-4o anomaly on Trip Planning:** GPT-4o scores 3.7%, far below GPT-4's 31.1%. In the paper's small manual analysis of ten GPT-4o errors, seven involve flight-connectivity violations and three involve travel-date violations.
- **Long context:** Gemini 1.5 Pro is the only tested model that continues improving up to the 800-shot, approximately 355k-token condition.

## 6. Direct relevance to my thesis

### Role in the thesis
NATURAL PLAN is the central benchmark predecessor for my Meeting Planning task and provides the closest published formulation to my experimental domain.

### Relationship to the hypotheses

| Hypothesis | Relevance |
|---|---|
| **H1: MAS outperforms a single agent at high complexity** | Indirect support: strong single models deteriorate severely as the number of people and constraints grows, leaving room for structured decomposition to help. The paper itself does not test MAS. |
| **H2: A complexity threshold exists** | Direct motivation: performance curves show a marked decline as instance size grows, including a severe low-accuracy regime for larger meeting instances. |
| **H0: No architecture difference** | Not evaluated because the paper compares models and prompting conditions rather than single-agent and multi-agent architectures. |

### Methodological ideas to reuse
1. **Meeting Planning formulation:** People, travel times, availability windows, durations, and the objective of scheduling meetings provide a direct predecessor to my synthetic generator.
2. **Systematic scaling:** The paper varies the number of entities; my design refines this by parameterizing the number of interacting constraints.
3. **Programmatically validated solutions:** The benchmark is generated so that solutions can be checked exactly. My work uses OR-Tools CP-SAT as an explicit oracle and additionally awards partial credit.
4. **Separating planning from information retrieval:** Supplying relevant information in context isolates planning ability. My controlled slice similarly separates reasoning quality from external data-access variation.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Evidence that current LLMs remain below 50% on demanding natural-language planning tasks. |
| **Related Work** | Primary Meeting Planning benchmark and baseline results. |
| **Methodology** | Predecessor for task structure and controlled complexity scaling. |
| **Discussion** | Comparison between my crossover point and the benchmark's complexity-related collapse. |

## 8. Important differences

| Aspect | NATURAL PLAN | My thesis |
|---|---|---|
| **Metric** | Binary exact match | Partial-credit meeting-satisfaction rate relative to solver optimum |
| **Architecture** | Few-shot prompting of individual models | ReAct, ReAct with verification/revision, hierarchical MAS, and planner/critic conditions |
| **Architecture comparison** | No direct SAS-versus-MAS comparison | Direct single-agent versus multi-agent comparison |
| **Complexity** | Number of people, cities, or days | Number and structure of interacting constraints |
| **Token budget** | Not matched across architectures | Equal total token budget is a central control |
| **Models** | GPT and Gemini models | Self-hosted Qwen3 32B and Qwen3 8B |
| **Self-correction** | Prompt-based ablation | Explicit verify-revise architecture condition |
| **Ground truth** | Exact generated reference solution | OR-Tools CP-SAT oracle with partial-credit scoring |

## 9. Short quotations

> "model performance drops drastically as the complexity of the problem increases"

> "Self-correction leads to significant model performance drop across all models"

> "all models perform below 5% when there are 10 cities"

## 10. Verification notes
- The metadata, dataset sizes, model list, 5-shot scores, and the reported large-instance performance collapse match the original paper.
- The Russian statement that performance falls **exponentially** was too strong. The paper demonstrates and describes a drastic or sharp decline, but does not establish an exponential functional relationship.
- The statement that the instances have a “unique solution” has been avoided here because exact evaluation and programmatic construction do not necessarily imply mathematical uniqueness in every instance.
- Live tool execution is excluded from the benchmark, but it is more precise to say that tool outputs are provided in context than that the task contains no tool-derived information.

## 11. BibTeX

```bibtex
@article{zheng2024naturalplan,
  title     = {{NATURAL PLAN}: Benchmarking {LLM}s on Natural Language Planning},
  author    = {Zheng, Huaixiu Steven and Mishra, Swaroop and Zhang, Hugh and
               Chen, Xinyun and Chen, Minmin and Nova, Azade and Hou, Le and
               Cheng, Heng-Tze and Le, Quoc V. and Chi, Ed H. and Zhou, Denny},
  year      = {2024},
  journal   = {arXiv preprint arXiv:2406.04520},
  url       = {https://arxiv.org/abs/2406.04520},
  note      = {Preprint. Submitted 6 June 2024}
}
```
