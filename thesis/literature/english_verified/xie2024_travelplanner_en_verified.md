# TravelPlanner: A Benchmark for Real-World Planning with Language Agents

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
| **Authors** | Jian Xie*, Kai Zhang*, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, Yu Su |
| **Equal contribution** | Jian Xie and Kai Zhang |
| **Affiliations** | Fudan University; The Ohio State University; Pennsylvania State University; Meta AI |
| **Year** | 2024 |
| **Venue** | ICML 2024 |
| **arXiv ID** | 2402.01622v4, 23 October 2024 |
| **BibTeX key** | `xie2024travelplanner` |
| **Qwen used?** | No. Evaluated models include Mistral-7B-32K, Mixtral-8×7B, Gemini Pro, GPT-3.5-Turbo, and GPT-4-Turbo. |

## 2. One-sentence summary

TravelPlanner is a large benchmark for realistic multi-constraint travel planning, containing 1,225 requests, nearly four million data records, and six search tools. Its central result is that even GPT-4-Turbo with ReAct achieves only a 0.6% final pass rate in the full two-stage setting, indicating severe difficulty with holistic constraint tracking.

## 3. Key arguments

1. **Leading LLM agents fail on realistic multi-constraint planning.** GPT-4-Turbo with ReAct reaches a 0.6% final pass rate in the two-stage setting; the other reported two-stage systems score 0% on the test set.

2. **ReAct and Reflexion do not solve the global-planning problem.** Agents struggle with tool arguments, repeated invalid actions, information management, and maintaining global constraints while simultaneously gathering data and composing an itinerary.

3. **More hard constraints reduce success.** For GPT-4-Turbo in the two-stage setting, final pass rate declines from 1.1% on easy instances to 0.3% on medium and hard instances. The corresponding sole-planning scores also fall from 8.0% to 2.7% and 2.2%.

4. **Global constraints are especially difficult.** Budget, minimum-stay, room, and route constraints require reasoning about the itinerary as a whole rather than validating each item independently.

5. **Micro scores can hide complete-plan failure.** An agent may satisfy many individual constraints while still producing an itinerary that fails the macro requirement. The paper concludes that current agents do not reliably consider multiple constraints holistically.

## 4. Methodology

### Dataset
- **1,225 requests:** 45 training, 180 validation, and 1,000 test instances.
- Instances span nine groups formed by trip length—3, 5, or 7 days—and difficulty—easy, medium, or hard.
- Difficulty is defined by hard-constraint count: easy includes budget; medium adds one extra hard constraint; hard adds two extra hard constraints.
- Queries are generated from structured templates and manually checked for feasibility.

### Environment
- A static, closed sandbox built from approximately four million records, largely drawn from 2022 data.
- Six main retrieval tools: CitySearch, FlightSearch, DistanceMatrix, RestaurantSearch, AttractionSearch, and AccommodationSearch.
- NotebookWrite provides an external scratchpad for information management.

### Constraint taxonomy

| Type | Examples | Count |
|---|---|---:|
| Environment constraints | Unavailable transportation or attractions | 2 |
| Commonsense constraints | Staying inside the sandbox, complete information, valid routes, diversity, non-conflicting transport, minimum nights | 8 |
| Hard constraints | Budget, room rule, room type, cuisine, transportation preference | 5 |

### Evaluation modes
- **Two-stage:** The agent retrieves information with tools and then creates the plan.
- **Sole-planning:** Relevant information is supplied in advance, isolating plan construction from retrieval.

### Metrics
- **Delivery Rate:** Whether the agent produces a plan within the step limit.
- **Commonsense Pass Rate:** Micro and macro satisfaction of commonsense constraints.
- **Hard-Constraint Pass Rate:** Micro and macro satisfaction of hard constraints.
- **Final Pass Rate:** Percentage of itineraries satisfying every required constraint.

### Models and strategies

| Models | Strategies |
|---|---|
| GPT-3.5-Turbo, GPT-4-Turbo | Direct, zero-shot CoT, ReAct, Reflexion, depending on mode |
| Gemini Pro, Mixtral-8×7B, Mistral-7B-32K | ReAct in two-stage evaluation; Direct in sole-planning evaluation |

## 5. Results and conclusions

### Main test-set results

| Model / condition | Delivery | Commonsense micro | Hard micro | Hard macro | Final |
|---|---:|---:|---:|---:|---:|
| GPT-4-Turbo, two-stage ReAct | 93.1% | 63.3% | **10.5%** | **5.5%** | **0.6%** |
| Other reported two-stage systems | — | — | — | — | 0% |
| GPT-4-Turbo, sole-planning Direct | 100% | 80.6% | 44.3% | 23.1% | 4.4% |
| Greedy Search | 100% | 72.0% | 31.8% | — | 0% |

### Effect of hard-constraint count

| Difficulty | Final pass, two-stage | Final pass, sole-planning |
|---|---:|---:|
| Easy: one hard constraint | 1.1% | 8.0% |
| Medium: two hard constraints | 0.3% | 2.7% |
| Hard: three hard constraints | 0.3% | 2.2% |

The sole-planning condition consistently outperforms two-stage planning, showing that tool use and information management impose a large additional burden. Both settings remain difficult even after retrieval is removed.

### Tool-use errors
- Argument errors: 37.3%
- Invalid-action dead loops: 56.7%
- Repetition of the same action: 6.0%

### Representative failure modes
1. **Persistent error loops:** Incorrect dates produce empty results, after which the agent repeats invalid actions or abandons the task.
2. **Information confusion:** Similar records are mixed together, such as reusing one flight number for both directions.
3. **Reasoning-action mismatch:** The agent notices that the budget is exceeded but modifies minor food expenses instead of the dominant accommodation or transportation costs.

### Main conclusion
Agents often achieve respectable micro-level constraint scores but near-zero end-to-end validity. The principal unresolved challenge is holistic planning over many interacting constraints.

## 6. Direct relevance to my thesis

### Position in the argument
TravelPlanner is one of the closest existing benchmarks to my task. Both involve tool-supported planning, hard and softer constraints, and evaluation of a final plan rather than an isolated answer.

### Relationship to the hypotheses
- **H1:** The decline with additional constraints motivates the possibility that decomposition or verification may become more useful at high complexity. TravelPlanner itself does not establish that MAS is the solution.
- **H0:** Even sole-planning GPT-4-Turbo reaches only 4.4%, showing that planning remains difficult after removing the tool-use bottleneck. Reflexion also does not transform performance.
- **External-verifier connection:** Later LLM-Modulo work uses TravelPlanner to motivate explicit constraint critics, which is relevant to my planner/critic condition.

### Direct parallels

| TravelPlanner | My thesis |
|---|---|
| Multi-day itinerary planning | Meeting scheduling |
| Budget, accommodation, cuisine, and transportation constraints | Availability, duration, priority, travel-time, and compatibility constraints |
| Commonsense itinerary requirements | Soft preferences and scheduling quality criteria |
| Binary final pass | Partial-credit meeting-satisfaction rate plus perfect-schedule rate |
| Easy/medium/hard based on 1–3 hard constraints | Continuous or finely graded interacting-constraint complexity |
| Closed retrieval sandbox | Synthetic controlled task generator and external validator |

### Methodological ideas to reuse
1. Distinguish formal hard constraints from softer or commonsense requirements.
2. Report both partial/micro satisfaction and complete/macro validity.
3. Parameterize complexity explicitly rather than reporting only aggregate scores.
4. Analyze retrieval and planning separately when possible.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Demonstrate the very low success of current agents on realistic planning. |
| **Related Work — Planning Benchmarks** | Closest benchmark and a key comparison point. |
| **Related Work — Agent Failures** | Tool loops, information confusion, and reasoning-action mismatch. |
| **Methodology — Task Design** | Motivation for a formally verifiable scheduling domain. |
| **Methodology — Metrics** | Motivation for combining micro-style partial credit with macro complete validity. |
| **Discussion** | Empirical precedent for declining performance as constraints accumulate. |

## 8. Important differences from my thesis

1. **Domain and verifiability:** TravelPlanner includes subjective or commonsense itinerary constraints. My scheduling constraints are designed to be fully machine-verifiable.
2. **Complexity scale:** TravelPlanner uses three discrete hard-constraint levels. My generator provides a finer interaction-based complexity axis intended for crossover estimation.
3. **Budget control:** TravelPlanner does not match total token use across strategies. My architecture comparison does.
4. **Architecture question:** TravelPlanner primarily compares models and single-agent prompting strategies, not a fixed strong SAS against hierarchical MAS.
5. **Metric:** TravelPlanner's main final score is all-or-nothing. My primary metric gives partial credit and retains a separate perfect-schedule metric.
6. **Data control:** TravelPlanner emphasizes realistic records; my synthetic generation emphasizes experimental control and known optima.

## 9. Short quotations

> "current language agents are not yet capable of handling such complex planning tasks"

> "even GPT-4 only achieves a success rate of 0.6%"

> "current agents fail to consider multiple constraints holistically"

## 10. Verification notes
- ICML 2024 is confirmed by the official project and proceedings records; it is not merely an inference from the PDF date.
- The original summary contained an important table error: **5.5% is hard-constraint macro pass rate, not hard-constraint micro pass rate**. The correct GPT-4-Turbo two-stage hard scores are **10.5% micro** and **5.5% macro**.
- The 0.6% final pass rate, 4.4% sole-planning result, dataset size, and main failure modes match the paper.
- Claims about exact human annotation time and individual per-constraint percentages were omitted where they were not necessary for the main argument or were easy to misread outside their exact table context.
- The easy-to-medium-to-hard final score is non-increasing, but the two-stage medium and hard values are tied; “monotonic decrease” should therefore be read as **monotonic non-increase**, not a strict decrease at every level.

## 11. BibTeX

```bibtex
@inproceedings{xie2024travelplanner,
  title     = {{TravelPlanner}: A Benchmark for Real-World Planning with Language Agents},
  author    = {Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and
               Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2024},
  publisher = {PMLR},
  note      = {arXiv:2402.01622}
}
```
