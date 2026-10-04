# Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline (OneFlow)

## Metadata
- **Authors:** Jiawei Xu, Arief Koesdwiady, Sisong Bei, Yan Han, Baixiang Huang, Dakuo Wang, Yutong Chen, Zheshen Wang, Peihao Wang, Pan Li, Ying Ding
- **Affiliations:** The University of Texas at Austin, Amazon, Emory University, Northeastern University, Georgia Institute of Technology
- **Year:** 2026
- **arXiv:** 2601.12307v1 (18 January 2026)
- **BibTeX key:** `xu2026oneflow`

## One-sentence summary
Most current MAS workflows are homogeneous: their agents use the same base LLM and differ mainly in prompts, tools, and graph position. The paper argues and demonstrates that one LLM can execute such a workflow through a multi-turn conversation with comparable performance and lower inference cost through KV-cache reuse.

## Key arguments
- **Homogeneity as a blind spot:** Many MAS frameworks create separate agent instances even though all agents share the same underlying model.
- **Simulation result:** Under the paper's formal assumptions—including fixed decoding, deterministic tool-side effects, and routing based on observable history—a single-LLM simulator can reproduce a homogeneous workflow without loss of expressivity.
- **KV-cache efficiency:** A single continuing conversation can reuse cached prefixes between role transitions. Separate stateless agent calls repeatedly encode overlapping context, increasing prefill cost.
- **Boundary of the claim:** A single LLM cannot faithfully collapse a genuinely heterogeneous workflow that uses different base models, because their internal states and KV caches cannot be shared.

## Methodology
- **OneFlow:** An automatic workflow-design method using Monte Carlo Tree Search and two meta-LLMs—a Creative Designer and a Critical Reviewer—to optimize the performance-cost trade-off for efficient execution.
- **Seven main public benchmarks:** HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP, and Shopping-MMLU.
- **Additional planning/tool-use evaluation:** TravelPlanner is reported separately as an additional experiment. It should not be counted as an eighth member of the paper's stated seven-benchmark main suite.
- **Models:** GPT-4o-mini as the main executor; Claude 3.5 Haiku for additional and heterogeneous experiments; Qwen3-8B with vLLM and a 16k context window for open-weight KV-cache measurements. Claude 4 Sonnet is used for workflow search/design in the reported setup.
- **Metrics:** Task accuracy, pass@1, F1, solve or success rate, estimated inference cost, latency, and throughput.

## Results and conclusions
- Single-agent execution of homogeneous AFlow and OneFlow workflows matches or slightly exceeds the corresponding multi-agent execution across the main benchmarks.
- **Cost reduction:** Single-LLM execution is substantially cheaper at comparable performance because overlapping prefixes can be reused through the KV cache.
- OneFlow's performance sometimes improves slightly under single-agent execution, which the authors attribute to greater continuity of context.
- **Qwen3-8B experiment:** On HumanEval, single-agent execution maintains or improves pass@1. Although the accumulated conversation contains more input tokens, latency and throughput remain broadly stable because of KV-cache reuse.
- **TravelPlanner:** A single LLM executing AFlow or OneFlow matches the task success rate of the corresponding homogeneous MAS workflow while lowering inference cost.
- **Heterogeneous pilot:** The automatically discovered GPT-4o-mini/Claude-3.5-Haiku workflow is largely bounded by the strongest homogeneous workflow in the pilot study. The authors explicitly present this as a limited pilot, not proof that heterogeneity is never useful.

## Direct relevance to my thesis
- **A central skeptical reference:** Together with Tran and Kiela and the MAST analysis, this paper supports the position that adding more agents does not automatically improve performance.
- **Planning evidence:** Its TravelPlanner result shows that, in a tool-intensive planning benchmark, separate homogeneous agent instances are not necessary to reproduce the workflow's success rate.
- **Different research question:**
  - OneFlow asks whether one LLM can **simulate an already decomposed homogeneous workflow**.
  - My thesis asks whether a hierarchical MAS can **outperform a strong single ReAct agent at some complexity threshold under an equal token budget**.
- **Important baseline distinction:** Their “single agent” receives the workflow decomposition and role sequence. My primary single ReAct baseline does not receive the MAS graph. Conditions that add draft-verify-revise or a separate critic therefore form useful intermediate comparisons.
- **Cost interpretation:** The paper shows that repeated context encoding is a genuine MAS overhead. In my experiments, inter-agent messages count toward the budget, so message passing and re-encoding must be reported rather than treated as free coordination.
- **Methodological reference:** Qwen3-8B, vLLM, 16k context, and latency/throughput reporting are close to my intended stack.

## Where to cite it in the thesis
- **Problem Statement / Introduction:** Evidence that MAS is not automatically superior.
- **Related Work:** Homogeneous workflows, single-agent simulation, and KV-cache reuse.
- **Discussion:** Coordination overhead and repeated context encoding in hierarchical MAS.

## Important differences from my work
- Their question is workflow **simulation**; mine is architecture **crossover and superiority** under equal budgets.
- Their single LLM executes an explicit decomposition; my main ReAct baseline does not inherit the MAS graph.
- Their evaluation spans general code, math, QA, domain reasoning, and TravelPlanner; mine isolates meeting-planning complexity using controlled interacting constraints and solver-based partial credit.
- Their primary comparison focuses on monetary and systems cost rather than strict matched output-token budgets.

## Short quotations
- "a single agent can reach the performance of homogeneous workflows"
- "single-LLM methods cannot capture heterogeneous workflows"

## Verification notes
- The original note incorrectly called the full list “seven benchmarks” while enumerating eight. The paper has **seven main public benchmarks**, with **TravelPlanner added as a separate experiment**.
- The metadata, model assignments, Qwen3-8B/vLLM setup, KV-cache conclusion, and TravelPlanner result match the arXiv paper.
- The simulation claim is stated conditionally and should not be generalized to stochastic tools, hidden state, unrestricted routing, or heterogeneous base models.

## BibTeX
```bibtex
@article{xu2026oneflow,
  title={Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline},
  author={Xu, Jiawei and Koesdwiady, Arief and Bei, Sisong and Han, Yan and Huang, Baixiang and Wang, Dakuo and Chen, Yutong and Wang, Zheshen and Wang, Peihao and Li, Pan and Ding, Ying},
  journal={arXiv preprint arXiv:2601.12307},
  year={2026}
}
```
