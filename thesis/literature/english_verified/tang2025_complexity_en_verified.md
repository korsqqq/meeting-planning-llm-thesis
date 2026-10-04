# On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems

## Metadata
- **Authors:** Bohan Tang, Huidong Liang, Keyue Jiang, Xiaowen Dong
- **Equal contribution:** Bohan Tang, Huidong Liang, and Keyue Jiang
- **Affiliations:** University of Oxford; University College London
- **Year:** 2025
- **Venue:** NeurIPS 2025 Workshop on Scaling Environments for Agents (SEA)
- **arXiv:** 2510.04311v1 (5 October 2025)
- **BibTeX key:** `tang2025complexity`

## One-sentence summary
The paper argues theoretically and empirically that the relative benefit of a multi-agent system over a single-agent system grows with task complexity, and that this benefit is more strongly associated with reasoning depth than with capability width.

## Key arguments
- Existing claims that “MAS is better than SAS” often rely on downstream scores without a principled account of **when** and **why** collaboration helps.
- The paper introduces a two-dimensional complexity model:
  - **Depth:** the length of the sequential reasoning chain.
  - **Width:** the breadth of capabilities or knowledge required at each step.
- The empirical MAS is a **multi-agent debate** system in which agents propose, criticize, and refine answers before an aggregator produces the final output.
- The central theoretical claim is that the relative MAS gain increases with both dimensions, but width-related gains saturate whereas depth-related gains can continue to grow in the authors' model.

## Methodology
- **Theory:** The single-agent success probability is modeled as `s(w)^d`; the MAS probability is modeled as `r · [1 − (1 − s(w))^N]^d`. Relative performance gain `Δ` is then analyzed as a function of depth `d` and width `w`.
- The paper derives that `∂Δ/∂d > 0` and `∂Δ/∂w > 0` under its assumptions. It also derives a finite width limit, `(rN)^d − 1`, while the modeled depth gain can become unbounded.
- **Two empirical task families:**
  - **Mathematical reasoning:** DyVal instances represented through tree/DAG structure, with controlled depth and width values between 2 and 4; 900 questions.
  - **Creative writing:** The authors' `DW²` benchmark, where depth is the required number of sentences and width is based on normalized Shannon entropy over keyword domains; 2,500 prompts.
- **Model:** Qwen2.5-32B-Instruct in the reported experiments.
- **Architectures:** SAS uses chain-of-thought prompting; MAS uses debate-style collaboration with four to six total agents, including aggregation.
- **Attribution analysis:** Shapley-R² decomposition estimates how much of the observed gain is explained by depth and width.

## Results and conclusions
- Across both task families, MAS gain increases as controlled complexity increases.
- Depth explains more of the gain than width, supporting the paper's theoretical model.
- The gain is substantially larger in creative writing than in mathematical reasoning. The paper associates this with a much larger solution space and multiple simultaneous content constraints, for which workload distribution and cross-checking can be useful.
- Overall answer quality can remain close while constraint-following differs: a single agent is more likely to omit required elements in the writing task.

## Direct relevance to my thesis
- **Motivation for a crossover hypothesis:** This is a key reference for the claim that a multi-agent advantage may emerge only after task complexity reaches a sufficient level.
- **Different complexity measure:** Tang et al. operationalize complexity through depth and width. My thesis uses the number and interaction structure of scheduling constraints, closer to a CSP-oriented view.
- **No strict equal-token control:** The paper does not isolate architecture under a matched total token budget, which is a central control in my design.
- **Different MAS and domain:** Their system is debate-based and their tasks are mathematics and creative writing. Mine is a hierarchical MAS for tool-based meeting planning.
- **Interaction complexity:** The paper's discussion of dependencies and conflicts among subtasks provides a useful conceptual bridge to my choice of interacting constraints as a complexity axis.
- **Model similarity:** Qwen2.5-32B is close in scale to my Qwen3 32B model, although it is a different model generation.

## Where to cite it in the thesis
- **Introduction / Motivation:** Rationale for expecting a complexity-dependent crossover.
- **Related Work:** Prior studies linking task structure to MAS effectiveness.
- **Discussion:** Contrast between depth/width and interacting-constraint complexity, and the effect of unequal coordination cost.

## Short quotations
- "the benefit of LLM-MAS over LLM-SAS increases with both task depth and width"
- "depth can provide unbounded improvements"

## Limitations useful for Related Work
- Only a debate-style MAS is evaluated, not hierarchical decomposition.
- The empirical tasks are not planning or scheduling tasks.
- Total token budgets and coordination overhead are not strictly matched.
- The main model is Qwen2.5-32B-Instruct, so generalization to much stronger or differently trained models remains open.
- The “unbounded” depth result belongs to the paper's probabilistic model and assumptions; it should not be presented as an unconditional real-world law.

## Verification notes
- The title, authors, venue, depth/width formulation, Qwen2.5-32B setup, and main conclusion match the arXiv and workshop records.
- Statements about unbounded depth gains have been explicitly qualified as theoretical results under the paper's model.
- The approximate “~5%” sentence from the Russian note was removed because its meaning was ambiguous without the table context and was not necessary for the core conclusion.

## BibTeX
```bibtex
@inproceedings{tang2025complexity,
  title={On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems},
  author={Tang, Bohan and Liang, Huidong and Jiang, Keyue and Dong, Xiaowen},
  booktitle={NeurIPS 2025 Workshop on Scaling Environments for Agents (SEA)},
  year={2025},
  note={arXiv:2510.04311}
}
```
