# Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets

## Metadata
- **Authors:** Dat Tran, Douwe Kiela
- **Affiliation:** Stanford University
- **Year:** 2026
- **Status:** Preprint / under review
- **arXiv:** 2604.02460v2 (11 April 2026)
- **BibTeX key:** `tran2026singleagent`

## One-sentence summary
Under a matched thinking-token budget, a single-agent system generally matches or outperforms multi-agent systems on multi-hop reasoning. Much of the apparent MAS advantage in uncontrolled comparisons can therefore be attributed to additional test-time compute rather than architecture alone. MAS becomes competitive when the single agent's effective use of context is deliberately degraded.

## Key arguments
- **Compute confounding:** Many SAS-versus-MAS comparisons allocate more test-time tokens to MAS. This makes it unclear whether gains come from architecture or simply from additional computation.
- **Data Processing Inequality argument:** If inter-agent messages `M` are derived from the original context `C`, then `I(Y; C) ≥ I(Y; M)`. Under the paper's assumptions, a single agent retaining full access to `C` should not be informationally disadvantaged relative to a system that only communicates through `M = g(C)`. This is a conditional information-theoretic argument, not a universal guarantee about every practical implementation.
- **When MAS may help:** Structured decomposition, filtering, and verification can become useful when a single agent cannot reliably exploit a long, noisy, masked, or misleading context.

## Methodology
- **Tasks:** FRAMES and MuSiQue; MuSiQue is restricted to four-hop questions.
- **Controlled resource:** **thinking tokens**, meaning intermediate reasoning tokens rather than prompts or final-answer tokens, under matched requested budgets.
- **Model families:** Qwen3-30B-A3B, DeepSeek-R1-Distill-Llama-70B, and Gemini 2.5 Flash/Pro.
- **Single-agent conditions:** A direct step-by-step reasoning pass and `SAS-L`, which uses a structured pre-answer scaffold under the same budget.
- **Five MAS architectures:** Sequential, Subtask-Parallel, Parallel-Roles, Debate, and Ensemble. The total budget is divided among agents; planner and aggregator stages are designed to remain approximately budget-neutral.
- **Evaluation:** LLM-as-a-judge with a fixed rubric assessing whether the gold answer is semantically present.
- **Budgets:** 100, 500, 1,000, 2,000, 5,000, and 10,000 tokens, with 95% bootstrap confidence intervals.

## Results and conclusions
- **Main result:** SAS is the strongest default architecture under matched requested thinking-token budgets. It is equal to or better than the tested MAS variants at nearly all budgets; the very small 100-token setting is an exceptional regime with almost no room for reasoning.
- The pattern `SAS ≥ Sequential MAS` remains stable in uncapped experiments and across multiple Gemini generations, suggesting that it is not merely an artifact of one checkpoint or one token cap.
- **Crossover under context degradation:** In the Qwen3-30B, 1,000-token experiments with substitution and masking, SAS leads at `α = 0.3`, approaches parity at `α = 0.5`, and Sequential MAS leads at `α = 0.7`. The interpretation is not simply that MAS benefits from longer context; rather, it benefits when the single agent has difficulty separating relevant evidence from distorted evidence.
- **Evaluation diagnostics:** API-side token accounting can diverge substantially from visible reasoning text, especially for Gemini 2.5. The paper reports inflation of up to about `4.7×` in some settings. Consequently, the experiment can reliably control the requested budget, but not necessarily the provider's hidden internal compute.
- **Error analysis:** SAS often succeeds by remaining focused on the question and preserving the correct evidence span through finalization. Sequential MAS can win when broader exploration is followed by effective late-stage verification. A recurring failure is finding the right span but losing it during answer extraction.

## Direct relevance to my thesis
- **Direct predecessor:** The exposé relies on the finding that, under an equal thinking-token budget, a single agent can match or outperform MAS on multi-hop reasoning.
- **My niche addresses an explicit limitation:** The paper studies text-only multi-hop reasoning and places tool-using, multimodal, and other settings outside its scope. My thesis transfers the equal-budget question to tool-based meeting planning with a hierarchical MAS.
- **Support for the null hypothesis:** If MAS does not outperform the single agent in my tested range, the result would extend this paper's skeptical finding to a new task class rather than constitute an uninformative failure.
- **Connection to my crossover hypothesis:** Their crossover is induced through context degradation; mine may arise through a growing number of interacting constraints. Both mechanisms can be framed as conditions under which one uninterrupted reasoning process becomes less reliable.
- **Methodological lessons:**
  - Sequential MAS is a clean comparison point because it resembles a single reasoning process split across explicit messages.
  - The API-budget problem strengthens the case for my self-hosted tokenizer-based accounting.
  - Matched budgets and bootstrap confidence intervals provide a useful template for my analysis.
  - Appendix prompts for the five MAS architectures are useful implementation references.
- **Model overlap:** The paper uses Qwen3-30B-A3B, which is close in scale to my Qwen3 32B setup, although the former is a mixture-of-experts model rather than a dense 32B model.

## Where to cite it in the thesis
- **Introduction / Motivation:** Equal-budget comparison and the central `SAS ≥ MAS` result.
- **Related Work:** The primary budget-controlled SAS-versus-MAS study, contrasted with my planning setting.
- **Methodology:** Justification for matched token budgets and self-hosted token accounting.
- **Discussion:** Comparison between context degradation and interacting-constraint complexity as possible crossover mechanisms.

## Short quotations
- "single-agent systems are more information-efficient"
- "MAS advantages with tools/vision or safety constraints are out of scope"

## Important differences from my work
- Their task: text-only **multi-hop reasoning**; mine: **tool-based meeting planning**.
- Their control: requested **thinking-token budgets** through model APIs; mine: self-hosted tokenizer-based accounting.
- Their MAS variants: sequential, debate, parallel, and ensemble architectures; mine: hierarchical supervisor-worker-aggregator-critic architecture.
- Their scoring: LLM-as-a-judge semantic matching; mine: solver-based partial-credit satisfaction rate.

## Verification notes
- The metadata, task families, model families, central equal-budget finding, DPI framing, and context-degradation result match the current arXiv version.
- The DPI statement has been rewritten more cautiously: it supports an information argument under stated assumptions and should not be presented as an unconditional guarantee that every single-agent implementation will always perform at least as well.
- The distinction between **requested token budget** and unobservable provider-side compute is retained because it is central to the paper's evaluation caveat.

## BibTeX
```bibtex
@article{tran2026singleagent,
  title={Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets},
  author={Tran, Dat and Kiela, Douwe},
  journal={arXiv preprint arXiv:2604.02460},
  year={2026}
}
```
