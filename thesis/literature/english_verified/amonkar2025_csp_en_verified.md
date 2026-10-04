# A Reality Check of Language Models as Formalizers on Constraint Satisfaction Problems

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
| **Authors** | Rikhil Amonkar*, Ceyhun Efe Kayan*, Qimei Lai, Ronan Le Bras, Li Zhang |
| **Affiliations** | Drexel University; University of Pennsylvania; Allen Institute for AI |
| **Year** | 2025; current cited revision v4 dated 31 March 2026 |
| **Venue** | arXiv preprint / under review |
| **arXiv ID** | 2505.13252v4 |
| **BibTeX key** | `amonkar2025csp` |
| **Qwen used?** | Yes. Qwen3-32B and Qwen2.5-32B are evaluated, including on NATURAL PLAN Meeting Planning. |

## 2. One-sentence summary

A systematic comparison of LLM-as-solver and LLM-as-formalizer approaches on four CSP-style benchmarks and six models finds that formalization underperforms direct solving in 15 of 24 model-dataset combinations, while both approaches deteriorate substantially as the number of constraints grows.

## 3. Key arguments

1. **LLM-as-formalizer is not generally better than LLM-as-solver.** Formalization loses in 15 of 24 model-dataset combinations; for reasoning models specifically, it loses in 12 of 16 combinations.

2. **Both approaches degrade with complexity.** Although generating a formal model should theoretically reduce the search burden, formalization does not show the expected robustness as constraint count increases.

3. **The dominant problem is semantic rather than syntactic.** In the analyzed Qwen3 errors, incorrectly represented constraints account for 95% of the relevant semantic failures. Revision can repair syntax errors or missing outputs, but frequently leaves a semantically wrong model intact.

4. **Reasoning models often solve instead of formalizing.** A substantial share of the reasoning trace contains enumeration, backtracking, or direct problem solving rather than code-oriented constraint construction. This leads to hard-coded answers and other spurious behavior.

5. **Python often outperforms Z3 generation.** Free-form Python formalization performs better than generated SMT/Z3 code in 15 of 24 combinations, despite SMT's conceptual suitability for CSPs.

## 4. Methodology

### Tasks and data

| Dataset | Domain |
|---|---|
| NATURAL PLAN — Calendar Scheduling | Time-window CSP |
| NATURAL PLAN — Trip Planning | Day/city assignment CSP |
| **NATURAL PLAN — Meeting Planning** | Person, start-time, duration, and travel constraints |
| ZebraLogic | Logic-grid puzzles |

The study samples 100 instances per NATURAL PLAN domain. The authors manually annotate constraints and judge a generated solution by whether it satisfies all annotated constraints rather than by string match alone.

### Models
- **Reasoning models:** DeepSeek-R1, Qwen3-32B, o3-mini-high, GPT-5
- **Non-reasoning models:** DeepSeek-V3, Qwen2.5-32B

### Pipelines
- **LLM-as-solver:** Directly produces the solution in a one-shot condition without an explicit chain-of-thought instruction.
- **LLM-as-formalizer with Python:** Generates and executes a Python program.
- **LLM-as-formalizer with SMT/Z3:** Generates a solver program using Z3.
- Formalizer variants are tested with and without revision, allowing up to five attempts when execution fails or no plan is produced.

### Metric and complexity
The primary score is the percentage of outputs that satisfy **all** annotated constraints. Complexity is measured by constraint count and stratified into five quantiles.

### Manual analysis
- 80 outputs are categorized as execution error, no plan, wrong plan, or correct.
- 160 reasoning traces are labeled as code-related, solver-like, or spurious.

## 5. Results and conclusions

### RQ1: Formalizer versus solver

| Group | Combinations where the formalizer wins |
|---|---:|
| Reasoning models | 4 of 16 |
| Non-reasoning models | 4 of 8 |
| Overall | 9 of 24 |

The formalizer therefore fails to outperform direct solving in most tested combinations.

### RQ2: Robustness to complexity
Both direct solving and formalization decline as constraint count rises. The paper's quantile analysis does not support the expectation that program generation is systematically protected from high-complexity instances.

### RQ3: Why formalization fails
- Syntax failures are relatively rare and are often repaired by revision.
- “No plan” cases can also be reduced through revision.
- **Wrong plans persist** because a syntactically valid program can encode the wrong semantics.
- Wrongly defined constraints dominate Qwen3's analyzed semantic failures.
- Example: a time such as 2:16 may be interpreted as an end time rather than a meeting start time.

### RQ4: Token behavior
- Formalizers often produce fewer reasoning tokens than direct solvers.
- Up to half of DeepSeek-R1 reasoning can be solver-like rather than code-focused; Qwen3 also exhibits this behavior, though at a lower reported proportion.
- Some R1 traces hard-code solutions instead of representing constraints.
- Python generally uses fewer tokens than the SMT condition.

## 6. Direct relevance to my thesis

### Role in the thesis
This paper provides empirical justification for using a direct LLM planning baseline rather than assuming that LLM-generated symbolic formalization is stronger. It also evaluates Qwen3-32B on the same NATURAL PLAN Meeting Planning family that motivates my task.

### Relationship to the hypotheses

| Hypothesis | Relevance |
|---|---|
| **H1: MAS wins at high complexity** | Indirect only. The paper establishes limitations of single-model solving and formalization but does not test MAS. |
| **H2: A complexity threshold exists** | Supports the general premise that performance declines with increasing constraint count, but does not by itself demonstrate an SAS/MAS crossover. |
| **H0: No architecture difference** | Not directly studied. The work instead helps define a strong direct-solver baseline. |

### Practical implications
1. **Direct solver/ReAct baseline:** Direct solution generation is a defensible strong baseline for Meeting Planning.
2. **OR-Tools as evaluator, not generated method:** My use of CP-SAT as a trusted oracle avoids making success depend on whether the LLM can correctly write solver code.
3. **Constraint-based complexity:** Their stratification by constraint count supports my decision to treat constraint structure as the primary complexity axis.
4. **Qwen3 comparison point:** Their Qwen3-32B results provide useful contextual evidence, although prompt format, task distribution, and metric differ from mine.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Motivation for a strong direct-solver baseline rather than assuming a neuro-symbolic formalizer is superior. |
| **Related Work** | Comparison of direct solution generation and formalization for planning/CSP tasks. |
| **Methodology** | Justification for evaluating with an external solver oracle while keeping agents in natural-language planning mode. |
| **Discussion** | Comparison of performance decline as constraint count increases and analysis of semantic constraint errors. |

## 8. Important differences

| Aspect | Amonkar et al. | My thesis |
|---|---|---|
| Systems compared | LLM-as-solver versus LLM-as-formalizer | Single ReAct versus hierarchical MAS and intermediate verification conditions |
| External solver | Python/Z3 as an LLM-generated solution mechanism | OR-Tools CP-SAT as a trusted evaluation oracle |
| Agent loop | Primarily one-shot generation plus formalizer revision | ReAct interaction, verify-revise, and hierarchical coordination |
| Multi-agent variable | Not evaluated | Central independent variable |
| Metric | Binary satisfaction of all constraints | Partial-credit meeting-satisfaction rate |
| Token budget | Not strictly matched | Equal total token budget |
| Complexity | Constraint-count quantiles | Number and interaction structure of constraints |

## 9. Short quotations

> "LLM-as-formalizer underperforms LLM-as-solver in 15 out of 24 model-dataset combinations"

> "LLM-as-formalizer still drastically degrades as problem complexity increases"

> "the majority of the semantic errors are due to wrongly defining constraints"

## 10. Verification notes
- The title, authors, arXiv revision, six-model setup, four task families, and the `15/24` central result match the current arXiv record.
- The original note's headline is accurate, but “both approaches degrade equally” is stronger than necessary. The paper supports **similar substantial degradation**, not literal equality at every point.
- The statement “95% of Qwen3 errors” has been narrowed to **the analyzed semantic constraint errors**, which is the appropriate scope.
- The quantile sentence in the Russian version was linguistically ambiguous; it has been replaced with the paper's safer qualitative conclusion rather than preserving a potentially inverted comparison.

## 11. BibTeX

```bibtex
@article{amonkar2025csp,
  title     = {A Reality Check of Language Models as Formalizers on Constraint Satisfaction Problems},
  author    = {Amonkar, Rikhil and Kayan, Ceyhun Efe and Lai, Qimei and
               {Le Bras}, Ronan and Zhang, Li},
  year      = {2025},
  journal   = {arXiv preprint arXiv:2505.13252},
  url       = {https://arxiv.org/abs/2505.13252},
  note      = {Preprint. Current cited revision v4, 31 March 2026}
}
```
