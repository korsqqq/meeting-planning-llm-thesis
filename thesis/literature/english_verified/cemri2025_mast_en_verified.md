# Why Do Multi-Agent LLM Systems Fail?

## 1. Metadata

| Field | Value |
|---|---|
| **Authors** | Mert Cemri*, Melissa Z. Pan*, Shuyi Yang*, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica |
| **Affiliations** | University of California, Berkeley; Intesa Sanpaolo |
| **Equal contribution** | Mert Cemri, Melissa Z. Pan, Shuyi Yang |
| **Year** | 2025 |
| **Venue** | NeurIPS 2025, Datasets and Benchmarks Track |
| **arXiv ID** | 2503.13657v3, 26 October 2025 |
| **BibTeX key** | `cemri2025mast` |
| **Qwen used?** | Yes. Qwen2.5-Coder-32B-Instruct is one of the open-source model variants examined. |

## 2. One-sentence summary

The paper introduces MAST, a taxonomy of 14 multi-agent failure modes in three categories, and MAST-Data, containing more than 1,600 annotated traces from seven MAS frameworks. It shows that failures arise not only from base-model capability but also from system specification, coordination, and verification design.

## 3. Key arguments

1. **Existing MAS frameworks fail frequently.** Across the seven analyzed systems, reported failure rates range from roughly 41% to 86.7%, depending on framework and task.

2. **Fourteen failure modes fall into three broad categories:**
   - **FC1 — System Design Issues:** specification errors, repeated steps, context loss, and termination problems.
   - **FC2 — Inter-Agent Misalignment:** task drift, information withholding, and mismatch between reasoning and actions.
   - **FC3 — Task Verification:** premature completion, incomplete checking, and incorrect validation.

3. **Architecture matters independently of the underlying model.** The same LLM can exhibit different failure profiles under different workflows, demonstrating that model quality alone does not determine MAS reliability.

4. **Verification and coordination trade-offs differ by framework.** MetaGPT and ChatDev, even with the same base model, show different strengths because their role definitions, SOPs, review stages, and test phases differ.

5. **Some modes are strongly associated with failed traces, while others also occur in successful runs.** In particular, lack of termination awareness and information withholding are especially damaging, whereas verification weaknesses may be present without always causing the final task to fail.

## 4. Methodology

### MAST taxonomy
- Developed using a grounded-theory process rather than a predefined list of errors.
- Six expert annotators analyze more than 150 traces during taxonomy development.
- Iterative inter-annotator agreement reaches Cohen's `κ = 0.88`.
- Fourteen modes are organized by category and by pre-execution, execution, and post-execution stage.
- Generalization is tested on previously unseen systems and tasks, with reported agreement around `κ = 0.79` in that validation setting.

### MAST-Data
- More than 1,600 annotated traces; the current version reports seven MAS frameworks.
- Frameworks include MetaGPT, ChatDev, HyperAgent, AppWorld, AG2, Magentic-One, and OpenManus.
- Tasks span coding, mathematics, and general-purpose agents, including ProgramDev, SWE-Bench, GSM-Plus, OlympiadBench, MMLU, GAIA, and AppWorld Test-C.

### Models

| Model family / variant | Type |
|---|---|
| GPT-4 and GPT-4o variants | Closed-source |
| Claude 3 / Claude 3.7 Sonnet variants | Closed-source |
| Qwen2.5-Coder-32B-Instruct | Open-source |
| CodeLlama-7B-Instruct | Open-source |

The arXiv abstract groups these into four model families—GPT-4, Claude 3, Qwen2.5, and CodeLlama—while some detailed experiments distinguish individual versions such as GPT-4 and GPT-4o.

### LLM-as-a-judge pipeline
- OpenAI o1 is prompted with MAST definitions and examples.
- Reported agreement with human annotations is 94% accuracy and Cohen's `κ = 0.77`.
- Annotation cost varies substantially with trace length and framework.

## 5. Results and conclusions

### Failure distribution

| Category | Share | Most frequent mode |
|---|---:|---|
| FC1 — System Design Issues | 44.2% | FM-1.3 Step Repetition, 15.7% |
| FC2 — Inter-Agent Misalignment | 32.3% | FM-2.6 Reasoning-Action Mismatch, 13.2% |
| FC3 — Task Verification | 23.5% | FM-3.3 Incorrect Verification, 9.1% |

### Model and framework comparisons
- In a MetaGPT/ProgramDev comparison, GPT-4o shows fewer system-design and inter-agent failures than Claude 3.7 Sonnet, while verification remains difficult for both.
- With GPT-4o held fixed, MetaGPT has fewer FC1 and FC2 failures than ChatDev, while ChatDev benefits from explicit review and testing phases and has fewer FC3 failures.
- Qwen2.5-Coder-32B performs substantially better than CodeLlama-7B across the reported open-source comparisons, but the stronger closed models still have an advantage.

### Intervention studies

| Intervention | Reported result |
|---|---|
| Improved ChatDev role prompts | +9.4% task success |
| Added high-level verification stage | +15.6% on ProgramDev |
| AG2 topology restructuring | Statistically significant improvement with GPT-4o, `p = 0.03` |

The paper's broader conclusion is that isolated prompt patches can help, but persistent failure patterns often require structural redesign of roles, topology, information flow, and verification.

## 6. Direct relevance to my thesis

### Position in the thesis argument
- **Support for a skeptical null hypothesis:** High failure rates show that coordination overhead does not automatically produce better task performance.
- **Conditional support for MAS:** A well-designed workflow can improve results using the same underlying model, but only when system design avoids recurrent coordination and verification errors.
- **Failure-analysis framework:** MAST offers a ready-made vocabulary for qualitative analysis of my scheduling traces.

### Particularly relevant failure modes

| MAST mode | Meeting-planning interpretation |
|---|---|
| FM-1.3 Step Repetition | Repeatedly testing the same slots without progress |
| FM-1.5 Unaware of Termination Conditions | Failing to recognize that a feasible or optimal stopping condition has been reached |
| FM-2.2 Failure to Request Clarification | Proceeding despite ambiguous or conflicting constraints |
| FM-2.6 Reasoning-Action Mismatch | Correctly identifying a conflict but outputting an incompatible slot |
| FM-3.2 Missing or Incomplete Verification | Checking only some meetings or constraints |
| FM-3.3 Incorrect Verification | Declaring an invalid schedule feasible |

### Methodological ideas to reuse
1. Label traces with a fixed failure taxonomy rather than relying only on anecdotal examples.
2. Compare failure profiles across complexity levels and architectures.
3. Distinguish prompt-level interventions from topology-level interventions.
4. Use automated annotation only after validating it against a human-labeled subset.

## 7. Where to cite it in the thesis

| Section | Function |
|---|---|
| **Introduction** | Evidence that current MAS implementations have high and poorly understood failure rates. |
| **Related Work — MAS Failures** | Primary taxonomy and dataset reference. |
| **Related Work — SAS versus MAS** | Complement to equal-budget studies showing that more agents do not guarantee gains. |
| **Methodology — Error Analysis** | Source of failure labels for trace inspection. |
| **Discussion** | Interpretation of repeated actions, misalignment, and failed verification. |

## 8. Important differences

1. MAST spans coding, mathematics, and general-agent tasks rather than controlled meeting scheduling.
2. It is primarily an observational analysis of existing traces, whereas my thesis runs a parameterized controlled experiment.
3. Token budgets are not equalized across the analyzed systems.
4. MAST does not define a continuous quantitative complexity measure comparable to my interacting-constraint axis.
5. Its research question is why MAS fails; mine is when MAS begins to outperform a strong single-agent baseline, if such a crossover exists.
6. MAST's LLM judge labels failure modes, whereas my primary task score is computed by a formal solver-based evaluator.

## 9. Short quotations

> "MAS failure is not merely a function of challenges in the underlying model"

> "performance gains on popular benchmarks are often minimal"

> "good MAS design requires organizational understanding"

## 10. Verification notes
- The current arXiv version confirms seven frameworks, more than 1,600 traces, 14 modes, three categories, and `κ = 0.88`.
- The exact trace count is reported as 1,642 in the detailed paper materials; the abstract deliberately uses “1600+”.
- The original statement that failures occur “rather than because of base-LLM limitations” was too categorical. The paper shows that **system design is an additional and major source of failure**, not that model capability is irrelevant.
- “One of four tested LLMs” was ambiguous: the abstract names four model families, while detailed tables distinguish multiple model versions. The verified wording avoids an incorrect simple count.
- MAST's LLM-as-a-judge pipeline can motivate trace categorization, but it is not a justification for replacing my deterministic solver-based outcome metric with an LLM judge.

## 11. BibTeX

```bibtex
@inproceedings{cemri2025mast,
  title     = {Why Do Multi-Agent {LLM} Systems Fail?},
  author    = {Cemri, Mert and Pan, Melissa Z. and Yang, Shuyi and
               Agrawal, Lakshya A. and Chopra, Bhavya and Tiwari, Rishabh and
               Keutzer, Kurt and Parameswaran, Aditya and Klein, Dan and
               Ramchandran, Kannan and Zaharia, Matei and
               Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  note      = {Datasets and Benchmarks Track; arXiv:2503.13657}
}
```
