# Verification Report for the 11 Paper Notes

## Scope

Each Russian note was translated into polished academic English and checked against the corresponding original paper or its current official publication record. The verification focused on:

- title, authors, affiliations, year, venue, arXiv version, and BibTeX;
- datasets, models, baselines, metrics, and experimental conditions;
- headline numerical results and table interpretation;
- claims connecting the paper to the planned thesis;
- wording that was stronger than the source actually supports.

The English files are corrected editions rather than mechanical word-for-word translations. Repetition and informal phrasing were reduced, but the original thesis-oriented structure and substantive content were retained.

## Summary table

| File | Overall match | Main corrections or qualifications |
|---|---|---|
| `tran2026_singleagent` | High | Qualified the Data Processing Inequality argument: it is conditional on the paper's assumptions, not an unconditional real-world performance guarantee. Clarified requested-token control versus hidden provider-side compute. |
| `xie2024_travelplanner` | High, with one material table error | Corrected GPT-4-Turbo two-stage hard-constraint scores: **10.5% micro** and **5.5% macro**. Confirmed ICML 2024. Rephrased the difficulty trend as non-increasing because medium and hard two-stage final scores are tied. |
| `xu2026_oneflow` | High, with a counting error | The paper has **seven main public benchmarks**; TravelPlanner is an additional separate experiment. The original note listed eight while calling them seven. Qualified the workflow-simulation proposition by its assumptions. |
| `yao2023_react` | High | Retained the important caveat that pure ReAct does not beat CoT-SC on HotpotQA. Limited hallucination claims to the paper's manually inspected subset. |
| `zheng2024_naturalplan` | High | Replaced “exponential decline” with “sharp decline”; the paper does not fit or prove an exponential law. Avoided claiming every generated instance has a mathematically unique solution. |
| `amonkar2025_csp` | High | Replaced “degrade equally” with “degrade similarly/substantially.” Restricted the 95% claim to the analyzed semantic constraint errors. Removed an ambiguous quantile sentence that risked reversing the comparison. |
| `cemri2025_mast` | High | Softened “failures come from design rather than model limits” to the supported claim that design, coordination, and verification are major additional sources of failure. Clarified four model families versus several individual model versions. |
| `kambhampati2024_llmmodulo` | Moderate-to-high, with an important interpretation correction | The **~12%** figure is an average GPT-4 result from earlier PlanBench work cited by the position paper, not the maximum result in every 2024 table. Ordinary Blocksworld results can be much higher, while Mystery Blocksworld remains near zero. |
| `ke2026_masorchestra` | High | Qualified “first controlled benchmark” as the authors' positioning. Removed an overly universal threshold interpretation: MAS benefits depend jointly on structural axis and sub-agent strength, and Depth is an explicit counterexample. |
| `parmar2025_plangen` | Outdated metadata corrected; content otherwise high | The work is no longer merely an arXiv preprint: it was published in the **main proceedings of EMNLP 2025**, pages 20640–20666, DOI `10.18653/v1/2025.emnlp-main.1042`. Updated BibTeX. Qualified complexity-dependent rankings as evidence for adaptation, not a strict equal-budget SAS/MAS crossover. |
| `tang2025_complexity` | High | Explicitly framed “unbounded depth improvement” as a result of the paper's probabilistic model and assumptions. Removed an ambiguous approximate percentage statement that was unnecessary for the main conclusion. |

## Most important factual corrections

### 1. TravelPlanner hard-constraint columns
The Russian table placed `5.5%` under hard-constraint micro pass rate. In the original results, GPT-4-Turbo with two-stage ReAct has:

- hard-constraint micro pass rate: **10.5%**;
- hard-constraint macro pass rate: **5.5%**;
- final pass rate: **0.6%**.

### 2. OneFlow benchmark count
The paper states that it evaluates seven main benchmarks:

`HumanEval`, `MBPP`, `GSM8K`, `MATH`, `HotpotQA`, `DROP`, and `Shopping-MMLU`.

`TravelPlanner` is presented later as an additional planning/tool-use experiment. The original note combined all eight items under “seven benchmarks.”

### 3. LLM-Modulo and the 12% result
The approximately 12% GPT-4 result is a cross-domain average from earlier autonomous PlanBench work cited by the ICML position paper. It should not be summarized as “the best model in the 2024 paper reaches only 12% in every setting.” The same paper reports Claude 3 Opus at 59.3% on ordinary zero-shot Blocksworld, while obfuscated Mystery Blocksworld remains approximately 0%.

### 4. PlanGEN publication status
The original note correctly described the February 2025 arXiv v1 available at the time, but current bibliographic information is:

- **Venue:** EMNLP 2025 main conference;
- **Pages:** 20640–20666;
- **Publisher:** Association for Computational Linguistics;
- **DOI:** `10.18653/v1/2025.emnlp-main.1042`.

### 5. Claims that needed qualification
Several claims were directionally reasonable but too categorical:

- DPI does not guarantee that every practical single-agent implementation always beats every MAS.
- NATURAL PLAN establishes a sharp complexity-related decline, not necessarily an exponential function.
- MAST shows that system design is crucial; it does not show that base-model limitations are irrelevant.
- MAS-Orchestra does not imply that MAS wins whenever a task is simply “more complex.” The structural axis and sub-agent capability matter.
- Complexity-dependent method rankings in PlanGEN are not equivalent to a matched-budget architectural crossover experiment.

## Files produced

The folder contains one verified English Markdown file per source note, plus this report and a README. Each translated file includes its own `Verification notes` section documenting paper-specific changes.
