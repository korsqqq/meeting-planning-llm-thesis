# Parmar et al. (2025) — PlanGEN

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

**Файл:** `parmar2025_plangen.md`  
**Статус для thesis:** релевантная статья для `Related Work`, `Methodology` и `Discussion`  
**Источник:** PDF `arXiv:2502.16111v1`, 22 Feb 2025  
**Важно:** разбор основан только на приложенном PDF. В PDF не указана conference venue; поэтому venue здесь записан как `arXiv preprint`.

---

## 1. Метаданные

| Поле | Значение |
|---|---|
| **Название** | *PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving* |
| **Авторы** | Mihir Parmar, Xin Liu, Palash Goyal, Yanfei Chen, Long Le, Swaroop Mishra, Hossein Mobahi, Jindong Gu, Zifeng Wang, Hootan Nakhost, Chitta Baral, Chen-Yu Lee, Tomas Pfister, Hamid Palangi |
| **Аффилиации** | Google; Arizona State University |
| **Год** | 2025 |
| **Дата версии PDF** | 22 Feb 2025 |
| **Venue** | arXiv preprint; conference/journal venue в PDF не указан |
| **arXiv ID** | `2502.16111v1` |
| **Primary class** | `cs.AI` |
| **BibTeX key** | `parmar2025plangen` |
| **Использует Qwen?** | Нет. В PDF Qwen не упоминается. Основные эксперименты: Gemini-1.5-Pro; дополнительные case studies: Gemini-2.0-Flash и GPT-4o. |

---

## 2. Одна строка

PlanGEN показывает, что multi-agent framework с `constraint agent`, `verification agent` и `selection agent` может улучшать natural-language planning и reasoning за счёт constraint-guided verification и выбора inference-time algorithm в зависимости от instance-level complexity.

---

## 3. Ключевые аргументы

1. **Главная проблема:** существующие agent frameworks и inference-time algorithms часто плохо справляются со сложным planning, потому что verification недостаточно учитывает instance-specific constraints, а выбор алгоритма не адаптируется к complexity конкретного instance.

2. **Главная идея:** PlanGEN добавляет три роли: `constraint agent` извлекает ограничения, `verification agent` оценивает plan по этим ограничениям и выдаёт reward score, а `selection agent` выбирает между `Best of N`, `Tree-of-Thought` и `REBASE`.

3. **Сильная методологическая мысль:** качество planning зависит не только от количества LLM calls или от того, что система multi-agent, а от того, есть ли строгая проверка constraints и правильный выбор стратегии под сложность задачи.

4. **Результат:** PlanGEN улучшает результаты на NATURAL PLAN, OlympiadBench, DocFinQA и GPQA относительно baselines; особенно важен результат, что `Mixture of Algorithms` лучше работает на более complex problems.

5. **Ограничение:** статья обсуждает overhead через LLM calls, но не ставит строгий equal token budget и не сравнивает напрямую с сильным single ReAct agent в тех же budget conditions.

---

## 4. Методология

### 4.1. Задачи / benchmarks

Статья использует четыре benchmarks:

| Benchmark | Что проверяет | Размер evaluation set по PDF |
|---|---:|---:|
| **NATURAL PLAN** | natural-language planning: calendar scheduling, meeting planning, trip planning | Calendar Scheduling: 1000; Meeting Planning: 1000; Trip Planning: 1600 |
| **GPQA** | graduate-level scientific reasoning | Diamond set: 198 instances |
| **OlympiadBench** | text-only mathematical/scientific reasoning | MATH: 674; PHY: 236 |
| **DocFinQA** | financial document reasoning | 922 instances |

NATURAL PLAN особенно релевантен для моей thesis, потому что содержит `calendar scheduling`, `meeting planning` и `trip planning`, то есть задачи, близкие к constraint-based planning.

### 4.2. Модели

- Основная модель: **Gemini-1.5-Pro**.
- Дополнительные case studies для model-agnostic claim: **Gemini-2.0-Flash** и **GPT-4o**.
- **Qwen не используется и не упоминается**.

Это важно: статья близка по типу задачи, но не по model stack. В моей thesis будут self-hosted **Qwen3 32B** и **Qwen3 8B**, поэтому результаты PlanGEN нельзя напрямую переносить на мой сетап.

### 4.3. Baselines

Статья сравнивает PlanGEN с:

- `Zero-shot Chain-of-Thought (CoT)`;
- `Vanilla Multi-Agent Baseline`, где та же модель итеративно улучшает ответ через feedback loop;
- отдельными model baselines на некоторых benchmarks, например Gemini/GPT/Claude results.

Сильного single ReAct baseline под равным token budget в статье нет. Это ключевое отличие от моей thesis.

### 4.4. Архитектура PlanGEN

PlanGEN состоит из трёх agents:

| Agent | Функция |
|---|---|
| **Constraint agent** | Extracts instance-specific constraints from the problem statement. |
| **Verification agent** | Evaluates generated plans against constraints and assigns reward score from `-100` to `100`. |
| **Selection agent** | Selects the best inference-time algorithm using modified UCB with LLM-guided priors. |

### 4.5. Варианты PlanGEN

Статья тестирует четыре variants:

1. **PlanGEN (Best of N)** — генерирует несколько candidate plans и выбирает plan с максимальным reward.
2. **PlanGEN (ToT)** — строит tree of reasoning/planning steps, оценивает steps через verification agent.
3. **PlanGEN (REBASE)** — использует reward-guided tree search / pruning.
4. **PlanGEN (Mixture of Algorithms)** — selection agent динамически выбирает между Best of N, ToT и REBASE.

### 4.6. Метрики

| Benchmark | Metric |
|---|---|
| NATURAL PLAN | Exact Match (EM) |
| OlympiadBench | micro-average accuracy |
| GPQA | accuracy |
| DocFinQA | accuracy + F1-score |

В моей thesis метрика отличается: **meeting-satisfaction rate**, то есть partial credit relative to solver optimum. Это сильнее подходит для planning, потому что не превращает весь plan в бинарный exact match.

### 4.7. Hyperparameters / setup

По PDF:

- temperature всех agents для deterministic behavior: `0`;
- `PlanGEN (Best of N)`: 5 samples, temperature `0.7`;
- `PlanGEN (ToT)`: 3 children per root node, temperature `0.7`;
- `REBASE`: initial width `10`, temperature `0.7`, width decremented after each expansion;
- verification score: от `-100` до `100`;
- high-quality threshold example: `95` или выше.

---

## 5. Результаты и выводы

### 5.1. Главные результаты

По abstract и main results, PlanGEN достигает улучшений относительно strongest baseline:

| Benchmark | Reported improvement |
|---|---:|
| NATURAL PLAN | примерно `+8%` average across categories |
| OlympiadBench | примерно `+4%` |
| DocFinQA | примерно `+7%` |
| GPQA | примерно `+1%` |

На NATURAL PLAN лучший вариант — **PlanGEN (Best of N)**:

| NATURAL PLAN task | Best reported PlanGEN score |
|---|---:|
| Calendar | `60.70 EM` |
| Meeting | `43.80 EM` |
| Trip | `41.63 EM` |

На GPQA лучший вариант — **PlanGEN (Mixture of Algorithms)** с `59.6% accuracy`. Это важно, потому что individual inference-time algorithms показывают ниже performance, а значит selection может быть полезным.

### 5.2. Complexity analysis

Самый важный для моей thesis результат: PlanGEN анализирует performance across complexity levels.

- Для **calendar scheduling**: `PlanGEN (ToT)` лучше на simple problems, `PlanGEN (Best of N)` лучше на intermediate problems, а `PlanGEN (Mixture of Algorithms)` лучше при росте complexity.
- Для **meeting planning**: `PlanGEN (Best of N)` лучше на simple/intermediate problems, а `PlanGEN (Mixture of Algorithms)` лучше на complex problems.
- Для **trip planning**: `PlanGEN (Best of N)` и `PlanGEN (Mixture of Algorithms)` consistently outperform other approaches, но в very complex cases все algorithms работают плохо.

Это прямо поддерживает идею моей thesis: нужно искать не общий ответ “MAS лучше или хуже”, а **crossover point** относительно task complexity.

### 5.3. Verification agent

Статья показывает, что reward score verification agent коррелирует с successful outcome. На DocFinQA и GPQA авторы строят logistic regression / KDE analysis и показывают, что высокие reward values связаны с большей probability of success.

Для моей thesis это полезно как аргумент, что отдельная verification stage может быть не просто “лишним агентом”, а реальным quality-control mechanism. Но в моей работе лучше заменить или дополнить LLM-verification внешним `OR-Tools CP-SAT solver` / hidden validator, потому что planning constraints можно проверять формально.

### 5.4. Overhead

Статья обсуждает overhead через **number of LLM calls**. Например, single-agent CoT требует один call, а PlanGEN variants требуют больше calls. Для ToT и REBASE каждый explored path может включать calls для step generation, reward evaluation и completion verification.

Но это не equal token budget. Поэтому для моей thesis эта статья является motivation, а не прямым доказательством cost-efficiency MAS.

### 5.5. Limitations из статьи

Авторы сами отмечают:

- selection strategy опирается на predefined heuristics и может не generalize optimally across tasks/domains;
- computational overhead требует оптимизации для real-world applications;
- будущая работа может использовать reinforcement learning / meta-learning для adaptive agent strategies;
- возможны расширения на multi-modal и multi-lingual reasoning.

---

## 6. Прямая связь с моим thesis

### Как я буду на неё ссылаться

Эту статью можно использовать как **близкую work on multi-agent planning with verification and complexity-adaptive strategy selection**.

Формулировка для thesis:

> Parmar et al. (2025) propose PlanGEN, a multi-agent framework that extracts constraints, verifies generated plans, and dynamically selects inference-time algorithms based on instance-level complexity. Their results suggest that adaptive verification and selection can improve planning performance, but they do not evaluate a strong single ReAct baseline under an equal token budget.

### Что поддерживает относительно H1/H2/H0

Я использую следующие рабочие гипотезы:

- **H1:** MAS начинает превосходить single agent после определённого complexity threshold.
- **H2:** выигрыш MAS может оправдать overhead, если смотреть на quality-cost trade-off.
- **H0:** после строгого budget control у MAS нет устойчивого преимущества.

| Hypothesis | Поддержка от PlanGEN | Комментарий |
|---|---|---|
| **H1** | **частично поддерживает** | Complexity analysis показывает, что `Mixture of Algorithms` лучше на более complex calendar/meeting tasks. Это близко к идее threshold/crossover. |
| **H2** | **слабо / косвенно поддерживает** | Есть discussion of LLM calls vs performance, но нет equal token budget, token-normalized metric или latency-normalized metric. |
| **H0** | **не опровергает напрямую** | Baseline не является strong ReAct agent under equal budget. Кроме того, Best of N иногда лучше сложной MAS-логики, что предупреждает против простого вывода “more agents = better”. |

### Методологические заимствования

1. **Constraint extraction before verification**  
   В моей thesis можно формально определить constraints через generator + CP-SAT solver, но идея отдельного constraint layer полезна.

2. **Separate verification/critic stage**  
   Это поддерживает мои условия `ReAct+verify-revise`, `planner+critic` и `hierarchical MAS`.

3. **Complexity-stratified evaluation**  
   Статья показывает важность анализа performance по complexity levels, а не только average score.

4. **Ablation of components**  
   Можно повторить идею ablation: single agent vs single+critic vs MAS, чтобы отделить эффект decomposition от эффекта verification.

5. **Cost-performance plot**  
   Их LLM-calls-vs-performance plot можно адаптировать в моей thesis как tokens/latency/calls vs meeting-satisfaction rate.

6. **Model-agnostic evaluation**  
   Их Gemini/GPT case study мотивирует мой robustness check с Qwen3 8B после main Qwen3 32B.

---

## 7. Где цитируется в thesis

| Раздел thesis | Как использовать |
|---|---|
| **Introduction** | Как motivation: natural-language planning remains hard, especially when verification and instance-level complexity matter. |
| **Related Work** | Как work on multi-agent planning with constraint-guided verification and inference-time algorithm selection. |
| **Methodology** | Как inspiration для separate verification stage, complexity-based evaluation и cost-performance analysis. |
| **Discussion** | Как comparison: PlanGEN получает gains, но не контролирует equal token budget и не сравнивает с strong ReAct baseline. |

---

## 8. Важные различия

Эти различия нужно подчеркнуть в `Related Work`, чтобы показать contribution моей thesis:

1. **Нет equal token budget.**  
   PlanGEN обсуждает LLM calls, но не заставляет single-agent и multi-agent systems работать под одинаковым token cap.

2. **Нет сильного single ReAct baseline.**  
   Основной single-agent baseline — Zero-shot CoT, а не полноценный ReAct agent с теми же tools и возможностью planning/checking.

3. **Другая MAS-архитектура.**  
   PlanGEN — это constraint/verification/selection framework. Моя MAS — hierarchical system: supervisor, workers, aggregator, critic.

4. **Другая метрика.**  
   PlanGEN использует EM/accuracy/F1. Моя thesis использует `meeting-satisfaction rate` with partial credit vs solver optimum.

5. **Другая ground truth philosophy.**  
   PlanGEN использует LLM-based verification/reward. Моя thesis использует external formal solver / hidden validator через OR-Tools CP-SAT.

6. **Другая complexity definition.**  
   В PlanGEN complexity часто задаётся number of people/days/cities. В моей thesis complexity = number of interacting constraints, not number of people.

7. **Другая model stack.**  
   PlanGEN использует Gemini/GPT-4o. Моя thesis использует self-hosted Qwen3 32B and Qwen3 8B with vLLM.

8. **Overhead измеряется слабее.**  
   PlanGEN рассматривает LLM calls, но не делает строгий analysis по total tokens, repeated context, tool schemas, inter-agent messages и latency.

9. **Best of N часто оказывается очень сильным.**  
   На NATURAL PLAN лучший overall variant — PlanGEN (Best of N), что важно для моей thesis: complex MAS нужно сравнивать не только с простым baseline, но и с сильным single-agent / sampling / verify baseline.

---

## 9. Точные цитаты

Все цитаты ниже строго короче 15 слов.

1. “PlanGEN consists of three specialized agents”

2. “constraint-guided iterative verification improves inference-time algorithms”

3. “all algorithms exhibit poor performance”

---

## 10. BibTeX запись

```bibtex
@misc{parmar2025plangen,
  title        = {PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving},
  author       = {Parmar, Mihir and Liu, Xin and Goyal, Palash and Chen, Yanfei and Le, Long and Mishra, Swaroop and Mobahi, Hossein and Gu, Jindong and Wang, Zifeng and Nakhost, Hootan and Baral, Chitta and Lee, Chen-Yu and Pfister, Tomas and Palangi, Hamid},
  year         = {2025},
  eprint       = {2502.16111},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  note         = {arXiv:2502.16111v1},
  url          = {https://arxiv.org/abs/2502.16111}
}
```

---

## Короткая готовая заметка для Related Work

Parmar et al. (2025) introduce PlanGEN, a multi-agent framework for complex planning and reasoning that combines constraint extraction, plan verification, and adaptive inference-time algorithm selection. Their results on NATURAL PLAN are especially relevant because they include calendar, meeting, and trip planning tasks and show that different algorithms perform better at different complexity levels. However, their setup differs from this thesis because it does not compare against a strong ReAct agent under an equal token budget and relies on EM/accuracy rather than partial satisfaction against a solver optimum.

---

## Как это влияет на дизайн моей thesis

PlanGEN усиливает мой аргумент, что comparison должен быть **complexity-aware**. Но моя thesis закрывает методологический gap: я сравниваю single ReAct и hierarchical MAS при равном token budget, использую formal solver ground truth, считаю partial meeting satisfaction и отдельно измеряю overhead in tokens/calls/latency.
