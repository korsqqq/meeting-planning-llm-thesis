# Yao et al. 2023 — ReAct: Synergizing Reasoning and Acting in Language Models

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

## 1. Метаданные

| Поле | Значение |
|---|---|
| Полное название | *ReAct: Synergizing Reasoning and Acting in Language Models* |
| Авторы | Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao |
| Аффилиации | Department of Computer Science, Princeton University; Google Research, Brain team |
| Год | 2023 |
| Venue | ICLR 2023, conference paper |
| arXiv ID | arXiv:2210.03629v3 |
| BibTeX key | `yao2023react` |
| Основные модели | PaLM-540B; дополнительные результаты с GPT-3 в Appendix A.1; fine-tuning experiments с PaLM-8B/62B |
| Использует ли Qwen? | Нет. В PDF Qwen не используется и не упоминается как экспериментальная модель. |

## 2. Одна строка

Статья вводит **ReAct** — single-agent prompting paradigm, где LLM чередует **reasoning traces** и **task-specific actions**, чтобы одновременно планировать, искать внешнюю информацию и корректировать действия.

## 3. Ключевые аргументы

1. **Reasoning и acting лучше работают вместе, чем по отдельности.**  
   Reasoning помогает модели планировать, отслеживать прогресс, исправлять ошибки и выбирать следующие действия; actions позволяют получать внешнюю информацию из среды.

2. **ReAct уменьшает hallucination и error propagation по сравнению с чистым CoT.**  
   На knowledge-intensive tasks ReAct может проверять информацию через Wikipedia API, поэтому его траектории становятся более grounded и interpretable.

3. **Action-only agents недостаточны для сложных задач.**  
   Без reasoning traces агент часто не умеет декомпозировать цель, отслеживать состояние и понимать, какое действие нужно дальше.

4. **ReAct работает как few-shot single-agent baseline.**  
   На ALFWorld и WebShop ReAct с одним или несколькими in-context examples превосходит imitation/reinforcement learning baselines, обученные на тысячах траекторий.

5. **ReAct не является универсально лучшим методом.**  
   В HotpotQA чистый ReAct немного уступает CoT, а лучшая стратегия — гибрид ReAct и CoT-SC. Это важно: даже сильный single-agent pipeline зависит от типа задачи и источников информации.

## 4. Методология

### 4.1. Основная идея

Авторы рассматривают agent-environment setup: агент получает observation, выбирает action и строит контекст из прошлых observations/actions. ReAct расширяет action space языковыми reasoning traces:

- обычные действия взаимодействуют со средой;
- thoughts/reasoning traces не меняют среду напрямую;
- reasoning traces обновляют внутренний контекст агента и помогают выбрать следующее действие.

Типичная траектория имеет формат:

```text
Thought: ...
Act: ...
Obs: ...
Thought: ...
Act: ...
Obs: ...
Finish: ...
```

### 4.2. Задачи

| Группа задач | Benchmark | Что проверяется |
|---|---|---|
| Knowledge-intensive reasoning | HotpotQA | multi-hop question answering |
| Fact verification | FEVER | проверка утверждений по Wikipedia |
| Interactive decision making | ALFWorld | text-based household tasks |
| Web navigation / shopping | WebShop | поиск и покупка продукта по инструкции |

### 4.3. Action space

Для HotpotQA и FEVER авторы используют простой Wikipedia API:

| Action | Функция |
|---|---|
| `search[entity]` | возвращает первые предложения страницы или похожие entities |
| `lookup[string]` | ищет следующую sentence с нужной строкой |
| `finish[answer]` | завершает задачу с ответом |

Для ALFWorld и WebShop action space зависит от среды: navigation, object interaction, search, product selection, option selection, buy/finish.

### 4.4. Модели

| Роль | Модель |
|---|---|
| Основные prompting experiments | PaLM-540B |
| Дополнительные эксперименты | GPT-3 в Appendix A.1 |
| Fine-tuning experiments | PaLM-8B и PaLM-62B |
| Qwen | Не используется |

### 4.5. Baselines

| Baseline | Описание |
|---|---|
| Standard prompting | direct answer без thoughts/actions |
| CoT | reasoning-only baseline |
| CoT-SC | self-consistency over CoT samples |
| Act | action-only baseline без thoughts |
| ReAct | reasoning + acting |
| ReAct → CoT-SC | fallback to CoT-SC, если ReAct не завершился |
| CoT-SC → ReAct | fallback to ReAct, если CoT-SC недостаточно уверен |
| BUTLER | imitation learning baseline для ALFWorld |
| IL / IL+RL | imitation learning и reinforcement learning baselines для WebShop |

### 4.6. Метрики

| Benchmark | Метрика |
|---|---|
| HotpotQA | Exact Match (EM) |
| FEVER | Accuracy |
| ALFWorld | Success Rate |
| WebShop | Score и Success Rate |
| Human analysis | категории ошибок: hallucination, reasoning error, search result error, false positive |

### 4.7. Экспериментальный сетап

- HotpotQA: 6 manually composed ReAct exemplars.
- FEVER: 3 manually composed ReAct exemplars.
- CoT-SC: 21 sampled CoT trajectories with temperature 0.7.
- ALFWorld: 134 unseen evaluation games; task-specific setup; несколько prompt permutations.
- WebShop: 500 test instructions.
- Fine-tuning: 3,000 trajectories with correct answers generated by ReAct/baselines.

## 5. Результаты и выводы

### 5.1. HotpotQA и FEVER

| Method | HotpotQA EM | FEVER Accuracy |
|---|---:|---:|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | 64.6 |
| ReAct → CoT-SC | 35.1 | 62.0 |

Вывод: ReAct лучше action-only baseline, но не всегда лучше CoT. Лучшие результаты дают гибридные стратегии, где internal knowledge из CoT-SC комбинируется с external knowledge через ReAct.

### 5.2. Ошибки и hallucination

На HotpotQA авторы вручную анализируют 200 trajectories. Главный вывод:

- CoT чаще даёт hallucinated reasoning/facts.
- ReAct более factual и grounded, потому что использует external knowledge.
- Но ReAct может ошибаться из-за плохих search results или повторяющихся действий.

Ключевые numbers из анализа:

| Категория | ReAct | CoT |
|---|---:|---:|
| False positive в success mode | 6% | 14% |
| Hallucination как failure mode | 0% | 56% |
| Reasoning error как failure mode | 47% | 16% |
| Search result error | 23% | — |

### 5.3. ALFWorld

| Method | Overall Success Rate |
|---|---:|
| Act, best of 6 | 45% |
| ReAct, average | 57% |
| ReAct, best of 6 | 71% |
| BUTLER, best of 8 | 37% |

Вывод: reasoning traces помогают агенту декомпозировать цели, отслеживать состояние и выбирать likely locations для объектов.

### 5.4. WebShop

| Method | Score | Success Rate |
|---|---:|---:|
| Act | 62.3 | 30.1 |
| ReAct | 66.6 | 40.0 |
| IL | 59.9 | 29.1 |
| IL+RL | 62.4 | 28.7 |
| Human Expert | 82.1 | 59.6 |

Вывод: ReAct даёт +10% absolute success rate over the previous best baseline, но всё ещё сильно уступает humans.

### 5.5. Общий вывод статьи

ReAct показывает, что strong single-agent systems могут быть значительно сильнее простых prompting baselines, если они имеют:

- explicit reasoning traces;
- external actions/tools;
- observation feedback;
- interpretable trajectory structure;
- возможность fallback/combination with CoT.

## 6. Прямая связь с моим thesis

### Как именно ссылаться

Эта статья должна быть основной citation для **single ReAct baseline** в thesis:

> В моей работе ReAct используется как canonical strong single-agent architecture: один LLM-agent чередует reasoning, tool/action calls и observations. Поэтому MAS нужно сравнивать не с простым CoT или naive single-agent, а именно с сильным ReAct-style baseline.

### Что поддерживает: H1 / H2 / H0

| Hypothesis | Поддержка |
|---|---|
| H1: MAS начинает превосходить single agent после порога сложности | Прямо не поддерживает. Статья не сравнивает MAS vs single-agent. |
| H2: MAS может быть оправдан, если performance gain перекрывает overhead | Прямо не поддерживает. В статье нет equal token budget, latency/cost analysis или solver-normalized utility. |
| H0: strong single agent остаётся конкурентным, MAS advantage не гарантирован | Поддерживает косвенно. ReAct показывает, что well-designed single-agent prompting может сильно превосходить action-only и learned baselines. |

### Методологические заимствования

1. **ReAct loop как baseline architecture**

```text
Thought → Action → Observation → Thought → Action → Observation → Finish
```

В thesis это можно адаптировать как:

```text
Reason → Tool/Planner Step → Observation/Constraint Check → Revise → Final Schedule
```

2. **Ablation logic**

Статья сравнивает:

- Standard;
- CoT;
- Act;
- ReAct.

В моей работе аналогично можно сравнивать:

- ReAct;
- ReAct + verify-revise;
- planner + critic;
- hierarchical MAS.

3. **Trajectory-level interpretability**

ReAct показывает, что важно не только final answer, но и trajectory:

- где агент искал информацию;
- какие intermediate assumptions сделал;
- где ошибся;
- как observation повлияло на следующий step.

Для моего thesis это полезно при error analysis: invalid meeting, missed constraint, wrong merge, unnecessary tool calls, repeated checking.

4. **Internal vs external knowledge**

В ReAct важна разница между:

- internal reasoning;
- external environment feedback.

В моей постановке похожее разделение:

- internal LLM planning;
- external CP-SAT solver / validator / constraint checker.

5. **Strong baseline requirement**

Главная польза статьи: она задаёт стандарт, что single-agent baseline должен быть сильным. Иначе вывод “MAS beats single agent” будет методологически слабым.

### Как формулировать связь в thesis

> ReAct motivates the single-agent baseline used in this thesis. It shows that interleaving reasoning traces with environment/tool actions produces stronger and more interpretable agents than reasoning-only or action-only prompting. Therefore, the multi-agent system in this thesis is compared against a strong ReAct-style agent rather than a weak direct-prompting baseline.

## 7. Где цитируется в thesis

| Chapter | Как использовать |
|---|---|
| Introduction | Обосновать, почему reasoning + acting является стандартной архитектурой для LLM agents. |
| Related Work | Описать ReAct как canonical single-agent agent framework. |
| Methodology | Обосновать дизайн условия `ReAct`: Thought/Action/Observation loop, tool calls, final answer. |
| Methodology / Baselines | Пояснить, почему single-agent baseline не должен быть simple CoT. |
| Discussion | Обсудить, что strong single-agent может быть трудно превзойти; MAS должен оправдать overhead. |
| Error Analysis | Использовать идею trajectory inspection для анализа ошибок агента. |

## 8. Важные различия

Эти различия нужно подчеркнуть в Related Work, чтобы не смешивать ReAct с моей постановкой.

| ReAct paper | Мой thesis |
|---|---|
| Single-agent prompting paradigm | Сравнение single ReAct vs hierarchical MAS |
| Нет MAS architecture | Есть hierarchical supervisor-worker-critic / planner+critic conditions |
| Нет equal token budget | Equal token budget — центральное условие эксперимента |
| Нет systematic overhead analysis | Измеряются tokens, calls, latency |
| Нет CP-SAT solver optimum | OR-Tools CP-SAT даёт ground truth / optimum |
| Метрики: EM, accuracy, success rate | Meeting-satisfaction rate vs solver optimum |
| Задачи: HotpotQA, FEVER, ALFWorld, WebShop | Meeting planning with interacting constraints |
| Complexity не является главной independent variable | Complexity = number of interacting constraints |
| Модели: PaLM-540B, GPT-3, PaLM-8/62B | Self-hosted Qwen3 32B + Qwen3 8B |
| Важен tool/environment feedback | Важны constraint validation, schedule feasibility, optimum comparison |

### Related Work formulation

> Unlike Yao et al. (2023), this thesis does not propose ReAct itself. Instead, ReAct is used as a strong single-agent baseline. The main research question is whether a hierarchical multi-agent system can outperform this baseline under equal token budgets as constraint interaction complexity increases.

## 9. Точные цитаты

1. “synergize reasoning and acting”

2. “ReAct outperforms Act consistently”

3. “Human aligned and controllable”

Все цитаты короткие и используются только как точные phrases из PDF.

## 10. BibTeX запись

```bibtex
@inproceedings{yao2023react,
  title     = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  author    = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=WE_vluYUL-X},
  note      = {arXiv:2210.03629}
}
```
