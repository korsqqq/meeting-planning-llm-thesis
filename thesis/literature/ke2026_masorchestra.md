# ke2026_masorchestra

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

---

## 1. Метаданные

| Поле           | Значение |
|----------------|----------|
| **Авторы**     | Zixuan Ke, Yifei Ming, Austin Xu, Ryan Chin, Xuan-Phi Nguyen, Prathyusha Jwalapuram, Jiayu Wang, Semih Yavuz, Caiming Xiong, Shafiq Joty |
| **Аффилиации** | Salesforce Research (1), MIT (2), University of Wisconsin-Madison (3) |
| **Год**        | 2026 |
| **Venue**      | ICML 2026 — Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306 |
| **arXiv ID**   | 2601.14652v5 (v5: 21 May 2026) |
| **BibTeX key** | `ke2026masorchestra` |
| **Qwen?**      | ✅ **ДА** — Qwen-2.5-7B-Instruct используется как orchestrator во всех контролируемых экспериментах (близко к Qwen3 8B из thesis) |

---

## 2. Одна строка

MAS-Orchestra формулирует оркестрацию MAS как holistic function-calling RL-задачу и вводит MASBENCH (5 осей сложности) — первый контролируемый бенчмарк для систематического изучения того, когда MAS превосходит single-agent, обнаруживая, что выигрыш зависит от структуры задачи и возможностей агентов, а не является универсальным.

---

## 3. Ключевые аргументы

1. **MAS-выигрыш не универсален — зависит от структуры задачи.** Авторы выделяют 5 структурных осей (Depth, Horizon, Breadth, Parallel, Robustness) и показывают, что MAS превосходит SAS по Breadth, Parallel и Robustness, но не по Depth (строго последовательные задачи — SAS лучше или равен).

2. **MAS наиболее эффективен «на краю компетентности» sub-agent'а.** Когда sub-agent силён (GPT-120b), прирост MAS исчезает или уходит в минус из-за coordination overhead и error propagation. Когда sub-agent слабее, MAS стабильно выигрывает.

3. **Robustness — единственный axis, где MAS доминирует при любой силе sub-agent'а.** SAS коллапсирует до нуля при adversarial inputs, тогда как MAS через cross-verification и разделение sub-задач остаётся робастным.

4. **Holistic orchestration (вся структура MAS за один шаг) эффективнее sequential (инкрементальное добавление агентов).** MAS-Orchestra достигает >10× снижения затрат (51 LLM call vs 1288 у AFlow на AIME24) при более высокой точности, находясь на Pareto-фронтире.

5. **Instruction-tuned LLM лучше как orchestrator, чем RLM.** RLM-оркестраторы склонны решать задачу самостоятельно и не делегировать sub-agent'ам, даже когда те сильнее — следствие end-to-end training objective RLM.

---

## 4. Методология

### Фреймворк: MAS-Orchestra
- Оркестрация как function-calling RL-задача: два примитива `create_agent` и `create_flow`.
- **Degree of MAS (DoM)**: Low (≤1 sub-agent) vs High (неограниченное количество агентов).
- **Holistic orchestration**: вся структура MAS генерируется за один шаг решения.
- RL-алгоритм: GRPO (Group Relative Policy Optimization).

### Бенчмарк: MASBENCH

| Ось | Определение | Тип координации |
|-----|-------------|-----------------|
| **Depth** | Длина наидлиннейшей цепочки зависимостей | Последовательный |
| **Horizon** | Промежуточные sub-задачи, чьи ответы передаются дальше | Промежуточная верификация |
| **Breadth** | Максимальный in-degree sub-задачи (fan-in) | Агрегация |
| **Parallel** | Независимые компоненты sub-задач (fan-out) | Параллельный |
| **Robustness** | Sub-задачи с adversarial атаками | Верификация + коррекция |

Значения осей: 2–12. Данные: синтетический генератор iGSM (math). Размеры: Depth 3993/1195 train/test, Horizon 2174/567, Breadth 2000/676, Parallel 1807/567, Robustness 3000/600.

### Модели
- Orchestrator: **Qwen-2.5-7B-Instruct** (✅ Qwen!)
- Sub-agents: GPT-OSS-120B (GPT-120b) в режимах low/mid/high reasoning effort
- Для RLM-ablation: GPT-OSS-20B, DeepSeek-R1-Distill-Qwen-7B

### Sub-agents
CoTAgent, SCAgent (self-consistency, 5 samples), DebateAgent (max 5 rounds), ReflexionAgent (critic loop), SearchAgent (multi-turn + DuckDuckGo/BM25).

### Публичные бенчмарки
AIME24, AIME25 (math), GPQA (graduate QA, OOD), HotpotQA (multi-hop QA), BrowseComp+ (search-based QA).

### Метрика
Avg@8 accuracy. Оценка корректности: string matching (AIME) или Llama-3.3-70B-Instruct как LLM-as-judge (остальные).

---

## 5. Результаты и выводы

### Контролируемые эксперименты (MASBENCH, слабый sub-agent = Qwen-7b)

| Axis | MAS > SAS? | Комментарий |
|------|------------|-------------|
| Depth | ❌ Нет | Sequential chains = SAS efficient, MAS adds overhead |
| Breadth | ✅ Да | Fan-in aggregation benefits from specialization |
| Parallel | ✅ Да | Independent sub-tasks — MAS learns parallel structure |
| Horizon | ✅ Да | Intermediate tracking benefits from explicit decomposition |
| Robustness | ✅✅ Явно | SAS collapses, MAS remains robust |

Когда sub-agent сильнее (GPT-120b low): **выигрыш MAS исчезает по всем структурным осям** — coordination cost offset potential gains.

### Публичные бенчмарки

| Benchmark | MAS-Orchestra | Best SAS baseline | Improvement |
|-----------|--------------|-------------------|-------------|
| AIME24 | **66.25%** | DebateAgent 62.08% | +4.17% |
| AIME25 | **61.25%** | DebateAgent 57.50% | +3.75% |
| GPQA (OOD) | **65.21%** | AFlow 65.43% | ≈ equal |
| HotpotQA | **49.00%** | DeepResearchAgent 46.44% | +2.56% |
| BrowseComp+ | **11.00%** | DeepResearchAgent 8.56% | +2.44% |

**Эффективность**: 51 LLM calls vs 1288 (AFlow) на AIME24 — >10× снижение при более высокой точности.

---

## 6. Прямая связь с моим thesis

### Роль в thesis
Это **центральная методологическая опора** для сравнения MAS vs single-agent — единственная работа с контролируемым бенчмарком, специально разработанным для ответа на вопрос «когда MAS лучше SAS?».

### Поддержка гипотез

| Гипотеза | Поддержка |
|----------|-----------|
| **H1** (MAS > single agent при высокой сложности) | Прямая для Breadth/Parallel/Robustness axes; ключевая фраза: "MAS are most effective at the edge of sub-agent competence" |
| **H2** (порог сложности существует) | Прямая: Fig. 3 показывает нелинейный паттерн — при низкой сложности MAS ≈ SAS, при средней MAS выигрывает, при очень высокой (если сильный sub-agent) снова выравнивается |
| **H0** (нет разницы) | Активно опровергается для Parallel/Robustness; поддерживается для Depth |

### Ключевой вывод для thesis
Тезис "MAS are most effective at the edge of sub-agent competence" критически важен: мой Qwen3 32B является относительно сильным агентом → по логике этой статьи, MAS выигрыш может быть меньше, чем ожидается. Это честная нулевая гипотеза для Discussion.

### Методологические заимствования
1. **5-осевая структура** как язык для описания структуры ограничений в моей meeting planning задаче: мои задачи высокой сложности содержат Breadth (несколько взаимозависимых constraint-групп) и Parallel (независимые временные слоты).
2. **DoM-концепция** — обоснование для моего hierarchical MAS condition: выбор между low/high DoM соответствует выбору между single-agent ReAct и плановщик+исполнитель.
3. **Robustness axis** — обоснование для моего verify-revise condition: MAS через internal verification аналогичен их cross-verification между sub-agents.
4. **Оркестратор = Qwen-7b** → близко к моему Qwen3 8B robustness model → прямая сопоставимость.

---

## 7. Где цитируется в thesis

| Раздел | Роль |
|--------|------|
| **Introduction** | "Recent work [Ke et al., 2026] demonstrates that MAS benefits depend critically on task structure rather than holding universally" — мотивация исследовательского вопроса |
| **Related Work** | §Controlled MAS-SAS Comparisons: описание MASBENCH и их 5 осей; единственный предшествующий controlled benchmark |
| **Methodology** | §Complexity Operationalization: мои "interacting constraints" соответствуют комбинации Depth/Breadth/Parallel осей по MASBENCH |
| **Discussion** | §Boundary Conditions: сравнение моего threshold с их "edge of competence"; обсуждение Depth-axis результата как caveat для sequential meeting planning |

---

## 8. Важные различия

> Эти различия необходимо явно указать в Related Work, чтобы показать оригинальный вклад thesis.

| Аспект | MAS-Orchestra (Ke et al.) | Мой thesis |
|--------|--------------------------|------------|
| **Домен задач** | Математика (iGSM), QA, поиск | Meeting planning / scheduling |
| **Тип задач** | Reasoning/QA по структурным графам | CSP scheduling с temporal и resource constraints |
| **Определение сложности** | 5 осей (Depth/Horizon/Breadth/Parallel/Robustness) | Число взаимодействующих ограничений |
| **Обучение** | Training-time: GRPO (RL) на оркестраторе | Inference-time: prompting (ReAct, hierarchical) |
| **Token budget** | Не контролируется (измеряется как cost-metric) | Равный token budget — центральная переменная |
| **Ground truth** | String matching / LLM-as-judge | OR-Tools CP-SAT solver как oracle |
| **Метрика** | Binary accuracy (Avg@8) | Partial credit: meeting satisfaction rate |
| **Sub-agents** | CoT/SC/Debate/Reflexion/Search | Нет pre-built sub-agents; условия — архитектурные паттерны |
| **Задача** | Найти когда MAS > SAS → обучить лучшую систему | Найти complexity threshold для fixed architectures |
| **Модель** | Qwen-2.5-7B orchestrator + GPT-120b sub-agent | Qwen3 32B (main), Qwen3 8B (robustness) |

---

## 9. Точные цитаты

Все цитаты строго < 15 слов, взяты дословно из текста статьи.

> **[Q1]** "MAS are most effective at the edge of sub-agent competence"
> *(p. 7, §5.1)* — центральный тезис для Discussion: мой сильный Qwen3 32B может уменьшить MAS-выигрыш.

> **[Q2]** "not all problems benefit from MAS"
> *(p. 2, §Introduction)* — для Introduction thesis: обоснование нетривиальности вопроса о threshold.

> **[Q3]** "SAS performance collapses to near-zero accuracy in this adversarial setting"
> *(p. 6, §5.1)* — для Related Work / Discussion: единственный axis с бесспорным преимуществом MAS.

---

## 10. BibTeX запись

```bibtex
@inproceedings{ke2026masorchestra,
  title     = {{MAS-Orchestra}: Understanding and Improving Multi-Agent Reasoning
               Through Holistic Orchestration and Controlled Benchmarks},
  author    = {Ke, Zixuan and Ming, Yifei and Xu, Austin and Chin, Ryan and
               Nguyen, Xuan-Phi and Jwalapuram, Prathyusha and Wang, Jiayu and
               Yavuz, Semih and Xiong, Caiming and Joty, Shafiq},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  address   = {Seoul, South Korea},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2601.14652}
}
```
