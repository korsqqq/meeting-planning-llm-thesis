# xie2024_travelplanner — TravelPlanner: A Benchmark for Real-World Planning with Language Agents

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

| Поле | Значение |
|---|---|
| **Авторы** | Jian Xie\*, Kai Zhang\*, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, Yu Su |
| **\* Equal contribution** | Xie, Zhang (work done during Jian's internship at OSU NLP Group) |
| **Аффилиации** | Fudan University, Ohio State University, Pennsylvania State University, Meta AI |
| **Год** | 2024 |
| **Venue** | ICML 2024 (arXiv v4: 23 Oct 2024; venue не указан явно в PDF, основано на знании) |
| **arXiv ID** | 2402.01622v4 |
| **BibTeX key** | `xie2024travelplanner` |
| **Использует Qwen?** | ❌ Нет — Mistral-7B-32K, Mixtral-8×7B-MoE, Gemini Pro, GPT-3.5-Turbo, GPT-4-Turbo |

---

## 2. Суть в одном предложении

TravelPlanner — первый multi-constraint planning бенчмарк, реалистичный для language agents (1225 travel queries, ~4M data entries, 6 tools), показывающий, что даже GPT-4 достигает лишь 0.6% final pass rate: агенты не справляются с глобальным tracking множества взаимодействующих constraints.

---

## 3. Ключевые аргументы

1. **SOTA LLMs не справляются с multi-constraint planning.** GPT-4-Turbo с ReAct в two-stage режиме: 0.6% final pass rate; все остальные модели — 0% на тестовом наборе. Это при условии, что задачи, которые требуют ~12 минут ручной аннотации от обученного человека, агент генерирует за 1–2 минуты, но без соблюдения constraints.

2. **ReAct и Reflexion недостаточны для multi-constraint задач.** Стратегии, эффективные на простых задачах (math, web nav), не переносятся: agents fail to stay on task, use tools correctly, or track global constraints. Разница между sole-planning и two-stage mode >30% — агенты теряют "когнитивную ёмкость" при multitasking.

3. **Число hard constraints напрямую снижает performance.** По всем уровням difficulty (easy=1 hard constraint, medium=2, hard=3) pass rates ухудшаются с ростом числа ограничений: GPT-4-Turbo final pass rate = 1.1% (easy) → 0.3% (medium) → 0.3% (hard) в two-stage mode. Это прямое подтверждение complexity threshold эффекта.

4. **Global constraints особенно трудны.** Budget (10.1%) и Minimum Nights Stay (46.8%) показывают самые низкие pass rates в two-stage mode — именно constraints, требующие forward-looking и holistic optimization. LLM's авторегрессивная природа не позволяет учитывать будущие ветки разом.

5. **Agents не умеют глобально оптимизировать.** Micro pass rate значительно выше macro — агенты удовлетворяют часть constraints, но проваливают всё комплексно. "Current agents fail to consider multiple constraints holistically" (§4.2).

---

## 4. Методология

### Датасет
- **1,225 queries** (train/val/test = 45/180/1000), 9 групп по 2 критериям: длина (3/5/7 дней) × сложность (easy/medium/hard)
- Сложность определяется числом hard constraints: easy=budget, medium=budget+1 extra, hard=budget+2 extra
- Queries генерировались GPT-4 из JSON-шаблонов; **20 аннотаторов-аспирантов** проверяли feasibility планов ($0.80/план)

### Среда
- **Статическая закрытая sandbox** — данные 2022 года, ~4M записей
- 6 инструментов (Table A.2): CitySearch, FlightSearch, DistanceMatrix, RestaurantSearch, AttractionSearch, AccommodationSearch
- NotebookWrite — инструмент для working memory management

### Constraint taxonomy (Table 1)
| Тип | Примеры | Число |
|---|---|---|
| Environment | Unavailable Transport/Attractions | 2 |
| Commonsense | Within Sandbox, Complete Info, City Route, Diverse Restaurants/Attractions, Non-conf Transport, Min Nights | 8 |
| Hard | Budget, Room Rule, Room Type, Cuisine, Transportation | 5 |

### Evaluation modes
- **Two-stage**: агент сам собирает информацию через tools, затем планирует
- **Sole-planning**: информация предоставлена заранее (evaluates planning only, без tool use)

### Метрики (§3.4)
- **Delivery Rate**: смог ли агент выдать план в лимит шагов (30)
- **Commonsense Pass Rate**: micro (доля прошедших constraints) и macro (все constraints сразу)
- **Hard Constraint Pass Rate**: micro и macro
- **Final Pass Rate**: доля планов, прошедших ВСЕ constraints — основная метрика

### Модели и стратегии
| Модели | Стратегии |
|---|---|
| GPT-3.5-Turbo, GPT-4-Turbo | Direct, ZS-CoT, ReAct, Reflexion |
| Gemini Pro, Mixtral-8×7B-MoE, Mistral-7B-32K | Two-stage: ReAct; Sole: Direct |

---

## 5. Результаты и выводы

### Главные результаты (Table 3, тестовый набор)

| Модель / Стратегия | Delivery | Comm. Pass (micro) | Hard Pass (micro) | Final |
|---|---|---|---|---|
| GPT-4-Turbo two-stage (ReAct) | 93.1% | 63.3% | 5.5% | **0.6%** |
| Все остальные two-stage | — | — | — | 0% |
| Direct GPT-4-Turbo sole-planning | 100% | 80.6% | 44.3% | 4.4% |
| Greedy Search | 100% | 72.0% | 31.8% | 0% |

### Влияние числа hard constraints (Table 4, GPT-4-Turbo)
| Сложность | Final (two-stage) | Final (sole-planning) |
|---|---|---|
| Easy (1 hard constraint) | 1.1% | 8.0% |
| Medium (2 hard constraints) | 0.3% | 2.7% |
| Hard (3 hard constraints) | 0.3% | 2.2% |

→ **Monotonic decrease**: чем больше constraints, тем хуже результат

### Tool-use ошибки (Figure 2, GPT-4-Turbo)
- Argument errors: 37.3%
- Invalid action dead loops: 56.7%
- Same action dead loops: 6.0%

### Failure modes (Figure 3 / §5.3)
1. **Persistent errors**: неверные даты → null results → dead loop → агент сдаётся
2. **Information confusion hallucination**: "Lost in the Middle" — смешение похожих данных (один номер рейса для обеих направлений)
3. **Reasoning-action mismatch**: агент знает, что бюджет превышен, но трогает только мелкие статьи (еда), не трогая транспорт/жильё

### Ключевой вывод
Агенты достигают высоких micro scores по отдельным constraints, но провально низких macro scores — они не умеют рассматривать план as a whole. Holistic multi-constraint reasoning остаётся открытой проблемой.

---

## 6. Прямая связь с тезисом

### Позиционирование в аргументации thesis

TravelPlanner — **ближайший существующий бенчмарк к моей задаче**. Оба используют multi-constraint satisfaction с явными (hard) и неявными (commonsense) constraints, требуют tool use для сбора информации, и оба оценивают качество финального плана. Это делает TravelPlanner обязательным пунктом Related Work и базовой точкой сравнения.

**Поддержка H1 (MAS/planner+critic превосходит single agent при высокой сложности):**
- Table 4: более сложные queries (больше constraints) → хуже results → подтверждение complexity threshold эффекта
- Kambhampati 2024 (§4) показывает, что LLM-Modulo даёт 6x улучшение именно на этом бенчмарке → harder tasks benefit more from verifier

**Поддержка H0 (null: MAS не стабильно превосходит single agent):**
- Абсолютные числа всё равно низкие: 4.4% даже с sole-planning GPT-4 Direct → даже убрав tool use bottleneck, planning per se трудно
- Reflexion (≈ мой ReAct+verify-revise) не значительно лучше ReAct

### Прямые параллели к моему дизайну

| TravelPlanner | Мой thesis |
|---|---|
| Travel planning (multi-day itinerary) | Meeting scheduling (constraint satisfaction) |
| Hard constraints: budget, room rule, cuisine... | Hard constraints: availability, duration, priority... |
| Commonsense constraints: diverse restaurants... | Soft preferences: preferred time slots, contiguity... |
| Final pass rate (all-or-nothing) | Meeting satisfaction rate (partial credit) |
| Easy/medium/hard by constraint count (1/2/3) | Complexity = number of interacting constraints (1..N) |
| Static closed sandbox | Synthetic constraint generator (parameterized) |
| 6 search tools | OR-Tools CP-SAT solver (external verifier) |

### Методологические заимствования
1. **Constraint taxonomy design**: разделение на hard (formalized, verifiable) vs. commonsense (implicit, softer) — применить к my meeting scheduling constraint design
2. **Micro vs. macro pass rate**: различие между "удовлетворил часть meetings" (micro) и "удовлетворил все meetings" (macro) — информирует дизайн моей **meeting satisfaction rate**; использую partial credit (micro-style) для более информативной метрики
3. **Complexity parametrization**: TravelPlanner использует discrete levels (easy/medium/hard); мой thesis улучшает это, создавая **continuous complexity axis** — novel contribution по сравнению с TravelPlanner
4. **Sole-planning vs. two-stage gap**: аналог моего сравнения ReAct (всё сразу) vs. planner+critic (разделённые роли)

---

## 7. Где цитируется в thesis

| Раздел | Функция |
|---|---|
| **Introduction** | Motivation: даже GPT-4 = 0.6% на real-world planning → нужна архитектурная помощь |
| **Related Work § Planning Benchmarks** | TravelPlanner как ближайший prior benchmark; различия — мой thesis вводит continuous complexity + equal token budget + MAS comparison |
| **Related Work § LLM Planning Failures** | Failure modes (dead loops, info confusion, reasoning-action mismatch) аналогичны MAST taxonomy |
| **Methodology § Task Design** | Обоснование выбора meeting scheduling: аналогично travel planning, но с чисто формальными constraints → позволяет точную верификацию OR-Tools |
| **Methodology § Evaluation Metrics** | Micro pass rate → мотивация для partial credit meeting satisfaction rate (vs. binary macro) |
| **Discussion § Complexity Threshold** | Table 4: monotonic decrease с ростом # constraints → baseline empirical evidence для моей complexity threshold hypothesis |
| **Discussion § Single Agent Failures** | TravelPlanner failure modes как аналитическая параллель к наблюдениям в моих трассах |

---

## 8. Важные различия (для Related Work)

> **Обязательно подчеркнуть**, чтобы отделить contribution thesis от TravelPlanner:

1. **Domain & verifiability**: TravelPlanner использует real-world travel domain с commonsense constraints (частично субъективные: "diverse restaurants"). Мой thesis использует meeting scheduling с **полностью формально верифицируемыми constraints** (OR-Tools CP-SAT) — нет серой зоны в оценке, нет human judgment.

2. **Complexity metric**: TravelPlanner определяет сложность как дискретный уровень (easy/medium/hard = 1/2/3 hard constraints). Мой thesis вводит **continuous parameterized complexity** = число interacting constraints — это allows systematic threshold measurement, чего нет в TravelPlanner.

3. **Equal token budget**: TravelPlanner не контролирует вычислительный бюджет. Разные стратегии/модели используют произвольное число API calls. Мой thesis **уравнивает token budget** — необходимое условие для fair comparison.

4. **Architecture comparison**: TravelPlanner сравнивает LLMs и planning strategies (ReAct vs. Reflexion vs. Direct) внутри single-agent paradigm. Мой thesis сравнивает **single-agent vs. hierarchical MAS** — качественно другой вопрос.

5. **Partial credit evaluation**: TravelPlanner использует binary pass/fail per plan (macro) — агент либо выполняет всё, либо нет. Мой thesis использует **meeting satisfaction rate** с partial credit — более информативная метрика для measuring degrees of planning quality.

6. **Synthetic vs. real data**: TravelPlanner использует реальные данные (Kaggle flights, Google Places, Airbnb). Мой thesis использует **синтетический генератор** с параметризованными constraints — обеспечивает полный контроль над complexity.

---

## 9. Точные цитаты (≤14 слов, для thesis)

> ⚠️ Дословно из текста статьи. Проверены по PDF.

**Q1** — провал agents на complex planning (Introduction / Related Work):
> "current language agents are not yet capable of handling such complex planning tasks"
— (Xie et al., 2024, Abstract) — *13 слов* ✓

**Q2** — конкретный failure rate (Introduction / Motivation):
> "even GPT-4 only achieves a success rate of 0.6%"
— (Xie et al., 2024, Abstract) — *9 слов* ✓

**Q3** — multi-constraint holistic failure (Discussion):
> "current agents fail to consider multiple constraints holistically"
— (Xie et al., 2024, §4.2) — *8 слов* ✓

---

## 10. BibTeX запись

```bibtex
@inproceedings{xie2024travelplanner,
  title     = {{TravelPlanner}: A Benchmark for Real-World Planning
               with Language Agents},
  author    = {Xie, Jian and Zhang, Kai and Chen, Jiangjie and Zhu, Tinghui and
               Lou, Renze and Tian, Yuandong and Xiao, Yanghua and Su, Yu},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  year      = {2024},
  publisher = {PMLR},
  note      = {arXiv:2402.01622}
}
```

> ⚠️ Venue ICML 2024 не указан явно в PDF — выведено из контекста (kambhampati2024 цитирует как arXiv, ICML 2024 подтверждается по дате конференции и PMLR 235). Проверить при `/bib-validate`.

---

## Дополнительные заметки

### Ключевая таблица для Discussion
**Table 4 (§5.2)** — самая важная таблица для моего thesis:
- Easy → Medium → Hard: monotonic deterioration при добавлении constraints
- Sole-planning всегда лучше two-stage: tool use bottleneck существен
- Budget constraint (10.1%) и Room Rule (5.6%) — самые трудные: требуют global optimization
- Аналог: "meeting budget" (все встречи в доступное время) vs. "individual meeting" (одна встреча)

### Case C.7 особенно важен
Figure C.7 (Reflexion, sole-planning): агент **знает**, что бюджет превышен ($3247 > $3000), пытается исправить через еду ($9 экономии), но не трогает транспорт/жильё ($1000+), и в конце принимает нарушение. Это FM-2.6 (Reasoning-Action Mismatch) из MAST + FM-3.3 (Incorrect Verification) — аналог возможного failure mode в meeting scheduling где агент знает о конфликте, но "округляет" его.

### Связь с другими paper notes
- **→ kambhampati2024_llmmodulo.md**: TravelPlanner используется как case study; LLM-Modulo показывает 6x improvement. Цитировать вместе.
- **→ cemri2025_mast.md**: Failure modes в §5.3 (dead loops = FM-1.3 Step Repetition, reasoning-action mismatch = FM-2.6, hallucinations = FM-1.1). Использовать MAST как общий язык.
- **→ yao2023_react.md**: ReAct тестируется в TravelPlanner и показывает 0.6% → слабость ReAct для complex multi-constraint tasks.

### Metric design для моего thesis
TravelPlanner использует macro (all-or-nothing) как primary metric с micro как вспомогательной. Я инвертирую приоритеты:
- **Primary: meeting satisfaction rate (micro-style)** = partial credit per satisfied meeting → более информативна для gradient measurements
- **Secondary: perfect schedule rate (macro-style)** = доля задач с 100% meeting satisfaction → сравниваемо с TravelPlanner

- [ ] Добавить в `references.bib` под ключом `xie2024travelplanner`
- [ ] Уточнить venue через DOI поиск перед `/bib-validate`
- [ ] В Related Work написать параграф: "The closest existing benchmark to our task is TravelPlanner... However, TravelPlanner differs from our setup in three key ways: [continuous complexity / equal budget / MAS comparison]"
- [ ] Использовать Table 4 monotonic trend как empirical motivation для H1 в Introduction

---

*Обработано: 15 июня 2026 | Статус: ✅ Готово к интеграции в thesis | Файл: `xie2024_travelplanner.md`*
