# kambhampati2024_llmmodulo — LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks

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
| **Авторы** | Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Saldyt, Anil Murthy |
| **Аффилиация** | School of Computing and AI, Arizona State University |
| **Год** | 2024 |
| **Venue** | ICML 2024 — 41st International Conference on Machine Learning, Vienna (PMLR 235) |
| **Тип** | Position paper |
| **arXiv ID** | 2402.01817v3 (12 Jun 2024) |
| **BibTeX key** | `kambhampati2024llmmodulo` |
| **Использует Qwen?** | ❌ Нет — только GPT-4(o/Turbo), Claude-3-Opus, LLaMA-3 70B, Gemini Pro |

---

## 2. Суть в одном предложении

LLM в автономном режиме не способны выполнять планирование или самоверификацию (GPT-4 достигает лишь ~12% правильных планов), но могут играть конструктивную роль как генераторы идей и источники знаний в LLM-Modulo Framework — архитектуре Generate-Test-Critique с внешними звуковыми верификаторами.

---

## 3. Ключевые аргументы

1. **LLM — это pseudo-System 1, не System 2.** Авторегрессивные LLM работают как гигантские приближённые памяти и не способны к принципиальному рассуждению сами по себе: constant-time generation per token физически несовместим с planful reasoning. Это структурное ограничение, не артефакт конкретной модели.

2. **Эмпирическое подтверждение провала автономного планирования.** На PlanBench (PDDL-домены: Blocksworld, Logistics, Mystery BW) лучшая модель GPT-4 генерирует исполнимые планы лишь ~12% случаев. Все протестированные SOTA-модели (GPT-4o, Claude-3-Opus, Gemini Pro, LLaMA-3 70B) демонстрируют "dismal performance". ReAct-style и Chain-of-Thought prompting "largely ineffective" для улучшения этих результатов.

3. **LLM не умеют самоверификации, и итеративный промптинг ухудшает результат.** В задаче graph coloring (NP-complete): LLM столь же плохи при верификации, как и при генерации. Self-critiquing с LLM-as-verifier *хуже* baseline — система не распознаёт случайно правильные планы и отбрасывает их.

4. **LLM-Modulo Framework = Generate-Test-Critique с внешними верификаторами.** Архитектура разделяет роли: LLM генерирует кандидатов, внешние hard critics (model-based verifiers: VAL, CP-SAT и т.д.) проверяют корректность, Meta Controller backprompts LLM. Soundness гарантии исходят исключительно от внешних верификаторов.

5. **Практический прирост подтверждён.** На Blocksworld с VAL как верификатором: 82% success в 15 backprompting-раундах (vs ~35% baseline). На TravelPlanner benchmark: LLM-Modulo даёт **6x улучшение** над baselines (у которых 0.7% pass rate). Ключевое условие успеха — наличие внешнего звукового критика, а не мощность LLM.

---

## 4. Методология

### Оценка автономного планирования (§2)
- **Benchmark:** PlanBench (Valmeekam et al., 2023b) — PDDL-домены IPC: Blocksworld (600 instances), Mystery Blocksworld (deceptive), Logistics
- **Модели:** GPT-4, GPT-4-Turbo, GPT-4o, Claude-3-Opus, LLaMA-3 70B, Gemini Pro
- **Prompting:** one-shot и zero-shot
- **Оценка:** автоматическая (VAL plan validator) — исполнимость плана и достижение цели
- **Mystery BW:** те же задачи с obfuscated именами объектов → проверка, является ли LLM "retrieval machine" vs подлинным планировщиком

### Оценка самоверификации (§2.2)
- **Задача:** Graph Coloring (NP-complete CSP)
- **Модель:** GPT-4
- **Условия:** (a) direct mode — прямое решение; (b) iterative с LLM-как-верификатором; (c) iterative с внешним корректным верификатором
- **Вывод:** только внешний верификатор улучшает результаты

### LLM-Modulo Case Studies (§4)

| Исследование | Benchmark | Верификатор | Результат |
|---|---|---|---|
| Classical planning | PlanBench Blocksworld | VAL (PDDL validator) | 82% за 15 раундов (vs ~35%) |
| Classical planning | PlanBench Logistics | VAL | 70% |
| Travel planning | TravelPlanner (Xie et al. 2024) | Hard constraint critics + commonsense critics | 6x над baseline (0.7% → ~4-6%) |

### Архитектура LLM-Modulo (Figure 3)
```
Problem Spec → LLM (candidate generation) → Plan Blackboard
                                                    ↓
                               Bank of Critics (hard + soft)
                                    ↓              ↓
                              Agreement      Disagreement
                                  ↓              ↓
                            Valid Solution  Meta Controller
                                            (Backprompt) → LLM
```
Роли LLM в системе:
1. Генерация кандидатов планов
2. Реформатирование в синтаксис верификаторов
3. Уточнение неполных спецификаций (с end user)
4. Помощь доменным экспертам в создании моделей критиков

---

## 5. Результаты и выводы

### Автономный режим (Table 1)
| Домен | Лучшая модель | Accuracy |
|---|---|---|
| Blocksworld zero-shot | Claude-3-Opus | **59.3%** |
| Blocksworld one-shot | Claude-3-Opus | 48.17% |
| Mystery BW zero-shot | Все модели | **0–0.16%** |
| Mystery BW one-shot | Все модели | 0.4–4.3% |

→ Obfuscation полностью разрушает performance → LLM делают retrieval, не планирование

### LLM-Modulo vs baseline
- Blocksworld: 35% (zero-shot) → **82%** (15 VAL backprompts)
- TravelPlanner: 0.7% (CoT/ReAct) → ~4–6% (LLM-Modulo, 10 iterations)
- Mystery BW: ~10% — LLM не может генерировать правдоподобных кандидатов, поэтому внешний верификатор не помогает

### Ключевые принципы архитектуры
- LLM soundness ≠ system soundness: гарантии корректности исходят только от external critics
- Completeness зависит от способности LLM генерировать разнообразных кандидатов
- Hard critics (model-based) vs soft critics (LLM-based, для style)
- "LLM-Modulo based agentification with automated critics in the loop significantly improves the performance" (§4)

---

## 6. Прямая связь с тезисом

### Позиционирование в аргументации thesis

> ⚠️ **ИСПРАВЛЕНО 2026-07-26 (аудит THESIS_DECISIONS §8).** Ранее здесь утверждалось, что
> `planner+critic` (C4) — это «конкретная реализация LLM-Modulo», где OR-Tools CP-SAT
> служит in-loop hard critic. Это **противоречит залоченному инварианту**
> THESIS_DECISIONS §4: *«The critic is an LLM with fresh context, **not** CP-SAT — the
> solver is never shown to the agents»*, и общему hard invariant «no oracle access» для
> всех четырёх условий. Ни C3, ни C4 **не являются** реализациями LLM-Modulo, и так их
> описывать нельзя. Правильная формулировка ниже.

**Что тезис берёт у LLM-Modulo, а что — намеренно нет:**

Из работы заимствуется её *soundness*-аргумент: корректность плана обязана исходить от
внешнего формального проверяющего, а не от самой LLM (эмпирика §2.2 — self-critique хуже
baseline). В тезисе это реализовано так: валидность решает скрытый `validator`
(`src/oracle/validator.py`), независимый от солвера, а вердикт **никогда** не показывается
агенту. Плюс скрытый гейт («валиден ∧ строго длиннее fallback») делает плохого критика
безвредным: он может не помочь, но не может ухудшить результат.

Механизм *выигрыша* LLM-Modulo — backprompting-цикл (внешний верификатор многократно
возвращает LLM на доработку: Blocksworld ~35% → 82% за 15 раундов; ~6× на TravelPlanner) —
**сознательно не воспроизводится**. Связывающая причина одна и она дизайнерская, не
бюджетная: **никакая обратная связь формального проверяющего не доходит до модели, и ни один
агент не имеет доступа к оптимуму солвера** (точная формулировка инварианта —
THESIS_DECISIONS §8). Показ диагностики верификатора планировщику изменил бы
экспериментальное условие и смешал сравнение *LLM-архитектур* со сравнением *ремонта с
помощью верификатора*. Equal-budget сам по себе такую систему НЕ запрещает — её можно
запустить под тем же капом с меньшим числом раундов; бюджет лишь ограничивает, сколько
раундов влезет. Следствие для текста: тезис **не даёт** никаких свидетельств об
LLM-Modulo-подобном ремонте; такое условие потребовало бы снятия инварианта и отнесено к
future work.

| Компонент LLM-Modulo | Статус в тезисе |
|---|---|
| LLM candidate generator | Qwen3 32B — есть (воркеры C3 / планировщик C4) |
| Hard critic / model-based verifier **в петле** | **НЕТ (запрещено инвариантом).** CP-SAT существует только harness-side как оракул-знаменатель и как независимый скорер |
| Sound external checker **вне петли** | Есть: скрытый `validator` как молчаливый фильтр (гейт), вердикт агенту не сообщается |
| Backprompt controller (Meta Controller) | **НЕТ.** Ровно один вызов критика, без revision-раунда (§4 C3-4) |
| Soft critics | Есть: LLM-критик C3/C4 — но его выход проходит формальный гейт, а не заменяет его |
| Problem specification | Meeting scheduling constraints (параметризованные генератором) |

**Поддержка H0 (null: MAS/verify не превосходит single agent):**
- ReAct "largely ineffective" для планирования согласно Verma et al. 2024a (цитируется в §2.1) → ожидаем слабые результаты для ReAct baseline при высокой сложности
- Self-critique (ReAct+verify-revise) worsen results в CSP tasks → H0-поддержка для этого условия

**Поддержка H1 (MAS/planner+critic превосходит при высокой сложности):**
- LLM-Modulo показывает 6x improvement на TravelPlanner (constraint-rich задача) → именно в constraint-heavy домене внешний верификатор критически важен
- Ключевой тезис: чем больше constraint interactions, тем более необходим soundness-guaranteeing verifier

**Методологические заимствования (в исправленном виде):**
1. **Soundness-принцип**: корректность решает внешний формальный проверяющий, не LLM. Это
   обоснование для скрытого `validator` как единственного авторитета валидности и для
   solver-based скорера — но НЕ для присутствия солвера в агентной петле.
2. **Разделение hard/soft**: hard-проверка = `validator` harness-side (молчаливый гейт);
   soft-роль = LLM-критик, чьё предложение обязано пройти hard-гейт. Ключевое отличие от
   LLM-Modulo: у них hard critic **говорит с моделью**, у нас — нет.
3. **Meeting scheduling как "constraint-rich" domain**: аналогично TravelPlanner, но с
   формально верифицируемыми ограничениями.
4. **Ceiling-предсказание из Mystery BW**: если модель не генерирует правдоподобных
   кандидатов, никакой внешний контроль не спасает → предсказывает плато на высокой
   сложности (см. ниже «Важное для Discussion»).

### Релевантность к task domain
TravelPlanner (case study §4) — closest существующий benchmark к meeting scheduling:
- Оба имеют natural language constraint specifications
- Оба требуют multi-constraint satisfaction
- LLM-Modulo показал 6x improvement именно там → поддерживает мой выбор domain

---

## 7. Где цитируется в thesis

| Раздел | Функция |
|---|---|
| **Introduction** | Центральная проблема: LLM в autonomous mode плохо планируют → нужны гибридные подходы; мотивация для сравнения архитектур |
| **Related Work § LLM Planning** | Фундаментальное ограничение: ~12% autonomous planning accuracy; ReAct "largely ineffective" |
| **Related Work § LLM-Modulo / Hybrid Approaches** | LLM-Modulo как контрастная точка: гибрид «LLM + внешний верификатор в петле». Явно отметить, что тезис берёт soundness-принцип, но НЕ backprompting-петлю (запрещена no-oracle инвариантом) |
| **Methodology § System Design** | Обоснование того, что валидность решает внешний формальный checker (`validator`), а не LLM. НЕ ссылаться на CP-SAT как на in-loop hard critic — солвер только harness-side |
| **Methodology § Experimental Conditions** | Контраст уровней верификации: C2 — внутренняя рефлексия без инструментов; C3/C4 — свежий контекст + формальный скрытый гейт; LLM-Modulo-стиль (верификатор возвращает модель на доработку) НЕ реализован ни в одном условии |
| **Discussion § Complexity Threshold** | Пример: Mystery BW (~0%) vs Blocksworld (82%) — даже LLM-Modulo не помогает когда LLM не может генерировать кандидатов → complexity ceiling hypothesis |
| **Discussion § ReAct baseline** | Self-critique actively harmful (graph coloring result) → объяснение, почему ReAct+verify-revise может underperform |

---

## 8. Важные различия (для Related Work)

> **Подчеркнуть в тексте Related Work:**

1. **Domain of planning**: Kambhampati использует PDDL-based classical planning (Blocksworld, Logistics) и TravelPlanner. Мой thesis работает с meeting scheduling как constraint satisfaction задачей — это позволяет точнее параметризовать сложность через число interacting constraints, чего нет в PDDL-бенчмарках.

2. **Внешний верификатор**: Kambhampati использует VAL (PDDL plan validator) — domain-specific инструмент. Мой thesis использует OR-Tools CP-SAT solver — general-purpose constraint solver, применимый к любой CSP задаче. Это обобщение дизайна LLM-Modulo.

3. **Отсутствие token budget control**: Kambhampati не фиксирует вычислительный бюджет — LLM-Modulo может использовать произвольное число итераций (до 15 backprompts). Мой thesis обеспечивает **equal token budget** для всех 4 условий — необходимое условие fair comparison.

4. **Нет comparison с hierarchical MAS**: LLM-Modulo — это преимущественно single-agent + verifier архитектура. Мой thesis добавляет hierarchical MAS (planner + specialist subagents) как отдельное условие, что является новым вкладом.

5. **Complexity threshold не изучается**: Kambhampati показывает, что LLM-Modulo помогает в одних доменах (Blocksworld) и не помогает в других (Mystery BW), но не вводит количественную меру сложности. Мой thesis систематически изучает этот порог через parameterized constraint complexity.

6. **Partial credit metric**: Kambhampati использует бинарный pass/fail (VAL верификация). Мой thesis использует meeting satisfaction rate с partial credit — более информативная метрика для scheduling.

---

## 9. Точные цитаты (≤14 слов, для thesis)

> ⚠️ Дословно из текста статьи. Проверены по PDF.

**Q1** — провал автономного планирования (Introduction / Related Work):
> "results in the autonomous mode are pretty bleak"
— (Kambhampati et al., 2024, §2.1) — *8 слов* ✓

**Q2** — риски автономных агентов без планирования (Introduction):
> "acting without the ability to plan is surely a recipe for unpleasant consequences"
— (Kambhampati et al., 2024, §1) — *14 слов* ✓

**Q3** — эффект LLM-Modulo (Discussion / Methodology):
> "LLM-Modulo based agentification with automated critics in the loop significantly improves the performance"
— (Kambhampati et al., 2024, §4) — *13 слов* ✓

---

## 10. BibTeX запись

```bibtex
@inproceedings{kambhampati2024llmmodulo,
  title     = {Position: {LLMs} Can't Plan, But Can Help Planning
               in {LLM-Modulo} Frameworks},
  author    = {Kambhampati, Subbarao and Valmeekam, Karthik and Guan, Lin and
               Verma, Mudit and Stechly, Kaya and Bhambri, Siddhant and
               Saldyt, Lucas and Murthy, Anil},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024},
  publisher = {PMLR},
  note      = {arXiv:2402.01817}
}
```

---

## Дополнительные заметки

### Связь с другими paper notes
- **→ xie2024_travelplanner.md**: TravelPlanner (Xie et al. 2024) используется как case study в §4. Kambhampati показывает 6x improvement LLM-Modulo над методами из TravelPlanner paper. Цитировать оба вместе в Related Work.
- **→ cemri2025_mast.md**: MAST описывает failure modes в MAS — Kambhampati объясняет *почему* они возникают (LLM не умеют планировать/верифицировать). Complementary: один описывает что ломается, другой — почему.
- **→ tran2026_singleagent.md**: Tran & Kiela показывают конкурентоспособность single-agent с MAS — Kambhampati объясняет, что разница не в LLM, а в наличии внешнего верификатора.

### Важное для Discussion
Mystery BW результат (~0% даже с LLM-Modulo) — это "ceiling effect": если LLM не может генерировать правдоподобных кандидатов, внешний верификатор не помогает. Применительно к thesis: при экстремально высокой complexity даже `planner+critic` может не помочь, если LLM теряет способность генерировать feasible кандидатов. Это предсказывает нелинейный или plateau эффект в моей complexity curve.

### Терминологическая связь
- Kambhampati: "approximate knowledge source" = LLM как нечёткий генератор идей
- Мой thesis: Qwen3 32B используется именно в этой роли (воркеры C3, планировщик C4)
- "External sound model-based verifier": в тезисе эту роль играет скрытый `validator`
  **вне** агентной петли (молчаливый гейт). CP-SAT — оракул для знаменателя метрики и
  независимый скорер, он НЕ верификатор в петле и агентам недоступен

- [ ] Добавить в `references.bib` под ключом `kambhampati2024llmmodulo`
- [ ] В Related Work создать параграф, объединяющий Kambhampati 2024 + Xie 2024 (TravelPlanner)
      как контекст для верификационных условий — с явной оговоркой, что backprompting-петля
      в тезисе не реализована
- [ ] В Methodology ссылаться на LLM-Modulo как на источник soundness-принципа, НЕ как на
      реализованную архитектуру (см. THESIS_DECISIONS §7 «Literature claims this design
      deliberately does NOT implement»)
- [ ] В Discussion использовать Mystery BW аналогию для объяснения plateau при высокой complexity

---

*Обработано: 15 июня 2026 | Статус: ✅ Готово к интеграции в thesis | Файл: `kambhampati2024_llmmodulo.md`*
