# cemri2025_mast — Why Do Multi-Agent LLM Systems Fail?

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
| **Авторы** | Mert Cemri\*, Melissa Z. Pan\*, Shuyi Yang\*, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica |
| **Аффилиации** | UC Berkeley (основная), Intesa Sanpaolo (Yang) |
| **\* Equal contribution** | Cemri, Pan, Yang |
| **Год** | 2025 |
| **Venue** | NeurIPS 2025, Track on Datasets and Benchmarks |
| **arXiv ID** | 2503.13657v3 (26 Oct 2025) |
| **BibTeX key** | `cemri2025mast` |
| **Использует Qwen?** | ✅ Да — `Qwen2.5-Coder-32B-Instruct` (один из 4 протестированных LLM) |

---

## 2. Суть в одном предложении

Первое эмпирически обоснованное исследование причин отказов в MAS: авторы вводят таксономию MAST из 14 режимов отказа в 3 категориях, проверенную на 1642 аннотированных трассах 7 MAS-фреймворков, и показывают, что большинство отказов происходят из-за ошибок дизайна системы и нарушений координации агентов — а не из-за ограничений базовых LLM.

---

## 3. Ключевые аргументы

1. **Высокий уровень отказов в существующих MAS.** Failure rate составляет 41–86.7% на 7 SOTA open-source фреймворках (MetaGPT, ChatDev, HyperAgent, AppWorld, AG2, Magentic-One, OpenManus) — при этом чёткого консенсуса по причинам нет.

2. **Три категории отказов охватывают 100% наблюдаемых случаев:**
   - FC1 System Design Issues (44.2%) — ошибки спецификации, зацикливание, потеря контекста
   - FC2 Inter-Agent Misalignment (32.3%) — несовпадение логики и действий, дрейф задачи, сокрытие информации
   - FC3 Task Verification (23.5%) — преждевременное завершение, неполная или неправильная верификация

3. **Отказы MAS — это прежде всего проблема дизайна, не модели.** Одна и та же LLM под разными MAS-дизайнами даёт очень разные failure profiles; простые исправления промптов дают лишь скромный прирост (+9.4% / +15.6%), но не решают проблему системно.

4. **Архитектурный дизайн важен не меньше, чем выбор LLM.** MetaGPT vs ChatDev (оба с GPT-4o) показывают противоположные профили: MetaGPT значительно лучше в FC1/FC2, но хуже в FC3 — потому что ChatDev встраивает явные фазы тестирования, а MetaGPT полагается на SOP-ролевые спецификации.

5. **Некоторые отказы "фатальны", другие — нет.** FM-1.5 (Unaware of Termination Conditions) и FM-2.4 (Information Withholding) встречаются почти исключительно в failed traces; FM-3.2/3.3 (верификационные ошибки) присутствуют и в успешных трассах — системная слабость, не всегда критичная.

---

## 4. Методология

### Таксономия (MAST)
- Основана на Grounded Theory (Glaser & Strauss 1967): анализ 150+ трасс 6 экспертами без предварительных гипотез
- Итеративная Inter-Annotator Agreement (IAA): 3 раунда × 5 трасс, Cohen's κ = **0.88** (финальный)
- 14 режимов отказа → 3 категории → 3 стадии (Pre-Execution / Execution / Post-Execution)
- Проверена на unseen MAS (OpenManus, Magentic-One) и новых задачах (GAIA, MMLU): κ = 0.79

### Датасет (MAST-Data)
- **1642 аннотированных трасс** из 7 MAS-фреймворков
- Каждая трасса в среднем >15 000 строк текста
- Задачи: coding (ProgramDev, SWE-Bench), math (GSM-Plus, OlympiadBench, MMLU), general agent (GAIA, AppWorld Test-C)

### Модели
| LLM | Тип |
|---|---|
| GPT-4o, GPT-4 | Closed-source |
| Claude-3.7-Sonnet | Closed-source |
| **Qwen2.5-Coder-32B-Instruct** | Open-source ✅ |
| CodeLlama-7b-Instruct-hf | Open-source |

### LLM-as-a-Judge pipeline
- Модель-аннотатор: OpenAI o1 (few-shot с MAST-определениями)
- Согласие с людьми: accuracy 94%, Cohen's κ = **0.77**
- Стоимость: $0.37–$4.14 на трассу (в зависимости от MAS)

---

## 5. Результаты и выводы

### Распределение отказов (1642 трассы)
| Категория | Доля | Топ-режим |
|---|---|---|
| FC1 System Design Issues | 44.2% | FM-1.3 Step Repetition (15.7%) |
| FC2 Inter-Agent Misalignment | 32.3% | FM-2.6 Reasoning-Action Mismatch (13.2%) |
| FC3 Task Verification | 23.5% | FM-3.3 Incorrect Verification (9.1%) |

### GPT-4o vs Claude-3.7-Sonnet (MetaGPT на ProgramDev-v2)
- GPT-4o: на **39% меньше** FC1-отказов, значительно меньше FC2
- Оба показывают высокое число FC3 — верификация остаётся проблемой независимо от модели

### MetaGPT vs ChatDev (оба GPT-4o, ProgramDev-v2)
- MetaGPT: 60–68% меньше FC1 и FC2-отказов
- ChatDev: в **1.56×** меньше FC3-отказов (встроенные фазы review/testing)

### Open-source модели
- Qwen2.5-Coder-32B значительно лучше CodeLlama-7b (меньше отказов во всех категориях)
- Обе уступают закрытым моделям — performance gap существенный для multi-agent задач

### Интервенционные эксперименты (Case Studies)
| Вмешательство | Результат |
|---|---|
| Улучшение ролевых промптов (ChatDev) | +9.4% task success |
| Добавление high-level верификационного шага (ChatDev) | +15.6% на ProgramDev |
| Реструктуризация топологии AG2 | статистически значимо (p=0.03 с GPT-4o) |
- **Топологические изменения эффективнее промпт-патчей** для обоих систем (Figures 10–11)
- Тактические фиксы не решают проблему комплексно — нужны structural redesigns

---

## 6. Прямая связь с тезисом

### Позиционирование в аргументации thesis

**MAST поддерживает H0 (null hypothesis: MAS не стабильно превосходит single agent):**
- Failure rates 41–86.7% в SOTA MAS фреймворках — сильный аргумент, что overhead агентов не всегда окупается
- Простые тактические фиксы дают лишь +9.4–15.6% — скромный gain за сложность системы
- Single-agent baselines (упомянутые в Related Work как Kapoor et al. 2024, Xia et al. 2024) остаются конкурентоспособными

**MAST поддерживает H1 (MAS превосходит single agent при достаточной сложности):**
- "A well-designed MAS can result in performance gain when using the same underlying model" (Insight 1) — но только если дизайн действительно хороший
- Failures stem from design, not LLM limits → правильно спроектированный MAS имеет потенциал

**Методологические заимствования:**
1. **Таксономия MAST как аналитический lens**: использовать 14 failure modes для качественного анализа трасс моих экспериментов (особенно FM-1.3 Step Repetition, FM-1.5 Unaware of Termination, FM-3.2/3.3 Verification failures — все релевантны для scheduling)
2. **Идея failure profile по условиям сложности**: аналогично таблице 8 (failure rates по сложности бенчмарка — GSM vs MMLU vs Olympiad) — я могу строить failure breakdowns по уровням constraint complexity
3. **LLM-as-judge подход**: обоснование для использования автоматизированной оценки meeting satisfaction rate
4. **Intervention framework**: логика "baseline → prompt improvement → topology change" применима к моим 4 условиям (ReAct / ReAct+verify / hierarchical MAS / planner+critic)

### Релевантные failure modes для planning/scheduling

| MAST mode | Relevance для meeting planning |
|---|---|
| FM-1.3 Step Repetition | Агент циклически перебирает слоты, не достигая решения |
| FM-1.5 Unaware of Termination | Агент не понимает, когда расписание оптимально/допустимо |
| FM-2.2 Fail to Ask Clarification | Неясные/конфликтующие ограничения не уточняются |
| FM-2.6 Reasoning-Action Mismatch | Рассуждает правильно, но выводит неверный слот |
| FM-3.2 No/Incomplete Verification | Не проверяет все constraint violations в найденном расписании |
| FM-3.3 Incorrect Verification | Считает расписание валидным, хотя оно нарушает ограничения |

---

## 7. Где цитируется в thesis

| Раздел | Функция |
|---|---|
| **Introduction** | Motivation: failure rates 41–86.7% в SOTA MAS → нет чёткого понимания, когда MAS worth it |
| **Related Work § MAS Failures** | Основная цитата: MAST как первая эмпирическая таксономия MAS-отказов; три категории |
| **Related Work § Single vs Multi Agent** | Цитировать совместно с Kapoor 2024 и Tran 2026 — все три показывают, что MAS не гарантирует выигрыш |
| **Methodology § Evaluation** | Обоснование для LLM-as-judge оценки трасс (аналогично LLM annotator в MAST) |
| **Discussion § Failure Analysis** | Интерпретация failure patterns через MAST lens; соотнести собственные наблюдения с FC1/FC2/FC3 |
| **Discussion § Complexity Threshold** | Insight из Table 8: failure rates растут с ростом сложности задачи — параллель к complexity threshold |

---

## 8. Важные различия (для Related Work)

> **Подчеркнуть в тексте Related Work**, чтобы отделить свой contribution:

1. **Задача vs домен**: MAST охватывает coding, math, general agent tasks — без фокуса на scheduling/planning. Мой thesis нацелен исключительно на **meeting scheduling как proxy для constraint-satisfaction задач** с параметризованной сложностью.

2. **Absence of budget control**: MAST не контролирует token budget. В MAST-Data разные MAS могут использовать произвольное количество вызовов и токенов. Мой thesis обеспечивает **equal token budget** — необходимое условие для fair comparison.

3. **Observational vs experimental**: MAST — наблюдательное исследование существующих MAS-трасс (grounded theory). Мой thesis — **controlled experiment** с параметризованным генератором задач и измерением complexity threshold.

4. **Complexity metric**: MAST не вводит количественную меру сложности задачи. Мой thesis определяет сложность как **число взаимодействующих ограничений** — novel contribution, позволяющий измерить threshold эффект.

5. **Модели**: MAST использует GPT-4o, Claude-3.7-Sonnet, Qwen2.5-Coder-32B, CodeLlama-7b. Мой thesis использует **Qwen3 32B (self-hosted, vLLM)** — ближайший открытый родственник Qwen2.5-Coder, что делает сравнение особенно уместным.

6. **Вопрос исследования**: MAST отвечает "почему MAS failing?"; мой thesis отвечает "**при каком уровне сложности MAS начинает превосходить single agent?**" — complementary, не конкурирующие вопросы.

---

## 9. Точные цитаты (≤14 слов, для thesis)

> ⚠️ Все цитаты взяты дословно из текста статьи. Проверены по PDF.

**Q1** — обоснование проблемы (Introduction):
> "MAS failure is not merely a function of challenges in the underlying model"
— (Cemri et al., 2025, §4 FC1, Insight 1) — *14 слов* ✓

**Q2** — масштаб проблемы (Introduction / Motivation):
> "performance gains on popular benchmarks are often minimal"
— (Cemri et al., 2025, Abstract) — *8 слов* ✓

**Q3** — дизайн важнее LLM (Discussion):
> "good MAS design requires organizational understanding"
— (Cemri et al., 2025, §5.3) — *6 слов* ✓

---

## 10. BibTeX запись

```bibtex
@inproceedings{cemri2025mast,
  title     = {Why Do Multi-Agent {LLM} Systems Fail?},
  author    = {Cemri, Mert and Pan, Melissa Z. and Yang, Shuyi and
               Agrawal, Lakshya A and Chopra, Bhavya and Tiwari, Rishabh and
               Keutzer, Kurt and Parameswaran, Aditya and Klein, Dan and
               Ramchandran, Kannan and Zaharia, Matei and
               Gonzalez, Joseph E. and Stoica, Ion},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2025},
  note      = {arXiv:2503.13657}
}
```

---

## Заметки для дальнейшей работы

- [ ] Добавить в `references.bib` под ключом `cemri2025mast`
- [ ] В Related Work написать параграф, соединяющий Cemri 2025 + Tran 2026 + Kapoor 2024 как тройное обоснование H0
- [ ] В Discussion Section использовать MAST failure modes как аналитическую рамку для интерпретации трасс (особенно FM-1.3, FM-1.5, FM-3.2)
- [ ] Проверить: **используется ли `Qwen2.5-Coder-32B`** — *да*, это семейство Qwen, близкое к Qwen3 32B. Укажи в тексте: "близкая модельная семья к используемой в данном thesis"
- [ ] Посмотреть Table 8 (failure rates по сложности задачи GSM→MMLU→Olympiad) — прямая параллель к моей complexity gradient гипотезе

---

*Обработано: 15 июня 2026 | Статус: ✅ Готово к интеграции в thesis | Файл: `cemri2025_mast.md`*
