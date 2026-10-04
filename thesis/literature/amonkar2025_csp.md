# amonkar2025_csp

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
| **Авторы**     | Rikhil Amonkar*, Ceyhun Efe Kayan*, Qimei Lai, Ronan Le Bras, Li Zhang |
| **Аффилиации** | Drexel University, University of Pennsylvania, Allen Institute for AI |
| **Год**        | 2025 (первая версия: май 2025; v4: 31 марта 2026) |
| **Venue**      | Preprint. Under review (arXiv) |
| **arXiv ID**   | 2505.13252v4 |
| **BibTeX key** | `amonkar2025csp` |
| **Qwen?**      | ✅✅ **ДА** — Qwen3-32B и Qwen2.5-32B тестируются как основные модели, в том числе именно на Meeting Planning из NATURAL PLAN |

---

## 2. Одна строка

Систематическое сравнение LLM-as-solver vs LLM-as-formalizer (Python/Z3 SMT) на реальных CSP-задачах (включая Meeting Planning) с 6 LLM, в том числе Qwen3-32B, показывает: формализация проигрывает прямому решению в 15/24 случаях, а оба подхода одинаково деградируют с ростом числа ограничений.

---

## 3. Ключевые аргументы

1. **LLM-as-formalizer НЕ превосходит LLM-as-solver**: вопреки установившемуся мнению, формализация проигрывает прямому решению в 15 из 24 комбинаций модель-датасет (и в 12 из 16 для LRM в частности).

2. **Оба подхода одинаково деградируют с ростом сложности**: несмотря на то что пространство формализации на порядки меньше пространства поиска, производительность формализатора падает так же резко, как у solver'а, при росте числа constraints — обещанная робастность к сложности не реализуется.

3. **Главная ошибка — семантическая, а не синтаксическая**: 95% ошибок Qwen3 — это неправильно определённые constraints (wrong constraint), а не синтаксические ошибки. Revision устраняет syntax errors и no-plan, но не wrong-plan.

4. **LRM «решают задачу мысленно» вместо формализации**: до 50% reasoning tokens DeepSeek-R1 и >25% Qwen3 — это «solver-like» рассуждения (перебор, backtracking), а не code-focused. Модели обучены решать, а не формализовать, что проявляется в spurious reasoning и hard-coded solutions.

5. **Python > Z3 SMT**: свободный Python превосходит код для SMT-solver'а в 15 из 24 комбинаций, несмотря на то что Z3 теоретически более подходит для CSP.

---

## 4. Методология

### Задачи и данные
| Датасет | Домен |
|---------|-------|
| NaturalPlan – Calendar Scheduling | CSP: start/end time ∈ [9,17] |
| NaturalPlan – Trip Planning | CSP: day/city assignment |
| **NaturalPlan – Meeting Planning** | **CSP: person, start, duration** |
| ZebraLogic | Logic grid puzzle |

По 100 случайно выбранных примеров из 1000 (NATURAL PLAN). Constraints **вручную аннотированы** авторами — оценка корректна при прохождении всех аннотированных constraints (не string matching).

### Модели
- **LRM**: DeepSeek-R1, **Qwen3-32B** (✅), o3-mini-high, GPT-5
- **Non-LRM**: DeepSeek-V3, Qwen2.5-32B

### Методы (pipelines)
- **LLM-as-solver**: прямой ответ (1-shot, без explicit CoT prompt)
- **LLM-as-formalizer Python**: LLM генерирует Python-программу, запускается
- **LLM-as-formalizer SMT (Z3)**: LLM генерирует код для Z3 solver
- С ревизией и без: до 5 попыток при ошибке/отсутствии плана

### Метрика
Процент планов, формально удовлетворяющих **всем аннотированным constraints** (binary). Сложность измеряется числом constraints → стратификация на 5 квантилей.

### Анализ ошибок
- 80 примеров вручную аннотированы: error / no plan / wrong plan / correct
- 160 reasoning chains вручную размечены: code-related / solver-like / spurious

---

## 5. Результаты и выводы

### RQ1: Formalizer vs Solver

| Условие | Formalizer > Solver |
|---------|---------------------|
| LRM (16 комбинаций) | 4 из 16 (25%) |
| Non-LRM (8 комбинаций) | 4 из 8 (50%) |
| Итого | 9 из 24 (37.5%) |

Формализатор НЕ превосходит solver в большинстве случаев.

### RQ2: Робастность к сложности
Performance на top-20% сложных задач менее чем вдвое ниже bottom-20% у **10 из 12** комбинаций для Python+revision. Оба подхода деградируют аналогично.

### RQ3: Причины неудачи формализации
- Syntax errors: редки, revision их устраняет
- No plan: устраняется revision
- **Wrong plan: персистентен, не устраняется revision**
- Основная причина wrong plan: **wrong constraint** (95% у Qwen3)
- Примеры: время 2:16 интерпретируется как end time вместо start time

### RQ4: Эффективность токенов
- Формализатор генерирует **меньше** reasoning tokens, чем solver (для большинства доменов)
- Но до 50% reasoning chains у DeepSeek-R1 — solver-like (не code-related)
- Spurious reasoning (hard-coded solutions без constraints) — до 90% у R1
- Python: меньше токенов, чем SMT

---

## 6. Прямая связь с моим thesis

### Роль в thesis
Эта статья — **ключевое empirical обоснование** для выбора LLM-as-solver (ReAct) как главного условия вместо LLM-as-formalizer. Она также напрямую оценивает **Qwen3-32B** на **Meeting Planning** из NATURAL PLAN — прямой прообраз моей задачи.

### Поддержка гипотез

| Гипотеза | Связь |
|----------|-------|
| **H1** (MAS > single agent при высокой сложности) | Косвенная: если single-agent solver уже ограничен при high complexity, MAS потенциально помогает; но статья не тестирует MAS |
| **H2** (порог сложности существует) | Прямая: Fig.3 показывает нелинейную деградацию обоих подходов с ростом числа constraints → эмпирическое свидетельство complexity degradation в Meeting Planning |
| **H0** (нет разницы) | Не затрагивается напрямую; но LRM-as-solver как сильный baseline обоснован этой статьёй |

### Практическое значение для дизайна thesis
1. **Обоснование LLM-as-solver (ReAct) как baseline**: статья показывает, что для Meeting Planning LLM-as-solver (прямое решение, ≈ мой ReAct) сравним или превосходит formalizer → правильный выбор baseline.
2. **OR-Tools CP-SAT как oracle**: мой выбор использовать solver как ground truth (не как метод агента) разумен — статья показывает, что LLM плохо генерирует code для solver'а.
3. **Complexity threshold**: рис. 3 для Meeting Planning с Qwen3 — прямые данные о сложности, которые можно цитировать рядом с моими результатами.
4. **Qwen3-32B baseline numbers**: результаты из рис. 2 дают точки сравнения для моего ReAct-агента на Qwen3-32B.

### Методологические заимствования
- **Formal constraint annotation** (оценка по всем аннотированным constraints): я использую аналогичный подход через OR-Tools oracle вместо string matching.
- **Stratification по числу constraints**: мой complexity axis аналогичен их percentile bucketing.

---

## 7. Где цитируется в thesis

| Раздел | Роль |
|--------|------|
| **Introduction** | Мотивация выбора LLM-as-solver (ReAct) как архитектурного baseline вместо neurosymbolic подхода |
| **Related Work** | §LLM Planning Approaches: сравнение solver vs formalizer для CSP-задач, в том числе Meeting Planning; позиционирование моей работы |
| **Methodology** | §Baseline Justification: обоснование того, что ReAct ≈ LLM-as-solver является правомерным сильным baseline |
| **Discussion** | §Complexity Effects: сравнение паттерна деградации с их рис. 3; обсуждение схожести decay с ростом constraints |

---

## 8. Важные различия

> Необходимо явно указать в Related Work, чтобы показать различие в постановке.

| Аспект | Amonkar et al. (2025) | Мой thesis |
|--------|----------------------|------------|
| **Сравниваемые системы** | LLM-as-solver vs LLM-as-formalizer (neurosymbolic) | Single-agent ReAct vs hierarchical MAS |
| **Внешний solver** | Z3 SMT / Python (как метод генерации плана) | OR-Tools CP-SAT (как oracle для оценки) |
| **Агентная архитектура** | 1-shot prompting, no agentic loop | ReAct loop, verify-revise, hierarchical MAS |
| **Multi-agent** | Отсутствует | Центральная переменная |
| **Метрика** | Binary exact pass (все constraints) | Partial credit: meeting satisfaction rate |
| **Token budget** | Не контролируется | Равный token budget |
| **Revision** | До 5 attempts на syntax/no-plan errors | verify-revise как отдельное архитектурное условие |
| **Масштаб оценки** | 100 примеров / домен | Полный экспериментальный сетап с синтетическим генератором |
| **Тип сложности** | Число constraints (квантильная стратификация) | Число **взаимодействующих** constraints (структурная мера) |

---

## 9. Точные цитаты

Все цитаты строго < 15 слов, взяты дословно из текста статьи.

> **[Q1]** "LLM-as-formalizer underperforms LLM-as-solver in 15 out of 24 model-dataset combinations"
> *(p. 1, Abstract)* — для Related Work: обоснование выбора LLM-as-solver как сильного baseline.

> **[Q2]** "LLM-as-formalizer still drastically degrades as problem complexity increases similar to LLM-as-solver"
> *(p. 1, Abstract)* — для Discussion: общий паттерн деградации с ростом сложности.

> **[Q3]** "the majority of the semantic errors are due to wrongly defining constraints"
> *(p. 7, §5.3)* — для Discussion: почему constraint complexity challenges persist независимо от архитектуры.

---

## 10. BibTeX запись

```bibtex
@article{amonkar2025csp,
  title     = {A Reality Check of Language Models as Formalizers on
               Constraint Satisfaction Problems},
  author    = {Amonkar, Rikhil and Kayan, Ceyhun Efe and Lai, Qimei and
               {Le Bras}, Ronan and Zhang, Li},
  year      = {2025},
  journal   = {arXiv preprint arXiv:2505.13252},
  url       = {https://arxiv.org/abs/2505.13252},
  note      = {Preprint. Under review. v4: 31 March 2026}
}
```
