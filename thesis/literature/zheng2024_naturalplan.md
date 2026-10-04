# zheng2024_naturalplan

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

| Поле          | Значение |
|---------------|----------|
| **Авторы**    | Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou |
| **Аффилиации**| Google DeepMind |
| **Год**       | 2024 |
| **Venue**     | Preprint (arXiv) — не опубликовано в конференции на момент v1 |
| **arXiv ID**  | 2406.04520v1 (submitted 6 June 2024) |
| **BibTeX key**| `zheng2024naturalplan` |
| **Qwen?**     | ❌ Нет — используются GPT-3.5, GPT-4, GPT-4o, Gemini 1.5 Flash/Pro |

---

## 2. Одна строка

NATURAL PLAN — бенчмарк трёх реальных задач планирования (Trip, Meeting, Calendar), на котором все SoTA LLM показывают менее 50% exact match, а производительность катастрофически падает с ростом числа ограничений.

---

## 3. Ключевые аргументы

1. **Planning in NL is fundamentally hard for LLMs.** Даже при наличии всей контекстной информации от инструментов (Flights, Maps, Calendar) лучшая модель (Gemini 1.5 Pro) не превышает 34.8% на Trip Planning и 48.9% на Calendar Scheduling.

2. **Complexity is the primary failure mode.** Производительность экспоненциально падает с числом constraint-участников: в Meeting Planning все модели опускаются ниже 10% при ≥8 людях; в Trip Planning — ниже 5% при 10 городах.

3. **Self-correction hurts, not helps.** Prompt-based самокоррекция вызывает значительное падение точности у всех моделей, причём сильнее у более мощных (GPT-4, Gemini 1.5 Pro), которые слишком самоуверенно «исправляют» корректные решения.

4. **Long-context in-context learning is promising but model-dependent.** Только Gemini 1.5 Pro способен устойчиво использовать до 800 примеров (355K токенов) и улучшать точность (2.7% → 39.9% в Trip Planning). GPT-4 и Gemini 1.5 Flash деградируют после 20 примеров.

5. **Easy-to-hard generalization dominates hard-to-easy.** Демонстрации с более простыми примерами в few-shot дают лучший результат, что свидетельствует о том, что модели плохо извлекают стратегии из сложных exemplars.

---

## 4. Методология

### Задачи
| Задача | Примеров | Переменная сложности |
|--------|----------|---------------------|
| Trip Planning | 1 600 | N городов ∈ [3, 10] |
| Meeting Planning | 1 000 | N людей ∈ [1, 10] |
| Calendar Scheduling | 1 000 | N участников ∈ [2, 7] или N дней ∈ [1, 5] |

Данные синтетически сгенерированы на основе реальных API (Google Flights, Google Maps, Google Calendar). Tool outputs предоставлены **в контексте** — tool-use execution из оценки исключён.

### Модели
- GPT-3.5 (`gpt-3.5-turbo-0125`)
- GPT-4 (`gpt-4-turbo-2024-04-09`)
- GPT-4o (`gpt-4o-2024-05-13`)
- Gemini 1.5 Flash
- Gemini 1.5 Pro

### Метрика
**Exact Match (EM)** — бинарная: план правильный тогда и только тогда, когда все поля (дата/место/время) точно совпадают с ground truth. Partial credit отсутствует.

### Сетап
- Базовый: 5-shot prompting.
- Ablations: self-correction (prompt-based), easy-to-hard / hard-to-easy generalization, in-context planning с до 800 shots (long context).

---

## 5. Результаты и выводы

### Основные числа (5-shot)

| Задача | GPT-3.5 | GPT-4 | GPT-4o | Gem 1.5 Flash | Gem 1.5 Pro |
|--------|---------|-------|--------|---------------|-------------|
| Trip Planning | 7.3% | 31.1% | 3.7% | 25.6% | **34.8%** |
| Meeting Planning | 19.1% | **47.0%** | 45.2% | 23.9% | 39.1% |
| Calendar Scheduling | 19.9% | 41.2% | 43.7% | 34.3% | **48.9%** |

### Ключевые выводы
- **Complexity cliff**: При 10 городах (Trip) все модели < 5%; при ≥8 людях (Meeting) все модели < 10%.
- **Self-correction backfires**: Значительное падение точности у всех моделей после self-correction; более сильные модели теряют больше.
- **GPT-4o аномалия**: GPT-4o (3.7% на Trip) резко уступает GPT-4 (31.1%); анализ 10 ошибок показал, что 7/10 — нарушение flight connectivity constraints, 3/10 — travel date constraints.
- **Long context**: Только Gemini 1.5 Pro непрерывно улучшается до 800-shot (355K токенов). GPT-4 и Flash деградируют после 20-shot.

---

## 6. Прямая связь с моим thesis

### Роль в thesis
Это **центральный benchmarking baseline** для задачи Meeting Planning — прямой прототип моего экспериментального сетапа.

### Поддержка гипотез
| Гипотеза | Поддержка |
|----------|-----------|
| **H1** (MAS > Single Agent при высокой сложности) | Косвенная: одиночные агенты (даже GPT-4) деградируют при N ≥ 5–6 людях, что создаёт «потолок» производительности, который MAS потенциально может преодолеть |
| **H2** (порог сложности существует) | Прямая: Fig. 5 показывает излом кривой производительности в Meeting Planning при N = 3–4 → мотивирует гипотезу о complexity threshold |
| **H0** (нет разницы) | Не затрагивается — статья не сравнивает агентные архитектуры |

### Методологические заимствования
1. **Структура задачи Meeting Planning** (N людей, travel time constraints, maximize meetings) — прямой предшественник моего synthetic task generator.
2. **Systematic complexity scaling** (варьирование N) — я адаптирую под «число взаимодействующих ограничений» вместо просто числа людей.
3. **Ground truth через solver** — Zheng et al. конструируют задачи с единственным решением; я аналогично использую OR-Tools CP-SAT как oracle.
4. **Decoupling tool-use from reasoning** — авторы убирают tool execution из оценки; я аналогично исключаю latency tool calls из token budget подсчёта.

---

## 7. Где цитируется в thesis

| Раздел | Роль |
|--------|------|
| **Introduction** | Мотивация: "Even SoTA models achieve <50% on natural language planning benchmarks [Zheng et al., 2024]" |
| **Related Work** | Центральная ссылка в §Meeting Planning Benchmarks: описание задачи и результатов базовых моделей |
| **Methodology** | §Task Design: "Our Meeting Planning task follows the formulation of [Zheng et al., 2024], adapted to parameterize constraint interaction count" |
| **Discussion** | §Complexity Threshold: сравнение моего порога с их complexity cliff при N ≥ 8 |

---

## 8. Важные различия

> Эти различия необходимо явно подчеркнуть в секции Related Work, чтобы обозначить оригинальный вклад thesis.

| Аспект | NATURAL PLAN (Zheng et al.) | Мой thesis |
|--------|----------------------------|------------|
| **Метрика** | Binary Exact Match (EM) | Partial credit: meeting satisfaction rate vs solver optimum |
| **Архитектура агента** | Чистый prompting (5-shot), без ReAct/agentic loop | ReAct / ReAct+verify-revise / hierarchical MAS / planner+critic |
| **Сравнение архитектур** | Только одиночные модели | Прямое сравнение single vs multi-agent |
| **Определение сложности** | Число людей / городов / дней | Число **взаимодействующих ограничений** (более точная мера CSP-сложности) |
| **Token budget** | Не контролируется | Равный token budget — центральная экспериментальная переменная |
| **Модели** | GPT-4/4o/3.5, Gemini 1.5 | Qwen3 32B (main), Qwen3 8B (robustness) — open-weight, self-hosted |
| **Самокоррекция** | Prompt-based SC (отдельный ablation) | verify-revise как одно из 4 условий эксперимента |
| **Ground truth** | Единственное решение по конструкции | OR-Tools CP-SAT solver как oracle с partial credit |

---

## 9. Точные цитаты

Все цитаты строго < 15 слов, взяты дословно из текста статьи.

> **[Q1]** "model performance drops drastically as the complexity of the problem increases"
> *(p. 1, Abstract)* — для мотивации complexity threshold hypothesis.

> **[Q2]** "Self-correction leads to significant model performance drop across all models"
> *(p. 8, §4.4)* — для критики verify-revise как простой self-correction; мой метод структурно отличается.

> **[Q3]** "all models perform below 5% when there are 10 cities"
> *(p. 7, §4.2)* — конкретный пример complexity cliff; аналог для Meeting Planning: <10% при ≥8 людях.

---

## 10. BibTeX запись

```bibtex
@article{zheng2024naturalplan,
  title     = {{NATURAL PLAN}: Benchmarking {LLM}s on Natural Language Planning},
  author    = {Zheng, Huaixiu Steven and Mishra, Swaroop and Zhang, Hugh and
               Chen, Xinyun and Chen, Minmin and Nova, Azade and Hou, Le and
               Cheng, Heng-Tze and Le, Quoc V. and Chi, Ed H. and Zhou, Denny},
  year      = {2024},
  journal   = {arXiv preprint arXiv:2406.04520},
  url       = {https://arxiv.org/abs/2406.04520},
  note      = {Preprint. Submitted 6 June 2024}
}
```
