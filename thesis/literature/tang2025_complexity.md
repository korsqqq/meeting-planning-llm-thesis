# On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

## Метаданные
- **Авторы:** Bohan Tang, Huidong Liang, Keyue Jiang (равный вклад), Xiaowen Dong
- **Аффилиации:** University of Oxford, University College London
- **Год:** 2025
- **Venue:** NeurIPS 2025 Workshop (Scaling Environments for Agents, SEA)
- **arXiv:** 2510.04311v1 (5 Oct 2025)
- **BibTeX key:** tang2025complexity

## Одна строка (суть)
Теоретически и эмпирически показывает, что преимущество MAS над single-agent растёт с
ростом сложности задачи — и сильнее зависит от «глубины» (длины рассуждений), чем от «ширины».

## Ключевые аргументы
- Текущие выводы «MAS лучше SAS» опираются только на downstream-метрики, без принципиального
  понимания *когда и почему* MAS эффективнее. Это и есть пробел, который статья закрывает.
- Вводит **двумерную меру сложности**: **depth** (длина цепочки рассуждений, число
  последовательных шагов) и **width** (широта способностей/знаний, нужных на каждом шаге).
- Фокус на одном классе MAS — **multi-agent debate** (несколько агентов предлагают, критикуют,
  уточняют ответы; финальный агрегатор сводит результат).
- Главный вывод: выигрыш MAS растёт по обоим измерениям, но **depth доминирует** — выигрыш
  по width насыщается (saturates), а по depth растёт неограниченно.

## Методология
- **Теория:** формализует success rate SAS как `s(w)^d` и MAS как `r·[1−(1−s(w))^N]^d`,
  определяет performance gain Δ как относительное улучшение. Доказывает две propositions:
  (2.1) ∂Δ/∂d > 0 и ∂Δ/∂w > 0; (2.2) по width предел конечный `(rN)^d − 1`, по depth → +∞.
- **Эмпирика, 2 задачи:**
  - Math reasoning — бенчмарк DyVal (дерево-DAG), depth/width заданы структурой DAG, значения 2–4,
    900 вопросов.
  - Creative writing — собственный бенчмарк **DW² (Depth-Width Writing)**: depth = число
    предложений K, width = нормализованная энтропия Шеннона доменов ключевых слов, 2500 вопросов.
- **Модель:** Qwen-2.5-32B-Instruct во всех экспериментах. MAS = multi-agent debate (4–6 агентов
  включая агрегатор), SAS = chain-of-thought.
- **Анализ вклада измерений:** Shapley-R² декомпозиция (какое измерение сильнее объясняет gain).

## Результаты и выводы
- По обеим задачам два устойчивых паттерна: (1) выигрыш MAS растёт со сложностью; (2) рост по
  depth значимее, чем по width — подтверждает теорию.
- В creative writing магнитуда выигрыша заметно больше, чем в math reasoning. Гипотеза: это
  генеративная задача с огромным пространством решений и **сетью взаимодействующих ограничений** —
  SAS чаще проваливает constraints (например, покрытие всех keyword'ов), MAS распределяет нагрузку.
- Качество ответов SAS и MAS близко (~5%), но SAS чаще нарушает ограничения.

## Прямая связь с моим thesis
- **Ключевая опора мотивации (H2 — «crossover может существовать»).** Это одна из работ, на
  которую exposé прямо ссылается: «multi-agent advantage grows with task complexity».
- **Контраст с моим дизайном (важно подчеркнуть в Related Work):**
  - Tang измеряет сложность как depth/width; я — как **число взаимодействующих ограничений**
    (вслед за Amonkar 2025). Это другая ось сложности.
  - Tang НЕ выравнивает token budget; у меня equal budget — главный контроль.
  - Tang использует **debate-style** MAS на **non-planning** задачах (math, writing);
    у меня — **hierarchical** MAS на **tool-based meeting planning**. Exposé прямо отмечает этот
    разрыв как то, что я закрываю.
- **Полезная связка:** их финальное замечание из «Beyond depth and width» про *interaction
  complexity* (насколько подзадачи взаимозависимы/конфликтуют) почти буквально совпадает с моей
  осью сложности — можно процитировать как мотивацию выбора constraint-based меры.
- **Совпадение по модели:** они тоже используют Qwen-2.5-32B — близко к моему Qwen3 32B.

## Где цитируется в thesis
- **Related Work** — раздел про связь сложности задачи и эффективности MAS.
- **Introduction / Motivation** — обоснование гипотезы о crossover.
- **Discussion** — контраст осей сложности (depth/width vs interacting constraints) и
  бюджетного контроля.

## Точные цитаты (под 15 слов каждая)
- "the benefit of LLM-MAS over LLM-SAS increases with both task depth and width"
- "depth can provide unbounded improvements"

## Ограничения работы (полезно для критики в Related Work)
- Только debate-style MAS (не hierarchical decomposition, как у меня).
- Только non-planning задачи (math, creative writing).
- Не контролирует token budget и coordination overhead — признаётся как открытый вызов.
- Модель не больше Qwen-2.5-32B (вычислительные ограничения).

## BibTeX
```bibtex
@inproceedings{tang2025complexity,
  title={On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems},
  author={Tang, Bohan and Liang, Huidong and Jiang, Keyue and Dong, Xiaowen},
  booktitle={NeurIPS 2025 Workshop: Scaling Environments for Agents (SEA)},
  year={2025},
  note={arXiv:2510.04311}
}
```
