# Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline (OneFlow)

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

## Метаданные
- **Авторы:** Jiawei Xu, Arief Koesdwiady, Sisong Bei, Yan Han, Baixiang Huang, Dakuo Wang,
  Yutong Chen, Zheshen Wang, Peihao Wang, Pan Li, Ying Ding
- **Аффилиации:** UT Austin, Amazon, Emory, Northeastern, Georgia Tech
- **Год:** 2026
- **arXiv:** 2601.12307v1 (18 Jan 2026)
- **BibTeX key:** xu2026oneflow

## Одна строка (суть)
Большинство MAS **гомогенны** (все агенты = одна базовая LLM, различаются только промптами/ролями),
а значит один агент может симулировать такой workflow через multi-turn диалог — с равным качеством
и дешевле за счёт переиспользования KV-cache.

## Ключевые аргументы
- **Гомогенность как слепое пятно:** в большинстве MAS все агенты делят одну базовую модель и
  отличаются только system-промптами, инструментами и позицией в графе. Тогда зачем отдельные
  инстансы?
- **Теорема симуляции (Proposition 1):** при детерминированных tool-side-effects, маршрутизации,
  зависящей только от видимой истории, и фиксированном декодировании — single-LLM-симулятор даёт
  **ту же распределённость транскриптов**, что и MAS с раздельными инстансами.
- **KV-cache даёт экономию:** один агент переиспользует KV-cache между «ролями», поэтому prefill
  растёт как Δ-токены, а не как полный префикс → стоимость single ≤ стоимость multi.
- **Граница метода:** single-LLM **не может** симулировать *гетерогенные* workflow (разные базовые
  модели не делят KV-cache). Это и есть оставленное направление для будущей работы.

## Методология
- **OneFlow** — алгоритм авто-дизайна workflow через MCTS + два мета-LLM (Creative Designer +
  Critical Reviewer), оптимизирует Pareto «performance − cost», заточен под single-agent исполнение.
- **7 бенчмарков:** код (HumanEval, MBPP), математика (GSM8K, MATH), QA (HotpotQA, DROP),
  domain (Shopping-MMLU), **planning/tool-use (TravelPlanner)**.
- **Модели:** GPT-4o-mini (основной), Claude 3.5 Haiku, **Qwen-3 8B** (open-weight, через vLLM,
  16k контекст). Designer = Claude-4-Sonnet.
- **Метрики:** task accuracy/pass@1/F1/success rate + стоимость USD + latency/throughput (Qwen).

## Результаты и выводы
- Single-agent исполнение гомогенного workflow **равно или чуть лучше** multi-agent версии на всех
  бенчмарках — подтверждает Proposition 1.
- **Существенно дешевле:** single-LLM execution резко снижает стоимость при том же качестве за счёт
  KV-cache reuse. Для OneFlow качество даже слегка растёт (больше контекста у агента).
- **Qwen-3 8B + vLLM (Table 4):** single-agent держит качество и эффективность; KV-cache reuse
  сохраняет latency/throughput, хотя multi-turn накапливает больше входных токенов.
- **TravelPlanner (planning!):** single-agent исполнение AFlow/OneFlow workflow **достигает того же
  success rate**, что multi-agent, при меньшей стоимости — Pareto-фронт сдвинут в их пользу.
- Гетерогенный pilot: производительность в основном **ограничена лучшим гомогенным workflow**.

## Прямая связь с моим thesis
- **Третий столп «скептической» линии** (вместе с Tran & Kiela 2026 и Cemri 2025), на которую
  опирается мой Problem Statement: «more agents ≠ automatically better».
- **Прямое доказательство для planning:** их TravelPlanner-эксперимент показывает, что на
  **tool-based планировании** single-agent держится наравне с MAS — это поддерживает мою H1
  (на низкой сложности single ≥ MAS) и мотивирует поиск именно *порога* сложности.
- **Важное различие в постановке (подчеркнуть!):**
  - Они спрашивают «может ли ОДИН агент **симулировать** MAS-workflow?» — про эквивалентность
    исполнения. Я спрашиваю «существует ли **порог сложности**, где MAS реально *превосходит*
    сильный single ReAct под равным бюджетом?» — это разные вопросы.
  - Их «single-agent» = одна модель, ролеплеящая весь граф через multi-turn (то есть фактически
    исполняющая декомпозицию). Мой single ReAct (Условие 1) декомпозицию **не** получает. Это
    помогает мне чётко позиционировать мои Условия 2/4 как промежуточные точки.
- **KV-cache аргумент прямо релевантен моему equal-budget контролю:** они показывают, что
  «справедливое» сравнение должно учитывать KV-cache reuse. У меня MAS передаёт сообщения между
  агентами (входят в бюджет) — стоит отметить, что часть «overhead» MAS — это re-encoding,
  которого single-agent избегает. Полезно для интерпретации cost-результатов.
- **Методологический референс:** Qwen-3 8B + vLLM + 16k контекст + измерение latency/throughput —
  почти мой сетап (я на Qwen3 8B/32B, vLLM). Их таблицы input/output-токенов — образец отчётности.

## Где цитируется в thesis
- **Problem Statement / Introduction** — линия «MAS не автоматически лучше».
- **Related Work** — гомогенные MAS, KV-cache, single-agent симуляция; контраст «симуляция vs
  превосходство под равным бюджетом».
- **Discussion** — интерпретация coordination overhead через KV-cache re-encoding; почему мой
  hierarchical MAS платит за message-passing.

## Важные различия (подчеркнуть в Related Work)
- Они: «может ли single **симулировать** homogeneous workflow». Я: «есть ли **порог**, где MAS
  *побеждает* сильный single ReAct под равным token budget».
- Их single-agent **исполняет декомпозицию** (ролеплей графа). Мой ReAct (Усл. 1) — нет.
- Они: общие бенчмарки (код/QA/math) + TravelPlanner с exact success rate. Я: синтетический
  meeting-planning с **solver-based partial-credit** и контролем числа взаимодействующих ограничений.
- Они НЕ выравнивают token budget явно (фокус на cost/KV-cache); у меня equal budget — главный контроль.

## Точные цитаты (под 15 слов каждая)
- "a single agent can reach the performance of homogeneous workflows"
- "single-LLM methods cannot capture heterogeneous workflows due to the lack of KV cache sharing"

## BibTeX
```bibtex
@article{xu2026oneflow,
  title={Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline},
  author={Xu, Jiawei and Koesdwiady, Arief and Bei, Sisong and Han, Yan and Huang, Baixiang and Wang, Dakuo and Chen, Yutong and Wang, Zheshen and Wang, Peihao and Li, Pan and Ding, Ying},
  journal={arXiv preprint arXiv:2601.12307},
  year={2026}
}
```
