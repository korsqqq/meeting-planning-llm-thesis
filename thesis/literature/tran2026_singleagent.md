# Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets

<!-- h-labels-banner -->
> ⚠️ **О метках H0/H1/H2 в этой заметке (добавлено 2026-07-26).** Канонические определения
> гипотез теперь живут ТОЛЬКО в `THESIS_DECISIONS.md` §5. Заметки писались раньше и
> используют как минимум две несовместимые схемы нумерации (например, в одних `H2` — это
> «порог существует», в других — «выигрыш оправдывает overhead»), и ни одна не обязана
> совпадать со схемой экспозе. **Авторитетна суть в скобках, а не номер.** Перед переносом
> в текст тезиса сверь метку с §5 и переформулируй по существу. Полное перенумерование
> заметок — отдельный проход (см. §8, «Two known gaps left open on purpose»).

## Метаданные
- **Авторы:** Dat Tran, Douwe Kiela
- **Аффилиация:** Stanford University
- **Год:** 2026
- **Статус:** Preprint, under review
- **arXiv:** 2604.02460v2 (11 Apr 2026)
- **BibTeX key:** tran2026singleagent

## Одна строка (суть)
Под равным бюджетом thinking-токенов single-agent стабильно ≥ multi-agent на multi-hop
reasoning; кажущееся превосходство MAS объясняется неучтённым compute, а не архитектурой.
MAS становится конкурентным только при деградации контекста единственного агента.

## Ключевые аргументы
- Сравнения MAS vs SAS обычно **запутаны (confounded)** разным test-time compute: MAS тратит
  больше токенов, и неясно, выигрыш от архитектуры или просто от лишних вычислений.
- **Теоретический аргумент через Data Processing Inequality (DPI):** сообщения M между агентами —
  функция от контекста C, поэтому I(Y;C) ≥ I(Y;M). Single-agent с полным доступом к C
  информационно-теоретически гарантированно работает **не хуже** MAS на M=g(C).
- **Предсказание, когда MAS помогает:** когда эффективное использование контекста единственным
  агентом **деградирует** (длинный/шумный/искажённый контекст) — тогда структурированный MAS
  через фильтрацию/декомпозицию/верификацию может восстановить релевантную информацию.

## Методология
- **Задачи:** FRAMES и MuSiQue (multi-hop QA; MuSiQue отфильтрован до 4-hop).
- **Контролируемая переменная:** именно **thinking tokens** (промежуточные рассуждения, без
  промптов и финального ответа), при matched budget.
- **Модели (3 семейства):** Qwen3-30B-A3B, DeepSeek-R1-Distill-Llama-70B, Gemini 2.5 (Flash/Pro).
- **SAS:** один проход «think step by step, then answer» + вариант SAS-L (структурный pre-answer
  scaffold, тот же бюджет).
- **5 MAS-архитектур:** Sequential (главный baseline), Subtask-parallel, Parallel-roles, Debate,
  Ensemble. Бюджет B делится между агентами; planner/aggregator держат near-budget-neutral.
- **Метрика:** LLM-as-judge по фиксированной рубрике (семантическое присутствие gold-ответа).
- **Бюджеты:** 100/500/1k/2k/5k/10k токенов. 95% bootstrap CI.

## Результаты и выводы
- **Главный результат:** SAS — лучшая default-архитектура; при matched budget SAS равен или лучше
  всех MAS-вариантов на всех бюджетах, кроме самого низкого (100 токенов, где рассуждения вообще
  нет). SAS при этом тратит меньше токенов.
- Паттерн **SAS ≥ Sequential устойчив** даже без token cap и через несколько поколений Gemini —
  это свойство самого сравнения, а не артефакт одного чекпойнта.
- **Crossover при деградации контекста (Section 5.3, ключевое для тебя):** при искажающих
  деградациях (substitution, masking) на Qwen3-30B при 1k бюджете — SAS лидирует при α=0.3,
  паритет при α=0.5, **Sequential MAS лучше при α=0.7**. То есть MAS помогает не когда контекст
  длиннее, а когда single-агенту трудно отличить релевантное от вводящего в заблуждение.
- **Диагностика evaluation (важно методологически):** API-budget control ненадёжен (особенно
  Gemini 2.5 — до 4.7x инфляция API-счётчика токенов vs видимый текст); бенчмарки уязвимы к
  перефразированию (намёк на memorization). Поэтому контролировать можно только **requested budget**.
- **Error analysis:** SAS выигрывает, держась близко к вопросу и донося найденный span в финал;
  Sequential MAS выигрывает, когда его широта сочетается с поздней проверкой ограничений;
  частый провал — потеря верного span на финализации (extraction failure).

## Прямая связь с моим thesis
- **ПРЯМОЙ предшественник.** Exposé прямо ссылается: «under an equal thinking-token budget a
  single agent can match or beat a multi-agent system on multi-hop reasoning».
- **Моя ниша = их явный пробел.** Их Limitations (C.i) дословно: они изучают **только text-only
  multi-hop reasoning**, а «MAS advantages with tools/vision ... are out of scope». Мой thesis —
  это **tool-based planning** с hierarchical MAS — ровно та область, которую они оставляют открытой.
  Это сильнейший аргумент для моего Introduction: я переношу их вопрос на planning.
- **H0 (null result) обоснован именно этой работой:** если у меня MAS не побеждает в диапазоне —
  это «расширяет Tran & Kiela на новый тип задач (planning)», как и записано в моём exposé.
- **Их crossover-механизм = моя гипотеза H2.** У них crossover появляется при деградации контекста;
  у меня кандидат-механизм — рост числа взаимодействующих ограничений. Можно провести параллель:
  высокая constraint-сложность ≈ «трудный для одного прохода» контекст.
- **Методологические заимствования (прямо применимы):**
  - Их Sequential MAS — «чистейший» аналог single-agent (та же серия рассуждений, но через
    явные сообщения). Полезно как идейная опора для моего Условия 2 (verify/revise) vs MAS.
  - Их вывод про **ненадёжность API-budget** — аргумент в пользу моего **self-hosted** контроля
    бюджета токенайзером (у меня этого confound нет — большой плюс, который стоит подчеркнуть).
  - Их **bootstrap CI** + matched-budget — методологический шаблон для моего анализа.
  - Промпты всех 5 MAS-архитектур (Appendix D) — готовый референс для моей реализации.
- **Совпадение модели:** они используют **Qwen3-30B-A3B** — очень близко к моему Qwen3 32B
  (хотя у них MoE-вариант A3B, у меня dense 32B — стоит отметить различие).

## Где цитируется в thesis
- **Introduction / Motivation** — центральная мотивация и постановка (equal budget, SAS≥MAS).
- **Related Work** — главный budget-controlled SAS-vs-MAS источник; явно показать, что они на
  reasoning, я на planning.
- **Methodology** — обоснование matched-budget контроля; критика API-budget → мой self-hosted подход.
- **Discussion** — параллель «context degradation ↔ constraint complexity» как объяснение crossover.

## Точные цитаты (под 15 слов каждая)
- "single-agent systems are more information-efficient"
- "MAS advantages with tools/vision or safety constraints are out of scope"

## Важные различия (подчеркнуть в Related Work)
- Они: text-only **multi-hop reasoning**; я: **tool-based meeting planning**.
- Они: контроль **thinking tokens** через API (ненадёжно); я: **self-hosted**, точный токенайзер.
- Они: **debate/sequential/ensemble** MAS; я: **hierarchical** supervisor-worker-aggregator-critic.
- Они: scoring = LLM-as-judge (semantic match); я: **solver-based partial-credit** satisfaction rate.

## BibTeX
```bibtex
@article{tran2026singleagent,
  title={Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets},
  author={Tran, Dat and Kiela, Douwe},
  journal={arXiv preprint arXiv:2604.02460},
  year={2026}
}
```
