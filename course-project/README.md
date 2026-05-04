# Dev Team — Мультиагентна система розробки ПЗ

Мультиагентна система, що симулює AI-команду розробки за патерном **Business Analyst → Developer ↔ QA** з Human-in-the-Loop затвердженням специфікації.

---

## Архітектура

```
User Story
    │
    ▼
Business Analyst ──► SpecOutput
    │                (title, requirements, acceptance_criteria, complexity)
    ▼
HITL Approval ──► Відхилено → BA переробляє spec (цикл)
    │
    ▼
Developer ──► CodeOutput
    │          (files_created, tests_passed, dependencies_installed)
    ▼
QA Engineer ──► ReviewOutput (verdict, score 0–1, issues)
    │
    ├── REVISION_NEEDED → Developer (до 5 ітерацій)
    └── APPROVED ──► Готово
```

**Патерн:** Evaluator-Optimizer — QA оцінює, Developer оптимізує. Лінійна частина (BA → Developer) — Prompt Chaining з HITL gate.

---

## Де шукати головне

> Веб-інтерфейс і TraceVault — це розширення понад умови завдання. Основна реалізація — у пакетах нижче.

| Що | Де | Навіщо |
|----|-----|--------|
| Агенти (BA / Developer / QA) | `agents/ba.py`, `agents/developer.py`, `agents/qa.py` | Реалізація ролей, ReAct-цикл, structured output |
| Схеми Pydantic | `agents/schemas.py` | `SpecOutput`, `CodeOutput`, `ReviewOutput` |
| Pipeline orchestration | `pipeline.py` | BA→HITL→Developer↔QA, max 5 ітерацій |
| ReAct-цикл + middleware | `agentflow/agent.py`, `agentflow/middleware.py` | `AgentRunner`, `BudgetMiddleware`, retry |
| MCP-інтеграція | `tools/mcp_fs.py`, `tools/mcp_repl.py` | Filesystem + Python REPL через MCP |
| RAG | `tools/rag.py` | Hybrid FAISS + BM25 по `rag_docs/` |
| Промпти агентів | `prompts/*.md` | System prompts, JSON-suffix для structured output |
| Юніт-тести | `tests/unit/` | Без зовнішніх сервісів |
| Інтеграційні тести | `tests/integration/` | MCP, MongoDB, embedding |
| Live-тести агентів | `tests/live/` | З живим LLM |

---

## Реалізовані вимоги завдання

### Агенти та інструменти

| Агент | Інструменти |
|-------|------------|
| **Business Analyst** | DuckDuckGo Search, RAG (FAISS + BM25 по `rag_docs/`) |
| **Developer** | Filesystem MCP (read/write файлів), Python REPL MCP (виконання коду, pytest, pip install), DuckDuckGo Search, RAG |
| **QA Engineer** | Filesystem MCP (читання), Python REPL MCP (запуск тестів) |

### Structured Output

Pydantic-контракти (`agents/schemas.py`):

```python
class SpecOutput(BaseModel):
    title: str
    requirements: list[str]
    acceptance_criteria: list[str]
    estimated_complexity: Literal["simple", "medium", "complex"]

class CodeOutput(BaseModel):
    summary: str
    files_created: list[str]
    dependencies_installed: list[str]
    tests_passed: bool

class ReviewOutput(BaseModel):
    verdict: Literal["APPROVED", "REVISION_NEEDED"]
    score: float          # 0.0–1.0
    issues: list[str]
    suggestions: list[str]
```

Workaround для локальних LLM: JSON-suffix у system prompt → regex-extract → Pydantic validate → retry до 3 разів із повним збереженням контексту.

### Human-in-the-Loop

HITL gate між BA і Developer: специфікація виводиться на затвердження. При відхиленні — BA переробляє з урахуванням feedback. Синхронізація через `asyncio.Event`.

### MCP-інтеграція (бонус)

- **Filesystem MCP** — безпечний доступ до файлової системи через стандартизований протокол (`tools/mcp_fs.py` + `PathNormalizer`)
- **Python REPL MCP** — ізольоване виконання коду в Docker-контейнері

### Моніторинг — TraceVault (замість Langfuse/LangSmith)

Власна реалізація системи трейсингу (`tracevault/`):

- Кожен виклик LLM зберігається як **span** з input/output, токенами, latency, tool calls
- Фільтрація трейсів по агенту, статусу, сесії, проєкту
- Версіонування промптів з rollback
- **LLM-as-a-Judge** оцінки з агрегованою статистикою pass rate по агентах
- SSE real-time оновлення

Dashboard: `http://localhost:8090`

### RAG Knowledge Base

`rag_docs/` — 4 документи (Python stdlib, FastAPI tutorial, Google Python Style Guide, coding standards). Hybrid retrieval: FAISS (dense) + BM25 (sparse) з RRF fusion. Індекс auto-rebuild при зміні документів.

---

## Запуск

### 1. Налаштування

Створіть `.env` файл у корені проєкту та заповніть змінні (див. секцію **Configuration** у [`README_full.md`](README_full.md) для прикладу та налаштування зовнішніх сервісів):

```bash
cp .env.example .env   # якщо є, або створіть вручну
# Відредагуйте .env під своє оточення
```

### 2. Залежності

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Seed промптів у MongoDB

```bash
python scripts/seed_prompts.py
```

### 4. Перевірка сервісів

```bash
python scripts/check_services.py
```

### 5. Запуск застосунку

```bash
# Основний застосунок
uvicorn app:app --port 8000 --reload

# TraceVault dashboard (окремий термінал)
uvicorn tracevault.server:app --port 8090 --reload
```

**Основний UI:** `http://localhost:8000`  
**TraceVault:** `http://localhost:8090`

---

## Тести

```bash
# Юніт-тести (без зовнішніх сервісів)
.venv/bin/pytest tests/unit/

# Інтеграційні тести (MongoDB + MCP + embedding)
.venv/bin/pytest tests/integration/

# Live-тести агентів (потрібен LLM)
.venv/bin/pytest tests/live/
```

**105 юніт-тестів, 38 інтеграційних.** LLM-as-a-Judge тести — у `tracevault` через Evaluations screen.

---

## Структура проєкту

```
course-project/
├── agents/              # BA, Developer, QA — реалізація агентів
├── agentflow/           # AgentRunner (ReAct), Pipeline, middleware, MCP client
├── tracevault/          # Моніторинг: traces, spans, prompts, evaluations, SSE
├── tools/               # MCP wrappers, RAG, DuckDuckGo search
├── prompts/             # System prompts агентів (.md → MongoDB)
├── rag_docs/            # Knowledge base для RAG
├── tests/               # unit / integration / live
├── pipeline.py          # Orchestration BA→HITL→Dev↔QA
├── config.py            # Pydantic Settings (читає .env)
└── README_full.md       # Повна технічна документація
```

Повна технічна документація та інструкції з налаштування зовнішніх сервісів (LLM, MCP, MongoDB, Embedding): [`README_full.md`](README_full.md)
