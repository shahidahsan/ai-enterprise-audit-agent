# Enterprise Compliance & Audit Agent

A production-grade financial/compliance audit agent. It ingests SEC filings (10-Ks) into a
vector store, answers questions with a ReAct agent backed by deterministic tools, enforces a
structured Pydantic output schema, traces every run to LangSmith, and scores itself against a
ground-truth Q&A set with DeepEval. A two-column Streamlit UI ties it all together.

## Architecture

```
                    +-----------------------------+
                    |       PDF Filing             |
                    |  (data/sample_filing.pdf)    |
                    +--------------+---------------+
                                   |
                     src/indexer/ (chunker.py)
                     RecursiveCharacterTextSplitter,
                     800/150-token chunks, page + section
                                   |
                                   v
                    +-----------------------------+
                    | Postgres + pgvector          |
                    | (src/indexer/vector_store.py)|
                    +--------------+---------------+
                                   |
+-------------------+              v
|                   |    +-------------------+    +--------------------+
|  Streamlit UI     |<-->|    ReAct Agent    |<-->| LangSmith Tracing  |
|  (src/ui/app.py)  |    | (src/agent/)      |    | (src/observability)|
+-------------------+    +---------+---------+    +--------------------+
                                   | (native tool calling)
                    +--------------+--------------+--------------------+
                    |                             |                    |
                    v                             v                    v
       +-----------------------+     +--------------------------+  +------------------------+
       | document_search       |     | calculate_variance       |  | compliance_flag_checker |
       | (src/tools/search_    |     | (src/tools/calculator_   |  | (src/tools/compliance_  |
       |  tool.py)             |     |  tool.py)                |  |  tool.py)                |
       +-----------------------+     +--------------------------+  +------------------------+

                    src/evals/ (DeepEval) scores the whole pipeline against
                    a ground-truth dataset -> eval_results.json -> shown in the UI
```

The agent's final answer is always validated against `AuditFinding`
(`src/agent/schemas.py`): `summary`, `data_points`, `compliance_status`
(`PASSED` / `FLAGGED` / `NEEDS_REVIEW`), `sources`. That's the "Pydantic output guardrail."

## Tech stack

| Concern | Choice |
|---|---|
| Language / packaging | Python 3.11, `uv` |
| LLM orchestration | `langchain` + `langchain-openai` / `langchain-anthropic` (pluggable, see below) |
| Vector store | PostgreSQL + `pgvector`, via `langchain-postgres`, run through Docker |
| Embeddings | OpenAI `text-embedding-3-small` |
| Observability | LangSmith (`langsmith`) |
| Evaluation | DeepEval (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualRecallMetric`) |
| UI | Streamlit |
| Testing | `pytest`, `pytest-mock`, `pytest-asyncio` |

**LLM provider is pluggable.** `Settings.llm_provider` (`.env`) is `"openai"` by default
(`gpt-4o-mini`); set it to `"anthropic"` plus `ANTHROPIC_API_KEY` and `AGENT_MODEL=claude-sonnet-5`
(or similar) to switch. `build_llm()` in `src/agent/react_agent.py` picks the right client. Note
DeepEval's real metric class is `ContextualRecallMetric` — the PRD's generic "ContextRecallMetric"
naming is Ragas terminology, not DeepEval's.

## Setup

**Prerequisites:** `uv`, Docker Desktop, an OpenAI API key (embeddings always use OpenAI), and
optionally an Anthropic key and/or a free LangSmith account.

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env: OPENAI_API_KEY, optionally ANTHROPIC_API_KEY / LANGCHAIN_API_KEY

# 3. Start Postgres (pgvector-enabled)
docker compose up -d

# 4. Index the sample filing (Apple's real FY2025 10-K, from SEC EDGAR/Apple IR)
uv run python -m src.indexer data/sample_filing.pdf
# (re-running this appends a fresh copy of the chunks; there's no dedup yet)

# 5. Run the tests
uv run pytest

# 6. Run the evaluation suite (writes eval_results.json)
uv run python -m src.evals.runner

# 7. Launch the UI
PYTHONPATH=. uv run streamlit run src/ui/app.py
```

`PYTHONPATH=.` matters: Streamlit puts the script's own directory (`src/ui/`) on `sys.path`
instead of the project root, so plain `src.*` imports fail without it.

## System flow

1. **Ingest** (`src/indexer/`): a PDF is loaded page-by-page via `pypdf`, tagged with page number
   and a detected "Item N." section heading, then split into ~800-token chunks (150-token
   overlap) with a tiktoken-based length function.
2. **Index**: chunks are embedded (OpenAI) and stored in Postgres via `pgvector`
   (`AuditVectorStore`).
3. **Ask**: a user question comes in through the Streamlit chat panel (or directly via
   `run_audit_query()`).
4. **Reason**: the ReAct agent (`src/agent/react_agent.py`) runs a native-tool-calling loop,
   choosing among `document_search`, `calculate_variance`, and `compliance_flag_checker`
   (`src/tools/`) until it has enough information.
5. **Guardrail**: the free-text answer is passed through a second LLM call with
   `.with_structured_output(AuditFinding)`, enforcing the schema.
6. **Trace**: every LLM call and tool invocation is traced to LangSmith
   (`src/observability/tracing.py` bridges `Settings` into the env vars LangChain's tracer reads).
7. **Evaluate**: `src/evals/runner.py` re-runs the agent against a 6-question ground-truth dataset
   (`src/evals/dataset.py`), scores each answer with DeepEval, and writes `eval_results.json`.
8. **Display**: the Streamlit UI's left panel shows the live chat + collapsible tool-call steps;
   the right panel shows the latest eval scorecard, last-query latency, and a LangSmith link.

## Known nuances / limitations

- Re-running the indexer CLI appends duplicate chunks rather than replacing the collection.
- DeepEval's LLM-as-judge is itself imperfect — in one real eval run it appeared to misread a
  figure from the retrieval context. Treat eval scores as strong signal, not ground truth.
- The agent tends to include more context than the literal question asks for (e.g. prior-year
  comparisons), which DeepEval's `AnswerRelevancyMetric` penalizes even though it's arguably
  useful audit behavior.

## Loom demo outline (~7 min)

1. **Intro (30s)** — what this is: an audit agent over a real 10-K, built strictly TDD.
2. **Setup (30s)** — `.env`, `docker compose up -d`, mention Postgres+pgvector.
3. **Ingestion (1 min)** — run `python -m src.indexer`, pop open TablePlus/psql to show the
   `langchain_pg_embedding` rows with page/section metadata.
4. **Tests (1 min)** — `uv run pytest`, all green; call out the TDD-first workflow and the
   50-line-function / 300-line-file discipline.
5. **Live agent (2 min)** — open the Streamlit UI, ask 2-3 real questions (net sales, a variance
   question, a compliance check), show the compliance badge and tool-call expanders.
6. **Tracing (1 min)** — jump to the LangSmith project, open the trace for one of those queries,
   walk through the nested LLM/tool spans.
7. **Evals (1 min)** — show the eval scorecard in the UI and `eval_results.json`, discuss the
   Faithfulness/Relevancy/Context-Recall numbers and what they mean.
8. **Wrap-up (30s)** — recap the pluggable-provider design and the overall architecture.
