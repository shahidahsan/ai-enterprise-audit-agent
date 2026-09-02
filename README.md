# Enterprise Compliance & Audit Agent

A financial/compliance audit agent positioned as an **internal-audit / SOX-compliance analyst
copilot** (the category MindBridge AI, Trullion, and Datasnipper compete in) rather than a generic
"chat with a PDF" tool. It ingests SEC filings (10-Ks) into a vector store, then applies real
audit techniques on top of retrieval: **tie-out** (cross-checking an LLM-read figure against the
filing's own structured SEC data, not just trusting the PDF prose) and a **disclosure checklist**
grounded in actual cited requirements (FASB ASC codification, SEC Regulation S-K), not keyword
guessing. A ReAct agent backed by 4 deterministic tools handles free-form Q&A; a separate "Run
Full Audit" mode runs the tie-out + disclosure checks end-to-end as a fixed checklist. Every run
traces to LangSmith and the whole pipeline is scored against a ground-truth Q&A set with DeepEval.

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
                    +-----------------------------+      +----------------------------+
                    | Postgres + pgvector          |      | SEC EDGAR XBRL API          |
                    | (src/indexer/vector_store.py)|      | (data.sec.gov, live)        |
                    +--------------+---------------+      +--------------+-------------+
                                   |                                      |
+-------------------+              v                                     |
|                   |    +-------------------+    +--------------------+ |
|  Streamlit UI     |<-->|    ReAct Agent    |<-->| LangSmith Tracing  | |
|  (src/ui/app.py)  |    | (src/agent/)      |    | (src/observability)| |
+-------------------+    +---------+---------+    +--------------------+ |
        |                          | (native tool calling)               |
        |          +---------------+------+-------------------+---------+
        |          |                      |                   |
        |          v                      v                   v
        |  +----------------+  +---------------------+  +------------------------+
        |  | document_search|  | calculate_variance   |  | compliance_flag_checker |
        |  +----------------+  +---------------------+  +------------------------+
        |          |
        |          v (tie-out)
        |  +--------------------------+
        |  | verify_against_xbrl      |
        |  | (src/tools/xbrl_tool.py) |
        |  +--------------------------+
        v
+----------------------------------------------------------+
| "Run Full Audit" (src/agent/audit_checklist.py)           |
| fixed checklist: 4 tie-out checks + 4 disclosure checks   |
| -> one summary table, instead of open-ended chat          |
+----------------------------------------------------------+

                    src/evals/ (DeepEval) scores the ReAct agent path against
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

# 5. Run the tests (mocked, no external calls -- live tests are excluded by default)
uv run pytest

# 5b. Optional: run the live regression against real OpenAI + live SEC XBRL data
uv run pytest -m live -v

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
   choosing among 4 tools (`src/tools/`) until it has enough information:
   - `document_search` -- semantic retrieval with page/section citations
   - `calculate_variance` -- deterministic period-over-period variance math
   - `compliance_flag_checker` -- checks a clause against a cited disclosure-element checklist
   - `verify_against_xbrl` -- **tie-out**: cross-checks a figure against SEC's live structured
     XBRL data, catching cases where the LLM misread the filing's prose
5. **Guardrail**: the free-text answer is passed through a second LLM call with
   `.with_structured_output(AuditFinding)`, enforcing the schema.
6. **Trace**: every LLM call and tool invocation is traced to LangSmith
   (`src/observability/tracing.py` bridges `Settings` into the env vars LangChain's tracer reads).
7. **Run Full Audit** (`src/agent/audit_checklist.py`): a separate, deterministic path -- instead
   of open-ended chat, it runs a fixed checklist (4 tie-out checks + 4 disclosure checks) end to
   end and returns one summary row per check. `extract_figure` pulls a number out of retrieved
   passages by asking the LLM for the value *exactly as printed* plus its stated unit (never the
   converted raw-dollar amount); the unit multiplication happens in Python, deterministically --
   same reasoning as `calculate_variance`, arithmetic belongs in code, not a prompt.
8. **Evaluate**: `src/evals/runner.py` re-runs the ReAct agent path against a 6-question
   ground-truth dataset (`src/evals/dataset.py`), scores each answer with DeepEval, and writes
   `eval_results.json`.
9. **Display**: the Streamlit UI has an "Interactive Audit" tab (chat + collapsible tool-call
   steps) and a "Run Full Audit" tab (checklist table with live per-check progress), plus a right
   panel showing the latest eval scorecard, last-query latency, and a LangSmith link.

## A real bug this caught (and how it was fixed)

Manual testing of "Run Full Audit" surfaced a genuine LLM extraction failure: the agent read
Apple's **FY2024** operating income and R&D figures out of a 3-year comparison table instead of
FY2025's, because both years sit in the same table/column-adjacent layout. The XBRL tie-out
tool flagged both as mismatches against SEC's structured data -- exactly the failure mode it
exists to catch. Root cause and fix:
1. **Retrieval drift**: the fixed query for those two concepts matched a segment-reconciliation
   table (which repeats the same "Research and development" line-item label) instead of the
   actual Consolidated Statements of Operations. Fixed by anchoring the retrieval query on the
   statement's own title/structure.
2. **Unit-scale arithmetic**: the first fix asked the LLM to do the "in millions" -> raw-USD
   conversion itself via a prompt instruction. That fixed the immediate case but *introduced a
   different bug* on a later run (over-multiplying gross margin by 1000x) -- confirmed via 3
   repeated live runs. The durable fix: `ExtractedFigure` now has separate `value`/`unit` fields;
   the LLM only reads the number as printed plus its unit label, and Python does the
   multiplication deterministically. Same principle as `calculate_variance` -- never let an LLM
   do arithmetic that code can do exactly.

This is also why `tests/test_live_extraction.py` exists as a real (unmocked) regression test
(`pytest -m live`) -- the DeepEval suite never caught this because it exercises a different code
path (the full ReAct loop), not the one-shot extractor added for the tie-out checklist.

## Known nuances / limitations

- Re-running the indexer CLI appends duplicate chunks rather than replacing the collection.
- DeepEval's LLM-as-judge is itself imperfect — in one real eval run it appeared to misread a
  figure from the retrieval context. Treat eval scores as strong signal, not ground truth.
- The agent tends to include more context than the literal question asks for (e.g. prior-year
  comparisons), which DeepEval's `AnswerRelevancyMetric` penalizes even though it's arguably
  useful audit behavior.
- `compliance_flag_checker`'s keyword/regex matching per required element is a crude proxy for
  whether that element is actually addressed -- illustrative disclosure-checklist tooling, not a
  certified compliance opinion. Citations are real; the text-matching heuristic is not legal advice.
- The XBRL tool and the "Run Full Audit" checklist are hardcoded to Apple's CIK and fiscal 2025;
  multi-company/multi-year support would need the CIK and fiscal year threaded through as
  parameters (natural extension once PDF upload lands).

## Loom demo outline (~8 min)

1. **Intro (30s)** — positioning: not "chat with a PDF" but an internal-audit copilot doing
   tie-out and disclosure-checklist review, the same category as MindBridge AI / Trullion.
2. **Setup (30s)** — `.env`, `docker compose up -d`, mention Postgres+pgvector.
3. **Ingestion (1 min)** — run `python -m src.indexer`, pop open TablePlus/psql to show the
   `langchain_pg_embedding` rows with page/section metadata.
4. **Tests (1 min)** — `uv run pytest`, all green; call out the TDD-first workflow, the
   50-line-function / 300-line-file discipline, and the `-m live` regression suite split.
5. **Live agent (1.5 min)** — open the Streamlit UI's chat tab, ask 2-3 real questions, show the
   compliance badge and tool-call expanders including a `verify_against_xbrl` call.
6. **Run Full Audit (1.5 min)** — click into the checklist tab, walk through the live per-check
   progress, land on the summary table. Narrate the real bug story above: this is where the tie-out
   check caught the agent reading the wrong fiscal year, and how the fix works.
7. **Tracing (1 min)** — jump to the LangSmith project, open the trace for one of those queries.
8. **Evals (1 min)** — show the eval scorecard and `eval_results.json`, discuss what the scores
   mean and why this feature's own bug lived outside the eval suite's coverage.
9. **Wrap-up (30s)** — recap the pluggable-provider design and the overall architecture.
