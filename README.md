# Enterprise Compliance & Audit Agent

A financial/compliance audit agent positioned as an **internal-audit / SOX-compliance analyst
copilot** (the category MindBridge AI, Trullion, and Datasnipper compete in) rather than a generic
"chat with a PDF" tool. Upload any company's 10-K and it: ingests it into a vector store, then
applies real audit techniques on top of retrieval -- **tie-out** (cross-checking an LLM-read
figure against that company's own structured SEC data, not just trusting the PDF prose) and a
**disclosure checklist** grounded in actual cited requirements (FASB ASC codification, SEC
Regulation S-K), not keyword guessing. Multiple filings can be indexed and queried side by side,
each auto-identified by company/CIK/fiscal-year from its own cover page -- verified against real
filings from three different companies, not just the one it was originally built around. A ReAct
agent backed by 4 deterministic tools handles free-form Q&A; a separate "Run Full Audit" mode runs
the tie-out + disclosure checks end-to-end as a fixed checklist. Every run traces to LangSmith and
the whole pipeline is scored against a ground-truth Q&A set with DeepEval.

## Architecture

```
                    +-----------------------------+
                    |   Any company's 10-K PDF     |
                    |  (upload, or the bundled      |
                    |   Apple sample_filing.pdf)   |
                    +--------------+---------------+
                                   |
                src/indexer/document_indexer.py: cover page ->
                extract_filing_metadata() (company/fiscal year) ->
                sec_lookup.resolve_cik() -> document_id = {cik}_{fy} ->
                chunker.py (800/150-token chunks, tagged with document_id)
                                   |
                                   v
                    +-----------------------------+      +----------------------------+
                    | Postgres + pgvector          |      | SEC EDGAR XBRL API          |
                    | one collection, many filings |      | (data.sec.gov, live)        |
                    | filtered by document_id      |      +--------------+-------------+
                    | (src/indexer/vector_store.py)|                     |
                    | + document_registry.py       |                     |
                    | (data/documents.json)        |                     |
                    +--------------+---------------+                     |
                                   |                                     |
+---------------------+            v                                    |
| Streamlit UI         |    +-------------------+    +--------------------+ |
| sidebar: doc picker  |<-->|    ReAct Agent    |<-->| LangSmith Tracing  | |
| + upload; (app.py)   |    | (src/agent/)      |    | (src/observability)| |
+---------------------+    +---------+---------+    +--------------------+ |
        |                          | (native tool calling, scoped to      |
        |                          |  the selected document_id/cik)       |
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
| fixed checklist: 4 tie-out checks + 4 disclosure checks,  |
| scoped to whichever filing is selected -> one summary     |
| table, instead of open-ended chat                         |
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
# auto-extracts company name + fiscal year from the cover page and resolves its SEC CIK.
# Index any other company's 10-K the same way, or upload one from the Streamlit sidebar --
# multiple filings coexist and are selectable. (Re-running on the same file re-indexes it;
# there's no dedup against a prior run of the identical file yet.)

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

1. **Upload & identify** (`src/indexer/document_indexer.py`): a PDF's cover page is read and an
   LLM call (`metadata_extractor.py`) extracts the filer's company name and fiscal year; that
   name is fuzzy-matched against SEC's public ticker directory (`sec_lookup.resolve_cik()`) to
   get its CIK. `document_id = "{cik}_{fiscal_year}"` identifies this filing from here on.
2. **Ingest** (`chunker.py`): the PDF is loaded page-by-page via `pypdf`, tagged with page number,
   a detected "Item N." section heading, and the `document_id`, then split into ~800-token chunks
   (150-token overlap) with a tiktoken-based length function.
3. **Index**: chunks are embedded (OpenAI) and stored in one shared Postgres/`pgvector` collection
   (`AuditVectorStore`); the filing is registered in a local JSON registry
   (`document_registry.py`) so the UI can list and select among all indexed filings. Multiple
   companies' chunks coexist in that one collection, distinguished only by `document_id` metadata
   -- every retrieval call filters on it, so filings never cross-contaminate each other.
4. **Ask**: a user question comes in through the Streamlit chat panel (or directly via
   `run_audit_query()`), scoped to whichever filing is selected in the sidebar.
5. **Reason**: the ReAct agent (`src/agent/react_agent.py`) runs a native-tool-calling loop,
   choosing among 4 tools (`src/tools/`) until it has enough information:
   - `document_search` -- semantic retrieval with page/section citations, scoped to the selected
     `document_id`
   - `calculate_variance` -- deterministic period-over-period variance math
   - `compliance_flag_checker` -- checks a clause against a cited disclosure-element checklist
   - `verify_against_xbrl` -- **tie-out**: cross-checks a figure against the selected company's
     live structured XBRL data, catching cases where the LLM misread the filing's prose
   The system prompt also names the selected company and fiscal year explicitly (`build_system_prompt()`),
   so the agent can answer even meta-questions like "what filing are we looking at."
6. **Guardrail**: the free-text answer is passed through a second LLM call with
   `.with_structured_output(AuditFinding)`, enforcing the schema.
7. **Trace**: every LLM call and tool invocation is traced to LangSmith
   (`src/observability/tracing.py` bridges `Settings` into the env vars LangChain's tracer reads).
8. **Run Full Audit** (`src/agent/audit_checklist.py`): a separate, deterministic path -- instead
   of open-ended chat, it runs a fixed checklist (4 tie-out checks + 4 disclosure checks) against
   the selected filing and returns one summary row per check. `extract_figure` pulls a number out
   of retrieved passages by asking the LLM for the value *exactly as printed* plus its stated unit
   (never the converted raw-dollar amount); the unit multiplication happens in Python,
   deterministically -- same reasoning as `calculate_variance`, arithmetic belongs in code, not a
   prompt.
9. **Evaluate**: `src/evals/runner.py` re-runs the ReAct agent path against a 6-question
   ground-truth dataset (`src/evals/dataset.py`), scores each answer with DeepEval, and writes
   `eval_results.json`.
10. **Display**: the Streamlit UI has a sidebar for selecting/uploading filings, an "Interactive
    Audit" tab (chat + collapsible tool-call steps), a "Run Full Audit" tab (checklist table with
    live per-check progress), and a right panel showing the latest eval scorecard, last-query
    latency, and a LangSmith link.

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
- The financial-statement retrieval query (`audit_checklist._STATEMENT_QUERY`) assumes the common
  "Consolidated Statements of Operations" caption; a filing captioned differently (e.g. "...of
  Income") may retrieve less reliably. Verified working for Apple, NVIDIA, and Microsoft.
- No dedup: indexing the same file twice creates duplicate chunks rather than replacing them.
- The document registry (`data/documents.json`) is a local JSON file, not part of Postgres itself
  -- fine for a single-user demo, would need a real table (or at least file locking) for
  concurrent multi-user use.
