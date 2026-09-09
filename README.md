# Enterprise Compliance & Audit Agent

An AI-powered audit agent for SEC Form 10-K filings, built as an internal audit and compliance
analyst copilot. The system ingests financial filings, verifies extracted figures against
authoritative government data, and checks disclosures against real accounting and securities
regulations. It supports multiple companies and filings at once, each automatically identified
from its own cover page.

Rather than simply summarizing unstructured text, the system applies two techniques used in
professional audit practice:

- **Tie-out.** Every financial figure the agent extracts from a filing is cross-checked against
  that company's own structured data filed with the SEC (XBRL), rather than trusted at face value.
- **Disclosure review.** Filing language is checked against specific, cited requirements from
  FASB accounting standards and SEC Regulation S-K, rather than matched against generic keywords.

A ReAct-based agent with four purpose-built tools answers natural language questions about a
filing, and a "Run Full Audit" mode executes the full tie-out and disclosure review automatically
as a fixed checklist. Every agent run is traced through LangSmith, and answer quality is measured
with an automated evaluation suite (DeepEval) against a ground-truth question set.

## Key Features

- **Multi-document support.** Upload any company's 10-K. The system identifies the company,
  fiscal year, and SEC CIK directly from the filing, and multiple filings can be indexed and
  queried side by side.
- **Tie-out verification.** Cross-checks figures the agent extracts against live SEC XBRL data.
- **Disclosure compliance checks** grounded in cited FASB and SEC requirements, not keyword
  matching.
- **ReAct agent with four tools**: semantic search, deterministic variance calculation,
  compliance checking, and XBRL verification.
- **Structured, schema-validated output** for every agent response, built on Pydantic.
- **Full observability** through LangSmith tracing of every LLM call and tool invocation.
- **Automated evaluation suite** using DeepEval, scoring faithfulness, answer relevancy, and
  context recall against a ground-truth question set.
- **Two-mode Streamlit interface**: an interactive chat view and a one-click full audit
  checklist.
- **Pluggable LLM provider.** OpenAI or Anthropic, selected through configuration.

## Architecture

```
                     Any company's 10-K PDF
                (upload, or the bundled Apple filing)
                              |
                              v
        document_indexer.py: read cover page, extract company
        name and fiscal year, resolve SEC CIK, assign a
        document_id, then chunk and tag the filing (chunker.py)
                              |
                              v
   +------------------------------------+     +----------------------------+
   |  Postgres + pgvector                |     |  SEC EDGAR XBRL API        |
   |  One collection, many filings,      |     |  (data.sec.gov, live)      |
   |  filtered by document_id            |     +--------------+-------------+
   |  (vector_store.py, document_        |                    |
   |   registry.py)                      |                    |
   +------------------+-------------------+                    |
                       |                                        |
   +-------------------+-------+     +-------------------+     |
   |  Streamlit UI              |<--->|  ReAct Agent      |<----+
   |  document picker, upload,  |     |  (src/agent/)     |
   |  chat, full audit view     |     +---------+---------+
   +-------------------+-------+               |
                       |               (native tool calling,
                       |                scoped to the selected
                       |                document and company)
                       |          +----------------+----------------+---------------------------+
                       |          |                |                |                           |
                       |          v                v                v                           v
                       |   document_search   calculate_variance  compliance_flag_checker  verify_against_xbrl
                       |                                                                    (tie-out check)
                       v
        "Run Full Audit" (audit_checklist.py): a fixed checklist of
        tie-out and disclosure checks against the selected filing,
        producing one summary table instead of open-ended chat

        LangSmith traces every step above. A separate evaluation
        suite (DeepEval) scores the agent against a ground-truth
        question set and writes eval_results.json, shown in the UI.
```

Every agent response is validated against a schema (`AuditFinding` in `src/agent/schemas.py`)
before it is returned: a summary, key data points, a compliance status of `PASSED`, `FLAGGED`,
or `NEEDS_REVIEW`, and the sources cited.

## Technology Stack

| Concern | Choice |
|---|---|
| Language and packaging | Python 3.11, `uv` |
| LLM orchestration | `langchain`, with `langchain-openai` and `langchain-anthropic` |
| Vector store | PostgreSQL with `pgvector`, via `langchain-postgres`, run through Docker |
| Embeddings | OpenAI `text-embedding-3-small` |
| Observability | LangSmith |
| Evaluation | DeepEval (`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualRecallMetric`) |
| Interface | Streamlit |
| Testing | `pytest`, `pytest-mock`, `pytest-asyncio` |

The LLM provider is configurable rather than hardcoded. `Settings.llm_provider` defaults to
`openai` (model `gpt-4o-mini`); setting it to `anthropic` along with `ANTHROPIC_API_KEY` and an
Anthropic model name switches the agent to Claude with no code changes. `build_llm()` in
`src/agent/react_agent.py` selects the client at runtime.

## Getting Started

**Prerequisites:** `uv`, Docker Desktop, and an OpenAI API key (embeddings always use OpenAI).
An Anthropic API key and a LangSmith account are optional.

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# Edit .env: set OPENAI_API_KEY, and optionally ANTHROPIC_API_KEY and LANGCHAIN_API_KEY.

# 3. Start Postgres (pgvector-enabled)
docker compose up -d

# 4. Index the sample filing (Apple's FY2025 10-K)
uv run python -m src.indexer data/sample_filing.pdf
# This extracts the company name and fiscal year from the cover page and resolves its SEC CIK
# automatically. Any other company's 10-K can be indexed the same way, or uploaded from the
# Streamlit sidebar. Multiple filings can be indexed and selected independently.

# 5. Run the test suite (mocked; no external calls; live tests are excluded by default)
uv run pytest

# 5b. Optional: run the live regression suite against real OpenAI and SEC XBRL data
uv run pytest -m live -v

# 6. Run the evaluation suite (writes eval_results.json)
uv run python -m src.evals.runner

# 7. Launch the interface
PYTHONPATH=. uv run streamlit run src/ui/app.py
```

`PYTHONPATH=.` is required because Streamlit adds the script's own directory to `sys.path`
instead of the project root, which would otherwise break the project's internal imports.

## System Flow

1. **Upload and identify.** A filing's cover page is read, and an LLM call extracts the
   company's legal name and fiscal year. That name is matched against SEC's public ticker
   directory to resolve its CIK. A `document_id` (company CIK plus fiscal year) identifies the
   filing from this point forward.
2. **Ingest.** The PDF is loaded page by page, tagged with page number, a detected filing
   section, and its `document_id`, then split into overlapping token-bounded chunks.
3. **Index.** Chunks are embedded and stored in a shared Postgres and pgvector collection. The
   filing is also registered in a local index so the interface can list and select among all
   indexed filings. Every retrieval is filtered by `document_id`, so filings never cross
   contaminate one another.
4. **Ask.** A user question arrives through the chat interface, scoped to whichever filing is
   currently selected.
5. **Reason.** The agent runs a native tool-calling loop, choosing among four tools as needed:
   - `document_search`: semantic retrieval with page and section citations, scoped to the
     selected filing.
   - `calculate_variance`: deterministic period-over-period variance calculation.
   - `compliance_flag_checker`: checks a clause against a cited disclosure requirement.
   - `verify_against_xbrl`: the tie-out check, cross-referencing a figure against the selected
     company's live structured XBRL data.

   The agent's system prompt also states the selected company and fiscal year explicitly, so it
   can correctly answer even a question about which filing is currently loaded.
6. **Enforce structure.** The agent's answer is passed through a second call that enforces the
   `AuditFinding` schema before it is returned to the user.
7. **Trace.** Every LLM call and tool invocation is traced to LangSmith.
8. **Run a full audit.** As an alternative to open-ended chat, this mode runs a fixed checklist
   of four tie-out checks and four disclosure checks against the selected filing and returns one
   summary row per check. Figures are extracted as printed, with their unit stated separately;
   the unit conversion to a raw dollar value is performed deterministically in code rather than
   left to the model.
9. **Evaluate.** A separate evaluation script re-runs the agent against a ground-truth question
   set, scores each answer with DeepEval, and writes the results to `eval_results.json`.
10. **Display.** The interface provides a sidebar for selecting or uploading filings, a chat
    view with expandable tool-call detail, a full audit view with live per-check progress, and a
    panel showing the latest evaluation scores, response latency, and a link to LangSmith.

## Known Limitations

- Indexing the same file twice creates duplicate chunks rather than replacing the existing ones.
- DeepEval's LLM-based judge is itself imperfect. In one evaluation run it appeared to misread a
  figure from the retrieved context, so evaluation scores should be treated as a strong signal
  rather than ground truth.
- The agent sometimes includes more context than a question strictly requires (for example,
  prior-year comparisons), which the relevancy metric penalizes even though this is often useful
  audit behavior.
- The compliance checker's keyword and pattern matching for each required disclosure element is
  an approximation, not a certified compliance determination. The cited requirements are real;
  the text-matching heuristic used to check for them is not.
- The financial statement retrieval query assumes the common "Consolidated Statements of
  Operations" caption. A filing captioned differently may retrieve less reliably. This has been
  verified working for Apple, NVIDIA, and Microsoft filings.
- The local document registry is a JSON file rather than a database table, which is sufficient
  for single-user use but would need a proper table, or at least file locking, to support
  concurrent multi-user access.
