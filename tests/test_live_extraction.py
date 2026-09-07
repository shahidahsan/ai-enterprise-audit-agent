"""Live regression test for the tie-out extraction pipeline.

Exercises the real indexed filing, a real LLM call, and the live SEC XBRL API -- no
mocks. This is what actually caught the FY2025-vs-FY2024 column-confusion bug during
manual testing of the "Run Full Audit" UI. Costs real API calls and needs Postgres up
(`docker compose up -d`) with the sample filing indexed, so it's opt-in, not part of the
default suite: run explicitly with `uv run pytest -m live -v`.

conftest.py's autouse fixture resets Settings env vars to a fake OPENAI_API_KEY for test
isolation; `real_openai_key` below restores the real key from .env for this file only.

document_id is pinned to Apple's filing explicitly: since Phase 8 (multi-document upload),
the same vector store collection can hold several companies' chunks side by side, so an
unscoped search here would silently mix in whichever other filings happen to be indexed
(this actually happened -- adding NVIDIA's and Microsoft's filings during manual Phase 8
testing broke this test until it was pinned, the same way real production usage always
scopes a query to one selected document).
"""
import pytest
from dotenv import dotenv_values

from src.agent.audit_checklist import FINANCIAL_CHECKS, run_tie_out_check
from src.agent.react_agent import build_llm
from src.config import get_settings
from src.indexer.vector_store import AuditVectorStore
from src.tools.xbrl_tool import APPLE_CIK

APPLE_DOCUMENT_ID = f"{APPLE_CIK}_2025"

pytestmark = pytest.mark.live


@pytest.fixture
def real_openai_key(monkeypatch):
    real_key = dotenv_values(".env").get("OPENAI_API_KEY")
    if not real_key:
        pytest.skip("No real OPENAI_API_KEY in .env; skipping live regression test")
    monkeypatch.setenv("OPENAI_API_KEY", real_key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def live_llm(real_openai_key):
    return build_llm()


@pytest.fixture
def live_store(real_openai_key):
    return AuditVectorStore()


@pytest.mark.parametrize("check", FINANCIAL_CHECKS, ids=lambda c: c["concept"])
def test_tie_out_extraction_matches_real_xbrl_figure(check, live_llm, live_store):
    result = run_tie_out_check(
        check, llm=live_llm, store=live_store, fiscal_year=2025, document_id=APPLE_DOCUMENT_ID, cik=APPLE_CIK
    )

    assert result["status"] == "PASSED", (
        f"{check['label']} extraction mismatch -- extracted={result.get('extracted_value')}, "
        f"detail={result.get('detail')}"
    )
