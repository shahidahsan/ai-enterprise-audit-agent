"""Tests for extracting company name / fiscal year from an uploaded filing's cover page."""
from unittest.mock import MagicMock

from src.agent.schemas import FilingMetadata
from src.indexer.metadata_extractor import extract_filing_metadata


def _fake_llm(company_name, fiscal_year):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = FilingMetadata(
        company_name=company_name, fiscal_year=fiscal_year
    )
    return llm


def test_extract_filing_metadata_returns_company_and_year():
    llm = _fake_llm("Apple Inc.", 2025)

    metadata = extract_filing_metadata(llm, "Apple Inc. Annual Report on Form 10-K for fiscal year ended 2025")

    assert metadata.company_name == "Apple Inc."
    assert metadata.fiscal_year == 2025


def test_extract_filing_metadata_passes_cover_page_text_to_llm():
    llm = _fake_llm("Apple Inc.", 2025)
    cover_page = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION FORM 10-K Apple Inc."

    extract_filing_metadata(llm, cover_page)

    prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert cover_page in prompt
