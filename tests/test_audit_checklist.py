"""Tests for the 'Run Full Audit' checklist orchestration (tie-out + disclosure checks)."""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.agent.audit_checklist import extract_figure, run_disclosure_check, run_full_audit, run_tie_out_check
from src.agent.schemas import ExtractedFigure


def _fake_llm(value, unit="raw"):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = ExtractedFigure(value=value, unit=unit)
    return llm


def _fake_store(docs):
    store = MagicMock()
    store.similarity_search.return_value = docs
    return store


# --- extract_figure ------------------------------------------------------------


def test_extract_figure_returns_value_from_structured_output():
    llm = _fake_llm(416161000000.0)

    value = extract_figure(llm, ["Total net sales $416,161 million"], "total net sales", fiscal_year=2025)

    assert value == 416161000000.0


def test_extract_figure_prompt_names_the_target_fiscal_year_to_disambiguate_comparison_tables():
    llm = _fake_llm(416161000000.0)

    extract_figure(llm, ["2025 2024 2023\n$416,161 $391,035 $383,285"], "total net sales", fiscal_year=2025)

    prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "2025" in prompt
    assert "fiscal year" in prompt.lower()


def test_extract_figure_prompt_asks_for_unit_separately_not_llm_arithmetic():
    llm = _fake_llm(34550.0, unit="millions")

    extract_figure(llm, ["Research and development 34,550"], "R&D expense", fiscal_year=2025)

    prompt = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert "unit" in prompt.lower()
    assert "exactly as printed" in prompt.lower()


def test_extract_figure_converts_millions_to_raw_usd_deterministically():
    llm = _fake_llm(34550.0, unit="millions")

    value = extract_figure(llm, ["Research and development 34,550"], "R&D expense", fiscal_year=2025)

    assert value == 34550000000.0


def test_extract_figure_converts_billions_to_raw_usd_deterministically():
    llm = _fake_llm(1.5, unit="billions")

    value = extract_figure(llm, ["some text"], "some concept", fiscal_year=2025)

    assert value == 1500000000.0


def test_extract_figure_returns_none_when_not_found():
    llm = _fake_llm(None, unit=None)

    value = extract_figure(llm, ["irrelevant text"], "total net sales", fiscal_year=2025)

    assert value is None


def test_extract_figure_returns_none_when_value_present_but_unit_missing():
    llm = _fake_llm(34550.0, unit=None)

    value = extract_figure(llm, ["some text"], "some concept", fiscal_year=2025)

    assert value is None


# --- run_tie_out_check -----------------------------------------------------------


def test_run_tie_out_check_passes_when_xbrl_confirms_the_figure(mocker):
    check = {"concept": "net_sales", "query": "total net sales", "label": "Net sales tie-out"}
    store = _fake_store([Document(page_content="Total net sales $416,161", metadata={"page": 26})])
    llm = _fake_llm(416161000000.0)
    mocker.patch(
        "src.agent.audit_checklist.verify_against_xbrl",
        return_value={"match": True, "xbrl_value": 416161000000.0, "difference_pct": 0.0, "accession_number": "x"},
    )

    result = run_tie_out_check(check, llm=llm, store=store, fiscal_year=2025)

    assert result["status"] == "PASSED"
    assert result["label"] == "Net sales tie-out"
    assert result["extracted_value"] == 416161000000.0


def test_run_tie_out_check_flags_when_xbrl_disagrees(mocker):
    check = {"concept": "net_sales", "query": "total net sales", "label": "Net sales tie-out"}
    store = _fake_store([Document(page_content="Total net sales $400,000", metadata={"page": 26})])
    llm = _fake_llm(400000000000.0)
    mocker.patch(
        "src.agent.audit_checklist.verify_against_xbrl",
        return_value={"match": False, "xbrl_value": 416161000000.0, "difference_pct": 3.9, "accession_number": "x"},
    )

    result = run_tie_out_check(check, llm=llm, store=store, fiscal_year=2025)

    assert result["status"] == "FLAGGED"


def test_run_tie_out_check_needs_review_when_no_figure_extracted():
    check = {"concept": "net_sales", "query": "total net sales", "label": "Net sales tie-out"}
    store = _fake_store([Document(page_content="unrelated", metadata={"page": 1})])
    llm = _fake_llm(None)

    result = run_tie_out_check(check, llm=llm, store=store, fiscal_year=2025)

    assert result["status"] == "NEEDS_REVIEW"


def test_run_tie_out_check_needs_review_when_xbrl_lookup_raises(mocker):
    check = {"concept": "net_sales", "query": "total net sales", "label": "Net sales tie-out"}
    store = _fake_store([Document(page_content="Total net sales $416,161", metadata={"page": 26})])
    llm = _fake_llm(416161000000.0)
    mocker.patch("src.agent.audit_checklist.verify_against_xbrl", side_effect=ValueError("no data"))

    result = run_tie_out_check(check, llm=llm, store=store, fiscal_year=2025)

    assert result["status"] == "NEEDS_REVIEW"
    assert "no data" in result["detail"]


def test_run_tie_out_check_scopes_search_to_document_id_and_verifies_with_cik(mocker):
    check = {"concept": "net_sales", "query": "total net sales", "label": "Net sales tie-out"}
    store = _fake_store([Document(page_content="Total net sales $211,915", metadata={"page": 30})])
    llm = _fake_llm(211915000000.0)
    mock_verify = mocker.patch(
        "src.agent.audit_checklist.verify_against_xbrl",
        return_value={"match": True, "xbrl_value": 211915000000.0, "difference_pct": 0.0, "accession_number": "x"},
    )

    run_tie_out_check(check, llm=llm, store=store, fiscal_year=2025, document_id="msft_2025", cik="0000789019")

    store.similarity_search.assert_called_once_with("total net sales", document_id="msft_2025")
    mock_verify.assert_called_once_with(
        concept="net_sales", fiscal_year=2025, reported_value=211915000000.0, cik="0000789019"
    )


# --- run_disclosure_check --------------------------------------------------------


def test_run_disclosure_check_runs_compliance_checker_on_top_passage():
    check = {"rule_type": "revenue_recognition", "query": "revenue recognition", "label": "Revenue recognition"}
    passage = (
        "The Company disaggregates revenue by product category. Contract liabilities and "
        "performance obligations are described, along with the transaction price allocated "
        "to remaining performance obligations."
    )
    store = _fake_store([Document(page_content=passage, metadata={"page": 40, "section": "Item 8"})])

    result = run_disclosure_check(check, store=store)

    assert result["label"] == "Revenue recognition"
    assert result["citation"] == "ASC 606-10-50"
    assert result["status"] in {"PASSED", "NEEDS_REVIEW", "FLAGGED"}
    assert result["source_page"] == 40


def test_run_disclosure_check_needs_review_when_no_passages_found():
    check = {"rule_type": "revenue_recognition", "query": "revenue recognition", "label": "Revenue recognition"}
    store = _fake_store([])

    result = run_disclosure_check(check, store=store)

    assert result["status"] == "NEEDS_REVIEW"


def test_run_disclosure_check_scopes_search_to_document_id():
    check = {"rule_type": "revenue_recognition", "query": "revenue recognition", "label": "Revenue recognition"}
    store = _fake_store([Document(page_content="some clause", metadata={"page": 10})])

    run_disclosure_check(check, store=store, document_id="msft_2025")

    store.similarity_search.assert_called_once_with("revenue recognition", document_id="msft_2025")


# --- run_full_audit ---------------------------------------------------------------


def test_run_full_audit_returns_one_row_per_check(mocker):
    store = _fake_store([Document(page_content="Total net sales $416,161", metadata={"page": 26})])
    llm = _fake_llm(416161000000.0)
    mocker.patch(
        "src.agent.audit_checklist.verify_against_xbrl",
        return_value={"match": True, "xbrl_value": 416161000000.0, "difference_pct": 0.0, "accession_number": "x"},
    )

    results = run_full_audit(llm=llm, store=store, fiscal_year=2025)

    assert len(results) > 4
    assert all("label" in r and "status" in r for r in results)
