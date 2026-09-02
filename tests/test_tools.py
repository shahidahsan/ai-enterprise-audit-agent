"""Tests for the agent's deterministic tools."""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.tools.calculator_tool import calculate_variance
from src.tools.compliance_tool import compliance_flag_checker
from src.tools.search_tool import document_search


def _fake_store(results):
    store = MagicMock()
    store.similarity_search.return_value = results
    return store


# --- document_search ---------------------------------------------------------

def test_document_search_formats_passages_with_citations():
    results = [
        Document(page_content="Net sales increased 6% year over year.", metadata={"page": 26, "section": "Item 7"}),
        Document(page_content="Gross margin was 46.9%.", metadata={"page": 27, "section": "Item 7"}),
    ]
    store = _fake_store(results)

    output = document_search("What was net sales growth?", store=store)

    assert "page 26" in output
    assert "Item 7" in output
    assert "Net sales increased 6%" in output
    assert "Gross margin was 46.9%" in output


def test_document_search_returns_message_when_no_results():
    store = _fake_store([])

    output = document_search("something irrelevant", store=store)

    assert output == "No relevant passages found."


def test_document_search_rejects_blank_query():
    store = _fake_store([])

    with pytest.raises(ValueError):
        document_search("   ", store=store)


def test_document_search_delegates_query_to_store():
    store = _fake_store([])

    document_search("iPhone revenue trend", store=store)

    store.similarity_search.assert_called_once_with("iPhone revenue trend")


# --- calculate_variance -------------------------------------------------------

def test_calculate_variance_computes_increase():
    result = calculate_variance(current_val=110.0, prior_val=100.0)

    assert result["absolute_change"] == 10.0
    assert result["percentage_variance"] == 10.0
    assert result["trend"] == "increase"


def test_calculate_variance_computes_decrease():
    result = calculate_variance(current_val=80.0, prior_val=100.0)

    assert result["absolute_change"] == -20.0
    assert result["percentage_variance"] == -20.0
    assert result["trend"] == "decrease"


def test_calculate_variance_computes_no_change():
    result = calculate_variance(current_val=50.0, prior_val=50.0)

    assert result["absolute_change"] == 0.0
    assert result["percentage_variance"] == 0.0
    assert result["trend"] == "no_change"


def test_calculate_variance_rejects_zero_prior_value():
    with pytest.raises(ValueError):
        calculate_variance(current_val=10.0, prior_val=0.0)


def test_calculate_variance_rejects_non_numeric_input():
    with pytest.raises(TypeError):
        calculate_variance(current_val="ten", prior_val=5.0)


def test_calculate_variance_rejects_bool_input():
    with pytest.raises(TypeError):
        calculate_variance(current_val=True, prior_val=5.0)


# --- compliance_flag_checker ---------------------------------------------------

def test_compliance_flag_checker_passes_when_all_required_elements_present():
    clause = (
        "The Company disaggregates revenue by product category and disaggregates revenue further "
        "by geographic segment. Contract liabilities and deferred revenue are recognized as "
        "performance obligations are satisfied. The transaction price allocated to remaining "
        "performance obligations as of period end is disclosed below."
    )

    result = compliance_flag_checker(clause, "revenue_recognition")

    assert result["status"] == "PASSED"
    assert result["citation"] == "ASC 606-10-50"
    assert all(el["matched"] for el in result["elements"])
    assert result["coverage"] == 1.0


def test_compliance_flag_checker_needs_review_when_some_elements_missing():
    clause = (
        "The Company satisfies its performance obligations over time as services are delivered "
        "to customers, consistent with the terms of each arrangement entered into during the year."
    )

    result = compliance_flag_checker(clause, "revenue_recognition")

    assert result["status"] == "NEEDS_REVIEW"
    matched = {el["citation"] for el in result["elements"] if el["matched"]}
    assert "ASC 606-10-50-12" in matched
    assert 0 < result["coverage"] < 1.0


def test_compliance_flag_checker_flags_when_no_elements_present():
    clause = "The weather in Cupertino was mild this quarter, with clear skies most days."

    result = compliance_flag_checker(clause, "revenue_recognition")

    assert result["status"] == "FLAGGED"
    assert result["coverage"] == 0.0
    assert all(not el["matched"] for el in result["elements"])


def test_compliance_flag_checker_needs_review_when_too_short_despite_element_match():
    clause = "Performance obligations exist."

    result = compliance_flag_checker(clause, "revenue_recognition")

    assert result["status"] == "NEEDS_REVIEW"


def test_compliance_flag_checker_detects_dollar_amount_via_pattern_for_related_party():
    clause = (
        "The Company purchased components from an affiliate during the year for $12.4 million, "
        "with $3.1 million due to the affiliate as of period end under normal settlement terms."
    )

    result = compliance_flag_checker(clause, "related_party_transaction")

    matched = {el["citation"]: el["matched"] for el in result["elements"]}
    assert matched["ASC 850-10-50-1(c)"] is True


def test_compliance_flag_checker_includes_real_regulatory_citations():
    result = compliance_flag_checker(
        "Material weakness identified; remediation plan to implement additional controls is underway.",
        "material_weakness",
    )

    assert "308" in result["citation"] or "SOX" in result["citation"]
    for element in result["elements"]:
        assert element["citation"]
        assert element["description"]


def test_compliance_flag_checker_rejects_unknown_rule_type():
    with pytest.raises(ValueError):
        compliance_flag_checker("some clause", "not_a_real_rule")


def test_compliance_flag_checker_rejects_blank_clause_text():
    with pytest.raises(ValueError):
        compliance_flag_checker("   ", "revenue_recognition")
