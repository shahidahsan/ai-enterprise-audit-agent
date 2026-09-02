"""Tests for the SEC XBRL tie-out tool (cross-checking figures against structured filing data)."""
from unittest.mock import MagicMock

import pytest
import requests

from src.tools.xbrl_tool import verify_against_xbrl

FAKE_RESPONSE_BODY = {
    "cik": 320193,
    "taxonomy": "us-gaap",
    "tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "entityName": "Apple Inc.",
    "units": {
        "USD": [
            {"fy": 2024, "fp": "FY", "form": "10-K", "val": 391035000000, "accn": "old-accn"},
            {"fy": 2025, "fp": "Q1", "form": "10-Q", "val": 100000000000, "accn": "q-accn"},
            {"fy": 2025, "fp": "FY", "form": "10-K", "val": 416161000000, "accn": "new-accn"},
        ]
    },
}


def _mock_get(mocker, json_body=None, status_code=200, raise_for_status_error=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body or {}
    if raise_for_status_error:
        response.raise_for_status.side_effect = raise_for_status_error
    return mocker.patch("src.tools.xbrl_tool.requests.get", return_value=response)


def test_verify_against_xbrl_matches_when_value_agrees(mocker):
    _mock_get(mocker, FAKE_RESPONSE_BODY)

    result = verify_against_xbrl(concept="net_sales", fiscal_year=2025, reported_value=416161000000)

    assert result["match"] is True
    assert result["xbrl_value"] == 416161000000
    assert result["difference_pct"] == 0.0
    assert result["accession_number"] == "new-accn"


def test_verify_against_xbrl_flags_mismatch_when_value_disagrees(mocker):
    _mock_get(mocker, FAKE_RESPONSE_BODY)

    result = verify_against_xbrl(concept="net_sales", fiscal_year=2025, reported_value=400000000000)

    assert result["match"] is False
    assert result["difference_pct"] > 0


def test_verify_against_xbrl_selects_full_year_figure_not_quarterly(mocker):
    _mock_get(mocker, FAKE_RESPONSE_BODY)

    result = verify_against_xbrl(concept="net_sales", fiscal_year=2025, reported_value=416161000000)

    assert result["xbrl_value"] == 416161000000


def test_verify_against_xbrl_tries_alias_tags_until_one_resolves(mocker):
    mock_get = mocker.patch("src.tools.xbrl_tool.requests.get")
    not_found = MagicMock(status_code=404)
    not_found.raise_for_status.side_effect = requests.HTTPError("404")
    found = MagicMock(status_code=200)
    found.json.return_value = FAKE_RESPONSE_BODY
    mock_get.side_effect = [not_found, found]

    result = verify_against_xbrl(concept="net_sales", fiscal_year=2025, reported_value=416161000000)

    assert result["match"] is True
    assert mock_get.call_count == 2


def test_verify_against_xbrl_raises_for_unknown_concept(mocker):
    mock_get = mocker.patch("src.tools.xbrl_tool.requests.get")
    not_found = MagicMock(status_code=404)
    not_found.raise_for_status.side_effect = requests.HTTPError("404")
    mock_get.return_value = not_found

    with pytest.raises(ValueError):
        verify_against_xbrl(concept="not_a_real_concept", fiscal_year=2025, reported_value=1.0)


def test_verify_against_xbrl_raises_when_fiscal_year_has_no_full_year_figure(mocker):
    _mock_get(mocker, FAKE_RESPONSE_BODY)

    with pytest.raises(ValueError):
        verify_against_xbrl(concept="net_sales", fiscal_year=1999, reported_value=1.0)


def test_verify_against_xbrl_sends_required_sec_user_agent_header(mocker):
    mock_get = _mock_get(mocker, FAKE_RESPONSE_BODY)

    verify_against_xbrl(concept="net_sales", fiscal_year=2025, reported_value=416161000000)

    _, kwargs = mock_get.call_args
    assert "User-Agent" in kwargs["headers"]
