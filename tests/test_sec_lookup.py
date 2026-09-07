"""Tests for resolving a company name to its SEC CIK via SEC's public ticker directory."""
from unittest.mock import MagicMock

import pytest

from src.tools.sec_lookup import resolve_cik

FAKE_TICKERS_JSON = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "2": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}


def _mock_get(mocker):
    response = MagicMock()
    response.json.return_value = FAKE_TICKERS_JSON
    return mocker.patch("src.tools.sec_lookup.requests.get", return_value=response)


def test_resolve_cik_finds_exact_case_insensitive_match(mocker):
    _mock_get(mocker)

    cik = resolve_cik("apple inc.")

    assert cik == "0000320193"


def test_resolve_cik_finds_close_fuzzy_match(mocker):
    _mock_get(mocker)

    cik = resolve_cik("Apple")

    assert cik == "0000320193"


def test_resolve_cik_pads_to_ten_digits(mocker):
    _mock_get(mocker)

    cik = resolve_cik("NVIDIA CORP")

    assert cik == "0001045810"


def test_resolve_cik_raises_when_no_reasonable_match(mocker):
    _mock_get(mocker)

    with pytest.raises(ValueError):
        resolve_cik("Definitely Not A Real Company Name XYZ")


def test_resolve_cik_sends_required_sec_user_agent_header(mocker):
    mock_get = _mock_get(mocker)

    resolve_cik("Apple Inc.")

    _, kwargs = mock_get.call_args
    assert "User-Agent" in kwargs["headers"]
