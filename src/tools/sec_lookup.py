"""Resolves a company name (as extracted from an uploaded filing) to its SEC CIK."""
import difflib

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_USER_AGENT = "compliance-audit-agent research@example.com"
_MATCH_CUTOFF = 0.6


def _fetch_company_titles() -> dict[str, str]:
    response = requests.get(TICKERS_URL, headers={"User-Agent": SEC_USER_AGENT})
    response.raise_for_status()
    entries = response.json().values()
    return {entry["title"]: str(entry["cik_str"]) for entry in entries}


def resolve_cik(company_name: str) -> str:
    """Fuzzy-match a company name against SEC's public ticker directory and return its CIK."""
    titles_to_cik = _fetch_company_titles()
    lowered_to_title = {title.lower(): title for title in titles_to_cik}

    matches = difflib.get_close_matches(company_name.lower(), lowered_to_title.keys(), n=1, cutoff=_MATCH_CUTOFF)
    if not matches:
        raise ValueError(f"No SEC-registered company found matching {company_name!r}")

    cik = titles_to_cik[lowered_to_title[matches[0]]]
    return cik.zfill(10)
