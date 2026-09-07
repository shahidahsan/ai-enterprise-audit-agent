"""Tie-out tool: cross-checks LLM-extracted figures against SEC's structured XBRL data.

"Tie-out" is a real audit technique -- confirming a number stated in a filing's narrative
(e.g. the MD&A) matches the authoritative, structured source. The SEC requires every
reported figure to also be filed as a machine-readable XBRL "fact"; this tool queries
that structured data directly rather than trusting the LLM's read of PDF prose.
"""
import requests

# Apple's SEC CIK (Central Index Key), zero-padded to 10 digits per SEC's API convention.
APPLE_CIK = "0000320193"
SEC_USER_AGENT = "compliance-audit-agent research@example.com"

# Common audit concepts mapped to candidate us-gaap XBRL tags, tried in order. Apple has used
# different tags for the same concept across years (e.g. after adopting ASC 606).
CONCEPT_ALIASES: dict[str, list[str]] = {
    "net_sales": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_margin": ["GrossProfit"],
    "net_income": ["NetIncomeLoss"],
    "research_development_expense": ["ResearchAndDevelopmentExpense"],
}


def _fetch_concept_facts(tag: str, cik: str) -> dict | None:
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    response = requests.get(url, headers={"User-Agent": SEC_USER_AGENT})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _select_full_year_fact(facts: dict, fiscal_year: int) -> dict | None:
    candidates = [
        fact
        for fact in facts.get("units", {}).get("USD", [])
        if fact.get("fy") == fiscal_year and fact.get("fp") == "FY" and "10-K" in fact.get("form", "")
    ]
    return candidates[-1] if candidates else None


def verify_against_xbrl(
    concept: str, fiscal_year: int, reported_value: float, cik: str = APPLE_CIK
) -> dict:
    """Cross-check a reported figure against SEC's structured XBRL company-facts data."""
    tags_to_try = CONCEPT_ALIASES.get(concept, [concept])

    for tag in tags_to_try:
        facts = _fetch_concept_facts(tag, cik)
        if facts is None:
            continue
        fact = _select_full_year_fact(facts, fiscal_year)
        if fact is None:
            # This tag exists for the company but has no data for the requested year -- a
            # company can switch which XBRL tag it reports a concept under between filings
            # (found via NVIDIA: its FY2025 revenue moved from the first alias to "Revenues").
            # Try the next alias rather than failing on the first tag that merely exists.
            continue

        xbrl_value = fact["val"]
        difference_pct = round(abs(reported_value - xbrl_value) / xbrl_value * 100, 4)
        return {
            "concept": concept,
            "xbrl_tag": tag,
            "fiscal_year": fiscal_year,
            "reported_value": reported_value,
            "xbrl_value": xbrl_value,
            "difference_pct": difference_pct,
            "match": difference_pct < 0.01,
            "accession_number": fact["accn"],
            "source": "SEC EDGAR XBRL company facts API",
        }

    raise ValueError(
        f"No fiscal year {fiscal_year} 10-K figure found for concept {concept!r} "
        f"(tried tags: {tags_to_try})"
    )
