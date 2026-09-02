"""Deterministic compliance disclosure checker, grounded in real cited requirements.

Each rule's required elements are drawn from actual regulatory/accounting-standard
sources (FASB ASC codification, SEC Regulation S-K). Keyword/pattern matching is a
crude proxy for whether a required element is addressed -- this is illustrative
disclosure-checklist tooling, not a certified compliance opinion. The material_weakness
rule's element framing (nature/impact/remediation) reflects SEC guidance and common
practice around Item 308/Item 9A, not a single verbatim statutory sentence.
"""
import re
from typing import TypedDict


class RequiredElement(TypedDict):
    citation: str
    description: str
    keywords: list[str]
    pattern: str | None


class ComplianceRule(TypedDict):
    citation: str
    min_length: int
    required_elements: list[RequiredElement]


def _element(citation: str, description: str, keywords: list[str], pattern: str | None = None) -> RequiredElement:
    return {"citation": citation, "description": description, "keywords": keywords, "pattern": pattern}


RULES: dict[str, ComplianceRule] = {
    "revenue_recognition": {
        "citation": "ASC 606-10-50",
        "min_length": 100,
        "required_elements": [
            _element(
                "ASC 606-10-50-5",
                "Disaggregation of revenue by category (type, geography, market, timing)",
                ["disaggregat", "by type of", "by geograph", "by category", "by segment", "by product"],
            ),
            _element(
                "ASC 606-10-50-8",
                "Contract balances (receivables, contract assets/liabilities, deferred revenue)",
                ["contract asset", "contract liabilit", "deferred revenue", "receivable"],
            ),
            _element(
                "ASC 606-10-50-12",
                "Description of performance obligations",
                ["performance obligation"],
            ),
            _element(
                "ASC 606-10-50-13",
                "Transaction price allocated to remaining performance obligations",
                ["remaining performance obligation", "transaction price allocated"],
            ),
        ],
    },
    "risk_factor": {
        "citation": "17 CFR 229.105 (Item 105)",
        "min_length": 100,
        "required_elements": [
            _element(
                "Item 105(a)",
                "States a material adverse effect, not a generic possibility",
                ["material adverse effect", "materially adversely affect", "materially and adversely affect"],
            ),
            _element(
                "Item 105(a)",
                "Explains the specific mechanism or consequence, not just naming the risk",
                ["could result in", "may result in", "which could", "which may", "as a result"],
            ),
        ],
    },
    "related_party_transaction": {
        "citation": "ASC 850-10-50",
        "min_length": 50,
        "required_elements": [
            _element(
                "ASC 850-10-50-1(a)",
                "Nature of the relationship with the related party",
                ["related part", "affiliate", "subsidiary of", "controlled by", "nature of the relationship"],
            ),
            _element(
                "ASC 850-10-50-1(b)",
                "Description of the transaction(s)",
                ["transaction", "purchased", "sold", "provided services", "lease", "loan"],
            ),
            _element(
                "ASC 850-10-50-1(c)",
                "Dollar amount of the transaction(s) disclosed",
                ["million", "thousand"],
                pattern=r"\$[\d,]+(\.\d+)?",
            ),
            _element(
                "ASC 850-10-50-1(d)",
                "Amounts due to/from the related party and settlement terms",
                ["due from", "due to", "amounts owed", "settlement", "payable to", "receivable from"],
            ),
        ],
    },
    "material_weakness": {
        "citation": "17 CFR 229.308 (Item 308) / SOX Section 404 -- element framing per SEC guidance",
        "min_length": 50,
        "required_elements": [
            _element(
                "SEC guidance (nature)",
                "Describes the nature of the identified material weakness",
                ["material weakness", "deficiency"],
            ),
            _element(
                "SEC guidance (impact)",
                "Describes the impact on financial reporting / ICFR",
                ["reasonable possibility", "misstatement", "impact on", "resulted in"],
            ),
            _element(
                "SEC guidance (remediation)",
                "Describes remediation plan or corrective actions taken",
                ["remediat", "plan to address", "implement", "corrective action"],
            ),
        ],
    },
}


def _element_matched(clause_lower: str, element: RequiredElement) -> bool:
    if any(kw in clause_lower for kw in element["keywords"]):
        return True
    return bool(element["pattern"] and re.search(element["pattern"], clause_lower, re.IGNORECASE))


def compliance_flag_checker(clause_text: str, rule_type: str) -> dict:
    """Check extracted clause text against a required-element checklist for one disclosure rule."""
    if not clause_text.strip():
        raise ValueError("clause_text must not be blank")
    if rule_type not in RULES:
        raise ValueError(f"unknown rule_type: {rule_type!r}. Valid types: {sorted(RULES)}")

    rule = RULES[rule_type]
    lowered = clause_text.lower()
    elements = [
        {"citation": el["citation"], "description": el["description"], "matched": _element_matched(lowered, el)}
        for el in rule["required_elements"]
    ]
    coverage = sum(el["matched"] for el in elements) / len(elements)
    length_ok = len(clause_text.strip()) >= rule["min_length"]

    if coverage == 1.0 and length_ok:
        status = "PASSED"
    elif coverage > 0:
        status = "NEEDS_REVIEW"
    else:
        status = "FLAGGED"

    return {
        "rule_type": rule_type,
        "citation": rule["citation"],
        "status": status,
        "coverage": round(coverage, 4),
        "elements": elements,
        "clause_length": len(clause_text.strip()),
    }
