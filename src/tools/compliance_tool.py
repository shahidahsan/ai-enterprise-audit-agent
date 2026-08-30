"""Deterministic compliance rule checker tool."""
from typing import TypedDict


class ComplianceRule(TypedDict):
    keywords: list[str]
    min_length: int


RULES: dict[str, ComplianceRule] = {
    "revenue_recognition": {
        "keywords": ["revenue recognition", "asc 606", "performance obligation"],
        "min_length": 100,
    },
    "risk_factor": {
        "keywords": ["risk factor", "may adversely affect", "could harm"],
        "min_length": 100,
    },
    "related_party_transaction": {
        "keywords": ["related party", "affiliate"],
        "min_length": 50,
    },
    "material_weakness": {
        "keywords": ["material weakness", "internal control over financial reporting"],
        "min_length": 50,
    },
}


def compliance_flag_checker(clause_text: str, rule_type: str) -> dict:
    """Check extracted clause text against a predefined compliance disclosure rule."""
    if not clause_text.strip():
        raise ValueError("clause_text must not be blank")
    if rule_type not in RULES:
        raise ValueError(f"unknown rule_type: {rule_type!r}. Valid types: {sorted(RULES)}")

    rule = RULES[rule_type]
    lowered = clause_text.lower()
    matched_keywords = [kw for kw in rule["keywords"] if kw in lowered]
    length_ok = len(clause_text.strip()) >= rule["min_length"]

    if matched_keywords and length_ok:
        status = "PASSED"
    elif matched_keywords:
        status = "NEEDS_REVIEW"
    else:
        status = "FLAGGED"

    return {
        "rule_type": rule_type,
        "status": status,
        "matched_keywords": matched_keywords,
        "clause_length": len(clause_text.strip()),
    }
