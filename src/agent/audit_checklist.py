"""'Run Full Audit' checklist: a fixed set of tie-out and disclosure checks over the filing.

Unlike the free-form chat (which answers whatever the user asks), this runs a
deterministic checklist end-to-end and produces one summary table -- the audit-report
shape a real internal-audit workflow expects, rather than open-ended Q&A.
"""
from langchain_core.language_models import BaseChatModel

from src.agent.schemas import ExtractedFigure
from src.indexer.vector_store import AuditVectorStore
from src.tools.compliance_tool import compliance_flag_checker
from src.tools.xbrl_tool import verify_against_xbrl

# All four figures live on the same Consolidated Statements of Operations page. A query naming
# only the target concept (e.g. "research and development expense") tends to drift toward other
# tables that repeat the same line-item label (e.g. a segment-reconciliation table) -- anchoring
# every query on the statement's own title reliably retrieves the correctly-labeled table instead.
_STATEMENT_QUERY = (
    "Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS Years ended September 27 2025 net sales "
    "cost of sales gross margin operating expenses research and development selling general "
    "administrative operating income"
)

FINANCIAL_CHECKS = [
    {"concept": "net_sales", "query": _STATEMENT_QUERY, "label": "Net sales tie-out"},
    {"concept": "operating_income", "query": _STATEMENT_QUERY, "label": "Operating income tie-out"},
    {"concept": "gross_margin", "query": _STATEMENT_QUERY, "label": "Gross margin tie-out"},
    {"concept": "research_development_expense", "query": _STATEMENT_QUERY, "label": "R&D expense tie-out"},
]

DISCLOSURE_CHECKS = [
    {
        "rule_type": "revenue_recognition",
        "query": "revenue recognition performance obligations disclosure",
        "label": "Revenue recognition disclosure",
    },
    {
        "rule_type": "risk_factor",
        "query": "risk factors that could materially and adversely affect our business",
        "label": "Risk factor disclosure",
    },
    {
        "rule_type": "related_party_transaction",
        "query": "related party transactions",
        "label": "Related party transaction disclosure",
    },
    {
        "rule_type": "material_weakness",
        "query": "material weakness internal control over financial reporting",
        "label": "Material weakness disclosure",
    },
]


_UNIT_MULTIPLIERS = {"raw": 1, "thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}


def extract_figure(
    llm: BaseChatModel, passages: list[str], concept_description: str, fiscal_year: int
) -> float | None:
    """Ask the LLM to pull a single figure out of retrieved passages, converting units in code.

    10-K tables routinely show 3 fiscal years side by side, so the prompt must explicitly name
    the target year -- without it the model can silently grab the adjacent (prior-year) column,
    which is exactly the failure mode this tie-out tool exists to catch. The LLM is asked for the
    number exactly as printed plus its unit, never the converted raw-dollar value: an earlier
    version asked the LLM to do the x1,000,000 conversion itself, which fixed one failure only to
    introduce a different one elsewhere (over-multiplying a different figure by 1000x on a later
    run) -- the same reason calculate_variance exists, arithmetic belongs in code, not a prompt.
    """
    prompt = (
        f"The text below may show multiple fiscal years side by side in a comparison table. "
        f"Find the {concept_description} figure for fiscal year {fiscal_year} specifically -- "
        f"carefully match the correct column/year label, do not use an adjacent year's figure. "
        f"Extract the number exactly as printed in the table (do not convert units yourself) "
        f"and separately state its unit based on the table's header, e.g. '(In millions...)' "
        f"-> 'millions'; no stated multiplier -> 'raw'. Return null for both if fiscal year "
        f"{fiscal_year}'s figure is not present.\n\n" + "\n\n".join(passages)
    )
    result: ExtractedFigure = llm.with_structured_output(ExtractedFigure).invoke(prompt)
    if result.value is None or result.unit is None:
        return None
    return result.value * _UNIT_MULTIPLIERS[result.unit]


def run_tie_out_check(check: dict, llm: BaseChatModel, store: AuditVectorStore, fiscal_year: int) -> dict:
    """Extract a figure from retrieved passages and tie it out against SEC XBRL data."""
    docs = store.similarity_search(check["query"])
    passages = [doc.page_content for doc in docs]
    extracted_value = extract_figure(llm, passages, check["label"], fiscal_year)

    if extracted_value is None:
        return {**check, "status": "NEEDS_REVIEW", "extracted_value": None, "detail": "No figure extracted"}

    try:
        tie_out = verify_against_xbrl(concept=check["concept"], fiscal_year=fiscal_year, reported_value=extracted_value)
    except ValueError as exc:
        return {**check, "status": "NEEDS_REVIEW", "extracted_value": extracted_value, "detail": str(exc)}

    status = "PASSED" if tie_out["match"] else "FLAGGED"
    return {**check, "status": status, "extracted_value": extracted_value, "detail": tie_out}


def run_disclosure_check(check: dict, store: AuditVectorStore) -> dict:
    """Retrieve the most relevant passage and run it through the compliance checklist."""
    docs = store.similarity_search(check["query"])
    if not docs:
        return {**check, "status": "NEEDS_REVIEW", "citation": None, "source_page": None, "detail": "No passages found"}

    top_doc = docs[0]
    result = compliance_flag_checker(top_doc.page_content, check["rule_type"])
    return {
        **check,
        "status": result["status"],
        "citation": result["citation"],
        "source_page": top_doc.metadata.get("page"),
        "detail": result,
    }


def run_full_audit(
    llm: BaseChatModel | None = None, store: AuditVectorStore | None = None, fiscal_year: int = 2025
) -> list[dict]:
    """Run every tie-out and disclosure check and return one result row per check."""
    from src.agent.react_agent import build_llm

    llm = llm or build_llm()
    store = store or AuditVectorStore()

    results = [run_tie_out_check(check, llm=llm, store=store, fiscal_year=fiscal_year) for check in FINANCIAL_CHECKS]
    results += [run_disclosure_check(check, store=store) for check in DISCLOSURE_CHECKS]
    return results
