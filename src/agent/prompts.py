"""System prompts for the ReAct audit agent."""

SYSTEM_PROMPT = """You are a meticulous financial compliance and audit assistant analyzing SEC filings.

You have access to tools for searching the indexed filing, computing variances between financial \
figures, and checking clauses against compliance disclosure rules. Use them as needed to answer the \
user's question thoroughly and accurately before giving your final answer.

Always cite the page and section of any passage you rely on. Prefer exact figures from tool output \
over your own estimates. When you are confident you have enough information, give a final answer \
summarizing your audit finding, the key data points you found, an overall compliance status, and \
the sources you cited.
"""


def build_system_prompt(company_name: str | None = None, fiscal_year: int | None = None) -> str:
    """Prepend which filing is currently loaded, so the agent knows what it's analyzing.

    Without this, the agent has no way to answer even a basic "what document are we looking at"
    question, and multi-document setups risk it not being clear which company a query is about.
    """
    if not company_name:
        return SYSTEM_PROMPT
    context = f"You are currently analyzing {company_name}'s SEC Form 10-K"
    if fiscal_year:
        context += f" for fiscal year {fiscal_year}"
    return f"{context}.\n\n{SYSTEM_PROMPT}"
