"""Extracts company name and fiscal year from an uploaded filing's cover page text."""
from langchain_core.language_models import BaseChatModel

from src.agent.schemas import FilingMetadata


def extract_filing_metadata(llm: BaseChatModel, cover_page_text: str) -> FilingMetadata:
    """Ask the LLM to read the filer's legal name and fiscal year off the cover page."""
    prompt = (
        "This is the cover page of an SEC Form 10-K filing. Extract the filer's legal company "
        "name (as it appears on the filing, e.g. 'Apple Inc.') and the fiscal year this filing "
        "covers (as a 4-digit integer, e.g. 2025).\n\n" + cover_page_text
    )
    return llm.with_structured_output(FilingMetadata).invoke(prompt)
