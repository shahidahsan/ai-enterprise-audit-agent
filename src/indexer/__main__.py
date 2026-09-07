"""CLI entrypoint: index a PDF filing into the vector store.

Usage: uv run python -m src.indexer [path/to/filing.pdf]
Defaults to data/sample_filing.pdf when no path is given. Auto-extracts the company name
and fiscal year from the filing's cover page and resolves its SEC CIK, so the same filing
can be re-indexed alongside others and later selected by name in the UI.
"""
import sys

from src.agent.react_agent import build_llm
from src.indexer.document_indexer import index_filing


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_filing.pdf"
    record = index_filing(pdf_path, llm=build_llm())
    print(f"Indexed {record.chunk_count} chunks from {pdf_path}.")
    print(f"Registered as {record.document_id}: {record.company_name}, FY{record.fiscal_year} (CIK {record.cik}).")


if __name__ == "__main__":
    main()
