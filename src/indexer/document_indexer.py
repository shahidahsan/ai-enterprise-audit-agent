"""End-to-end orchestration: an uploaded PDF becomes a registered, queryable filing.

Ties together cover-page metadata extraction, SEC CIK resolution, document-tagged
chunking, vector storage, and the local document registry -- used by both the
indexer CLI and the Streamlit upload flow.
"""
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from src.indexer.chunker import chunk_documents, load_pdf_pages
from src.indexer.document_registry import DocumentRecord, DocumentRegistry
from src.indexer.metadata_extractor import extract_filing_metadata
from src.indexer.vector_store import AuditVectorStore
from src.tools.sec_lookup import resolve_cik


def index_filing(
    pdf_path: str | Path,
    llm: BaseChatModel,
    store: AuditVectorStore | None = None,
    registry: DocumentRegistry | None = None,
) -> DocumentRecord:
    """Index a PDF filing and register it, resolving company/CIK/fiscal year automatically."""
    store = store or AuditVectorStore()
    registry = registry or DocumentRegistry()

    pages = load_pdf_pages(pdf_path)
    metadata = extract_filing_metadata(llm, pages[0].page_content)
    cik = resolve_cik(metadata.company_name)
    document_id = f"{cik}_{metadata.fiscal_year}"

    chunks = chunk_documents(pages, document_id=document_id)
    ids = store.add_documents(chunks)

    record = DocumentRecord(
        document_id=document_id,
        filename=Path(pdf_path).name,
        company_name=metadata.company_name,
        cik=cik,
        fiscal_year=metadata.fiscal_year,
        chunk_count=len(ids),
    )
    registry.add(record)
    return record
