"""Tests for PDF loading and chunking."""
from pathlib import Path

from langchain_core.documents import Document

from src.indexer.chunker import chunk_documents, load_and_chunk_pdf, load_pdf_pages

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "sample_filing.pdf"


def test_load_pdf_pages_returns_one_document_per_page():
    pages = load_pdf_pages(SAMPLE_PDF)

    assert len(pages) > 1
    assert all(isinstance(page, Document) for page in pages)
    assert pages[0].metadata["page"] == 1
    assert pages[-1].metadata["page"] == len(pages)


def test_load_pdf_pages_tags_source_path():
    pages = load_pdf_pages(SAMPLE_PDF)

    assert all(page.metadata["source"] == str(SAMPLE_PDF) for page in pages)


def test_load_pdf_pages_detects_item_section_headers():
    pages = load_pdf_pages(SAMPLE_PDF)

    sections = {page.metadata["section"] for page in pages}
    assert any(section != "Unknown" for section in sections)


def test_chunk_documents_splits_long_text_into_multiple_chunks():
    long_text = "Apple reported strong revenue growth this fiscal year. " * 500
    documents = [Document(page_content=long_text, metadata={"page": 1, "section": "Item 7"})]

    chunks = chunk_documents(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) > 0 for chunk in chunks)


def test_chunk_documents_preserves_source_metadata():
    documents = [Document(page_content="Short filing text.", metadata={"page": 3, "section": "Item 1"})]

    chunks = chunk_documents(documents)

    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["section"] == "Item 1"


def test_chunk_documents_assigns_sequential_chunk_index():
    long_text = "Risk factors discussion regarding supply chain exposure. " * 500
    documents = [Document(page_content=long_text, metadata={"page": 1, "section": "Item 1A"})]

    chunks = chunk_documents(documents)

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_chunk_documents_tags_document_id_when_given():
    documents = [Document(page_content="Short filing text.", metadata={"page": 3, "section": "Item 1"})]

    chunks = chunk_documents(documents, document_id="apple_2025")

    assert chunks[0].metadata["document_id"] == "apple_2025"


def test_chunk_documents_omits_document_id_when_not_given():
    documents = [Document(page_content="Short filing text.", metadata={"page": 3, "section": "Item 1"})]

    chunks = chunk_documents(documents)

    assert "document_id" not in chunks[0].metadata


def test_chunk_documents_respects_configured_chunk_size_in_tokens():
    long_text = "Material weakness in internal controls over financial reporting. " * 500
    documents = [Document(page_content=long_text, metadata={"page": 1, "section": "Item 9A"})]

    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1


def test_load_and_chunk_pdf_end_to_end_on_sample_filing():
    chunks = load_and_chunk_pdf(SAMPLE_PDF)

    assert len(chunks) > 10
    assert all("page" in chunk.metadata for chunk in chunks)
    assert all("section" in chunk.metadata for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_load_and_chunk_pdf_tags_document_id_when_given():
    chunks = load_and_chunk_pdf(SAMPLE_PDF, document_id="apple_2025")

    assert all(chunk.metadata["document_id"] == "apple_2025" for chunk in chunks)
