"""Tests for the end-to-end document indexing orchestration (upload -> registered filing)."""
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.schemas import FilingMetadata
from src.indexer.document_indexer import index_filing
from src.indexer.document_registry import DocumentRegistry

SAMPLE_PDF = Path(__file__).parent.parent / "data" / "sample_filing.pdf"


def _fake_llm():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = FilingMetadata(
        company_name="Apple Inc.", fiscal_year=2025
    )
    return llm


def test_index_filing_registers_a_document_record(tmp_path, mocker):
    mocker.patch("src.indexer.document_indexer.resolve_cik", return_value="0000320193")
    fake_store = MagicMock()
    fake_store.add_documents.return_value = ["id1"] * 10
    registry = DocumentRegistry(path=tmp_path / "documents.json")

    record = index_filing(SAMPLE_PDF, llm=_fake_llm(), store=fake_store, registry=registry)

    assert record.company_name == "Apple Inc."
    assert record.cik == "0000320193"
    assert record.fiscal_year == 2025
    assert record.chunk_count == 10
    assert registry.get(record.document_id) == record


def test_index_filing_tags_stored_chunks_with_the_document_id(tmp_path, mocker):
    mocker.patch("src.indexer.document_indexer.resolve_cik", return_value="0000320193")
    fake_store = MagicMock()
    fake_store.add_documents.return_value = ["id1"]
    registry = DocumentRegistry(path=tmp_path / "documents.json")

    record = index_filing(SAMPLE_PDF, llm=_fake_llm(), store=fake_store, registry=registry)

    stored_chunks = fake_store.add_documents.call_args[0][0]
    assert all(chunk.metadata["document_id"] == record.document_id for chunk in stored_chunks)


def test_index_filing_derives_document_id_from_cik_and_fiscal_year(tmp_path, mocker):
    mocker.patch("src.indexer.document_indexer.resolve_cik", return_value="0000320193")
    fake_store = MagicMock()
    fake_store.add_documents.return_value = ["id1"]
    registry = DocumentRegistry(path=tmp_path / "documents.json")

    record = index_filing(SAMPLE_PDF, llm=_fake_llm(), store=fake_store, registry=registry)

    assert record.document_id == "0000320193_2025"
