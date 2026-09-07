"""Tests for the local registry tracking which filings have been indexed."""
import json

from src.indexer.document_registry import DocumentRecord, DocumentRegistry


def test_add_and_list_returns_the_record(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    record = DocumentRecord(
        document_id="apple_2025",
        filename="sample_filing.pdf",
        company_name="Apple Inc.",
        cik="0000320193",
        fiscal_year=2025,
        chunk_count=124,
    )

    registry.add(record)

    assert registry.list_documents() == [record]


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "documents.json"
    record = DocumentRecord(
        document_id="apple_2025",
        filename="sample_filing.pdf",
        company_name="Apple Inc.",
        cik="0000320193",
        fiscal_year=2025,
        chunk_count=124,
    )
    DocumentRegistry(path=path).add(record)

    reloaded = DocumentRegistry(path=path)

    assert reloaded.list_documents() == [record]


def test_list_documents_returns_empty_list_when_file_missing(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "does_not_exist.json")

    assert registry.list_documents() == []


def test_get_returns_matching_record_by_document_id(tmp_path):
    registry = DocumentRegistry(path=tmp_path / "documents.json")
    record = DocumentRecord(
        document_id="apple_2025",
        filename="sample_filing.pdf",
        company_name="Apple Inc.",
        cik="0000320193",
        fiscal_year=2025,
        chunk_count=124,
    )
    registry.add(record)

    assert registry.get("apple_2025") == record
    assert registry.get("not_a_real_id") is None


def test_add_writes_valid_json_to_disk(tmp_path):
    path = tmp_path / "documents.json"
    registry = DocumentRegistry(path=path)
    record = DocumentRecord(
        document_id="apple_2025",
        filename="sample_filing.pdf",
        company_name="Apple Inc.",
        cik="0000320193",
        fiscal_year=2025,
        chunk_count=124,
    )

    registry.add(record)

    on_disk = json.loads(path.read_text())
    assert on_disk[0]["document_id"] == "apple_2025"
