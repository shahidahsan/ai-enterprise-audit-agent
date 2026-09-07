"""Tests for the pgvector-backed vector store, with the underlying PGVector client mocked."""
import pytest
from langchain_core.documents import Document

from src.indexer.vector_store import AuditVectorStore


@pytest.fixture
def mock_pgvector_cls(mocker):
    return mocker.patch("src.indexer.vector_store.PGVector")


@pytest.fixture
def mock_embeddings(mocker):
    return mocker.patch("src.indexer.vector_store.build_embeddings", return_value=object())


def test_init_wires_embeddings_collection_and_connection(mock_pgvector_cls, mock_embeddings):
    AuditVectorStore()

    _, kwargs = mock_pgvector_cls.call_args
    assert kwargs["embeddings"] is mock_embeddings.return_value
    assert kwargs["collection_name"] == "audit_filings"
    assert "postgresql+psycopg://" in kwargs["connection"]


def test_add_documents_delegates_to_underlying_store(mock_pgvector_cls, mock_embeddings):
    mock_instance = mock_pgvector_cls.return_value
    mock_instance.add_documents.return_value = ["id-1", "id-2"]
    documents = [Document(page_content="Revenue grew 6% YoY.", metadata={"page": 12})]

    store = AuditVectorStore()
    ids = store.add_documents(documents)

    mock_instance.add_documents.assert_called_once_with(documents)
    assert ids == ["id-1", "id-2"]


def test_similarity_search_delegates_with_default_top_k(mock_pgvector_cls, mock_embeddings):
    mock_instance = mock_pgvector_cls.return_value
    expected = [Document(page_content="Gross margin was 46.9%.", metadata={"page": 20})]
    mock_instance.similarity_search.return_value = expected

    store = AuditVectorStore()
    results = store.similarity_search("What was gross margin?")

    mock_instance.similarity_search.assert_called_once_with("What was gross margin?", k=5, filter=None)
    assert results == expected


def test_similarity_search_respects_explicit_top_k(mock_pgvector_cls, mock_embeddings):
    mock_instance = mock_pgvector_cls.return_value
    mock_instance.similarity_search.return_value = []

    store = AuditVectorStore()
    store.similarity_search("iPhone revenue", k=2)

    mock_instance.similarity_search.assert_called_once_with("iPhone revenue", k=2, filter=None)


def test_similarity_search_filters_by_document_id_when_given(mock_pgvector_cls, mock_embeddings):
    mock_instance = mock_pgvector_cls.return_value
    mock_instance.similarity_search.return_value = []

    store = AuditVectorStore()
    store.similarity_search("iPhone revenue", document_id="apple_2025")

    mock_instance.similarity_search.assert_called_once_with(
        "iPhone revenue", k=5, filter={"document_id": "apple_2025"}
    )


def test_similarity_search_rejects_blank_query(mock_pgvector_cls, mock_embeddings):
    store = AuditVectorStore()

    with pytest.raises(ValueError):
        store.similarity_search("   ")
