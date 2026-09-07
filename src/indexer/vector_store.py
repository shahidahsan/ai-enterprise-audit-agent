"""Pgvector-backed vector store for indexed filing chunks."""
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from src.config import get_settings

COLLECTION_NAME = "audit_filings"


def build_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


class AuditVectorStore:
    """Thin wrapper around PGVector for indexing and retrieving filing chunks."""

    def __init__(self, connection: str | None = None) -> None:
        settings = get_settings()
        self._store = PGVector(
            embeddings=build_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=connection or settings.postgres_dsn,
            use_jsonb=True,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int | None = None, document_id: str | None = None) -> list[Document]:
        if not query.strip():
            raise ValueError("query must not be blank")
        settings = get_settings()
        doc_filter = {"document_id": document_id} if document_id else None
        return self._store.similarity_search(query, k=k or settings.retrieval_top_k, filter=doc_filter)
