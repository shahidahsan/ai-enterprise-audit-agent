"""Deterministic document search tool backed by the pgvector store."""
from src.indexer.vector_store import AuditVectorStore


def document_search(query: str, store: AuditVectorStore | None = None) -> str:
    """Search indexed filing chunks and return top passages with source citations."""
    if not query.strip():
        raise ValueError("query must not be blank")
    store = store or AuditVectorStore()
    results = store.similarity_search(query)
    if not results:
        return "No relevant passages found."

    passages = []
    for doc in results:
        page = doc.metadata.get("page", "unknown")
        section = doc.metadata.get("section", "Unknown")
        passages.append(f"[Source: page {page}, {section}]\n{doc.page_content.strip()}")
    return "\n\n".join(passages)
