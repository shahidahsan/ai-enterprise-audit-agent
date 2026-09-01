"""CLI entrypoint: index a PDF filing into the vector store.

Usage: uv run python -m src.indexer [path/to/filing.pdf]
Defaults to data/sample_filing.pdf when no path is given.
"""
import sys

from src.indexer.chunker import load_and_chunk_pdf
from src.indexer.vector_store import AuditVectorStore


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_filing.pdf"
    chunks = load_and_chunk_pdf(pdf_path)
    ids = AuditVectorStore().add_documents(chunks)
    print(f"Indexed {len(ids)} chunks from {pdf_path}.")


if __name__ == "__main__":
    main()
