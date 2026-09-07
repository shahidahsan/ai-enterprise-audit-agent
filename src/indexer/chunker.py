"""PDF loading and chunking for the audit filing indexer."""
import re
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import get_settings

SECTION_PATTERN = re.compile(r"(Item\s+\d+[A-Za-z]?\.?\s*[^\n]{0,80})", re.IGNORECASE)
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _detect_section(text: str, current_section: str) -> str:
    match = SECTION_PATTERN.search(text)
    return match.group(1).strip() if match else current_section


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Load a PDF into one Document per page, tagged with page number and section."""
    reader = PdfReader(str(pdf_path))
    pages: list[Document] = []
    current_section = "Unknown"
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        current_section = _detect_section(text, current_section)
        metadata = {"page": page_number, "section": current_section, "source": str(pdf_path)}
        pages.append(Document(page_content=text, metadata=metadata))
    return pages


def build_text_splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=_count_tokens,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    document_id: str | None = None,
) -> list[Document]:
    """Split page-level documents into overlapping, token-bounded chunks."""
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        if document_id is not None:
            chunk.metadata["document_id"] = document_id
    return chunks


def load_and_chunk_pdf(pdf_path: str | Path, document_id: str | None = None) -> list[Document]:
    """Load a PDF and split it into indexable chunks in one step."""
    pages = load_pdf_pages(pdf_path)
    return chunk_documents(pages, document_id=document_id)
