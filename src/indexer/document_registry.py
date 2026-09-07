"""Local registry tracking which filings have been indexed into the vector store.

Chunks/embeddings live in Postgres; this small JSON file just tracks which document_ids
exist and their company/CIK/fiscal-year metadata, so the UI can offer a document picker
and the audit checklist can resolve which company/year a selected document refers to.
"""
import json
from pathlib import Path

from pydantic import BaseModel

DEFAULT_REGISTRY_PATH = Path("data/documents.json")


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    company_name: str
    cik: str
    fiscal_year: int
    chunk_count: int


class DocumentRegistry:
    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self._path = path

    def list_documents(self) -> list[DocumentRecord]:
        if not self._path.exists():
            return []
        raw = json.loads(self._path.read_text())
        return [DocumentRecord(**item) for item in raw]

    def get(self, document_id: str) -> DocumentRecord | None:
        return next((r for r in self.list_documents() if r.document_id == document_id), None)

    def add(self, record: DocumentRecord) -> None:
        records = [r for r in self.list_documents() if r.document_id != record.document_id]
        records.append(record)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps([r.model_dump() for r in records], indent=2))
