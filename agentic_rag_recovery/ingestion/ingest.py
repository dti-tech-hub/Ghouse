"""Ingestion pipeline: extract → clean → chunk → store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models import Chunk, Document
from ..settings import Settings, get_settings
from .chunker import Chunker
from .cleaner import clean_text
from .loaders import LoadedDocument, load_directory, load_file


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass
class IngestionResult:
    documents: List[Document] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def summary(self) -> Dict[str, Any]:
        return {
            "documents": self.document_count,
            "chunks": self.chunk_count,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class _Ingestor:
    """Ingestion worker bound to a settings object (uses a repository)."""

    def __init__(self, repository: Any, settings: Optional[Settings] = None, chunker: Optional[Chunker] = None):
        self.repository = repository
        self.settings = settings or get_settings()
        self.chunker = chunker or Chunker(self.settings.chunking)

    def add_document(self, doc: Document) -> Document:
        doc.content = clean_text(doc.content)
        doc.checksum = doc.checksum or _checksum(doc.content)
        self.repository.upsert_document(doc)
        chunks = self.chunker.chunk_document(doc.document_id, doc.source, doc.content)
        self.repository.replace_chunks(doc.document_id, chunks)
        return doc

    def ingest_file(self, path: str | Path, skip_existing: bool = True) -> tuple[Optional[Document], List[Chunk], Optional[str]]:
        try:
            loaded = load_file(path)
        except Exception as exc:
            return None, [], str(exc)
        checksum = _checksum(loaded.content)
        existing = self.repository.db.fetch_one(
            "SELECT document_id, version FROM documents WHERE checksum=?", (checksum,)
        )
        if existing and skip_existing:
            return None, [], None
        doc = Document(
            source=loaded.source,
            title=loaded.title,
            doc_type=loaded.doc_type,
            content=loaded.content,
            checksum=checksum,
            version=(existing["version"] + 1) if existing else 1,
            metadata=loaded.metadata,
        )
        self.add_document(doc)
        return doc, self.repository.get_chunks(doc.document_id), None

    def ingest_text(
        self, text: str, source: str = "manual", title: str = "", metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[Document, List[Chunk]]:
        cleaned = clean_text(text)
        doc = Document(
            source=source,
            title=title or source,
            doc_type="txt",
            content=cleaned,
            checksum=_checksum(cleaned),
            metadata=metadata or {},
        )
        self.add_document(doc)
        return doc, self.repository.get_chunks(doc.document_id)

    def reingest_with_chunking(
        self, document_id: str, chunk_size: int, chunk_overlap: int
    ) -> List[Chunk]:
        """Re-chunk an existing document (used by the CHUNK_EXPANSION recovery)."""
        doc = self.repository.get_document(document_id)
        if doc is None:
            return []
        cfg = self.settings.chunking.copy()
        cfg.chunk_size = int(chunk_size)
        cfg.chunk_overlap = int(chunk_overlap)
        chunks = Chunker(cfg).chunk_document(doc.document_id, doc.source, doc.content)
        self.repository.replace_chunks(doc.document_id, chunks)
        return chunks

    def purge_document(self, document_id: str) -> bool:
        doc = self.repository.get_document(document_id)
        if not doc:
            return False
        self.repository.db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.repository.db.execute("DELETE FROM documents WHERE document_id=?", (document_id,))
        return True


def ingest_path(
    path: str | Path, repository: Any, settings: Optional[Settings] = None, skip_existing: bool = True
) -> IngestionResult:
    """Ingest a file or a directory of documents."""
    ingestor = _Ingestor(repository, settings)
    result = IngestionResult()
    p = Path(path)
    if not p.exists():
        result.errors.append(f"path not found: {p}")
        return result
    targets: Sequence[Path] = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.is_file()
    )
    for target in targets:
        doc, chunks, error = ingestor.ingest_file(target, skip_existing=skip_existing)
        if error:
            result.errors.append(error)
        elif doc is None:
            result.skipped += 1
        else:
            result.documents.append(doc)
            result.chunks.extend(chunks)
    return result


def ingest_paths(
    paths: Sequence[str | Path], repository: Any, settings: Optional[Settings] = None
) -> IngestionResult:
    combined = IngestionResult()
    for path in paths:
        one = ingest_path(path, repository, settings)
        combined.documents.extend(one.documents)
        combined.chunks.extend(one.chunks)
        combined.errors.extend(one.errors)
        combined.skipped += one.skipped
    return combined


def build_ingestor(repository: Any, settings: Optional[Settings] = None, chunker: Optional[Chunker] = None) -> _Ingestor:
    return _Ingestor(repository, settings, chunker)