"""Bootstrap helpers: build the runtime stack (db, indexes, orchestrator)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .db.database import Database
from .db.repository import Repository
from .embeddings.factory import build_embedder
from .ingestion.chunker import Chunker
from .ingestion.ingest import IngestionResult, _Ingestor, ingest_path
from .pipeline.orchestrator import RAGOrchestrator
from .retrieval.retriever import HybridRetriever
from .settings import Settings, get_settings


@dataclass
class Application:
    """Everything a running system needs, wired together."""

    settings: Settings
    database: Database
    repository: Repository
    orchestrator: RAGOrchestrator

    @property
    def ingestor(self) -> _Ingestor:
        return self.orchestrator.ingestor

    def rebuild_indexes(self) -> Dict[str, Any]:
        self.orchestrator.reindex()
        return {
            "documents": self.repository.document_count(),
            "chunks": self.orchestrator.retriever.size,
            "vector_backend": self.orchestrator.retriever.vector_index.backend,
            "embedding": self.orchestrator.embedder.name,
        }


def create_application(
    settings: Optional[Settings] = None,
    create_schema: bool = True,
    reindex: bool = True,
    auto_bootstrap: bool = True,
) -> Application:
    settings = settings or get_settings()
    settings.ensure_dirs()
    database = Database(settings.db_path(), create_schema=create_schema)
    repository = Repository(database)
    if auto_bootstrap:
        # an empty knowledge base is a valid state, but loading the demonstration
        # corpus makes the system usable out of the box
        try:
            demo = Path(settings.paths.demo_documents)
            if repository.document_count() == 0 and demo.exists():
                ingest_path(demo, repository, settings, skip_existing=True)
        except Exception:
            pass
    embedder = build_embedder(settings=settings)
    orchestrator = RAGOrchestrator(settings=settings, repository=repository, embedder=embedder)
    if reindex:
        orchestrator.reindex()
    repository.save_configuration(
        name="default",
        kind="settings",
        payload=settings.to_dict(),
        description="Runtime configuration of this deployment",
    )
    repository.register_model(
        name=settings.generation.model,
        provider=settings.generation.provider,
        role="generation",
        params={"temperature": settings.generation.temperature, "max_tokens": settings.generation.max_tokens},
        is_default=True,
    )
    return Application(settings=settings, database=database, repository=repository, orchestrator=orchestrator)


def bootstrap_demo_data(
    app: Application, documents_dir: Optional[str] = None, force: bool = False
) -> IngestionResult:
    """Ingest the demonstration corpus (idempotent unless ``force``)."""
    path = Path(documents_dir or app.settings.paths.demo_documents)
    if not path.exists():
        raise FileNotFoundError(f"demo documents directory not found: {path}")
    result = ingest_path(path, app.repository, app.settings, skip_existing=not force)
    if result.document_count or force:
        app.orchestrator.reindex()
    return result


def ensure_demo_ready(app: Application) -> Dict[str, Any]:
    """Idempotently ensure a populated knowledge base."""
    if app.repository.document_count() == 0:
        bootstrap_demo_data(app)
    elif app.orchestrator.retriever.size == 0:
        app.orchestrator.reindex()
    return {
        "documents": app.repository.document_count(),
        "chunks": app.orchestrator.retriever.size,
        "index_dir": str(app.settings.index_dir()),
        "db_path": str(app.settings.db_path()),
    }