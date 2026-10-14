"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import (  # noqa: E402
    Application,
    bootstrap_demo_data,
    create_application,
    ensure_demo_ready,
)
from agentic_rag_recovery.embeddings.factory import build_embedder  # noqa: E402
from agentic_rag_recovery.evaluation.datasets import load_demo_dataset  # noqa: E402
from agentic_rag_recovery.ingestion.chunker import Chunker  # noqa: E402
from agentic_rag_recovery.models import Chunk  # noqa: E402
from agentic_rag_recovery.retrieval.retriever import HybridRetriever  # noqa: E402
from agentic_rag_recovery.settings import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    """Isolated settings rooted in a temporary directory."""
    root = tmp_path_factory.mktemp("arr")
    config = Settings.from_dict(
        {
            "mode": "demo",
            "seed": 42,
            "paths": {
                "root": str(root),
                "data_dir": str(ROOT / "data"),
                "demo_documents": str(ROOT / "data" / "demo_documents"),
                "demo_questions": str(ROOT / "data" / "demo_questions"),
                "demo_ground_truth": str(ROOT / "data" / "demo_ground_truth"),
                "storage_dir": str(root / "storage"),
                "index_dir": str(root / "storage" / "index"),
                "upload_dir": str(root / "storage" / "uploads"),
                "db_path": str(root / "storage" / "test.db"),
                "experiment_config_dir": str(ROOT / "experiments" / "configs"),
                "experiment_results_dir": str(root / "experiments"),
            },
        }
    )
    config.ensure_dirs()
    return config


@pytest.fixture(scope="session")
def app(settings: Settings) -> Application:
    application = create_application(settings=settings)
    bootstrap_demo_data(application, force=True)
    ensure_demo_ready(application)
    return application


@pytest.fixture(scope="session")
def dataset(settings: Settings):
    return load_demo_dataset(settings)


@pytest.fixture
def fresh_app(settings: Settings, tmp_path) -> Application:
    """A per-test application with an empty knowledge base."""
    import shutil

    isolated = Settings.from_dict(settings.to_dict())
    isolated.paths.db_path = str(tmp_path / "fresh.db")
    isolated.paths.index_dir = str(tmp_path / "index")
    isolated.ensure_dirs()
    application = create_application(settings=isolated)
    for document in application.repository.list_documents():
        application.repository.purge = None  # noqa: B010 - explicit no-op for clarity
    yield application
    application.database.close()


@pytest.fixture(autouse=True)
def _isolate_backend(settings: Settings):
    """Point the FastAPI backend at the isolated test database.

    Without this the API tests would ingest into the developer's real
    ``storage/agentic_rag.db`` and pollute it for every later run.
    """
    from backend.services import reset_app, set_settings

    previous = settings.paths.db_path
    set_settings(settings)
    yield
    settings.paths.db_path = previous
    reset_app()


@pytest.fixture
def chunker(settings: Settings) -> Chunker:
    return Chunker(settings.chunking)


@pytest.fixture
def embedder(settings: Settings):
    return build_embedder(settings=settings)


@pytest.fixture
def sample_chunks(chunker: Chunker) -> list:
    text = (
        "# Policy\n\nThe escalation policy requires on-call handover within 30 minutes. "
        "Severity 1 incidents must be acknowledged in 5 minutes.\n\n"
        "Backups are retained for 35 days and a restore drill runs every quarter."
    )
    return chunker.chunk_document("doc_test", "policy.md", text)


@pytest.fixture
def retriever(sample_chunks, embedder, settings: Settings) -> HybridRetriever:
    return HybridRetriever(sample_chunks, embedder, settings)