"""Application wiring for the HTTP API.

One :class:`Application` per process; created lazily so that importing
``backend.main`` is cheap and does not touch the database.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_rag_recovery import __version__
from agentic_rag_recovery.bootstrap import Application, create_application, ensure_demo_ready
from agentic_rag_recovery.evaluation.external import availability as external_availability
from agentic_rag_recovery.evaluation.runner import ExperimentRunner
from agentic_rag_recovery.faults.definitions import FaultRegistry
from agentic_rag_recovery.settings import Settings, get_settings

_lock = threading.Lock()
_app: Optional[Application] = None


def get_app(
    settings: Optional[Settings] = None,
    auto_bootstrap: bool = True,
) -> Application:
    """Return the process-wide application, creating it on first use."""
    global _app
    with _lock:
        if _app is None:
            settings = settings or get_settings()
            _app = create_application(settings=settings)
            if auto_bootstrap and _app.repository.document_count() == 0:
                try:
                    ensure_demo_ready(_app)
                except Exception:
                    # an empty knowledge base is a valid state; the API reports it
                    pass
        return _app


def reset_app() -> None:
    global _app
    with _lock:
        _app = None


def get_runner() -> ExperimentRunner:
    return ExperimentRunner(get_app())


def system_info() -> Dict[str, Any]:
    app = get_app()
    health = app.orchestrator.health()
    return {
        "version": __version__,
        "health": health,
        "external_evaluation": external_availability(),
        "fault_catalogue_size": len(FaultRegistry.types()),
        "experiment_configs": get_runner().available_configs(),
        "root": str(Path(__file__).resolve().parent.parent),
    }


def set_settings(settings: Settings) -> None:
    """Point the backend at a different configuration (used by tests)."""
    global _app
    with _lock:
        _settings = settings
        _app = create_application(settings=settings, auto_bootstrap=False)


__all__ = [
    "get_app",
    "reset_app",
    "get_runner",
    "system_info",
    "set_settings",
    "Application",
]