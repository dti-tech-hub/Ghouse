"""Agentic RAG pipeline: stages, state machine and orchestrator.

Imports are resolved lazily (PEP 562).  The fault injector needs
``pipeline.runtime`` and the orchestrator needs the fault injector, so eager
re-exports here would create an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .state import PipelineState, StageResult, SubQueryResult

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import OrchestratorResult, RAGOrchestrator
    from .runtime import GenerationKnobs, Runtime
    from .sequential import (
        LangGraphPipelineRunner,
        SequentialPipelineRunner,
        build_runner,
    )

_LAZY = {
    "RAGOrchestrator": ("agentic_rag_recovery.pipeline.orchestrator", "RAGOrchestrator"),
    "OrchestratorResult": ("agentic_rag_recovery.pipeline.orchestrator", "OrchestratorResult"),
    "Runtime": ("agentic_rag_recovery.pipeline.runtime", "Runtime"),
    "GenerationKnobs": ("agentic_rag_recovery.pipeline.runtime", "GenerationKnobs"),
    "SequentialPipelineRunner": (
        "agentic_rag_recovery.pipeline.sequential",
        "SequentialPipelineRunner",
    ),
    "LangGraphPipelineRunner": (
        "agentic_rag_recovery.pipeline.sequential",
        "LangGraphPipelineRunner",
    ),
    "build_runner": ("agentic_rag_recovery.pipeline.sequential", "build_runner"),
}

__all__ = [
    "RAGOrchestrator",
    "OrchestratorResult",
    "Runtime",
    "GenerationKnobs",
    "PipelineState",
    "StageResult",
    "SubQueryResult",
    "SequentialPipelineRunner",
    "LangGraphPipelineRunner",
    "build_runner",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))