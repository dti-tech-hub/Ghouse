"""Agentic RAG Failure Diagnosis and Recovery System.

A locally-executable research system that makes failures inside an agentic
RAG pipeline observable, diagnoses them with a two-tier (rule + LLM)
diagnoser, applies targeted bounded recovery, verifies the recovery and
abstains honestly when no reliable answer can be produced.

Public entry points:
    * :class:`agentic_rag_recovery.pipeline.orchestrator.RAGOrchestrator`
    * :func:`agentic_rag_recovery.settings.get_settings`
"""

__version__ = "1.0.0"
__all__ = ["__version__"]