"""Execution runtime: the mutable object shared by every stage, the fault
injector, the recovery strategies and the experiment engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..db.repository import Repository
from ..diagnosis.engine import DiagnosisEngine
from ..diagnosis.llm_diagnoser import LLMDiagnoser
from ..embeddings.base import Embedder
from ..llm.base import LLMProvider
from ..llm.factory import build_diagnoser_provider, build_provider
from ..models import Chunk, utcnow
from ..observability.trace import MetricsRegistry, TraceCollector, get_logger
from ..retrieval.retriever import HybridRetriever
from ..settings import Settings, get_settings
from ..validation.verifier import RecoveryVerifier

logger = get_logger(__name__)


@dataclass
class GenerationKnobs:
    """Switches the fault injector and recovery strategies may flip."""

    ground_answer: bool = True
    require_grounding: bool = True
    cite_sources: bool = True
    max_context_tokens: Optional[int] = None
    max_chunks: Optional[int] = None
    retrieval_method: Optional[str] = None
    top_k: Optional[int] = None
    candidate_k: Optional[int] = None
    use_rerank: Optional[bool] = None
    force_decomposition: Optional[bool] = None
    force_rewrite: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunk_expansion: bool = False
    context_expansion: bool = False
    alternate_provider: Optional[str] = None


@dataclass
class Runtime:
    """Everything a pipeline execution needs."""

    settings: Settings = field(default_factory=get_settings)
    repository: Optional[Repository] = None
    embedder: Optional[Embedder] = None
    retriever: Optional[HybridRetriever] = None
    provider: Optional[LLMProvider] = None
    diagnoser_provider: Optional[LLMProvider] = None
    diagnosis_engine: Optional[DiagnosisEngine] = None
    verifier: RecoveryVerifier = field(default_factory=RecoveryVerifier)
    ingestor: Any = None
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    trace: TraceCollector = field(default_factory=TraceCollector)

    knobs: GenerationKnobs = field(default_factory=GenerationKnobs)
    fault_flags: Dict[str, Any] = field(default_factory=dict)
    active_fault_ids: List[str] = field(default_factory=list)
    baseline_retriever: Optional[HybridRetriever] = None
    baseline_chunks: List[Chunk] = field(default_factory=list)
    chunk_config_snapshot: Optional[Dict[str, Any]] = None
    document_removals: List[str] = field(default_factory=list)
    decoy_chunks: List[Chunk] = field(default_factory=list)
    gold_chunk_ids: Set[str] = field(default_factory=set)
    gold_document_ids: Set[str] = field(default_factory=set)
    answer_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def t(self):
        return self.settings.thresholds

    @property
    def budget_config(self):
        return self.settings.budget

    def set_trace(self, trace: TraceCollector) -> None:
        self.trace = trace

    def ensure_diagnosis_engine(self) -> DiagnosisEngine:
        if self.diagnosis_engine is None:
            llm = None
            if self.settings.diagnosis.use_llm_fallback:
                provider = self.diagnoser_provider
                if provider is None:
                    provider = build_diagnoser_provider(self.settings)
                    self.diagnoser_provider = provider
                llm = LLMDiagnoser(provider, self.settings)
            self.diagnosis_engine = DiagnosisEngine(
                detector=None, llm_diagnoser=llm, settings=self.settings
            )
        return self.diagnosis_engine

    def ensure_provider(self) -> LLMProvider:
        if self.provider is None:
            self.provider = build_provider(settings=self.settings)
        return self.provider

    # ------------------------------------------------------------------
    def snapshot_chunks(self) -> List[Chunk]:
        """Deep copy of the current chunk set (safe to restore later)."""
        if self.retriever is None:
            return []
        return [c.model_copy(deep=True) for c in self.retriever.chunks]

    def restore_clean_state(self) -> None:
        """Undo every active fault (used between baseline and injected runs)."""
        self.knobs = GenerationKnobs()
        self.fault_flags = {}
        self.active_fault_ids = []
        self.document_removals = []
        self.decoy_chunks = []
        self.gold_chunk_ids = set()
        self.gold_document_ids = set()
        if self.retriever is not None:
            self.retriever.inject = {}
            self.retriever.service_available = True
            self.retriever.embedding_errors = 0
            self.retriever.timeouts = 0
            self.retriever.errors = []
            if self.baseline_chunks:
                # restore the pristine corpus captured before any fault was
                # activated - the retriever object itself is shared across runs
                self.retriever.rebuild([c.model_copy(deep=True) for c in self.baseline_chunks])
        self._restore_chunking()

    def _restore_chunking(self) -> None:
        snapshot = self.chunk_config_snapshot
        if not snapshot:
            return
        self.settings.chunking.chunk_size = int(snapshot["chunk_size"])
        self.settings.chunking.chunk_overlap = int(snapshot["chunk_overlap"])
        self.settings.chunking.min_chunk_chars = int(snapshot.get("min_chunk_chars", 30))
        if self.ingestor is not None:
            self.ingestor.chunker.config = self.settings.chunking
        self.chunk_config_snapshot = None
        if self.repository is not None and self.baseline_chunks:
            self._rechunk_all(self.baseline_chunks)

    def _rechunk_all(self, source_chunks: List[Chunk]) -> None:
        if self.ingestor is None or self.repository is None:
            return
        by_doc: Dict[str, List[Chunk]] = {}
        for chunk in source_chunks:
            by_doc.setdefault(chunk.document_id, []).append(chunk)
        rebuilt: List[Chunk] = []
        for document_id, chunks in by_doc.items():
            document = self.repository.get_document(document_id)
            if document is None:
                rebuilt.extend(chunks)
                continue
            cfg = copy.copy(self.settings.chunking)
            from ..ingestion.chunker import Chunker

            new_chunks = Chunker(cfg).chunk_document(
                document.document_id, document.source, document.content
            )
            self.repository.replace_chunks(document_id, new_chunks)
            rebuilt.extend(new_chunks)
        if self.retriever is not None:
            self.retriever.rebuild(rebuilt)
        self.baseline_chunks = rebuilt
        self._remap_gold(rebuilt)

    def _remap_gold(self, chunks: List[Chunk]) -> None:
        """Re-anchor the gold chunk ids after the corpus was re-chunked.

        Re-chunking creates new chunk ids, so ids captured before the fault no
        longer exist; without this, every experiment would report recall = 0
        for reasons that have nothing to do with the system under test.
        """
        if not self.gold_chunk_ids:
            return
        from ..textutil import tokenize

        existing = {c.chunk_id for c in chunks}
        surviving = self.gold_chunk_ids & existing
        keywords = [k.lower() for k in self.answer_keywords if k]
        if surviving and not keywords:
            self.gold_chunk_ids = surviving
            return
        pool = [c for c in chunks if keywords and sum(1 for k in keywords if k in c.content.lower())]
        if not pool:
            pool = list(chunks)
        query_tokens = set(tokenize(" ".join(keywords))) or set(tokenize(" ".join(pool[:1])))
        scored = sorted(
            ((len(query_tokens & set(tokenize(c.content))), c.chunk_id) for c in pool), reverse=True
        )
        self.gold_chunk_ids = {scored[0][1]} if scored else surviving

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": getattr(self.provider, "name", None),
            "model": getattr(self.provider, "model", None),
            "retrieval_method": self.knobs.retrieval_method or self.settings.retrieval.method,
            "top_k": self.knobs.top_k or self.settings.retrieval.top_k,
            "chunks": self.retriever.size if self.retriever else 0,
            "documents": self.repository.document_count() if self.repository else 0,
            "active_faults": list(self.active_fault_ids),
            "knobs": vars(self.knobs),
            "mode": self.settings.mode,
            "labels": dict(self.settings.labels),
            "created_at": utcnow(),
        }