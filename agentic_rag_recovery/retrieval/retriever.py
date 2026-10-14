"""Hybrid retriever + diagnostics.

Pipeline: query → (dense | sparse | RRF-fused) → candidate pool → rerank →
top-k → :class:`RetrievalDiagnostics`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..embeddings.base import Embedder, EmbeddingError
from ..models import Chunk, RetrievalDiagnostics, RetrievalMethod, RetrievedChunk
from ..settings import RetrievalConfig, Settings, get_settings
from ..textutil import mean, median, stddev, tokenize
from .bm25_index import BM25Index
from .reranker import CorruptReranker, build_reranker
from .vector_index import VectorIndex


class RetrievalError(RuntimeError):
    """Maps to INFRASTRUCTURE_FAILURE / retrieval_service_unavailable."""

    failure_subtype = "retrieval_service_unavailable"


class RetrievalTimeout(RetrievalError):
    failure_subtype = "timeout"


@dataclass
class _Candidate:
    chunk: Chunk
    dense: float = 0.0
    sparse: float = 0.0
    fused: float = 0.0
    original_rank: int = 0
    rerank: float = 0.0


class HybridRetriever:
    """Owns the indexes and executes vector / BM25 / hybrid search."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        settings: Optional[Settings] = None,
    ):
        from ..observability.trace import get_logger

        self._logger = get_logger(__name__)
        self.settings = settings or get_settings()
        self.config: RetrievalConfig = self.settings.retrieval
        self.embedder = embedder
        self.chunks: List[Chunk] = list(chunks)
        self.chunk_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self.vector_index = VectorIndex(embedder, backend=self.config.vector_backend)
        self.bm25_index = BM25Index()
        self.vector_index.build(self.chunks)
        self.bm25_index.build(self.chunks)
        self.errors: List[str] = []
        self.timeout_ms: float = 0.0
        self.service_available: bool = True
        self.embedding_errors: int = 0
        self.timeouts: int = 0
        self._dense_raw: Dict[str, float] = {}
        self._sparse_raw: Dict[str, float] = {}
        # fault-injection switches (set only by the injector, never by user input)
        self.inject: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.chunks)

    def rebuild(self, chunks: Sequence[Chunk]) -> "HybridRetriever":
        """Replace the corpus and refit both indexes.

        The embedding statistics must be refitted whenever the corpus changes
        (re-chunking, document removal, injected decoys), otherwise the dense
        vectors keep IDF weights from a corpus that no longer exists and the
        dense ranking silently degrades.
        """
        self.chunks = list(chunks)
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        fitter = getattr(self.embedder, "fit", None)
        if callable(fitter) and self.chunks:
            try:
                fitter([c.content for c in self.chunks])
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning("could not refit embedder: %s", exc)
        self.vector_index.build(self.chunks)
        self.bm25_index.build(self.chunks)
        return self

    def reindex_document(self, chunks: Sequence[Chunk]) -> "HybridRetriever":
        if not chunks:
            return self
        doc_id = chunks[0].document_id
        self.chunks = [c for c in self.chunks if c.document_id != doc_id] + list(chunks)
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.vector_index.build(self.chunks)
        self.bm25_index.build(self.chunks)
        return self

    def remove_document(self, document_id: str) -> "HybridRetriever":
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        self.vector_index.build(self.chunks)
        self.bm25_index.build(self.chunks)
        return self

    def remove_document_ids(self, document_ids: Sequence[str]) -> "HybridRetriever":
        targets = set(document_ids)
        if not targets:
            return self
        return self.remove_document_ids_bulk(targets)

    def remove_document_ids_bulk(self, document_ids: Set[str]) -> "HybridRetriever":
        keep = [c for c in self.chunks if c.document_id not in document_ids]
        return self.rebuild(keep)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        method: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        candidate_k: Optional[int] = None,
        exclude_chunk_ids: Optional[Set[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievedChunk], RetrievalDiagnostics]:
        started = time.perf_counter()
        method = (method or self.config.method or RetrievalMethod.HYBRID.value).lower()
        top_k = int(top_k or self.config.top_k)
        candidate_k = int(candidate_k or max(self.config.candidate_k, top_k * 2))
        rerank_enabled = self.config.rerank_enabled if use_rerank is None else bool(use_rerank)
        exclude = exclude_chunk_ids or set()

        diagnostics = RetrievalDiagnostics(top_k=top_k, candidate_pool=candidate_k, method=method)

        if self.inject.get("force_empty_retrieval"):
            diagnostics.notes.append("empty retrieval forced by fault injection")
            diagnostics.latency_ms = (time.perf_counter() - started) * 1000.0
            return [], diagnostics

        if self.inject.get("force_timeout"):
            self.timeouts += 1
            diagnostics.latency_ms = (time.perf_counter() - started) * 1000.0
            self.timeout_ms = diagnostics.latency_ms
            raise RetrievalTimeout("retrieval timeout (fault injection)")

        if not self.service_available:
            self.errors.append("retrieval backend marked unavailable (fault injection)")
            diagnostics.notes.append("service_unavailable")
            diagnostics.latency_ms = (time.perf_counter() - started) * 1000.0
            raise RetrievalError("retrieval service unavailable")

        if not query or not query.strip():
            diagnostics.notes.append("empty query")
            diagnostics.latency_ms = (time.perf_counter() - started) * 1000.0
            return [], diagnostics

        pool = self._candidate_pool(query, method, candidate_k)
        if exclude:
            pool = [c for c in pool if c.chunk.chunk_id not in exclude]
        if metadata_filter:
            pool = [c for c in pool if _matches(c.chunk.metadata, metadata_filter)]

        ranked: List[_Candidate] = sorted(pool, key=lambda c: -c.fused)[:candidate_k]
        results: List[RetrievedChunk] = []
        for position, cand in enumerate(ranked, start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=cand.chunk.chunk_id,
                    document_id=cand.chunk.document_id,
                    content=cand.chunk.content,
                    score=round(cand.fused, 6),
                    rank=position,
                    method=method,
                    dense_score=round(cand.dense, 6),
                    sparse_score=round(cand.sparse, 6),
                    original_rank=position,
                    page=cand.chunk.page,
                    source=cand.chunk.source,
                    metadata=dict(cand.chunk.metadata),
                )
            )

        diagnostics.candidate_chunk_ids = [rc.chunk_id for rc in results]

        if rerank_enabled and results:
            pool_size = max(self.config.rerank_candidates, top_k)
            candidates_for_rerank = results[:pool_size]
            reranker = (
                CorruptReranker(mode=self.inject.get("rerank_mode", "reverse"), seed=self.settings.seed)
                if self.inject.get("corrupt_rerank")
                else build_reranker(self.config.rerank_model, seed=self.settings.seed)
            )
            before = {r.chunk_id: r.rank for r in candidates_for_rerank}
            results = reranker.rerank(query, candidates_for_rerank, top_k=top_k)
            after = {r.chunk_id: r.rank for r in results}
            diagnostics.rank_changes = sum(
                1 for cid, rank in after.items() if before.get(cid, rank) != rank
            )
            diagnostics.reranker_applied = True
        else:
            results = results[:top_k]

        if self.inject.get("irrelevant_injection"):
            results = self._inject_irrelevant(results, query)
            diagnostics.notes.append("irrelevant documents injected by fault injection")

        diagnostics = self._diagnostics(results, diagnostics, query)
        diagnostics.latency_ms = (time.perf_counter() - started) * 1000.0
        return results, diagnostics

    # ------------------------------------------------------------------
    def _candidate_pool(self, query: str, method: str, candidate_k: int) -> List[_Candidate]:
        self._dense_raw = {}
        self._sparse_raw = {}
        dense: List[Tuple[str, float]] = []
        sparse: List[Tuple[str, float]] = []
        state_error = False

        if method in {RetrievalMethod.VECTOR.value, RetrievalMethod.HYBRID.value}:
            try:
                if self.inject.get("force_embedding_failure"):
                    self.embedding_errors += 1
                    raise EmbeddingError(
                        f"embedding backend failure injected (mode="
                        f"{self.inject.get('embedding_failure_mode', 'raise')})"
                    )
                dense = self.vector_index.search(query, top_k=max(candidate_k, 10))
            except EmbeddingError as exc:
                self.embedding_errors += 1
                self.errors.append(str(exc))
                state_error = True
                if method == RetrievalMethod.VECTOR.value:
                    raise RetrievalError(f"embedding failure: {exc}") from exc
            except Exception as exc:
                self.errors.append(str(exc))
                if method == RetrievalMethod.VECTOR.value:
                    raise RetrievalError(str(exc)) from exc
        self._dense_raw = dict(dense)

        if state_error and method == RetrievalMethod.HYBRID.value:
            # dense retrieval is unusable: the sparse index needs no embedding
            # service, so the hybrid path degrades to BM25 and stays honest
            # about it (the failure is still recorded as an infrastructure
            # error so the diagnoser sees it).
            self.errors.append("dense branch unavailable - hybrid degraded to BM25")
        if method in {RetrievalMethod.BM25.value, RetrievalMethod.HYBRID.value}:
            sparse = self.bm25_index.search(query, top_k=max(candidate_k, 10))
        self._sparse_raw = dict(sparse)

        alpha = (
            float(self.config.hybrid_alpha)
            if method == RetrievalMethod.HYBRID.value
            else (1.0 if method == RetrievalMethod.VECTOR.value else 0.0)
        )
        candidates: Dict[str, _Candidate] = {}
        if method == RetrievalMethod.HYBRID.value and dense and sparse:
            rrf_k = float(self.config.rrf_k)
            for rank, (cid, _score) in enumerate(dense, start=1):
                c = self._candidate(cid)
                c.fused = alpha / (rrf_k + rank)
                candidates[cid] = c
            for rank, (cid, _score) in enumerate(sparse, start=1):
                c = self._candidate(cid)
                c.fused += (1.0 - alpha) / (rrf_k + rank)
                candidates[cid] = c
        else:
            # when the dense branch is unavailable the sparse index carries the
            # search on its own (an independent backend, not a silent fallback)
            primary = dense if (method != RetrievalMethod.BM25.value and dense) else sparse
            for rank, (cid, score) in enumerate(primary, start=1):
                c = self._candidate(cid)
                c.fused = float(score)
                candidates[cid] = c

        dense_map = dict(dense)
        sparse_map = dict(sparse)
        for cid, cand in candidates.items():
            cand.dense = dense_map.get(cid, 0.0)
            cand.sparse = sparse_map.get(cid, 0.0)
        return list(candidates.values())

    def _candidate(self, chunk_id: str) -> _Candidate:
        chunk = self.chunk_by_id.get(chunk_id)
        if chunk is None:
            raise RetrievalError(f"index references unknown chunk {chunk_id}")
        return _Candidate(chunk=chunk)

    def _inject_irrelevant(self, results: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        present = {r.chunk_id for r in results}
        q_tokens = set(tokenize(query))
        unrelated = [
            c
            for c in self.chunks
            if c.chunk_id not in present and (q_tokens & set(tokenize(c.content))) == set()
        ]
        unrelated.sort(key=lambda c: c.chunk_id)
        injected: List[RetrievedChunk] = []
        for position, chunk in enumerate(unrelated[: max(1, len(results))], start=1):
            injected.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=round(0.9 - 0.05 * position, 6),
                    rank=position,
                    method="injected",
                    dense_score=0.0,
                    sparse_score=0.0,
                    original_rank=position,
                    page=chunk.page,
                    source=chunk.source,
                    metadata=dict(chunk.metadata),
                )
            )
        return injected

    # ------------------------------------------------------------------
    def build_diagnostics(
        self,
        query: str,
        results: Sequence[RetrievedChunk],
        base: Optional[RetrievalDiagnostics] = None,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
    ) -> RetrievalDiagnostics:
        """Compute retrieval diagnostics for the *final* ranked result list.

        Sub-query diagnostics are aggregated separately for reporting, but the
        authoritative diagnostics must describe exactly the chunks that reached
        context assembly.
        """
        diagnostics = base.model_copy() if base is not None else RetrievalDiagnostics()
        diagnostics.top_k = int(top_k or diagnostics.top_k or self.config.top_k)
        if candidate_pool is not None:
            diagnostics.candidate_pool = int(candidate_pool)
        diagnostics.method = diagnostics.method or self.config.method
        return self._diagnostics(list(results), diagnostics, query)

    # ------------------------------------------------------------------
    def _diagnostics(
        self,
        results: Sequence[RetrievedChunk],
        diagnostics: RetrievalDiagnostics,
        query: str,
    ) -> RetrievalDiagnostics:
        scores = [float(r.score) for r in results]
        diagnostics.returned = len(results)
        diagnostics.score_distribution = [round(s, 6) for s in scores]
        if scores:
            diagnostics.top_score = round(scores[0], 6)
            diagnostics.mean_score = round(mean(scores), 6)
            diagnostics.median_score = round(median(scores), 6)
            diagnostics.min_score = round(min(scores), 6)
            diagnostics.max_score = round(max(scores), 6)
            diagnostics.score_gap = round(scores[0] - scores[1], 6) if len(scores) > 1 else round(scores[0], 6)
            diagnostics.score_std = round(stddev(scores), 6)

        q_tokens = set(tokenize(query))
        overlaps = [
            _query_chunk_overlap(query, r.content) if q_tokens else 0.0 for r in results
        ]
        useful_overlap = float(self.settings.thresholds.useful_overlap)
        useful = sum(1 for o in overlaps if o >= useful_overlap)
        diagnostics.top_overlap = round(max(overlaps), 6) if overlaps else 0.0
        diagnostics.mean_overlap = round(mean(overlaps), 6) if overlaps else 0.0
        diagnostics.useful_overlap = useful_overlap
        diagnostics.useful_count = useful
        diagnostics.useful_ratio = round(useful / len(results), 6) if results else 0.0

        seen: Dict[str, int] = {}
        for r in results:
            seen[r.chunk_id] = seen.get(r.chunk_id, 0) + 1
        duplicates = sum(count - 1 for count in seen.values() if count > 1)
        diagnostics.duplicate_ratio = round(duplicates / len(results), 6) if results else 0.0

        diagnostics.unique_documents = len({r.document_id for r in results})
        diagnostics.source_diversity = len(
            {(r.source or r.document_id).split("/")[-1] for r in results}
        )
        rerank_scores = [float(r.rerank_score) for r in results if r.rerank_score is not None]
        if len(rerank_scores) > 1:
            diagnostics.rerank_score_margin = round(rerank_scores[0] - rerank_scores[1], 6)
        return diagnostics

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "chunks": self.size,
            "vector_index": self.vector_index.health(),
            "bm25_index": self.bm25_index.health(),
            "errors": self.errors[-5:],
            "service_available": self.service_available,
            "embedding_errors": self.embedding_errors,
            "timeouts": self.timeouts,
        }

    def save(self, directory: Optional[str] = None) -> str:
        root = directory or self.settings.index_dir()
        self.vector_index.save(root)
        self.bm25_index.save(root)
        return root


def _matches(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    return all(metadata.get(key) == expected for key, expected in filters.items())


def _query_chunk_overlap(query: str, content: str) -> float:
    """Fraction of query content tokens that appear in the chunk."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    c = set(tokenize(content))
    return len(q & c) / len(q)