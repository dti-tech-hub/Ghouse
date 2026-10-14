"""Retrieval diagnostics helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from ..models import Chunk, RetrievalDiagnostics, RetrievedChunk
from ..settings import Settings, get_settings
from ..textutil import mean, tokenize


def retrieval_ground_truth(
    query: str, chunks: Sequence[Chunk], answer_keywords: Sequence[str] = ()
) -> Set[str]:
    """Identify answer-bearing chunks using lexical evidence.

    A chunk is *gold* when it contains a majority of the answer keywords, or -
    failing that - when it has the highest query overlap in the corpus.  This is
    a transparent, documented proxy rather than a human annotation.
    """
    if not chunks:
        return set()
    keywords = [k.lower() for k in answer_keywords if k]
    q_tokens = set(tokenize(query))

    def score(chunk: Chunk) -> float:
        tokens = set(tokenize(chunk.content))
        kw = sum(1 for k in keywords if k in chunk.content.lower())
        kw_score = (kw / len(keywords)) if keywords else 0.0
        overlap = (len(q_tokens & tokens) / len(q_tokens)) if q_tokens else 0.0
        return 0.7 * kw_score + 0.3 * overlap

    scored = sorted(((score(c), c.chunk_id) for c in chunks), reverse=True)
    if not scored:
        return set()
    best = scored[0][0]
    if best <= 0.0:
        return set()
    return {cid for value, cid in scored if value >= max(best * 0.6, best - 1e-9) and value > 0}


def mark_relevance(
    retrieved: Sequence[RetrievedChunk],
    gold_chunk_ids: Set[str],
    gold_document_ids: Optional[Set[str]] = None,
) -> List[RetrievedChunk]:
    """Attach ground-truth relevance flags to retrieved chunks (experiments only)."""
    out: List[RetrievedChunk] = []
    for r in retrieved:
        relevant = r.chunk_id in gold_chunk_ids or (
            gold_document_ids is not None and r.document_id in gold_document_ids
        )
        r.is_relevant = relevant
        out.append(r)
    return out


def diagnose_retrieval(
    diagnostics: RetrievalDiagnostics,
    query: str,
    retrieved: Sequence[RetrievedChunk],
    settings: Optional[Settings] = None,
    timeouts: int = 0,
    embedding_errors: int = 0,
) -> Dict[str, Any]:
    """Human-readable retrieval diagnostics for the trace viewer / dashboard."""
    settings = settings or get_settings()
    t = settings.thresholds
    flags: List[str] = []
    if timeouts:
        flags.append("retrieval_timeout")
    if embedding_errors:
        flags.append("embedding_failure")
    if diagnostics.returned == 0:
        flags.append("empty_retrieval")
    else:
        if diagnostics.top_overlap and diagnostics.top_overlap < t.min_top_overlap:
            flags.append("weak_top_coverage")
        if diagnostics.mean_overlap and diagnostics.mean_overlap < t.min_mean_overlap:
            flags.append("weak_mean_coverage")
        if diagnostics.top_score and diagnostics.top_score < t.min_top_score:
            flags.append("weak_top_score")
        if diagnostics.score_gap < t.min_score_gap:
            flags.append("flat_score_distribution")
        if diagnostics.useful_ratio < t.min_useful_chunk_ratio:
            flags.append("low_useful_ratio")
        if diagnostics.duplicate_ratio > t.max_duplicate_ratio:
            flags.append("high_duplicate_ratio")
    q_tokens = set(tokenize(query))
    overlaps = [
        len(q_tokens & set(tokenize(r.content))) / max(1, len(q_tokens)) for r in retrieved
    ]
    max_overlap = mean(overlaps) if overlaps else 0.0
    return {
        "flags": flags,
        "top_k": diagnostics.top_k,
        "returned": diagnostics.returned,
        "candidate_pool": diagnostics.candidate_pool,
        "top_score": diagnostics.top_score,
        "top_overlap": diagnostics.top_overlap,
        "mean_overlap": diagnostics.mean_overlap,
        "useful_overlap": diagnostics.useful_overlap,
        "mean_score": diagnostics.mean_score,
        "median_score": diagnostics.median_score,
        "score_gap": diagnostics.score_gap,
        "score_std": diagnostics.score_std,
        "score_distribution": diagnostics.score_distribution,
        "useful_count": diagnostics.useful_count,
        "useful_ratio": diagnostics.useful_ratio,
        "duplicate_ratio": diagnostics.duplicate_ratio,
        "source_diversity": diagnostics.source_diversity,
        "unique_documents": diagnostics.unique_documents,
        "latency_ms": round(diagnostics.latency_ms, 3),
        "method": diagnostics.method,
        "reranker_applied": diagnostics.reranker_applied,
        "rank_changes": diagnostics.rank_changes,
        "max_chunk_overlap": round(max_overlap, 4),
        "timeouts": timeouts,
        "embedding_errors": embedding_errors,
        "notes": diagnostics.notes,
    }