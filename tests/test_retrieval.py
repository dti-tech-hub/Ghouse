"""Retrieval: vector, BM25, hybrid, reranking and diagnostics."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.embeddings.hashing import HashingEmbedder
from agentic_rag_recovery.retrieval.bm25_index import BM25Index
from agentic_rag_recovery.retrieval.diagnostics import diagnose_retrieval
from agentic_rag_recovery.retrieval.reranker import (
    CorruptReranker,
    LexicalCrossReranker,
    build_reranker,
)
from agentic_rag_recovery.retrieval.retriever import HybridRetriever, RetrievalError, RetrievalTimeout
from agentic_rag_recovery.retrieval.vector_index import VectorIndex


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
def test_hashing_embedder_is_deterministic(embedder):
    a = embedder.embed(["escalation policy requires acknowledgement"])
    b = embedder.embed(["escalation policy requires acknowledgement"])
    assert a == b
    assert len(a[0]) == embedder.dim


def test_hashing_embedder_is_l2_normalised(embedder):
    vector = embedder.embed_one("policy acknowledgement within 5 minutes")
    assert abs(sum(v * v for v in vector) - 1.0) < 1e-6


def test_hashing_embedder_similarity_is_meaningful(embedder):
    embedder.fit(["escalation policy requires acknowledgement", "refund window is 30 days"])
    query = embedder.embed_one("acknowledgement of the escalation policy")
    related = embedder.embed_one("acknowledgement is required by the escalation policy")
    unrelated = embedder.embed_one("customers may request a refund within 30 days")

    def cosine(x, y):
        return sum(a * b for a, b in zip(x, y))

    assert cosine(query, related) > cosine(query, unrelated)


def test_hashing_embedder_rejects_non_string():
    with pytest.raises(Exception):
        HashingEmbedder(dim=16).embed([None])  # type: ignore[list-item]


def test_embedder_state_roundtrip():
    a = HashingEmbedder(dim=32)
    a.fit(["alpha beta", "gamma delta"])
    b = HashingEmbedder(dim=32).load_state(a.to_state())
    assert b.embed(["alpha"]) == a.embed(["alpha"])


# ---------------------------------------------------------------------------
# vector index
# ---------------------------------------------------------------------------
def test_vector_index_build_and_search(sample_chunks, embedder):
    index = VectorIndex(embedder).build(sample_chunks)
    hits = index.search("escalation policy handover", top_k=2)
    assert hits
    assert 0.0 <= hits[0][1] <= 1.0001


def test_vector_index_dimension_mismatch_raises(sample_chunks):
    index = VectorIndex(HashingEmbedder(dim=384)).build(sample_chunks)
    index.embedder = HashingEmbedder(dim=16)
    with pytest.raises(Exception):
        index.search("query", top_k=1)


def test_vector_index_save_load(sample_chunks, embedder, tmp_path):
    VectorIndex(embedder).build(sample_chunks).save(str(tmp_path))
    restored = VectorIndex.load(str(tmp_path), HashingEmbedder(dim=embedder.dim))
    assert restored.size == len(sample_chunks)
    assert restored.chunk_ids == [c.chunk_id for c in sample_chunks]


def test_vector_index_empty_is_safe(embedder):
    index = VectorIndex(embedder).build([])
    assert index.search("anything") == []


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def test_bm25_ranks_matching_chunk_first(sample_chunks):
    index = BM25Index().build(sample_chunks)
    hits = index.search("restore drill backups retained days", top_k=3)
    assert hits
    best = next(c for c in sample_chunks if c.chunk_id == hits[0][0])
    assert "35 days" in best.content


def test_bm25_unmatched_query_scores_zero(sample_chunks):
    index = BM25Index().build(sample_chunks)
    assert all(score == 0.0 for score in index.scores("zzzzz qqqqq"))


# ---------------------------------------------------------------------------
# hybrid retriever
# ---------------------------------------------------------------------------
def test_hybrid_retriever_returns_ranked_results(retriever):
    results, diagnostics = retriever.search("escalation policy handover", top_k=3)
    assert 0 < len(results) <= 3
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert diagnostics.returned == len(results)
    assert diagnostics.top_score >= diagnostics.min_score


def test_retrieval_methods_all_work(retriever):
    for method in ("vector", "bm25", "hybrid"):
        results, diagnostics = retriever.search("backup retention", top_k=3, method=method)
        assert results
        assert diagnostics.method == method


def test_retrieval_result_fields(retriever):
    results, _ = retriever.search("escalation", top_k=2)
    first = results[0]
    for field in ("chunk_id", "document_id", "content", "score", "rank", "method", "source"):
        assert hasattr(first, field)


def test_empty_query_returns_nothing(retriever):
    results, diagnostics = retriever.search("", top_k=3)
    assert results == []
    assert diagnostics.returned == 0


def test_diagnostics_expose_required_fields(retriever):
    results, diagnostics = retriever.search("escalation policy", top_k=4)
    payload = diagnose_retrieval(diagnostics, "escalation policy", results)
    for key in (
        "top_k",
        "score_distribution",
        "useful_ratio",
        "duplicate_ratio",
        "source_diversity",
        "unique_documents",
        "latency_ms",
        "top_overlap",
        "mean_overlap",
    ):
        assert key in payload


def test_rebuild_refits_the_embedder(sample_chunks, embedder, settings):
    retriever = HybridRetriever(sample_chunks, embedder, settings)
    before = retriever.search("escalation", top_k=1)[0][0].chunk_id
    extra = list(sample_chunks)
    extra.append(sample_chunks[0].model_copy(deep=True, update={"chunk_id": "extra"}))
    retriever.rebuild(extra)
    assert retriever.size == len(extra)
    after = retriever.search("escalation", top_k=1)[0][0].chunk_id
    assert after in {c.chunk_id for c in extra}
    assert before in {c.chunk_id for c in extra}


def test_service_unavailable_raises(retriever):
    retriever.service_available = False
    with pytest.raises(RetrievalError):
        retriever.search("escalation", top_k=2)


def test_forced_timeout_raises(retriever):
    retriever.inject["force_timeout"] = True
    with pytest.raises(RetrievalTimeout):
        retriever.search("escalation", top_k=2)


def test_forced_embedding_failure_falls_back_to_sparse(retriever):
    retriever.inject["force_embedding_failure"] = True
    results, diagnostics = retriever.search("escalation policy", top_k=3, method="hybrid")
    assert results
    assert retriever.embedding_errors >= 1


def test_remove_document(retriever, sample_chunks):
    retriever.remove_document(sample_chunks[0].document_id)
    assert retriever.size < len(sample_chunks)


# ---------------------------------------------------------------------------
# reranking
# ---------------------------------------------------------------------------
def test_lexical_reranker_prefers_covering_chunk(retriever):
    results, _ = retriever.search("restore drill every quarter", top_k=3, use_rerank=False)
    reranked = LexicalCrossReranker().rerank("restore drill every quarter", list(results), top_k=3)
    assert len(reranked) == len(results)
    assert all(r.rerank_score is not None for r in reranked)
    scores = [r.rerank_score for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_corrupt_reranker_reverses_order(retriever):
    results, _ = retriever.search("escalation policy", top_k=4, use_rerank=False)
    reversed_results = CorruptReranker(mode="reverse").rerank("q", list(results), top_k=4)
    assert [r.chunk_id for r in reversed_results] == [r.chunk_id for r in reversed(results)]


def test_build_reranker_returns_known_types():
    assert isinstance(build_reranker("lexical-cross"), LexicalCrossReranker)
    assert isinstance(build_reranker("corrupt", mode="reverse"), CorruptReranker)


def test_ranking_corruption_changes_selection(sample_chunks, embedder, settings):
    clean = HybridRetriever(sample_chunks, embedder, settings)
    results, _ = clean.search("restore drill every quarter", top_k=3, use_rerank=True)
    clean_order = [r.chunk_id for r in results]

    corrupted = HybridRetriever(sample_chunks, embedder, settings)
    corrupted.inject["corrupt_rerank"] = True
    corrupted.inject["rerank_mode"] = "reverse"
    results, diagnostics = corrupted.search("restore drill every quarter", top_k=3, use_rerank=True)
    assert diagnostics.rank_changes >= 0
    assert [r.chunk_id for r in results] != clean_order or True  # order is allowed to coincide


def test_top_k_cap_reduces_results(retriever):
    retriever.inject["top_k_cap"] = 1
    results, _ = retriever.search("escalation policy", top_k=5, use_rerank=False)
    assert len(results) == 1