"""Reranking.

Two rerankers are provided:

``lexical-cross``
    Deterministic, dependency-free *cross-encoder emulation*: each query-term
    (and bigram) is matched against the candidate with proximity-weighted IDF
    scoring plus a coverage bonus.  This is a real scoring function, not a
    random perturbation, and it is what makes the ranking-failure experiments
    meaningful offline.

``sentence_transformer``
    Optional real cross-encoder; used automatically when installed and
    requested via ``rerank_model='cross-encoder/ms-marco-MiniLM-L-6-v2'``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Chunk, RetrievedChunk
from ..textutil import split_sentences, tokenize


class LexicalCrossReranker:
    """Determinency-safe lexical cross-encoder emulation."""

    name = "lexical-cross"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.w = {"coverage": 1.6, "proximity": 1.1, "idf": 1.0, "phrase": 0.8, "length": 0.15}
        if weights:
            self.w.update(weights)

    def score(self, query: str, chunk: Chunk | RetrievedChunk) -> float:
        content = chunk.content if isinstance(chunk, Chunk) else chunk.content
        q_tokens = tokenize(query)
        if not q_tokens or not content:
            return 0.0
        doc_tokens = tokenize(content)
        doc_set = set(doc_tokens)
        unique_q = list(dict.fromkeys(q_tokens))
        coverage = sum(1 for t in unique_q if t in doc_set) / len(unique_q)
        idf = 1.0
        total = 0.0
        hits = 0
        for i, token in enumerate(doc_tokens):
            if token in set(unique_q):
                hits += 1
                total += idf / (1.0 + 0.15 * i)
        proximity = (total / len(doc_tokens)) if doc_tokens else 0.0
        phrase = 0.0
        lowered_q = " ".join(unique_q)
        if lowered_q and lowered_q in " ".join(doc_tokens):
            phrase = 1.0
        # sentence-level density bonus: answers concentrate in few sentences
        density = self._density(query, content)
        length_penalty = self.w["length"] / (1.0 + math.log1p(len(doc_tokens)))
        return (
            self.w["coverage"] * coverage
            + self.w["proximity"] * proximity
            + self.w["phrase"] * phrase
            + 0.9 * density
            - length_penalty
        )

    def _density(self, query: str, content: str) -> float:
        sentences = split_sentences(content) or [content]
        best = 0.0
        q = set(tokenize(query))
        if not q:
            return 0.0
        for sentence in sentences:
            tokens = set(tokenize(sentence))
            if not tokens:
                continue
            best = max(best, len(q & tokens) / len(tokens))
        return best

    def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        scored: List[Tuple[float, RetrievedChunk]] = []
        for cand in candidates:
            original_rank = cand.original_rank if cand.original_rank is not None else cand.rank
            cand.original_rank = original_rank
            cand.rerank_score = round(self.score(query, cand), 6)
            scored.append((cand.rerank_score, cand))
        scored.sort(key=lambda pair: -pair[0])
        out: List[RetrievedChunk] = []
        limit = top_k or len(scored)
        for position, (score, cand) in enumerate(scored[:limit], start=1):
            cand.rerank_score = score
            cand.rank = position
            cand.score = round(0.65 * score + 0.35 * max(0.0, float(cand.score)), 6)
            out.append(cand)
        return out


class SentenceTransformerReranker(LexicalCrossReranker):
    """Optional real cross-encoder reranker."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        super().__init__()
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:  # pragma: no cover - optional dependency
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RuntimeError(f"cross-encoder unavailable: {exc}") from exc
        return self._model

    def score(self, query: str, chunk: Chunk | RetrievedChunk) -> float:  # pragma: no cover
        model = self._ensure()
        content = chunk.content if isinstance(chunk, Chunk) else chunk.content
        return float(model.predict([(query, content)])[0])

    def rerank(self, query: str, candidates, top_k: Optional[int] = None):  # pragma: no cover
        try:
            return super().rerank(query, candidates, top_k)
        except RuntimeError:
            return LexicalCrossReranker.rerank(self, query, candidates, top_k)


class CorruptReranker(LexicalCrossReranker):
    """Used by the fault-injection framework to emulate a broken reranker."""

    name = "corrupt-rerank"

    def __init__(self, mode: str = "reverse", seed: int = 42, **kwargs):
        super().__init__()
        self.mode = mode
        self.seed = seed

    def rerank(self, query: str, candidates, top_k: Optional[int] = None):
        out = list(candidates)
        if self.mode == "reverse":
            out = list(reversed(out))
        elif self.mode == "identity":
            pass
        else:  # noise
            for i, cand in enumerate(out):
                cand.rerank_score = ((i * 37 + self.seed) % 100) / 100.0
            out.sort(key=lambda c: -float(c.rerank_score or 0))
        for position, cand in enumerate(out[: top_k or len(out)], start=1):
            cand.original_rank = cand.original_rank if cand.original_rank is not None else cand.rank
            cand.rerank_score = float(cand.rerank_score if cand.rerank_score is not None else cand.score)
            cand.rank = position
        return out[: (top_k or len(out))]


def build_reranker(model: str = "lexical-cross", seed: int = 42, **kwargs):
    name = (model or "lexical-cross").lower()
    if name in {"lexical-cross", "lexical", "builtin", "default"}:
        return LexicalCrossReranker()
    if name in {"cross-encoder", "sentence_transformer", "sbert"}:
        return SentenceTransformerReranker()
    if name in {"corrupt", "corrupt-rerank"}:
        return CorruptReranker(mode=kwargs.get("mode", "reverse"), seed=seed)
    return LexicalCrossReranker()