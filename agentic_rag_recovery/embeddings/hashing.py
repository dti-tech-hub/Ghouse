"""Deterministic hashed TF-IDF embedding (default offline backend)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from ..textutil import tokenize
from .base import Embedder, EmbeddingError


class HashingEmbedder(Embedder):
    """Signed random-projection of an IDF-weighted bag of stemmed tokens.

    * ``fit`` learns document frequencies from the corpus.
    * ``embed`` maps tokens to ``dim`` buckets (hashed), applies IDF weights and
      a sign flip to reduce collision bias, then L2-normalises.

    Deterministic: identical corpus + seed -> identical vectors, always.
    """

    name = "hashing"

    def __init__(self, dim: int = 384, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.idf: Dict[str, float] = {}
        self.default_idf: float = 1.0
        self.fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, corpus: Sequence[str]) -> "HashingEmbedder":
        doc_freq: Dict[str, int] = {}
        total = 0
        for text in corpus:
            total += 1
            for token in set(tokenize(text)):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        total = max(1, total)
        self.idf = {
            token: math.log((total + 1.0) / (count + 1.0)) + 1.0 for token, count in doc_freq.items()
        }
        self.default_idf = math.log((total + 1.0) / 1.0) + 1.0
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            if text is None:
                raise EmbeddingError("cannot embed None")
            if not isinstance(text, str):
                raise EmbeddingError(f"expected str, got {type(text).__name__}")
            vector = [0.0] * self.dim
            tokens = tokenize(text)
            if not tokens:
                # deterministic unit vector for empty text (keeps cosine defined)
                idx, sign = self._hash_index("__empty__", self.dim)
                vector[idx] = 1.0 * sign
                out.append(vector)
                continue
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                weight = (1.0 + math.log(count)) * self.idf.get(token, self.default_idf)
                idx, sign = self._hash_index(token, self.dim)
                vector[idx] += weight * sign
            norm = math.sqrt(sum(v * v for v in vector))
            if norm < 1e-12:
                idx, sign = self._hash_index(text[:64], self.dim)
                vector[idx] = 1.0 * sign
                norm = 1.0
            out.append([v / norm for v in vector])
        return out

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {"dim": self.dim, "idf": self.idf, "default_idf": self.default_idf, "fitted": self.fitted}

    def load_state(self, state: dict) -> "HashingEmbedder":
        self.dim = int(state.get("dim", self.dim))
        self.idf = dict(state.get("idf") or {})
        self.default_idf = float(state.get("default_idf", self.default_idf))
        self.fitted = bool(state.get("fitted", False))
        return self

    def health(self) -> dict:
        return {
            "backend": self.name,
            "dim": self.dim,
            "available": True,
            "fitted": self.fitted,
            "vocabulary": len(self.idf),
        }


class SentenceTransformerEmbedder(Embedder):
    """Optional backend; requires ``sentence-transformers`` to be installed."""

    name = "sentence_transformer"

    def __init__(self, dim: int = 384, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise EmbeddingError(
                    "sentence-transformers is not installed; use embedding_backend='hashing'"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - network/model failure
                raise EmbeddingError(f"could not load '{self.model_name}': {exc}") from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - optional
        model = self._ensure()
        vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        self.dim = int(vectors.shape[1])
        return [[float(v) for v in row] for row in vectors]

    def health(self) -> dict:  # pragma: no cover - optional
        try:
            self._ensure()
            return {"backend": self.name, "dim": self.dim, "available": True, "model": self.model_name}
        except Exception as exc:
            return {"backend": self.name, "dim": self.dim, "available": False, "error": str(exc)}