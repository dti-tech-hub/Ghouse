"""Embedding factory."""

from __future__ import annotations

from typing import Any, Optional

from ..settings import Settings, get_settings
from .base import Embedder, EmbeddingError
from .hashing import HashingEmbedder, SentenceTransformerEmbedder

BACKENDS = ("hashing", "sentence_transformer")


def build_embedder(
    backend: Optional[str] = None,
    dim: Optional[int] = None,
    settings: Optional[Settings] = None,
    **overrides: Any,
) -> Embedder:
    settings = settings or get_settings()
    name = (backend or settings.retrieval.embedding_backend or "hashing").lower()
    dimension = int(dim or settings.retrieval.embedding_dim)
    if name in {"hashing", "hash", "tfidf", "local"}:
        return HashingEmbedder(dim=dimension, **overrides)
    if name in {"sentence_transformer", "st", "minilm", "sbert"}:
        return SentenceTransformerEmbedder(
            dim=dimension, model_name=overrides.pop("model_name", "sentence-transformers/all-MiniLM-L6-v2"), **overrides
        )
    raise ValueError(f"unknown embedding backend '{name}'. Available: {', '.join(BACKENDS)}")