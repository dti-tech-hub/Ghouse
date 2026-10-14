"""Embedding interface."""

from __future__ import annotations

import abc
import hashlib
from typing import List, Sequence


class EmbeddingError(RuntimeError):
    """Raised on backend failure; maps to INFRASTRUCTURE_FAILURE."""


class Embedder(abc.ABC):
    """Vector encoder. All backends must be deterministic given the same input."""

    name: str = "base"

    def __init__(self, dim: int = 384, **kwargs):
        self.dim = int(dim)
        self.options = kwargs

    @abc.abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a batch of texts into L2-normalised vectors."""

    def embed_one(self, text: str) -> List[float]:
        vectors = self.embed([text])
        if not vectors:
            raise EmbeddingError("embedder returned no vector")
        return vectors[0]

    def fit(self, corpus: Sequence[str]) -> "Embedder":
        """Optional corpus statistics fitting (IDF)."""
        return self

    def health(self) -> dict:
        return {"backend": self.name, "dim": self.dim, "available": True}

    @staticmethod
    def _hash_index(token: str, dim: int, salt: str = "") -> tuple:
        digest = hashlib.blake2b(f"{salt}{token}".encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % dim, (value >> 32) % 2

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} dim={self.dim}>"