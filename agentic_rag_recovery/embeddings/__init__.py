"""Embedding backends.

The default backend is a deterministic, dependency-free **hashed TF-IDF
projector**: tokens are hashed into a fixed vector space and weighted by
inverse document frequency.  It is a genuine bag-of-words embedding (not a
random stub), so it produces meaningful similarity structure offline and
reproduces exactly across runs.
"""

from .base import Embedder, EmbeddingError
from .factory import build_embedder

__all__ = ["Embedder", "EmbeddingError", "build_embedder"]