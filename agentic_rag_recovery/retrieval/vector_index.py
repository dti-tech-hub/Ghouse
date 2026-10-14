"""Vector index (FAISS with a pure-NumPy fallback)."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..embeddings.base import Embedder, EmbeddingError
from ..models import Chunk


class VectorIndex:
    """Dense index over chunk embeddings.

    Backends:
      * ``faiss``  - ``IndexFlatIP`` on L2-normalised vectors (cosine).
      * ``numpy``  - exact brute-force matmul (identical semantics, slower for
        very large corpora, but dependency-free and always available).
    """

    def __init__(self, embedder: Embedder, backend: str = "auto"):
        self.embedder = embedder
        self.dim = embedder.dim
        self.chunk_ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._faiss_index = None
        self._backend = "numpy"
        self.backend_requested = backend
        if backend in {"faiss", "auto"}:
            try:  # pragma: no cover - depends on environment
                import faiss  # type: ignore

                self._faiss = faiss
                self._backend = "faiss"
            except Exception:
                self._faiss = None
                self._backend = "numpy"
                if backend == "faiss":
                    raise EmbeddingError("faiss requested but not installed; use vector_backend='numpy'")
        self._ready = False

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def build(self, chunks: Sequence[Chunk], embedder: Optional[Embedder] = None) -> "VectorIndex":
        if embedder is not None:
            self.embedder = embedder
            self.dim = embedder.dim
        if not chunks:
            self.chunk_ids = []
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            self._faiss_index = None
            self._ready = True
            return self
        texts = [c.content for c in chunks]
        vectors = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise EmbeddingError(
                f"embedding dimension mismatch: got {vectors.shape}, expected {(self.dim,)}"
            )
        if self._backend == "faiss":
            index = self._faiss.IndexFlatIP(self.dim)
            index.add(np.ascontiguousarray(vectors))
            self._faiss_index = index
        self._vectors = vectors
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._ready = True
        return self

    def add(self, chunks: Sequence[Chunk]) -> "VectorIndex":
        """Incremental add (appends to an existing index)."""
        if not chunks:
            return self
        new_vectors = np.asarray(self.embedder.embed([c.content for c in chunks]), dtype=np.float32)
        new_ids = [c.chunk_id for c in chunks]
        if self._vectors is None or self._vectors.size == 0:
            return self.build(list(chunks))
        self._vectors = np.vstack([self._vectors, new_vectors])
        self.chunk_ids.extend(new_ids)
        if self._backend == "faiss":
            self._faiss_index.reset()
            self._faiss_index.add(np.ascontiguousarray(self._vectors))
        self._ready = True
        return self

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.size == 0:
            return []
        qvec = np.asarray(self.embedder.embed([query]), dtype=np.float32)
        if qvec.shape[1] != self.dim:
            raise EmbeddingError(
                f"query embedding dimension {qvec.shape[1]} does not match index dimension {self.dim}"
            )
        if self._backend == "faiss" and self._faiss_index is not None:
            scores, ids = self._faiss_index.search(np.ascontiguousarray(qvec), min(top_k, self.size))
            return [
                (self.chunk_ids[int(i)], float(s))
                for s, i in zip(scores[0], ids[0])
                if int(i) >= 0
            ]
        assert self._vectors is not None
        sims = self._vectors @ qvec[0]
        k = min(top_k, self.size)
        top = np.argpartition(-sims, k - 1)[:k] if self.size > k else np.arange(self.size)
        top = top[np.argsort(-sims[top])]
        return [(self.chunk_ids[int(i)], float(sims[int(i)])) for i in top]

    def similarity_matrix(self, queries: Sequence[str], chunk_ids: Sequence[str]) -> np.ndarray:
        """Raw cosine similarity matrix (diagnostics / ground-truth evaluation)."""
        if self.size == 0 or not chunk_ids:
            return np.zeros((len(queries), 0), dtype=np.float32)
        q = np.asarray(self.embedder.embed(list(queries)), dtype=np.float32)
        index_of = {cid: i for i, cid in enumerate(self.chunk_ids)}
        rows = np.asarray([index_of.get(cid, -1) for cid in chunk_ids], dtype=np.int64)
        valid = rows >= 0
        out = np.zeros((len(queries), len(chunk_ids)), dtype=np.float32)
        if not valid.any() or self._vectors is None:
            return out
        subset = self._vectors[rows[valid]]
        out[:, valid] = q @ subset.T
        return out

    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        import json
        from pathlib import Path

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "vectors.npy", self._vectors if self._vectors is not None else np.zeros((0, self.dim), np.float32))
        (root / "chunk_ids.json").write_text(json.dumps(self.chunk_ids), encoding="utf-8")
        (root / "meta.json").write_text(
            json.dumps({"backend": self._backend, "dim": self.dim, "size": self.size}, indent=2), encoding="utf-8"
        )
        state = getattr(self.embedder, "to_state", None)
        if callable(state):
            (root / "embedder.json").write_text(json.dumps(state(), default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str, embedder: Embedder) -> "VectorIndex":
        import json
        from pathlib import Path

        root = Path(directory)
        vectors = np.load(root / "vectors.npy")
        chunk_ids = json.loads((root / "chunk_ids.json").read_text(encoding="utf-8"))
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        state_path = root / "embedder.json"
        loader = getattr(embedder, "load_state", None)
        if state_path.exists() and callable(loader):
            loader(json.loads(state_path.read_text(encoding="utf-8")))
        index = cls(embedder, backend=meta.get("backend", "numpy"))
        index._vectors = vectors
        index.chunk_ids = chunk_ids
        index.dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else int(meta.get("dim", embedder.dim))
        if index._backend == "faiss":
            faiss_index = index._faiss.IndexFlatIP(index.dim)
            faiss_index.add(np.ascontiguousarray(vectors))
            index._faiss_index = faiss_index
        index._ready = True
        return index

    def health(self) -> dict:
        return {
            "backend": self._backend,
            "size": self.size,
            "dim": self.dim,
            "embedding": self.embedder.health(),
            "ready": self._ready,
        }