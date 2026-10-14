"""BM25 sparse index.

Uses ``rank_bm25`` when available and otherwise falls back to an equivalent
in-house Okapi BM25 implementation so the project never hard-depends on it.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Chunk
from ..textutil import tokenize


class BM25Index:
    """Okapi BM25 with k1=1.5, b=0.75."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._backend = "builtin"

    # ------------------------------------------------------------------
    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_tokens = [tokenize(c.content) for c in chunks]
        self._finalise()
        return self

    def _finalise(self) -> None:
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df = {}
        for counter in self.doc_freqs:
            for token in counter:
                self.df[token] = self.df.get(token, 0) + 1
        n = max(1, len(self.doc_tokens))
        self.idf = {
            token: max(0.01, math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)))
            for token, freq in self.df.items()
        }
        self._backend = "builtin"

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    def scores(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return [0.0] * len(self.chunk_ids)
        scores = []
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i] or 1
            score = 0.0
            for token in q_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                idf = self.idf.get(token, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: max(1, top_k)]
        return [(self.chunk_ids[i], scores[i]) for i in order]

    def save(self, directory: str) -> None:
        import json
        from pathlib import Path

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "bm25.json").write_text(
            json.dumps({"chunk_ids": self.chunk_ids, "df": self.df, "k1": self.k1, "b": self.b}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, chunks: Sequence[Chunk], directory: str = "") -> "BM25Index":
        return cls().build(list(chunks))

    def health(self) -> dict:
        return {"backend": self._backend, "size": self.size, "vocabulary": len(self.df)}