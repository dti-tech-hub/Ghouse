"""Configurable chunking.

The configuration (size / overlap / separators) is a first-class, *runtime*
knob because chunk fragmentation is one of the injected failure scenarios: a
recovery action may re-chunk with a larger window and re-index.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import Chunk
from ..settings import ChunkingConfig, get_settings
from ..textutil import estimate_tokens


def split_with_metadata(
    text: str,
    chunk_size: int = 320,
    chunk_overlap: int = 60,
    separators: Optional[Sequence[str]] = None,
    min_chunk_chars: int = 30,
) -> List[Tuple[str, int, int]]:
    """Split ``text`` into ``(chunk, char_start, char_end)`` triples.

    Uses a recursive character splitter: the highest-priority separator that
    appears in the window wins, matching LangChain's approach but implemented
    locally so no optional dependency is required.
    """
    if not text or not text.strip():
        return []
    seps: List[str] = list(separators if separators is not None else ["\n\n", "\n", ". ", " ", ""])
    if not seps:
        seps = [" "]
    overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))

    raw = _recursive_split(text, seps, chunk_size)
    merged = _merge_with_overlap(raw, chunk_size, overlap, min_chunk_chars)
    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for piece in merged:
        content = piece["text"].strip()
        if not content:
            continue
        index = text.find(content, cursor)
        if index < 0:
            index = text.find(content)
        if index < 0:
            index = max(0, min(cursor, len(text)))
        out.append((content, index, min(len(text), index + len(content))))
        cursor = index + max(1, len(content) - overlap)
    if not out:
        stripped = text.strip()
        out = [(stripped, 0, len(stripped))]
    return out


def _recursive_split(text: str, seps: Sequence[str], chunk_size: int) -> List[Dict[str, Any]]:
    return [{"text": t, "start": s} for t, s in _split_recursive(text, 0, list(seps) or [" "], chunk_size)]


def _split_recursive(
    text: str, base_offset: int, seps: Sequence[str], chunk_size: int
) -> List[Tuple[str, int]]:
    """Split ``text`` honouring separator priority while tracking char offsets."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [(text, base_offset)]
    for i, sep in enumerate(seps):
        if not sep or sep not in text:
            continue
        rest = list(seps[i + 1 :]) or [" "]
        out: List[Tuple[str, int]] = []
        pos = 0
        for part in text.split(sep):
            if part.strip():
                # keep the separator so punctuation and markdown survive chunking
                out.extend(_split_recursive(part + sep, base_offset + pos, rest, chunk_size))
            pos += len(part) + len(sep)
        return out
    # no usable separator: fixed-width windows
    return [
        (text[i : i + chunk_size], base_offset + i) for i in range(0, len(text), chunk_size)
    ]


def _merge_with_overlap(
    pieces: Sequence[Dict[str, Any]], chunk_size: int, overlap: int, min_chunk_chars: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    current = ""
    current_start: Optional[int] = None
    for piece in pieces:
        candidate = f"{current}\n{piece['text']}" if current else piece["text"]
        if len(candidate) <= chunk_size or not current:
            if current_start is None:
                current_start = piece["start"]
            current = candidate
        else:
            out.append({"text": current.strip(), "start": current_start or 0})
            if overlap > 0:
                tail = current[-overlap:]
                new_start = max(current_start or 0, (current_start or 0) + max(1, len(current) - overlap))
                current = f"{tail}\n{piece['text']}".strip()
                current_start = new_start
            else:
                current_start = piece["start"]
                current = piece["text"]
    if current.strip():
        out.append({"text": current.strip(), "start": current_start or 0})
    # merge very short fragments forward so that context is not fragmented
    merged: List[Dict[str, Any]] = []
    for item in out:
        if merged and (len(item["text"]) < min_chunk_chars or len(merged[-1]["text"]) < min_chunk_chars):
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + item["text"]).strip()
        else:
            merged.append(dict(item))
    return merged


class Chunker:
    """Stateful chunker bound to a :class:`ChunkingConfig`."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or get_settings().chunking

    def split(self, text: str) -> List[Tuple[str, int, int]]:
        return split_with_metadata(
            text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            min_chunk_chars=self.config.min_chunk_chars,
        )

    def chunk_document(self, document_id: str, source: str, content: str, page_offset: int = 0) -> List[Chunk]:
        chunks: List[Chunk] = []
        pieces = self.split(content)
        for index, (piece, start, end) in enumerate(pieces):
            page = self._detect_page(piece, page_offset)
            section = self._detect_section(piece)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=index,
                    content=piece,
                    token_estimate=estimate_tokens(piece),
                    char_start=start,
                    char_end=end,
                    page=page,
                    section=section,
                    source=source,
                    metadata={"chars": len(piece), "chunk_size": self.config.chunk_size},
                )
            )
        return chunks

    @staticmethod
    def _detect_page(text: str, page_offset: int) -> Optional[int]:
        marker = text.find("[page:")
        if marker < 0:
            return page_offset or None
        end = text.find("]", marker)
        try:
            return int(text[marker + 6 : end])
        except ValueError:
            return page_offset or None

    @staticmethod
    def _detect_section(text: str) -> Optional[str]:
        for line in text.split("\n")[:3]:
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()[:80]
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "separators": list(self.config.separators),
            "min_chunk_chars": self.config.min_chunk_chars,
        }