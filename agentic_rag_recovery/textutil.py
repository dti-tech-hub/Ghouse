"""Deterministic text utilities shared by the mock provider, signal layer
and the custom evaluation metrics.

Everything here is pure, dependency-free and reproducible - that is a hard
requirement because several metrics must produce the same number twice.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves", "please", "tell", "give", "explain", "describe", "does",
    "do", "did", "is", "are", "was", "were", "there", "known", "used", "using", "per",
}

# Lightweight, fully deterministic stemmer.
#
# It is not Porter: it combines ordinary suffix stripping (so "charged" and
# "charge" agree) with a fixed-width prefix fold (so "cancellation" and
# "cancel" agree).  The fold is what makes lexical coverage - and therefore
# the retrieval metrics - robust to the morphological gaps that cause
# real-world low-recall failures.
SUFFIX_RULES: Tuple[Tuple[str, str], ...] = (
    ("ies", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

#: suffixes that must not be stripped when they are the whole word ending
KEEP_SUFFIXES = ("ss", "us", "is", "as", "os", "ics")

PREFIX_FOLD_LENGTH = 5


def _strip_suffix(word: str) -> str:
    for suffix, replacement in SUFFIX_RULES:
        if not word.endswith(suffix) or len(word) <= len(suffix) + 1:
            continue
        if suffix in ("s", "es") and word.endswith(KEEP_SUFFIXES):
            continue
        return word[: -len(suffix)] + replacement
    return word


def stem(word: str, _depth: int = 0) -> str:
    w = (word or "").lower().strip("'-")
    if not w:
        return ""
    if len(w) > 8 and (w.endswith("ation") or w.endswith("ations")):
        w = w[:-4] if w.endswith("ations") else w[:-3]
    w = _strip_suffix(w)
    if _depth == 0 and len(w) > len(w.rstrip("e")) + 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > PREFIX_FOLD_LENGTH:
        w = w[:PREFIX_FOLD_LENGTH]
    return w


def tokenize(text: str, remove_stopwords: bool = True, do_stem: bool = True) -> List[str]:
    if not text:
        return []
    tokens = [m.group(0).lower() for m in WORD_RE.finditer(text)]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if do_stem:
        tokens = [stem(t) for t in tokens]
    # keep numbers of any length: "5 minutes" must not lose its "5"
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\s{2,}(?=[A-Z0-9])")
_BLOCK_START_RE = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||>|\[page:)")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Hard-wrapped source lines are re-joined first (otherwise a line break in
    the middle of a sentence would look like a sentence boundary and would
    corrupt both the groundedness and the conflict metrics).  Markdown block
    markers (headings, bullets, table rows, page markers) stay separate.
    """
    if not text:
        return []
    parts: List[str] = []
    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        buffer = ""
        for line in block.split("\n"):
            stripped = line.strip()
            if not stripped:
                if buffer:
                    parts.append(buffer)
                    buffer = ""
                continue
            if _BLOCK_START_RE.match(line):
                if buffer:
                    parts.append(buffer)
                    buffer = ""
                parts.append(stripped)
                continue
            if buffer and not re.search(r"[.!?:\-]$", buffer) and not stripped.startswith(("-", "*")):
                buffer = f"{buffer} {stripped}"
            else:
                if buffer:
                    parts.append(buffer)
                buffer = stripped
        if buffer:
            parts.append(buffer)
    sentences: List[str] = []
    for part in parts:
        for sentence in SENTENCE_RE.split(part):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def token_count(text: str) -> int:
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_coefficient(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def lexical_overlap(a: str, b: str) -> float:
    """Normalised [0,1] overlap between two strings based on stemmed tokens."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return jaccard(ta, tb)


def coverage_of_terms(text: str, terms: Sequence[str]) -> float:
    """Fraction of ``terms`` present in ``text``."""
    if not terms:
        return 1.0
    tokens = set(tokenize(text))
    hits = sum(1 for t in terms if stem(t.lower()) in tokens)
    return hits / len(terms)


def numeric_values(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""))


def capitalized_terms(text: str) -> Set[str]:
    return {m.group(0) for m in re.finditer(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", text or "")}


NEGATION_WORDS = {
    "not", "no", "never", "none", "cannot", "n't", "without", "neither", "nor", "unable",
    "fails", "failed", "disables", "disabled", "prohibited", "forbidden",
}


def has_negation(text: str) -> bool:
    toks = set(re.findall(r"[A-Za-z']+", (text or "").lower()))
    return bool(toks & NEGATION_WORDS)


def stable_int(*parts: object, modulo: int = 1000) -> int:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:12], 16) % max(1, modulo)


def stable_choice(options: Sequence[str], *parts: object) -> str:
    if not options:
        return ""
    return options[stable_int(*parts, modulo=len(options))]


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def clip(text: str, limit: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def dedupe_preserving_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def char_ngrams(text: str, n: int = 3) -> List[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def qa_pair_map(pairs: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    return {a.lower(): b for a, b in pairs}


def mean(values: Sequence[float]) -> float:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return float(vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2)


def stddev(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def normalise_scores(scores: Sequence[float]) -> List[float]:
    """Min-max normalise to [0,1]; a flat list becomes all ones when equal & >0."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def apply_softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(1e-6, temperature)
    exps = [math.exp((s - max(scores)) / t) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None