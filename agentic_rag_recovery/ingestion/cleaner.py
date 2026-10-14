"""Text cleaning."""

from __future__ import annotations

import re
import unicodedata
from typing import List

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"[ ]{2,}")
_SOFT_HYPHEN = re.compile(r"\u00ad")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str, keep_markdown: bool = True) -> str:
    """Normalise unicode, drop control characters and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    lines: List[str] = []
    for line in text.split("\n"):
        if not keep_markdown and line.lstrip().startswith(("#", "*", "-", "|", ">")):
            line = re.sub(r"^[#>*\-\|\s]+", "", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def normalize_whitespace_for_index(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text.replace("\n", " ")).strip()


def sentence_case_fix(text: str) -> str:
    """Join lines that were hard-wrapped in the source document."""
    lines = [l.strip() for l in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if not line:
            out.append("")
            continue
        if out and out[-1] and not out[-1].endswith((".", "!", "?", ":", "-")) and not line.startswith(("#", "-", "*", "|")):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)