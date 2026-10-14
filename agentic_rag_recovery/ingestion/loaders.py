"""Document loaders for PDF / TXT / Markdown / DOCX / HTML / JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cleaner import clean_text

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".json", ".csv", ".rst"}


@dataclass
class LoadedDocument:
    source: str
    doc_type: str
    content: str
    title: str = ""
    pages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def supported_extensions() -> List[str]:
    return sorted(SUPPORTED_EXTENSIONS)


def load_file(path: str | Path) -> LoadedDocument:
    """Load a single file into text. Raises ``ValueError`` for unknown types."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"document not found: {p}")
    if not p.is_file():
        raise ValueError(f"not a file: {p}")
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(p)
    if ext == ".docx":
        return _load_docx(p)
    if ext in {".html", ".htm"}:
        return _load_html(p)
    if ext == ".json":
        return _load_json(p)
    if ext == ".csv":
        return _load_csv(p)
    if ext in {".txt", ".md", ".markdown", ".rst"}:
        return _load_text(p, ext)
    raise ValueError(f"unsupported document type '{ext}' for {p.name}")


def load_directory(path: str | Path, recursive: bool = True) -> List[LoadedDocument]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"directory not found: {root}")
    pattern = "**/*" if recursive else "*"
    out: List[LoadedDocument] = []
    for candidate in sorted(root.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            if candidate.name.startswith((".", "~$")):
                continue
            out.append(load_file(candidate))
    return out


# --------------------------------------------------------------------------
def _load_text(path: Path, ext: str) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = _markdown_title(raw) or path.stem.replace("_", " ").replace("-", " ")
    return LoadedDocument(
        source=str(path),
        doc_type="markdown" if ext in {".md", ".markdown"} else "txt",
        content=clean_text(raw),
        title=title.strip()[:200],
        metadata={"bytes": path.stat().st_size, "filename": path.name},
    )


def _load_pdf(path: Path) -> LoadedDocument:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return _load_pdf_fallback(path)
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pragma: no cover - malformed pdf
        raise ValueError(f"could not read PDF '{path.name}': {exc}") from exc
    body_parts = []
    for number, text in enumerate(pages, start=1):
        cleaned = clean_text(text)
        if cleaned:
            body_parts.append(f"[page:{number}]\n{cleaned}")
    title = path.stem.replace("_", " ").replace("-", " ")
    try:
        meta_title = (reader.metadata or {}).get("/Title")
        if meta_title:
            title = str(meta_title)
    except Exception:
        pass
    return LoadedDocument(
        source=str(path),
        doc_type="pdf",
        content="\n\n".join(body_parts),
        title=title.strip()[:200],
        pages=pages,
        metadata={"pages": len(pages), "bytes": path.stat().st_size, "filename": path.name},
    )


def _load_pdf_fallback(path: Path) -> LoadedDocument:  # pragma: no cover - optional dependency
    """Very small pypdf-free fallback (extracts text between BT/ET markers)."""
    raw = path.read_bytes().decode("latin-1", errors="replace")
    strings = re.findall(r"\((?:\\.|[^()\\])*\)", raw)
    text = " ".join(s[1:-1] for s in strings)
    text = text.replace("\\(", "(").replace("\\)", ")").replace("\\n", " ")
    text = re.sub(r"\\(\d{3})", lambda m: chr(int(m.group(1), 8)), text)
    return LoadedDocument(
        source=str(path),
        doc_type="pdf",
        content=clean_text(text),
        title=path.stem,
        metadata={"filename": path.name, "fallback_extractor": True},
    )


def _load_docx(path: Path) -> LoadedDocument:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError(
            f"python-docx is required to read '{path.name}'. Install with: pip install python-docx"
        ) from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # pragma: no cover - malformed docx
        raise ValueError(f"could not read DOCX '{path.name}': {exc}") from exc
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    tables: List[str] = []
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                tables.append(" | ".join(cells))
    title = ""
    for paragraph in document.paragraphs:
        if paragraph.style and paragraph.style.name.lower().startswith("heading") and paragraph.text.strip():
            title = paragraph.text.strip()
            break
    body = "\n\n".join(paragraphs + tables)
    return LoadedDocument(
        source=str(path),
        doc_type="docx",
        content=clean_text(body),
        title=title or path.stem.replace("_", " ").replace("-", " "),
        metadata={"filename": path.name, "paragraphs": len(paragraphs), "tables": len(document.tables)},
    )


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _load_html(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = _SCRIPT_RE.sub(" ", raw)
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.DOTALL | re.IGNORECASE)
    title = clean_text(title_match.group(1)) if title_match else path.stem
    body = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", raw, flags=re.IGNORECASE)
    body = _TAG_RE.sub(" ", body)
    return LoadedDocument(
        source=str(path),
        doc_type="html",
        content=clean_text(body),
        title=title[:200],
        metadata={"filename": path.name},
    )


def _load_json(path: Path) -> LoadedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in '{path.name}': {exc}") from exc

    records: List[Any]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("documents"), list):
        records = data["documents"]
    else:
        records = [data]
    parts: List[str] = []
    for i, record in enumerate(records):
        if isinstance(record, dict):
            title = record.get("title") or record.get("id") or f"entry_{i+1}"
            body = record.get("text") or record.get("content") or record.get("body") or ""
            parts.append(f"# {title}\n{body}")
        else:
            parts.append(str(record))
    return LoadedDocument(
        source=str(path),
        doc_type="json",
        content=clean_text("\n\n".join(parts)),
        title=path.stem,
        metadata={"filename": path.name, "records": len(records)},
    )


def _load_csv(path: Path) -> LoadedDocument:
    import csv

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return LoadedDocument(source=str(path), doc_type="csv", content="", title=path.stem)
    header = rows[0]
    lines = [" | ".join(header)]
    for row in rows[1:]:
        lines.append(" | ".join(row))
    return LoadedDocument(
        source=str(path),
        doc_type="csv",
        content=clean_text("\n".join(lines)),
        title=path.stem,
        metadata={"filename": path.name, "rows": len(rows) - 1},
    )


def _markdown_title(text: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""