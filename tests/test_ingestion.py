"""Document ingestion, cleaning and chunking."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.ingestion.chunker import Chunker, split_with_metadata
from agentic_rag_recovery.ingestion.cleaner import clean_text
from agentic_rag_recovery.ingestion.loaders import load_directory, load_file, supported_extensions


def test_supported_extensions_cover_required_formats():
    extensions = set(supported_extensions())
    for required in {".pdf", ".txt", ".md", ".docx"}:
        assert required in extensions


def test_clean_text_collapses_whitespace_and_keeps_markdown():
    cleaned = clean_text("Title\n\n\n\nBody   with   spaces  .\n- bullet")
    assert "\n\n\n\n" not in cleaned
    assert "Body with spaces." in cleaned
    assert "- bullet" in cleaned


def test_clean_text_removes_control_characters():
    assert "\x00" not in clean_text("a\x00b")


def test_split_with_metadata_respects_chunk_size():
    text = " ".join(f"Sentence number {i} about escalation policy." for i in range(40))
    pieces = split_with_metadata(text, chunk_size=200, chunk_overlap=40)
    assert len(pieces) > 1
    for content, start, end in pieces:
        assert 0 <= start <= end <= len(text) + 1
        assert content.strip()


def test_split_with_metadata_respects_offsets():
    text = "Alpha section text. " * 30
    pieces = split_with_metadata(text, chunk_size=120, chunk_overlap=20)
    for content, _start, _end in pieces:
        assert content[:20] in text


def test_chunk_produces_metadata(chunker: Chunker):
    chunks = chunker.chunk_document(
        "doc_1", "source.md", "# Heading\n\nBody text that is long enough to matter."
    )
    assert chunks
    first = chunks[0]
    assert first.document_id == "doc_1"
    assert first.source == "source.md"
    assert first.chunk_index == 0
    assert first.token_estimate > 0
    assert first.created_at


def test_chunking_is_configurable(settings):
    from agentic_rag_recovery.settings import ChunkingConfig

    small = Chunker(ChunkingConfig(chunk_size=80, chunk_overlap=10))
    large = Chunker(ChunkingConfig(chunk_size=600, chunk_overlap=50))
    text = " ".join(f"Fact number {i} about vendor audit intervals." for i in range(60))
    assert len(small.split(text)) > len(large.split(text))


def test_load_markdown_file(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nSome content.", encoding="utf-8")
    document = load_file(path)
    assert document.doc_type == "markdown"
    assert document.title == "Title"
    assert "Some content." in document.content


def test_load_txt_and_json(tmp_path):
    (tmp_path / "a.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "b.json").write_text('[{"title": "T", "text": "body"}]', encoding="utf-8")
    assert load_file(tmp_path / "a.txt").content == "plain text"
    assert "body" in load_file(tmp_path / "b.json").content


def test_load_unsupported_type_raises(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        load_file(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.md")


def test_load_directory(tmp_path):
    from pathlib import Path

    (tmp_path / "one.md").write_text("# One\n\nalpha", encoding="utf-8")
    (tmp_path / "two.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    documents = load_directory(tmp_path)
    assert {Path(d.source).name for d in documents} == {"one.md", "two.txt"}


def test_ingestion_is_idempotent(app):
    from agentic_rag_recovery.bootstrap import bootstrap_demo_data

    before = app.repository.document_count()
    result = bootstrap_demo_data(app, force=False)
    assert app.repository.document_count() == before
    assert result.skipped >= 1 or result.document_count == 0


def test_ingest_text_roundtrip(app):
    document, chunks = app.ingestor.ingest_text(
        "Escalation to the incident commander must happen within 5 minutes.",
        source="unit-test",
        title="Unit test note",
    )
    try:
        assert document.document_id
        assert chunks
        assert app.repository.get_document(document.document_id) is not None
    finally:
        app.ingestor.purge_document(document.document_id)