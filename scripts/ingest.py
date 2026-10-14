#!/usr/bin/env python
"""Ingest documents and build the vector + BM25 indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import bootstrap_demo_data, create_application  # noqa: E402
from agentic_rag_recovery.ingestion.ingest import ingest_path  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="", help="file or directory (default: demo corpus)")
    parser.add_argument("--force", action="store_true", help="re-ingest unchanged documents")
    parser.add_argument("--text", default="", help="ingest a literal string instead of a path")
    parser.add_argument("--source", default="manual", help="source name for --text")
    args = parser.parse_args(argv)

    app = create_application(auto_bootstrap=False)
    if args.text:
        document, chunks = app.ingestor.ingest_text(args.text, source=args.source, title=args.source)
        app.orchestrator.reindex()
        print(f"ingested {document.document_id} -> {len(chunks)} chunk(s)")
        return 0

    target = args.path or app.settings.paths.demo_documents
    result = bootstrap_demo_data(app, target, force=args.force)
    status = app.rebuild_indexes()
    print(f"documents : {result.document_count} (skipped {result.skipped})")
    print(f"chunks    : {result.chunk_count}")
    print(f"index     : {status}")
    for error in result.errors:
        print(f"  ERROR {error}")
    return 1 if result.errors and not result.document_count else 0


if __name__ == "__main__":
    raise SystemExit(main())