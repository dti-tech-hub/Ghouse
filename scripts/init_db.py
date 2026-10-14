#!/usr/bin/env python
"""Initialise (or reset) the SQLite database and register provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_recovery.bootstrap import create_application  # noqa: E402
from agentic_rag_recovery.settings import get_settings  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop every table first")
    parser.add_argument("--db", default="", help="override the database path")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.db:
        from pathlib import Path as _Path

        from agentic_rag_recovery.settings import Settings

        settings = Settings.load()
        settings.paths.db_path = args.db
        settings.ensure_dirs()

    settings.ensure_dirs()
    app = create_application(settings=settings, auto_bootstrap=False)
    if args.reset:
        app.database.reset()
        print(f"reset {settings.db_path()}")

    tables = app.database.tables()
    print(f"database : {settings.db_path()}")
    print(f"tables   : {len(tables)}")
    for table in tables:
        count = app.database.count(table)
        print(f"  {table:<22} {count:>8} rows")
    print("\nNext: python scripts/ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())