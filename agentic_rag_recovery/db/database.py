"""SQLite database layer.

A single :class:`Database` object owns the connection factory and schema
bootstrap.  Connections are per-thread (SQLite objects are not shareable),
which is required for FastAPI + Streamlit usage.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..settings import PROJECT_ROOT

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Database:
    """Thread-safe SQLite wrapper with a small query helper API."""

    def __init__(self, path: Optional[str | Path] = None, create_schema: bool = True):
        self.path = Path(path) if path else PROJECT_ROOT / "storage" / "agentic_rag.db"
        self._local = threading.local()
        self._write_lock = threading.RLock()
        if create_schema:
            self.initialise()

    # -- connection ------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def initialise(self) -> "Database":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        script = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._write_lock:
            conn = self.connect()
            conn.executescript(script)
            conn.commit()
        return self

    # -- helpers ---------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute(self, sql: str, params: Sequence[Any] | Dict[str, Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            conn = self.connect()
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

    def execute_many(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self._write_lock:
            conn = self.connect()
            cur = conn.executemany(sql, rows)
            conn.commit()
            return cur.rowcount

    def insert(self, table: str, record: Dict[str, Any]) -> str:
        """Insert a dict; ``_id`` supplies the value of the table's PK column."""
        record = dict(record)
        pk_value = record.pop("_id", None)
        if pk_value is not None:
            pk_column = self.primary_key_of(table)
            if pk_column:
                record[pk_column] = pk_value
        columns = list(record.keys())
        sql = (
            f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})"
        )
        self.execute(sql, [_encode(record[c]) for c in columns])
        return str(pk_value) if pk_value is not None else ""

    def primary_key_of(self, table: str) -> Optional[str]:
        cached = getattr(self._local, "_pk_cache", None)
        if cached is None:
            cached = {}
            self._local._pk_cache = cached
        if table in cached:
            return cached[table]
        column: Optional[str] = None
        for row in self.fetch_all(f"PRAGMA table_info({table})"):
            if int(row.get("pk") or 0) > 0:
                column = row["name"]
                break
        cached[table] = column
        return column

    def update(self, table: str, pk_column: str, pk_value: Any, updates: Dict[str, Any]) -> int:
        if not updates:
            return 0
        sets = ",".join(f"{k}=?" for k in updates)
        params = [_encode(v) for v in updates.values()] + [pk_value]
        cur = self.execute(f"UPDATE {table} SET {sets} WHERE {pk_column}=?", params)
        return cur.rowcount

    def fetch_one(self, sql: str, params: Sequence[Any] | Dict[str, Any] = ()) -> Optional[Dict[str, Any]]:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: Sequence[Any] | Dict[str, Any] = ()) -> List[Dict[str, Any]]:
        cur = self.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fetch_value(self, sql: str, params: Sequence[Any] | Dict[str, Any] = (), default: Any = None) -> Any:
        cur = self.execute(sql, params)
        row = cur.fetchone()
        if not row:
            return default
        return row[0] if row[0] is not None else default

    def count(self, table: str, where: str = "", params: Sequence[Any] = ()) -> int:
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.fetch_value(sql, params, 0) or 0)

    def table_exists(self, table: str) -> bool:
        return bool(
            self.fetch_value(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,), None
            )
        )

    def tables(self) -> List[str]:
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r["name"] for r in rows]

    def vacuum(self) -> None:
        with self._write_lock:
            self.connect().execute("VACUUM")

    def reset(self) -> None:
        """Drop and recreate every table (used by scripts/tests)."""
        names = [name for name in self.tables() if not name.startswith("sqlite_")]
        if names:
            with self._write_lock:
                conn = self.connect()
                conn.execute("PRAGMA foreign_keys=OFF")
                for name in names:
                    conn.execute(f"DROP TABLE IF EXISTS \"{name}\"")
                conn.commit()
                conn.execute("PRAGMA foreign_keys=ON")
        self.initialise()


def _encode(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return value


def decode(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def decode_row(row: Optional[Dict[str, Any]], json_fields: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out = dict(row)
    for field in json_fields:
        if field in out:
            out[field] = decode(out[field])
    return out