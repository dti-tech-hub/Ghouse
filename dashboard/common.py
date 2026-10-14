"""Shared page scaffolding: connection handling and session helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.api_client import DEFAULT_BASE_URL, ApiClient  # noqa: E402

_PAGE_CACHE: Dict[str, ApiClient] = {}


def client() -> ApiClient:
    """Return a cached client for the configured base URL."""
    base_url = st.session_state.get("base_url") or DEFAULT_BASE_URL
    if base_url not in _PAGE_CACHE:
        _PAGE_CACHE[base_url] = ApiClient(base_url)
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = base_url
    return _PAGE_CACHE[base_url]


def init_state() -> None:
    for key, value in {
        "base_url": DEFAULT_BASE_URL,
        "last_query_response": None,
        "last_trace": None,
        "last_experiment": None,
        "last_injection": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar() -> None:
    st.sidebar.title("🛠️ Agentic RAG")
    st.sidebar.caption("Failure Diagnosis & Recovery")
    st.sidebar.text_input("API base URL", key="base_url")
    if st.sidebar.button("Check connection", use_container_width=True):
        try:
            health = client().health()
            st.sidebar.success(f"{health['provider']} · {health['chunks']} chunks")
        except Exception as exc:
            st.sidebar.error(str(exc).splitlines()[0])
    st.sidebar.divider()
    # The navigation bar itself is produced by st.navigation() in the
    # entrypoint (app.py). This fallback link list is only rendered when a
    # page is executed outside that context, and must never break a page.
    try:
        from dashboard.api_client import PAGES

        for page_file, label in PAGES:
            st.sidebar.page_link(f"pages/{page_file}.py", label=label)
    except Exception:
        pass
    st.sidebar.divider()
    st.sidebar.caption(
        "Demo mode needs no API key. Results produced with the deterministic "
        "mock provider are labelled DEMO / MOCK RESULTS."
    )


def scalar_text(value):
    """Render a heterogeneous verification value as a plain string.

    A verification criterion's ``observed``/``required`` pair is deliberately
    polymorphic - ``0.62``, ``"< 0.62"``, ``False``, ``"[60, 3000]"`` - so a
    column of them defeats pandas' dtype inference and pyarrow raises
    ``ArrowInvalid`` part-way through rendering the table.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(scalar_text(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {scalar_text(v)}" for k, v in value.items()) + "}"
    if value is None:
        return "-"
    return str(value)


def table(records, columns=None):
    """st.dataframe with every cell coerced to text, for mixed-type payloads."""
    rows = []
    for record in records:
        source = columns or record.keys()
        rows.append({column: scalar_text(record.get(column)) for column in source})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    return rows


def build_pages():
    """Build the st.Page objects used by st.navigation() in the entrypoint."""
    from dashboard.api_client import PAGES

    icons = ["🛠️", "📚", "🎮", "🔍", "🚨", "🧠", "🛠", "🧪", "📈", "💉", "🧾", "📤"]
    pages = []
    for index, (page_file, title) in enumerate(PAGES):
        pages.append(
            st.Page(
                f"pages/{page_file}.py",
                title=title,
                icon=icons[index] if index < len(icons) else "📄",
                default=index == 0,
            )
        )
    return pages


def guard(fn):
    """Run a page body, rendering connection/API errors instead of a traceback."""

    def wrapper(*args, **kwargs):
        init_state()
        sidebar()
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            st.error(str(exc))
            st.code("uvicorn backend.main:app --reload", language="bash")
        except Exception as exc:
            st.exception(exc)
        return None

    wrapper.__name__ = fn.__name__
    return wrapper


def page(title: str, icon: str = "") -> None:
    st.title(f"{icon} {title}".strip())


def label_banner() -> None:
    try:
        health = client().health()
        label = health.get("labels", {}).get("results", "")
        if "MOCK" in label.upper():
            st.warning(f"⚠️ **{label}** — figures on this page come from the deterministic mock provider.", icon="⚠️")
    except Exception:
        pass


def json_block(payload: Any, language: str = "json") -> None:
    import json

    st.code(json.dumps(payload, indent=2, default=str), language=language)