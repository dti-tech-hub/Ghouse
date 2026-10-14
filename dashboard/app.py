"""Streamlit entry point.

Run with::

    streamlit run dashboard/app.py

The dashboard is a *thin client* of the FastAPI backend, so the system can be
demonstrated as two processes (which is the realistic deployment) or against a
remote instance.

This file only declares the navigation (``st.Page`` + ``st.navigation``). The
pages themselves live in ``dashboard/pages/`` and each one renders the sidebar
(API URL box, connection check) through ``dashboard.common.sidebar()``.

Opening ``app.py`` directly lands on the default page, System Overview.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.common import build_pages  # noqa: E402

st.set_page_config(
    page_title="Agentic RAG - Failure Diagnosis & Recovery",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# default API endpoint; every page shows an editable box in the sidebar
if "base_url" not in st.session_state:
    st.session_state["base_url"] = os.environ.get("ARR_API_URL", "http://127.0.0.1:8000")

st.navigation(build_pages()).run()