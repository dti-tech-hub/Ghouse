"""Shared helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

DEFAULT_BASE_URL = os.environ.get("ARR_API_URL", "http://127.0.0.1:8000")

PAGES = [
    ("1_System_Overview", "System Overview"),
    ("2_Knowledge_Base", "Knowledge Base"),
    ("3_Query_Playground", "Query Playground"),
    ("4_Agent_Trace", "Agent Trace"),
    ("5_Failure_Detection", "Failure Detection"),
    ("6_Diagnosis", "Diagnosis"),
    ("7_Recovery", "Recovery"),
    ("8_Experiments", "Experiments"),
    ("9_Metrics", "Metrics"),
    ("10_Fault_Injection", "Fault Injection"),
    ("11_Model_Comparison", "Model Comparison"),
    ("12_Export", "Export"),
]


class ApiClient:
    """Thin HTTP client; every method raises a readable error."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ConnectionError(
                f"cannot reach the API at {self.base_url}: {exc}\n"
                "Start it with:  uvicorn backend.main:app --reload"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{method} {path} -> HTTP {response.status_code}: {detail}")
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def get(self, path: str, **kwargs: Any) -> Any:
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self._request("POST", path, **kwargs)

    # -- convenience wrappers ------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self.get("/health")

    def system(self) -> Dict[str, Any]:
        return self.get("/system")

    def config(self) -> Dict[str, Any]:
        return self.get("/config")

    def documents(self) -> List[Dict[str, Any]]:
        return self.get("/documents")

    def ingest(self, path: str, force: bool = False) -> Dict[str, Any]:
        return self.post("/documents/ingest", json={"path": path, "force": force})

    def query(self, **payload: Any) -> Dict[str, Any]:
        return self.post("/query", json=payload)

    def demo_question(self, question_id: str, fault: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"question_id": question_id}
        if fault:
            params["fault"] = fault
        return self.get("/query/demo", params=params)

    def faults(self) -> List[Dict[str, Any]]:
        return self.get("/faults")

    def inject(self, fault_type: str, parameters: Optional[Dict[str, Any]] = None, query: Optional[str] = None) -> Dict[str, Any]:
        return self.post(
            "/faults/inject",
            json={"fault_type": fault_type, "parameters": parameters or {}, "query": query},
        )

    def experiments(self) -> List[Dict[str, Any]]:
        return self.get("/experiments")

    def experiment(self, experiment_id: str) -> Dict[str, Any]:
        return self.get(f"/experiments/{experiment_id}")

    def experiment_configs(self) -> Dict[str, Any]:
        return self.get("/experiments/configs")

    def run_experiment(self, **payload: Any) -> Dict[str, Any]:
        return self.post("/experiments/run", json=payload)

    def run_experiments_batch(self, configs: List[str]) -> Dict[str, Any]:
        return self.post("/experiments/batch", json={"configs": configs})

    def traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.get("/traces", params={"limit": limit}).get("traces", [])

    def trace(self, run_id: str) -> Dict[str, Any]:
        return self.get(f"/traces/{run_id}")

    def diagnoses(self, limit: int = 200) -> List[Dict[str, Any]]:
        return self.get("/diagnoses", params={"limit": limit}).get("diagnoses", [])

    def diagnosis(self, diagnosis_id: str) -> Dict[str, Any]:
        return self.get(f"/diagnoses/{diagnosis_id}")

    def metrics(self) -> Dict[str, Any]:
        return self.get("/metrics")

    def per_run_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.get("/metrics/runs", params={"limit": limit}).get("metrics", [])

    def abstentions(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.get("/abstentions", params={"limit": limit}).get("abstentions", [])

    def datasets(self) -> Dict[str, Any]:
        return self.get("/datasets")


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def json_block(payload: Any, height: int = 320) -> None:
    import streamlit as st

    st.code(json.dumps(payload, indent=2, default=str), language="json")
    st.caption(f"(scrollable block, {height}px)")


def flatten_signals(signals: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for section, values in (signals or {}).items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            rows.append({"section": section, "signal": key, "value": value})
    return to_frame(rows)


def numeric_frame(metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    return to_frame(
        [
            {
                "run_id": m.get("run_id"),
                "name": m.get("name"),
                "value": m.get("value"),
                "group": m.get("group_name"),
                "label": m.get("label"),
            }
            for m in metrics
            if m.get("value") is not None
        ]
    )