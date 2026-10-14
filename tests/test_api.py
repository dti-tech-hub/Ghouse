"""HTTP API: schema validation, safety and endpoint behaviour."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app as fastapi_app
from backend.services import get_app, reset_app  # noqa: F401


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(fastapi_app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------
def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["chunks"] >= 0
    assert "DEMO" in payload["labels"]["results"].upper()


def test_root_lists_endpoints(client: TestClient):
    payload = client.get("/").json()
    assert "/query" in payload["endpoints"]
    assert payload["openapi"] == "/openapi.json"


def test_openapi_is_generated(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"]
    for path in ("/health", "/query", "/experiments/run", "/faults/inject", "/metrics"):
        assert path in schema["paths"]


def test_config_exposes_taxonomy_and_budgets(client: TestClient):
    payload = client.get("/config").json()
    assert payload["taxonomy"]["stages"]
    assert payload["budgets"]["max_recovery_attempts"] >= 1
    assert len(payload["faults"]) >= 15


def test_taxonomy_endpoint(client: TestClient):
    taxonomy = client.get("/taxonomy").json()
    assert taxonomy["classes"]
    assert taxonomy["subtypes"]


def test_recovery_strategies_endpoint(client: TestClient):
    payload = client.get("/recovery/strategies").json()
    assert "QUERY_REWRITE" in payload["strategies"]
    assert "ABSTAIN" in payload["actions"]


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------
def test_documents_listed(client: TestClient):
    documents = client.get("/documents").json()
    assert isinstance(documents, list)
    if documents:
        assert "chunk_count" in documents[0]


def test_document_detail_and_404(client: TestClient):
    documents = client.get("/documents").json()
    if documents:
        detail = client.get(f"/documents/{documents[0]['document_id']}").json()
        assert "chunks" in detail
    assert client.get("/documents/does_not_exist").status_code == 404


def test_ingest_missing_path_returns_404(client: TestClient):
    response = client.post("/documents/ingest", json={"path": "/no/such/place"})
    assert response.status_code == 404


def test_ingest_text_creates_document(client: TestClient):
    response = client.post(
        "/documents/text",
        json={"text": "Escalation to the incident commander must happen within 5 minutes.", "source": "api-test"},
    )
    assert response.status_code == 200
    assert response.json()["chunk_count"] >= 1


def test_ingest_text_rejects_blank(client: TestClient):
    assert client.post("/documents/text", json={"text": "   "}).status_code == 422


def test_datasets_endpoint(client: TestClient):
    payload = client.get("/datasets").json()
    assert payload["demo"]["total"] > 0
    assert isinstance(payload["public"], list)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def test_query_returns_full_pipeline_payload(client: TestClient):
    response = client.post(
        "/query",
        json={"query": "Within how many minutes must a Severity 1 incident be acknowledged?"},
    )
    assert response.status_code == 200
    payload = response.json()
    for key in (
        "run_id",
        "answer",
        "understanding",
        "retrieved",
        "retrieval_diagnostics",
        "context",
        "generation",
        "signals",
        "diagnoses",
        "recoveries",
        "steps",
        "metrics",
    ):
        assert key in payload
    assert payload["retrieved"]


def test_query_with_fault_runs_recovery(client: TestClient):
    response = client.post(
        "/query",
        json={"query": "What is the minimum password length?", "faults": ["GENERATION_CONSTRAINT_REMOVAL"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnoses"]
    assert payload["diagnoses"][0]["failure_detected"]
    assert payload["recoveries"]


def test_query_recover_endpoint(client: TestClient):
    response = client.post(
        "/query/recover",
        json={"query": "How long are backups retained?", "recover": True},
    )
    assert response.status_code == 200


def test_query_mode2_never_recovers(client: TestClient):
    payload = client.post(
        "/query",
        json={
            "query": "What is the minimum password length?",
            "faults": ["GENERATION_CONSTRAINT_REMOVAL"],
            "mode": "mode2_agentic_no_recovery",
        },
    ).json()
    assert payload["recoveries"] == []


def test_query_demo_endpoint(client: TestClient):
    payload = client.get("/query/demo", params={"question_id": "q001"}).json()
    assert payload["answer"]
    assert client.get("/query/demo", params={"question_id": "nope"}).status_code == 404


# ---------------------------------------------------------------------------
# input validation / safety
# ---------------------------------------------------------------------------
def test_empty_query_rejected(client: TestClient):
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "   "}).status_code == 422


def test_unknown_mode_rejected(client: TestClient):
    assert client.post("/query", json={"query": "x", "mode": "nope"}).status_code == 422


def test_extra_fields_rejected(client: TestClient):
    assert client.post("/query", json={"query": "x", "evil": 1}).status_code == 422


def test_unknown_fault_returns_404(client: TestClient):
    response = client.post("/faults/inject", json={"fault_type": "NOT_A_FAULT"})
    assert response.status_code == 404


def test_fault_injection_rejects_non_object_parameters(client: TestClient):
    response = client.post(
        "/faults/inject", json={"fault_type": "LOW_TOP_K", "parameters": [1, 2, 3]}
    )
    assert response.status_code in (200, 422)


def test_code_like_input_is_treated_as_text(client: TestClient):
    """User input must never be executed; it is only ever stored as text."""
    payload = client.post(
        "/query",
        json={"query": "__import__('os').system('echo pwned')"},
    ).json()
    assert "__import__" in payload["query"]


def test_query_length_limit(client: TestClient):
    assert client.post("/query", json={"query": "x" * 5000}).status_code == 422


# ---------------------------------------------------------------------------
# faults
# ---------------------------------------------------------------------------
def test_fault_catalogue(client: TestClient):
    faults = client.get("/faults").json()
    assert len(faults) >= 15
    for fault in faults:
        assert fault["expected_failure_class"]
        assert fault["expected_subtype"]


def test_inject_fault_records_ground_truth(client: TestClient):
    payload = client.post("/faults/inject", json={"fault_type": "LOW_TOP_K"}).json()
    assert payload["fault_id"]
    assert payload["active"] is True
    assert payload["expected_subtype"]


def test_inject_fault_with_query_returns_run(client: TestClient):
    payload = client.post(
        "/faults/inject",
        json={"fault_type": "GENERATION_CONSTRAINT_REMOVAL", "query": "What is the minimum password length?"},
    ).json()
    assert payload["run_id"]
    assert payload["result"]["answer"]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def test_experiment_configs_listed(client: TestClient):
    payload = client.get("/experiments/configs").json()
    assert "baseline" in payload["available"]


def test_run_small_experiment(client: TestClient):
    response = client.post(
        "/experiments/run",
        json={
            "name": "api-test",
            "modes": ["mode3_agentic_with_recovery"],
            "fault_type": "GENERATION_CONSTRAINT_REMOVAL",
            "question_ids": ["q013"],
            "repetitions": 1,
            "seed": 42,
            "persist": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode3_agentic_with_recovery"]["total_runs"] == 1
    assert payload["label"]


def test_run_unknown_experiment_returns_404(client: TestClient):
    assert client.post("/experiments/run", json={"name": "does-not-exist"}).status_code == 404


def test_list_and_get_experiments(client: TestClient):
    experiments = client.get("/experiments").json()
    assert isinstance(experiments, list)
    if experiments:
        detail = client.get(f"/experiments/{experiments[0]['experiment_id']}")
        assert detail.status_code == 200
    assert client.get("/experiments/nope").status_code == 404


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------
def test_traces_and_trace_detail(client: TestClient):
    run_id = client.post("/query", json={"query": "What is the minimum password length?"}).json()["run_id"]
    traces = client.get("/traces").json()["traces"]
    assert any(t["run_id"] == run_id for t in traces)
    trace = client.get(f"/traces/{run_id}").json()
    assert trace["steps"]
    assert trace["signals"]
    assert client.get("/traces/nope").status_code == 404


def test_diagnosis_endpoints(client: TestClient):
    run_id = client.post(
        "/query",
        json={"query": "What is the minimum password length?", "faults": ["GENERATION_CONSTRAINT_REMOVAL"]},
    ).json()["run_id"]
    diagnoses = client.get("/diagnoses").json()["diagnoses"]
    assert diagnoses
    detail = client.get(f"/diagnoses/{diagnoses[0]['diagnosis_id']}").json()
    assert "failure_class" in detail
    assert client.get("/diagnoses/nope").status_code == 404


def test_metrics_endpoint(client: TestClient):
    payload = client.get("/metrics").json()
    assert payload["counts"]
    assert "aggregates" in payload
    assert payload["label"]


def test_abstentions_endpoint(client: TestClient):
    assert "abstentions" in client.get("/abstentions").json()


def test_models_endpoint(client: TestClient):
    assert "models" in client.get("/models").json()


def test_response_time_header_present(client: TestClient):
    assert "X-Response-Time-ms" in client.get("/health").headers