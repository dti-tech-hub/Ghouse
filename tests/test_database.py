"""SQLite persistence: schema, repositories and round-trips."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.db.database import Database, decode
from agentic_rag_recovery.db.repository import Repository
from agentic_rag_recovery.models import Chunk, Document, utcnow

EXPECTED_TABLES = {
    "documents",
    "chunks",
    "queries",
    "retrieval_runs",
    "retrieved_chunks",
    "agent_runs",
    "agent_steps",
    "diagnoses",
    "fault_injections",
    "recovery_attempts",
    "experiments",
    "metrics",
    "models",
    "configurations",
    "trace_spans",
    "abstentions",
    "index_versions",
    "experiments_config",
}


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "unit.db")
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> Repository:
    return Repository(db)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def test_schema_creates_every_required_table(db: Database):
    tables = set(db.tables())
    missing = EXPECTED_TABLES - tables
    assert not missing, f"missing tables: {sorted(missing)}"


def test_primary_key_lookup(db: Database):
    assert db.primary_key_of("documents") == "document_id"
    assert db.primary_key_of("chunks") == "chunk_id"
    assert db.primary_key_of("agent_runs") == "run_id"


def test_insert_uses_primary_key_column(db: Database):
    db.insert(
        "documents",
        {
            "_id": "doc_manual",
            "source": "s",
            "title": "t",
            "doc_type": "txt",
            "content": "c",
            "checksum": "abc",
            "version": 1,
            "is_active": True,
            "metadata": {"a": 1},
            "created_at": utcnow(),
        },
    )
    row = db.fetch_one("SELECT * FROM documents WHERE document_id=?", ("doc_manual",))
    assert row is not None
    assert decode(row["metadata"]) == {"a": 1}


def test_reset_recreates_schema(db: Database):
    db.insert("documents", {"_id": "d", "source": "s", "checksum": "c", "created_at": utcnow()})
    db.reset()
    assert EXPECTED_TABLES <= set(db.tables())
    assert db.count("documents") == 0


def test_decode_helper():
    assert decode('{"a": 1}') == {"a": 1}
    assert decode("not json", {}) == {}
    assert decode(None, "fallback") == "fallback"
    assert decode({"already": True}) == {"already": True}


# ---------------------------------------------------------------------------
# documents and chunks
# ---------------------------------------------------------------------------
def test_document_roundtrip(repo: Repository):
    document = Document(source="a.md", title="A", content="hello world", checksum="sum1")
    repo.upsert_document(document)
    loaded = repo.get_document(document.document_id)
    assert loaded is not None
    assert loaded.title == "A"
    assert loaded.metadata == {}


def test_chunk_roundtrip(repo: Repository):
    document = Document(source="b.md", content="content", checksum="sum2")
    repo.upsert_document(document)
    chunks = [
        Chunk(document_id=document.document_id, chunk_index=i, content=f"part {i}", source="b.md")
        for i in range(3)
    ]
    assert repo.replace_chunks(document.document_id, chunks) == 3
    loaded = repo.get_chunks(document.document_id)
    assert [c.chunk_index for c in loaded] == [0, 1, 2]
    assert repo.chunk_count() == 3


def test_replace_chunks_deletes_previous(repo: Repository):
    document = Document(source="c.md", content="content", checksum="sum3")
    repo.upsert_document(document)
    repo.replace_chunks(document.document_id, [Chunk(document_id=document.document_id, content="one")])
    repo.replace_chunks(document.document_id, [Chunk(document_id=document.document_id, content="two")])
    chunks = repo.get_chunks(document.document_id)
    assert len(chunks) == 1
    assert chunks[0].content == "two"


def test_get_chunks_by_ids(repo: Repository):
    document = Document(source="d.md", content="content", checksum="sum4")
    repo.upsert_document(document)
    ids = [f"chk_{i}" for i in range(5)]
    repo.replace_chunks(
        document.document_id,
        [Chunk(chunk_id=cid, document_id=document.document_id, content=f"c{i}") for i, cid in enumerate(ids)],
    )
    fetched = repo.get_chunks_by_ids([ids[3], ids[1]])
    assert [c.chunk_id for c in fetched] == [ids[3], ids[1]]


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
def test_run_roundtrip(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        experiment_id="exp_test",
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
        persist=True,
    )
    run = app.repository.get_run(outcome.run.run_id)
    assert run is not None
    assert run["query"] == question.question
    assert run["experiment_id"] == "exp_test"
    assert run["signals"]
    assert run["diagnoses"]
    assert run["recoveries"]
    assert run["steps"]
    assert run["retrieved_chunks"]
    assert run["faults"]


def test_diagnosis_roundtrip(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(question.question, faults=["GENERATION_CONSTRAINT_REMOVAL"], persist=True)
    diagnosis_id = outcome.run.diagnoses[0].diagnosis_id
    stored = app.repository.get_diagnosis(diagnosis_id)
    assert stored is not None
    assert stored["subtype"]
    assert isinstance(stored["evidence"], list)
    assert isinstance(stored["recovery_ladder"], list)


def test_abstention_is_persisted(app, dataset):
    question = dataset.by_id("q016")
    outcome = app.orchestrator.answer(question.question, faults=["CONFLICTING_EVIDENCE"], persist=True)
    if outcome.abstained:
        rows = app.repository.list_abstentions(limit=10)
        assert rows
        assert rows[0]["run_id"] == outcome.run.run_id
        assert isinstance(rows[0]["attempted"], list)


def test_fault_is_persisted_with_ground_truth(app, dataset):
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(question.question, faults=["LOW_TOP_K"], persist=True)
    faults = app.repository.list_faults(limit=5)
    assert faults
    assert faults[0]["expected_failure_class"]
    assert isinstance(faults[0]["injection_parameters"], dict)


def test_metrics_are_persisted(app, dataset):
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(question.question, persist=True)
    rows = app.repository.get_metrics(run_id=outcome.run.run_id, group_name="per_run")
    assert rows
    names = {r["name"] for r in rows}
    assert "groundedness" in names


def test_experiment_lifecycle(repo: Repository):
    repo.create_experiment(
        {
            "_id": "exp_1",
            "name": "test",
            "description": "d",
            "dataset": "demo",
            "fault_types": ["LOW_TOP_K"],
            "model": "mock",
            "retrieval_config": {"top_k": 5},
            "recovery_config": {"max_recovery_attempts": 4},
            "repetitions": 1,
            "seed": 42,
            "mode": "mode3_agentic_with_recovery",
            "status": "running",
            "created_at": utcnow(),
        }
    )
    row = repo.get_experiment("exp_1")
    assert row["fault_types"] == ["LOW_TOP_K"]
    assert row["retrieval_config"]["top_k"] == 5
    repo.update_experiment("exp_1", {"status": "completed"})
    assert repo.get_experiment("exp_1")["status"] == "completed"
    assert repo.get_experiment("missing") is None


def test_model_and_config_registration(repo: Repository):
    repo.register_model("m1", "mock", params={"temperature": 0.0}, is_default=True)
    models = repo.list_models()
    assert any(m["name"] == "m1" and m["is_default"] for m in models)
    repo.save_configuration("c1", "settings", {"a": 1})
    assert repo.db.count("configurations") == 1


def test_counts_and_breakdown(app):
    counts = app.repository.counts()
    for key in ("documents", "chunks", "runs", "diagnoses", "experiments", "metrics"):
        assert key in counts
    breakdown = app.repository.failure_breakdown()
    assert set(breakdown) == {"by_class", "by_stage", "by_severity"}


def test_trace_listing(app, dataset):
    app.orchestrator.answer("What is the minimum password length?", persist=True)
    traces = app.repository.recent_trace(limit=5)
    assert traces
    assert "query" in traces[0]


def test_list_runs_filters(app, dataset):
    app.orchestrator.answer("What is the minimum password length?", persist=True)
    assert app.repository.list_runs(limit=5)
    assert app.repository.count_runs() >= 1


def test_repository_tolerates_database_error(repo: Repository):
    with pytest.raises(Exception):
        repo.db.fetch_all("SELECT * FROM table_that_does_not_exist")