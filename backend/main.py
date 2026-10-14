"""FastAPI application for the Agentic RAG Failure Diagnosis and Recovery System."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agentic_rag_recovery import __version__
from agentic_rag_recovery.bootstrap import bootstrap_demo_data
from agentic_rag_recovery.evaluation.datasets import load_demo_dataset, list_public_datasets
from agentic_rag_recovery.faults.definitions import FaultRegistry
from agentic_rag_recovery.models import RunMode
from agentic_rag_recovery.observability.trace import get_logger
from agentic_rag_recovery.recovery.strategies import registered_strategies
from agentic_rag_recovery.taxonomy import FailureClass, FailureTaxonomy, RecoveryAction, Stage

from .schemas import (
    ConfigResponse,
    DiagnosisResponse,
    DocumentResponse,
    ErrorResponse,
    ExperimentBatchRequest,
    ExperimentDetail,
    ExperimentRequest,
    ExperimentSummary,
    FaultDefinitionResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    InjectFaultRequest,
    InjectFaultResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
    TraceResponse,
)
from .services import get_app, get_runner, system_info

logger = get_logger("backend.main")

DESCRIPTION = """
**Agentic RAG Failure Diagnosis and Recovery System**

`MONITOR → DETECT → DIAGNOSE → RECOVER → VERIFY → RESPOND or ABSTAIN`

* Every endpoint is safe to call: user input is validated and never executed.
* Demo mode needs **no API key** (deterministic mock provider).
* Results produced with the mock provider are labelled `DEMO / MOCK RESULTS`.
"""

app = FastAPI(
    title="Agentic RAG Failure Diagnosis and Recovery System",
    version=__version__,
    description=DESCRIPTION,
    openapi_tags=[
        {"name": "system", "description": "Health, configuration and taxonomy."},
        {"name": "knowledge-base", "description": "Documents, ingestion and indexing."},
        {"name": "query", "description": "End-to-end agentic RAG with diagnosis and recovery."},
        {"name": "faults", "description": "Controlled fault injection with ground truth."},
        {"name": "experiments", "description": "Reproducible experiment runs."},
        {"name": "observability", "description": "Traces, diagnoses and metrics."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(detail=str(exc), code="invalid_input").model_dump(),
    )


@app.exception_handler(FileNotFoundError)
async def not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(detail=str(exc), code="not_found").model_dump(),
    )


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.3f}"
    return response


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    application = get_app()
    info = application.orchestrator.health()
    return HealthResponse(
        status="ok",
        version=__version__,
        mode=application.settings.mode,
        provider=application.settings.generation.provider,
        model=application.settings.generation.model,
        documents=application.repository.document_count(),
        chunks=application.orchestrator.retriever.size,
        database=str(application.settings.db_path()),
        labels=application.settings.labels,
    )


@app.get("/system", tags=["system"])
def system() -> Dict[str, Any]:
    return system_info()


@app.get("/config", response_model=ConfigResponse, tags=["system"])
def config() -> ConfigResponse:
    application = get_app()
    return ConfigResponse(
        settings=application.settings.to_dict(),
        taxonomy=application.orchestrator.taxonomy(),
        faults=FaultRegistry.catalogue(),
        budgets=application.settings.budget.__dict__,
    )


@app.get("/taxonomy", tags=["system"])
def taxonomy() -> Dict[str, Any]:
    return get_app().orchestrator.taxonomy()


@app.get("/recovery/strategies", tags=["system"])
def recovery_strategies() -> Dict[str, Any]:
    return {
        "strategies": registered_strategies(),
        "actions": [a.value for a in RecoveryAction],
        "classes": [c.value for c in FailureClass],
        "stages": [s.value for s in Stage],
        "subtypes": len(FailureTaxonomy.all()),
    }


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------
@app.get("/documents", response_model=List[DocumentResponse], tags=["knowledge-base"])
def list_documents(
    active_only: bool = True,
    limit: int = Query(default=200, ge=1, le=2000),
) -> List[DocumentResponse]:
    application = get_app()
    out: List[DocumentResponse] = []
    for document in application.repository.list_documents(active_only=active_only, limit=limit):
        chunks = application.repository.get_chunks(document.document_id)
        out.append(
            DocumentResponse(
                document_id=document.document_id,
                source=document.source,
                title=document.title,
                doc_type=document.doc_type,
                version=document.version,
                is_active=document.is_active,
                chunk_count=len(chunks),
                chars=len(document.content),
                metadata=document.metadata,
                created_at=document.created_at,
            )
        )
    return out


@app.get("/documents/{document_id}", tags=["knowledge-base"])
def get_document(document_id: str) -> Dict[str, Any]:
    application = get_app()
    document = application.repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"document '{document_id}' not found")
    chunks = application.repository.get_chunks(document_id)
    return {
        **document.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
        "chunk_count": len(chunks),
    }


@app.post("/documents/ingest", response_model=IngestResponse, tags=["knowledge-base"])
def ingest(payload: IngestRequest) -> IngestResponse:
    application = get_app()
    candidate = Path(payload.path)
    if not candidate.exists():
        demo = Path(application.settings.paths.demo_documents)
        candidate = demo / payload.path
        if not candidate.exists():
            raise HTTPException(
                status_code=404,
                detail=f"path '{payload.path}' not found (also tried '{demo}')",
            )
    result = bootstrap_demo_data(application, str(candidate), force=payload.force)
    application.orchestrator.reindex()
    return IngestResponse(
        documents=result.document_count,
        chunks=result.chunk_count,
        skipped=result.skipped,
        errors=result.errors,
    )


@app.post("/documents/text", response_model=DocumentResponse, tags=["knowledge-base"])
def ingest_text(payload: TextIngestRequest) -> DocumentResponse:
    application = get_app()
    document, _chunks = application.ingestor.ingest_text(
        payload.text, source=payload.source, title=payload.title
    )
    application.orchestrator.reindex()
    return DocumentResponse(
        document_id=document.document_id,
        source=document.source,
        title=document.title,
        doc_type=document.doc_type,
        version=document.version,
        is_active=document.is_active,
        chunk_count=len(application.repository.get_chunks(document.document_id)),
        chars=len(document.content),
        metadata=document.metadata,
        created_at=document.created_at,
    )


@app.post("/documents/reindex", tags=["knowledge-base"])
def reindex() -> Dict[str, Any]:
    application = get_app()
    return application.rebuild_indexes()


@app.get("/datasets", tags=["knowledge-base"])
def datasets() -> Dict[str, Any]:
    application = get_app()
    demo = load_demo_dataset(application.settings)
    return {
        "demo": demo.summary(),
        "demo_questions": [q.to_dict() for q in demo.questions],
        "public": list_public_datasets(application.settings),
    }


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def _to_query_response(outcome: Any) -> QueryResponse:
    run = outcome.run
    return QueryResponse(
        run_id=run.run_id,
        mode=run.mode.value,
        status=run.status.value,
        query=run.query,
        answer=run.answer,
        baseline_answer=run.baseline_answer,
        abstained=run.abstained,
        abstention=run.abstention.model_dump() if run.abstention else None,
        recovered=run.recovered,
        understanding=run.understanding.model_dump() if run.understanding else {},
        sub_results=run.sub_results,
        retrieved=[r.model_dump() for r in run.retrieved],
        retrieval_diagnostics=run.retrieval_diagnostics.model_dump() if run.retrieval_diagnostics else {},
        context=run.context.model_dump() if run.context else {},
        generation=run.generation.model_dump() if run.generation else {},
        signals=run.signals.model_dump() if run.signals else {},
        diagnoses=[DiagnosisResponse(**_diagnosis_fields(d)) for d in run.diagnoses],
        recoveries=[r.model_dump(mode="json") for r in run.recoveries],
        steps=[
            {
                "index": s.index,
                "stage": s.stage,
                "action": s.action,
                "status": s.status,
                "latency_ms": s.latency_ms,
                "input_summary": s.input_summary,
                "output_summary": s.output_summary,
                "tool_name": s.tool_name,
                "error": s.error,
            }
            for s in run.steps
        ],
        metrics=run.metrics,
        faults=run.faults,
        labels=run.labels,
        total_latency_ms=run.total_latency_ms,
        baseline_latency_ms=run.baseline_latency_ms,
        added_latency_ms=run.added_latency_ms,
        created_at=run.created_at,
    )


def _diagnosis_fields(diagnosis: Any) -> Dict[str, Any]:
    return {
        "diagnosis_id": diagnosis.diagnosis_id,
        "failure_detected": diagnosis.failure_detected,
        "failure_class": diagnosis.class_value,
        "stage": diagnosis.stage.value if diagnosis.stage else None,
        "subtype": diagnosis.subtype,
        "severity": diagnosis.severity.value if hasattr(diagnosis.severity, "value") else str(diagnosis.severity),
        "confidence": diagnosis.confidence,
        "tier": diagnosis.tier.value if hasattr(diagnosis.tier, "value") else str(diagnosis.tier),
        "rule_id": diagnosis.rule_id,
        "recommended_recovery": (
            diagnosis.recommended_recovery.value
            if hasattr(diagnosis.recommended_recovery, "value")
            else diagnosis.recommended_recovery
        ),
        "recovery_ladder": [
            a.value if hasattr(a, "value") else str(a) for a in diagnosis.recovery_ladder
        ],
        "evidence": diagnosis.evidence,
        "validation_errors": diagnosis.validation_errors,
        "latency_ms": diagnosis.latency_ms,
        "created_at": diagnosis.created_at,
    }


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(payload: QueryRequest) -> QueryResponse:
    application = get_app()
    mode = payload.mode
    if not payload.recover and mode == RunMode.MODE_3_AGENTIC_WITH_RECOVERY.value:
        mode = RunMode.MODE_2_AGENTIC_NO_RECOVERY.value
    outcome = application.orchestrator.answer(
        payload.query,
        mode=mode,
        faults=payload.faults or None,
        fault_parameters=payload.fault_parameters,
        top_k=payload.top_k,
        retrieval_method=payload.retrieval_method,
        use_rerank=payload.use_rerank,
        relevant_documents=payload.relevant_documents or None,
        answer_keywords=payload.answer_keywords or None,
        persist=payload.persist,
    )
    return _to_query_response(outcome)


@app.post("/query/recover", response_model=QueryResponse, tags=["query"])
def query_recover(payload: QueryRequest) -> QueryResponse:
    """Explicit recovery entry point (same behaviour as POST /query)."""
    payload.recover = True
    return query(payload)


@app.get("/query/demo", response_model=QueryResponse, tags=["query"])
def query_demo(
    question_id: str = Query(default="q001"),
    fault: Optional[str] = Query(default=None),
    recover: bool = Query(default=True),
) -> QueryResponse:
    """Run one question from the offline demonstration dataset."""
    application = get_app()
    dataset = load_demo_dataset(application.settings)
    question = dataset.by_id(question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"question '{question_id}' not found")
    mode = (
        RunMode.MODE_3_AGENTIC_WITH_RECOVERY
        if recover
        else RunMode.MODE_2_AGENTIC_NO_RECOVERY
    )
    outcome = application.orchestrator.answer(
        question.question,
        mode=mode,
        faults=[fault] if fault else None,
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
    )
    return _to_query_response(outcome)


# ---------------------------------------------------------------------------
# faults
# ---------------------------------------------------------------------------
@app.get("/faults", response_model=List[FaultDefinitionResponse], tags=["faults"])
def list_faults() -> List[FaultDefinitionResponse]:
    return [FaultDefinitionResponse(**entry) for entry in FaultRegistry.catalogue()]


@app.post("/faults/inject", response_model=InjectFaultResponse, tags=["faults"])
def inject_fault(payload: InjectFaultRequest) -> InjectFaultResponse:
    application = get_app()
    definition = FaultRegistry.get(payload.fault_type)
    if definition is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown fault type '{payload.fault_type}'",
        )
    if payload.parameters and not isinstance(payload.parameters, dict):
        raise HTTPException(status_code=422, detail="parameters must be an object")

    if not payload.query:
        fault = definition.build(payload.parameters)
        application.repository.record_fault(fault)
        return InjectFaultResponse(
            fault_id=fault.fault_id,
            fault_type=fault.fault_type,
            target_stage=fault.target_stage.value,
            severity=fault.severity.value,
            expected_failure_class=fault.expected_failure_class.value,
            expected_subtype=fault.expected_subtype or "",
            expected_symptom=fault.expected_symptom,
            injection_parameters=fault.injection_parameters,
            active=fault.active,
        )

    outcome = application.orchestrator.answer(
        payload.query,
        faults=[definition.build(payload.parameters)],
        fault_parameters=payload.parameters,
    )
    fault = outcome.run.faults[0] if outcome.run.faults else None
    return InjectFaultResponse(
        fault_id=fault.get("fault_id", "") if fault else "",
        fault_type=payload.fault_type,
        target_stage=fault.get("target_stage", definition.target_stage.value) if fault else definition.target_stage.value,
        severity=fault.get("severity", definition.severity.value) if fault else definition.severity.value,
        expected_failure_class=definition.expected_failure_class.value,
        expected_subtype=definition.expected_subtype or "",
        expected_symptom=definition.expected_symptom,
        injection_parameters=definition.build(payload.parameters).injection_parameters,
        active=True,
        run_id=outcome.run.run_id,
        result=_to_query_response(outcome),
    )


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
@app.get("/experiments", response_model=List[ExperimentSummary], tags=["experiments"])
def list_experiments(limit: int = Query(default=100, ge=1, le=1000)) -> List[ExperimentSummary]:
    rows = get_app().repository.list_experiments(limit=limit)
    return [ExperimentSummary(**_experiment_row(row)) for row in rows]


@app.get("/experiments/configs", tags=["experiments"])
def list_experiment_configs() -> Dict[str, Any]:
    return {
        "available": get_runner().available_configs(),
        "stored": get_app().repository.list_experiment_configs(),
    }


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetail, tags=["experiments"])
def get_experiment(experiment_id: str) -> ExperimentDetail:
    application = get_app()
    row = application.repository.get_experiment(experiment_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"experiment '{experiment_id}' not found")
    metrics = application.repository.get_metrics(experiment_id=experiment_id)
    runs = application.repository.list_runs(experiment_id=experiment_id, limit=500)
    detail = ExperimentDetail(**_experiment_row(row))
    detail.metrics = {
        m["name"] + (f":{m['group_name']}" if m["group_name"] else ""): m["value"]
        for m in metrics
        if m["value"] is not None
    }
    detail.runs = runs
    return detail


def _experiment_row(row: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "experiment_id",
        "name",
        "description",
        "dataset",
        "fault_types",
        "model",
        "repetitions",
        "seed",
        "mode",
        "status",
        "result_label",
        "started_at",
        "finished_at",
        "error",
        "created_at",
        "retrieval_config",
        "recovery_config",
    }
    return {key: row.get(key) for key in allowed if key in row}


@app.post("/experiments/run", tags=["experiments"])
def run_experiment(payload: ExperimentRequest) -> Dict[str, Any]:
    application = get_app()
    runner = get_runner()
    if payload.config:
        config = dict(payload.config)
        config.setdefault("name", payload.name or "ad-hoc")
    elif payload.name:
        try:
            config = runner.load_config(payload.name)
        except FileNotFoundError:
            # an unknown name with explicit parameters means "run ad-hoc"
            if not (payload.fault_type or payload.fault_types or payload.question_ids or payload.limit):
                raise
            config = {"name": payload.name, "description": "ad-hoc experiment created through the API"}
    else:
        config = {
                "name": "ad-hoc",
                "description": "ad-hoc experiment created through the API",
                "dataset": payload.dataset,
                "fault_type": payload.fault_type,
                "fault_sweep": payload.fault_types or None,
                "question_ids": payload.question_ids or None,
                "limit": payload.limit,
                "repetitions": payload.repetitions,
                "seed": payload.seed,
                "modes": payload.modes,
            }
    config["repetitions"] = payload.repetitions or config.get("repetitions", 1)
    config["seed"] = payload.seed if payload.seed is not None else config.get("seed", 42)
    if payload.fault_types:
        config["fault_sweep"] = payload.fault_types
    if payload.question_ids:
        config["question_ids"] = payload.question_ids
    if payload.limit:
        config["limit"] = payload.limit
    if payload.modes:
        config["modes"] = payload.modes

    try:
        result = runner.run(config, persist=payload.persist)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/experiments/batch", tags=["experiments"])
def run_experiments_batch(payload: ExperimentBatchRequest) -> Dict[str, Any]:
    runner = get_runner()
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for name in payload.configs:
        try:
            config = runner.load_config(name)
            results.append(runner.run(config, persist=payload.persist).to_dict())
        except Exception as exc:
            errors.append({"config": name, "error": str(exc)})
    return {"executed": len(results), "errors": errors, "experiments": results}


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------
@app.get("/traces", tags=["observability"])
def list_traces(limit: int = Query(default=50, ge=1, le=500)) -> Dict[str, Any]:
    return {"traces": get_app().repository.recent_trace(limit=limit)}


@app.get("/traces/{run_id}", response_model=TraceResponse, tags=["observability"])
def get_trace(run_id: str) -> TraceResponse:
    application = get_app()
    row = application.repository.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"run '{run_id}' not found")
    return TraceResponse(
        run_id=row["run_id"],
        query=row["query"],
        mode=row["mode"],
        status=row["status"],
        abstained=bool(row.get("abstained")),
        created_at=row["created_at"],
        total_latency_ms=row.get("total_latency_ms", 0.0),
        added_latency_ms=row.get("added_latency_ms", 0.0),
        steps=[
            {
                "index": s.get("step_index", 0),
                "stage": s.get("stage", ""),
                "action": s.get("action", ""),
                "status": s.get("status", ""),
                "latency_ms": s.get("latency_ms", 0.0),
                "input_summary": s.get("input_summary", ""),
                "output_summary": s.get("output_summary", ""),
                "tool_name": s.get("tool_name"),
                "error": s.get("error"),
            }
            for s in row.get("steps", [])
        ],
        spans=row.get("spans", []),
        signals=row.get("signals") or {},
        diagnoses=[DiagnosisResponse(**_diagnosis_dict(d)) for d in row.get("diagnoses", [])],
        recoveries=row.get("recoveries", []),
        retrieved=row.get("retrieved_chunks", []),
        faults=row.get("faults") or [],
        metrics=row.get("metrics") or {},
    )


def _diagnosis_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "diagnosis_id": row.get("diagnosis_id"),
        "failure_detected": bool(row.get("failure_detected")),
        "failure_class": row.get("failure_class") or "NONE",
        "stage": row.get("stage"),
        "subtype": row.get("subtype"),
        "severity": row.get("severity") or "low",
        "confidence": row.get("confidence") or 0.0,
        "tier": row.get("tier") or "none",
        "rule_id": row.get("rule_id"),
        "recommended_recovery": row.get("recommended_recovery"),
        "recovery_ladder": row.get("recovery_ladder") or [],
        "evidence": row.get("evidence") or [],
        "validation_errors": row.get("validation_errors") or [],
        "latency_ms": row.get("latency_ms") or 0.0,
        "created_at": row.get("created_at") or "",
    }


@app.get("/diagnoses/{diagnosis_id}", response_model=DiagnosisResponse, tags=["observability"])
def get_diagnosis(diagnosis_id: str) -> DiagnosisResponse:
    row = get_app().repository.get_diagnosis(diagnosis_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"diagnosis '{diagnosis_id}' not found")
    return DiagnosisResponse(**_diagnosis_dict(row))


@app.get("/diagnoses", tags=["observability"])
def list_diagnoses(limit: int = Query(default=100, ge=1, le=1000)) -> Dict[str, Any]:
    rows = get_app().repository.db.fetch_all(
        "SELECT diagnosis_id, run_id, failure_class, stage, subtype, severity, confidence, tier, "
        "created_at FROM diagnoses ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (limit,),
    )
    for row in rows:
        row["failure_detected"] = bool(row.get("failure_detected", True))
    return {"diagnoses": rows}


@app.get("/metrics", response_model=MetricsResponse, tags=["observability"])
def metrics() -> MetricsResponse:
    application = get_app()
    breakdown = application.repository.failure_breakdown()
    return MetricsResponse(
        counts=application.repository.counts(),
        failures_by_class=breakdown["by_class"],
        failures_by_stage=breakdown["by_stage"],
        severity_distribution=breakdown["by_severity"],
        aggregates=application.repository.aggregate_metrics("summary"),
        label=application.settings.labels.get("results", ""),
    )


@app.get("/metrics/runs", tags=["observability"])
def per_run_metrics(limit: int = Query(default=100, ge=1, le=1000)) -> Dict[str, Any]:
    rows = get_app().repository.get_metrics(group_name="per_run", limit=limit)
    return {"metrics": rows, "label": get_app().settings.labels.get("results", "")}


@app.get("/abstentions", tags=["observability"])
def abstentions(limit: int = Query(default=100, ge=1, le=1000)) -> Dict[str, Any]:
    return {"abstentions": get_app().repository.list_abstentions(limit=limit)}


@app.get("/models", tags=["observability"])
def models() -> Dict[str, Any]:
    return {"models": get_app().repository.list_models()}


@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    return {
        "name": app.title,
        "version": __version__,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": sorted(
            route.path for route in app.routes if getattr(route, "methods", None) and route.path.startswith("/")
        ),
    }