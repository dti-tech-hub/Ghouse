"""Pydantic request/response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# health / meta
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str
    provider: str
    model: str
    documents: int
    chunks: int
    database: str
    labels: Dict[str, str] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    settings: Dict[str, Any]
    taxonomy: Dict[str, Any]
    faults: List[Dict[str, Any]]
    budgets: Dict[str, Any]


# ---------------------------------------------------------------------------
# knowledge base
# ---------------------------------------------------------------------------
class DocumentResponse(BaseModel):
    document_id: str
    source: str
    title: str
    doc_type: str
    version: int
    is_active: bool
    chunk_count: int = 0
    chars: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class IngestRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=1024)
    force: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    skipped: int
    errors: List[str] = Field(default_factory=list)


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    source: str = Field(default="manual", max_length=256)
    title: str = Field(default="", max_length=256)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field(default="mode3_agentic_with_recovery")
    faults: List[str] = Field(default_factory=list, max_length=8)
    fault_parameters: Dict[str, Any] = Field(default_factory=dict)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    retrieval_method: Optional[str] = Field(default=None, pattern="^(vector|bm25|hybrid)$")
    use_rerank: Optional[bool] = None
    answer_keywords: List[str] = Field(default_factory=list, max_length=32)
    relevant_documents: List[str] = Field(default_factory=list, max_length=32)
    persist: bool = True
    recover: bool = Field(default=True, description="apply the recovery loop (mode 3 only)")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("mode")
    @classmethod
    def known_mode(cls, value: str) -> str:
        allowed = {
            "mode1_normal_rag",
            "mode2_agentic_no_recovery",
            "mode3_agentic_with_recovery",
        }
        if value not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        return value


class RetrievedChunkResponse(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    score: float
    method: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None
    source: str
    page: Optional[int] = None
    content: str
    is_relevant: Optional[bool] = None


class DiagnosisResponse(BaseModel):
    diagnosis_id: str
    failure_detected: bool
    failure_class: str
    stage: Optional[str] = None
    subtype: Optional[str] = None
    severity: str
    confidence: float
    tier: str
    rule_id: Optional[str] = None
    recommended_recovery: Optional[str] = None
    recovery_ladder: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    created_at: str


class RecoveryResponse(BaseModel):
    attempt_id: str
    attempt_index: int
    action: str
    target_stage: str
    status: str
    rationale: str
    reused_stage: bool
    skipped_stages: List[str] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    added_latency_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    error: Optional[str] = None
    created_at: str


class TraceStep(BaseModel):
    index: int
    stage: str
    action: str
    status: str
    latency_ms: float
    input_summary: str
    output_summary: str
    tool_name: Optional[str] = None
    error: Optional[str] = None


class QueryResponse(BaseModel):
    run_id: str
    mode: str
    status: str
    query: str
    answer: str
    baseline_answer: str
    abstained: bool
    abstention: Optional[Dict[str, Any]] = None
    recovered: bool
    understanding: Dict[str, Any]
    sub_results: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved: List[RetrievedChunkResponse] = Field(default_factory=list)
    retrieval_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    generation: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Any] = Field(default_factory=dict)
    diagnoses: List[DiagnosisResponse] = Field(default_factory=list)
    recoveries: List[RecoveryResponse] = Field(default_factory=list)
    steps: List[TraceStep] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    faults: List[Dict[str, Any]] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    added_latency_ms: float = 0.0
    created_at: str


# ---------------------------------------------------------------------------
# faults
# ---------------------------------------------------------------------------
class FaultDefinitionResponse(BaseModel):
    fault_type: str
    title: str
    target_stage: str
    category: str
    description: str
    expected_symptom: str
    expected_failure_class: str
    expected_subtype: str
    severity: str
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    possible_subtypes: List[str] = Field(default_factory=list)


class InjectFaultRequest(BaseModel):
    fault_type: str = Field(..., min_length=1, max_length=64)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    query: Optional[str] = Field(default=None, max_length=4000)


class InjectFaultResponse(BaseModel):
    fault_id: str
    fault_type: str
    target_stage: str
    severity: str
    expected_failure_class: str
    expected_subtype: str
    expected_symptom: str
    injection_parameters: Dict[str, Any]
    active: bool
    run_id: Optional[str] = None
    result: Optional[QueryResponse] = None


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, max_length=128)
    config: Optional[Dict[str, Any]] = None
    dataset: str = Field(default="demo", max_length=64)
    fault_type: Optional[str] = Field(default=None, max_length=64)
    fault_types: List[str] = Field(default_factory=list, max_length=32)
    question_ids: List[str] = Field(default_factory=list, max_length=200)
    limit: Optional[int] = Field(default=None, ge=1, le=200)
    repetitions: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=42, ge=0)
    modes: List[str] = Field(default_factory=lambda: ["mode3_agentic_with_recovery"])
    persist: bool = True

    @field_validator("modes")
    @classmethod
    def known_modes(cls, value: List[str]) -> List[str]:
        allowed = {"mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"}
        for mode in value:
            if mode not in allowed:
                raise ValueError(f"unknown mode '{mode}'")
        return value


class ExperimentBatchRequest(BaseModel):
    configs: List[str] = Field(..., min_length=1, max_length=32)
    persist: bool = True


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    description: str = ""
    dataset: Optional[str] = None
    fault_types: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    repetitions: int = 1
    seed: Optional[int] = None
    mode: Optional[str] = None
    status: str = "pending"
    result_label: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None


class ExperimentDetail(ExperimentSummary):
    retrieval_config: Dict[str, Any] = Field(default_factory=dict)
    recovery_config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    runs: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# traces / metrics
# ---------------------------------------------------------------------------
class TraceResponse(BaseModel):
    run_id: str
    query: str
    mode: str
    status: str
    abstained: bool
    created_at: str
    total_latency_ms: float
    added_latency_ms: float
    steps: List[TraceStep] = Field(default_factory=list)
    spans: List[Dict[str, Any]] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)
    diagnoses: List[DiagnosisResponse] = Field(default_factory=list)
    recoveries: List[RecoveryResponse] = Field(default_factory=list)
    retrieved: List[RetrievedChunkResponse] = Field(default_factory=list)
    faults: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    counts: Dict[str, int]
    failures_by_class: Dict[str, int]
    failures_by_stage: Dict[str, int]
    severity_distribution: Dict[str, int]
    aggregates: Dict[str, float]
    label: str


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
    hint: Optional[str] = None