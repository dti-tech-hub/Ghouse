"""Repository layer: persistence of runs, diagnoses, recoveries, metrics."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..models import (
    Chunk,
    Diagnosis,
    Document,
    ExperimentMetrics,
    FaultInjection,
    PipelineRun,
    RecoveryAttempt,
    RetrievalDiagnostics,
    RetrievedChunk,
    utcnow,
)
from ..taxonomy import RecoveryAction
from .database import Database, decode

_JSON_FIELDS = {
    "agent_runs": [
        "understanding",
        "context",
        "generation",
        "retrieval_diag",
        "signals",
        "metrics",
        "faults",
        "labels",
        "abstention",
    ],
    "diagnoses": [
        "recovery_ladder",
        "evidence",
        "rule_matches",
        "signals_used",
        "validation_errors",
        "repairs",
    ],
    "recovery_attempts": ["skipped_stages", "verification", "before_state", "after_state"],
    "abstentions": ["attempted", "budget_consumed"],
    "experiments": ["fault_types", "retrieval_config", "recovery_config"],
    "metrics": ["value_json"],
    "documents": ["metadata"],
    "chunks": ["metadata"],
    "retrieval_runs": ["diagnostics"],
    "fault_injections": ["injection_parameters"],
    "models": ["params"],
    "configurations": ["payload"],
    "queries": ["subqueries"],
    "agent_steps": ["meta"],
    "trace_spans": ["inputs", "outputs", "metrics", "tags"],
}


class Repository:
    """High-level persistence operations for the whole system."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Knowledge base
    # ------------------------------------------------------------------
    def upsert_document(self, doc: Document) -> str:
        payload = doc.model_dump()
        payload["updated_at"] = utcnow()
        return self.db.insert("documents", {"_id": payload.pop("document_id"), **payload})

    def get_document(self, document_id: str) -> Optional[Document]:
        row = self.db.fetch_one("SELECT * FROM documents WHERE document_id=?", (document_id,))
        return self._to_document(row) if row else None

    def list_documents(self, active_only: bool = True, limit: int = 500) -> List[Document]:
        sql = "SELECT * FROM documents"
        if active_only:
            sql += " WHERE is_active=1"
        sql += " ORDER BY created_at DESC LIMIT ?"
        return [self._to_document(r) for r in self.db.fetch_all(sql, (limit,))]

    def document_count(self, active_only: bool = True) -> int:
        return self.db.count("documents", "is_active=1" if active_only else "")

    def _to_document(self, row: Dict[str, Any]) -> Document:
        row = dict(row)
        row["metadata"] = decode(row.get("metadata"), {}) or {}
        row["is_active"] = bool(row.get("is_active", 1))
        return Document(**row)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    def replace_chunks(self, document_id: str, chunks: Sequence[Chunk]) -> int:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            rows = []
            for c in chunks:
                rows.append(
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.token_estimate,
                        c.char_start,
                        c.char_end,
                        c.page,
                        c.section,
                        c.source,
                        json.dumps(c.metadata, default=str),
                        c.created_at,
                    )
                )
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id,document_id,chunk_index,content,token_estimate,char_start,char_end,"
                "page,section,source,metadata,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(chunks)

    def get_chunks(self, document_id: Optional[str] = None, limit: int = 100_000) -> List[Chunk]:
        if document_id:
            rows = self.db.fetch_all(
                "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index LIMIT ?",
                (document_id, limit),
            )
        else:
            rows = self.db.fetch_all("SELECT * FROM chunks ORDER BY document_id, chunk_index LIMIT ?", (limit,))
        out = []
        for row in rows:
            row = dict(row)
            row["metadata"] = decode(row.get("metadata"), {}) or {}
            out.append(Chunk(**row))
        return out

    def get_chunks_by_ids(self, chunk_ids: Iterable[str]) -> List[Chunk]:
        ids = list(chunk_ids)
        if not ids:
            return []
        out: List[Chunk] = []
        for i in range(0, len(ids), 400):
            batch = ids[i : i + 400]
            placeholders = ",".join("?" for _ in batch)
            for row in self.db.fetch_all(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", batch
            ):
                row = dict(row)
                row["metadata"] = decode(row.get("metadata"), {}) or {}
                out.append(Chunk(**row))
        order = {cid: i for i, cid in enumerate(ids)}
        out.sort(key=lambda c: order.get(c.chunk_id, 10**6))
        return out

    def chunk_count(self) -> int:
        return self.db.count("chunks")

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def save_run(self, run: PipelineRun) -> str:
        self.save_query(run)
        u = run.understanding.model_dump() if run.understanding else {}
        ctx = run.context.model_dump() if run.context else {}
        gen = run.generation.model_dump() if run.generation else {}
        diag = run.retrieval_diagnostics.model_dump() if run.retrieval_diagnostics else {}
        signals = run.signals.model_dump() if run.signals else {}
        abstention = run.abstention.model_dump() if run.abstention else {}
        self.db.insert(
            "agent_runs",
            {
                "_id": run.run_id,
                "experiment_id": run.experiment_id,
                "query_id": f"qy_{run.run_id}",
                "mode": run.mode.value,
                "status": run.status.value,
                "query": run.query,
                "answer": run.answer,
                "baseline_answer": run.baseline_answer,
                "understanding": u,
                "context": ctx,
                "generation": gen,
                "retrieval_diag": diag,
                "signals": signals,
                "metrics": run.metrics,
                "faults": run.faults,
                "labels": run.labels,
                "baseline_latency_ms": run.baseline_latency_ms,
                "total_latency_ms": run.total_latency_ms,
                "added_latency_ms": run.added_latency_ms,
                "tokens_used": run.tokens_used,
                "llm_calls": run.llm_calls,
                "cost": run.cost,
                "abstained": int(run.abstained),
                "abstention": abstention,
                "created_at": run.created_at,
                "finished_at": run.finished_at,
            },
        )
        for step in run.steps:
            self.db.insert(
                "agent_steps",
                {
                    "_id": step.step_id,
                    "run_id": run.run_id,
                    "step_index": step.index,
                    "stage": step.stage,
                    "action": step.action,
                    "tool_name": step.tool_name,
                    "input_summary": step.input_summary,
                    "output_summary": step.output_summary,
                    "status": step.status,
                    "latency_ms": step.latency_ms,
                    "action_signature": step.action_signature,
                    "error": step.error,
                    "meta": step.meta,
                    "created_at": step.created_at,
                },
            )
        for span in run.spans:
            self.db.insert(
                "trace_spans",
                {
                    "_id": span.span_id,
                    "run_id": run.run_id,
                    "parent_id": span.parent_id,
                    "stage": span.stage,
                    "name": span.name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "latency_ms": span.latency_ms,
                    "status": span.status,
                    "inputs": span.inputs,
                    "outputs": span.outputs,
                    "metrics": span.metrics,
                    "error": span.error,
                    "tags": span.tags,
                },
            )
        for rec in run.recoveries:
            self.save_recovery(rec, experiment_id=run.experiment_id, persist_run=False)
        for dg in run.diagnoses:
            self.save_diagnosis(dg, experiment_id=run.experiment_id, persist_run=False)
        if run.retrieved:
            self.save_retrieval(run)
        if run.abstention and run.abstention.abstained:
            self.save_abstention(run)
        return run.run_id

    def save_query(self, run: PipelineRun) -> str:
        """Record the request itself so runs can be grouped by question."""
        understanding = run.understanding
        query_id = f"qy_{run.run_id}"
        self.db.insert(
            "queries",
            {
                "_id": query_id,
                "text": run.query,
                "normalized": understanding.normalized_query if understanding else run.query,
                "rewrite": understanding.rewrite if understanding else None,
                "subqueries": understanding.subqueries if understanding else [],
                "intent": understanding.intent if understanding else None,
                "is_complex": int(bool(understanding.is_complex)) if understanding else 0,
                "source": "playground" if not run.experiment_id else "experiment",
                "created_at": utcnow(),
            },
        )
        return query_id

    def save_retrieval(self, run: PipelineRun) -> Optional[str]:
        if not run.retrieved:
            return None
        retrieval_run_id = f"rr_{run.run_id}"
        diag = run.retrieval_diagnostics.model_dump() if run.retrieval_diagnostics else {}
        self.db.insert(
            "retrieval_runs",
            {
                "_id": retrieval_run_id,
                "run_id": run.run_id,
                "query_id": f"qy_{run.run_id}",
                "query": run.query,
                "method": diag.get("method", "unknown"),
                "top_k": diag.get("top_k", 0),
                "candidate_k": diag.get("candidate_pool", 0),
                "latency_ms": diag.get("latency_ms", 0.0),
                "diagnostics": diag,
                "created_at": utcnow(),
            },
        )
        rows = []
        for rc in run.retrieved:
            rows.append(
                (
                    retrieval_run_id,
                    rc.chunk_id,
                    rc.document_id,
                    rc.rank,
                    rc.score,
                    rc.method,
                    rc.dense_score,
                    rc.sparse_score,
                    rc.rerank_score,
                    rc.original_rank,
                    None if rc.is_relevant is None else int(rc.is_relevant),
                    rc.source,
                    rc.page,
                    utcnow(),
                )
            )
        self.db.execute_many(
            "INSERT INTO retrieved_chunks (retrieval_run_id,chunk_id,document_id,rank,score,method,"
            "dense_score,sparse_score,rerank_score,original_rank,is_relevant,source,page,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        return retrieval_run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM agent_runs WHERE run_id=?", (run_id,))
        if not row:
            return None
        for field in _JSON_FIELDS["agent_runs"]:
            row[field] = decode(row.get(field))
        row["abstained"] = bool(row.get("abstained"))
        row["diagnoses"] = self.list_diagnoses(run_id)
        row["recoveries"] = self.list_recoveries(run_id)
        row["steps"] = self.list_steps(run_id)
        row["spans"] = self.list_spans(run_id)
        retrieved = self.db.fetch_all(
            "SELECT * FROM retrieved_chunks WHERE retrieval_run_id=? ORDER BY rank",
            (f"rr_{run_id}",),
        )
        if retrieved:
            chunks = {c.chunk_id: c for c in self.get_chunks_by_ids([r["chunk_id"] for r in retrieved])}
            for entry in retrieved:
                chunk = chunks.get(entry["chunk_id"])
                entry["content"] = chunk.content if chunk else ""
                entry["metadata"] = dict(chunk.metadata) if chunk else {}
                entry["is_relevant"] = None if entry.get("is_relevant") is None else bool(entry["is_relevant"])
        row["retrieved_chunks"] = retrieved
        return row

    def list_runs(
        self,
        experiment_id: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id=?")
            params.append(experiment_id)
        if mode:
            clauses.append("mode=?")
            params.append(mode)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        rows = self.db.fetch_all(
            f"SELECT run_id,experiment_id,mode,status,query,answer,abstained,baseline_answer,"
            f"total_latency_ms,added_latency_ms,tokens_used,cost,created_at FROM agent_runs "
            f"{where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            params,
        )
        for r in rows:
            r["abstained"] = bool(r.get("abstained"))
        return rows

    def count_runs(self, experiment_id: Optional[str] = None, mode: Optional[str] = None) -> int:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id=?")
            params.append(experiment_id)
        if mode:
            clauses.append("mode=?")
            params.append(mode)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return self.db.count("agent_runs", where, params)

    # ------------------------------------------------------------------
    # Diagnosis / recovery / abstention
    # ------------------------------------------------------------------
    def save_diagnosis(
        self, diag: Diagnosis, experiment_id: Optional[str] = None, persist_run: bool = True
    ) -> str:
        self.db.insert(
            "diagnoses",
            {
                "_id": diag.diagnosis_id,
                "run_id": diag.run_id,
                "experiment_id": experiment_id,
                "failure_detected": int(diag.failure_detected),
                "failure_class": _enum_value(diag.failure_class),
                "stage": _enum_value(diag.stage),
                "subtype": diag.subtype,
                "severity": _enum_value(diag.severity),
                "confidence": diag.confidence,
                "tier": _enum_value(diag.tier),
                "rule_id": diag.rule_id,
                "recommended_recovery": _enum_value(diag.recommended_recovery),
                "recovery_ladder": [_enum_value(a) for a in diag.recovery_ladder],
                "evidence": diag.evidence,
                "rule_matches": diag.rule_matches,
                "signals_used": diag.signals_used,
                "latency_ms": diag.latency_ms,
                "raw_llm_output": diag.raw_llm_output,
                "validation_errors": diag.validation_errors,
                "repairs": diag.repairs,
                "created_at": diag.created_at,
            },
        )
        return diag.diagnosis_id

    def list_diagnoses(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM diagnoses WHERE run_id=? ORDER BY created_at, rowid", (run_id,)
        )
        for r in rows:
            r["failure_detected"] = bool(r.get("failure_detected"))
            for f in ("recovery_ladder", "evidence", "rule_matches", "signals_used", "validation_errors", "repairs"):
                r[f] = decode(r.get(f))
        return rows

    def get_diagnosis(self, diagnosis_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM diagnoses WHERE diagnosis_id=?", (diagnosis_id,))
        if not row:
            return None
        row["failure_detected"] = bool(row.get("failure_detected"))
        for f in ("recovery_ladder", "evidence", "rule_matches", "signals_used", "validation_errors", "repairs"):
            row[f] = decode(row.get(f))
        return row

    def save_recovery(
        self,
        attempt: RecoveryAttempt,
        experiment_id: Optional[str] = None,
        persist_run: bool = True,
    ) -> str:
        self.db.insert(
            "recovery_attempts",
            {
                "_id": attempt.attempt_id,
                "run_id": attempt.run_id,
                "experiment_id": experiment_id,
                "attempt_index": attempt.attempt_index,
                "action": attempt.action.value,
                "target_stage": _enum_value(attempt.target_stage),
                "diagnosis_subtype": attempt.diagnosis_subtype,
                "status": _enum_value(attempt.status),
                "rationale": attempt.rationale,
                "reused_stage": int(attempt.reused_stage),
                "skipped_stages": [_enum_value(s) for s in attempt.skipped_stages],
                "verification": attempt.verification,
                "before_state": attempt.before,
                "after_state": attempt.after,
                "latency_ms": attempt.latency_ms,
                "added_latency_ms": attempt.added_latency_ms,
                "tokens_used": attempt.tokens_used,
                "cost": attempt.cost,
                "llm_calls": attempt.llm_calls,
                "error": attempt.error,
                "created_at": attempt.created_at,
            },
        )
        return attempt.attempt_id

    def list_recoveries(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM recovery_attempts WHERE run_id=? ORDER BY attempt_index", (run_id,)
        )
        for r in rows:
            r["reused_stage"] = bool(r.get("reused_stage"))
            for f in ("skipped_stages", "verification", "before_state", "after_state"):
                r[f] = decode(r.get(f))
        return rows

    def save_abstention(self, run: PipelineRun) -> str:
        abst = run.abstention
        if not abst:
            return ""
        abstention_id = f"abs_{run.run_id}"
        self.db.insert(
            "abstentions",
            {
                "_id": abstention_id,
                "run_id": run.run_id,
                "reason": abst.reason,
                "failure_class": _enum_value(abst.failure_class),
                "stage": _enum_value(abst.stage),
                "subtype": abst.subtype,
                "attempted": [a.value if isinstance(a, RecoveryAction) else str(a) for a in abst.attempted_recoveries],
                "attempts_used": abst.attempts_used,
                "budget_consumed": abst.budget_consumed,
                "message": abst.message,
                "created_at": abst.created_at,
            },
        )
        return abstention_id

    def list_abstentions(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM abstentions ORDER BY created_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["attempted"] = decode(r.get("attempted"))
            r["budget_consumed"] = decode(r.get("budget_consumed"))
        return rows

    # ------------------------------------------------------------------
    # Faults
    # ------------------------------------------------------------------
    def record_fault(self, fault: FaultInjection, run_id: Optional[str] = None) -> str:
        self.db.insert(
            "fault_injections",
            {
                "_id": fault.fault_id,
                "fault_type": fault.fault_type,
                "target_stage": fault.target_stage.value,
                "severity": fault.severity.value,
                "description": fault.description,
                "expected_symptom": fault.expected_symptom,
                "expected_failure_class": fault.expected_failure_class.value,
                "expected_subtype": fault.expected_subtype,
                "injection_parameters": fault.injection_parameters,
                "active": int(fault.active),
                "experiment_id": fault.experiment_id,
                "run_id": run_id,
                "created_at": fault.created_at,
                "released_at": utcnow() if not fault.active else None,
            },
        )
        return fault.fault_id

    def list_faults(self, experiment_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM fault_injections"
        params: List[Any] = []
        if experiment_id:
            sql += " WHERE experiment_id=?"
            params.append(experiment_id)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self.db.fetch_all(sql, params)
        for r in rows:
            r["active"] = bool(r.get("active"))
            r["injection_parameters"] = decode(r.get("injection_parameters"))
        return rows

    # ------------------------------------------------------------------
    # Experiments & metrics
    # ------------------------------------------------------------------
    def create_experiment(self, record: Dict[str, Any]) -> str:
        payload = dict(record)
        experiment_id = payload.pop("_id") or payload.pop("experiment_id")
        payload.setdefault("created_at", utcnow())
        self.db.insert("experiments", {"_id": experiment_id, **payload})
        return experiment_id

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> int:
        return self.db.update("experiments", "experiment_id", experiment_id, updates)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
        if not row:
            return None
        for f in ("fault_types", "retrieval_config", "recovery_config"):
            row[f] = decode(row.get(f))
        return row

    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT experiment_id,name,description,dataset,fault_types,model,mode,repetitions,seed,"
            "status,result_label,started_at,finished_at,created_at FROM experiments "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        )
        for r in rows:
            r["fault_types"] = decode(r.get("fault_types"), [])
        return rows

    def record_metrics(
        self,
        experiment_id: Optional[str],
        metrics: Dict[str, Any],
        group_name: str = "summary",
        run_id: Optional[str] = None,
        label: str = "",
        seed: Optional[int] = None,
    ) -> None:
        rows = []
        for name, value in (metrics or {}).items():
            numeric = None
            payload = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
            elif isinstance(value, bool):
                numeric = float(value)
            else:
                payload = value
            rows.append(
                (
                    f"met_{abs(hash((experiment_id, group_name, name, str(metrics.get('seed'))))) % (10**16)}",
                    experiment_id,
                    run_id,
                    name,
                    numeric,
                    json.dumps(payload, default=str) if payload is not None else None,
                    group_name,
                    label,
                    seed,
                    utcnow(),
                )
            )
        self.db.execute_many(
            "INSERT OR REPLACE INTO metrics (metric_id,experiment_id,run_id,name,value,value_json,"
            "group_name,label,seed,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    def get_metrics(
        self,
        experiment_id: Optional[str] = None,
        group_name: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if experiment_id:
            clauses.append("experiment_id=?")
            params.append(experiment_id)
        if group_name:
            clauses.append("group_name=?")
            params.append(group_name)
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.db.fetch_all(
            f"SELECT * FROM metrics {where} ORDER BY created_at DESC, rowid DESC LIMIT ?", params
        )
        for r in rows:
            r["value_json"] = decode(r.get("value_json"))
        return rows

    def aggregate_metrics(self, group_name: str = "summary") -> Dict[str, float]:
        rows = self.db.fetch_all(
            "SELECT name, AVG(value) AS v, COUNT(*) AS n FROM metrics "
            "WHERE value IS NOT NULL AND group_name=? GROUP BY name ORDER BY name",
            (group_name,),
        )
        return {r["name"]: float(r["v"]) for r in rows if r["v"] is not None}

    # ------------------------------------------------------------------
    # Steps / spans
    # ------------------------------------------------------------------
    def list_steps(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM agent_steps WHERE run_id=? ORDER BY step_index", (run_id,)
        )
        for r in rows:
            r["meta"] = decode(r.get("meta"))
        return rows

    def list_spans(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM trace_spans WHERE run_id=? ORDER BY start_time, rowid", (run_id,)
        )
        for r in rows:
            for f in ("inputs", "outputs", "metrics", "tags"):
                r[f] = decode(r.get(f))
        return rows

    def recent_trace(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.fetch_all(
            "SELECT run_id, query, mode, status, abstained, total_latency_ms, added_latency_ms, "
            "created_at FROM agent_runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def register_model(
        self,
        name: str,
        provider: str,
        role: str = "generation",
        params: Optional[Dict[str, Any]] = None,
        cost_in: float = 0.0,
        cost_out: float = 0.0,
        is_default: bool = False,
    ) -> str:
        model_id = f"mdl_{provider}_{name}".replace(" ", "_").replace("/", "_")
        self.db.insert(
            "models",
            {
                "_id": model_id,
                "name": name,
                "provider": provider,
                "role": role,
                "version": "1",
                "params": params or {},
                "cost_per_1k_in": cost_in,
                "cost_per_1k_out": cost_out,
                "is_default": int(is_default),
                "created_at": utcnow(),
            },
        )
        return model_id

    def list_models(self) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM models ORDER BY created_at")
        for r in rows:
            r["params"] = decode(r.get("params"))
            r["is_default"] = bool(r.get("is_default"))
        return rows

    def save_configuration(
        self, name: str, kind: str, payload: Dict[str, Any], description: str = ""
    ) -> str:
        import hashlib

        blob = json.dumps(payload, sort_keys=True, default=str)
        config_id = f"cfg_{hashlib.sha256(f'{kind}:{name}'.encode()).hexdigest()[:12]}"
        self.db.insert(
            "configurations",
            {
                "_id": config_id,
                "name": name,
                "kind": kind,
                "description": description,
                "payload": payload,
                "hash": hashlib.sha256(blob.encode()).hexdigest(),
                "created_at": utcnow(),
            },
        )
        return config_id

    def save_experiment_config(self, name: str, payload: Dict[str, Any], seed: int = 42) -> str:
        import hashlib

        config_id = f"exc_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        self.db.insert(
            "experiments_config",
            {
                "_id": config_id,
                "name": name,
                "description": payload.get("description", ""),
                "payload": payload,
                "seed": seed,
                "created_at": utcnow(),
            },
        )
        return config_id

    def list_experiment_configs(self) -> List[Dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM experiments_config ORDER BY name")
        for r in rows:
            r["payload"] = decode(r.get("payload"))
        return rows

    # ------------------------------------------------------------------
    # Dashboard summaries
    # ------------------------------------------------------------------
    def failure_breakdown(self, limit: int = 500) -> Dict[str, Dict[str, int]]:
        rows = self.db.fetch_all(
            "SELECT failure_class, stage, severity, COUNT(*) AS n FROM diagnoses "
            "WHERE failure_detected=1 ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        by_class: Dict[str, int] = {}
        by_stage: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for r in rows:
            by_class[r["failure_class"] or "UNKNOWN"] = by_class.get(r["failure_class"] or "UNKNOWN", 0) + r["n"]
            by_stage[r["stage"] or "unknown"] = by_stage.get(r["stage"] or "unknown", 0) + r["n"]
            by_severity[r["severity"] or "unknown"] = by_severity.get(r["severity"] or "unknown", 0) + r["n"]
        return {"by_class": by_class, "by_stage": by_stage, "by_severity": by_severity}

    def counts(self) -> Dict[str, int]:
        return {
            "documents": self.db.count("documents"),
            "chunks": self.db.count("chunks"),
            "runs": self.db.count("agent_runs"),
            "diagnoses": self.db.count("diagnoses"),
            "failures": self.db.count("diagnoses", "failure_detected=1"),
            "recovery_attempts": self.db.count("recovery_attempts"),
            "successful_recoveries": self.db.count("recovery_attempts", "status='success'"),
            "abstentions": self.db.count("abstentions"),
            "faults": self.db.count("fault_injections"),
            "experiments": self.db.count("experiments"),
            "metrics": self.db.count("metrics"),
        }


def _enum_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


__all__ = ["Repository", "Database", "decode", "Chunk", "Document", "RetrievedChunk", "RetrievalDiagnostics"]