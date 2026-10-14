"""Unified orchestrator.

This is the single entry point used by the API, the dashboard, the demo script
and the experiment engine.  It supports the three comparison modes:

* ``mode1_normal_rag``            - plain RAG (no agentic control, no recovery)
* ``mode2_agentic_no_recovery``   - agentic RAG, diagnosis reported but no action
* ``mode3_agentic_with_recovery`` - full MONITOR→DETECT→DIAGNOSE→RECOVER→VERIFY
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..db.repository import Repository
from ..embeddings.base import Embedder
from ..embeddings.factory import build_embedder
from ..faults.definitions import FaultRegistry
from ..faults.injector import FaultInjector
from ..ingestion.chunker import Chunker
from ..ingestion.ingest import _Ingestor
from ..llm.factory import build_provider
from ..models import (
    Diagnosis,
    FaultInjection,
    PipelineRun,
    RunMode,
    RunStatus,
    utcnow,
)
from ..observability.trace import TraceCollector, get_logger
from ..recovery.budget import BudgetTracker
from ..recovery.executor import ABSTENTION_MESSAGE, RecoveryController
from ..retrieval.diagnostics import retrieval_ground_truth
from ..retrieval.retriever import HybridRetriever
from ..settings import Settings, get_settings
from ..taxonomy import Stage
from .nodes import STAGE_SEQUENCE, run_stages_from, stage_retrieval
from .runtime import Runtime
from .state import PipelineState

logger = get_logger(__name__)


@dataclass
class OrchestratorResult:
    run: PipelineRun
    state: PipelineState
    runtime: Runtime
    trace: TraceCollector

    @property
    def answer(self) -> str:
        return self.run.answer

    @property
    def abstained(self) -> bool:
        return self.run.abstained

    @property
    def recovered(self) -> bool:
        return self.run.recovered

    @property
    def diagnosis(self) -> Optional[Diagnosis]:
        return self.run.final_diagnosis

    def to_dict(self) -> Dict[str, Any]:
        return self.run.model_dump(mode="json")


class RAGOrchestrator:
    """Owns the runtime and executes end-to-end agentic RAG requests."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        repository: Optional[Repository] = None,
        embedder: Optional[Embedder] = None,
        provider: Optional[Any] = None,
        chunks: Optional[Sequence[Any]] = None,
    ):
        self.settings = settings or get_settings()
        self.repository = repository or Repository(
            __import__(
                "agentic_rag_recovery.db.database", fromlist=["Database"]
            ).Database(self.settings.db_path())
        )
        self.embedder = embedder or build_embedder(settings=self.settings)
        all_chunks = list(chunks) if chunks is not None else self.repository.get_chunks()
        self.retriever = HybridRetriever(all_chunks, self.embedder, self.settings)
        self.provider = provider
        self.ingestor = _Ingestor(self.repository, self.settings, Chunker(self.settings.chunking))
        self.faults = FaultInjector(self.runtime_base())
        self.controller = RecoveryController(self.runtime_base(), settings=self.settings)

    # ------------------------------------------------------------------
    def runtime_base(self) -> Runtime:
        runtime = Runtime(
            settings=self.settings,
            repository=self.repository,
            embedder=self.embedder,
            retriever=self.retriever,
            provider=self.provider,
            ingestor=self.ingestor,
        )
        runtime.baseline_chunks = [c.model_copy(deep=True) for c in self.retriever.chunks]
        return runtime

    def build_runtime(self) -> Runtime:
        runtime = self.runtime_base()
        if self.provider is not None:
            runtime.diagnoser_provider = None
        return runtime

    # ------------------------------------------------------------------
    def reindex(self) -> "RAGOrchestrator":
        """Reload chunks from SQLite and rebuild both indexes."""
        self.retriever.rebuild(self.repository.get_chunks())
        return self

    def answer(
        self,
        query: str,
        mode: RunMode | str = RunMode.MODE_3_AGENTIC_WITH_RECOVERY,
        faults: Optional[Sequence[FaultInjection | str | Dict[str, Any]]] = None,
        fault_parameters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        retrieval_method: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        experiment_id: Optional[str] = None,
        gold_chunk_ids: Optional[Sequence[str]] = None,
        gold_document_ids: Optional[Sequence[str]] = None,
        relevant_documents: Optional[Sequence[str]] = None,
        answer_keywords: Optional[Sequence[str]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        persist: bool = True,
        run_baseline: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ) -> OrchestratorResult:
        """Execute one full request and return the persisted :class:`PipelineRun`."""
        mode = RunMode(mode) if not isinstance(mode, RunMode) else mode
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        if len(query) > 4000:
            raise ValueError("query exceeds the 4000 character limit")

        runtime = self.build_runtime()
        if top_k:
            runtime.knobs.top_k = int(top_k)
        if retrieval_method:
            runtime.knobs.retrieval_method = retrieval_method
        if use_rerank is not None:
            runtime.knobs.use_rerank = bool(use_rerank)
        trace = TraceCollector()
        runtime.set_trace(trace)
        self.faults = FaultInjector(runtime)
        self.controller = RecoveryController(runtime, settings=self.settings)

        state = PipelineState(query=query, mode=mode, experiment_id=experiment_id)

        # ---- ground truth (experiments only; never fed to the diagnoser) ---
        if answer_keywords:
            runtime.answer_keywords = list(answer_keywords)
        doc_by_name: Dict[str, str] = {}
        for document in self.repository.list_documents():
            doc_by_name[Path(document.source).name.lower()] = document.document_id
        resolved_docs: set = set()
        if relevant_documents:
            resolved_docs = {
                doc_by_name[Path(name).name.lower()]
                for name in relevant_documents
                if Path(name).name.lower() in doc_by_name
            }
        if resolved_docs:
            state.gold_document_ids = resolved_docs
            runtime.gold_document_ids = resolved_docs
        if gold_chunk_ids:
            state.gold_chunk_ids = set(gold_chunk_ids)
            runtime.gold_chunk_ids = set(gold_chunk_ids)
        elif self.repository.chunk_count():
            state.gold_chunk_ids = self._resolve_gold(query, runtime.answer_keywords, resolved_docs)
            runtime.gold_chunk_ids = set(state.gold_chunk_ids)
        elif not state.gold_document_ids:
            gold = retrieval_ground_truth(query, self.retriever.chunks, runtime.answer_keywords)
            state.gold_chunk_ids = set(gold)
            runtime.gold_chunk_ids = set(gold)
        if not state.gold_document_ids and state.gold_chunk_ids:
            state.gold_document_ids = {
                c.document_id for c in self.retriever.chunks if c.chunk_id in state.gold_chunk_ids
            }
            runtime.gold_document_ids = state.gold_document_ids

        # ---- faults -------------------------------------------------------
        active: List[FaultInjection] = []
        for item in faults or []:
            if isinstance(item, FaultInjection):
                active.append(item)
            elif isinstance(item, str):
                active.append(self.faults.build(item, fault_parameters or {}, experiment_id))
            elif isinstance(item, dict):
                active.append(
                    self.faults.build(
                        item.get("fault_type", ""),
                        {**(fault_parameters or {}), **(item.get("parameters") or {})},
                        experiment_id,
                    )
                )
        for fault in active:
            self.faults.activate(fault, query)
        state.active_faults = [f.ground_truth() | {"active": True} for f in active]
        state.ground_truth = ground_truth or _default_ground_truth(active)

        try:
            budget = BudgetTracker(self.settings.budget)

            # ---- MODE 1: plain RAG, no agentic control -----------------
            if mode is RunMode.MODE_1_NORMAL_RAG:
                self._execute_mode1(state, runtime, budget)
            # ---- MODE 2 / 3: agentic ----------------------------------
            else:
                self._execute_agentic(state, runtime, budget, mode, run_baseline=run_baseline)

            state.finished_at = utcnow()
            run = state.to_run(
                labels=labels
                or {
                    "results": self.settings.labels.get("results", "DEMO / MOCK RESULTS"),
                    "provider": getattr(self.provider or runtime.provider, "name", "unknown"),
                    "mode": mode.value,
                }
            )
            run.steps = list(trace.steps)
            run.spans = list(trace.spans)
            run.total_latency_ms = round(state.total_latency_ms, 3)
            run.added_latency_ms = round(
                max(0.0, run.total_latency_ms - run.baseline_latency_ms), 3
            )
            if persist and self.repository is not None:
                self._persist(run, state, active, budget)
            return OrchestratorResult(run=run, state=state, runtime=runtime, trace=trace)
        finally:
            self.faults.release()
            runtime.restore_clean_state()

    # ------------------------------------------------------------------
    # Mode 1
    # ------------------------------------------------------------------
    def _execute_mode1(self, state: PipelineState, runtime: Runtime, budget: BudgetTracker) -> None:
        runtime.knobs.force_decomposition = False
        runtime.knobs.use_rerank = None
        run_stages_from(Stage.QUERY_UNDERSTANDING, state, runtime)
        state.metadata["mode_note"] = "MODE 1: retrieval-augmented generation only"
        budget.consume_baseline(state.total_latency_ms, state.llm_calls, state.tokens_used, state.cost)
        state.baseline_latency_ms = state.total_latency_ms
        state.baseline_answer = state.answer()
        state.baseline_signals = state.signals
        state.metadata["agent_signals_snapshot"] = _agent_signals(self.settings, runtime)

    # ------------------------------------------------------------------
    # Modes 2 and 3
    # ------------------------------------------------------------------
    def _execute_agentic(
        self,
        state: PipelineState,
        runtime: Runtime,
        budget: BudgetTracker,
        mode: RunMode,
        run_baseline: bool = True,
    ) -> None:
        # ---- happy-path execution (this is the "normal" pipeline) --------
        run_stages_from(Stage.QUERY_UNDERSTANDING, state, runtime)

        if state.signals is None:
            from .nodes import stage_validation

            stage_validation(state, runtime)

        # Freeze the agent-control signals from the *original* agent workflow:
        # repeats introduced by our own recovery loop are not agent failures.
        state.metadata["agent_signals_snapshot"] = _agent_signals(self.settings, runtime)

        state.baseline_latency_ms = round(state.total_latency_ms, 3)
        state.baseline_answer = state.answer()
        state.baseline_signals = state.signals
        state.baseline_generation = state.generation
        if run_baseline:
            budget.consume_baseline(
                state.total_latency_ms, state.llm_calls, state.tokens_used, state.cost
            )
        state.metadata["baseline_latency_ms"] = state.baseline_latency_ms

        # ---- MONITOR → DETECT → DIAGNOSE -------------------------------
        engine = runtime.ensure_diagnosis_engine()
        diagnosis = engine.diagnose(
            state.signals, run_id=state.run_id, query=state.query
        )
        diagnosis.run_id = state.run_id
        state.diagnoses.append(diagnosis)
        runtime.trace.step(
            "diagnosis",
            "diagnose",
            input_summary=f"tier-1 rules + tier-2 LLM (threshold "
            f"{self.settings.diagnosis.rule_confidence_threshold})",
            output_summary=(
                f"{diagnosis.tier.value}: {diagnosis.class_value}/{diagnosis.subtype} "
                f"confidence={diagnosis.confidence}"
            ),
            latency_ms=diagnosis.latency_ms,
            failure_detected=diagnosis.failure_detected,
            severity=diagnosis.severity.value if diagnosis.severity else None,
        )

        if not diagnosis.failure_detected:
            state.metadata["stop_reason"] = "healthy pipeline - no recovery required"
            return

        if mode is RunMode.MODE_2_AGENTIC_NO_RECOVERY:
            state.metadata["stop_reason"] = "MODE 2: diagnosis reported, recovery disabled"
            runtime.trace.step(
                "recovery",
                "disabled",
                output_summary="MODE 2 deliberately does not apply recovery",
            )
            return

        # ---- RECOVER → VERIFY (bounded) --------------------------------
        outcome = self.controller.run(state, diagnosis, budget)
        state.metadata["recovery_stop_reason"] = outcome.stop_reason
        state.metadata["recovery_plans"] = outcome.plans

        if outcome.recovered:
            state.metadata["recovered"] = True
            # re-diagnose so the trace shows the post-recovery health
            post = engine.diagnose(state.signals, run_id=state.run_id, query=state.query)
            post.run_id = state.run_id
            post.metadata["post_recovery"] = True
            state.diagnoses.append(post)
        else:
            state.abstention = outcome.abstention
            state.generation = state.generation.model_copy(update={"answer": ABSTENTION_MESSAGE})
            runtime.trace.step(
                "abstention",
                "abstain",
                output_summary=ABSTENTION_MESSAGE,
                reason=outcome.abstention.reason if outcome.abstention else "",
            )

    # ------------------------------------------------------------------
    def _resolve_gold(
        self, query: str, answer_keywords: Sequence[str], documents: set
    ) -> set:
        """Identify the *answer-bearing* chunks.

        Marking every chunk of a document as gold would make recall structurally
        <1 for any top_k, so only the chunks that actually carry the answer
        keywords (or, failing that, the best-matching chunk of the relevant
        document) are treated as gold.
        """
        from ..textutil import tokenize

        pool = [c for c in self.retriever.chunks if not documents or c.document_id in documents]
        if not pool:
            return set()
        keywords = [k.lower() for k in answer_keywords if k]
        if keywords:
            scored = sorted(
                ((sum(1 for k in keywords if k in c.content.lower()), c.chunk_id) for c in pool),
                reverse=True,
            )
            best = scored[0][0] if scored else 0
            if best > 0:
                gold = {cid for hits, cid in scored if hits >= max(1, best)}
                # never return more gold chunks than the retrieval cut can hold
                if len(gold) > self.settings.retrieval.top_k:
                    gold = {cid for hits, cid in scored if hits >= best}
                return gold
        query_tokens = set(tokenize(query))
        scored = sorted(
            (
                (len(query_tokens & set(tokenize(c.content))), c.chunk_id)
                for c in pool
            ),
            reverse=True,
        )
        return {scored[0][1]} if scored else set()

    # ------------------------------------------------------------------
    def _persist(
        self,
        run: PipelineRun,
        state: PipelineState,
        faults: List[FaultInjection],
        budget: BudgetTracker,
    ) -> None:
        run.metrics = self.compute_metrics(run, state, budget)
        self.repository.save_run(run)
        for fault in faults:
            self.repository.record_fault(fault, run_id=run.run_id)
        self.repository.record_metrics(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            group_name="per_run",
            label=run.labels.get("results", ""),
            metrics=run.metrics,
        )

    # ------------------------------------------------------------------
    def compute_metrics(
        self, run: PipelineRun, state: PipelineState, budget: BudgetTracker
    ) -> Dict[str, Any]:
        """Per-request metrics used by both the dashboard and the experiments."""
        signals = state.signals
        baseline = state.baseline_signals
        retrieval = signals.retrieval if signals else {}
        generation = signals.generation if signals else {}
        context = signals.context if signals else {}
        ranking = signals.ranking if signals else {}
        base_gen = baseline.generation if baseline else {}
        base_ret = baseline.retrieval if baseline else {}

        metrics: Dict[str, Any] = {
            "groundedness": float(generation.get("groundedness", 0.0) or 0.0),
            "baseline_groundedness": float(base_gen.get("groundedness", 0.0) or 0.0),
            "citation_coverage": float(generation.get("citation_coverage", 0.0) or 0.0),
            "context_coverage": float(context.get("context_coverage", 0.0) or 0.0),
            "useful_ratio": float(retrieval.get("useful_ratio", 0.0) or 0.0),
            "top_score": float(retrieval.get("top_score", 0.0) or 0.0),
            "recalled_count": int(retrieval.get("retrieved_count", 0) or 0),
            "added_latency_ms": round(max(0.0, state.total_latency_ms - state.baseline_latency_ms), 3),
            "total_latency_ms": round(state.total_latency_ms, 3),
            "baseline_latency_ms": round(state.baseline_latency_ms, 3),
            "llm_calls": int(state.llm_calls),
            "tokens_used": int(state.tokens_used),
            "cost": round(state.cost, 8),
            "recovery_attempts": len(state.recoveries),
            "successful_recoveries": sum(
                1 for a in state.recoveries if a.status.value == "success"
            ),
            "abstained": int(run.abstained),
            "recovered": int(run.recovered),
            "failures_detected": sum(1 for d in state.diagnoses if d.failure_detected),
            "rank_changes": int(ranking.get("rank_changes", 0) or 0),
            "budget_attempts_used": budget.attempts_used,
            "budget_tokens_used": budget.tokens_used,
            "cost_per_recovery": round(
                budget.cost / max(1, sum(1 for a in state.recoveries if a.status.value == "success")), 8
            ),
        }
        for key in ("recall", "precision", "mrr"):
            if retrieval.get(key) is not None:
                metrics[f"retrieval_{key}"] = float(retrieval[key])
            if base_ret.get(key) is not None:
                metrics[f"baseline_retrieval_{key}"] = float(base_ret[key])
        return metrics

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __import__("agentic_rag_recovery", fromlist=["__version__"]).__version__,
            "settings": {
                "mode": self.settings.mode,
                "provider": self.settings.generation.provider,
                "model": self.settings.generation.model,
                "retrieval_method": self.settings.retrieval.method,
                "top_k": self.settings.retrieval.top_k,
                "chunk_size": self.settings.chunking.chunk_size,
            },
            "knowledge_base": {
                "documents": self.repository.document_count(),
                "chunks": self.retriever.size,
            },
            "retrieval": self.retriever.health(),
            "labels": dict(self.settings.labels),
            "budget": {
                "max_recovery_attempts": self.settings.budget.max_recovery_attempts,
                "max_total_latency_ms": self.settings.budget.max_total_latency_ms,
                "max_token_budget": self.settings.budget.max_token_budget,
                "max_llm_calls": self.settings.budget.max_llm_calls,
            },
        }

    def taxonomy(self) -> Dict[str, Any]:
        from ..taxonomy import FailureClass, FailureTaxonomy

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for name, definition in sorted(FailureTaxonomy.all().items()):
            from ..taxonomy import class_for_subtype

            key = class_for_subtype(name).value
            grouped.setdefault(key, []).append(
                {
                    "subtype": name,
                    "stage": definition.stage.value,
                    "description": definition.description,
                    "severity": definition.default_severity.value,
                    "recovery_actions": [a.value for a in definition.recovery_actions],
                    "evidence_hints": list(definition.evidence_hints),
                    "reusable_stage": definition.reusable_stage.value
                    if definition.reusable_stage
                    else None,
                }
            )
        return {
            "stages": [s.value for s in Stage],
            "classes": {c.value for c in FailureClass},
            "subtypes": grouped,
        }


def _default_ground_truth(faults: List[FaultInjection]) -> Dict[str, Any]:
    if not faults:
        return {}
    fault = faults[0]
    return fault.ground_truth()


def _agent_signals(settings: Settings, runtime: Runtime) -> Dict[str, Any]:
    from ..signals.collector import SignalCollector

    return SignalCollector(settings)._agent_signals(runtime.trace.steps)


__all__ = ["RAGOrchestrator", "OrchestratorResult"]