"""Runtime signal collection.

One :class:`SignalCollector` produces the full :class:`SignalBundle` for a
single pipeline execution.  Every signal is a plain number so that the rule
engine, the LLM diagnoser and the dashboard all read the same object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from ..models import (
    AgentStep,
    AssembledContext,
    Diagnosis,
    GenerationResult,
    QueryUnderstanding,
    RetrievalDiagnostics,
    RetrievedChunk,
    SignalBundle,
)
from ..settings import Settings, get_settings
from ..taxonomy import Stage
from ..textutil import capitalized_terms, estimate_tokens, mean, split_sentences, tokenize
from .metrics import (
    claim_support,
    constraint_coverage,
    context_coverage,
    conflict_score,
    duplicate_ratio,
    evidence_fragmentation,
    groundedness_score,
    repetition_score,
)


@dataclass
class QuerySignals:
    query_words: int = 0
    query_tokens: int = 0
    query_chars: int = 0
    question_marks: int = 0
    conjunctions: int = 0
    is_complex: bool = False
    is_ambiguous: bool = False
    ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    subquery_count: int = 0
    expected_subqueries: int = 0
    rewrite_present: bool = False
    rewrite_change_ratio: float = 0.0
    rewrite_drift: float = 0.0
    key_terms_dropped: List[str] = field(default_factory=list)
    keyword_count: int = 0
    entity_count: int = 0
    constraint_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RetrievalSignals:
    retrieved_count: int = 0
    top_k: int = 0
    candidate_pool: int = 0
    top_score: float = 0.0
    mean_score: float = 0.0
    median_score: float = 0.0
    score_gap: float = 0.0
    score_std: float = 0.0
    useful_ratio: float = 0.0
    top_overlap: float = 0.0
    mean_overlap: float = 0.0
    pre_rank_top_overlap: float = 0.0
    pre_rank_useful_ratio: float = 0.0
    ranking_coverage_loss: float = 0.0
    max_chunk_overlap: float = 0.0
    duplicate_ratio: float = 0.0
    source_diversity: int = 0
    unique_documents: int = 0
    recall: Optional[float] = None
    precision: Optional[float] = None
    mrr: Optional[float] = None
    relevant_retrieved: int = 0
    gold_count: int = 0
    relevant_below_cut: int = 0
    stale_ratio: float = 0.0
    superseded_ratio: float = 0.0
    latency_ms: float = 0.0
    method: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RankingSignals:
    reranker_applied: bool = False
    rank_changes: int = 0
    rank_regression_rate: float = 0.0
    mrr_delta_after_rerank: float = 0.0
    rerank_score_margin: float = 0.0
    irrelevant_at_rank1: bool = False
    relevant_at_rank1: bool = False
    rerank_flat: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ContextSignals:
    context_tokens: int = 0
    context_chars: int = 0
    chunk_count: int = 0
    source_diversity: int = 0
    context_coverage: float = 0.0
    constraint_coverage: float = 1.0
    unconstrained_constraints: int = 0
    fragmentation_ratio: float = 0.0
    conflict_score: float = 0.0
    conflict_count: int = 0
    relevant_buried: bool = False
    truncation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GenerationSignals:
    answer_words: int = 0
    answer_chars: int = 0
    groundedness: float = 1.0
    claim_count: int = 0
    unsupported_claims: int = 0
    min_claim_support: float = 1.0
    citation_coverage: float = 0.0
    citations: int = 0
    confidence: float = 0.5
    repetition_score: float = 0.0
    grounded_mode: bool = True
    provider: str = "mock"
    model: str = "mock"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AgentSignals:
    tool_calls: int = 0
    failed_actions: int = 0
    repeat_actions: int = 0
    duplicate_action_rate: float = 0.0
    steps: int = 0
    state_transitions: int = 0
    workflow_duration_ms: float = 0.0
    unused_tool_results: int = 0
    max_steps: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class InfraSignals:
    latency_ms: float = 0.0
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    timeouts: int = 0
    infra_errors: int = 0
    provider_errors: int = 0
    parse_errors: int = 0
    embedding_errors: int = 0
    retrieval_service_unavailable: bool = False
    service_status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class SignalCollector:
    """Builds a :class:`SignalBundle` from pipeline artefacts."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def collect(
        self,
        query: str,
        understanding: QueryUnderstanding,
        retrieved: Sequence[RetrievedChunk],
        diagnostics: RetrievalDiagnostics,
        context: Optional[AssembledContext] = None,
        generation: Optional[GenerationResult] = None,
        steps: Sequence[AgentStep] = (),
        budget: Optional[Dict[str, Any]] = None,
        infra: Optional[Dict[str, Any]] = None,
        gold_chunk_ids: Optional[Set[str]] = None,
        corpus_chunks: Optional[Sequence[Chunk]] = None,
        document_texts: Optional[Dict[str, str]] = None,
        citations: Optional[Sequence[str]] = None,
        run_id: str = "",
        agent_signals: Optional[Dict[str, Any]] = None,
    ) -> SignalBundle:
        understanding = understanding or QueryUnderstanding(original_query=query, normalized_query=query)
        diagnostics = diagnostics or RetrievalDiagnostics(top_k=self.settings.retrieval.top_k)
        query_signals = self._query_signals(query, understanding)
        retrieval_signals = self._retrieval_signals(
            query, retrieved, diagnostics, gold_chunk_ids, corpus_chunks
        )
        ranking_signals = self._ranking_signals(retrieved, diagnostics, gold_chunk_ids)
        context_signals = self._context_signals(
            query,
            understanding,
            context or AssembledContext(),
            retrieved,
            gold_chunk_ids,
            corpus_chunks,
            document_texts,
        )
        generation_signals = self._generation_signals(
            query,
            generation or GenerationResult(answer="", grounded=False, groundedness=0.0),
            context or AssembledContext(),
            citations,
        )
        agent_payload = agent_signals or self._agent_signals(steps)
        agent = AgentSignals(**agent_payload)
        infra_signals = self._infra_signals(infra or {}, steps, diagnostics)

        return SignalBundle(
            run_id=run_id,
            query=query_signals.to_dict(),
            retrieval=retrieval_signals.to_dict(),
            ranking=ranking_signals.to_dict(),
            context=context_signals.to_dict(),
            generation=generation_signals.to_dict(),
            agent=agent.to_dict(),
            infrastructure=infra_signals.to_dict(),
            budget=dict(budget or {}),
        )

    # ------------------------------------------------------------------
    def _query_signals(self, query: str, u: QueryUnderstanding) -> QuerySignals:
        tokens = tokenize(query)
        words = query.split()
        conjunctions = sum(
            1 for w in words if w.lower() in {"and", "or", "also", "plus", "as", "then", "while", "but"}
        )
        rewrite_change = 0.0
        dropped: List[str] = []
        if u.rewrite:
            original = set(tokens)
            rewritten = set(tokenize(u.rewrite))
            if original:
                rewrite_change = len(rewritten - original) / len(original)
                dropped = sorted(original - rewritten)[:8]
        return QuerySignals(
            query_words=len(words),
            query_tokens=len(tokens),
            query_chars=len(query),
            question_marks=query.count("?"),
            conjunctions=conjunctions,
            is_complex=u.is_complex,
            is_ambiguous=u.is_ambiguous,
            ambiguous=u.is_ambiguous,
            ambiguity_reason=u.ambiguity_reason,
            subquery_count=len(u.subqueries),
            expected_subqueries=2 if u.is_complex else 1,
            rewrite_present=bool(u.rewrite),
            rewrite_change_ratio=round(rewrite_change, 6),
            rewrite_drift=round(u.rewrite_drift, 6),
            key_terms_dropped=dropped,
            keyword_count=len(u.keywords),
            entity_count=len(u.entities),
            constraint_count=len(u.constraints),
        )

    def _retrieval_signals(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        diagnostics: RetrievalDiagnostics,
        gold: Optional[Set[str]],
        corpus_chunks: Optional[Sequence[Chunk]] = None,
    ) -> RetrievalSignals:
        from .metrics import mrr, precision_at_k, recall_at_k

        q_tokens = set(tokenize(query))
        overlaps = [
            (len(q_tokens & set(tokenize(r.content))) / len(q_tokens)) if q_tokens else 0.0
            for r in retrieved
        ]
        ids = [r.chunk_id for r in retrieved]
        recall = precision = mrr_value = None
        relevant_retrieved = 0
        below_cut = 0
        # gold ids that no longer exist in the corpus (e.g. the fault removed the
        # document) must not be counted as missed retrievals
        if corpus_chunks:
            existing = {c.chunk_id for c in corpus_chunks}
            gold = gold & existing if gold else set()
        if gold:
            recall = recall_at_k(ids, gold)
            precision = precision_at_k(ids, gold)
            mrr_value = mrr(ids, gold)
            relevant_retrieved = len(gold & set(ids))
            below_cut = sum(
                1 for r in retrieved if r.chunk_id in gold and r.rank > diagnostics.top_k
            )
        return RetrievalSignals(
            retrieved_count=len(retrieved),
            top_k=diagnostics.top_k,
            candidate_pool=diagnostics.candidate_pool,
            top_score=diagnostics.top_score,
            mean_score=diagnostics.mean_score,
            median_score=diagnostics.median_score,
            score_gap=diagnostics.score_gap,
            score_std=diagnostics.score_std,
            useful_ratio=diagnostics.useful_ratio,
            top_overlap=diagnostics.top_overlap or (max(overlaps) if overlaps else 0.0),
            mean_overlap=diagnostics.mean_overlap,
            pre_rank_top_overlap=diagnostics.pre_rank_top_overlap or diagnostics.top_overlap,
            pre_rank_useful_ratio=diagnostics.pre_rank_useful_ratio or diagnostics.useful_ratio,
            ranking_coverage_loss=round(
                max(0.0, diagnostics.pre_rank_top_overlap - diagnostics.top_overlap), 6
            ),
            max_chunk_overlap=round(mean(overlaps), 6) if overlaps else 0.0,
            duplicate_ratio=duplicate_ratio(ids),
            source_diversity=diagnostics.source_diversity,
            unique_documents=diagnostics.unique_documents,
            recall=recall,
            precision=precision,
            mrr=mrr_value,
            relevant_retrieved=relevant_retrieved,
            gold_count=len(gold) if gold else 0,
            relevant_below_cut=below_cut,
            stale_ratio=_stale_ratio(retrieved),
            superseded_ratio=_superseded_ratio(retrieved),
            latency_ms=round(diagnostics.latency_ms, 3),
            method=diagnostics.method,
        )

    def _ranking_signals(
        self,
        retrieved: Sequence[RetrievedChunk],
        diagnostics: RetrievalDiagnostics,
        gold: Optional[Set[str]],
    ) -> RankingSignals:
        regressions = 0
        compared = 0
        for r in retrieved:
            if r.original_rank is None:
                continue
            compared += 1
            if r.rank > r.original_rank:
                regressions += 1
        rate = round(regressions / compared, 6) if compared else 0.0
        mrr_delta = 0.0
        irrelevant_top = False
        relevant_top = False
        if gold and retrieved:
            from .metrics import mrr

            original_order = sorted(retrieved, key=lambda r: (r.original_rank or r.rank))
            mrr_before = mrr([r.chunk_id for r in original_order], gold)
            mrr_after = mrr([r.chunk_id for r in retrieved], gold)
            mrr_delta = round(mrr_after - mrr_before, 6)
            irrelevant_top = retrieved[0].chunk_id not in gold
            relevant_top = retrieved[0].chunk_id in gold
        rerank_scores = [float(r.rerank_score) for r in retrieved if r.rerank_score is not None]
        margin = diagnostics.rerank_score_margin
        return RankingSignals(
            reranker_applied=diagnostics.reranker_applied,
            rank_changes=diagnostics.rank_changes,
            rank_regression_rate=rate,
            mrr_delta_after_rerank=mrr_delta,
            rerank_score_margin=margin,
            irrelevant_at_rank1=irrelevant_top,
            relevant_at_rank1=relevant_top,
            rerank_flat=bool(len(rerank_scores) > 1 and (max(rerank_scores) - min(rerank_scores)) < 1e-6),
        )

    def _context_signals(
        self,
        query: str,
        understanding: QueryUnderstanding,
        context: AssembledContext,
        retrieved: Sequence[RetrievedChunk],
        gold: Optional[Set[str]],
        corpus_chunks: Optional[Sequence[Chunk]] = None,
        document_texts: Optional[Dict[str, str]] = None,
    ) -> ContextSignals:
        item_texts = [[item.text] for item in context.items]
        frag = evidence_fragmentation(
            query,
            corpus_chunks or [],
            [item.chunk.chunk_id for item in context.items],
            document_texts or {},
        )
        conflict, conflicts = conflict_score(item_texts)
        cov_value, missing = constraint_coverage(query, understanding.constraints, context.text)
        coverage_value = context_coverage(query, context.text, understanding.subqueries, understanding.constraints)
        buried = False
        if gold and context.items:
            positions = [i.position for i in context.items if i.chunk.chunk_id in gold]
            if positions and max(positions) > max(1, len(context.items) // 2):
                buried = True
        return ContextSignals(
            context_tokens=context.token_estimate,
            context_chars=context.char_estimate,
            chunk_count=len(context.items),
            source_diversity=context.source_diversity,
            context_coverage=coverage_value,
            constraint_coverage=cov_value,
            unconstrained_constraints=len(missing),
            fragmentation_ratio=frag,
            conflict_score=conflict,
            conflict_count=len(conflicts),
            relevant_buried=buried,
            truncation_applied=context.truncation_applied,
        )

    def _generation_signals(
        self,
        query: str,
        generation: GenerationResult,
        context: AssembledContext,
        citations: Optional[Sequence[str]] = None,
    ) -> GenerationSignals:
        markers = citations or [f"S{i}" for i in range(1, len(context.items) + 1)]
        scores = groundedness_score(
            question=query,
            answer=generation.answer,
            context=[item.text for item in context.items],
            citation_markers=markers,
            support_threshold=self.settings.thresholds.min_claim_support,
        )
        return GenerationSignals(
            answer_words=len(generation.answer.split()),
            answer_chars=len(generation.answer),
            groundedness=scores["groundedness"],
            claim_count=scores["claim_count"],
            unsupported_claims=len(scores["unsupported_claims"]),
            min_claim_support=scores["min_claim_support"],
            citation_coverage=scores["citation_coverage"],
            citations=len(scores["citations"]),
            confidence=generation.confidence,
            repetition_score=repetition_score(generation.answer),
            grounded_mode=generation.grounded_mode,
            provider=generation.provider,
            model=generation.model,
        )

    def _agent_signals(self, steps: Sequence[AgentStep]) -> Dict[str, Any]:
        """Repeat/failure detection over the *agent's own* actions.

        Pipeline stages (retrieval, generation, ...) are excluded: repeating a
        *recovery* action is the bounded loop's job, not an agent-control bug.
        """
        agent_steps = [
            s
            for s in steps
            if s.tool_name or s.stage in {"query_understanding", "query_decomposition", "agent_control"}
        ]
        signatures = [s.action_signature for s in agent_steps if s.action_signature]
        repeats = len(signatures) - len(set(signatures))
        tool_calls = sum(1 for s in steps if s.tool_name)
        failed = sum(1 for s in agent_steps if s.status not in {"ok", "success"})
        transitions = len({s.stage for s in steps})
        return {
            "tool_calls": tool_calls,
            "failed_actions": failed,
            "repeat_actions": max(0, repeats),
            "duplicate_action_rate": round(repeats / len(signatures), 6) if signatures else 0.0,
            "steps": len(steps),
            "state_transitions": transitions,
            "workflow_duration_ms": round(sum(s.latency_ms for s in steps), 3),
            "unused_tool_results": 0,
            "max_steps": self.settings.agent.max_steps,
        }

    def _infra_signals(
        self, infra: Dict[str, Any], steps: Sequence[AgentStep], diagnostics: RetrievalDiagnostics
    ) -> InfraSignals:
        stage_latency = {
            k: round(float(v), 3) for k, v in (infra.get("stage_latency_ms") or {}).items()
        }
        errors = [s.error for s in steps if s.error]
        status = "degraded" if errors or infra.get("timeouts") else "ok"
        return InfraSignals(
            latency_ms=round(float(infra.get("latency_ms", diagnostics.latency_ms)), 3),
            stage_latency_ms=stage_latency,
            timeouts=int(infra.get("timeouts", 0)),
            infra_errors=len(errors) + int(infra.get("infra_errors", 0)),
            provider_errors=int(infra.get("provider_errors", 0)),
            parse_errors=int(infra.get("parse_errors", 0)),
            embedding_errors=int(infra.get("embedding_errors", 0)),
            retrieval_service_unavailable=bool(infra.get("retrieval_service_unavailable", False)),
            service_status=str(infra.get("service_status", status)),
        )


def _stale_ratio(retrieved: Sequence[RetrievedChunk]) -> float:
    """Fraction of retrieved chunks that are flagged as outdated versions."""
    if not retrieved:
        return 0.0
    stale = 0
    for rc in retrieved:
        meta = rc.metadata or {}
        if meta.get("stale") or meta.get("injected_stale"):
            stale += 1
        elif int(meta.get("version", 1) or 1) < int(meta.get("latest_version", 1) or 1):
            stale += 1
    return round(stale / len(retrieved), 6)


def _superseded_ratio(retrieved: Sequence[RetrievedChunk]) -> float:
    """Fraction of retrieved chunks explicitly marked as superseded."""
    if not retrieved:
        return 0.0
    marked = sum(1 for rc in retrieved if (rc.metadata or {}).get("superseded"))
    return round(marked / len(retrieved), 6)


def flatten_signals(bundle: SignalBundle) -> Dict[str, Any]:
    """Flat name→value view used by the rule engine and the LLM diagnoser."""
    return bundle.flat()


def signals_for_diagnoser(bundle: SignalBundle, max_items: int = 8) -> Dict[str, Any]:
    """Compact, JSON-serialisable signal view handed to the Tier-2 diagnoser."""
    flat = bundle.flat()
    interesting = {
        "retrieval": dict(bundle.retrieval),
        "ranking": dict(bundle.ranking),
        "context": dict(bundle.context),
        "generation": dict(bundle.generation),
        "agent": dict(bundle.agent),
        "infrastructure": dict(bundle.infrastructure),
        "query": {
            k: v
            for k, v in bundle.query.items()
            if k in {"query_words", "is_ambiguous", "rewrite_drift", "subquery_count", "rewrite_change_ratio"}
        },
    }
    for section in interesting.values():
        for key, value in list(section.items()):
            if isinstance(value, (list, tuple)) and len(value) > max_items:
                section[key] = value[:max_items]
            elif isinstance(value, dict):
                section.pop(key)
    return {"flat": {k: v for k, v in flat.items() if isinstance(v, (int, float, bool, str))}, "sections": interesting}


__all__ = [
    "SignalCollector",
    "QuerySignals",
    "RetrievalSignals",
    "RankingSignals",
    "ContextSignals",
    "GenerationSignals",
    "AgentSignals",
    "InfraSignals",
    "flatten_signals",
    "signals_for_diagnoser",
]