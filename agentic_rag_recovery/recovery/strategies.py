"""Targeted recovery strategies.

Every strategy:
  * mutates only the knobs/state it needs,
  * declares the **earliest stage that must be re-run** (everything before it
    is reused verbatim - that is what makes recovery targeted rather than a
    full pipeline restart),
  * returns a :class:`StrategyOutcome` with an auditable rationale.

No strategy is ever allowed to widen a threshold or hide a signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm.base import LLMError
from ..observability.trace import get_logger
from ..pipeline.runtime import Runtime
from ..pipeline.state import PipelineState
from ..prompts import rewrite_prompt
from ..taxonomy import PIPELINE_STAGES, RecoveryAction, Stage
from ..textutil import clip, tokenize

logger = get_logger(__name__)

_PIPELINE_ORDER = list(PIPELINE_STAGES)


def _reused_before(rerun_from: Optional[Stage]) -> List[Stage]:
    """Stages whose results are reused verbatim when re-running from ``rerun_from``."""
    if rerun_from is None or rerun_from not in _PIPELINE_ORDER:
        return []
    return _PIPELINE_ORDER[: _PIPELINE_ORDER.index(rerun_from)]


@dataclass
class StrategyOutcome:
    applied: bool = False
    rerun_from: Optional[Stage] = None
    rationale: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    llm_calls: int = 0
    tokens: int = 0
    cost: float = 0.0
    reused_stages: List[Stage] = field(default_factory=list)


@dataclass
class RecoveryContext:
    state: PipelineState
    runtime: Runtime
    diagnosis: Any
    attempt_index: int
    failed_stages: List[Stage] = field(default_factory=list)
    history: List[str] = field(default_factory=list)


StrategyFn = Callable[[RecoveryContext], StrategyOutcome]
STRATEGIES: Dict[RecoveryAction, StrategyFn] = {}


def strategy(action: RecoveryAction) -> Callable[[StrategyFn], StrategyFn]:
    def decorator(fn: StrategyFn) -> StrategyFn:
        STRATEGIES[action] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Query-stage recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.QUERY_REWRITE)
def query_rewrite(ctx: RecoveryContext) -> StrategyOutcome:
    """Rewrite the query to close the vocabulary gap, then re-run retrieval.

    Query-understanding and decomposition results are **reused**; only retrieval,
    ranking, context assembly and generation are repeated.
    """
    state, runtime = ctx.state, ctx.runtime
    understanding = state.understanding
    original = state.query

    feedback = _rewrite_feedback(state, ctx)
    prompt = rewrite_prompt(original, feedback=feedback, failed_queries=ctx.history[-2:])
    try:
        result = runtime.ensure_provider().generate(
            prompt, system="You rewrite search queries; never answer the question."
        )
    except LLMError as exc:
        return StrategyOutcome(applied=False, error=f"rewrite provider error: {exc}")

    rewritten = (result.text or "").strip()
    if not rewritten or rewritten.lower() == original.lower():
        return StrategyOutcome(
            applied=False,
            error="provider returned an unusable rewrite",
            llm_calls=1,
            tokens=result.total_tokens,
            cost=result.cost,
        )

    from ..pipeline.nodes import _drift

    if understanding is not None:
        understanding.rewrite = rewritten
        understanding.rewrite_drift = _drift(original, rewritten)
    state.metadata.setdefault("rewrites", []).append(rewritten)
    runtime.trace.step(
        Stage.QUERY_UNDERSTANDING.value,
        "rewrite",
        input_summary=clip(original, 140),
        output_summary=clip(rewritten, 220),
        latency_ms=result.latency_ms,
        drift=round(understanding.rewrite_drift, 4) if understanding else 0.0,
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.RETRIEVAL,
        rationale=(
            "expanded the query with alternative surface forms; query understanding and "
            "decomposition are reused, retrieval onward is re-run"
        ),
        detail={"rewrite": rewritten, "drift": understanding.rewrite_drift if understanding else None},
        llm_calls=1,
        tokens=result.total_tokens,
        cost=result.cost,
        reused_stages=[Stage.QUERY_UNDERSTANDING, Stage.QUERY_DECOMPOSITION],
    )


def _rewrite_feedback(state: PipelineState, ctx: RecoveryContext) -> str:
    signals = state.signals
    if signals is None:
        return "the previous retrieval did not surface evidence for the question"
    parts: List[str] = []
    retrieval = signals.retrieval
    if retrieval.get("retrieved_count", 0) == 0:
        parts.append("the previous retrieval returned zero results")
    if float(retrieval.get("useful_ratio", 1.0)) < 0.4:
        parts.append(
            f"only {retrieval.get('useful_ratio')} of retrieved chunks shared vocabulary with the question"
        )
    if float(retrieval.get("top_score", 0.0)) < 0.2:
        parts.append(f"the best similarity score was only {retrieval.get('top_score')}")
    if signals.query.get("key_terms_dropped"):
        parts.append(f"terms already lost: {', '.join(signals.query['key_terms_dropped'])}")
    return "; ".join(parts) or "the previous retrieval did not surface evidence for the question"


# ---------------------------------------------------------------------------
# Retrieval-stage recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.HYBRID_SEARCH)
def hybrid_search(ctx: RecoveryContext) -> StrategyOutcome:
    """Fuse dense + lexical retrieval (RRF). Retrieval onward is re-run."""
    runtime, state = ctx.runtime, ctx.state
    previous = runtime.knobs.retrieval_method or runtime.settings.retrieval.method
    if previous == "hybrid":
        return StrategyOutcome(
            applied=False,
            error="hybrid retrieval is already active; escalating instead of repeating",
        )
    runtime.knobs.retrieval_method = "hybrid"
    runtime.trace.step(
        Stage.RETRIEVAL.value,
        "switch_method",
        input_summary=f"method={previous}",
        output_summary="method=hybrid",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.RETRIEVAL,
        rationale=(
            f"switched retrieval from '{previous}' to reciprocal-rank fusion of dense and "
            "BM25 signals; query understanding is reused"
        ),
        detail={"previous_method": previous, "new_method": "hybrid"},
        reused_stages=[Stage.QUERY_UNDERSTANDING, Stage.QUERY_DECOMPOSITION],
    )


@strategy(RecoveryAction.EXPAND_TOP_K)
def expand_top_k(ctx: RecoveryContext) -> StrategyOutcome:
    """Widen the cut-off (and the candidate pool) for one re-run."""
    runtime, state = ctx.runtime, ctx.state
    settings = runtime.settings
    current = int(runtime.knobs.top_k or settings.retrieval.top_k)
    # escalation is bounded by a fixed multiple of the configured cut-off, so a
    # runaway loop can never inflate top-k without limit
    ceiling = max(settings.retrieval.candidate_k, settings.retrieval.top_k * 3)
    if current >= ceiling:
        return StrategyOutcome(
            applied=False,
            error=f"top_k already at the ceiling ({current}); refusing to repeat",
        )
    new_top_k = min(ceiling, max(current * 2, current + 2))
    runtime.knobs.top_k = new_top_k
    runtime.knobs.candidate_k = max(
        int(runtime.knobs.candidate_k or settings.retrieval.candidate_k), new_top_k * 2
    )
    if runtime.retriever is not None:
        runtime.retriever.inject.pop("top_k_cap", None)
    runtime.trace.step(
        Stage.RETRIEVAL.value,
        "expand_top_k",
        input_summary=f"top_k={current}",
        output_summary=f"top_k={new_top_k}",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.RETRIEVAL,
        rationale=(
            f"raised top-k from {current} to {new_top_k} so that relevant chunks below the "
            "previous cut can enter the context"
        ),
        detail={"previous_top_k": current, "new_top_k": new_top_k},
        reused_stages=[Stage.QUERY_UNDERSTANDING, Stage.QUERY_DECOMPOSITION],
    )


@strategy(RecoveryAction.BROADEN_RETRIEVAL)
def broaden_retrieval(ctx: RecoveryContext) -> StrategyOutcome:
    """Last retrieval resort: wider cut-off and hybrid fusion, reranking kept."""
    runtime = ctx.runtime
    settings = runtime.settings
    previous_top_k = int(runtime.knobs.top_k or settings.retrieval.top_k)
    runtime.knobs.top_k = max(previous_top_k * 2, settings.retrieval.candidate_k)
    runtime.knobs.candidate_k = max(
        int(runtime.knobs.candidate_k or settings.retrieval.candidate_k), runtime.knobs.top_k * 2
    )
    runtime.knobs.retrieval_method = "hybrid"
    if runtime.retriever is not None:
        runtime.retriever.inject.pop("corrupt_rerank", None)
        runtime.retriever.inject.pop("top_k_cap", None)
    runtime.trace.step(
        Stage.RETRIEVAL.value,
        "broaden",
        input_summary=f"top_k={previous_top_k}, method={settings.retrieval.method}",
        output_summary=f"top_k={runtime.knobs.top_k}, method=hybrid",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.RETRIEVAL,
        rationale=(
            "broadened the search: hybrid fusion over a much wider candidate pool; "
            "quality thresholds are untouched"
        ),
        detail={"top_k": runtime.knobs.top_k, "method": "hybrid"},
        reused_stages=[Stage.QUERY_UNDERSTANDING, Stage.QUERY_DECOMPOSITION],
    )


# ---------------------------------------------------------------------------
# Ranking-stage recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.RERANK)
def rerank(ctx: RecoveryContext) -> StrategyOutcome:
    """Re-order the *existing candidate pool*; retrieval is NOT repeated."""
    runtime, state = ctx.runtime, ctx.state
    if not state.retrieved_candidates:
        return StrategyOutcome(applied=False, error="no candidate pool to re-rank")
    runtime.knobs.use_rerank = True
    runtime.knobs.retrieval_method = None
    if runtime.retriever is not None:
        runtime.retriever.inject.pop("corrupt_rerank", None)
        runtime.retriever.inject.pop("top_k_cap", None)
    runtime.trace.step(
        Stage.RANKING.value,
        "rerank",
        input_summary=f"{len(state.retrieved_candidates)} cached candidate(s)",
        output_summary="re-running ranking with the healthy cross-encoder",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.RANKING,
        rationale=(
            "re-ran ranking over the already-retrieved candidate pool with a healthy "
            "cross-encoder; retrieval results are reused verbatim"
        ),
        detail={"candidates": len(state.retrieved_candidates)},
        reused_stages=[Stage.QUERY_UNDERSTANDING, Stage.QUERY_DECOMPOSITION, Stage.RETRIEVAL],
    )


# ---------------------------------------------------------------------------
# Context-stage recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.CHUNK_EXPANSION)
def chunk_expansion(ctx: RecoveryContext) -> StrategyOutcome:
    """Pull the neighbouring chunks so split evidence becomes contiguous."""
    runtime, state = ctx.runtime, ctx.state
    if runtime.retriever is None or not state.retrieved:
        return StrategyOutcome(applied=False, error="nothing to expand")
    runtime.knobs.chunk_expansion = True
    runtime.trace.step(
        Stage.CONTEXT_ASSEMBLY.value,
        "expand_chunks",
        input_summary=f"{len(state.retrieved)} retrieved chunk(s)",
        output_summary="pulling adjacent chunks of the same document",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.CONTEXT_ASSEMBLY,
        rationale=(
            "added the immediate neighbours of each retrieved chunk to repair fragmented "
            "evidence; retrieval and ranking are reused"
        ),
        detail={"expansions": [rc.chunk_id for rc in state.retrieved[:3]]},
        reused_stages=[
            Stage.QUERY_UNDERSTANDING,
            Stage.QUERY_DECOMPOSITION,
            Stage.RETRIEVAL,
            Stage.RANKING,
        ],
    )


@strategy(RecoveryAction.CONTEXT_EXPANSION)
def context_expansion(ctx: RecoveryContext) -> StrategyOutcome:
    """Remove an artificial context budget and add more evidence."""
    runtime, state = ctx.runtime, ctx.state
    previous_chunks = runtime.knobs.max_chunks
    previous_tokens = runtime.knobs.max_context_tokens
    runtime.knobs.max_chunks = None
    runtime.knobs.max_context_tokens = None
    runtime.knobs.chunk_expansion = True
    runtime.trace.step(
        Stage.CONTEXT_ASSEMBLY.value,
        "expand_context",
        input_summary=f"max_chunks={previous_chunks}, max_tokens={previous_tokens}",
        output_summary="context budget restored to the configured maximum",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.CONTEXT_ASSEMBLY,
        rationale=(
            "lifted the artificial context truncation and expanded with neighbouring "
            "chunks so the constraints of the question can be covered"
        ),
        detail={
            "previous_max_chunks": previous_chunks,
            "previous_max_tokens": previous_tokens,
        },
        reused_stages=[
            Stage.QUERY_UNDERSTANDING,
            Stage.QUERY_DECOMPOSITION,
            Stage.RETRIEVAL,
            Stage.RANKING,
        ],
    )


# ---------------------------------------------------------------------------
# Generation-stage recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.CONSTRAINED_REGENERATION)
def constrained_regeneration(ctx: RecoveryContext) -> StrategyOutcome:
    """Regenerate under a hard grounding constraint, reusing the context."""
    runtime, state = ctx.runtime, ctx.state
    previous_ground = runtime.knobs.ground_answer
    previous_requirement = runtime.knobs.require_grounding
    runtime.knobs.ground_answer = True
    runtime.knobs.require_grounding = True
    runtime.knobs.cite_sources = True
    state.baseline_answer = state.answer()
    runtime.trace.step(
        Stage.GENERATION.value,
        "constrained_regeneration",
        input_summary=f"previous grounded_mode={previous_ground and previous_requirement}",
        output_summary="regenerating with evidence-only + citation constraints",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.GENERATION,
        rationale=(
            "re-generated the answer with grounding and citation constraints re-enabled, "
            "using the same retrieved context (no retrieval repeat)"
        ),
        detail={"previous_grounded_mode": previous_ground and previous_requirement},
        reused_stages=[
            Stage.QUERY_UNDERSTANDING,
            Stage.QUERY_DECOMPOSITION,
            Stage.RETRIEVAL,
            Stage.RANKING,
            Stage.CONTEXT_ASSEMBLY,
        ],
    )


# ---------------------------------------------------------------------------
# Query-decomposition recovery
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.DECOMPOSITION_REPAIR)
def decomposition_repair(ctx: RecoveryContext) -> StrategyOutcome:
    """Re-run decomposition without the injected truncation."""
    runtime, state = ctx.runtime, ctx.state
    runtime.fault_flags.pop("INCOMPLETE_DECOMPOSITION", None)
    runtime.knobs.force_decomposition = None
    understanding = state.understanding
    if understanding is not None:
        understanding.is_complex = True
    runtime.trace.step(
        Stage.QUERY_DECOMPOSITION.value,
        "repair_decomposition",
        input_summary=f"previous sub-queries={len(understanding.subqueries) if understanding else 0}",
        output_summary="re-planning sub-questions",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=Stage.QUERY_DECOMPOSITION,
        rationale=(
            "re-ran query decomposition without the truncation fault so every constraint of "
            "the question produces its own evidence request"
        ),
        detail={"previous_subqueries": len(understanding.subqueries) if understanding else 0},
        reused_stages=[Stage.QUERY_UNDERSTANDING],
    )


# ---------------------------------------------------------------------------
# Infrastructure recoveries
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.BOUNDED_RETRY)
def bounded_retry(ctx: RecoveryContext) -> StrategyOutcome:
    """One retry of the failed stage after clearing transient conditions."""
    runtime, state = ctx.runtime, ctx.state
    retriever = runtime.retriever
    cleared: List[str] = []
    if retriever is not None and retriever.inject.get("force_timeout"):
        retriever.inject.pop("force_timeout", None)
        retriever.timeouts = max(0, retriever.timeouts - 1)
        cleared.append("retrieval timeout")
    if retriever is not None and retriever.service_available is False:
        retriever.service_available = True
        cleared.append("retrieval service availability")
    state.infra.timeouts = max(0, state.infra.timeouts - 1)
    state.infra.service_status = "ok"
    rerun_from = _earliest_failed(ctx) or Stage.RETRIEVAL
    runtime.trace.step(
        Stage.INFRASTRUCTURE.value,
        "bounded_retry",
        input_summary=f"cleared={cleared or 'nothing transient'}",
        output_summary=f"retrying from {rerun_from.value}",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=rerun_from,
        rationale=(
            f"performed one bounded retry from '{rerun_from.value}' after clearing transient "
            f"conditions ({', '.join(cleared) if cleared else 'none'})"
        ),
        detail={"cleared": cleared, "retry_stage": rerun_from.value},
        reused_stages=_reused_before(rerun_from),
    )


@strategy(RecoveryAction.ALTERNATE_PATH)
def alternate_path(ctx: RecoveryContext) -> StrategyOutcome:
    """Switch to an independent backend instead of retrying the broken one."""
    runtime, state = ctx.runtime, ctx.state
    retriever = runtime.retriever
    previous_method = runtime.knobs.retrieval_method or runtime.settings.retrieval.method
    changes: List[str] = []

    if previous_method != "bm25":
        # BM25 is a genuinely independent index: no embeddings, no vector store.
        runtime.knobs.retrieval_method = "bm25"
        changes.append("retrieval_method -> bm25 (independent of the vector store)")
    if retriever is not None:
        retriever.service_available = True
        retriever.inject.pop("force_timeout", None)
        retriever.inject.pop("corrupt_rerank", None)
        changes.append("cleared retrieval outage flags")
    state.infra.retrieval_service_unavailable = False
    state.infra.timeouts = max(0, state.infra.timeouts - 1)
    state.infra.service_status = "ok"
    rerun_from = _earliest_failed(ctx) or Stage.RETRIEVAL
    runtime.trace.step(
        Stage.INFRASTRUCTURE.value,
        "alternate_path",
        input_summary=f"method={previous_method}",
        output_summary="; ".join(changes),
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=rerun_from,
        rationale=(
            "abandoned the failing backend and switched to an independent retrieval path "
            "(Okapi BM25 needs no embedding service); prior stages are reused"
        ),
        detail={"previous_method": previous_method, "changes": changes},
        reused_stages=_reused_before(rerun_from),
    )


@strategy(RecoveryAction.TOOL_REPAIR)
def tool_repair(ctx: RecoveryContext) -> StrategyOutcome:
    """Disable the misbehaving tool (corrupt reranker) and continue."""
    runtime, state = ctx.runtime, ctx.state
    retriever = runtime.retriever
    changes: List[str] = []
    if retriever is not None and retriever.inject.get("corrupt_rerank"):
        retriever.inject.pop("corrupt_rerank", None)
        changes.append("disabled the corrupted reranker")
    if retriever is not None and retriever.inject.get("force_timeout"):
        retriever.inject.pop("force_timeout", None)
        changes.append("disabled the timing-out retrieval tool")
    if not changes:
        runtime.knobs.use_rerank = False
        changes.append("bypassed reranking for this request")
    rerun_from = Stage.RANKING if any("rerank" in c for c in changes) else Stage.RETRIEVAL
    runtime.trace.step(
        Stage.AGENT_CONTROL.value,
        "tool_repair",
        input_summary="; ".join(changes),
        output_summary=f"continue from {rerun_from.value}",
    )
    return StrategyOutcome(
        applied=True,
        rerun_from=rerun_from,
        rationale=(
            "removed the misbehaving tool from the agent plan and continued the workflow "
            "without it"
        ),
        detail={"changes": changes},
        reused_stages=_reused_before(rerun_from),
    )


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------
@strategy(RecoveryAction.ABSTAIN)
def abstain(ctx: RecoveryContext) -> StrategyOutcome:
    return StrategyOutcome(
        applied=True,
        rerun_from=None,
        rationale=(
            "no safe, targeted recovery remains within the attempt/latency/token budget; "
            "declining to answer rather than emitting unsupported text"
        ),
    )


# ---------------------------------------------------------------------------
def _earliest_failed(ctx: RecoveryContext) -> Optional[Stage]:
    failed = [s for s in ctx.state.stage_results if not s.ok]
    if not failed:
        return None
    indexes = []
    for result in failed:
        if result.stage in _PIPELINE_ORDER:
            indexes.append(_PIPELINE_ORDER.index(result.stage))
    return _PIPELINE_ORDER[min(indexes)] if indexes else None


def get_strategy(action: RecoveryAction) -> Optional[StrategyFn]:
    return STRATEGIES.get(action)


def registered_strategies() -> List[str]:
    return [a.value for a in STRATEGIES]