"""Pipeline stages.

Each stage is an explicit function ``(state, runtime) -> StageResult``.  That
makes targeted recovery trivial: re-running from stage *k* just means calling
``stages[k:]`` and **reusing** the state produced by stages ``<k``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.base import LLMError, LLMResponseError, ProviderTimeout
from ..models import (
    AssembledContext,
    ContextItem,
    GenerationResult,
    QueryUnderstanding,
    RetrievedChunk,
    RetrievalDiagnostics,
)
from ..observability.trace import get_logger
from ..prompts import answer_prompt, decompose_prompt, rewrite_prompt
from ..settings import Settings
from ..taxonomy import RecoveryAction, Stage
from ..textutil import (
    capitalized_terms,
    clip,
    estimate_tokens,
    split_sentences,
    stem,
    tokenize,
)
from .runtime import Runtime
from .state import PipelineState, StageResult, SubQueryResult

logger = get_logger(__name__)

CONSTRAINT_PATTERNS = (
    r"\bwithin\s+(\d+)\s*(minutes|hours|days|working days|business days)\b",
    r"\bafter\s+(\d+)\s*(minutes|hours|days)\b",
    r"\bbefore\s+(\d+)\s*(minutes|hours|days)\b",
    r"\b(\d+)\s*(percent|%|days|hours|minutes|attempts|retries|times)\b",
    r"\bat least\s+(\d+)\b",
    r"\bat most\s+(\d+)\b",
    r"\bmaximum\s+of\s+(\d+)\b",
    r"\bfee of\s+(\d+)\b",
    r"\bwithin\s+(\d+)\s*(business|calendar)\s+days\b",
)


# ---------------------------------------------------------------------------
# 1. Query understanding
# ---------------------------------------------------------------------------
def stage_query_understanding(state: PipelineState, runtime: Runtime) -> StageResult:
    started = time.perf_counter()
    query = state.query
    settings: Settings = runtime.settings
    fault = runtime.fault_flags.get("QUERY_CORRUPTION")
    if fault:
        query = _corrupt_query(query, float(fault.get("dropout", 0.4)), int(fault.get("seed", 7)))
        state.metadata["corrupted_query"] = query

    forced_rewrite = runtime.knobs.force_rewrite
    is_ambiguous = runtime.fault_flags.get("AMBIGUOUS_QUERY") is not None

    understanding = QueryUnderstanding(
        original_query=state.query,
        normalized_query=re.sub(r"\s+", " ", query).strip(),
        rewrite=forced_rewrite or None,
        is_ambiguous=is_ambiguous,
    )
    tokens = tokenize(understanding.normalized_query)
    understanding.keywords = list(dict.fromkeys(tokens))[:24]
    understanding.entities = sorted(capitalized_terms(understanding.normalized_query))[:8]
    understanding.constraints = _extract_constraints(understanding.normalized_query)
    understanding.is_complex = _is_complex(understanding.normalized_query, settings)
    if is_ambiguous:
        understanding.ambiguity_reason = "query reduced to an under-specified fragment"
    elif len(understanding.normalized_query.split()) < settings.thresholds.min_query_words:
        understanding.is_ambiguous = True
        understanding.ambiguity_reason = "fewer than the minimum number of content words"
    understanding.intent = _intent_of(understanding.normalized_query)
    if understanding.rewrite:
        understanding.rewrite_drift = _drift(state.query, understanding.rewrite)

    latency = (time.perf_counter() - started) * 1000.0
    state.understanding = understanding
    runtime.trace.step(
        Stage.QUERY_UNDERSTANDING.value,
        "understand",
        input_summary=clip(state.query, 160),
        output_summary=clip(
            f"complex={understanding.is_complex} ambiguous={understanding.is_ambiguous} "
            f"constraints={len(understanding.constraints)} rewrite={bool(understanding.rewrite)}",
            200,
        ),
        latency_ms=latency,
        **{"keywords": understanding.keywords[:8]},
    )
    return StageResult(
        stage=Stage.QUERY_UNDERSTANDING,
        latency_ms=latency,
        detail={
            "is_complex": understanding.is_complex,
            "is_ambiguous": understanding.is_ambiguous,
            "constraints": understanding.constraints,
            "query_used": understanding.normalized_query,
        },
    )


# ---------------------------------------------------------------------------
# 2. Decomposition
# ---------------------------------------------------------------------------
def stage_decomposition(state: PipelineState, runtime: Runtime) -> StageResult:
    started = time.perf_counter()
    understanding = state.understanding
    if understanding is None:
        stage_query_understanding(state, runtime)
        understanding = state.understanding
    assert understanding is not None

    settings: Settings = runtime.settings
    latent = runtime.fault_flags.get("INCOMPLETE_DECOMPOSITION")
    forced = runtime.knobs.force_decomposition

    subqueries: List[str] = []
    latency = 0.0
    if understanding.is_complex and settings.agent.enable_decomposition:
        if forced is False:
            subqueries = [understanding.normalized_query]
        else:
            prompt = decompose_prompt(understanding.normalized_query, max_subqueries=4)
            runtime.trace.step(Stage.QUERY_DECOMPOSITION.value, "decompose", input_summary=clip(understanding.normalized_query, 160))
            try:
                result = runtime.ensure_provider().generate(prompt, system="You decompose questions into sub-questions.")
                latency = result.latency_ms
                state.add_llm(1, result.total_tokens, result.cost)
                parsed = _parse_json_list(result.text)
                subqueries = parsed or _heuristic_decompose(understanding.normalized_query)
                runtime.trace.step(
                    Stage.QUERY_DECOMPOSITION.value,
                    "decompose_result",
                    output_summary=clip(" | ".join(subqueries), 240),
                    latency_ms=latency,
                )
            except (LLMError, ProviderTimeout) as exc:
                _note_infra_error(state, runtime, exc)
                subqueries = _heuristic_decompose(understanding.normalized_query)
    else:
        subqueries = [understanding.normalized_query]
        if forced:
            subqueries = forced if isinstance(forced, list) else [str(forced)]

    if latent:
        keep = max(1, int(latent.get("keep", 1)))
        subqueries = subqueries[:keep]
        state.infra.stage_latency_ms.setdefault("fault_injection", 0.0)

    understanding.subqueries = [s for s in subqueries if s.strip()]
    total_latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.QUERY_DECOMPOSITION.value,
        "plan",
        output_summary=f"{len(understanding.subqueries)} sub-question(s): "
        + clip(" | ".join(understanding.subqueries), 220),
        latency_ms=total_latency,
    )
    return StageResult(
        stage=Stage.QUERY_DECOMPOSITION,
        latency_ms=total_latency,
        detail={"subqueries": understanding.subqueries, "provider_latency_ms": latency},
    )


# ---------------------------------------------------------------------------
# 3 + 4. Retrieval and ranking
# ---------------------------------------------------------------------------
def stage_retrieval(
    state: PipelineState,
    runtime: Runtime,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    method: Optional[str] = None,
) -> StageResult:
    """Retrieve for the whole query set, aggregate and (optionally) rerank."""
    started = time.perf_counter()
    understanding = state.understanding
    if understanding is None:
        stage_query_understanding(state, runtime)
        stage_decomposition(state, runtime)
        understanding = state.understanding
    assert understanding is not None

    retriever = runtime.retriever
    if retriever is None:
        return StageResult(
            stage=Stage.RETRIEVAL, ok=False, error="no retriever configured",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    settings = runtime.settings
    effective_top_k = int(top_k or runtime.knobs.top_k or settings.retrieval.top_k)
    effective_candidate_k = int(runtime.knobs.candidate_k or settings.retrieval.candidate_k)
    effective_method = (method or runtime.knobs.retrieval_method or settings.retrieval.method).lower()

    # The rewritten query is the *retrieval* query; understanding and
    # decomposition results are still reused.
    base_query = query or understanding.rewrite or understanding.normalized_query or state.query
    state.metadata["retrieval_query"] = base_query
    if query:
        queries = [query]
    elif understanding.rewrite:
        queries = [base_query] + [s for s in understanding.subqueries if s != base_query]
    else:
        queries = list(understanding.subqueries or [state.query])

    aggregated: Dict[str, RetrievedChunk] = {}
    sub_results: List[SubQueryResult] = []
    all_diags: List[RetrievalDiagnostics] = []
    error: Optional[str] = None

    for subquery in queries:
        sub_started = time.perf_counter()
        try:
            # request the *candidate pool*, not just top-k: the pool is what
            # lets the ranking stage be evaluated separately from retrieval.
            results, diag = retriever.search(
                subquery,
                top_k=max(effective_top_k, effective_candidate_k),
                candidate_k=max(effective_candidate_k, effective_top_k * 2),
                method=effective_method,
                use_rerank=False,  # ranking is owned by stage_ranking
            )
        except Exception as exc:
            latency = (time.perf_counter() - sub_started) * 1000.0
            runtime.trace.step(
                Stage.RETRIEVAL.value,
                "retrieve",
                input_summary=clip(subquery, 140),
                output_summary="error",
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=latency,
            )
            _note_infra_error(state, runtime, exc)
            sub_results.append(SubQueryResult(subquery=subquery, error=str(exc), latency_ms=latency))
            error = str(exc)
            continue
        all_diags.append(diag)
        sub_results.append(
            SubQueryResult(
                subquery=subquery,
                retrieved=list(results),
                diagnostics=diag,
                latency_ms=(time.perf_counter() - sub_started) * 1000.0,
            )
        )
        for position, rc in enumerate(results, start=1):
            existing = aggregated.get(rc.chunk_id)
            score = float(rc.score) + (0.05 / position)
            if existing is None:
                rc.rank = position
                aggregated[rc.chunk_id] = rc
                aggregated[rc.chunk_id].metadata = dict(rc.metadata)
                aggregated[rc.chunk_id].metadata["_agg_score"] = score
            else:
                existing.metadata["_agg_score"] = float(
                    existing.metadata.get("_agg_score", existing.score)
                ) + score

    state.sub_results = sub_results
    state.retrieved_candidates = sorted(
        aggregated.values(),
        key=lambda rc: -float(rc.metadata.get("_agg_score", rc.score)),
    )
    state.retrieval_diagnostics = _merge_diagnostics(
        all_diags, top_k=effective_top_k, method=effective_method
    )

    latency = (time.perf_counter() - started) * 1000.0
    if error:
        if state.retrieval_diagnostics is None:
            state.retrieval_diagnostics = RetrievalDiagnostics(
                top_k=effective_top_k, method=effective_method
            )
        state.retrieval_diagnostics.notes.append(f"retrieval_error={error}")
        runtime.trace.step(
            Stage.RETRIEVAL.value,
            "retrieve",
            input_summary=clip(" ; ".join(queries), 180),
            output_summary=f"error: {error}",
            status="error",
            error=error,
            latency_ms=latency,
        )
        return StageResult(stage=Stage.RETRIEVAL, ok=False, error=error, latency_ms=latency)

    ranking_result = stage_ranking(state, runtime, effective_top_k)

    # Authoritative diagnostics describe exactly the chunks that reach context
    # assembly.  We also record what the *candidate pool* looked like before
    # ranking: if retrieval already had the evidence and ranking lost it, the
    # root cause is the ranking stage, not retrieval.
    if retriever is not None:
        pre_rank = retriever.build_diagnostics(
            state.query,
            state.retrieved_candidates[: max(effective_top_k * 4, 20)],
            None,
            top_k=effective_top_k,
            candidate_pool=len(state.retrieved_candidates),
        )
        state.metadata["pre_rank_diagnostics"] = pre_rank.model_dump()
        state.retrieval_diagnostics = retriever.build_diagnostics(
            state.query,
            state.retrieved,
            state.retrieval_diagnostics,
            top_k=effective_top_k,
            candidate_pool=len(state.retrieved_candidates),
        )
        state.retrieval_diagnostics.pre_rank_top_overlap = pre_rank.top_overlap
        state.retrieval_diagnostics.pre_rank_useful_ratio = pre_rank.useful_ratio
        state.retrieval_diagnostics.ranking_coverage_loss = round(
            max(0.0, pre_rank.top_overlap - state.retrieval_diagnostics.top_overlap), 6
        )
    total_latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.RETRIEVAL.value,
        "retrieve",
        input_summary=clip(" ; ".join(queries), 180),
        output_summary=f"{len(state.retrieved)} candidate(s) via {effective_method}",
        latency_ms=latency,
        **{"top_k": effective_top_k, "method": effective_method},
    )
    return StageResult(
        stage=Stage.RETRIEVAL,
        latency_ms=total_latency,
        detail={
            "top_k": effective_top_k,
            "method": effective_method,
            "candidates": len(state.retrieved),
            "ranking_latency_ms": ranking_result.latency_ms,
        },
    )


def stage_ranking(
    state: PipelineState, runtime: Runtime, top_k: Optional[int] = None
) -> StageResult:
    """Select + order the final context candidates (reranking stage)."""
    started = time.perf_counter()
    settings = runtime.settings
    effective_top_k = int(top_k or runtime.knobs.top_k or settings.retrieval.top_k)
    candidates = list(state.retrieved_candidates)
    diagnostics = state.retrieval_diagnostics

    if runtime.knobs.use_rerank is None and settings.retrieval.rerank_enabled and candidates:
        from ..retrieval.reranker import CorruptReranker, build_reranker

        if runtime.retriever is not None and runtime.retriever.inject.get("corrupt_rerank"):
            reranker = CorruptReranker(
                mode=runtime.retriever.inject.get("rerank_mode", "reverse"), seed=settings.seed
            )
        else:
            reranker = build_reranker(settings.retrieval.rerank_model, seed=settings.seed)
        pool = candidates[: max(settings.retrieval.rerank_candidates, effective_top_k)]
        before = {rc.chunk_id: i for i, rc in enumerate(pool, start=1)}
        rerank_query = state.metadata.get("retrieval_query") or state.query
        ordered = reranker.rerank(rerank_query, pool, top_k=effective_top_k)
        changes = sum(
            1 for i, rc in enumerate(ordered, start=1) if before.get(rc.chunk_id, i) != i
        )
        candidates = ordered + candidates[len(pool) :]
        if diagnostics is not None:
            diagnostics.reranker_applied = True
            diagnostics.rank_changes = changes
            scores = [float(rc.rerank_score or 0.0) for rc in ordered]
            if len(scores) > 1:
                diagnostics.rerank_score_margin = round(scores[0] - scores[1], 6)
    else:
        candidates = candidates[:effective_top_k]

    for position, rc in enumerate(candidates[:effective_top_k], start=1):
        rc.rank = position
        if rc.original_rank is None:
            rc.original_rank = position

    state.retrieved = list(candidates[:effective_top_k])
    latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.RANKING.value,
        "rank",
        input_summary=f"{len(candidates)} candidate(s)",
        output_summary=f"{len(state.retrieved)} selected; top={state.retrieved[0].chunk_id if state.retrieved else '-'}",
        latency_ms=latency,
    )
    return StageResult(
        stage=Stage.RANKING,
        latency_ms=latency,
        detail={"selected": len(state.retrieved), "reranked": bool(state.retrieved and state.retrieved[0].rerank_score is not None)},
    )


# ---------------------------------------------------------------------------
# 5. Context assembly
# ---------------------------------------------------------------------------
def stage_context_assembly(state: PipelineState, runtime: Runtime) -> StageResult:
    started = time.perf_counter()
    settings = runtime.settings
    max_chunks = runtime.knobs.max_chunks
    max_tokens = runtime.knobs.max_context_tokens
    neighbour_chunks: List[Chunk] = []

    if runtime.knobs.chunk_expansion and runtime.retriever is not None:
        neighbour_chunks = _expand_with_neighbours(state.retrieved, runtime.retriever)

    pool = list(state.retrieved)
    for rc in neighbour_chunks:
        if rc.chunk_id not in {p.chunk_id for p in pool}:
            pool.append(rc)

    # CONFLICTING_EVIDENCE: the decoy is placed immediately after the chunk it
    # contradicts so that both statements are present in the same context.
    if runtime.decoy_chunks and runtime.fault_flags.get("CONFLICTING_EVIDENCE"):
        anchor_ids = {rc.chunk_id for rc in pool}
        for decoy in runtime.decoy_chunks:
            if decoy.chunk_id in anchor_ids:
                continue
            injected = RetrievedChunk(
                chunk_id=decoy.chunk_id,
                document_id=decoy.document_id,
                content=decoy.content,
                score=0.0,
                rank=0,
                method="injected",
                page=decoy.page,
                source=decoy.source,
                metadata=dict(decoy.metadata),
            )
            position = next(
                (i for i, rc in enumerate(pool) if rc.chunk_id == decoy.metadata.get("conflict_with")),
                len(pool) - 1,
            )
            pool.insert(min(position + 1, len(pool)), injected)

    items: List[ContextItem] = []
    total_chars = 0
    truncated = False
    seen_docs: set = set()
    sources: set = set()

    neighbour_ids = {c.chunk_id for c in neighbour_chunks}
    for position, rc in enumerate(pool, start=1):
        text = rc.content
        item_truncated = False
        if max_tokens is not None:
            budget = max(0, int(max_tokens) - total_chars // 4)
            if budget <= 0:
                break
            if estimate_tokens(text) > budget:
                text = text[: max(40, budget * 4)]
                item_truncated = True
                truncated = True
        items.append(
            ContextItem(
                chunk=rc,
                text=text,
                position=position,
                truncated=item_truncated,
                expansion_reason="neighbour_chunk" if rc.chunk_id in neighbour_ids else None,
            )
        )
        total_chars += len(text)
        seen_docs.add(rc.document_id)
        sources.add((rc.source or rc.document_id).split("/")[-1])
        if max_chunks is not None and len(items) >= int(max_chunks):
            truncated = truncated or len(pool) > len(items)
            break

    context = AssembledContext(
        items=items,
        text="\n\n".join(item.text for item in items),
        token_estimate=estimate_tokens("\n\n".join(item.text for item in items)),
        char_estimate=total_chars,
        source_diversity=len(sources),
        unique_documents=len(seen_docs),
        truncation_applied=truncated,
    )
    understanding = state.understanding
    from ..signals.metrics import constraint_coverage, context_coverage

    coverage_value = context_coverage(
        state.query,
        context.text,
        understanding.subqueries if understanding else (),
        understanding.constraints if understanding else (),
    )
    constraint_value, missing = constraint_coverage(
        state.query, understanding.constraints if understanding else (), context.text
    )
    context.coverage = coverage_value
    context.unconstrained_constraints = missing
    context.notes.append(f"constraint_coverage={constraint_value:.3f}")

    state.context = context
    runtime.knobs.chunk_expansion = False
    latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.CONTEXT_ASSEMBLY.value,
        "assemble",
        input_summary=f"{len(pool)} ranked chunk(s)",
        output_summary=f"{len(items)} item(s), ~{context.token_estimate} tokens, coverage={coverage_value:.2f}",
        latency_ms=latency,
    )
    return StageResult(
        stage=Stage.CONTEXT_ASSEMBLY,
        latency_ms=latency,
        detail={
            "items": len(items),
            "tokens": context.token_estimate,
            "coverage": coverage_value,
            "truncated": truncated,
        },
    )


# ---------------------------------------------------------------------------
# 6. Generation
# ---------------------------------------------------------------------------
def stage_generation(state: PipelineState, runtime: Runtime) -> StageResult:
    started = time.perf_counter()
    settings = runtime.settings
    understanding = state.understanding
    context = state.context
    provider = runtime.ensure_provider()

    if runtime.fault_flags.get("MODEL_FAILURE"):
        runtime.trace.step(
            Stage.GENERATION.value, "generate", status="error",
            error="model failure injected", output_summary="empty completion",
        )
        _note_infra_error(state, runtime, LLMResponseError("model failure injected"))
        state.generation = GenerationResult(
            answer="",
            provider=getattr(provider, "name", "mock"),
            model=getattr(provider, "model", "mock"),
            grounded=False,
            groundedness=0.0,
        )
        return StageResult(
            stage=Stage.GENERATION, ok=False, error="model failure injected",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    items = context.items if context else []
    context_payload = [
        {
            "marker": f"S{i}",
            "text": item.text,
            "source": item.chunk.source or item.chunk.document_id,
            "document_id": item.chunk.document_id,
            "chunk_id": item.chunk.chunk_id,
        }
        for i, item in enumerate(items, start=1)
    ]
    prompt = answer_prompt(
        question=state.query,
        context_items=context_payload,
        require_grounding=runtime.knobs.require_grounding and settings.generation.require_grounding,
        cite=runtime.knobs.cite_sources and settings.generation.cite_sources,
        ground=runtime.knobs.ground_answer and settings.generation.ground_answer,
        max_sentences=4 if not runtime.knobs.ground_answer else 4,
        constraints=understanding.constraints if understanding else [],
        previous_answer=state.baseline_answer or None if runtime.knobs.require_grounding else None,
    )
    try:
        result = provider.generate(
            prompt, system="You are a careful, evidence-grounded retrieval assistant."
        )
        state.add_llm(1, result.total_tokens, result.cost)
        answer = (result.text or "").strip()
        latency = result.latency_ms
    except (ProviderTimeout, LLMError) as exc:
        latency = (time.perf_counter() - started) * 1000.0
        _note_infra_error(state, runtime, exc)
        state.generation = GenerationResult(
            answer="",
            provider=getattr(provider, "name", "mock"),
            model=getattr(provider, "model", "mock"),
            grounded=False,
            groundedness=0.0,
            prompt_tokens=estimate_tokens(prompt),
        )
        runtime.trace.step(
            Stage.GENERATION.value, "generate", status="error", error=str(exc), latency_ms=latency
        )
        return StageResult(stage=Stage.GENERATION, ok=False, error=str(exc), latency_ms=latency)

    from ..signals.metrics import extract_citations, groundedness_score

    markers = [f"S{i}" for i in range(1, len(items) + 1)]
    scores = groundedness_score(
        question=state.query,
        answer=answer,
        context=[item.text for item in items],
        citation_markers=markers,
        support_threshold=settings.thresholds.min_claim_support,
    )
    grounded_mode = runtime.knobs.ground_answer and runtime.knobs.require_grounding
    state.generation = GenerationResult(
        answer=answer,
        raw_answer=answer,
        citations=extract_citations(answer),
        claims=scores["claims"],
        grounded=scores["groundedness"] >= settings.thresholds.min_groundedness,
        groundedness=scores["groundedness"],
        citation_coverage=scores["citation_coverage"],
        unsupported_claims=scores["unsupported_claims"],
        confidence=round(
            0.5 * scores["groundedness"] + 0.5 * scores["citation_coverage"], 4
        ),
        grounded_mode=grounded_mode,
        provider=getattr(provider, "name", "mock"),
        model=getattr(provider, "model", "mock"),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost=result.cost,
        latency_ms=latency,
    )
    total_latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.GENERATION.value,
        "generate",
        input_summary=f"{len(items)} context item(s), ~{estimate_tokens(prompt)} prompt tokens",
        output_summary=clip(answer, 220),
        latency_ms=total_latency,
        groundedness=state.generation.groundedness,
        citations=len(state.generation.citations),
    )
    return StageResult(
        stage=Stage.GENERATION,
        latency_ms=total_latency,
        detail={
            "groundedness": state.generation.groundedness,
            "citations": state.generation.citations,
            "answer_chars": len(answer),
        },
    )


# ---------------------------------------------------------------------------
# 7. Validation / signal collection
# ---------------------------------------------------------------------------
def stage_validation(state: PipelineState, runtime: Runtime) -> StageResult:
    from ..signals.collector import SignalCollector

    started = time.perf_counter()
    budget_payload: Dict[str, Any] = {}
    collector = SignalCollector(runtime.settings)
    agent_signals = state.metadata.get("agent_signals_snapshot")
    signals = collector.collect(
        query=state.query,
        understanding=state.understanding,
        retrieved=state.retrieved,
        diagnostics=state.retrieval_diagnostics,
        context=state.context,
        generation=state.generation,
        steps=runtime.trace.steps,
        budget=budget_payload,
        infra={
            **state.infra.as_dict(state.total_latency_ms),
            "embedding_errors": (runtime.retriever.embedding_errors if runtime.retriever else 0),
            "timeouts": max(state.infra.timeouts, runtime.retriever.timeouts if runtime.retriever else 0),
        },
        gold_chunk_ids=(set(runtime.gold_chunk_ids) or set(state.gold_chunk_ids)) or None,
        citations=None,
        run_id=state.run_id,
        corpus_chunks=list(runtime.retriever.chunks) if runtime.retriever else None,
        document_texts=_document_texts(runtime),
        agent_signals=agent_signals,
    )
    state.signals = signals
    latency = (time.perf_counter() - started) * 1000.0
    runtime.trace.step(
        Stage.VALIDATION.value,
        "validate",
        input_summary="pipeline artefacts",
        output_summary=(
            f"groundedness={signals.generation.get('groundedness')} "
            f"useful_ratio={signals.retrieval.get('useful_ratio')} "
            f"coverage={signals.context.get('context_coverage')}"
        ),
        latency_ms=latency,
    )
    return StageResult(
        stage=Stage.VALIDATION,
        latency_ms=latency,
        detail={"signals": signals.model_dump()},
    )


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------
STAGE_SEQUENCE: Tuple[Stage, ...] = (
    Stage.QUERY_UNDERSTANDING,
    Stage.QUERY_DECOMPOSITION,
    Stage.RETRIEVAL,
    Stage.RANKING,
    Stage.CONTEXT_ASSEMBLY,
    Stage.GENERATION,
    Stage.VALIDATION,
)

STAGE_RUNNERS = {
    Stage.QUERY_UNDERSTANDING: stage_query_understanding,
    Stage.QUERY_DECOMPOSITION: stage_decomposition,
    Stage.RETRIEVAL: stage_retrieval,
    Stage.RANKING: stage_ranking,
    Stage.CONTEXT_ASSEMBLY: stage_context_assembly,
    Stage.GENERATION: stage_generation,
    Stage.VALIDATION: stage_validation,
}


def run_stages_from(
    start: Stage, state: PipelineState, runtime: Runtime
) -> List[StageResult]:
    """Execute every stage from ``start`` onwards, reusing earlier state.

    Every stage is wrapped in a trace span so the trace viewer can show the
    input/output and the latency of each stage independently of the agent-step
    log.
    """
    results: List[StageResult] = []
    started_index = STAGE_SEQUENCE.index(start)
    for stage in STAGE_SEQUENCE[started_index:]:
        runner = STAGE_RUNNERS[stage]
        if stage is Stage.RANKING and start is not Stage.RANKING:
            continue  # ranking already executed inside stage_retrieval
        with runtime.trace.span(stage.value, name=stage.value) as span:
            span.inputs = {"start_from": start.value, "query": state.query[:200]}
            result = runner(state, runtime)
            state.record_stage(result)
            span.outputs = {"ok": result.ok, "detail": result.detail}
            span.metrics = {"latency_ms": result.latency_ms}
            span.status = "ok" if result.ok else "error"
            if result.error:
                span.error = result.error
        results.append(result)
        if not result.ok and stage in {Stage.RETRIEVAL, Stage.GENERATION}:
            break
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _document_texts(runtime: Runtime) -> Dict[str, str]:
    """Full document text keyed by document_id (cached on the runtime).

    Sentence-level fragmentation can only be measured against whole documents:
    re-assembling sentences from chunk text would hide the very split we are
    trying to detect.
    """
    cache = runtime.metadata.setdefault("_document_texts", {})
    if runtime.repository is None:
        return {}
    for document in runtime.repository.list_documents():
        if document.document_id not in cache or cache[document.document_id][0] != document.version:
            cache[document.document_id] = (document.version, document.content)
    return {doc_id: text for doc_id, (version, text) in cache.items()}


def _corrupt_query(query: str, dropout: float, seed: int) -> str:
    from ..textutil import stable_int

    words = query.split()
    if not words:
        return query
    keep_ratio = max(0.2, 1.0 - dropout)
    n_keep = max(1, int(round(len(words) * keep_ratio)))
    chosen = sorted({stable_int(word, seed, modulo=len(words)) for word in words})
    kept = [words[i] for i in chosen[:n_keep]]
    return " ".join(kept) if kept else words[0]


def _drift(original: str, rewrite: str) -> float:
    a = set(tokenize(original))
    b = set(tokenize(rewrite))
    if not a:
        return 0.0
    return round(1.0 - len(a & b) / len(a), 6)


def _is_complex(query: str, settings: Settings) -> bool:
    words = query.split()
    if len(words) < settings.agent.decomposition_min_words:
        return False
    conjunction_hits = sum(
        1 for w in words if w.lower().strip(",") in {"and", "or", "also", "plus", "then", "while", "both"}
    )
    question_marks = query.count("?")
    return conjunction_hits >= 2 or question_marks >= 2 or (conjunction_hits >= 1 and len(words) >= 18)


def _intent_of(query: str) -> str:
    low = query.lower().strip()
    if re.match(r"^(what|which|who|when|where)\b", low):
        return "factual_lookup"
    if re.match(r"^(how|why)\b", low):
        return "explanation"
    if re.match(r"^(list|name|give|provide)\b", low):
        return "enumeration"
    if re.match(r"^(compare|contrast|difference|versus|vs)\b", low):
        return "comparison"
    if re.match(r"^(define|what does .* mean)\b", low):
        return "definition"
    return "factual_lookup"


def _extract_constraints(query: str) -> List[str]:
    """Extract explicit, checkable constraints (numbers + durations).

    Only quantitative or temporal expressions count: a bare topic word like
    "threshold" is not a constraint, and treating it as one produces false
    "missing evidence" diagnoses.
    """
    out: List[str] = []
    lowered = query.lower()
    for pattern in CONSTRAINT_PATTERNS:
        for match in re.finditer(pattern, lowered):
            phrase = match.group(0).strip()
            if phrase and phrase not in out:
                out.append(phrase)
    # "how many <unit>" / "how much <unit>" demands a quantity of that unit in
    # the evidence - a checkable constraint that needs no ground truth.
    for match in re.finditer(
        r"\bhow (?:many|much|often|long)\s+((?:\w+\s+){0,2}?"
        r"(?:minutes|hours|days|weeks|months|percent|units|attempts|retries|times|people))\b",
        lowered,
    ):
        phrase = match.group(1).strip()
        if phrase and phrase not in out:
            out.append(f"how many {phrase}")
    return out[:6]


def _heuristic_decompose(query: str) -> List[str]:
    parts = [
        p.strip()
        for p in re.split(r"(?<=[?])\s+|;\s+|\s+and\s+(?=(?:what|which|who|when|where|how|why)\b)", query)
        if p.strip()
    ]
    if len(parts) <= 1:
        parts = re.split(r"\s+and\s+|\s+then\s+", query)
    out = [p if p.endswith("?") else f"{p}?" for p in parts if len(p.split()) > 1]
    return out[:4] or [query]


def _parse_json_list(text: str) -> List[str]:
    import json
    import re as _re

    cleaned = _re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=_re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _re.search(r"\[.*\]", cleaned, _re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    if isinstance(data, dict):
        for key in ("subqueries", "sub_questions", "questions"):
            if isinstance(data.get(key), list):
                return [str(x) for x in data[key] if str(x).strip()]
    return []


def _merge_diagnostics(
    diagnostics: Sequence[RetrievalDiagnostics], top_k: int, method: str
) -> RetrievalDiagnostics:
    """Combine per-sub-query diagnostics into one retrieval view."""
    merged = RetrievalDiagnostics(top_k=top_k, candidate_pool=sum(d.candidate_pool for d in diagnostics), method=method)
    if not diagnostics:
        merged.notes.append("no successful retrieval")
        return merged
    distribution: List[float] = []
    for d in diagnostics:
        merged.candidate_chunk_ids = list(d.candidate_chunk_ids)
        distribution.extend(d.score_distribution)
        merged.returned += d.returned
        merged.useful_count += d.useful_count
        merged.rank_changes += d.rank_changes
        merged.latency_ms += d.latency_ms
        merged.reranker_applied = merged.reranker_applied or d.reranker_applied
        merged.useful_overlap = d.useful_overlap or merged.useful_overlap
        merged.top_overlap = max(merged.top_overlap, d.top_overlap)
        merged.notes.extend(d.notes)
    merged.score_distribution = [round(s, 6) for s in distribution]
    merged.useful_ratio = round(merged.useful_count / merged.returned, 6) if merged.returned else 0.0
    if distribution:
        from ..textutil import mean, median, stddev

        merged.top_score = round(max(distribution), 6)
        merged.mean_score = round(mean(distribution), 6)
        merged.median_score = round(median(distribution), 6)
        merged.min_score = round(min(distribution), 6)
        merged.max_score = round(max(distribution), 6)
        merged.score_gap = round(max(distribution) - min(distribution), 6)
        merged.score_std = round(stddev(distribution), 6)
    merged.latency_ms = round(merged.latency_ms, 3)
    return merged


def _expand_with_neighbours(
    retrieved: Sequence[RetrievedChunk], retriever: Any, max_neighbours: int = 2
) -> List[RetrievedChunk]:
    """CHUNK_EXPANSION: pull the immediate neighbours of each retrieved chunk."""
    out: List[RetrievedChunk] = []
    index = {c.chunk_id: c for c in retriever.chunks}
    by_doc: Dict[str, List[Chunk]] = {}
    for chunk in retriever.chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)
    for rc in retrieved[:3]:
        current = index.get(rc.chunk_id)
        if current is None:
            continue
        siblings = sorted(by_doc.get(current.document_id, []), key=lambda c: c.chunk_index)
        position = next((i for i, c in enumerate(siblings) if c.chunk_id == current.chunk_id), -1)
        if position < 0:
            continue
        for offset in range(1, max_neighbours + 1):
            for candidate_index in (position - offset, position + offset):
                if 0 <= candidate_index < len(siblings):
                    sibling = siblings[candidate_index]
                    out.append(
                        RetrievedChunk(
                            chunk_id=sibling.chunk_id,
                            document_id=sibling.document_id,
                            content=sibling.content,
                            score=round(float(rc.score) * 0.55, 6),
                            rank=rc.rank,
                            method=rc.method,
                            original_rank=rc.original_rank,
                            page=sibling.page,
                            source=sibling.source,
                            metadata=dict(sibling.metadata),
                            expanded_from=rc.chunk_id,
                        )
                    )
    return out


def _note_infra_error(state: PipelineState, runtime: Runtime, exc: Exception) -> None:
    subtype = getattr(exc, "failure_subtype", "model_failure")
    if subtype == "timeout":
        state.infra.timeouts += 1
    elif subtype == "malformed_response":
        state.infra.parse_errors += 1
    elif subtype == "embedding_failure":
        state.infra.embedding_errors += 1
    else:
        state.infra.provider_errors += 1
    state.infra.infra_errors += 1
    state.infra.service_status = "degraded"
    state.infra.stage_latency_ms.setdefault("_errors", 0.0)
    state.metadata.setdefault("infra_errors", []).append(f"{type(exc).__name__}: {exc}")