"""Tier-1 deterministic failure detector.

Every rule is a pure function of the :class:`SignalBundle` plus the configured
thresholds.  Rules are evaluated in a fixed order (infrastructure first,
because it explains downstream symptoms) and each returns a confidence that
reflects how strongly its evidence fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..models import DiagnosisTier, SignalBundle
from ..settings import Settings, get_settings
from ..taxonomy import (
    FailureClass,
    FailureTaxonomy,
    RecoveryAction,
    Severity,
    Stage,
    recovery_ladder,
)


@dataclass
class RuleMatch:
    rule_id: str
    subtype: str
    failure_class: FailureClass
    stage: Stage
    confidence: float
    severity: Severity
    evidence: List[str] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "subtype": self.subtype,
            "failure_class": self.failure_class.value,
            "stage": self.stage.value,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "evidence": self.evidence,
            "signals": self.signals,
        }


@dataclass
class RuleContext:
    signals: SignalBundle
    settings: Settings

    # convenience accessors -------------------------------------------
    def s(self, section: str, key: str, default: Any = None) -> Any:
        return getattr(self.signals, section, {}).get(key, default)

    @property
    def t(self):
        return self.settings.thresholds

    @property
    def b(self):
        return self.settings.budget


Rule = Callable[[RuleContext], Optional[RuleMatch]]


def _mk(
    rule_id: str,
    subtype: str,
    confidence: float,
    evidence: Sequence[str],
    signals: Dict[str, Any],
    severity: Optional[Severity] = None,
) -> RuleMatch:
    definition = FailureTaxonomy.get(subtype)
    return RuleMatch(
        rule_id=rule_id,
        subtype=subtype,
        failure_class=_class_of(subtype, definition),
        stage=definition.stage if definition else Stage.RETRIEVAL,
        confidence=confidence,
        severity=severity or (definition.default_severity if definition else Severity.MEDIUM),
        evidence=list(evidence),
        signals=signals,
    )


def _class_of(subtype: str, definition) -> FailureClass:
    from ..taxonomy import class_for_subtype

    return class_for_subtype(subtype)


# ---------------------------------------------------------------------------
# INFRASTRUCTURE rules (highest priority)
# ---------------------------------------------------------------------------
def rule_retrieval_unavailable(ctx: RuleContext) -> Optional[RuleMatch]:
    if ctx.s("infrastructure", "retrieval_service_unavailable"):
        return _mk(
            "R_INFRA_001",
            "retrieval_service_unavailable",
            0.97,
            ["Retrieval backend reported as unavailable"],
            {"retrieval_service_unavailable": True},
            Severity.CRITICAL,
        )
    return None


def rule_timeout(ctx: RuleContext) -> Optional[RuleMatch]:
    timeouts = int(ctx.s("infrastructure", "timeouts", 0) or 0)
    latency = float(ctx.s("infrastructure", "latency_ms", 0.0) or 0.0)
    if timeouts > 0:
        return _mk(
            "R_INFRA_002",
            "timeout",
            0.93,
            [f"{timeouts} timeout event(s) recorded", f"observed latency {latency:.1f} ms"],
            {"timeouts": timeouts, "latency_ms": latency},
            Severity.HIGH,
        )
    if latency > ctx.b.max_total_latency_ms:
        return _mk(
            "R_INFRA_003",
            "latency_budget_exceeded",
            0.9,
            [f"latency {latency:.1f} ms exceeds budget {ctx.b.max_total_latency_ms} ms"],
            {"latency_ms": latency},
            Severity.HIGH,
        )
    return None


def rule_malformed_response(ctx: RuleContext) -> Optional[RuleMatch]:
    if int(ctx.s("infrastructure", "parse_errors", 0) or 0) > 0:
        return _mk(
            "R_INFRA_004",
            "malformed_response",
            0.92,
            [f"{ctx.s('infrastructure', 'parse_errors')} unparseable model response(s)"],
            {"parse_errors": ctx.s("infrastructure", "parse_errors")},
            Severity.HIGH,
        )
    return None


def rule_embedding_failure(ctx: RuleContext) -> Optional[RuleMatch]:
    if int(ctx.s("infrastructure", "embedding_errors", 0) or 0) > 0:
        return _mk(
            "R_INFRA_005",
            "embedding_failure",
            0.94,
            [f"{ctx.s('infrastructure', 'embedding_errors')} embedding error(s)"],
            {"embedding_errors": ctx.s("infrastructure", "embedding_errors")},
            Severity.CRITICAL,
        )
    return None


def rule_model_failure(ctx: RuleContext) -> Optional[RuleMatch]:
    if int(ctx.s("infrastructure", "provider_errors", 0) or 0) > 0:
        return _mk(
            "R_INFRA_006",
            "model_failure",
            0.92,
            [f"{ctx.s('infrastructure', 'provider_errors')} provider error(s)"],
            {"provider_errors": ctx.s("infrastructure", "provider_errors")},
            Severity.CRITICAL,
        )
    return None


# ---------------------------------------------------------------------------
# BUDGET rules
# ---------------------------------------------------------------------------
def rule_budget_exhausted(ctx: RuleContext) -> Optional[RuleMatch]:
    budget = ctx.signals.budget or {}
    exhausted = budget.get("exhausted") or {}
    if exhausted.get("attempts"):
        return _mk(
            "R_BUDGET_001",
            "budget_exhausted",
            0.95,
            [f"recovery attempts exhausted ({budget.get('attempts_used')} used)"],
            {"attempts_used": budget.get("attempts_used")},
            Severity.HIGH,
        )
    if exhausted.get("tokens"):
        return _mk(
            "R_BUDGET_002",
            "token_budget_exceeded",
            0.94,
            [f"token budget exhausted ({budget.get('tokens_used')} tokens)"],
            {"tokens_used": budget.get("tokens_used")},
            Severity.HIGH,
        )
    if exhausted.get("latency"):
        return _mk(
            "R_BUDGET_003",
            "latency_budget_exceeded",
            0.94,
            ["latency budget exhausted"],
            {"latency_used_ms": budget.get("latency_used_ms")},
            Severity.HIGH,
        )
    return None


# ---------------------------------------------------------------------------
# AGENT CONTROL rules
# ---------------------------------------------------------------------------
def rule_repeated_action(ctx: RuleContext) -> Optional[RuleMatch]:
    repeats = int(ctx.s("agent", "repeat_actions", 0) or 0)
    if repeats > ctx.b.max_repeated_actions:
        return _mk(
            "R_AGENT_001",
            "repeated_action",
            0.88,
            [f"{repeats} repeated agent action(s) exceed limit {ctx.b.max_repeated_actions}"],
            {"repeat_actions": repeats, "duplicate_action_rate": ctx.s("agent", "duplicate_action_rate")},
            Severity.HIGH,
        )
    return None


def rule_unnecessary_tool_call(ctx: RuleContext) -> Optional[RuleMatch]:
    if int(ctx.s("agent", "unused_tool_results", 0) or 0) > 0:
        return _mk(
            "R_AGENT_002",
            "unnecessary_tool_call",
            0.55,
            [f"{ctx.s('agent', 'unused_tool_results')} tool result(s) not used by the answer"],
            {"unused_tool_results": ctx.s("agent", "unused_tool_results")},
            Severity.LOW,
        )
    return None


def rule_wrong_tool_selection(ctx: RuleContext) -> Optional[RuleMatch]:
    tool_calls = int(ctx.s("agent", "tool_calls", 0) or 0)
    useful = float(ctx.s("retrieval", "useful_ratio", 0.0) or 0.0)
    if tool_calls > 0 and useful < ctx.t.min_useful_chunk_ratio and int(ctx.s("retrieval", "retrieved_count", 0) or 0):
        return _mk(
            "R_AGENT_003",
            "wrong_tool_selection",
            0.6,
            [
                f"{tool_calls} tool call(s) executed but useful_ratio={useful:.2f}",
                "tool output did not serve the current subgoal",
            ],
            {"tool_calls": tool_calls, "useful_ratio": useful},
            Severity.MEDIUM,
        )
    return None


# ---------------------------------------------------------------------------
# GENERATION rules
# ---------------------------------------------------------------------------
def rule_hallucination(ctx: RuleContext) -> Optional[RuleMatch]:
    groundedness = float(ctx.s("generation", "groundedness", 1.0) or 0.0)
    citations = int(ctx.s("generation", "citations", 0) or 0)
    unsupported = int(ctx.s("generation", "unsupported_claims", 0) or 0)
    claims = int(ctx.s("generation", "claim_count", 0) or 0)
    if claims == 0:
        return None
    if groundedness < ctx.t.min_groundedness:
        confidence = min(0.95, 0.6 + (ctx.t.min_groundedness - groundedness))
        severity = Severity.CRITICAL if groundedness < 0.4 or citations == 0 else Severity.HIGH
        return _mk(
            "R_GEN_001",
            "hallucination",
            confidence,
            [
                f"groundedness={groundedness:.3f} below threshold {ctx.t.min_groundedness}",
                f"{unsupported}/{claims} claims unsupported by retrieved evidence",
            ],
            {"groundedness": groundedness, "citations": citations, "unsupported_claims": unsupported},
            severity,
        )
    return None


def rule_unsupported_claim(ctx: RuleContext) -> Optional[RuleMatch]:
    unsupported = int(ctx.s("generation", "unsupported_claims", 0) or 0)
    claims = int(ctx.s("generation", "claim_count", 0) or 0)
    min_support = float(ctx.s("generation", "min_claim_support", 1.0) or 0.0)
    if unsupported > ctx.t.max_unsupported_claims and claims:
        return _mk(
            "R_GEN_002",
            "unsupported_claim",
            min(0.9, 0.6 + 0.1 * unsupported),
            [
                f"{unsupported} unsupported claim(s) detected",
                f"minimum claim support {min_support:.3f}",
            ],
            {"unsupported_claims": unsupported, "min_claim_support": min_support},
            Severity.HIGH,
        )
    return None


def rule_citation_gap(ctx: RuleContext) -> Optional[RuleMatch]:
    """Low citation coverage is only a defect when grounding is weak too.

    A short answer may legitimately cite two of five passages, so this rule is
    deliberately conjunctive - on its own it is evidence, not a verdict.
    """
    coverage_value = float(ctx.s("generation", "citation_coverage", 1.0) or 0.0)
    chunks = int(ctx.s("context", "chunk_count", 0) or 0)
    groundedness = float(ctx.s("generation", "groundedness", 1.0) or 0.0)
    unsupported = int(ctx.s("generation", "unsupported_claims", 0) or 0)
    if chunks < 2 or unsupported == 0 or groundedness >= 0.95:
        return None
    if coverage_value < ctx.t.min_citation_coverage:
        return _mk(
            "R_GEN_003",
            "unsupported_claim",
            0.6,
            [
                f"citation_coverage={coverage_value:.3f} below {ctx.t.min_citation_coverage}",
                f"{chunks} evidence chunk(s) available but {unsupported} claim(s) lack citations",
            ],
            {"citation_coverage": coverage_value, "chunk_count": chunks},
            Severity.MEDIUM,
        )
    return None


def rule_incomplete_answer(ctx: RuleContext) -> Optional[RuleMatch]:
    words = int(ctx.s("generation", "answer_words", 0) or 0)
    chunks = int(ctx.s("context", "chunk_count", 0) or 0)
    subqueries = int(ctx.s("query", "subquery_count", 0) or 0)
    if chunks >= 2 and subqueries >= 2 and 0 < words < ctx.t.min_answer_words:
        return _mk(
            "R_GEN_004",
            "incomplete_answer",
            0.66,
            [f"answer has only {words} words for {subqueries} sub-questions"],
            {"answer_words": words, "subquery_count": subqueries},
            Severity.MEDIUM,
        )
    return None


def rule_repetition(ctx: RuleContext) -> Optional[RuleMatch]:
    repetition = float(ctx.s("generation", "repetition_score", 0.0) or 0.0)
    if repetition > 0.45:
        return _mk(
            "R_GEN_005",
            "incomplete_answer",
            0.6,
            [f"degenerate repetition detected (score {repetition:.2f})"],
            {"repetition_score": repetition},
            Severity.MEDIUM,
        )
    return None


# ---------------------------------------------------------------------------
# CONTEXT rules
# ---------------------------------------------------------------------------
def rule_missing_evidence(ctx: RuleContext) -> Optional[RuleMatch]:
    missing = int(ctx.s("context", "unconstrained_constraints", 0) or 0)
    if missing > 0:
        return _mk(
            "R_CTX_001",
            "missing_evidence",
            min(0.9, 0.65 + 0.12 * missing),
            [
                f"{missing} explicit constraint(s) of the question are not covered",
                f"context_coverage={ctx.s('context', 'context_coverage')}",
            ],
            {"unconstrained_constraints": missing},
            Severity.HIGH,
        )
    return None


def rule_insufficient_context(ctx: RuleContext) -> Optional[RuleMatch]:
    tokens = int(ctx.s("context", "context_tokens", 0) or 0)
    coverage_value = float(ctx.s("context", "context_coverage", 1.0) or 0.0)
    if tokens < ctx.t.min_context_tokens:
        return _mk(
            "R_CTX_002",
            "insufficient_context",
            0.88,
            [f"context_tokens={tokens} below minimum {ctx.t.min_context_tokens}"],
            {"context_tokens": tokens},
            Severity.HIGH,
        )
    if coverage_value < ctx.t.min_context_coverage:
        return _mk(
            "R_CTX_003",
            "insufficient_context",
            min(0.85, 0.55 + (ctx.t.min_context_coverage - coverage_value)),
            [f"context_coverage={coverage_value:.3f} below {ctx.t.min_context_coverage}"],
            {"context_coverage": coverage_value},
            Severity.HIGH,
        )
    return None


def rule_fragmented_context(ctx: RuleContext) -> Optional[RuleMatch]:
    fragmentation = float(ctx.s("context", "fragmentation_ratio", 0.0) or 0.0)
    coverage_value = float(ctx.s("context", "context_coverage", 1.0) or 0.0)
    if fragmentation > ctx.t.max_fragmentation_ratio and coverage_value < 0.85:
        return _mk(
            "R_CTX_004",
            "fragmented_context",
            min(0.85, 0.55 + fragmentation),
            [
                f"fragmentation_ratio={fragmentation:.2f} above {ctx.t.max_fragmentation_ratio}",
                f"context_coverage={coverage_value:.2f}",
            ],
            {"fragmentation_ratio": fragmentation, "context_coverage": coverage_value},
            Severity.HIGH,
        )
    return None


def rule_excessive_context(ctx: RuleContext) -> Optional[RuleMatch]:
    tokens = int(ctx.s("context", "context_tokens", 0) or 0)
    if tokens > ctx.t.max_context_tokens:
        return _mk(
            "R_CTX_005",
            "excessive_context",
            min(0.8, 0.5 + tokens / (ctx.t.max_context_tokens * 4)),
            [
                f"context_tokens={tokens} above maximum {ctx.t.max_context_tokens}",
                f"relevant_buried={ctx.s('context', 'relevant_buried')}",
            ],
            {"context_tokens": tokens},
            Severity.MEDIUM,
        )
    return None


def rule_conflicting_evidence(ctx: RuleContext) -> Optional[RuleMatch]:
    conflicts = int(ctx.s("context", "conflict_count", 0) or 0)
    score = float(ctx.s("context", "conflict_score", 0.0) or 0.0)
    if conflicts > 0 and score > ctx.t.max_conflict_score:
        return _mk(
            "R_CTX_006",
            "conflicting_evidence",
            min(0.9, 0.6 + 0.1 * conflicts),
            [f"{conflicts} conflicting evidence pair(s) detected", f"conflict_score={score:.3f}"],
            {"conflict_count": conflicts, "conflict_score": score},
            Severity.HIGH,
        )
    return None


# ---------------------------------------------------------------------------
# RANKING rules
# ---------------------------------------------------------------------------
def rule_relevant_ranked_low(ctx: RuleContext) -> Optional[RuleMatch]:
    regression = float(ctx.s("ranking", "rank_regression_rate", 0.0) or 0.0)
    below_cut = int(ctx.s("retrieval", "relevant_below_cut", 0) or 0)
    mrr_delta = float(ctx.s("ranking", "mrr_delta_after_rerank", 0.0) or 0.0)
    coverage_loss = float(ctx.s("retrieval", "ranking_coverage_loss", 0.0) or 0.0)
    pre_rank = float(ctx.s("retrieval", "pre_rank_top_overlap", 0.0) or 0.0)
    top_overlap = float(ctx.s("retrieval", "top_overlap", 0.0) or 0.0)
    if below_cut > 0 and pre_rank >= ctx.t.min_top_overlap:
        return _mk(
            "R_RANK_001",
            "relevant_ranked_low",
            min(0.9, 0.72 + 0.1 * below_cut),
            [
                f"{below_cut} relevant chunk(s) pushed beyond top_k={ctx.s('retrieval', 'top_k')}",
                f"the candidate pool reached {pre_rank:.2f} coverage but ranking kept only {top_overlap:.2f}",
                f"rank_regression_rate={regression:.2f}",
            ],
            {
                "relevant_below_cut": below_cut,
                "rank_regression_rate": regression,
                "ranking_coverage_loss": coverage_loss,
            },
            Severity.HIGH,
        )
    if mrr_delta < -0.1:
        return _mk(
            "R_RANK_002",
            "poor_reranking",
            min(0.85, 0.6 - mrr_delta),
            [f"reranking reduced MRR by {abs(mrr_delta):.3f}"],
            {"mrr_delta_after_rerank": mrr_delta, "ranking_coverage_loss": coverage_loss},
            Severity.MEDIUM,
        )
    if coverage_loss >= 0.15 and pre_rank >= ctx.t.min_top_overlap:
        return _mk(
            "R_RANK_005",
            "irrelevant_ranked_high",
            min(0.85, 0.6 + coverage_loss),
            [
                f"ranking removed {coverage_loss:.2f} of question coverage that retrieval had found",
                f"pre-rank coverage {pre_rank:.2f} -> ranked coverage {top_overlap:.2f}",
            ],
            {"ranking_coverage_loss": coverage_loss, "pre_rank_top_overlap": pre_rank},
            Severity.MEDIUM,
        )
    return None


def rule_irrelevant_ranked_high(ctx: RuleContext) -> Optional[RuleMatch]:
    if not ctx.s("ranking", "irrelevant_at_rank1", False):
        return None
    useful = float(ctx.s("retrieval", "useful_ratio", 0.0) or 0.0)
    if useful < ctx.t.min_useful_chunk_ratio:
        return _mk(
            "R_RANK_003",
            "irrelevant_ranked_high",
            min(0.82, 0.55 + (ctx.t.min_useful_chunk_ratio - useful)),
            [
                "rank-1 chunk is not relevant while useful evidence exists in the candidate pool",
                f"useful_ratio={useful:.2f}",
            ],
            {"useful_ratio": useful, "top_score": ctx.s("retrieval", "top_score")},
            Severity.MEDIUM,
        )
    return None


def rule_flat_rerank(ctx: RuleContext) -> Optional[RuleMatch]:
    if not ctx.s("ranking", "rerank_flat", False):
        return None
    return _mk(
        "R_RANK_004",
        "poor_reranking",
        0.58,
        ["reranker assigned (near) identical scores to every candidate"],
        {"rerank_score_margin": ctx.s("ranking", "rerank_score_margin")},
        Severity.MEDIUM,
    )


# ---------------------------------------------------------------------------
# RETRIEVAL rules
# ---------------------------------------------------------------------------
def rule_stale_documents(ctx: RuleContext) -> Optional[RuleMatch]:
    stale = float(ctx.s("retrieval", "stale_ratio", 0.0) or 0.0)
    superseded = float(ctx.s("retrieval", "superseded_ratio", 0.0) or 0.0)
    if superseded > 0 or stale > 0:
        return _mk(
            "R_RET_008",
            "stale_documents",
            min(0.88, 0.65 + 0.2 * max(stale, superseded)),
            [
                f"{superseded:.0%} of retrieved chunks are explicitly superseded versions",
                f"{stale:.0%} of retrieved chunks are older than the current document version",
            ],
            {"stale_ratio": stale, "superseded_ratio": superseded},
            Severity.MEDIUM,
        )
    return None


def rule_empty_retrieval(ctx: RuleContext) -> Optional[RuleMatch]:
    if int(ctx.s("retrieval", "retrieved_count", 0) or 0) == 0:
        return _mk(
            "R_RET_001",
            "empty_retrieval",
            0.9,
            ["retrieval returned zero candidates"],
            {"retrieved_count": 0},
            Severity.CRITICAL,
        )
    return None


def rule_no_useful_evidence(ctx: RuleContext) -> Optional[RuleMatch]:
    count = int(ctx.s("retrieval", "retrieved_count", 0) or 0)
    useful = float(ctx.s("retrieval", "useful_ratio", 0.0) or 0.0)
    top_overlap = float(ctx.s("retrieval", "top_overlap", 0.0) or 0.0)
    if count > 0 and useful < ctx.t.min_useful_chunk_ratio and top_overlap < ctx.t.useful_overlap:
        return _mk(
            "R_RET_002",
            "no_useful_evidence",
            min(0.9, 0.6 + (ctx.t.min_useful_chunk_ratio - useful)),
            [
                f"{count} chunk(s) returned but useful_ratio={useful:.2f}",
                f"best lexical coverage of the question is only {top_overlap:.3f}",
            ],
            {"useful_ratio": useful, "top_overlap": top_overlap},
            Severity.CRITICAL,
        )
    return None


def rule_low_recall(ctx: RuleContext) -> Optional[RuleMatch]:
    """Retrieval-stage low recall.

    Decided on the **pre-ranking** candidate pool: if the pool already lacked
    the evidence the fault is retrieval, whereas poor evidence *after* ranking
    is a ranking failure (see :func:`rule_relevant_ranked_low`).
    """
    top_overlap = float(ctx.s("retrieval", "top_overlap", 0.0) or 0.0)
    pre_rank = float(ctx.s("retrieval", "pre_rank_top_overlap", 0.0) or 0.0) or top_overlap
    mean_overlap = float(ctx.s("retrieval", "mean_overlap", 0.0) or 0.0)
    gap = float(ctx.s("retrieval", "score_gap", 0.0) or 0.0)
    recall = ctx.s("retrieval", "recall")
    top_score = float(ctx.s("retrieval", "top_score", 0.0) or 0.0)
    below_cut = int(ctx.s("retrieval", "relevant_below_cut", 0) or 0)
    count = int(ctx.s("retrieval", "retrieved_count", 0) or 0)
    gold_count = int(ctx.s("retrieval", "gold_count", 0) or 0)
    if (
        count > 0
        and recall is not None
        and float(recall) < 0.5
        and gold_count > 0
        and below_cut == 0
        and top_overlap < ctx.t.min_top_overlap * 2
    ):
        # the answer-bearing chunk is in the corpus but nothing close to it was
        # retrieved: there is simply no usable evidence in the result set
        return _mk(
            "R_RET_009",
            "no_useful_evidence",
            "low_recall",
            min(0.93, 0.7 + (0.5 - float(recall))),
            [
                f"measured recall@{ctx.s('retrieval', 'top_k')}={float(recall):.3f} with "
                f"{gold_count} answer-bearing chunk(s) known to exist",
                f"best retrieved chunk covers {top_overlap:.3f} of the question",
            ],
            {"recall": recall, "top_score": top_score, "gold_count": gold_count},
            Severity.CRITICAL,
        )
    if (
        count > 0
        and recall is not None
        and float(recall) < 0.5
        and gold_count > 0
        and below_cut == 0
        and top_overlap < ctx.t.min_top_overlap * 2
    ):
        return _mk(
            "R_RET_003",
            "low_recall",
            min(0.93, 0.7 + (0.5 - float(recall))),
            [
                f"measured recall@{ctx.s('retrieval', 'top_k')}={float(recall):.3f}",
                f"gold chunks={gold_count} were never retrieved",
            ],
            {"recall": recall, "top_score": top_score},
            Severity.HIGH,
        )
    if pre_rank < ctx.t.min_top_overlap:
        severity = Severity.HIGH if pre_rank < ctx.t.min_top_overlap / 2 else Severity.MEDIUM
        return _mk(
            "R_RET_004",
            "low_recall",
            min(0.82, 0.55 + (ctx.t.min_top_overlap - pre_rank)),
            [
                f"the candidate pool only reaches {pre_rank:.3f} question coverage "
                f"(threshold {ctx.t.min_top_overlap}) before ranking",
                f"best ranked chunk covers {top_overlap:.3f}",
            ],
            {
                "pre_rank_top_overlap": pre_rank,
                "top_overlap": top_overlap,
                "mean_overlap": mean_overlap,
                "top_score": top_score,
            },
            severity,
        )
    if mean_overlap < ctx.t.min_mean_overlap and pre_rank < ctx.t.min_top_overlap * 1.2:
        return _mk(
            "R_RET_005",
            "low_recall",
            min(0.72, 0.5 + (ctx.t.min_mean_overlap - mean_overlap)),
            [
                f"mean_overlap={mean_overlap:.3f} below {ctx.t.min_mean_overlap}",
                f"score_gap={gap:.4f}",
            ],
            {"mean_overlap": mean_overlap, "score_gap": gap},
            Severity.MEDIUM,
        )
    return None


def rule_irrelevant_retrieval(ctx: RuleContext) -> Optional[RuleMatch]:
    useful = float(ctx.s("retrieval", "useful_ratio", 0.0) or 0.0)
    top_overlap = float(ctx.s("retrieval", "top_overlap", 0.0) or 0.0)
    mean_overlap = float(ctx.s("retrieval", "mean_overlap", 0.0) or 0.0)
    count = int(ctx.s("retrieval", "retrieved_count", 0) or 0)
    if count >= 2 and useful < ctx.t.min_useful_chunk_ratio and top_overlap < 0.30:
        return _mk(
            "R_RET_006",
            "irrelevant_retrieval",
            min(0.85, 0.55 + (ctx.t.min_useful_chunk_ratio - useful)),
            [
                f"useful_ratio={useful:.2f} below {ctx.t.min_useful_chunk_ratio}",
                f"mean lexical coverage with the question only {mean_overlap:.2f}",
            ],
            {"useful_ratio": useful, "top_overlap": top_overlap, "mean_overlap": mean_overlap},
            Severity.HIGH,
        )
    return None


def rule_duplicate_heavy(ctx: RuleContext) -> Optional[RuleMatch]:
    duplicates = float(ctx.s("retrieval", "duplicate_ratio", 0.0) or 0.0)
    diversity = int(ctx.s("retrieval", "source_diversity", 0) or 0)
    useful = float(ctx.s("retrieval", "useful_ratio", 1.0) or 0.0)
    if duplicates > ctx.t.max_duplicate_ratio or (
        diversity == 1 and int(ctx.s("retrieval", "retrieved_count", 0)) >= 4 and useful < 0.5
    ):
        return _mk(
            "R_RET_007",
            "over_decomposition",
            0.56,
            [
                f"duplicate_ratio={duplicates:.2f}",
                f"only {diversity} distinct source(s) across retrieved chunks",
            ],
            {"duplicate_ratio": duplicates, "source_diversity": diversity},
            Severity.LOW,
        )
    return None


# ---------------------------------------------------------------------------
# QUERY rules
# ---------------------------------------------------------------------------
def rule_ambiguous_query(ctx: RuleContext) -> Optional[RuleMatch]:
    if ctx.s("query", "is_ambiguous", False):
        reason = ctx.s("query", "ambiguity_reason", "underspecified query")
        return _mk(
            "R_QRY_001",
            "ambiguous_query",
            0.78,
            [str(reason)],
            {"ambiguity_reason": reason, "query_words": ctx.s("query", "query_words")},
            Severity.MEDIUM,
        )
    words = int(ctx.s("query", "query_words", 0) or 0)
    if words and words < ctx.t.min_query_words:
        return _mk(
            "R_QRY_002",
            "ambiguous_query",
            0.62,
            [f"query has only {words} content word(s)"],
            {"query_words": words},
            Severity.MEDIUM,
        )
    return None


def rule_incorrect_rewrite(ctx: RuleContext) -> Optional[RuleMatch]:
    if ctx.s("query", "is_ambiguous", False):
        # an under-specified query is diagnosed as ambiguous, not as a bad rewrite
        return None
    drift = float(ctx.s("query", "rewrite_drift", 0.0) or 0.0)
    dropped = ctx.s("query", "key_terms_dropped") or []
    if ctx.s("query", "rewrite_present", False) and drift > ctx.t.max_rewrite_drift:
        return _mk(
            "R_QRY_003",
            "incorrect_rewrite",
            min(0.88, 0.6 + drift / 2),
            [f"rewrite_drift={drift:.3f} above {ctx.t.max_rewrite_drift}", f"terms dropped: {dropped}"],
            {"rewrite_drift": drift, "key_terms_dropped": dropped},
            Severity.HIGH,
        )
    if ctx.s("query", "rewrite_present", False) and dropped and drift < 0.12:
        return _mk(
            "R_QRY_004",
            "incorrect_rewrite",
            0.62,
            [f"rewrite preserved no new terms (change={drift:.3f}) while terms were dropped: {dropped}"],
            {"rewrite_drift": drift},
            Severity.MEDIUM,
        )
    return None


def rule_incomplete_decomposition(ctx: RuleContext) -> Optional[RuleMatch]:
    actual = int(ctx.s("query", "subquery_count", 0) or 0)
    expected = int(ctx.s("query", "expected_subqueries", 1) or 1)
    if ctx.s("query", "is_complex", False) and actual < expected:
        return _mk(
            "R_QRY_005",
            "incomplete_decomposition",
            min(0.85, 0.6 + 0.15 * (expected - actual)),
            [f"complex query produced {actual} sub-question(s), expected >= {expected}"],
            {"subquery_count": actual, "expected_subqueries": expected},
            Severity.HIGH,
        )
    return None


def rule_over_decomposition(ctx: RuleContext) -> Optional[RuleMatch]:
    actual = int(ctx.s("query", "subquery_count", 0) or 0)
    if actual >= 4 and float(ctx.s("retrieval", "duplicate_ratio", 0.0) or 0.0) > 0.5:
        return _mk(
            "R_QRY_006",
            "over_decomposition",
            0.6,
            [f"{actual} sub-questions produced duplicated evidence"],
            {"subquery_count": actual, "duplicate_ratio": ctx.s("retrieval", "duplicate_ratio")},
            Severity.LOW,
        )
    return None


# ---------------------------------------------------------------------------
# Rule registry (order matters: most specific / most explanatory first)
# ---------------------------------------------------------------------------
RULES: List[Rule] = [
    # infrastructure & budgets first - they explain downstream symptoms
    rule_retrieval_unavailable,
    rule_timeout,
    rule_malformed_response,
    rule_embedding_failure,
    rule_model_failure,
    rule_budget_exhausted,
    # agent control
    rule_repeated_action,
    rule_wrong_tool_selection,
    rule_unnecessary_tool_call,
    # generation
    rule_hallucination,
    rule_unsupported_claim,
    rule_citation_gap,
    rule_incomplete_answer,
    rule_repetition,
    # context
    rule_conflicting_evidence,
    rule_missing_evidence,
    rule_fragmented_context,
    rule_insufficient_context,
    rule_excessive_context,
    # ranking
    rule_relevant_ranked_low,
    rule_irrelevant_ranked_high,
    rule_flat_rerank,
    # retrieval
    rule_stale_documents,
    rule_empty_retrieval,
    rule_no_useful_evidence,
    rule_low_recall,
    rule_irrelevant_retrieval,
    rule_duplicate_heavy,
    # query
    rule_incorrect_rewrite,
    rule_incomplete_decomposition,
    rule_over_decomposition,
    rule_ambiguous_query,
]


#: Execution order used for root-cause preference.  A downstream symptom must
#: not out-vote the stage that actually broke: if retrieval returned nothing,
#: "the answer is unsupported" is a consequence, not the cause.
STAGE_PRECEDENCE: Tuple[Stage, ...] = (
    Stage.AGENT_CONTROL,   # budget/loop violations first (they explain everything)
    Stage.INFRASTRUCTURE,
    Stage.QUERY_UNDERSTANDING,
    Stage.QUERY_DECOMPOSITION,
    Stage.RETRIEVAL,
    Stage.RANKING,
    Stage.CONTEXT_ASSEMBLY,
    Stage.GENERATION,
)

#: a downstream match must beat the earliest-stage match by this margin to win
DOWNSTREAM_OVERRIDE_MARGIN = 0.12


class RuleDetector:
    """Tier-1 detector."""

    def __init__(self, settings: Optional[Settings] = None, rules: Optional[List[Rule]] = None):
        self.settings = settings or get_settings()
        self.rules = rules or RULES
        self.tier = DiagnosisTier.RULE

    def evaluate(self, signals: SignalBundle) -> List[RuleMatch]:
        ctx = RuleContext(signals=signals, settings=self.settings)
        matches: List[RuleMatch] = []
        for rule in self.rules:
            try:
                match = rule(ctx)
            except Exception:
                match = None
            if match is not None and match.confidence > 0:
                matches.append(match)
        matches.sort(key=lambda m: -m.confidence)
        return matches

    def best(self, signals: SignalBundle) -> Optional[RuleMatch]:
        """Highest-confidence match, preferring the earliest failing stage.

        Both a retrieval failure and a generation failure may fire at once.  The
        root cause is the earliest one, so a downstream match must exceed the
        earliest-stage match by ``DOWNSTREAM_OVERRIDE_MARGIN`` to be selected.
        """
        matches = self.evaluate(signals)
        if not matches:
            return None
        top = matches[0]
        ranks = {stage: index for index, stage in enumerate(STAGE_PRECEDENCE)}
        candidates = [m for m in matches if ranks.get(m.stage, 99) == ranks.get(top.stage, 99)]
        earliest_rank = min(ranks.get(m.stage, 99) for m in matches)
        earliest = [m for m in matches if ranks.get(m.stage, 99) == earliest_rank]
        if earliest and top in earliest:
            return top
        best_earliest = max(earliest, key=lambda m: m.confidence)
        if best_earliest.confidence >= top.confidence - DOWNSTREAM_OVERRIDE_MARGIN:
            return best_earliest
        return top

    def confident(self, signals: SignalBundle, threshold: Optional[float] = None) -> Optional[RuleMatch]:
        threshold = self.settings.diagnosis.rule_confidence_threshold if threshold is None else threshold
        best = self.best(signals)
        if best is not None and best.confidence >= threshold:
            return best
        return None