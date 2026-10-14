"""Failure taxonomy for agentic RAG pipelines.

The taxonomy is *declarative*: every failure class carries metadata that the
rest of the system consumes (detector rules, recovery mapping, severity
defaults, expected evidence).  This keeps detection, diagnosis and recovery
consistent with one another.

Stage vocabulary
----------------
``query_understanding``, ``query_decomposition``, ``retrieval``, ``ranking``,
``context_assembly``, ``generation``, ``validation``, ``agent_control``,
``infrastructure``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Stage(str, Enum):
    """Stages of the agentic RAG pipeline that can fail."""

    QUERY_UNDERSTANDING = "query_understanding"
    QUERY_DECOMPOSITION = "query_decomposition"
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"
    VALIDATION = "validation"
    AGENT_CONTROL = "agent_control"
    INFRASTRUCTURE = "infrastructure"


PIPELINE_STAGES: Tuple[Stage, ...] = (
    Stage.QUERY_UNDERSTANDING,
    Stage.QUERY_DECOMPOSITION,
    Stage.RETRIEVAL,
    Stage.RANKING,
    Stage.CONTEXT_ASSEMBLY,
    Stage.GENERATION,
    Stage.VALIDATION,
)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]

    @classmethod
    def max(cls, a: "Severity", b: "Severity") -> "Severity":
        return a if a.rank >= b.rank else b


class RecoveryAction(str, Enum):
    """Targeted recovery actions available to the recovery planner."""

    QUERY_REWRITE = "QUERY_REWRITE"
    HYBRID_SEARCH = "HYBRID_SEARCH"
    EXPAND_TOP_K = "EXPAND_TOP_K"
    RERANK = "RERANK"
    CHUNK_EXPANSION = "CHUNK_EXPANSION"
    CONTEXT_EXPANSION = "CONTEXT_EXPANSION"
    CONSTRAINED_REGENERATION = "CONSTRAINED_REGENERATION"
    DECOMPOSITION_REPAIR = "DECOMPOSITION_REPAIR"
    BROADEN_RETRIEVAL = "BROADEN_RETRIEVAL"
    BOUNDED_RETRY = "BOUNDED_RETRY"
    ALTERNATE_PATH = "ALTERNATE_PATH"
    TOOL_REPAIR = "TOOL_REPAIR"
    ABSTAIN = "ABSTAIN"


class FailureClass(str, Enum):
    """Top level failure classes of the taxonomy."""

    NONE = "NONE"
    QUERY_FAILURE = "QUERY_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    RANKING_FAILURE = "RANKING_FAILURE"
    CONTEXT_FAILURE = "CONTEXT_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    AGENT_CONTROL_FAILURE = "AGENT_CONTROL_FAILURE"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    RESOURCE_FAILURE = "RESOURCE_FAILURE"


@dataclass(frozen=True)
class FailureDefinition:
    """Declarative metadata for one failure subtype."""

    subtype: str
    description: str
    stage: Stage
    default_severity: Severity
    recovery_actions: Tuple[RecoveryAction, ...]
    evidence_hints: Tuple[str, ...] = field(default_factory=tuple)
    cost_hint: int = 1
    reusable_stage: Optional[Stage] = None


class FailureTaxonomy:
    """Registry of every failure subtype understood by the system."""

    _registry: Dict[str, FailureDefinition] = {}

    @classmethod
    def register(cls, definition: FailureDefinition) -> FailureDefinition:
        cls._registry[definition.subtype] = definition
        return definition

    @classmethod
    def get(cls, subtype: str) -> Optional[FailureDefinition]:
        return cls._registry.get(subtype)

    @classmethod
    def all(cls) -> Dict[str, FailureDefinition]:
        return dict(cls._registry)

    @classmethod
    def subtypes(cls, failure_class: Optional[FailureClass] = None) -> List[str]:
        out = []
        for name, defn in cls._registry.items():
            if failure_class is None or _class_of(defn.subtype) == failure_class:
                out.append(name)
        return sorted(out)

    @classmethod
    def stages(cls) -> List[str]:
        return [s.value for s in Stage]


def _class_of(subtype: str) -> FailureClass:
    mapping = {
        "ambiguous_query": FailureClass.QUERY_FAILURE,
        "incorrect_rewrite": FailureClass.QUERY_FAILURE,
        "incomplete_decomposition": FailureClass.QUERY_FAILURE,
        "over_decomposition": FailureClass.QUERY_FAILURE,
        "low_recall": FailureClass.RETRIEVAL_FAILURE,
        "irrelevant_retrieval": FailureClass.RETRIEVAL_FAILURE,
        "no_useful_evidence": FailureClass.RETRIEVAL_FAILURE,
        "stale_documents": FailureClass.RETRIEVAL_FAILURE,
        "empty_retrieval": FailureClass.RETRIEVAL_FAILURE,
        "relevant_ranked_low": FailureClass.RANKING_FAILURE,
        "irrelevant_ranked_high": FailureClass.RANKING_FAILURE,
        "poor_reranking": FailureClass.RANKING_FAILURE,
        "fragmented_context": FailureClass.CONTEXT_FAILURE,
        "insufficient_context": FailureClass.CONTEXT_FAILURE,
        "excessive_context": FailureClass.CONTEXT_FAILURE,
        "conflicting_evidence": FailureClass.CONTEXT_FAILURE,
        "missing_evidence": FailureClass.CONTEXT_FAILURE,
        "hallucination": FailureClass.GENERATION_FAILURE,
        "unsupported_claim": FailureClass.GENERATION_FAILURE,
        "incomplete_answer": FailureClass.GENERATION_FAILURE,
        "contradiction": FailureClass.GENERATION_FAILURE,
        "wrong_tool_selection": FailureClass.AGENT_CONTROL_FAILURE,
        "unnecessary_tool_call": FailureClass.AGENT_CONTROL_FAILURE,
        "repeated_action": FailureClass.AGENT_CONTROL_FAILURE,
        "incorrect_workflow": FailureClass.AGENT_CONTROL_FAILURE,
        "retrieval_service_unavailable": FailureClass.INFRASTRUCTURE_FAILURE,
        "timeout": FailureClass.INFRASTRUCTURE_FAILURE,
        "malformed_response": FailureClass.INFRASTRUCTURE_FAILURE,
        "embedding_failure": FailureClass.INFRASTRUCTURE_FAILURE,
        "model_failure": FailureClass.INFRASTRUCTURE_FAILURE,
        "budget_exhausted": FailureClass.RESOURCE_FAILURE,
        "latency_budget_exceeded": FailureClass.RESOURCE_FAILURE,
        "token_budget_exceeded": FailureClass.RESOURCE_FAILURE,
    }
    return mapping.get(subtype, FailureClass.QUERY_FAILURE)


def _define(
    subtype: str,
    description: str,
    stage: Stage,
    severity: Severity,
    actions: Tuple[RecoveryAction, ...],
    hints: Tuple[str, ...] = (),
    cost: int = 1,
    reusable: Optional[Stage] = None,
) -> FailureDefinition:
    return FailureTaxonomy.register(
        FailureDefinition(
            subtype=subtype,
            description=description,
            stage=stage,
            default_severity=severity,
            recovery_actions=actions,
            evidence_hints=hints,
            cost_hint=cost,
            reusable_stage=reusable,
        )
    )


# --------------------------------------------------------------------------
# QUERY FAILURES
# --------------------------------------------------------------------------
_define(
    "ambiguous_query",
    "Query is ambiguous or under-specified; different readings yield different evidence needs.",
    Stage.QUERY_UNDERSTANDING,
    Severity.MEDIUM,
    (RecoveryAction.QUERY_REWRITE, RecoveryAction.DECOMPOSITION_REPAIR, RecoveryAction.ABSTAIN),
    ("query_too_short", "multiple_entity_mentions", "no_anchor_terms"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "incorrect_rewrite",
    "Query rewrite changed the meaning of the original question.",
    Stage.QUERY_UNDERSTANDING,
    Severity.HIGH,
    (RecoveryAction.QUERY_REWRITE, RecoveryAction.ABSTAIN),
    ("rewrite_drift_high", "key_terms_dropped"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "incomplete_decomposition",
    "Complex query was decomposed into too few / wrong sub-questions.",
    Stage.QUERY_DECOMPOSITION,
    Severity.HIGH,
    (RecoveryAction.DECOMPOSITION_REPAIR, RecoveryAction.QUERY_REWRITE, RecoveryAction.ABSTAIN),
    ("subquery_count_below_expected", "constraint_not_covered"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "over_decomposition",
    "Query was split into redundant sub-questions producing duplicated evidence.",
    Stage.QUERY_DECOMPOSITION,
    Severity.LOW,
    (RecoveryAction.DECOMPOSITION_REPAIR, RecoveryAction.ABSTAIN),
    ("redundant_subqueries",),
    reusable=Stage.RETRIEVAL,
)

# --------------------------------------------------------------------------
# RETRIEVAL FAILURES
# --------------------------------------------------------------------------
_define(
    "low_recall",
    "Relevant chunks exist in the corpus but were not retrieved.",
    Stage.RETRIEVAL,
    Severity.HIGH,
    (
        RecoveryAction.QUERY_REWRITE,
        RecoveryAction.HYBRID_SEARCH,
        RecoveryAction.EXPAND_TOP_K,
        RecoveryAction.BROADEN_RETRIEVAL,
        RecoveryAction.ABSTAIN,
    ),
    ("mean_score_below_threshold", "score_gap_small", "no_lexical_overlap"),
    cost=1,
    reusable=Stage.QUERY_UNDERSTANDING,
)
_define(
    "irrelevant_retrieval",
    "Retrieved chunks are topically unrelated to the question.",
    Stage.RETRIEVAL,
    Severity.HIGH,
    (RecoveryAction.QUERY_REWRITE, RecoveryAction.HYBRID_SEARCH, RecoveryAction.RERANK, RecoveryAction.ABSTAIN),
    ("top_score_low", "query_chunk_overlap_near_zero"),
    reusable=Stage.QUERY_UNDERSTANDING,
)
_define(
    "no_useful_evidence",
    "Retrieval returned nothing that can support an answer.",
    Stage.RETRIEVAL,
    Severity.CRITICAL,
    (
        RecoveryAction.QUERY_REWRITE,
        RecoveryAction.HYBRID_SEARCH,
        RecoveryAction.EXPAND_TOP_K,
        RecoveryAction.BROADEN_RETRIEVAL,
        RecoveryAction.ABSTAIN,
    ),
    ("zero_useful_chunks",),
    cost=1,
    reusable=Stage.QUERY_UNDERSTANDING,
)
_define(
    "stale_documents",
    "Retrieved evidence comes from outdated document versions.",
    Stage.RETRIEVAL,
    Severity.MEDIUM,
    (RecoveryAction.ALTERNATE_PATH, RecoveryAction.BOUNDED_RETRY, RecoveryAction.ABSTAIN),
    ("stale_timestamp", "superseded_document"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "empty_retrieval",
    "Retrieval returned zero results (index/service level emptiness).",
    Stage.RETRIEVAL,
    Severity.CRITICAL,
    (RecoveryAction.BOUNDED_RETRY, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("result_count_zero",),
    reusable=Stage.QUERY_UNDERSTANDING,
)

# --------------------------------------------------------------------------
# RANKING FAILURES
# --------------------------------------------------------------------------
_define(
    "relevant_ranked_low",
    "Relevant chunks exist in the candidate set but were pushed below the cut.",
    Stage.RANKING,
    Severity.HIGH,
    (RecoveryAction.RERANK, RecoveryAction.EXPAND_TOP_K, RecoveryAction.ABSTAIN),
    ("relevant_at_rank_beyond_k", "rank_regression"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "irrelevant_ranked_high",
    "Irrelevant chunk occupies a top context slot.",
    Stage.RANKING,
    Severity.MEDIUM,
    (RecoveryAction.RERANK, RecoveryAction.CHUNK_EXPANSION, RecoveryAction.ABSTAIN),
    ("irrelevant_at_rank_1", "score_margin_small"),
    reusable=Stage.RETRIEVAL,
)
_define(
    "poor_reranking",
    "Reranker produced a worse ordering than the base retriever.",
    Stage.RANKING,
    Severity.MEDIUM,
    (RecoveryAction.RERANK, RecoveryAction.ABSTAIN),
    ("rerank_score_flat", "mrr_regression_after_rerank"),
    reusable=Stage.RETRIEVAL,
)

# --------------------------------------------------------------------------
# CONTEXT FAILURES
# --------------------------------------------------------------------------
_define(
    "fragmented_context",
    "Answer-bearing sentences were split across chunks so the evidence is unusable.",
    Stage.CONTEXT_ASSEMBLY,
    Severity.HIGH,
    (RecoveryAction.CHUNK_EXPANSION, RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.ABSTAIN),
    ("high_fragmentation_ratio", "adjacent_chunk_gap", "coverage_low"),
    reusable=Stage.RANKING,
)
_define(
    "insufficient_context",
    "Context is too small to answer the question.",
    Stage.CONTEXT_ASSEMBLY,
    Severity.HIGH,
    (RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.EXPAND_TOP_K, RecoveryAction.CHUNK_EXPANSION, RecoveryAction.ABSTAIN),
    ("context_tokens_below_minimum", "coverage_below_threshold"),
    reusable=Stage.RANKING,
)
_define(
    "excessive_context",
    "Context is too large and dilutes the relevant evidence.",
    Stage.CONTEXT_ASSEMBLY,
    Severity.MEDIUM,
    (RecoveryAction.RERANK, RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.ABSTAIN),
    ("context_tokens_above_maximum", "relevant_chunk_buried", "low_precision_at_depth"),
    reusable=Stage.RANKING,
)
_define(
    "conflicting_evidence",
    "Context contains mutually inconsistent statements about the answer.",
    Stage.CONTEXT_ASSEMBLY,
    Severity.HIGH,
    (RecoveryAction.RERANK, RecoveryAction.CHUNK_EXPANSION, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("contradiction_detected", "negation_conflict", "numeric_conflict"),
    reusable=Stage.RANKING,
)
_define(
    "missing_evidence",
    "A constraint of the question is not covered by any retrieved chunk.",
    Stage.CONTEXT_ASSEMBLY,
    Severity.HIGH,
    (RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.EXPAND_TOP_K, RecoveryAction.HYBRID_SEARCH, RecoveryAction.ABSTAIN),
    ("constraint_coverage_below_1",),
    reusable=Stage.RANKING,
)

# --------------------------------------------------------------------------
# GENERATION FAILURES
# --------------------------------------------------------------------------
_define(
    "hallucination",
    "Answer contains fabricated content with no support in the evidence.",
    Stage.GENERATION,
    Severity.CRITICAL,
    (RecoveryAction.CONSTRAINED_REGENERATION, RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.ABSTAIN),
    ("groundedness_below_threshold", "unsupported_claims_present"),
    cost=2,
    reusable=Stage.CONTEXT_ASSEMBLY,
)
_define(
    "unsupported_claim",
    "At least one claim in the answer is not entailed by the retrieved evidence.",
    Stage.GENERATION,
    Severity.HIGH,
    (RecoveryAction.CONSTRAINED_REGENERATION, RecoveryAction.ABSTAIN),
    ("claim_support_below_threshold", "citation_coverage_low"),
    cost=2,
    reusable=Stage.CONTEXT_ASSEMBLY,
)
_define(
    "incomplete_answer",
    "Answer omits part of what the question asked for.",
    Stage.GENERATION,
    Severity.MEDIUM,
    (RecoveryAction.CONSTRAINED_REGENERATION, RecoveryAction.CONTEXT_EXPANSION, RecoveryAction.ABSTAIN),
    ("subquestion_coverage_below_1",),
    cost=2,
    reusable=Stage.CONTEXT_ASSEMBLY,
)
_define(
    "contradiction",
    "Answer contradicts the retrieved evidence.",
    Stage.GENERATION,
    Severity.CRITICAL,
    (RecoveryAction.CONSTRAINED_REGENERATION, RecoveryAction.ABSTAIN),
    ("answer_evidence_negation_conflict",),
    cost=2,
    reusable=Stage.CONTEXT_ASSEMBLY,
)

# --------------------------------------------------------------------------
# AGENT CONTROL FAILURES
# --------------------------------------------------------------------------
_define(
    "wrong_tool_selection",
    "Agent invoked a tool that cannot serve the current subgoal.",
    Stage.AGENT_CONTROL,
    Severity.MEDIUM,
    (RecoveryAction.TOOL_REPAIR, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("tool_result_relevance_zero",),
)
_define(
    "unnecessary_tool_call",
    "Agent invoked a tool whose result is unused by the final answer.",
    Stage.AGENT_CONTROL,
    Severity.LOW,
    (RecoveryAction.TOOL_REPAIR,),
    ("unused_tool_results",),
)
_define(
    "repeated_action",
    "Agent repeated an action that already failed.",
    Stage.AGENT_CONTROL,
    Severity.HIGH,
    (RecoveryAction.TOOL_REPAIR, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("duplicate_action_signature", "max_repeats_exceeded"),
)
_define(
    "incorrect_workflow",
    "Agent followed a plan that cannot reach the goal.",
    Stage.AGENT_CONTROL,
    Severity.HIGH,
    (RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("dead_end_state",),
)

# --------------------------------------------------------------------------
# INFRASTRUCTURE FAILURES
# --------------------------------------------------------------------------
_define(
    "retrieval_service_unavailable",
    "Retrieval backend/vector store could not be reached.",
    Stage.INFRASTRUCTURE,
    Severity.CRITICAL,
    (RecoveryAction.ALTERNATE_PATH, RecoveryAction.BOUNDED_RETRY, RecoveryAction.ABSTAIN),
    ("service_unavailable", "exception_type"),
)
_define(
    "timeout",
    "A pipeline stage exceeded its time budget.",
    Stage.INFRASTRUCTURE,
    Severity.HIGH,
    (RecoveryAction.BOUNDED_RETRY, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("latency_exceeds_budget",),
)
_define(
    "malformed_response",
    "Downstream model/service returned unparseable output.",
    Stage.INFRASTRUCTURE,
    Severity.HIGH,
    (RecoveryAction.BOUNDED_RETRY, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("json_parse_error", "schema_violation"),
)
_define(
    "embedding_failure",
    "Embedding backend failed (dimension mismatch, empty vector, service error).",
    Stage.INFRASTRUCTURE,
    Severity.CRITICAL,
    (RecoveryAction.ALTERNATE_PATH, RecoveryAction.BOUNDED_RETRY, RecoveryAction.ABSTAIN),
    ("embedding_exception", "dimension_mismatch"),
)
_define(
    "model_failure",
    "LLM provider returned an error or empty completion.",
    Stage.INFRASTRUCTURE,
    Severity.CRITICAL,
    (RecoveryAction.BOUNDED_RETRY, RecoveryAction.ALTERNATE_PATH, RecoveryAction.ABSTAIN),
    ("provider_error", "empty_completion"),
)

# --------------------------------------------------------------------------
# RESOURCE FAILURES (control-loop budget violations)
# --------------------------------------------------------------------------
_define(
    "budget_exhausted",
    "Recovery attempt budget exhausted before a verified answer.",
    Stage.AGENT_CONTROL,
    Severity.HIGH,
    (RecoveryAction.ABSTAIN,),
    ("attempts_remaining_zero",),
)
_define(
    "latency_budget_exceeded",
    "Total latency budget for the request was exhausted.",
    Stage.AGENT_CONTROL,
    Severity.HIGH,
    (RecoveryAction.ABSTAIN,),
    ("latency_budget_remaining_zero",),
)
_define(
    "token_budget_exceeded",
    "Token/cost budget for the request was exhausted.",
    Stage.AGENT_CONTROL,
    Severity.HIGH,
    (RecoveryAction.ABSTAIN,),
    ("token_budget_remaining_zero",),
)


def default_recovery_map(failure_class: FailureClass) -> Tuple[RecoveryAction, ...]:
    """Escalation ladder used when a subtype is not known."""
    return {
        FailureClass.QUERY_FAILURE: (
            RecoveryAction.QUERY_REWRITE,
            RecoveryAction.DECOMPOSITION_REPAIR,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.RETRIEVAL_FAILURE: (
            RecoveryAction.QUERY_REWRITE,
            RecoveryAction.HYBRID_SEARCH,
            RecoveryAction.EXPAND_TOP_K,
            RecoveryAction.BROADEN_RETRIEVAL,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.RANKING_FAILURE: (
            RecoveryAction.RERANK,
            RecoveryAction.EXPAND_TOP_K,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.CONTEXT_FAILURE: (
            RecoveryAction.CHUNK_EXPANSION,
            RecoveryAction.CONTEXT_EXPANSION,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.GENERATION_FAILURE: (
            RecoveryAction.CONSTRAINED_REGENERATION,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.AGENT_CONTROL_FAILURE: (
            RecoveryAction.TOOL_REPAIR,
            RecoveryAction.ALTERNATE_PATH,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.INFRASTRUCTURE_FAILURE: (
            RecoveryAction.BOUNDED_RETRY,
            RecoveryAction.ALTERNATE_PATH,
            RecoveryAction.ABSTAIN,
        ),
        FailureClass.RESOURCE_FAILURE: (RecoveryAction.ABSTAIN,),
        FailureClass.NONE: (RecoveryAction.ABSTAIN,),
    }[failure_class]


def recovery_ladder(failure_class: FailureClass, subtype: Optional[str] = None) -> Tuple[RecoveryAction, ...]:
    """Return the ordered escalation ladder for a diagnosed failure."""
    if subtype:
        defn = FailureTaxonomy.get(subtype)
        if defn and defn.recovery_actions:
            return defn.recovery_actions
    return default_recovery_map(failure_class)


def class_for_subtype(subtype: str) -> FailureClass:
    return _class_of(subtype)


def stage_for_subtype(subtype: str, default: Stage = Stage.RETRIEVAL) -> Stage:
    defn = FailureTaxonomy.get(subtype)
    return defn.stage if defn else default


def severity_for_subtype(subtype: str, default: Severity = Severity.MEDIUM) -> Severity:
    defn = FailureTaxonomy.get(subtype)
    return defn.default_severity if defn else default