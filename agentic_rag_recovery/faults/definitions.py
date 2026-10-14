"""Fault-injection framework.

Each fault has **declared ground truth** (``expected_failure_class`` /
``expected_subtype`` / ``expected_symptom``).  That ground truth is written to
SQLite at injection time and is used *only* when scoring diagnosis accuracy -
never by the detector, the diagnoser or the recovery planner.  The diagnoser
never receives this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..models import FaultInjection
from ..taxonomy import FailureClass, RecoveryAction, Severity, Stage


@dataclass(frozen=True)
class FaultDefinition:
    fault_type: str
    title: str
    target_stage: Stage
    description: str
    expected_symptom: str
    expected_failure_class: FailureClass
    expected_subtype: str
    severity: Severity = Severity.MEDIUM
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = "semantic"
    targets: tuple = ()  # taxonomy subtypes it may surface as

    def build(self, parameters: Optional[Dict[str, Any]] = None, experiment_id: Optional[str] = None) -> FaultInjection:
        params = {**self.default_parameters, **(parameters or {})}
        return FaultInjection(
            fault_type=self.fault_type,
            target_stage=self.target_stage,
            severity=self.severity,
            description=self.description,
            expected_symptom=self.expected_symptom,
            expected_failure_class=self.expected_failure_class,
            expected_subtype=self.expected_subtype,
            injection_parameters=params,
            active=True,
            experiment_id=experiment_id,
        )


# ---------------------------------------------------------------------------
# 1. QUERY CORRUPTION
# ---------------------------------------------------------------------------
QUERY_CORRUPTION = FaultDefinition(
    fault_type="QUERY_CORRUPTION",
    title="Query corruption (word dropout / noise injection)",
    target_stage=Stage.QUERY_UNDERSTANDING,
    description=(
        "Drop, duplicate or corrupt content words of the user query before it reaches "
        "retrieval, simulating typos, truncation and ASR/OCR noise."
    ),
    expected_symptom="retrieval returns topically unrelated chunks",
    expected_failure_class=FailureClass.RETRIEVAL_FAILURE,
    expected_subtype="low_recall",
    severity=Severity.HIGH,
    default_parameters={"dropout": 0.75, "seed": 7},
    category="semantic",
    targets=("low_recall", "irrelevant_retrieval"),
)

# 2. BAD QUERY REWRITE
BAD_QUERY_REWRITE = FaultDefinition(
    fault_type="BAD_QUERY_REWRITE",
    title="Bad query rewrite (semantic drift)",
    target_stage=Stage.QUERY_UNDERSTANDING,
    description=(
        "Replace the user's question with a drifting paraphrase that keeps none of the "
        "anchor terms, so retrieval runs on the wrong topic."
    ),
    expected_symptom="retrieved evidence is on a different topic than the question",
    expected_failure_class=FailureClass.QUERY_FAILURE,
    expected_subtype="incorrect_rewrite",
    severity=Severity.HIGH,
    default_parameters={"drift": 0.85},
    category="agentic",
    targets=("incorrect_rewrite", "low_recall"),
)

# 3. RELEVANT DOCUMENT REMOVAL
RELEVANT_DOC_REMOVAL = FaultDefinition(
    fault_type="RELEVANT_DOC_REMOVAL",
    title="Relevant document removal",
    target_stage=Stage.RETRIEVAL,
    description=(
        "Remove every chunk of the document that contains the answer from the searchable "
        "index, simulating a corpus that genuinely lacks the evidence."
    ),
    expected_symptom="no chunk can answer the question at any retrieval effort",
    expected_failure_class=FailureClass.RETRIEVAL_FAILURE,
    expected_subtype="no_useful_evidence",
    severity=Severity.CRITICAL,
    default_parameters={"remove_all_ground_truth": True},
    category="data",
    targets=("no_useful_evidence", "low_recall"),
)

# 4. IRRELEVANT DOCUMENT INJECTION
IRRELEVANT_DOC_INJECTION = FaultDefinition(
    fault_type="IRRELEVANT_DOC_INJECTION",
    title="Irrelevant document injection",
    target_stage=Stage.RETRIEVAL,
    description="Replace the result list with chunks that share no vocabulary with the query.",
    expected_symptom="top-k results are unrelated to the question",
    expected_failure_class=FailureClass.RETRIEVAL_FAILURE,
    expected_subtype="irrelevant_retrieval",
    severity=Severity.HIGH,
    default_parameters={},
    category="data",
    targets=("irrelevant_retrieval", "no_useful_evidence"),
)

# 5. LOW TOP-K
LOW_TOP_K = FaultDefinition(
    fault_type="LOW_TOP_K",
    title="Low top-k truncation",
    target_stage=Stage.RETRIEVAL,
    description="Truncate the candidate list to an unrealistically small top-k.",
    expected_symptom="relevant chunk exists in the corpus but falls outside the cut",
    expected_failure_class=FailureClass.RANKING_FAILURE,
    expected_subtype="relevant_ranked_low",
    severity=Severity.MEDIUM,
    default_parameters={"top_k_cap": 1},
    category="configuration",
    targets=("relevant_ranked_low", "insufficient_context"),
)

# 6. RANKING CORRUPTION
RANKING_CORRUPTION = FaultDefinition(
    fault_type="RANKING_CORRUPTION",
    title="Ranking / reranking corruption",
    target_stage=Stage.RANKING,
    description="Replace the reranker with one that inverts or randomises the candidate order.",
    expected_symptom="relevant chunks move down, irrelevant chunks rise",
    expected_failure_class=FailureClass.RANKING_FAILURE,
    expected_subtype="poor_reranking",
    severity=Severity.MEDIUM,
    default_parameters={"mode": "reverse"},
    category="algorithmic",
    targets=("poor_reranking", "irrelevant_ranked_high", "relevant_ranked_low"),
)

# 7. CHUNK FRAGMENTATION
CHUNK_FRAGMENTATION = FaultDefinition(
    fault_type="CHUNK_FRAGMENTATION",
    title="Chunk fragmentation (aggressive re-chunking)",
    target_stage=Stage.CONTEXT_ASSEMBLY,
    description=(
        "Re-chunk the corpus with a tiny window so answer sentences are split across "
        "neighbouring chunks and become unusable in isolation."
    ),
    expected_symptom="retrieved chunks individually cover only part of the evidence",
    expected_failure_class=FailureClass.CONTEXT_FAILURE,
    expected_subtype="fragmented_context",
    severity=Severity.HIGH,
    default_parameters={"chunk_size": 50, "chunk_overlap": 6},
    category="configuration",
    targets=("fragmented_context", "insufficient_context"),
)

# 8. CONTEXT TRUNCATION
CONTEXT_TRUNCATION = FaultDefinition(
    fault_type="CONTEXT_TRUNCATION",
    title="Context truncation (too few tokens)",
    target_stage=Stage.CONTEXT_ASSEMBLY,
    description="Truncate the assembled context to a tiny token budget.",
    expected_symptom="context cannot cover the question's constraints",
    expected_failure_class=FailureClass.CONTEXT_FAILURE,
    expected_subtype="insufficient_context",
    severity=Severity.HIGH,
    default_parameters={"max_context_tokens": 40, "max_chunks": 1},
    category="configuration",
    targets=("insufficient_context", "incomplete_answer"),
)

# 9. CONFLICTING EVIDENCE
CONFLICTING_EVIDENCE = FaultDefinition(
    fault_type="CONFLICTING_EVIDENCE",
    title="Conflicting evidence injection",
    target_stage=Stage.CONTEXT_ASSEMBLY,
    description=(
        "Inject a decoy chunk that negates or contradicts the key numeric/entity claim of "
        "the supporting evidence."
    ),
    expected_symptom="context contains mutually inconsistent statements",
    expected_failure_class=FailureClass.CONTEXT_FAILURE,
    expected_subtype="conflicting_evidence",
    severity=Severity.HIGH,
    default_parameters={},
    category="data",
    targets=("conflicting_evidence",),
)

# 10. STALE DOCUMENT
STALE_DOCUMENT = FaultDefinition(
    fault_type="STALE_DOCUMENT",
    title="Stale document (superseded version)",
    target_stage=Stage.RETRIEVAL,
    description="Pin the search to an outdated document version instead of the current one.",
    expected_symptom="evidence is outdated but structurally valid",
    expected_failure_class=FailureClass.RETRIEVAL_FAILURE,
    expected_subtype="stale_documents",
    severity=Severity.MEDIUM,
    default_parameters={},
    category="data",
    targets=("stale_documents",),
)

# 11. GENERATION CONSTRAINT REMOVAL
GENERATION_CONSTRAINT_REMOVAL = FaultDefinition(
    fault_type="GENERATION_CONSTRAINT_REMOVAL",
    title="Generation grounding-constraint removal",
    target_stage=Stage.GENERATION,
    description=(
        "Disable the grounding instruction so the generator is free to produce content "
        "that is not entailed by the retrieved evidence."
    ),
    expected_symptom="answer contains unsupported claims and no citations",
    expected_failure_class=FailureClass.GENERATION_FAILURE,
    expected_subtype="hallucination",
    severity=Severity.CRITICAL,
    default_parameters={},
    category="agentic",
    targets=("hallucination", "unsupported_claim"),
)

# 12. TOOL TIMEOUT
TOOL_TIMEOUT = FaultDefinition(
    fault_type="TOOL_TIMEOUT",
    title="Tool timeout (retrieval tool)",
    target_stage=Stage.RETRIEVAL,
    description="Make the retrieval tool exceed its time budget.",
    expected_symptom="stage exceeds latency budget",
    expected_failure_class=FailureClass.INFRASTRUCTURE_FAILURE,
    expected_subtype="timeout",
    severity=Severity.HIGH,
    default_parameters={},
    category="infrastructure",
    targets=("timeout",),
)

# 13. RETRIEVAL TIMEOUT (alias, distinct id kept for clarity)
RETRIEVAL_TIMEOUT = FaultDefinition(
    fault_type="RETRIEVAL_TIMEOUT",
    title="Retrieval service timeout",
    target_stage=Stage.INFRASTRUCTURE,
    description="The vector store becomes unreachable / times out mid-request.",
    expected_symptom="retrieval raises a timeout before returning any result",
    expected_failure_class=FailureClass.INFRASTRUCTURE_FAILURE,
    expected_subtype="timeout",
    severity=Severity.HIGH,
    default_parameters={},
    category="infrastructure",
    targets=("timeout", "retrieval_service_unavailable"),
)

# 14. EMPTY RETRIEVAL
EMPTY_RETRIEVAL = FaultDefinition(
    fault_type="EMPTY_RETRIEVAL",
    title="Empty retrieval result",
    target_stage=Stage.RETRIEVAL,
    description="Retrieval returns zero candidates.",
    expected_symptom="zero results returned to the ranking stage",
    expected_failure_class=FailureClass.RETRIEVAL_FAILURE,
    expected_subtype="empty_retrieval",
    severity=Severity.CRITICAL,
    default_parameters={},
    category="infrastructure",
    targets=("empty_retrieval", "no_useful_evidence"),
)

# 15. EMBEDDING FAILURE
EMBEDDING_FAILURE = FaultDefinition(
    fault_type="EMBEDDING_FAILURE",
    title="Embedding backend failure",
    target_stage=Stage.INFRASTRUCTURE,
    description="The embedding backend raises, producing dimension/NaN vectors.",
    expected_symptom="embedding error recorded; dense retrieval unusable",
    expected_failure_class=FailureClass.INFRASTRUCTURE_FAILURE,
    expected_subtype="embedding_failure",
    severity=Severity.CRITICAL,
    default_parameters={},
    category="infrastructure",
    targets=("embedding_failure",),
)

# 16. MODEL FAILURE (extra: malformed model output)
MODEL_FAILURE = FaultDefinition(
    fault_type="MODEL_FAILURE",
    title="Malformed model response",
    target_stage=Stage.INFRASTRUCTURE,
    description="The LLM returns output that cannot be parsed / is empty.",
    expected_symptom="parse error on model response",
    expected_failure_class=FailureClass.INFRASTRUCTURE_FAILURE,
    expected_subtype="malformed_response",
    severity=Severity.HIGH,
    default_parameters={},
    category="infrastructure",
    targets=("malformed_response", "model_failure"),
)

# 17. AMBIGUOUS QUERY (pre-evidence procedural failure)
AMBIGUOUS_QUERY = FaultDefinition(
    fault_type="AMBIGUOUS_QUERY",
    title="Ambiguous query injection",
    target_stage=Stage.QUERY_UNDERSTANDING,
    description="Reduce the query to an under-specified fragment before processing.",
    expected_symptom="query under-specified at query-understanding time",
    expected_failure_class=FailureClass.QUERY_FAILURE,
    expected_subtype="ambiguous_query",
    severity=Severity.MEDIUM,
    default_parameters={"keep_words": 2},
    category="agentic",
    targets=("ambiguous_query",),
)

# 18. INCOMPLETE DECOMPOSITION
INCOMPLETE_DECOMPOSITION = FaultDefinition(
    fault_type="INCOMPLETE_DECOMPOSITION",
    title="Incomplete query decomposition",
    target_stage=Stage.QUERY_DECOMPOSITION,
    description="Drop all but the first sub-question of a complex question.",
    expected_symptom="sub-questions cover only part of the question",
    expected_failure_class=FailureClass.QUERY_FAILURE,
    expected_subtype="incomplete_decomposition",
    severity=Severity.HIGH,
    default_parameters={"keep": 1},
    category="agentic",
    targets=("incomplete_decomposition", "missing_evidence"),
)


ALL_FAULTS: Dict[str, FaultDefinition] = {
    definition.fault_type: definition
    for definition in [
        QUERY_CORRUPTION,
        BAD_QUERY_REWRITE,
        RELEVANT_DOC_REMOVAL,
        IRRELEVANT_DOC_INJECTION,
        LOW_TOP_K,
        RANKING_CORRUPTION,
        CHUNK_FRAGMENTATION,
        CONTEXT_TRUNCATION,
        CONFLICTING_EVIDENCE,
        STALE_DOCUMENT,
        GENERATION_CONSTRAINT_REMOVAL,
        TOOL_TIMEOUT,
        RETRIEVAL_TIMEOUT,
        EMPTY_RETRIEVAL,
        EMBEDDING_FAILURE,
        MODEL_FAILURE,
        AMBIGUOUS_QUERY,
        INCOMPLETE_DECOMPOSITION,
    ]
}


class FaultRegistry:
    """Lookup of the declared fault catalogue."""

    @staticmethod
    def get(fault_type: str) -> Optional[FaultDefinition]:
        return ALL_FAULTS.get(str(fault_type).upper())

    @staticmethod
    def list_all() -> List[FaultDefinition]:
        return list(ALL_FAULTS.values())

    @staticmethod
    def by_category(category: str) -> List[FaultDefinition]:
        return [d for d in ALL_FAULTS.values() if d.category == category]

    @staticmethod
    def catalogue() -> List[Dict[str, Any]]:
        return [
            {
                "fault_type": d.fault_type,
                "title": d.title,
                "target_stage": d.target_stage.value,
                "category": d.category,
                "description": d.description,
                "expected_symptom": d.expected_symptom,
                "expected_failure_class": d.expected_failure_class.value,
                "expected_subtype": d.expected_subtype,
                "severity": d.severity.value,
                "default_parameters": d.default_parameters,
                "possible_subtypes": list(d.targets),
            }
            for d in ALL_FAULTS.values()
        ]

    @staticmethod
    def types() -> List[str]:
        return sorted(ALL_FAULTS.keys())