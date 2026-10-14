"""Signal collection, detection rules and diagnosis."""

from __future__ import annotations

import json

import pytest

from agentic_rag_recovery.detection.rules import RULES, RuleDetector
from agentic_rag_recovery.diagnosis.engine import DiagnosisEngine
from agentic_rag_recovery.diagnosis.llm_diagnoser import LLMDiagnoser
from agentic_rag_recovery.diagnosis.schema import (
    DiagnosisSchema,
    extract_json,
    taxonomy_for_prompt,
    validate_diagnosis_payload,
    validate_diagnosis_text,
)
from agentic_rag_recovery.llm.mock import MockProvider
from agentic_rag_recovery.models import DiagnosisTier, SignalBundle
from agentic_rag_recovery.signals.collector import SignalCollector, signals_for_diagnoser
from agentic_rag_recovery.signals.metrics import (
    answer_correctness,
    claim_support,
    constraint_coverage,
    context_coverage,
    conflict_score,
    groundedness_score,
    mrr,
    precision_at_k,
    recall_at_k,
    repetition_score,
    split_claims,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_split_claims_strips_citations():
    claims = split_claims("First claim here [S1]. Second claim there [S2].")
    assert len(claims) == 2
    assert all("[" not in c for c in claims)


def test_groundedness_high_for_supported_answer():
    context = ["The escalation policy requires acknowledgement within 5 minutes."]
    answer = "The escalation policy requires acknowledgement within 5 minutes. [S1]."
    result = groundedness_score("q", answer, context, ["S1"])
    assert result["groundedness"] > 0.9
    assert result["unsupported_claims"] == []


def test_groundedness_low_for_fabrication():
    context = ["The escalation policy requires acknowledgement within 5 minutes."]
    answer = "Refund processing always completes in exactly 17 hours. [S1]."
    result = groundedness_score("q", answer, context, ["S1"])
    assert result["groundedness"] < 0.5
    assert result["unsupported_claims"]


def test_groundedness_empty_answer_is_zero():
    result = groundedness_score("q", "", ["context"])
    assert result["groundedness"] == 0.0
    assert result["claim_count"] == 0


def test_citation_coverage_counts_markers():
    context = ["a", "b", "c"]
    result = groundedness_score("q", "Evidence here [S1] and [S3].", context, ["S1", "S2", "S3"])
    assert 0 < result["citation_coverage"] <= 1


def test_claim_support_detects_numeric_agreement():
    context = "Retries stop after 3 attempts."
    assert claim_support("Services stop after 3 attempts.", context) > 0.3


def test_conflict_detects_negation_contradiction():
    a = "Severity 1 incidents must be acknowledged within 5 minutes."
    b = "Severity 1 incidents must not be acknowledged within 5 minutes."
    score, conflicts = conflict_score([[a, b]])
    assert conflicts and score > 0


def test_conflict_detects_numeric_contradiction():
    a = "Backups are retained for 35 days in the retention window."
    b = "Backups are retained for 20 days in the retention window."
    _score, conflicts = conflict_score([[a, b]])
    assert any(c["type"] == "numeric" for c in conflicts)


def test_conflict_ignores_contrastive_claims():
    a = "Vendors that process customer data are audited every 12 months."
    b = "Vendors that do not process customer data are audited every 24 months."
    _score, conflicts = conflict_score([[a, b]])
    assert conflicts == []


def test_conflict_ignores_different_subjects():
    a = "Severity 1 incidents must be acknowledged within 5 minutes."
    b = "Severity 2 incidents must be acknowledged within 15 minutes."
    _score, conflicts = conflict_score([[a, b]])
    assert conflicts == []


def test_context_and_constraint_coverage():
    context = "The escalation policy requires on-call handover within 30 minutes."
    assert context_coverage("handover within 30 minutes", context) > 0.5
    value, missing = constraint_coverage(
        "q", ["how many minutes"], context
    )
    assert value == 1.0 and missing == []
    value, missing = constraint_coverage("q", ["how many hours"], context)
    assert value == 0.0 and missing


def test_retrieval_metrics():
    gold = {"a", "b"}
    ids = ["a", "x", "y", "b"]
    assert recall_at_k(ids, gold, 4) == 1.0
    assert precision_at_k(ids, gold, 4) == 0.5
    assert mrr(ids, gold) == 1.0
    assert mrr(["x", "y", "b"], gold) == pytest.approx(1 / 3)


def test_answer_correctness_scores():
    result = answer_correctness("5 minutes", "within 5 minutes", ["5", "minutes"])
    assert result["f1"] > 0.5
    assert result["keyword_coverage"] == 1.0
    assert answer_correctness("", "5 minutes")["f1"] == 0.0


def test_repetition_score_detects_degenerate_text():
    assert repetition_score("same words " * 20) > 0.5
    assert repetition_score("a varied sentence with many different words in it") < 0.2


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------
def test_signal_collector_produces_all_sections(app, dataset):
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(
        question.question,
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
        persist=False,
    )
    signals = outcome.run.signals
    assert signals is not None
    for section in ("query", "retrieval", "ranking", "context", "generation", "agent", "infrastructure"):
        assert isinstance(getattr(signals, section), dict)
        assert getattr(signals, section)


def test_signals_for_diagnoser_is_serialisable(app, dataset):
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(question.question, persist=False)
    payload = signals_for_diagnoser(outcome.run.signals)
    json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# detection rules
# ---------------------------------------------------------------------------
def _signals(**overrides) -> SignalBundle:
    base = {
        "query": {"query_words": 12, "is_ambiguous": False, "subquery_count": 1, "rewrite_present": False},
        "retrieval": {
            "retrieved_count": 5,
            "top_overlap": 0.8,
            "pre_rank_top_overlap": 0.8,
            "mean_overlap": 0.5,
            "useful_ratio": 0.8,
            "duplicate_ratio": 0.0,
            "source_diversity": 2,
            "score_gap": 0.4,
            "top_score": 1.4,
            "relevant_below_cut": 0,
        },
        "ranking": {"reranker_applied": True, "rank_regression_rate": 0.0, "irrelevant_at_rank1": False},
        "context": {
            "context_coverage": 0.9,
            "context_tokens": 400,
            "fragmentation_ratio": 0.0,
            "conflict_score": 0.0,
            "unconstrained_constraints": 0,
            "chunk_count": 5,
        },
        "generation": {
            "groundedness": 1.0,
            "unsupported_claims": 0,
            "citation_coverage": 0.6,
            "claim_count": 3,
            "answer_words": 40,
            "citations": 3,
        },
        "agent": {"repeat_actions": 0, "tool_calls": 0, "failed_actions": 0, "unused_tool_results": 0},
        "infrastructure": {"infra_errors": 0, "timeouts": 0, "latency_ms": 10.0},
        "budget": {"exhausted": {}},
    }
    for section, values in overrides.items():
        base[section].update(values)
    return SignalBundle(**base)


def test_rule_registry_is_populated():
    assert len(RULES) >= 25
    detector = RuleDetector()
    assert detector.tier == DiagnosisTier.RULE


def test_healthy_signals_produce_no_failure():
    detector = RuleDetector()
    assert detector.best(_signals()) is None


def test_empty_retrieval_rule():
    detector = RuleDetector()
    signals = _signals(retrieval={"retrieved_count": 0, "useful_ratio": 0.0, "top_overlap": 0.0})
    match = detector.best(signals)
    assert match is not None
    assert match.subtype == "empty_retrieval"
    assert match.confidence > 0.8


def test_hallucination_rule():
    detector = RuleDetector()
    signals = _signals(generation={"groundedness": 0.1, "unsupported_claims": 3, "citations": 0})
    match = detector.best(signals)
    assert match is not None and match.subtype == "hallucination"
    assert match.severity.value == "critical"


def test_context_failure_rules():
    detector = RuleDetector()
    signals = _signals(context={"context_tokens": 20})
    assert detector.best(signals).subtype == "insufficient_context"
    signals = _signals(context={"fragmentation_ratio": 0.9, "context_coverage": 0.5})
    assert detector.best(signals).subtype == "fragmented_context"
    signals = _signals(context={"conflict_score": 0.9, "conflict_count": 3})
    assert detector.best(signals).subtype == "conflicting_evidence"
    signals = _signals(context={"unconstrained_constraints": 2})
    assert detector.best(signals).subtype == "missing_evidence"


def test_ranking_failure_rules():
    detector = RuleDetector()
    signals = _signals(retrieval={"relevant_below_cut": 2})
    assert detector.best(signals).subtype == "relevant_ranked_low"
    signals = _signals(ranking={"irrelevant_at_rank1": True}, retrieval={"useful_ratio": 0.1})
    assert detector.best(signals).subtype == "irrelevant_ranked_high"


def test_infrastructure_rules_take_priority():
    detector = RuleDetector()
    signals = _signals(
        infrastructure={"timeouts": 1, "infra_errors": 1},
        retrieval={"retrieved_count": 0, "useful_ratio": 0.0, "top_overlap": 0.0},
    )
    match = detector.best(signals)
    assert match.stage.value == "infrastructure"


def test_block_repeat_rule_is_explicit():
    detector = RuleDetector()
    signals = _signals(agent={"repeat_actions": 5})
    match = detector.best(signals)
    assert match is not None and match.subtype == "repeated_action"


def test_budget_rule():
    detector = RuleDetector()
    signals = _signals(budget={"exhausted": {"attempts": True}, "attempts_used": 4})
    match = detector.best(signals)
    assert match is not None and match.subtype == "budget_exhausted"


# ---------------------------------------------------------------------------
# diagnosis schema
# ---------------------------------------------------------------------------
def test_extract_json_from_markdown_fence():
    payload, errors = extract_json('```json\n{"a": 1}\n```')
    assert payload == {"a": 1}
    assert errors == []


def test_extract_json_from_prose():
    payload, errors = extract_json('Here you go: {"a": 1} thanks')
    assert payload == {"a": 1}
    assert errors == []


def test_extract_json_failure_paths():
    assert extract_json("")[0] is None
    assert extract_json("no json here")[0] is None
    assert extract_json("[1,2,3]")[0] is None


def test_validate_diagnosis_payload_accepts_valid():
    outcome = validate_diagnosis_payload(
        {
            "failure_detected": True,
            "failure_class": "RETRIEVAL_FAILURE",
            "stage": "retrieval",
            "subtype": "low_recall",
            "severity": "high",
            "confidence": 0.8,
            "evidence": ["top score low"],
            "recommended_recovery": "QUERY_REWRITE",
        }
    )
    assert outcome.valid
    assert outcome.payload.subtype == "low_recall"


def test_validate_diagnosis_payload_repairs_unknown_subtype():
    outcome = validate_diagnosis_payload(
        {
            "failure_detected": True,
            "failure_class": "RETRIEVAL_FAILURE",
            "stage": "retrieval",
            "subtype": "totally_made_up",
            "severity": "high",
            "confidence": 0.8,
            "evidence": ["x"],
            "recommended_recovery": "QUERY_REWRITE",
        }
    )
    assert outcome.valid
    assert outcome.repairs
    assert outcome.payload.subtype in {t for t in _subtypes("RETRIEVAL_FAILURE")}


def test_validate_diagnosis_payload_fixes_confidence_scale():
    outcome = validate_diagnosis_payload(
        {
            "failure_detected": True,
            "failure_class": "GENERATION_FAILURE",
            "stage": "generation",
            "subtype": "hallucination",
            "severity": "critical",
            "confidence": "85",
            "evidence": "single string evidence",
            "recommended_recovery": "CONSTRAINED_REGENERATION",
        }
    )
    assert outcome.valid
    assert outcome.payload.confidence == pytest.approx(0.85)
    assert isinstance(outcome.payload.evidence, list)


def test_validate_diagnosis_payload_rejects_without_repair():
    outcome = validate_diagnosis_payload({"failure_detected": True}, repair=False)
    assert not outcome.valid
    assert outcome.errors


def test_validate_diagnosis_text_on_garbage():
    outcome = validate_diagnosis_text("<<<not json>>>")
    assert not outcome.valid


def test_taxonomy_for_prompt_is_complete():
    taxonomy = taxonomy_for_prompt()
    assert len(taxonomy) >= 30
    for entry in taxonomy:
        assert set(entry) >= {"subtype", "failure_class", "stage", "recovery_actions"}


def _subtypes(failure_class: str) -> list:
    from agentic_rag_recovery.taxonomy import FailureClass, FailureTaxonomy

    return FailureTaxonomy.subtypes(FailureClass[failure_class])


# ---------------------------------------------------------------------------
# two-tier engine
# ---------------------------------------------------------------------------
def test_rule_tier_wins_when_confident(settings):
    engine = DiagnosisEngine(settings=settings, llm_diagnoser=None)
    signals = _signals(retrieval={"retrieved_count": 0, "useful_ratio": 0.0, "top_overlap": 0.0})
    diagnosis = engine.diagnose(signals, query="q")
    assert diagnosis.tier == DiagnosisTier.RULE
    assert diagnosis.failure_detected
    assert engine.stats["rule"] == 1


def test_llm_tier_used_when_inconclusive(settings):
    from dataclasses import replace

    strict = replace(settings, diagnosis=replace(settings.diagnosis, rule_confidence_threshold=0.95))
    diagnoser = LLMDiagnoser(MockProvider(), strict)
    engine = DiagnosisEngine(settings=strict, llm_diagnoser=diagnoser)
    signals = _signals(
        retrieval={"top_overlap": 0.2, "pre_rank_top_overlap": 0.2, "useful_ratio": 0.3, "mean_overlap": 0.1}
    )
    diagnosis = engine.diagnose(signals, query="escalation policy")
    assert diagnosis.tier in {DiagnosisTier.RULE, DiagnosisTier.LLM}
    assert engine.stats["escalated"] == 1


def test_engine_never_raises_on_garbage_signals(settings):
    engine = DiagnosisEngine(settings=settings, llm_diagnoser=LLMDiagnoser(MockProvider(), settings))
    diagnosis = engine.diagnose(SignalBundle(), query="")
    assert diagnosis.tier in set(DiagnosisTier)


def test_engine_stats_snapshot(settings):
    engine = DiagnosisEngine(settings=settings, llm_diagnoser=None)
    engine.diagnose(_signals(), query="q")
    snapshot = engine.stats_snapshot()
    assert snapshot["total"] == 1
    assert "rule_rate" in snapshot


def test_llm_diagnoser_output_validates(settings):
    diagnoser = LLMDiagnoser(MockProvider(), settings)
    signals = _signals(generation={"groundedness": 0.0, "unsupported_claims": 2, "citations": 0})
    diagnosis = diagnoser.diagnose(signals, query="q")
    assert diagnosis.tier == DiagnosisTier.LLM
    assert diagnosis.failure_class.value == "GENERATION_FAILURE"
    assert diagnosis.recovery_ladder


def test_diagnosis_schema_rejects_unknown_subtype():
    with pytest.raises(Exception):
        DiagnosisSchema(
            failure_detected=True,
            failure_class="RETRIEVAL_FAILURE",
            stage="retrieval",
            subtype="not_a_subtype",
            severity="high",
            confidence=0.5,
            evidence=["x"],
            recommended_recovery="QUERY_REWRITE",
        )