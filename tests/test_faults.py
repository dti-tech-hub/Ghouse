"""Fault-injection framework: catalogue, ground truth and injection."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.faults.definitions import ALL_FAULTS, FaultDefinition, FaultRegistry
from agentic_rag_recovery.faults.injector import FaultInjector
from agentic_rag_recovery.taxonomy import FailureClass, FailureTaxonomy, RecoveryAction, Severity

REQUIRED_FAULTS = [
    "QUERY_CORRUPTION",
    "BAD_QUERY_REWRITE",
    "RELEVANT_DOC_REMOVAL",
    "IRRELEVANT_DOC_INJECTION",
    "LOW_TOP_K",
    "RANKING_CORRUPTION",
    "CHUNK_FRAGMENTATION",
    "CONTEXT_TRUNCATION",
    "CONFLICTING_EVIDENCE",
    "STALE_DOCUMENT",
    "GENERATION_CONSTRAINT_REMOVAL",
    "TOOL_TIMEOUT",
    "RETRIEVAL_TIMEOUT",
    "EMPTY_RETRIEVAL",
    "EMBEDDING_FAILURE",
]


def test_catalogue_covers_every_required_fault():
    available = FaultRegistry.types()
    for fault in REQUIRED_FAULTS:
        assert fault in available, f"{fault} is missing"


def test_every_fault_declares_ground_truth():
    for definition in FaultRegistry.list_all():
        assert definition.expected_failure_class is not None
        assert definition.expected_subtype
        assert FailureTaxonomy.get(definition.expected_subtype) is not None
        assert definition.expected_symptom
        assert definition.description


def test_expected_subtype_matches_declared_class():
    from agentic_rag_recovery.taxonomy import class_for_subtype

    for definition in FaultRegistry.list_all():
        assert class_for_subtype(definition.expected_subtype) is definition.expected_failure_class


def test_fault_build_carries_parameters():
    definition = FaultRegistry.get("LOW_TOP_K")
    fault = definition.build({"top_k_cap": 3}, experiment_id="exp_test")
    assert fault.injection_parameters["top_k_cap"] == 3
    assert fault.active
    assert fault.experiment_id == "exp_test"
    ground_truth = fault.ground_truth()
    assert ground_truth["expected_failure_class"]
    assert ground_truth["expected_symptom"]


def test_fault_recovery_ladder_is_defined():
    for definition in FaultRegistry.list_all():
        from agentic_rag_recovery.taxonomy import recovery_ladder

        ladder = recovery_ladder(definition.expected_failure_class, definition.expected_subtype)
        assert ladder[-1] is RecoveryAction.ABSTAIN


def test_catalogue_serialises():
    import json

    payload = FaultRegistry.catalogue()
    assert len(payload) == len(ALL_FAULTS)
    json.dumps(payload, default=str)


def test_unknown_fault_is_rejected(app):
    injector = FaultInjector(app.orchestrator.build_runtime())
    with pytest.raises(ValueError):
        injector.build("NOT_A_FAULT")


def test_injector_rejects_unknown_fault_type(app):
    from agentic_rag_recovery.models import FaultInjection
    from agentic_rag_recovery.taxonomy import Stage

    injector = FaultInjector(app.orchestrator.build_runtime())
    fault = FaultInjection(fault_type="NOT_A_FAULT", target_stage=Stage.RETRIEVAL)
    with pytest.raises(ValueError):
        injector.activate(fault, "q")


def test_ground_truth_is_recorded_not_leaked(app, dataset):
    """The diagnoser must never see the fault's ground truth."""
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        persist=False,
    )
    ground_truth = outcome.state.ground_truth
    assert ground_truth["expected_subtype"] == "hallucination"
    # the injected fault id / expected class must not appear in the signals
    signals = str(outcome.run.signals.model_dump())
    assert "expected_subtype" not in signals
    for diagnosis in outcome.run.diagnoses:
        assert "ground_truth" not in str(diagnosis.signals_used)


@pytest.mark.parametrize("fault_type", REQUIRED_FAULTS)
def test_every_fault_runs_end_to_end(app, dataset, fault_type):
    """Every declared fault must execute without crashing the request."""
    question = dataset.by_id("q001")
    outcome = app.orchestrator.answer(
        question.question,
        faults=[fault_type],
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
        persist=False,
    )
    assert outcome.run.run_id
    assert outcome.run.signals is not None
    assert outcome.state.active_faults
    if outcome.abstained:
        assert outcome.run.abstention.abstained


def test_release_restores_the_runtime(app, dataset):
    question = dataset.by_id("q001")
    runtime = app.orchestrator.build_runtime()
    injector = FaultInjector(runtime)
    before = runtime.retriever.size
    injector.activate(injector.build("RELEVANT_DOC_REMOVAL"), question.question)
    runtime.gold_document_ids = {
        c.document_id for c in runtime.retriever.chunks
    } & set(runtime.gold_document_ids)
    injector.release()
    assert runtime.retriever.size == before
    assert runtime.fault_flags == {}
    assert runtime.active_fault_ids == []
    assert runtime.knobs.ground_answer is True


def test_fault_severity_is_declared():
    for definition in FaultRegistry.list_all():
        assert definition.severity in set(Severity)


def test_fault_target_stage_is_declared():
    for definition in FaultRegistry.list_all():
        assert definition.target_stage is not None


def test_diagnosis_accuracy_of_generation_fault(app, dataset):
    """End-to-end: the declared ground truth matches the produced diagnosis."""
    from agentic_rag_recovery.evaluation.metrics import evaluate_run

    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
        persist=False,
    )
    record = evaluate_run(
        outcome.run,
        state=outcome.state,
        reference_answer=question.answer,
        answer_keywords=question.answer_keywords,
        ground_truth=outcome.state.ground_truth,
    )
    assert record.diagnosis_detected
    assert record.diagnosis_subtype_correct
    assert record.diagnosis_class_correct