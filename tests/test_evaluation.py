"""Evaluation metrics, datasets and the experiment engine."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.evaluation.datasets import (
    load_demo_dataset,
    list_public_datasets,
    load_public_dataset,
)
from agentic_rag_recovery.evaluation.metrics import (
    RunMetrics,
    aggregate,
    class_distribution,
    compare_modes,
    diagnosis_correct,
    evaluate_run,
    severity_distribution,
    stage_distribution,
)
from agentic_rag_recovery.evaluation.runner import ExperimentRunner
from agentic_rag_recovery.models import Diagnosis, FailureClass, RunMode, Severity, Stage


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def test_demo_dataset_loads(settings):
    dataset = load_demo_dataset(settings)
    assert len(dataset) >= 20
    first = dataset.questions[0]
    assert first.question
    assert first.answer
    assert first.answer_keywords


def test_demo_dataset_categories(settings):
    dataset = load_demo_dataset(settings)
    summary = dataset.summary()
    assert summary["total"] == len(dataset)
    assert "retrieval" in summary["categories"]


def test_demo_dataset_lookup(settings):
    dataset = load_demo_dataset(settings)
    question = dataset.by_id("q001")
    assert question is not None
    assert dataset.by_id("does-not-exist") is None


def test_public_datasets_reported_as_unavailable(settings):
    listing = list_public_datasets(settings)
    assert {d["name"] for d in listing} >= {"natural_questions", "hotpotqa", "ms_marco"}


def test_public_dataset_missing_raises(settings):
    with pytest.raises(FileNotFoundError):
        load_public_dataset("natural_questions", settings=settings)


# ---------------------------------------------------------------------------
# diagnosis scoring
# ---------------------------------------------------------------------------
def test_diagnosis_correct_strict_and_class():
    diagnosis = Diagnosis(
        failure_detected=True,
        failure_class=FailureClass.RETRIEVAL_FAILURE,
        subtype="low_recall",
        stage=Stage.RETRIEVAL,
        severity=Severity.HIGH,
        confidence=0.8,
        evidence=["e"],
    )
    ground_truth = {"expected_subtype": "low_recall", "expected_failure_class": "RETRIEVAL_FAILURE"}
    assert diagnosis_correct(diagnosis, ground_truth) == (True, True, True)

    ground_truth = {"expected_subtype": "no_useful_evidence", "expected_failure_class": "RETRIEVAL_FAILURE"}
    subtype_ok, class_ok, acceptable = diagnosis_correct(diagnosis, ground_truth)
    assert not subtype_ok
    assert class_ok
    assert acceptable


def test_diagnosis_correct_without_ground_truth():
    assert diagnosis_correct(Diagnosis(), {}) == (False, False, False)
    assert diagnosis_correct(None, {"expected_subtype": "x"}) == (False, False, False)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _record(**kwargs) -> RunMetrics:
    defaults = dict(
        run_id="run_x",
        mode=RunMode.MODE_3_AGENTIC_WITH_RECOVERY.value,
        status="success",
        query="q",
        groundedness=1.0,
        citation_coverage=0.6,
        context_coverage=0.9,
        recovery_attempts=2,
        successful_recoveries=1,
        recovery_success=True,
        diagnosis_detected=True,
        diagnosis_subtype_correct=True,
        diagnosis_class_correct=True,
        diagnosis_acceptable=True,
        diagnosis_tier="rule",
        added_latency_ms=20.0,
        cost=0.01,
        faults=["LOW_TOP_K"],
    )
    defaults.update(kwargs)
    return RunMetrics(**defaults)


def test_aggregate_headline_metrics():
    records = [
        _record(run_id="a", cost=0.01),
        _record(run_id="b", cost=0.03, successful_recoveries=2, recovery_attempts=3, added_latency_ms=40.0),
    ]
    metrics = aggregate(records, experiment_id="exp", mode="mode3", label="TEST")
    assert metrics.total_runs == 2
    assert metrics.diagnosable_failures == 2
    assert metrics.diagnosis_accuracy == 1.0
    assert metrics.recovery_success_rate == pytest.approx(3 / 5)
    assert metrics.groundedness == 1.0
    assert metrics.abstention_rate == 0.0
    assert metrics.added_latency_ms == pytest.approx(30.0)
    assert metrics.cost_per_recovery == pytest.approx(0.04 / 3, rel=1e-3)
    assert metrics.label == "TEST"


def test_aggregate_counts_abstentions():
    metrics = aggregate([_record(abstained=True), _record(abstained=False, run_id="b")])
    assert metrics.abstentions == 1
    assert metrics.abstention_rate == 0.5


def test_aggregate_empty_is_safe():
    metrics = aggregate([])
    assert metrics.total_runs == 0
    assert metrics.diagnosis_accuracy == 0.0


def test_compare_modes_reports_only_executed_numbers():
    m1 = aggregate([_record(mode="mode1_normal_rag", added_latency_ms=0.0)])
    m2 = aggregate([_record(mode="mode2_agentic_no_recovery", added_latency_ms=0.0)])
    m3 = aggregate([_record(mode="mode3_agentic_with_recovery", added_latency_ms=20.0)])
    comparison = compare_modes(m1, m2, m3)
    assert set(comparison) >= {
        "mode1_normal_rag",
        "mode2_agentic_no_recovery",
        "mode3_agentic_with_recovery",
        "mode3_minus_mode1",
    }
    assert comparison["mode3_minus_mode1"]["added_latency_ms"] == pytest.approx(20.0)


def test_distributions():
    diagnoses = [
        Diagnosis(failure_detected=True, failure_class=FailureClass.RETRIEVAL_FAILURE, subtype="low_recall", stage=Stage.RETRIEVAL, severity=Severity.HIGH, confidence=0.8, evidence=["e"]),
        Diagnosis(failure_detected=True, failure_class=FailureClass.GENERATION_FAILURE, subtype="hallucination", stage=Stage.GENERATION, severity=Severity.CRITICAL, confidence=0.9, evidence=["e"]),
        Diagnosis(failure_detected=False),
    ]
    assert class_distribution(diagnoses) == {"RETRIEVAL_FAILURE": 1, "GENERATION_FAILURE": 1}
    assert stage_distribution(diagnoses) == {"retrieval": 1, "generation": 1}
    assert severity_distribution(diagnoses) == {"high": 1, "critical": 1}


# ---------------------------------------------------------------------------
# run evaluation
# ---------------------------------------------------------------------------
def test_evaluate_run(app, dataset):
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
    assert record.run_id == outcome.run.run_id
    assert record.groundedness >= 0.0
    assert record.faults == ["GENERATION_CONSTRAINT_REMOVAL"]
    assert record.diagnosis_detected
    assert 0.0 <= record.answer_f1 <= 1.0
    assert record.added_latency_ms >= 0.0


# ---------------------------------------------------------------------------
# experiment engine
# ---------------------------------------------------------------------------
def test_experiment_configs_are_discoverable(app):
    runner = ExperimentRunner(app)
    configs = runner.available_configs()
    assert "baseline" in configs
    for name in ("baseline", "retrieval_failure", "ranking_failure", "context_failure", "generation_failure", "full_recovery"):
        assert name in configs


def test_every_config_has_a_seed(app):
    runner = ExperimentRunner(app)
    for name in runner.available_configs():
        config = runner.load_config(name)
        assert "seed" in config
        assert config["seed"] >= 0


def test_run_baseline_experiment(app):
    runner = ExperimentRunner(app)
    config = {
        "name": "unit-baseline",
        "seed": 42,
        "repetitions": 1,
        "question_ids": ["q001", "q008"],
        "modes": ["mode1_normal_rag", "mode2_agentic_no_recovery", "mode3_agentic_with_recovery"],
    }
    result = runner.run(config, persist=False)
    assert result.mode1.total_runs == 2
    assert result.mode3.total_runs == 2
    assert result.comparison
    assert result.duration_s >= 0


def test_run_fault_experiment(app):
    runner = ExperimentRunner(app)
    config = {
        "name": "unit-generation",
        "seed": 7,
        "repetitions": 1,
        "question_ids": ["q013"],
        "fault_type": "GENERATION_CONSTRAINT_REMOVAL",
        "modes": ["mode3_agentic_with_recovery"],
    }
    result = runner.run(config, persist=False)
    assert result.mode3.total_runs == 1
    assert result.mode3.diagnosable_failures == 1
    assert result.mode3.diagnosis_accuracy in (0.0, 1.0)
    assert result.label


def test_experiment_is_reproducible(app):
    runner = ExperimentRunner(app)
    config = {
        "name": "unit-repro",
        "seed": 123,
        "repetitions": 2,
        "question_ids": ["q001", "q016"],
        "fault_type": "GENERATION_CONSTRAINT_REMOVAL",
        "modes": ["mode3_agentic_with_recovery"],
    }
    first = runner.run(config, persist=False)
    second = runner.run(config, persist=False)
    assert first.mode3.total_runs == second.mode3.total_runs == 4
    assert first.mode3.diagnosis_accuracy == second.mode3.diagnosis_accuracy


def test_experiment_fault_sweep(app):
    runner = ExperimentRunner(app)
    config = {
        "name": "unit-sweep",
        "seed": 5,
        "repetitions": 1,
        "question_ids": ["q001"],
        "fault_sweep": ["LOW_TOP_K", "RANKING_CORRUPTION"],
        "modes": ["mode3_agentic_with_recovery"],
    }
    result = runner.run(config, persist=False)
    assert result.mode3.total_runs == 2
    assert set(result.mode3.failure_count_by_class) >= {"LOW_TOP_K", "RANKING_CORRUPTION"}


def test_experiment_persists_everything(app):
    runner = ExperimentRunner(app)
    config = {
        "name": "unit-persist",
        "seed": 11,
        "repetitions": 1,
        "question_ids": ["q001"],
        "fault_type": "GENERATION_CONSTRAINT_REMOVAL",
        "modes": ["mode3_agentic_with_recovery"],
    }
    result = runner.run(config, persist=True)
    stored = app.repository.get_experiment(result.experiment_id)
    assert stored["status"] == "completed"
    assert stored["seed"] == 11
    metrics = app.repository.get_metrics(experiment_id=result.experiment_id)
    assert metrics
    runs = app.repository.list_runs(experiment_id=result.experiment_id)
    assert runs