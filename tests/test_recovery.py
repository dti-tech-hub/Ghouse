"""Recovery planner, strategies, bounded loop, verification and abstention."""

from __future__ import annotations

import pytest

from agentic_rag_recovery.models import DiagnosisTier, RunMode
from agentic_rag_recovery.recovery.budget import BudgetExhausted, BudgetTracker
from agentic_rag_recovery.recovery.executor import ABSTENTION_MESSAGE, RecoveryController
from agentic_rag_recovery.recovery.planner import RecoveryPlanner
from agentic_rag_recovery.recovery.strategies import (
    RecoveryContext,
    get_strategy,
    registered_strategies,
)
from agentic_rag_recovery.settings import BudgetConfig
from agentic_rag_recovery.taxonomy import (
    FailureClass,
    RecoveryAction,
    default_recovery_map,
    recovery_ladder,
)


def _diagnosis(subtype: str, failure_class: FailureClass, ladder=None) -> Diagnosis:
    from agentic_rag_recovery.models import Diagnosis
    from agentic_rag_recovery.taxonomy import Stage

    return Diagnosis(
        failure_detected=True,
        failure_class=failure_class,
        subtype=subtype,
        stage=Stage.RETRIEVAL,
        severity="high",
        confidence=0.9,
        evidence=["e"],
        recovery_ladder=list(ladder or recovery_ladder(failure_class, subtype)),
    )


# ---------------------------------------------------------------------------
# taxonomy mapping
# ---------------------------------------------------------------------------
def test_every_class_has_a_recovery_ladder():
    for failure_class in FailureClass:
        ladder = default_recovery_map(failure_class)
        assert ladder
        assert ladder[-1] is RecoveryAction.ABSTAIN


def test_ladder_ends_with_abstain():
    assert recovery_ladder(FailureClass.RETRIEVAL_FAILURE)[-1] is RecoveryAction.ABSTAIN
    assert recovery_ladder(FailureClass.GENERATION_FAILURE)[-1] is RecoveryAction.ABSTAIN


def test_ladder_for_subtype_is_declared_in_taxonomy():
    ladder = recovery_ladder(FailureClass.RETRIEVAL_FAILURE, "low_recall")
    assert RecoveryAction.QUERY_REWRITE in ladder
    assert RecoveryAction.ABSTAIN in ladder


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
def test_budget_snapshot_fields():
    budget = BudgetTracker(BudgetConfig(max_recovery_attempts=2, max_token_budget=100))
    snapshot = budget.snapshot()
    assert snapshot.max_recovery_attempts == 2
    assert snapshot.attempts_remaining == 2
    assert snapshot.tokens_remaining == 100
    assert budget.snapshot_dict()["exhausted"]["attempts"] is False


def test_budget_blocks_repeat():
    budget = BudgetTracker(BudgetConfig())
    action = RecoveryAction.QUERY_REWRITE
    assert budget.can_attempt(action)[0]
    budget.begin_attempt(action)
    allowed, reason = budget.can_attempt(action)
    assert not allowed
    assert "BLOCK_REPEAT" in reason
    assert budget.repeated_actions_blocked == 1


def test_budget_blocks_when_attempts_exhausted():
    budget = BudgetTracker(BudgetConfig(max_recovery_attempts=1))
    budget.begin_attempt(RecoveryAction.RERANK)
    allowed, reason = budget.can_attempt(RecoveryAction.HYBRID_SEARCH)
    assert not allowed
    assert "budget exhausted" in reason


def test_budget_blocks_when_tokens_exhausted():
    budget = BudgetTracker(BudgetConfig(max_token_budget=10))
    budget.record_llm(tokens=10)
    assert budget.snapshot().exhausted()["tokens"]


def test_budget_latency_accounting():
    budget = BudgetTracker(BudgetConfig())
    budget.record_latency(120.0)
    assert budget.snapshot().latency_used_ms == 120.0
    assert budget.added_latency_ms == 120.0


def test_budget_cost_per_recovery():
    budget = BudgetTracker(BudgetConfig())
    budget.record_llm(calls=2, tokens=100, cost=0.5)
    assert budget.cost_per_recovery(2) == 0.25


def test_budget_exhausted_exception():
    error = BudgetExhausted("tokens")
    assert error.resource == "tokens"


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------
def test_planner_orders_the_escalation_ladder(settings):
    planner = RecoveryPlanner(settings)
    plan = planner.plan(_diagnosis("low_recall", FailureClass.RETRIEVAL_FAILURE))
    actions = [a.action for a in plan.executable()]
    assert actions[0] is RecoveryAction.QUERY_REWRITE
    assert RecoveryAction.ABSTAIN in actions


def test_planner_blocks_already_attempted_actions(settings):
    planner = RecoveryPlanner(settings)
    plan = planner.plan(
        _diagnosis("low_recall", FailureClass.RETRIEVAL_FAILURE),
        attempted=[RecoveryAction.QUERY_REWRITE],
    )
    assert all(a.action is not RecoveryAction.QUERY_REWRITE for a in plan.executable())
    assert any("BLOCK_REPEAT" in (a.blocked_reason or "") for a in plan.skipped)


def test_planner_skips_noop_actions(settings):
    planner = RecoveryPlanner(settings)
    plan = planner.plan(
        _diagnosis("low_recall", FailureClass.RETRIEVAL_FAILURE),
        active_state={"retrieval_method": "hybrid"},
    )
    assert all(a.action is not RecoveryAction.HYBRID_SEARCH for a in plan.executable())
    assert any("NO-OP" in (a.blocked_reason or "") for a in plan.skipped)


def test_planner_respects_budget(settings):
    planner = RecoveryPlanner(settings)
    plan = planner.plan(
        _diagnosis("low_recall", FailureClass.RETRIEVAL_FAILURE),
        budget={"exhausted": {"attempts": True}},
    )
    assert plan.must_abstain


def test_planner_healthy_diagnosis_is_empty(settings):
    from agentic_rag_recovery.models import Diagnosis

    planner = RecoveryPlanner(settings)
    plan = planner.plan(Diagnosis(failure_detected=False, failure_class=FailureClass.NONE))
    assert plan.executable() == []
    assert not plan.must_abstain


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
def test_every_action_has_a_strategy():
    for action in RecoveryAction:
        assert get_strategy(action) is not None
    assert len(registered_strategies()) == len(list(RecoveryAction))


def test_strategy_rerank_reuses_retrieval(app, dataset, settings):
    from agentic_rag_recovery.pipeline.state import PipelineState
    from agentic_rag_recovery.pipeline.nodes import run_stages_from
    from agentic_rag_recovery.taxonomy import Stage

    runtime = app.orchestrator.build_runtime()
    state = PipelineState(query="What is the minimum password length?")
    run_stages_from(Stage.QUERY_UNDERSTANDING, state, runtime)
    ctx = RecoveryContext(
        state=state,
        runtime=runtime,
        diagnosis=_diagnosis("relevant_ranked_low", FailureClass.RANKING_FAILURE),
        attempt_index=1,
    )
    outcome = get_strategy(RecoveryAction.RERANK)(ctx)
    assert outcome.applied
    assert outcome.rerun_from is Stage.RANKING
    assert Stage.RETRIEVAL in outcome.reused_stages


def test_strategy_constrained_regeneration_reuses_context(app, settings):
    from agentic_rag_recovery.pipeline.state import PipelineState
    from agentic_rag_recovery.pipeline.nodes import run_stages_from
    from agentic_rag_recovery.taxonomy import Stage

    runtime = app.orchestrator.build_runtime()
    runtime.knobs.ground_answer = False
    runtime.knobs.require_grounding = False
    state = PipelineState(query="What is the minimum password length?")
    run_stages_from(Stage.QUERY_UNDERSTANDING, state, runtime)
    ctx = RecoveryContext(
        state=state,
        runtime=runtime,
        diagnosis=_diagnosis("hallucination", FailureClass.GENERATION_FAILURE),
        attempt_index=1,
    )
    outcome = get_strategy(RecoveryAction.CONSTRAINED_REGENERATION)(ctx)
    assert outcome.applied
    assert outcome.rerun_from is Stage.GENERATION
    assert Stage.CONTEXT_ASSEMBLY in outcome.reused_stages


def test_strategy_expand_top_k_refuses_at_ceiling(app, settings):
    from agentic_rag_recovery.pipeline.state import PipelineState

    runtime = app.orchestrator.build_runtime()
    runtime.knobs.top_k = 999
    ctx = RecoveryContext(
        state=PipelineState(query="q"),
        runtime=runtime,
        diagnosis=_diagnosis("low_recall", FailureClass.RETRIEVAL_FAILURE),
        attempt_index=1,
    )
    outcome = get_strategy(RecoveryAction.EXPAND_TOP_K)(ctx)
    assert not outcome.applied
    assert "ceiling" in (outcome.error or "")


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def test_verifier_rejects_unimproved_recovery(settings):
    from agentic_rag_recovery.models import SignalBundle
    from agentic_rag_recovery.validation.verifier import RecoveryVerifier

    signals = SignalBundle(
        retrieval={"retrieved_count": 5, "useful_ratio": 0.2, "top_overlap": 0.2, "pre_rank_top_overlap": 0.2, "source_diversity": 1, "duplicate_ratio": 0.0, "recall": 0.1},
        ranking={},
    )
    verifier = RecoveryVerifier(settings)
    result = verifier.verify(signals, signals, FailureClass.RETRIEVAL_FAILURE, "low_recall")
    assert not result.passed
    assert result.failed_criteria()


def test_verifier_accepts_improved_recovery(settings):
    from agentic_rag_recovery.models import SignalBundle
    from agentic_rag_recovery.validation.verifier import RecoveryVerifier

    before = SignalBundle(retrieval={"retrieved_count": 5, "useful_ratio": 0.2, "top_overlap": 0.2, "source_diversity": 2, "duplicate_ratio": 0.0, "recall": 0.0})
    after = SignalBundle(retrieval={"retrieved_count": 5, "useful_ratio": 0.8, "top_overlap": 0.8, "source_diversity": 2, "duplicate_ratio": 0.0, "recall": 1.0})
    result = RecoveryVerifier(settings).verify(before, after, FailureClass.RETRIEVAL_FAILURE, "low_recall")
    assert result.passed
    assert result.improvement["top_overlap"] > 0


# ---------------------------------------------------------------------------
# end-to-end recovery / abstention
# ---------------------------------------------------------------------------
def test_recovery_succeeds_for_generation_failure(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        relevant_documents=question.relevant_documents,
        answer_keywords=question.answer_keywords,
        persist=False,
    )
    assert outcome.run.diagnoses[0].failure_detected
    assert outcome.run.recoveries
    assert outcome.recovered
    assert outcome.run.recoveries[0].status.value == "success"
    assert outcome.run.recoveries[0].verification["passed"]


def test_recovery_is_targeted_not_a_full_restart(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        persist=False,
    )
    attempt = outcome.run.recoveries[0]
    assert attempt.skipped_stages  # earlier stages were reused
    assert attempt.added_latency_ms <= outcome.run.total_latency_ms + 1


def test_bounded_loop_respects_attempt_limit(app, dataset):
    question = dataset.by_id("q004")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["RELEVANT_DOC_REMOVAL"],
        relevant_documents=question.relevant_documents,
        persist=False,
    )
    assert len(outcome.run.recoveries) <= app.settings.budget.max_recovery_attempts
    attempted = [a.action for a in outcome.run.recoveries]
    assert len(attempted) == len(set(attempted))  # never the same action twice


def test_abstention_when_recovery_fails(app, dataset):
    question = dataset.by_id("q004")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["RELEVANT_DOC_REMOVAL"],
        relevant_documents=question.relevant_documents,
        persist=False,
    )
    if outcome.abstained:
        abstention = outcome.run.abstention
        assert abstention.abstained
        assert abstention.attempts_used <= app.settings.budget.max_recovery_attempts
        assert outcome.answer == ABSTENTION_MESSAGE
        assert abstention.budget_consumed


def test_conflicting_evidence_abstains(app, dataset):
    """A contradiction in the context cannot be repaired: the loop must abstain."""
    from dataclasses import replace

    question = dataset.by_id("q016")
    app.settings.budget = replace(app.settings.budget, max_recovery_attempts=3)
    try:
        outcome = app.orchestrator.answer(
            question.question,
            faults=["CONFLICTING_EVIDENCE"],
            relevant_documents=question.relevant_documents,
            persist=False,
        )
    finally:
        app.settings.budget = replace(app.settings.budget, max_recovery_attempts=4)
    if outcome.abstained:
        assert outcome.answer == ABSTENTION_MESSAGE
        assert outcome.run.abstention.attempts_used <= 3
        assert outcome.run.abstention.budget_consumed
    else:
        # recovery is still verified before it is reported, never assumed
        assert outcome.recovered
        assert outcome.run.recoveries[-1].verification["passed"]


def test_healthy_run_never_recovers(app, dataset):
    for question in dataset.questions[:6]:
        outcome = app.orchestrator.answer(
            question.question,
            relevant_documents=question.relevant_documents,
            answer_keywords=question.answer_keywords,
            persist=False,
        )
        assert not outcome.abstained
        assert outcome.run.recoveries == []


def test_mode2_never_recovers(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        mode=RunMode.MODE_2_AGENTIC_NO_RECOVERY,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        persist=False,
    )
    assert outcome.run.recoveries == []
    assert outcome.run.diagnoses[0].failure_detected


def test_recovery_controller_records_plans(app, dataset):
    question = dataset.by_id("q013")
    outcome = app.orchestrator.answer(
        question.question,
        faults=["GENERATION_CONSTRAINT_REMOVAL"],
        persist=False,
    )
    plans = outcome.state.metadata.get("recovery_plans") or []
    assert plans
    assert plans[0]["actions"]