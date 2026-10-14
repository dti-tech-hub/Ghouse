"""Bounded recovery control loop.

    for attempt in 1..MAX:
        plan   = planner(diagnosis, attempted=..., budget=...)
        action = next executable action          # never repeats, never exceeds budget
        apply(action)                            # targeted: re-runs one stage onward
        verify()                                 # never assume success
        if verified: stop
    abstain()

Every attempt is recorded with its before/after state, its latency, token and
cost cost, and the verification outcome.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import (
    Abstention,
    BudgetSnapshot,
    Diagnosis,
    PipelineRun,
    RecoveryAttempt,
    RecoveryOutcome,
    RunStatus,
)
from ..observability.trace import get_logger
from ..pipeline.nodes import run_stages_from
from ..pipeline.runtime import Runtime
from ..pipeline.state import PipelineState
from ..settings import Settings, get_settings
from ..taxonomy import PIPELINE_STAGES, FailureClass, RecoveryAction, Stage
from ..validation.verifier import RecoveryVerifier, VerificationResult
from .budget import BudgetTracker
from .planner import RecoveryPlan, RecoveryPlanner
from .strategies import RecoveryContext, StrategyOutcome, get_strategy

logger = get_logger(__name__)

ABSTENTION_MESSAGE = (
    "I could not obtain sufficient reliable evidence to answer this question."
)


@dataclass
class RecoveryOutcomeSummary:
    recovered: bool = False
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    successful_attempts: int = 0
    added_latency_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    abstention: Optional[Abstention] = None
    budget: Optional[BudgetSnapshot] = None
    plans: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""


class RecoveryController:
    """Executes the bounded recovery loop for one request."""

    def __init__(
        self,
        runtime: Runtime,
        verifier: Optional[RecoveryVerifier] = None,
        planner: Optional[RecoveryPlanner] = None,
        settings: Optional[Settings] = None,
    ):
        self.runtime = runtime
        self.settings = settings or runtime.settings
        self.verifier = verifier or runtime.verifier
        self.planner = planner or RecoveryPlanner(self.settings)

    # ------------------------------------------------------------------
    def run(
        self,
        state: PipelineState,
        diagnosis: Diagnosis,
        budget: BudgetTracker,
    ) -> RecoveryOutcomeSummary:
        summary = RecoveryOutcomeSummary()
        signals_before = state.signals
        target = diagnosis.subtype or diagnosis.failure_class.value

        if diagnosis.failure_class is FailureClass.NONE or not diagnosis.failure_detected:
            summary.stop_reason = "no failure detected - no recovery needed"
            return summary

        attempts: List[RecoveryAttempt] = []
        for index in range(1, self.settings.budget.max_recovery_attempts + 1):
            plan = self.planner.plan(
                diagnosis,
                signals=state.signals,
                attempted=budget.attempted,
                budget=budget.snapshot_dict(),
                active_state=self._active_state(state, self.runtime),
            )
            summary.plans.append(plan.to_dict())
            executable = plan.executable()

            if plan.must_abstain or not executable:
                summary.stop_reason = plan.abstain_reason or "no executable recovery left"
                break

            planned = executable[0]
            action = planned.action
            if action is RecoveryAction.ABSTAIN:
                summary.stop_reason = "planner selected terminal ABSTAIN"
                break

            allowed, blocked_reason = budget.can_attempt(action)
            if not allowed:
                attempt = RecoveryAttempt(
                    run_id=state.run_id,
                    attempt_index=index,
                    action=action,
                    target_stage=planned.target_stage or Stage.RETRIEVAL,
                    diagnosis_subtype=diagnosis.subtype,
                    status=RecoveryOutcome.SKIPPED,
                    rationale=planned.rationale,
                    error=blocked_reason,
                )
                attempts.append(attempt)
                summary.attempts = attempts
                logger.info("recovery blocked: %s (%s)", action.value, blocked_reason)
                if blocked_reason and blocked_reason.startswith("budget"):
                    summary.stop_reason = blocked_reason
                    break
                continue

            attempt = self._execute(state, diagnosis, budget, plan, index)
            attempts.append(attempt)
            summary.attempts = attempts
            budget.record_latency(attempt.latency_ms)
            budget.record_llm(attempt.llm_calls, attempt.tokens_used, attempt.cost)

            if attempt.status is RecoveryOutcome.SUCCESS:
                summary.recovered = True
                summary.successful_attempts = 1
                summary.stop_reason = f"verified recovery via {action.value}"
                break
            if attempt.status is RecoveryOutcome.ERROR:
                logger.warning("recovery attempt errored: %s", attempt.error)
        else:
            summary.stop_reason = "maximum recovery attempts exhausted"

        if not summary.recovered and not summary.stop_reason:
            summary.stop_reason = "no verified recovery achieved"

        summary.added_latency_ms = budget.added_latency_ms
        summary.tokens_used = budget.tokens_used
        summary.llm_calls = budget.llm_calls_used
        summary.cost = budget.cost
        summary.budget = budget.snapshot()

        if not summary.recovered:
            summary.abstention = self._abstain(state, diagnosis, budget, attempts)
        return summary

    # ------------------------------------------------------------------
    @staticmethod
    def _active_state(state: PipelineState, runtime: Runtime) -> Dict[str, Any]:
        """Current configuration, used by the planner to avoid no-op actions."""
        settings = runtime.settings
        return {
            "retrieval_method": runtime.knobs.retrieval_method or settings.retrieval.method,
            "use_rerank": runtime.knobs.use_rerank if runtime.knobs.use_rerank is not None else settings.retrieval.rerank_enabled,
            "top_k": runtime.knobs.top_k or settings.retrieval.top_k,
            "max_top_k": max(settings.retrieval.candidate_k, settings.retrieval.top_k * 3),
            "rewrite_attempts": len(state.metadata.get("rewrites", [])),
            "has_rewrite": bool(state.understanding and state.understanding.rewrite),
            "max_chunks": runtime.knobs.max_chunks,
            "max_context_tokens": runtime.knobs.max_context_tokens,
            "ground_answer": runtime.knobs.ground_answer and runtime.knobs.require_grounding,
        }

    # ------------------------------------------------------------------
    def _execute(
        self,
        state: PipelineState,
        diagnosis: Diagnosis,
        budget: BudgetTracker,
        plan: RecoveryPlan,
        index: int,
    ) -> RecoveryAttempt:
        action = plan.executable()[0].action
        rationale = plan.executable()[0].rationale
        before_state = state.snapshot_for_recovery()
        signals_before = state.signals
        attempts_before = len(state.recoveries)
        llm_before = state.llm_calls
        tokens_before = state.tokens_used
        cost_before = state.cost
        stage_before = state.total_latency_ms

        strategy_fn = get_strategy(action)
        attempt = RecoveryAttempt(
            run_id=state.run_id,
            attempt_index=index,
            action=action,
            target_stage=(plan.executable()[0].target_stage or Stage.RETRIEVAL),
            diagnosis_subtype=diagnosis.subtype,
            rationale=rationale,
            before=before_state,
        )

        if strategy_fn is None:
            attempt.status = RecoveryOutcome.SKIPPED
            attempt.error = f"no strategy implemented for {action.value}"
            state.recoveries.append(attempt)
            return attempt

        started = time.perf_counter()
        budget.begin_attempt(action)
        ctx = RecoveryContext(
            state=state,
            runtime=self.runtime,
            diagnosis=diagnosis,
            attempt_index=index,
            failed_stages=[r.stage for r in state.stage_results if not r.ok],
            history=[a.action.value for a in state.recoveries],
        )
        try:
            outcome: StrategyOutcome = strategy_fn(ctx)
        except Exception as exc:  # a broken strategy must never kill the request
            attempt.status = RecoveryOutcome.ERROR
            attempt.error = f"{type(exc).__name__}: {exc}"
            attempt.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            state.recoveries.append(attempt)
            self.runtime.trace.step(
                action.value, "strategy_error", status="error", error=attempt.error,
                latency_ms=attempt.latency_ms,
            )
            return attempt

        if not outcome.applied:
            attempt.status = RecoveryOutcome.SKIPPED
            attempt.error = outcome.error or "strategy declined to apply"
            attempt.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            state.recoveries.append(attempt)
            self.runtime.trace.step(
                action.value, "strategy_declined", status="skipped",
                output_summary=attempt.error, latency_ms=attempt.latency_ms,
            )
            return attempt

        attempt.reused_stage = bool(outcome.reused_stages)
        attempt.skipped_stages = [s.value for s in outcome.reused_stages]

        # --- targeted re-run: only stages from ``rerun_from`` onward --------
        if outcome.rerun_from is not None:
            try:
                run_stages_from(outcome.rerun_from, state, self.runtime)
            except Exception as exc:
                attempt.status = RecoveryOutcome.ERROR
                attempt.error = f"re-run from {outcome.rerun_from.value} failed: {exc}"
                attempt.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
                state.recoveries.append(attempt)
                return attempt

        state.signals = state.signals  # refreshed by stage_validation

        # --- verify (never assume success) -------------------------------
        verification: VerificationResult = self.verifier.verify(
            signals_before if signals_before is not None else state.signals,
            state.signals,
            diagnosis.failure_class,
            diagnosis.subtype,
        )
        attempt.verification = verification.to_dict()
        attempt.after = state.snapshot_for_recovery()
        attempt.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        attempt.added_latency_ms = round(max(0.0, state.total_latency_ms - stage_before), 3)
        attempt.llm_calls = max(0, state.llm_calls - llm_before)
        attempt.tokens_used = max(0, state.tokens_used - tokens_before)
        attempt.cost = round(max(0.0, state.cost - cost_before), 8)
        attempt.status = RecoveryOutcome.SUCCESS if verification.passed else RecoveryOutcome.FAILURE

        state.recoveries.append(attempt)
        self.runtime.trace.step(
            action.value,
            "recovery_attempt",
            input_summary=clip_summary(before_state),
            output_summary=clip_summary(attempt.after),
            status="ok" if verification.passed else "recovered_not_verified",
            latency_ms=attempt.latency_ms,
            rerun_from=outcome.rerun_from.value if outcome.rerun_from else None,
            reused_stages=attempt.skipped_stages,
            verification_passed=verification.passed,
            failed_criteria=verification.failed_criteria(),
        )
        logger.info(
            "recovery attempt %d/%d %s -> %s (%s)",
            index,
            self.settings.budget.max_recovery_attempts,
            action.value,
            attempt.status.value,
            verification.summary,
        )
        return attempt

    # ------------------------------------------------------------------
    def _abstain(
        self,
        state: PipelineState,
        diagnosis: Diagnosis,
        budget: BudgetTracker,
        attempts: List[RecoveryAttempt],
    ) -> Abstention:
        diagnosis = state.diagnoses[-1] if state.diagnoses else diagnosis
        exhausted = budget.snapshot().exhausted()
        reason = (
            "no verified recovery was achieved; the remaining evidence cannot support a "
            "grounded answer"
        )
        if any(exhausted.values()):
            exhausted_names = [name for name, flag in exhausted.items() if flag]
            reason = f"recovery stopped: budget exhausted ({', '.join(exhausted_names)})"
        return Abstention(
            abstained=True,
            reason=reason,
            failure_class=diagnosis.failure_class,
            stage=diagnosis.stage,
            subtype=diagnosis.subtype,
            attempted_recoveries=[a.action for a in attempts],
            attempts_used=budget.attempts_used,
            max_attempts=self.settings.budget.max_recovery_attempts,
            budget_consumed=budget.snapshot_dict(),
            message=ABSTENTION_MESSAGE,
        )


def clip_summary(payload: Dict[str, Any], limit: int = 220) -> str:
    import json

    return json.dumps(payload, default=str)[:limit]


__all__ = ["RecoveryController", "RecoveryOutcomeSummary", "ABSTENTION_MESSAGE"]