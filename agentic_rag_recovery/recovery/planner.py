"""Recovery planner.

Given a diagnosis the planner produces an **ordered escalation ladder** where
each candidate fix is:

1. derived from the taxonomy mapping for the diagnosed failure class/subtype,
2. scored (relevance to the evidence, cost, blast radius, reusability),
3. filtered against *what has already been attempted* and *what budget remains*.

The planner never executes anything - :mod:`agentic_rag_recovery.recovery.executor`
does, and records the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..models import Diagnosis, SignalBundle
from ..settings import Settings, get_settings
from ..taxonomy import (
    FailureClass,
    FailureTaxonomy,
    RecoveryAction,
    Stage,
    recovery_ladder,
)


@dataclass
class PlannedAction:
    action: RecoveryAction
    rank: int = 0
    score: float = 0.0
    rationale: str = ""
    target_stage: Optional[Stage] = None
    cost: int = 1
    blocked_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "rank": self.rank,
            "score": round(self.score, 4),
            "rationale": self.rationale,
            "target_stage": self.target_stage.value if self.target_stage else None,
            "cost": self.cost,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class RecoveryPlan:
    diagnosis_class: FailureClass
    subtype: Optional[str]
    actions: List[PlannedAction] = field(default_factory=list)
    skipped: List[PlannedAction] = field(default_factory=list)
    must_abstain: bool = False
    abstain_reason: Optional[str] = None
    budget_remaining: Dict[str, Any] = field(default_factory=dict)

    def executable(self) -> List[PlannedAction]:
        return [a for a in self.actions if a.blocked_reason is None]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_class": self.diagnosis_class.value,
            "subtype": self.subtype,
            "actions": [a.to_dict() for a in self.actions],
            "skipped": [a.to_dict() for a in self.skipped],
            "must_abstain": self.must_abstain,
            "abstain_reason": self.abstain_reason,
            "budget_remaining": self.budget_remaining,
        }


#: blast radius: how many stages a recovery forces us to repeat
BLAST_RADIUS: Dict[RecoveryAction, Stage] = {
    RecoveryAction.QUERY_REWRITE: Stage.RETRIEVAL,
    RecoveryAction.HYBRID_SEARCH: Stage.RETRIEVAL,
    RecoveryAction.EXPAND_TOP_K: Stage.RETRIEVAL,
    RecoveryAction.BROADEN_RETRIEVAL: Stage.RETRIEVAL,
    RecoveryAction.DECOMPOSITION_REPAIR: Stage.QUERY_DECOMPOSITION,
    RecoveryAction.RERANK: Stage.RANKING,
    RecoveryAction.CHUNK_EXPANSION: Stage.CONTEXT_ASSEMBLY,
    RecoveryAction.CONTEXT_EXPANSION: Stage.CONTEXT_ASSEMBLY,
    RecoveryAction.CONSTRAINED_REGENERATION: Stage.GENERATION,
    RecoveryAction.BOUNDED_RETRY: Stage.RETRIEVAL,
    RecoveryAction.ALTERNATE_PATH: Stage.RETRIEVAL,
    RecoveryAction.TOOL_REPAIR: Stage.RANKING,
    RecoveryAction.ABSTAIN: None,
}

#: relative execution cost (LLM calls dominate)
ACTION_COST: Dict[RecoveryAction, int] = {
    RecoveryAction.QUERY_REWRITE: 2,
    RecoveryAction.DECOMPOSITION_REPAIR: 2,
    RecoveryAction.CONSTRAINED_REGENERATION: 2,
    RecoveryAction.HYBRID_SEARCH: 1,
    RecoveryAction.EXPAND_TOP_K: 1,
    RecoveryAction.RERANK: 1,
    RecoveryAction.CHUNK_EXPANSION: 1,
    RecoveryAction.CONTEXT_EXPANSION: 1,
    RecoveryAction.BROADEN_RETRIEVAL: 1,
    RecoveryAction.BOUNDED_RETRY: 1,
    RecoveryAction.ALTERNATE_PATH: 1,
    RecoveryAction.TOOL_REPAIR: 1,
    RecoveryAction.ABSTAIN: 0,
}


class RecoveryPlanner:
    """Builds and filters the recovery ladder."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def plan(
        self,
        diagnosis: Diagnosis,
        signals: Optional[SignalBundle] = None,
        attempted: Sequence[RecoveryAction] = (),
        budget: Optional[Dict[str, Any]] = None,
        active_state: Optional[Dict[str, Any]] = None,
    ) -> RecoveryPlan:
        failure_class = diagnosis.failure_class
        subtype = diagnosis.subtype if diagnosis.subtype not in {None, "none"} else None
        ladder = recovery_ladder(failure_class, subtype)
        plan = RecoveryPlan(diagnosis_class=failure_class, subtype=subtype, budget_remaining=dict(budget or {}))

        if failure_class is FailureClass.NONE or not diagnosis.failure_detected:
            plan.must_abstain = False
            return plan

        scores = self._scores(failure_class, subtype, signals, diagnosis)
        candidates: List[PlannedAction] = []
        for order, action in enumerate(ladder):
            definition = FailureTaxonomy.get(subtype) if subtype else None
            score = scores.get(action, 0.0) - 0.05 * order
            candidates.append(
                PlannedAction(
                    action=action,
                    score=score,
                    rank=order + 1,
                    rationale=self._rationale(action, failure_class, subtype),
                    target_stage=BLAST_RADIUS.get(action),
                    cost=(ACTION_COST.get(action, 1) or (definition.cost_hint if definition else 1)),
                )
            )

        candidates.sort(key=lambda p: -p.score)
        for index, planned in enumerate(candidates, start=1):
            planned.rank = index
            if planned.action in attempted:
                planned.blocked_reason = "BLOCK_REPEAT: action already attempted for this request"
                plan.skipped.append(planned)
                continue
            if planned.action is RecoveryAction.ABSTAIN:
                plan.actions.append(planned)
                continue
            noop = self._already_in_effect(planned.action, active_state or {})
            if noop:
                planned.blocked_reason = f"NO-OP: {noop}"
                plan.skipped.append(planned)
                continue
            if budget:
                exhausted = budget.get("exhausted") or {}
                if exhausted.get("attempts") and planned.cost > 0:
                    planned.blocked_reason = "budget exhausted: attempts"
                    plan.skipped.append(planned)
                    continue
                if exhausted.get("tokens") and planned.cost >= 2:
                    planned.blocked_reason = "budget exhausted: tokens"
                    plan.skipped.append(planned)
                    continue
            plan.actions.append(planned)

        executable = plan.executable()
        if executable and all(a.action is RecoveryAction.ABSTAIN for a in executable):
            plan.must_abstain = True
            plan.abstain_reason = "only ABSTAIN remains - a budget or repeat limit was reached"
            return plan
        if not executable:
            plan.must_abstain = True
            plan.abstain_reason = (
                "no un-tried, budget-affordable recovery remains for "
                f"{failure_class.value}{'/' + subtype if subtype else ''}"
            )
        return plan

    # ------------------------------------------------------------------
    @staticmethod
    def _already_in_effect(action: RecoveryAction, state: Dict[str, Any]) -> Optional[str]:
        """Return a reason string when the action could not change anything.

        Repeating a *successful-but-ineffective* configuration is exactly the
        failure mode the bounded loop exists to prevent, so the planner refuses
        to schedule it instead of burning an attempt.
        """
        if action is RecoveryAction.HYBRID_SEARCH and state.get("retrieval_method") == "hybrid":
            return "hybrid retrieval is already the active method"
        if action is RecoveryAction.RERANK and state.get("use_rerank") is True:
            return "a healthy reranker is already applied"
        if action is RecoveryAction.QUERY_REWRITE and state.get("rewrite_attempts", 0) >= 2:
            return "the query has already been rewritten twice"
        if action is RecoveryAction.EXPAND_TOP_K and int(state.get("top_k") or 0) >= int(
            state.get("max_top_k") or 10**6
        ):
            return "top_k already reached the retrieval ceiling"
        return None

    # ------------------------------------------------------------------
    def _scores(
        self,
        failure_class: FailureClass,
        subtype: Optional[str],
        signals: Optional[SignalBundle],
        diagnosis: Diagnosis,
    ) -> Dict[RecoveryAction, float]:
        """Heuristic relevance scores, derived from the collected signals."""
        scores: Dict[RecoveryAction, float] = {}
        default = 1.0

        def put(action: RecoveryAction, value: float) -> None:
            scores[action] = max(scores.get(action, 0.0), value)

        if diagnosis.recommended_recovery is not None:
            put(diagnosis.recommended_recovery, 2.0)

        if signals is None:
            return {a: scores.get(a, default) for a in BLAST_RADIUS}

        retrieval = signals.retrieval or {}
        context = signals.context or {}
        generation = signals.generation or {}
        ranking = signals.ranking or {}
        infra = signals.infrastructure or {}
        query = signals.query or {}

        if failure_class is FailureClass.RETRIEVAL_FAILURE:
            put(RecoveryAction.QUERY_REWRITE, 2.2)
            put(RecoveryAction.HYBRID_SEARCH, 2.0)
            put(RecoveryAction.EXPAND_TOP_K, 1.6)
            put(RecoveryAction.RERANK, 1.1)
            put(RecoveryAction.BROADEN_RETRIEVAL, 1.0)
            if float(retrieval.get("top_score", 1.0)) < 0.05:
                put(RecoveryAction.ALTERNATE_PATH, 0.8)
            if int(retrieval.get("duplicate_ratio", 0.0) * 10) >= 6:
                put(RecoveryAction.EXPAND_TOP_K, 1.9)
        elif failure_class is FailureClass.RANKING_FAILURE:
            put(RecoveryAction.RERANK, 2.3)
            put(RecoveryAction.EXPAND_TOP_K, 1.7)
            put(RecoveryAction.CHUNK_EXPANSION, 0.9)
        elif failure_class is FailureClass.CONTEXT_FAILURE:
            put(RecoveryAction.CHUNK_EXPANSION, 2.2)
            put(RecoveryAction.CONTEXT_EXPANSION, 2.0)
            if float(context.get("fragmentation_ratio", 0.0)) > 0.5:
                put(RecoveryAction.CHUNK_EXPANSION, 2.6)
            put(RecoveryAction.EXPAND_TOP_K, 1.2)
        elif failure_class is FailureClass.GENERATION_FAILURE:
            put(RecoveryAction.CONSTRAINED_REGENERATION, 2.5)
            put(RecoveryAction.CONTEXT_EXPANSION, 1.2)
        elif failure_class is FailureClass.QUERY_FAILURE:
            put(RecoveryAction.QUERY_REWRITE, 2.4)
            put(RecoveryAction.DECOMPOSITION_REPAIR, 1.9)
        elif failure_class is FailureClass.INFRASTRUCTURE_FAILURE:
            put(RecoveryAction.BOUNDED_RETRY, 2.3)
            put(RecoveryAction.ALTERNATE_PATH, 2.2)
            if int(infra.get("embedding_errors", 0)):
                put(RecoveryAction.ALTERNATE_PATH, 2.8)
        elif failure_class is FailureClass.AGENT_CONTROL_FAILURE:
            put(RecoveryAction.TOOL_REPAIR, 2.4)
            put(RecoveryAction.ALTERNATE_PATH, 1.4)
        elif failure_class is FailureClass.RESOURCE_FAILURE:
            put(RecoveryAction.ABSTAIN, 3.0)

        if query.get("is_ambiguous") or float(query.get("rewrite_drift", 0.0)) > 0.5:
            put(RecoveryAction.QUERY_REWRITE, 2.6)
        if float(generation.get("groundedness", 1.0)) < 0.5:
            put(RecoveryAction.CONSTRAINED_REGENERATION, 2.6)
        if float(ranking.get("rank_regression_rate", 0.0)) > 0.3:
            put(RecoveryAction.RERANK, 2.2)

        return {a: scores.get(a, default) for a in BLAST_RADIUS}

    @staticmethod
    def _rationale(action: RecoveryAction, failure_class: FailureClass, subtype: Optional[str]) -> str:
        definition = FailureTaxonomy.get(subtype) if subtype else None
        if definition and action in definition.recovery_actions:
            position = list(definition.recovery_actions).index(action)
            return (
                f"taxonomy step {position + 1} for {subtype}: {definition.description}"
            )
        return f"{action.value} is the next escalation step for {failure_class.value}"