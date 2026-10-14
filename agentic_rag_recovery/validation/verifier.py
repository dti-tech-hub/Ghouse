"""Recovery verification.

A recovery attempt is **never** assumed to have worked.  After every attempt
the pipeline re-collects signals and this module decides - against the criteria
that belong to the diagnosed failure class - whether the recovery succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import SignalBundle
from ..settings import Settings, get_settings
from ..taxonomy import FailureClass, Stage


@dataclass
class VerificationCriterion:
    name: str
    passed: bool
    observed: Any = None
    required: Any = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "required": self.required,
            "detail": self.detail,
        }


@dataclass
class VerificationResult:
    passed: bool = False
    criteria: List[VerificationCriterion] = field(default_factory=list)
    score: float = 0.0
    summary: str = ""
    improvement: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "summary": self.summary,
            "criteria": [c.to_dict() for c in self.criteria],
            "improvement": {k: round(v, 4) for k, v in self.improvement.items()},
        }

    def failed_criteria(self) -> List[str]:
        return [c.name for c in self.criteria if not c.passed]


class RecoveryVerifier:
    """Class-specific acceptance criteria + before/after delta checks."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------
    def verify(
        self,
        signals_before: SignalBundle,
        signals_after: SignalBundle,
        failure_class: FailureClass,
        subtype: Optional[str] = None,
    ) -> VerificationResult:
        builder = {
            FailureClass.QUERY_FAILURE: self._query_criteria,
            FailureClass.RETRIEVAL_FAILURE: self._retrieval_criteria,
            FailureClass.RANKING_FAILURE: self._ranking_criteria,
            FailureClass.CONTEXT_FAILURE: self._context_criteria,
            FailureClass.GENERATION_FAILURE: self._generation_criteria,
            FailureClass.AGENT_CONTROL_FAILURE: self._agent_criteria,
            FailureClass.INFRASTRUCTURE_FAILURE: self._infrastructure_criteria,
            FailureClass.RESOURCE_FAILURE: self._infrastructure_criteria,
        }.get(failure_class, self._generic_criteria)
        result = builder(signals_before, signals_after, subtype)
        if not result.criteria:
            return result
        result.passed = all(c.passed for c in result.criteria)
        result.score = sum(1.0 for c in result.criteria if c.passed) / len(result.criteria)
        failed = result.failed_criteria()
        result.summary = (
            "all verification criteria passed"
            if result.passed
            else f"verification failed on: {', '.join(failed)}"
        )
        return result

    # ------------------------------------------------------------------
    def _generic_criteria(self, before: SignalBundle, after: SignalBundle, subtype: Optional[str]):
        improved = self._deltas(before, after)
        keys = ["groundedness", "context_coverage", "useful_ratio", "top_score"]
        gains = [improved.get(k, 0.0) for k in keys]
        return VerificationResult(
            improvement=improved,
            criteria=[
                VerificationCriterion(
                    "quality_improved",
                    passed=any(g > 0 for g in gains),
                    observed=round(max(gains), 4),
                    required="any positive improvement",
                    detail=f"deltas={ {k: round(v, 4) for k, v in improved.items()} }",
                )
            ],
        )

    def _infrastructure_criteria(self, before, after, subtype):
        errors_before = _int(before, "infrastructure", "infra_errors")
        errors_after = _int(after, "infrastructure", "infra_errors")
        timeouts_after = _int(after, "infrastructure", "timeouts")
        unavailable = bool(getattr(after, "infrastructure", {}).get("retrieval_service_unavailable"))
        embed_after = _int(after, "infrastructure", "embedding_errors")
        return VerificationResult(
            improvement={"infra_errors": errors_after - errors_before},
            criteria=[
                VerificationCriterion(
                    "no_new_infrastructure_errors",
                    passed=errors_after <= errors_before,
                    observed=errors_after,
                    required=f"<= {errors_before}",
                ),
                VerificationCriterion(
                    "retrieval_service_available",
                    passed=not unavailable,
                    observed=unavailable,
                    required=False,
                ),
                VerificationCriterion(
                    "no_timeouts",
                    passed=timeouts_after == 0,
                    observed=timeouts_after,
                    required=0,
                ),
                VerificationCriterion(
                    "no_embedding_errors",
                    passed=embed_after == 0,
                    observed=embed_after,
                    required=0,
                ),
            ],
        )

    def _agent_criteria(self, before, after, subtype):
        repeats_after = _int(after, "agent", "repeat_actions")
        failures_after = _int(after, "agent", "failed_actions")
        return VerificationResult(
            criteria=[
                VerificationCriterion(
                    "no_repeated_actions",
                    passed=repeats_after <= self.settings.budget.max_repeated_actions,
                    observed=repeats_after,
                    required=f"<= {self.settings.budget.max_repeated_actions}",
                ),
                VerificationCriterion(
                    "no_failed_actions",
                    passed=failures_after == 0,
                    observed=failures_after,
                    required=0,
                ),
            ],
        )

    def _generation_criteria(self, before, after, subtype):
        t = self.settings.thresholds
        grounded_after = _float(after, "generation", "groundedness", 0.0)
        grounded_before = _float(before, "generation", "groundedness", 0.0)
        unsupported = _int(after, "generation", "unsupported_claims")
        coverage = _float(after, "generation", "citation_coverage", 0.0)
        claims = _int(after, "generation", "claim_count")
        return VerificationResult(
            improvement={"groundedness": grounded_after - grounded_before},
            criteria=[
                VerificationCriterion(
                    "groundedness_above_threshold",
                    passed=grounded_after >= t.min_groundedness,
                    observed=round(grounded_after, 4),
                    required=t.min_groundedness,
                    detail="answer claims must be supported by retrieved evidence",
                ),
                VerificationCriterion(
                    "no_unsupported_claims",
                    passed=unsupported <= t.max_unsupported_claims,
                    observed=unsupported,
                    required=f"<= {t.max_unsupported_claims}",
                ),
                VerificationCriterion(
                    "citation_coverage_above_threshold",
                    passed=coverage >= t.min_citation_coverage,
                    observed=round(coverage, 4),
                    required=t.min_citation_coverage,
                ),
                VerificationCriterion(
                    "groundedness_not_degraded",
                    passed=grounded_after >= grounded_before - 1e-9,
                    observed=round(grounded_after, 4),
                    required=f">= {round(grounded_before, 4)}",
                ),
            ],
        )

    def _context_criteria(self, before, after, subtype):
        t = self.settings.thresholds
        coverage_after = _float(after, "context", "context_coverage", 0.0)
        coverage_before = _float(before, "context", "context_coverage", 0.0)
        tokens = _int(after, "context", "context_tokens")
        missing = _int(after, "context", "unconstrained_constraints")
        conflicts = _float(after, "context", "conflict_score", 0.0)
        return VerificationResult(
            improvement={"context_coverage": coverage_after - coverage_before},
            criteria=[
                VerificationCriterion(
                    "context_coverage_above_threshold",
                    passed=coverage_after >= t.min_context_coverage,
                    observed=round(coverage_after, 4),
                    required=t.min_context_coverage,
                ),
                VerificationCriterion(
                    "constraints_covered",
                    passed=missing == 0,
                    observed=missing,
                    required=0,
                    detail="every explicit constraint of the question must appear in context",
                ),
                VerificationCriterion(
                    "context_within_token_budget",
                    passed=t.min_context_tokens <= tokens <= t.max_context_tokens,
                    observed=tokens,
                    required=f"[{t.min_context_tokens}, {t.max_context_tokens}]",
                ),
                VerificationCriterion(
                    "no_conflicting_evidence",
                    passed=conflicts <= t.max_conflict_score,
                    observed=round(conflicts, 4),
                    required=t.max_conflict_score,
                ),
                VerificationCriterion(
                    "coverage_not_degraded",
                    passed=coverage_after >= coverage_before - 1e-9,
                    observed=round(coverage_after, 4),
                    required=f">= {round(coverage_before, 4)}",
                ),
            ],
        )

    def _ranking_criteria(self, before, after, subtype):
        useful_after = _float(after, "retrieval", "useful_ratio", 0.0)
        below_cut = _int(after, "retrieval", "relevant_below_cut")
        mrr_after = getattr(after, "retrieval", {}).get("mrr")
        mrr_before = getattr(before, "retrieval", {}).get("mrr")
        regression = _float(after, "ranking", "rank_regression_rate", 0.0)
        t = self.settings.thresholds
        criteria = [
            VerificationCriterion(
                "no_relevant_chunk_below_cut",
                passed=below_cut == 0,
                observed=below_cut,
                required=0,
            ),
            VerificationCriterion(
                "useful_ratio_above_threshold",
                passed=useful_after >= t.min_useful_chunk_ratio,
                observed=round(useful_after, 4),
                required=t.min_useful_chunk_ratio,
            ),
            VerificationCriterion(
                "rank_regression_bounded",
                passed=regression <= t.max_rank_regression_rate,
                observed=round(regression, 4),
                required=t.max_rank_regression_rate,
            ),
        ]
        if mrr_after is not None and mrr_before is not None:
            criteria.append(
                VerificationCriterion(
                    "mrr_not_reduced",
                    passed=float(mrr_after) >= float(mrr_before) - 1e-9,
                    observed=round(float(mrr_after), 4),
                    required=f">= {round(float(mrr_before), 4)}",
                )
            )
        return VerificationResult(criteria=criteria)

    def _retrieval_criteria(self, before, after, subtype):
        """Acceptance criteria for a retrieval recovery.

        Deliberately based on **lexical coverage** rather than raw similarity
        scores: cosine, BM25 and RRF scores have different scales, so comparing
        them across retrievals would be meaningless.  Coverage of the question's
        content terms is comparable across every backend.
        """
        t = self.settings.thresholds
        useful_after = _float(after, "retrieval", "useful_ratio", 0.0)
        useful_before = _float(before, "retrieval", "useful_ratio", 0.0)
        top_after = _float(after, "retrieval", "top_overlap", 0.0)
        top_before = _float(before, "retrieval", "top_overlap", 0.0)
        mean_after = _float(after, "retrieval", "mean_overlap", 0.0)
        count = _int(after, "retrieval", "retrieved_count")
        recall = getattr(after, "retrieval", {}).get("recall")
        recall_before = getattr(before, "retrieval", {}).get("recall")
        diversity = _int(after, "retrieval", "source_diversity")
        duplicates = _float(after, "retrieval", "duplicate_ratio", 0.0)

        criteria = [
            VerificationCriterion(
                "retrieval_not_empty",
                passed=count > 0,
                observed=count,
                required="> 0",
            ),
            VerificationCriterion(
                "useful_ratio_above_threshold",
                passed=useful_after >= t.min_useful_chunk_ratio,
                observed=round(useful_after, 4),
                required=t.min_useful_chunk_ratio,
            ),
            VerificationCriterion(
                "query_coverage_above_threshold",
                passed=top_after >= t.min_top_overlap,
                observed=round(top_after, 4),
                required=t.min_top_overlap,
                detail="the best retrieved chunk must cover the question's content terms",
            ),
            VerificationCriterion(
                "retrieval_quality_improved",
                passed=(top_after > top_before + 1e-9) or (useful_after > useful_before + 1e-9),
                observed=round(top_after, 4),
                required=f"> {round(top_before, 4)}",
                detail="recovery must improve coverage, never degrade it",
            ),
            VerificationCriterion(
                "duplicate_ratio_bounded",
                passed=duplicates <= t.max_duplicate_ratio,
                observed=round(duplicates, 4),
                required=t.max_duplicate_ratio,
            ),
        ]
        if recall is not None:
            criteria.append(
                VerificationCriterion(
                    "recall_acceptable",
                    passed=float(recall) >= max(0.5, float(recall_before or 0.0)),
                    observed=round(float(recall), 4),
                    required=f">= max(0.5, {round(float(recall_before or 0.0), 4)})",
                )
            )
        if diversity is not None:
            criteria.append(
                VerificationCriterion(
                    "source_diversity_adequate",
                    passed=diversity >= t.min_source_diversity,
                    observed=diversity,
                    required=t.min_source_diversity,
                )
            )
        return VerificationResult(
            improvement={
                "useful_ratio": useful_after - useful_before,
                "top_overlap": top_after - top_before,
                "mean_overlap": mean_after,
            },
            criteria=criteria,
        )

    def _query_criteria(self, before, after, subtype):
        t = self.settings.thresholds
        drift = _float(after, "query", "rewrite_drift", 0.0)
        change = _float(after, "query", "rewrite_change_ratio", 0.0)
        top_after = _float(after, "retrieval", "top_overlap", 0.0)
        top_before = _float(before, "retrieval", "top_overlap", 0.0)
        useful_after = _float(after, "retrieval", "useful_ratio", 0.0)
        useful_before = _float(before, "retrieval", "useful_ratio", 0.0)
        criteria = [
            VerificationCriterion(
                "rewrite_drift_bounded",
                passed=drift <= t.max_rewrite_drift,
                observed=round(drift, 4),
                required=t.max_rewrite_drift,
            ),
            VerificationCriterion(
                "rewrite_actually_changed_query",
                passed=change >= t.min_rewrite_change,
                observed=round(change, 4),
                required=t.min_rewrite_change,
            ),
            VerificationCriterion(
                "retrieval_improved",
                passed=(top_after > top_before + 1e-9) or (useful_after > useful_before + 1e-9),
                observed=round(top_after, 4),
                required=f"> {round(top_before, 4)}",
            ),
        ]
        return VerificationResult(criteria=criteria)

    # ------------------------------------------------------------------
    @staticmethod
    def _deltas(before: SignalBundle, after: SignalBundle) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for section in ("retrieval", "ranking", "context", "generation"):
            for key, value in getattr(after, section, {}).items():
                old = getattr(before, section, {}).get(key)
                if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                    out[f"{section}.{key}"] = float(value) - float(old)
        return out


def _float(bundle: SignalBundle, section: str, key: str, default: float = 0.0) -> float:
    value = getattr(bundle, section, {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(bundle: SignalBundle, section: str, key: str, default: int = 0) -> int:
    value = getattr(bundle, section, {}).get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default