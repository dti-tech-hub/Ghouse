"""Two-tier diagnosis engine.

    signals ──▶ TIER 1 rule detector
                   │
                   ├─ confidence ≥ threshold ──▶ diagnosis (tier=rule)
                   │
                   └─ otherwise ──▶ TIER 2 LLM diagnoser
                                          │
                                          └─ schema validation ──▶ diagnosis (tier=llm)
                                                                     (tier=fallback on error)

Every decision is counted so the dashboard can show the tier mix.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..detection.rules import RuleDetector, RuleMatch
from ..models import Diagnosis, DiagnosisTier, SignalBundle
from ..observability.trace import get_logger
from ..settings import Settings, get_settings
from ..signals.collector import signals_for_diagnoser
from ..taxonomy import (
    FailureClass,
    RecoveryAction,
    Severity,
    class_for_subtype,
    recovery_ladder,
)
from .llm_diagnoser import LLMDiagnoser

logger = get_logger(__name__)


class DiagnosisEngine:
    """Dispatch between Tier-1 and Tier-2 and normalise the output."""

    def __init__(
        self,
        detector: Optional[RuleDetector] = None,
        llm_diagnoser: Optional[LLMDiagnoser] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.detector = detector or RuleDetector(self.settings)
        self.llm = llm_diagnoser
        self.stats: Dict[str, int] = {
            "rule": 0,
            "llm": 0,
            "fallback": 0,
            "none": 0,
            "total": 0,
            "escalated": 0,
        }

    # ------------------------------------------------------------------
    def diagnose(
        self,
        signals: SignalBundle,
        run_id: str = "",
        query: str = "",
        force_tier2: bool = False,
    ) -> Diagnosis:
        started = time.perf_counter()
        matches: List[RuleMatch] = self.detector.evaluate(signals)
        threshold = self.settings.diagnosis.rule_confidence_threshold
        best = self.detector.best(signals)

        self.stats["total"] += 1
        if not force_tier2 and best is not None and best.confidence >= threshold:
            return self._from_rule(best, matches, signals, run_id, started)

        if self.llm is None or not self.settings.diagnosis.use_llm_fallback:
            # Tier-2 disabled/unavailable: fall back to the strongest rule we have
            if best is not None:
                self.stats["fallback"] += 1
                diagnosis = self._from_rule(best, matches, signals, run_id, started)
                diagnosis.tier = DiagnosisTier.FALLBACK
                diagnosis.metadata["note"] = "Tier-2 disabled; using strongest rule below threshold"
                return diagnosis
            self.stats["none"] += 1
            return self._healthy(run_id, matches, started)

        self.stats["escalated"] += 1
        diagnosis = self.llm.diagnose(
            signals,
            run_id=run_id,
            rule_shortlist=[m.to_dict() for m in matches[:3]],
            query=query,
        )
        diagnosis.rule_matches = [m.to_dict() for m in matches]
        diagnosis.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        if diagnosis.tier == DiagnosisTier.FALLBACK:
            self.stats["fallback"] += 1
            # a fallback is not a diagnosis - surface it honestly
            diagnosis.failure_detected = bool(matches)
            if matches and diagnosis.subtype == "llm_diagnoser_unavailable":
                best_match = matches[0]
                diagnosis.failure_class = best_match.failure_class
                diagnosis.stage = best_match.stage
                diagnosis.subtype = best_match.subtype
                diagnosis.severity = best_match.severity
                diagnosis.confidence = round(best_match.confidence * 0.7, 4)
                diagnosis.recommended_recovery = best_match.subtype and recovery_ladder(
                    best_match.failure_class, best_match.subtype
                )[0]
                diagnosis.recovery_ladder = list(
                    recovery_ladder(best_match.failure_class, best_match.subtype)
                )
                diagnosis.evidence = best_match.evidence
                diagnosis.metadata["note"] = (
                    "Tier-2 unusable; degraded to Tier-1 rule evidence with reduced confidence"
                )
            return diagnosis

        self.stats["llm"] += 1
        if diagnosis.failure_detected:
            self.stats["rule"] = self.stats.get("rule", 0)  # no-op, keeps shape explicit
        return diagnosis

    # ------------------------------------------------------------------
    def _from_rule(
        self,
        match: RuleMatch,
        matches: List[RuleMatch],
        signals: SignalBundle,
        run_id: str,
        started: float,
    ) -> Diagnosis:
        self.stats["rule"] += 1
        ladder = recovery_ladder(match.failure_class, match.subtype)
        return Diagnosis(
            run_id=run_id,
            failure_detected=True,
            failure_class=match.failure_class,
            stage=match.stage,
            subtype=match.subtype,
            severity=match.severity,
            confidence=round(match.confidence, 4),
            evidence=match.evidence,
            recommended_recovery=ladder[0],
            recovery_ladder=list(ladder),
            tier=DiagnosisTier.RULE,
            rule_id=match.rule_id,
            rule_matches=[m.to_dict() for m in matches],
            signals_used=match.signals,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            metadata={"rule_count": len(matches)},
        )

    def _healthy(self, run_id: str, matches: List[RuleMatch], started: float) -> Diagnosis:
        return Diagnosis(
            run_id=run_id,
            failure_detected=False,
            failure_class=FailureClass.NONE,
            stage=None,
            subtype="none",
            severity=Severity.LOW,
            confidence=0.5,
            evidence=["no deterministic rule fired above threshold"],
            recommended_recovery=None,
            recovery_ladder=[],
            tier=DiagnosisTier.NONE,
            rule_matches=[m.to_dict() for m in matches],
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    # ------------------------------------------------------------------
    def stats_snapshot(self) -> Dict[str, Any]:
        snapshot = dict(self.stats)
        total = max(1, snapshot.get("total", 0))
        snapshot["rule_rate"] = round(snapshot.get("rule", 0) / total, 4)
        snapshot["llm_rate"] = round(snapshot.get("llm", 0) / total, 4)
        snapshot["fallback_rate"] = round(snapshot.get("fallback", 0) / total, 4)
        snapshot["escalation_rate"] = round(snapshot.get("escalated", 0) / total, 4)
        return snapshot

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0