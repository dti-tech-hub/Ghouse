"""Tier-2 LLM diagnoser."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..llm.base import LLMProvider, LLMResponseError, ProviderTimeout
from ..models import Diagnosis, DiagnosisTier, SignalBundle
from ..observability.trace import get_logger
from ..prompts import diagnose_prompt
from ..settings import Settings, get_settings
from ..signals.collector import signals_for_diagnoser
from ..taxonomy import (
    FailureClass,
    RecoveryAction,
    Severity,
    Stage,
    class_for_subtype,
    recovery_ladder,
    severity_for_subtype,
    stage_for_subtype,
)
from .schema import taxonomy_for_prompt, validate_diagnosis_text

logger = get_logger(__name__)


class LLMDiagnoser:
    """Wraps any :class:`LLMProvider` into a structured diagnoser."""

    tier = DiagnosisTier.LLM

    def __init__(self, provider: LLMProvider, settings: Optional[Settings] = None):
        self.provider = provider
        self.settings = settings or get_settings()
        self.stats = {"calls": 0, "valid": 0, "invalid": 0, "timeouts": 0, "errors": 0}

    def thresholds(self) -> Dict[str, float]:
        """Operating points the diagnoser is told about (single source of truth)."""
        t = self.settings.thresholds
        return {
            "min_top_overlap": t.min_top_overlap,
            "min_mean_overlap": t.min_mean_overlap,
            "useful_overlap": t.useful_overlap,
            "min_useful_chunk_ratio": t.min_useful_chunk_ratio,
            "min_groundedness": t.min_groundedness,
            "min_context_coverage": t.min_context_coverage,
            "min_citation_coverage": t.min_citation_coverage,
            "max_conflict_score": t.max_conflict_score,
            "max_fragmentation_ratio": t.max_fragmentation_ratio,
        }

    # ------------------------------------------------------------------
    def diagnose(
        self,
        signals: SignalBundle,
        run_id: str = "",
        rule_shortlist: Optional[List[Dict[str, Any]]] = None,
        query: str = "",
    ) -> Diagnosis:
        started = time.perf_counter()
        signal_view = signals_for_diagnoser(signals, self.settings.diagnosis.max_evidence_items)
        payload_signals = dict(signal_view["flat"])
        payload_signals["question_preview"] = (query or "")[:200]
        prompt = diagnose_prompt(
            signals=payload_signals,
            taxonomy=taxonomy_for_prompt(),
            rule_shortlist=rule_shortlist or [],
            thresholds=self.thresholds(),
        )
        system = (
            "You are a failure-diagnosis expert for an agentic RAG pipeline. "
            "Return exactly one JSON object."
        )

        raw_text: Optional[str] = None
        errors: List[str] = []
        usage = (0, 0)
        cost = 0.0

        self.stats["calls"] += 1
        try:
            if hasattr(self.provider, "generate_json"):
                parsed = self.provider.generate_json(prompt, system=system)
                usage_meta = parsed.pop("_usage", {}) if isinstance(parsed, dict) else {}
                usage = (int(usage_meta.get("prompt_tokens", 0)), int(usage_meta.get("completion_tokens", 0)))
                raw_text = self._dumps(parsed)
            else:  # pragma: no cover - defensive
                result = self.provider.generate(prompt, system=system)
                raw_text = result.text
                usage = (result.usage.prompt_tokens, result.usage.completion_tokens)
                cost = result.cost
        except ProviderTimeout:
            self.stats["timeouts"] += 1
            errors.append("diagnoser timed out")
        except LLMResponseError as exc:
            self.stats["invalid"] += 1
            errors.append(f"diagnoser response error: {exc}")
        except Exception as exc:
            self.stats["errors"] += 1
            errors.append(f"diagnoser failure: {type(exc).__name__}: {exc}")

        outcome = validate_diagnosis_text(raw_text) if raw_text else None
        if outcome is None or outcome.payload is None:
            self.stats["invalid"] += 1
            return self._fallback(
                signals,
                run_id,
                errors=errors + (outcome.errors if outcome else ["no diagnoser output"]),
                repairs=outcome.repairs if outcome else [],
                raw=raw_text,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        self.stats["valid"] += 1
        parsed = outcome.payload
        failure_class = parsed.failure_class
        subtype = parsed.subtype
        confidence = max(0.0, min(1.0, parsed.confidence - outcome.confidence_penalty))
        if not parsed.failure_detected or failure_class is FailureClass.NONE:
            confidence = min(confidence, 0.5)
        ladder = recovery_ladder(failure_class, None if subtype == "none" else subtype)
        recommended = parsed.recommended_recovery
        if recommended not in ladder:
            recommended = ladder[0]
        return Diagnosis(
            run_id=run_id,
            failure_detected=bool(parsed.failure_detected and failure_class is not FailureClass.NONE),
            failure_class=failure_class,
            stage=parsed.stage if parsed.failure_detected else None,
            subtype=subtype,
            severity=parsed.severity,
            confidence=confidence,
            evidence=parsed.evidence,
            recommended_recovery=recommended if parsed.failure_detected else None,
            recovery_ladder=[a for a in ladder],
            tier=DiagnosisTier.LLM,
            repairs=outcome.repairs,
            rule_matches=[],
            signals_used={k: v for k, v in payload_signals.items() if k != "question_preview"},
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            raw_llm_output=raw_text,
            validation_errors=errors + outcome.errors,
            metadata={
                "llm_calls": 1,
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "cost": cost,
                "repairs": outcome.repairs,
                "provider": getattr(self.provider, "name", "unknown"),
                "model": getattr(self.provider, "model", "unknown"),
            },
        )

    # ------------------------------------------------------------------
    def _fallback(
        self,
        signals: SignalBundle,
        run_id: str,
        errors: List[str],
        repairs: List[str],
        raw: Optional[str],
        latency_ms: float,
    ) -> Diagnosis:
        """When Tier-2 is unusable we say so explicitly (never invent a cause)."""
        return Diagnosis(
            run_id=run_id,
            failure_detected=False,
            failure_class=FailureClass.NONE,
            stage=None,
            subtype="llm_diagnoser_unavailable",
            severity=Severity.LOW,
            confidence=0.0,
            evidence=errors or ["LLM diagnoser produced no usable output"],
            recommended_recovery=None,
            recovery_ladder=[],
            tier=DiagnosisTier.FALLBACK,
            validation_errors=errors,
            latency_ms=round(latency_ms, 3),
            raw_llm_output=raw,
            metadata={"repairs": repairs, "fallback_reason": "tier2_unusable"},
        )

    @staticmethod
    def _dumps(payload: Dict[str, Any]) -> str:
        import json

        return json.dumps(payload, default=str)

    def health(self) -> Dict[str, Any]:
        return {"provider": self.provider.health(), "stats": dict(self.stats)}