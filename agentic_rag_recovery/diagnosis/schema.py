"""Strict schema for LLM-produced diagnoses.

The LLM diagnoser is never trusted: its output is parsed, validated against the
taxonomy, and repaired where possible.  Every repair and every violation is
recorded on the :class:`~agentic_rag_recovery.models.Diagnosis` object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..taxonomy import (
    FailureClass,
    FailureTaxonomy,
    RecoveryAction,
    Severity,
    Stage,
    class_for_subtype,
    recovery_ladder,
    stage_for_subtype,
)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class DiagnosisSchema(BaseModel):
    """The exact JSON shape the Tier-2 diagnoser must return."""

    failure_detected: bool
    failure_class: FailureClass
    stage: Stage
    subtype: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(min_length=1)
    recommended_recovery: RecoveryAction

    @field_validator("subtype")
    @classmethod
    def known_subtype(cls, value: str) -> str:
        if value in {"none", "", None}:
            return "none"
        if FailureTaxonomy.get(value) is None:
            raise ValueError(f"unknown subtype '{value}'")
        return value


class ValidationOutcome(BaseModel):
    valid: bool = False
    payload: Optional[DiagnosisSchema] = None
    raw: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    repairs: List[str] = Field(default_factory=list)

    @property
    def confidence_penalty(self) -> float:
        return min(0.4, 0.08 * len(self.errors) + 0.03 * len(self.repairs))


def extract_json(text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Pull a JSON object out of free-form model output."""
    errors: List[str] = []
    if not text or not text.strip():
        return None, ["empty model response"]
    raw = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(cleaned)
        if not match:
            return None, ["no JSON object found in model response"]
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, [f"invalid JSON: {exc.msg} (line {exc.lineno})"]
    if not isinstance(payload, dict):
        return None, ["model response is not a JSON object"]
    return payload, errors


_ALIASES: Dict[str, Tuple[str, ...]] = {
    "class": ("failure_class",),
    "error_class": ("failure_class",),
    "root_cause": ("failure_class",),
    "type": ("subtype",),
    "reason": ("evidence",),
    "reasons": ("evidence",),
    "why": ("evidence",),
    "recovery": ("recommended_recovery",),
    "fix": ("recommended_recovery",),
    "recommended_action": ("recommended_recovery",),
    "confidence_score": ("confidence",),
    "sev": ("severity",),
    "detected": ("failure_detected",),
}


def normalise_keys(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    repairs: List[str] = []
    out = dict(payload)
    for alias, targets in _ALIASES.items():
        if alias in out and targets[0] not in out:
            out[targets[0]] = out.pop(alias)
            repairs.append(f"renamed '{alias}' -> '{targets[0]}'")
    return out, repairs


def coerce_types(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    repairs: List[str] = []
    out = dict(payload)

    if "evidence" in out:
        evidence = out["evidence"]
        if isinstance(evidence, str):
            out["evidence"] = [evidence]
            repairs.append("wrapped evidence string into a list")
        elif isinstance(evidence, dict):
            out["evidence"] = [f"{k}: {v}" for k, v in evidence.items()]
            repairs.append("flattened evidence mapping into a list")
        elif isinstance(evidence, list):
            out["evidence"] = [str(e) for e in evidence if str(e).strip()]

    if "confidence" in out:
        value = out["confidence"]
        if isinstance(value, str):
            try:
                text = value.strip().rstrip("%")
                number = float(text)
                # both "0.85" and "85" / "85%" mean a probability
                if number > 1.0 or value.strip().endswith("%"):
                    number = number / 100.0
                out["confidence"] = max(0.0, min(1.0, number))
                repairs.append("coerced confidence from string")
            except ValueError:
                out["confidence"] = 0.5
                repairs.append("unparseable confidence replaced with 0.5")
        elif isinstance(value, (int, float)):
            if value > 1.0:
                out["confidence"] = round(float(value) / 100.0, 4)
                repairs.append("scaled confidence from percent to 0-1")
            out["confidence"] = max(0.0, min(1.0, float(value)))

    if "failure_detected" in out:
        value = out["failure_detected"]
        if isinstance(value, str):
            out["failure_detected"] = value.strip().lower() in {"true", "yes", "1", "y"}
            repairs.append("coerced failure_detected from string")

    return out, repairs


def _closest_subtype(subtype: Optional[str], failure_class: FailureClass) -> str:
    if subtype and FailureTaxonomy.get(subtype):
        return subtype
    candidates = FailureTaxonomy.subtypes(failure_class)
    if candidates:
        return candidates[0]
    fallback_by_class = {
        FailureClass.RETRIEVAL_FAILURE: "low_recall",
        FailureClass.RANKING_FAILURE: "relevant_ranked_low",
        FailureClass.CONTEXT_FAILURE: "insufficient_context",
        FailureClass.GENERATION_FAILURE: "unsupported_claim",
        FailureClass.QUERY_FAILURE: "ambiguous_query",
        FailureClass.AGENT_CONTROL_FAILURE: "incorrect_workflow",
        FailureClass.INFRASTRUCTURE_FAILURE: "timeout",
        FailureClass.RESOURCE_FAILURE: "budget_exhausted",
        FailureClass.NONE: "none",
    }
    return fallback_by_class.get(failure_class, "low_recall")


def validate_diagnosis_payload(
    payload: Dict[str, Any],
    repair: bool = True,
) -> ValidationOutcome:
    """Validate (and optionally repair) a raw LLM diagnosis dictionary."""
    errors: List[str] = []
    repairs: List[str] = []
    normalised, r1 = normalise_keys(payload)
    repairs.extend(r1)
    coerced, r2 = coerce_types(normalised)
    repairs.extend(r2)

    required = DiagnosisSchema.model_fields.keys()
    missing = [field for field in required if field not in coerced]
    if missing:
        errors.append(f"missing required field(s): {', '.join(sorted(missing))}")
        if not repair:
            return ValidationOutcome(valid=False, raw=payload, errors=errors, repairs=repairs)
        for field_name in missing:
            coerced.setdefault(field_name, {
                "failure_detected": False,
                "failure_class": FailureClass.NONE.value,
                "stage": Stage.VALIDATION.value,
                "subtype": "none",
                "severity": Severity.LOW.value,
                "confidence": 0.3,
                "evidence": ["field missing from model response; defaulted"],
                "recommended_recovery": RecoveryAction.ABSTAIN.value,
            }[field_name])
        repairs.append(f"defaulted missing field(s): {', '.join(sorted(missing))}")

    # reconcile class/stage/subtype consistency
    subtype = str(coerced.get("subtype") or "")
    known = FailureTaxonomy.get(subtype) is not None
    if not known:
        try:
            failure_class = FailureClass(str(coerced.get("failure_class")).upper())
        except ValueError:
            failure_class = FailureClass.NONE
        coerced["subtype"] = _closest_subtype(subtype, failure_class)
        repairs.append(f"unknown subtype '{subtype}' -> '{coerced['subtype']}'")
        subtype = coerced["subtype"]
    else:
        expected_class = class_for_subtype(subtype)
        try:
            given_class = FailureClass(str(coerced.get("failure_class")).upper())
        except ValueError:
            given_class = expected_class
            repairs.append("invalid failure_class replaced from taxonomy")
        if expected_class != given_class:
            coerced["failure_class"] = expected_class.value
            repairs.append(f"failure_class '{given_class.value}' -> '{expected_class.value}' (taxonomy)")

    try:
        given_stage = Stage(str(coerced.get("stage")).lower())
    except ValueError:
        given_stage = stage_for_subtype(subtype)
        repairs.append(f"invalid stage replaced with '{given_stage.value}'")
    coerced["stage"] = given_stage.value

    try:
        given_action = RecoveryAction(str(coerced.get("recommended_recovery")).upper())
    except ValueError:
        given_action = recovery_ladder(
            FailureClass(str(coerced.get("failure_class")).upper()), subtype
        )[0]
        repairs.append(f"invalid recovery -> '{given_action.value}'")

    ladder = recovery_ladder(FailureClass(str(coerced["failure_class"]).upper()), subtype)
    if given_action not in ladder and given_action is not RecoveryAction.ABSTAIN:
        given_action = ladder[0]
        repairs.append(f"recovery '{coerced['recommended_recovery']}' not valid for subtype -> '{given_action.value}'")
    coerced["recommended_recovery"] = given_action.value

    try:
        schema = DiagnosisSchema(**coerced)
        return ValidationOutcome(valid=True, payload=schema, raw=payload, errors=errors, repairs=repairs)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}")
        return ValidationOutcome(valid=False, raw=payload, errors=errors, repairs=repairs)


def validate_diagnosis_text(text: Optional[str], repair: bool = True) -> ValidationOutcome:
    payload, errors = extract_json(text)
    if payload is None:
        return ValidationOutcome(valid=False, errors=errors)
    outcome = validate_diagnosis_payload(payload, repair=repair)
    outcome.errors = errors + outcome.errors
    return outcome


def taxonomy_for_prompt() -> List[Dict[str, Any]]:
    """Compact taxonomy description handed to the Tier-2 diagnoser."""
    out: List[Dict[str, Any]] = []
    for name, definition in sorted(FailureTaxonomy.all().items()):
        out.append(
            {
                "subtype": name,
                "failure_class": class_for_subtype(name).value,
                "stage": definition.stage.value,
                "severity": definition.default_severity.value,
                "description": definition.description,
                "recovery_actions": [a.value for a in definition.recovery_actions],
                "evidence_hints": list(definition.evidence_hints),
            }
        )
    return out


def schema_as_json() -> str:
    return json.dumps(DiagnosisSchema.model_json_schema(), indent=2)