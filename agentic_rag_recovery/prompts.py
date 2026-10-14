"""Prompt construction.

Prompts carry a machine-readable payload block so that:

* :class:`~agentic_rag_recovery.llm.mock.MockProvider` can answer
  deterministically from structured data, and
* real providers receive natural-language instructions rendered from the same
  payload.

The payload is plain JSON only - no code is ever executed from it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

PAYLOAD_OPEN = "<<<ARR_PAYLOAD>>>"
PAYLOAD_CLOSE = "<<<ARR_END>>>"


class Task:
    ANSWER = "answer"
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    DIAGNOSE = "diagnose"
    VERIFY = "verify"


SYSTEM_PROMPTS: Dict[str, str] = {
    Task.ANSWER: (
        "You are a careful retrieval-augmented assistant. Answer ONLY using the "
        "provided context. Cite evidence with bracketed markers like [S1]. If the "
        "context does not support an answer, say so explicitly."
    ),
    Task.REWRITE: (
        "You rewrite search queries for a retrieval system. Produce a single "
        "expanded query that preserves the original intent and adds likely "
        "terminology. Never answer the question."
    ),
    Task.DECOMPOSE: (
        "You decompose complex questions into independent sub-questions that can "
        "each be answered from a document collection. Return a JSON array only."
    ),
    Task.DIAGNOSE: (
        "You are a failure-diagnosis expert for an agentic RAG pipeline. Given only "
        "runtime signals, return a single JSON object describing the most likely "
        "root cause. Never invent signal values."
    ),
    Task.VERIFY: (
        "You verify whether a generated answer is fully supported by the retrieved "
        "evidence. Return JSON only."
    ),
}


def render_payload(payload: Dict[str, Any]) -> str:
    return (
        f"{PAYLOAD_OPEN}\n"
        f"{json.dumps(payload, indent=2, default=str, sort_keys=True)}\n"
        f"{PAYLOAD_CLOSE}"
    )


def parse_payload(text: str) -> Optional[Dict[str, Any]]:
    """Extract the embedded payload from a prompt (used by the mock provider)."""
    if not text or PAYLOAD_OPEN not in text:
        return None
    try:
        body = text.split(PAYLOAD_OPEN, 1)[1].split(PAYLOAD_CLOSE, 1)[0]
        return json.loads(body)
    except (IndexError, json.JSONDecodeError):
        return None


def _render_context(items: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        marker = item.get("marker") or f"S{i}"
        src = item.get("source") or item.get("document_id") or "unknown"
        lines.append(f"[{marker}] (source: {src})\n{item.get('text', '')}")
    return "\n\n".join(lines)


def answer_prompt(
    question: str,
    context_items: Sequence[Dict[str, Any]],
    require_grounding: bool = True,
    cite: bool = True,
    ground: bool = True,
    max_sentences: int = 4,
    constraints: Optional[List[str]] = None,
    previous_answer: Optional[str] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    payload = {
        "task": Task.ANSWER,
        "question": question,
        "context": list(context_items),
        "require_grounding": bool(require_grounding),
        "cite_sources": bool(cite),
        "ground_answer": bool(ground),
        "max_sentences": int(max_sentences),
        "constraints": list(constraints or []),
        "previous_answer": previous_answer,
    }
    instructions = [
        f"QUESTION:\n{question}",
        "",
        "CONTEXT:",
        _render_context(context_items) if context_items else "(no context retrieved)",
        "",
        "RULES:",
        f"- Use ONLY the context above. Ground every claim in a retrieved passage. ({'required' if require_grounding else 'not required'})",
        f"- Cite each claim with the bracketed marker of the passage it came from. ({'required' if cite else 'optional'})",
        f"- Keep the answer to at most {max_sentences} evidence sentences.",
    ]
    if constraints:
        instructions.append("- Address every constraint: " + "; ".join(constraints))
    if previous_answer:
        instructions.append(
            "- The previous answer was defective. Produce a corrected answer that is fully "
            "supported by the context and resolves the defect."
        )
    if not context_items:
        instructions.append(
            "- No context is available. State that sufficient evidence is unavailable "
            "instead of inventing an answer."
        )
    if extra_instructions:
        instructions.append(f"- {extra_instructions}")
    instructions.append(render_payload(payload))
    return "\n".join(instructions)


def rewrite_prompt(query: str, feedback: Optional[str] = None, failed_queries: Optional[List[str]] = None) -> str:
    payload = {
        "task": Task.REWRITE,
        "query": query,
        "feedback": feedback,
        "failed_queries": list(failed_queries or []),
    }
    instructions = [
        f"ORIGINAL QUERY:\n{query}",
        "",
        "RULES:",
        "- Produce ONE expanded search query string.",
        "- Preserve the original intent; do not answer the question.",
        "- Keep the important nouns; add alternative surface forms a document might use.",
        "- Maximum 40 words.",
    ]
    if feedback:
        instructions.append(f"- Fix this specific problem: {feedback}")
    if failed_queries:
        instructions.append(f"- Previous attempts that failed: {'; '.join(failed_queries)}")
    instructions.append(render_payload(payload))
    return "\n".join(instructions)


def decompose_prompt(query: str, max_subqueries: int = 4) -> str:
    payload = {"task": Task.DECOMPOSE, "query": query, "max_subqueries": int(max_subqueries)}
    instructions = [
        f"QUESTION:\n{query}",
        "",
        "RULES:",
        f"- Decompose into at most {max_subqueries} independent sub-questions.",
        "- Each sub-question must be answerable from a single passage.",
        "- Preserve every constraint of the original question.",
        "- Return a JSON array of strings.",
        render_payload(payload),
    ]
    return "\n".join(instructions)


DIAGNOSIS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "failure_detected",
        "failure_class",
        "stage",
        "subtype",
        "severity",
        "confidence",
        "evidence",
        "recommended_recovery",
    ],
    "properties": {
        "failure_detected": {"type": "boolean"},
        "failure_class": {
            "type": "string",
            "enum": [
                "QUERY_FAILURE",
                "RETRIEVAL_FAILURE",
                "RANKING_FAILURE",
                "CONTEXT_FAILURE",
                "GENERATION_FAILURE",
                "AGENT_CONTROL_FAILURE",
                "INFRASTRUCTURE_FAILURE",
                "RESOURCE_FAILURE",
                "NONE",
            ],
        },
        "stage": {
            "type": "string",
            "enum": [
                "query_understanding",
                "query_decomposition",
                "retrieval",
                "ranking",
                "context_assembly",
                "generation",
                "validation",
                "agent_control",
                "infrastructure",
            ],
        },
        "subtype": {"type": "string"},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "recommended_recovery": {
            "type": "string",
            "enum": [
                "QUERY_REWRITE",
                "HYBRID_SEARCH",
                "EXPAND_TOP_K",
                "RERANK",
                "CHUNK_EXPANSION",
                "CONTEXT_EXPANSION",
                "CONSTRAINED_REGENERATION",
                "DECOMPOSITION_REPAIR",
                "BROADEN_RETRIEVAL",
                "BOUNDED_RETRY",
                "ALTERNATE_PATH",
                "TOOL_REPAIR",
                "ABSTAIN",
            ],
        },
    },
}


def diagnose_prompt(
    signals: Dict[str, Any],
    taxonomy: Sequence[Dict[str, Any]],
    rule_shortlist: Sequence[Dict[str, Any]] = (),
    thresholds: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "task": Task.DIAGNOSE,
        "signals": signals,
        "taxonomy": list(taxonomy),
        "rule_shortlist": list(rule_shortlist),
        "thresholds": dict(thresholds or {}),
    }
    instructions = [
        "The deterministic rule engine was INCONCLUSIVE. Diagnose the root cause "
        "using only the runtime signals below.",
        "",
        "RULES:",
        "- Return exactly ONE JSON object, no prose, no markdown fences.",
        "- failure_class, stage and recommended_recovery MUST come from the provided enums.",
        "- subtype MUST be one of the listed taxonomy subtypes (use the closest match).",
        "- evidence MUST cite concrete signal values that justify the decision.",
        "- If nothing is wrong, set failure_detected=false and confidence accordingly.",
        "- The thresholds below are the system's operating points; use them rather "
        "than inventing your own.",
        "",
        "ALLOWED TAXONOMY:",
        json.dumps(list(taxonomy), indent=2),
        "",
        "SIGNALS:",
        json.dumps(signals, indent=2, default=str),
        "",
        "THRESHOLDS:",
        json.dumps(thresholds or {}, indent=2),
        "",
        "RETURN THIS JSON SHAPE:",
        json.dumps(DIAGNOSIS_JSON_SCHEMA, indent=2),
        render_payload(payload),
    ]
    return "\n".join(instructions)


def verify_prompt(
    question: str,
    answer: str,
    context_items: Sequence[Dict[str, Any]],
    constraints: Optional[List[str]] = None,
) -> str:
    payload = {
        "task": Task.VERIFY,
        "question": question,
        "answer": answer,
        "context": list(context_items),
        "constraints": list(constraints or []),
    }
    instructions = [
        f"QUESTION:\n{question}",
        f"ANSWER:\n{answer}",
        "CONTEXT:",
        _render_context(context_items) if context_items else "(empty)",
        "",
        "Return JSON: {\"supported\": bool, \"groundedness\": float, \"unsupported_claims\": [str]}",
        render_payload(payload),
    ]
    return "\n".join(instructions)