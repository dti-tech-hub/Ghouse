"""Deterministic offline LLM provider.

This provider is what makes DEMO MODE work with **no API key and no network**.
It is *not* a language model: it is a rule-driven generator that consumes the
structured payload embedded in each prompt (see :mod:`agentic_rag_recovery.prompts`).

Determinism guarantees
----------------------
* identical (prompt, seed) -> identical completion
* no randomness, no wall-clock dependence, no network
* crucially it **never sees the injected fault ground truth**, so the
  diagnosis-accuracy numbers measured with it are not self-fulfilling.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Sequence

from ..prompts import Task, parse_payload
from ..taxonomy import FailureClass, RecoveryAction, Severity, Stage, recovery_ladder
from ..textutil import (
    clip,
    estimate_tokens,
    has_negation,
    numeric_values,
    split_sentences,
    stable_int,
    stem,
    tokenize,
)
from .base import ChatMessage, CompletionResult, LLMProvider, Usage

_ANSWER_OPENERS = {
    "list": "The retrieved evidence lists the following:",
    "definition": "According to the retrieved evidence:",
    "reason": "The retrieved evidence explains that:",
    "default": "Based on the retrieved evidence:",
}


class MockProvider(LLMProvider):
    """Rule-driven deterministic provider (implements ``LLMProvider`` API)."""

    name = "mock"

    def __init__(self, model: str = "mock-deterministic-v1", seed: int = 42, **kwargs: Any):
        self.model = model
        self.options = kwargs
        self.seed = int(kwargs.get("seed", seed))
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.max_tokens = int(kwargs.get("max_tokens", 512))
        self.timeout = float(kwargs.get("timeout", 30.0))
        self._call_count = 0
        self._last_task: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------
    def _complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResult:
        user_text = "\n".join(m.content for m in messages if m.role != "system")
        payload = parse_payload(user_text) or {}
        task = payload.get("task")
        if not task:
            task = self._infer_task(user_text)
        self._last_task = task
        started = time.perf_counter()
        if task == Task.ANSWER:
            text = self._answer(payload)
        elif task == Task.REWRITE:
            text = self._rewrite(payload)
        elif task == Task.DECOMPOSE:
            text = self._decompose(payload)
        elif task == Task.DIAGNOSE:
            text = self._diagnose(payload)
        elif task == Task.VERIFY:
            text = self._verify(payload)
        else:
            text = self._freeform(user_text)
        latency = (time.perf_counter() - started) * 1000.0
        self.history.append({"task": task, "chars": len(text)})
        return CompletionResult(
            text=text,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
                completion_tokens=estimate_tokens(text),
            ),
            latency_ms=latency,
            model=self.model,
            provider=self.name,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _infer_task(text: str) -> str:
        low = text.lower()
        if "return exactly one json object" in low or "failure_class" in low:
            return Task.DIAGNOSE
        if "json array of strings" in low:
            return Task.DECOMPOSE
        if "expanded search query" in low:
            return Task.REWRITE
        if "context:" in low and "[s1]" in low:
            return Task.ANSWER
        return "freeform"

    # ------------------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------------------
    def _answer(self, payload: Dict[str, Any]) -> str:
        question = str(payload.get("question") or "")
        context: List[Dict[str, Any]] = list(payload.get("context") or [])
        require_grounding = bool(payload.get("require_grounding", True))
        cite = bool(payload.get("cite_sources", True))
        ground = bool(payload.get("ground_answer", True))
        max_sentences = int(payload.get("max_sentences", 4) or 4)

        usable = [c for c in context if (c.get("text") or "").strip()]
        if not usable:
            return (
                "I could not obtain sufficient reliable evidence to answer this question: "
                "no usable passages were retrieved from the knowledge base."
            )

        q_tokens = tokenize(question)
        # ---- defect mode: grounding constraint removed -------------------
        if not ground or not require_grounding:
            return self._ungrounded_answer(question, usable, cite)

        # ---- defect mode: incomplete answer ------------------------------
        max_sentences = max(1, min(max_sentences, 3))

        ranked: List[Dict[str, Any]] = []
        for position, item in enumerate(usable, start=1):
            marker = item.get("marker") or f"S{position}"
            for sent in split_sentences(item.get("text", "")):
                overlap = len(set(tokenize(sent)) & set(q_tokens)) / max(1, len(set(q_tokens)))
                if overlap <= 0:
                    continue
                ranked.append(
                    {
                        "sentence": clip(sent, 260),
                        "marker": marker,
                        "score": overlap,
                        "order": len(ranked),
                    }
                )
        ranked.sort(key=lambda x: (-x["score"], x["order"]))
        if not ranked:
            # no sentence shares vocabulary with the question -> do not guess
            return (
                "I could not obtain sufficient reliable evidence to answer this question: "
                "the retrieved passages do not address the question."
            )
        # draw on as many distinct passages as possible so that the answer is
        # properly cited rather than resting on a single chunk
        best_per_marker: Dict[str, Dict[str, Any]] = {}
        for item in ranked:
            current = best_per_marker.get(item["marker"])
            if current is None or item["score"] > current["score"]:
                best_per_marker[item["marker"]] = item
        chosen = sorted(best_per_marker.values(), key=lambda x: x["order"])[:max_sentences]
        opener = self._opener(question)
        parts = [opener]
        for i, item in enumerate(chosen, start=1):
            marker = item["marker"]
            if cite:
                parts.append(f"{item['sentence']} [{marker}].")
            else:
                parts.append(f"{item['sentence']}.")
        return " ".join(parts)

    def _ungrounded_answer(self, question: str, context: List[Dict[str, Any]], cite: bool) -> str:
        """Fabricated answer used when the grounding constraint is removed.

        Content is derived deterministically from the question so that it is
        *plausible but unsupported* - exactly the failure mode we want to
        detect (hallucination / low groundedness / zero citation coverage).
        """
        q_tokens = [t for t in tokenize(question) if len(t) > 3]
        seed = stable_int(question, self.seed, modulo=10_000)
        years = [1998, 2003, 2009, 2014, 2018, 2021, 2023]
        pct = 10 + (seed % 80)
        pct2 = 5 + ((seed // 7) % 90)
        subject = " ".join(q_tokens[:3]) if q_tokens else "the subject"
        extra = " ".join(q_tokens[3:6]) if len(q_tokens) > 3 else "in general"
        claims = [
            f"The {subject} was formally established in {years[seed % len(years)]} and currently "
            f"covers roughly {pct}% of all recorded cases involving {extra}.",
            f"According to the latest consolidated review, the figure for {subject} is exactly "
            f"{pct2}.{seed % 10} and has remained unchanged since adoption.",
            f"Most authorities now treat {subject} as the primary determinant when {extra} is "
            f"considered alongside the standard criteria.",
        ]
        return " ".join(claims)

    @staticmethod
    def _opener(question: str) -> str:
        low = question.lower()
        if re.search(r"\b(what are|which|list|name)\b", low) and re.search(
            r"\b(types|kinds|categories|steps|components|methods|options)\b", low
        ):
            return _ANSWER_OPENERS["list"]
        if re.search(r"^(what is|define|what does)\b", low):
            return _ANSWER_OPENERS["definition"]
        if re.search(r"\bwhy\b|\bhow\b|\bexplain\b", low):
            return _ANSWER_OPENERS["reason"]
        return _ANSWER_OPENERS["default"]

    # ------------------------------------------------------------------
    # REWRITE
    # ------------------------------------------------------------------
    #: Deterministic domain-synonym table.  ``data/demo_questions/lexicon.json``
    #: is merged in when present, otherwise this built-in table is used.  It
    #: represents the mock LLM's "knowledge" and is *not* derived from any
    #: injected-fault ground truth.
    SYNONYMS: Dict[str, List[str]] = {
        "policy": ["guideline", "procedure", "standard"],
        "guideline": ["policy", "procedure"],
        "requirement": ["obligation", "rule", "mandate"],
        "deadline": ["cutoff", "due date", "time limit"],
        "retry": ["re-attempt", "second attempt"],
        "cache": ["buffer", "memo"],
        "latency": ["delay", "response time", "round trip time"],
        "throughput": ["volume", "requests per second"],
        "incident": ["outage", "failure event"],
        "outage": ["downtime", "incident"],
        "salary": ["compensation", "pay", "remuneration"],
        "leave": ["vacation", "time off", "absence"],
        "employee": ["staff member", "worker"],
        "security": ["safety", "protection"],
        "encryption": ["at rest", "data protection"],
        "backup": ["snapshot", "recovery copy"],
        "customer": ["client", "account holder"],
        "refund": ["money back", "reimbursement"],
        "shipping": ["delivery", "dispatch"],
        "warranty": ["guarantee", "service promise"],
        "password": ["credential", "secret"],
        "access": ["entry", "permission"],
        "escalation": ["on-call", "handover"],
        "monitoring": ["observability", "alerting"],
        "audit": ["review", "inspection"],
        "vendor": ["supplier", "third party"],
        "compliance": ["regulatory alignment", "conformance"],
        "candidate": ["applicant", "job seeker"],
        "interview": ["assessment", "screening"],
        "invoice": ["bill", "payment request"],
        "budget": ["cost plan", "spend limit"],
    }

    _LEXICON_LOADED = False

    @classmethod
    def _load_lexicon(cls) -> Dict[str, List[str]]:
        """Merge the optional project lexicon into the built-in synonym table."""
        if cls._LEXICON_LOADED:
            return cls.SYNONYMS
        cls._LEXICON_LOADED = True
        try:
            from pathlib import Path

            from ..settings import PROJECT_ROOT

            path = Path(PROJECT_ROOT) / "data" / "demo_questions" / "lexicon.json"
            if not path.exists():
                return cls.SYNONYMS
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        merged = list(cls.SYNONYMS.get(key, [])) + [
                            v for v in values if v not in cls.SYNONYMS.get(key, [])
                        ]
                        cls.SYNONYMS[key] = merged
        except Exception:  # pragma: no cover - lexicon is optional
            pass
        return cls.SYNONYMS

    def _rewrite(self, payload: Dict[str, Any]) -> str:
        synonyms = self._load_lexicon()
        query = str(payload.get("query") or "")
        if not query.strip():
            return query
        tokens = tokenize(query)
        keep = [t for t in tokens if len(t) > 2]
        expansions: List[str] = []
        seen = set(keep)
        for token in keep:
            for syn in synonyms.get(token, []):
                for word in tokenize(syn):
                    if word not in seen and len(word) > 2:
                        expansions.append(word)
                        seen.add(word)
        # also expand the stemmed forms of capitalised / compound query terms
        for original, stemmed in zip(
            [m.group(0).lower() for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]+", query)],
            [stem(m.group(0).lower()) for m in re.finditer(r"[A-Za-z][A-Za-z0-9\-]+", query)],
        ):
            for syn in synonyms.get(original, []):
                for word in tokenize(syn):
                    if word not in seen and len(word) > 2:
                        expansions.append(word)
                        seen.add(word)
        parts = [query.strip().rstrip("?")]
        if expansions:
            parts.append(" ".join(dict.fromkeys(expansions))[:220])
        rewritten = " ".join(p for p in parts if p).strip()
        return rewritten or query

    # ------------------------------------------------------------------
    # DECOMPOSE
    # ------------------------------------------------------------------
    def _decompose(self, payload: Dict[str, Any]) -> str:
        import json

        query = str(payload.get("query") or "")
        max_sub = int(payload.get("max_subqueries", 4) or 4)
        sentences = [s.strip() for s in re.split(r"(?<=[?])\s+|;\s+|\s+and\s+|\s+then\s+", query) if s.strip()]
        subqueries: List[str] = []
        for sentence in sentences:
            subqueries.append(sentence if sentence.endswith("?") else sentence + "?")
        if not subqueries:
            subqueries = [query]
        return json.dumps(subqueries[:max_sub])

    # ------------------------------------------------------------------
    # DIAGNOSE (Tier-2 stand-in for demo mode)
    # ------------------------------------------------------------------
    def _diagnose(self, payload: Dict[str, Any]) -> str:
        import json

        signals: Dict[str, Any] = dict(payload.get("signals") or {})
        taxonomy = {t["subtype"]: t for t in (payload.get("taxonomy") or []) if isinstance(t, dict)}
        th = dict(payload.get("thresholds") or {})
        min_useful = float(th.get("min_useful_chunk_ratio", 0.40))
        min_top_overlap = float(th.get("min_top_overlap", 0.34))
        min_groundedness = float(th.get("min_groundedness", 0.62))

        # --- infrastructure first: it explains everything downstream -------
        if signals.get("infra_errors", 0) > 0 or signals.get("timeouts", 0) > 0:
            subtype = "timeout" if signals.get("timeouts", 0) > 0 else _infer_infra_subtype(signals)
            return json.dumps(self._diag_dict(taxonomy, subtype, signals, 0.88, 0.9))

        # --- groundedness ------------------------------------------------
        groundedness = float(signals.get("groundedness", 1.0) or 0.0)
        unsupported = int(signals.get("unsupported_claims", 0) or 0)
        citation_cov = float(signals.get("citation_coverage", 1.0) or 0.0)
        if groundedness < min_groundedness or unsupported > 0:
            subtype = "unsupported_claim" if unsupported > 0 and citation_cov > 0 else "hallucination"
            if citation_cov == 0.0:
                subtype = "hallucination"
            return json.dumps(self._diag_dict(taxonomy, subtype, signals, 0.84, 0.86))

        # --- context -----------------------------------------------------
        min_context_coverage = float(th.get("min_context_coverage", 0.50))
        max_conflict = float(th.get("max_conflict_score", 0.35))
        max_fragmentation = float(th.get("max_fragmentation_ratio", 0.50))
        context_cov = float(signals.get("context_coverage", 1.0) or 0.0)
        conflicts = float(signals.get("conflict_score", 0.0) or 0.0)
        fragmentation = float(signals.get("fragmentation_ratio", 0.0) or 0.0)
        if (
            int(signals.get("unconstrained_constraints", 0) or 0) > 0
            or conflicts > max_conflict
            or fragmentation > max_fragmentation
            or context_cov < min_context_coverage
        ):
            subtype = (
                "missing_evidence"
                if int(signals.get("unconstrained_constraints", 0) or 0) > 0
                else "conflicting_evidence"
                if conflicts > max_conflict
                else "fragmented_context"
                if fragmentation > max_fragmentation
                else "insufficient_context"
            )
            return json.dumps(self._diag_dict(taxonomy, subtype, signals, 0.78, 0.8))

        # --- ranking -----------------------------------------------------
        if int(signals.get("relevant_below_cut", 0) or 0) > 0:
            return json.dumps(self._diag_dict(taxonomy, "relevant_ranked_low", signals, 0.72, 0.7))
        if float(signals.get("ranking_coverage_loss", 0.0) or 0.0) >= 0.15:
            return json.dumps(self._diag_dict(taxonomy, "irrelevant_ranked_high", signals, 0.7, 0.65))

        # --- retrieval ---------------------------------------------------
        if int(signals.get("retrieved_count", 0) or 0) == 0:
            return json.dumps(self._diag_dict(taxonomy, "no_useful_evidence", signals, 0.82, 0.85))
        if float(signals.get("top_overlap", 1.0) or 0.0) < min_top_overlap:
            return json.dumps(self._diag_dict(taxonomy, "low_recall", signals, 0.74, 0.75))

        # --- query understanding ------------------------------------------
        if not signals.get("ambiguous", False) and int(signals.get("query_words", 0) or 0) > 0:
            return json.dumps(
                {
                    "failure_detected": False,
                    "failure_class": "NONE",
                    "stage": "validation",
                    "subtype": "none",
                    "severity": "low",
                    "confidence": 0.6,
                    "evidence": [
                        "no rule fired above threshold",
                        f"question coverage {signals.get('top_overlap')} >= {min_top_overlap}",
                        f"useful ratio {signals.get('useful_ratio')} >= {min_useful}",
                    ],
                    "recommended_recovery": "ABSTAIN",
                }
            )

        return json.dumps(
            {
                "failure_detected": False,
                "failure_class": "NONE",
                "stage": "validation",
                "subtype": "none",
                "severity": "low",
                "confidence": 0.62,
                "evidence": [
                    "groundedness above threshold",
                    "retrieval scores above threshold",
                    "context coverage complete",
                ],
                "recommended_recovery": "ABSTAIN",
            }
        )

    def _diag_dict(
        self,
        taxonomy: Dict[str, Dict[str, Any]],
        subtype: str,
        signals: Dict[str, Any],
        confidence: float,
        severity_score: float,
    ) -> Dict[str, Any]:
        meta = taxonomy.get(subtype, {})
        recovery = (meta.get("recovery_actions") or [RecoveryAction.ABSTAIN.value])[0]
        severity = (
            Severity.CRITICAL
            if severity_score >= 0.85
            else Severity.HIGH
            if severity_score >= 0.7
            else Severity.MEDIUM
        )
        evidence = self._evidence_for(subtype, signals) or list(meta.get("evidence_hints", []))[:2]
        return {
            "failure_detected": True,
            "failure_class": meta.get("failure_class", "INFRASTRUCTURE_FAILURE"),
            "stage": meta.get("stage", "retrieval"),
            "subtype": subtype,
            "severity": severity.value if isinstance(severity, Severity) else str(severity),
            "confidence": round(float(confidence), 4),
            "evidence": evidence,
            "recommended_recovery": recovery,
        }

    @staticmethod
    def _evidence_for(subtype: str, signals: Dict[str, Any]) -> List[str]:
        def g(key: str, default: Any = "-") -> Any:
            return signals.get(key, default)

        table = {
            "low_recall": [f"top_score={g('top_score')}", f"mean_score={g('mean_score')}", f"useful_ratio={g('useful_ratio')}"],
            "no_useful_evidence": [f"retrieved_count={g('retrieved_count')}", f"useful_ratio={g('useful_ratio')}"],
            "empty_retrieval": ["retrieval returned zero results"],
            "irrelevant_retrieval": [f"top_score={g('top_score')}", f"max_chunk_overlap={g('max_chunk_overlap')}"],
            "irrelevant_ranked_high": [f"useful_ratio={g('useful_ratio')}", f"rank_regression_rate={g('rank_regression_rate')}"],
            "relevant_ranked_low": [f"rank_regression_rate={g('rank_regression_rate')}", f"relevant_below_cut={g('relevant_below_cut')}"],
            "poor_reranking": [f"mrr_delta_after_rerank={g('mrr_delta_after_rerank')}"],
            "insufficient_context": [f"context_tokens={g('context_tokens')}", f"context_coverage={g('context_coverage')}"],
            "fragmented_context": [f"fragmentation_ratio={g('fragmentation_ratio')}"],
            "conflicting_evidence": [f"conflict_score={g('conflict_score')}"],
            "missing_evidence": [f"unconstrained_constraints={g('unconstrained_constraints')}"],
            "excessive_context": [f"context_tokens={g('context_tokens')}", f"relevant_buried={g('relevant_buried')}"],
            "hallucination": [f"groundedness={g('groundedness')}", f"citation_coverage={g('citation_coverage')}"],
            "unsupported_claim": [f"unsupported_claims={g('unsupported_claims')}", f"groundedness={g('groundedness')}"],
            "incomplete_answer": [f"subquestion_coverage={g('subquestion_coverage')}"],
            "contradiction": [f"negation_conflicts={g('negation_conflicts')}"],
            "timeout": [f"timeouts={g('timeouts')}", f"latency_ms={g('latency_ms')}"],
            "malformed_response": [f"parse_errors={g('parse_errors')}"],
            "embedding_failure": [f"embedding_errors={g('embedding_errors')}"],
            "model_failure": [f"provider_errors={g('provider_errors')}"],
            "retrieval_service_unavailable": [f"infra_errors={g('infra_errors')}"],
            "repeated_action": [f"repeat_actions={g('repeat_actions')}", f"duplicate_action_rate={g('duplicate_action_rate')}"],
            "ambiguous_query": [f"query_words={g('query_words')}", f"ambiguous={g('ambiguous')}"],
            "incorrect_rewrite": [f"rewrite_drift={g('rewrite_drift')}"],
            "incomplete_decomposition": [f"subquery_count={g('subquery_count')}"],
            "budget_exhausted": ["recovery attempt budget exhausted"],
        }
        return table.get(subtype, [])

    # ------------------------------------------------------------------
    # VERIFY / freeform
    # ------------------------------------------------------------------
    def _verify(self, payload: Dict[str, Any]) -> str:
        import json

        from ..signals.metrics import groundedness_score

        context = list(payload.get("context") or [])
        result = groundedness_score(
            question=str(payload.get("question") or ""),
            answer=str(payload.get("answer") or ""),
            context=[c.get("text", "") for c in context],
            citation_markers=[c.get("marker") or f"S{i+1}" for i, c in enumerate(context)],
        )
        return json.dumps(
            {
                "supported": result["groundedness"] >= 0.6,
                "groundedness": round(result["groundedness"], 4),
                "unsupported_claims": result["unsupported_claims"],
            }
        )

    def _freeform(self, user_text: str) -> str:
        first = user_text.strip().splitlines()
        return first[0][:500] if first else ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MockProvider model={self.model!r}>"


def _infer_infra_subtype(signals: Dict[str, Any]) -> str:
    if signals.get("parse_errors", 0):
        return "malformed_response"
    if signals.get("embedding_errors", 0):
        return "embedding_failure"
    if signals.get("provider_errors", 0):
        return "model_failure"
    if signals.get("retrieval_service_unavailable"):
        return "retrieval_service_unavailable"
    return "timeout"