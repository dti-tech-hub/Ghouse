"""Fault injector: applies declared faults to a :class:`Runtime`.

The injector is the *only* component allowed to mutate ``runtime.fault_flags``
or ``runtime.retriever.inject``.  It never edits thresholds, and it never
reveals the ground truth to the detector, the diagnoser or the planner.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from ..models import Chunk, FaultInjection
from ..observability.trace import get_logger
from ..pipeline.runtime import Runtime
from ..taxonomy import FailureClass, Stage
from .definitions import ALL_FAULTS, FaultDefinition, FaultRegistry

logger = get_logger(__name__)


class FaultInjector:
    """Controlled fault injection with ground-truth bookkeeping."""

    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    # ------------------------------------------------------------------
    def available(self) -> List[str]:
        return FaultRegistry.types()

    def definition(self, fault_type: str) -> Optional[FaultDefinition]:
        return FaultRegistry.get(fault_type)

    def build(
        self,
        fault_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> FaultInjection:
        definition = FaultRegistry.get(fault_type)
        if definition is None:
            raise ValueError(
                f"unknown fault type '{fault_type}'. Available: {', '.join(FaultRegistry.types())}"
            )
        return definition.build(parameters, experiment_id)

    # ------------------------------------------------------------------
    def activate(self, fault: FaultInjection, query: str = "") -> FaultInjection:
        """Apply one fault to the runtime."""
        runtime = self.runtime
        handler = getattr(self, f"_apply_{fault.fault_type.lower()}", None)
        if handler is None:
            raise ValueError(f"no handler implemented for fault type '{fault.fault_type}'")
        runtime.active_fault_ids.append(fault.fault_id)
        runtime.fault_flags[fault.fault_type] = dict(fault.injection_parameters)
        handler(fault, query)
        logger.info(
            "fault activated: %s -> stage=%s expected=%s/%s",
            fault.fault_type,
            fault.target_stage.value,
            fault.expected_failure_class.value,
            fault.expected_subtype,
        )
        return fault

    def activate_all(
        self, faults: Sequence[FaultInjection], query: str = ""
    ) -> List[FaultInjection]:
        return [self.activate(fault, query) for fault in faults]

    def release(self) -> None:
        self.runtime.restore_clean_state()

    # ------------------------------------------------------------------
    # Query-stage faults
    # ------------------------------------------------------------------
    def _apply_query_corruption(self, fault: FaultInjection, query: str) -> None:
        self.runtime.fault_flags.setdefault("QUERY_CORRUPTION", {})["query"] = query

    def _apply_bad_query_rewrite(self, fault: FaultInjection, query: str) -> None:
        drift = float(fault.injection_parameters.get("drift", 0.8))
        runtime = self.runtime
        if drift >= 0.75:
            replacement = _drifted_query(query)
        else:
            replacement = _light_drift(query)
        runtime.knobs.force_rewrite = replacement
        runtime.fault_flags["BAD_QUERY_REWRITE"]["applied_rewrite"] = replacement

    def _apply_ambiguous_query(self, fault: FaultInjection, query: str) -> None:
        keep = int(fault.injection_parameters.get("keep_words", 2))
        words = query.split()
        runtime = self.runtime
        runtime.knobs.force_rewrite = " ".join(words[:keep]) if words else query
        runtime.fault_flags["AMBIGUOUS_QUERY"]["applied_rewrite"] = runtime.knobs.force_rewrite

    def _apply_incomplete_decomposition(self, fault: FaultInjection, query: str) -> None:
        keep = int(fault.injection_parameters.get("keep", 1))
        self.runtime.fault_flags["INCOMPLETE_DECOMPOSITION"]["keep"] = keep

    # ------------------------------------------------------------------
    # Retrieval-stage faults
    # ------------------------------------------------------------------
    def _apply_relevant_doc_removal(self, fault: FaultInjection, query: str) -> None:
        runtime = self.runtime
        gold_docs = set(getattr(runtime, "gold_document_ids", set()) or set())
        gold_chunks = set(getattr(runtime, "gold_chunk_ids", set()) or set())
        if not gold_docs and runtime.retriever is not None:
            from ..retrieval.diagnostics import retrieval_ground_truth

            gold = retrieval_ground_truth(
                query, runtime.retriever.chunks, runtime.fault_flags.get("_answer_keywords", [])
            )
            gold_chunks = set(gold)
            gold_docs = {c.document_id for c in runtime.retriever.chunks if c.chunk_id in gold}
        if runtime.retriever is None or not gold_docs:
            logger.warning("RELEVANT_DOC_REMOVAL: no ground-truth document identified")
            runtime.fault_flags["RELEVANT_DOC_REMOVAL"]["skipped"] = "no ground-truth document"
            return
        removed = [c for c in runtime.retriever.chunks if c.document_id in gold_docs]
        runtime.document_removals.extend(sorted(gold_docs))
        runtime.retriever.remove_document_ids(list(gold_docs))
        runtime.retriever.rebuild([c for c in runtime.retriever.chunks if c.document_id not in gold_docs])
        runtime.fault_flags["RELEVANT_DOC_REMOVAL"]["removed_documents"] = sorted(gold_docs)
        runtime.fault_flags["RELEVANT_DOC_REMOVAL"]["removed_chunks"] = len(removed)

    def _apply_irrelevant_doc_injection(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["irrelevant_injection"] = True

    def _apply_low_top_k(self, fault: FaultInjection, query: str) -> None:
        cap = int(fault.injection_parameters.get("top_k_cap", 2))
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["top_k_cap"] = cap
        self.runtime.knobs.top_k = cap

    def _apply_empty_retrieval(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["force_empty_retrieval"] = True

    def _apply_stale_document(self, fault: FaultInjection, query: str) -> None:
        """Replace the current version of the relevant document with a superseded one."""
        runtime = self.runtime
        if runtime.retriever is None:
            return
        gold_chunks = set(getattr(runtime, "gold_chunk_ids", set()) or set())
        gold_docs = set(getattr(runtime, "gold_document_ids", set()) or set())
        if not gold_docs:
            gold_docs = {c.document_id for c in runtime.retriever.chunks if c.chunk_id in gold_chunks}
        if not gold_docs:
            logger.warning("STALE_DOCUMENT: no ground-truth document identified")
            runtime.fault_flags["STALE_DOCUMENT"]["skipped"] = "no ground-truth document"
            return

        stale_ids: List[str] = []
        stale_chunks: List[Chunk] = []
        for document_id in sorted(gold_docs):
            current = [c for c in runtime.retriever.chunks if c.document_id == document_id]
            if not current:
                continue
            joined = "\n\n".join(c.content for c in current)
            outdated = _make_outdated(joined, document_id)
            stale_chunk = Chunk(
                document_id=f"{document_id}__v1",
                chunk_index=0,
                content=outdated,
                token_estimate=max(1, len(outdated) // 4),
                page=current[0].page,
                section="outdated-version",
                source=current[0].source,
                metadata={
                    "injected": True,
                    "stale": True,
                    "superseded": True,
                    "version": 1,
                    "latest_version": 2,
                    "supersedes": document_id,
                },
            )
            stale_chunks.append(stale_chunk)
            stale_ids.append(document_id)
        if not stale_ids:
            return
        runtime.document_removals.extend(stale_ids)
        runtime.retriever.remove_document_ids(stale_ids)
        runtime.retriever.rebuild(list(runtime.retriever.chunks) + stale_chunks)
        runtime.fault_flags["STALE_DOCUMENT"]["superseded_documents"] = stale_ids
        runtime.fault_flags["STALE_DOCUMENT"]["stale_document_ids"] = [c.document_id for c in stale_chunks]

    def _apply_retrieval_timeout(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["force_timeout"] = True

    def _apply_tool_timeout(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["force_timeout"] = True

    def _apply_retrieval_service_unavailable(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.service_available = False

    def _apply_embedding_failure(self, fault: FaultInjection, query: str) -> None:
        runtime = self.runtime
        if runtime.retriever is None:
            return
        runtime.retriever.inject["force_embedding_failure"] = True
        runtime.retriever.inject["embedding_failure_mode"] = fault.injection_parameters.get("mode", "raise")
        runtime.fault_flags["EMBEDDING_FAILURE"]["mode"] = runtime.retriever.inject["embedding_failure_mode"]

    def _apply_model_failure(self, fault: FaultInjection, query: str) -> None:
        self.runtime.fault_flags["MODEL_FAILURE"]["enabled"] = True

    # ------------------------------------------------------------------
    # Ranking-stage faults
    # ------------------------------------------------------------------
    def _apply_ranking_corruption(self, fault: FaultInjection, query: str) -> None:
        if self.runtime.retriever is not None:
            self.runtime.retriever.inject["corrupt_rerank"] = True
            self.runtime.retriever.inject["rerank_mode"] = fault.injection_parameters.get("mode", "reverse")

    # ------------------------------------------------------------------
    # Context-stage faults
    # ------------------------------------------------------------------
    def _apply_chunk_fragmentation(self, fault: FaultInjection, query: str) -> None:
        runtime = self.runtime
        runtime.chunk_config_snapshot = {
            "chunk_size": runtime.settings.chunking.chunk_size,
            "chunk_overlap": runtime.settings.chunking.chunk_overlap,
            "min_chunk_chars": runtime.settings.chunking.min_chunk_chars,
        }
        runtime.settings.chunking.chunk_size = int(fault.injection_parameters.get("chunk_size", 90))
        runtime.settings.chunking.chunk_overlap = int(fault.injection_parameters.get("chunk_overlap", 8))
        runtime.knobs.chunk_size = runtime.settings.chunking.chunk_size
        runtime.knobs.chunk_overlap = runtime.settings.chunking.chunk_overlap
        if runtime.ingestor is not None:
            runtime.ingestor.chunker.config = runtime.settings.chunking
        if runtime.baseline_chunks:
            runtime._rechunk_all(runtime.baseline_chunks)

    def _apply_context_truncation(self, fault: FaultInjection, query: str) -> None:
        runtime = self.runtime
        runtime.knobs.max_context_tokens = int(fault.injection_parameters.get("max_context_tokens", 40))
        runtime.knobs.max_chunks = int(fault.injection_parameters.get("max_chunks", 1))

    def _apply_conflicting_evidence(self, fault: FaultInjection, query: str) -> None:
        runtime = self.runtime
        if runtime.retriever is None:
            return
        from ..textutil import split_sentences, tokenize

        gold = set(getattr(runtime, "gold_chunk_ids", set()) or set())
        anchors = [c for c in runtime.retriever.chunks if not gold or c.chunk_id in gold]
        if not anchors:
            return
        # pick the anchor sentence with the highest overlap with the question so
        # the decoy genuinely competes for the same evidence slot
        q_tokens = set(tokenize(query))
        best_sentence, best_score, anchor = None, -1.0, anchors[0]
        for chunk in anchors:
            for sentence in split_sentences(chunk.content):
                tokens = set(tokenize(sentence))
                if len(tokens) < 4:
                    continue
                score = len(q_tokens & tokens) / max(1, len(tokens))
                if score > best_score:
                    best_sentence, best_score, anchor = sentence, score, chunk
        sentences = [s for s in split_sentences(anchor.content) if len(s.split()) > 4]
        decoy_source = best_sentence or (sentences[0] if sentences else anchor.content[:200])
        negated = _negate(decoy_source)
        decoy = Chunk(
            document_id=f"decoy_{anchor.document_id}",
            chunk_index=0,
            content=(
                "ARCHIVED NOTE. Corrections published by the operations team.\n"
                f"{negated.rstrip('.')}.\n"
                "Reviewers should follow this archived note until the review closes."
            ),
            token_estimate=max(1, len(negated) // 4),
            page=anchor.page,
            section="injected-conflict",
            source=anchor.source,
            metadata={"injected": True, "conflict_with": anchor.chunk_id},
        )
        runtime.decoy_chunks.append(decoy)
        # The declared target stage of this fault is context_assembly, so the
        # decoy is injected into the assembled context (not force-ranked into
        # retrieval).  Retrieval and ranking therefore stay untouched.
        runtime.fault_flags["CONFLICTING_EVIDENCE"]["decoy_chunk_id"] = decoy.chunk_id
        runtime.fault_flags["CONFLICTING_EVIDENCE"]["anchors"] = anchor.chunk_id
        runtime.fault_flags["CONFLICTING_EVIDENCE"]["decoy_text"] = decoy.content

    # ------------------------------------------------------------------
    # Generation-stage faults
    # ------------------------------------------------------------------
    def _apply_generation_constraint_removal(self, fault: FaultInjection, query: str) -> None:
        self.runtime.knobs.ground_answer = False
        self.runtime.knobs.require_grounding = False

    # ------------------------------------------------------------------
    def active_summary(self) -> List[Dict[str, Any]]:
        return [{"fault_type": k, **v} for k, v in self.runtime.fault_flags.items()]


# ---------------------------------------------------------------------------
def _make_outdated(text: str, document_id: str) -> str:
    """Deterministically rewrite the key figures to produce a superseded version."""
    def bump(match: "re.Match[str]") -> str:
        return str(int(match.group(0)) * 2 + 5)

    altered = re.sub(r"\b\d+\b", bump, text)
    if altered == text:
        altered = (
            "ARCHIVED - SUPERSEDED VERSION 1. This earlier revision is retained for "
            f"audit only and no longer reflects current practice. {text[:400]}"
        )
    return f"[SUPERSEDED v1 of {document_id}]\n{altered}"


def _negate(sentence: str) -> str:
    """Produce a minimally-edited contradicting statement (deterministic)."""
    text = sentence.strip().rstrip(".")
    lowered = text.lower()
    swaps = [
        ("must", "must not"),
        ("requires", "does not require"),
        ("shall", "shall not"),
        ("is ", "is not "),
        ("are ", "are not "),
        ("within ", "not within "),
        ("maximum", "no"),
    ]
    for original, replacement in swaps:
        if original in lowered:
            index = lowered.index(original)
            return text[:index] + replacement + text[index + len(original) :]
    numbers = re.findall(r"\d+", text)
    if numbers:
        bumped = str(int(numbers[0]) + 7)
        return text.replace(numbers[0], bumped, 1)
    return f"It is not true that {text[0].lower()}{text[1:]}" if text else "Not applicable."


def _drifted_query(query: str) -> str:
    """A same-shaped question about a completely different topic."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2]
    topic = " ".join(w.lower() for w in words[:3]) or "general"
    return f"How does the {topic} framework compare with legacy alternatives in practice?"


def _light_drift(query: str) -> str:
    """A partial rewrite: some anchor terms survive, but several are lost."""
    words = query.split()
    if len(words) < 4:
        return _drifted_query(query)
    kept = words[: max(2, len(words) // 3)]
    return " ".join(kept) + " overview and background information"


__all__ = ["FaultInjector", "ALL_FAULTS", "FaultDefinition", "FaultRegistry"]