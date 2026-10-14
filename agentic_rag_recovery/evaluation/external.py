"""Optional external evaluation adapters (RAGAS / DeepEval).

The project never depends on these packages.  When they are installed the
adapters below make their metrics available *alongside* the deterministic
custom metrics; the reported figures in the database still come from
:mod:`agentic_rag_recovery.signals.metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..observability.trace import get_logger

logger = get_logger(__name__)


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except Exception:
        return False


def deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401

        return True
    except Exception:
        return False


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: Sequence[str],
    reference: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    """Faithfulness / answer-relevancy via RAGAS, when installed."""
    if not ragas_available():
        return None
    try:  # pragma: no cover - optional dependency
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import answer_relevancy, context_precision, faithfulness  # type: ignore

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [list(contexts)],
                "ground_truth": [reference],
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            **kwargs,
        )
        frame = result.to_pandas()
        return {column: float(frame[column].iloc[0]) for column in frame.columns}
    except Exception as exc:  # pragma: no cover
        logger.warning("RAGAS evaluation failed: %s", exc)
        return None


def evaluate_with_deepeval(
    question: str, answer: str, contexts: Sequence[str], **kwargs: Any
) -> Optional[Dict[str, float]]:
    """Groundedness via DeepEval, when installed."""
    if not deepeval_available():
        return None
    try:  # pragma: no cover - optional dependency
        from deepeval.metrics import GroundednessMetric  # type: ignore
        from deepeval.test_case import LLMTestCase  # type: ignore

        metric = GroundednessMetric()
        case = LLMTestCase(input=question, actual_output=answer, retrieval_context=list(contexts))
        metric.measure(case)
        return {"groundedness": float(metric.score)}
    except Exception as exc:  # pragma: no cover
        logger.warning("DeepEval evaluation failed: %s", exc)
        return None


def availability() -> Dict[str, bool]:
    return {"ragas": ragas_available(), "deepeval": deepeval_available()}