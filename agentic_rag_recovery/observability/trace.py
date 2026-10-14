"""Internal observability: trace collection + structured logging.

Langfuse / OpenTelemetry are *optional* integrations; the system records its
own spans in SQLite and never requires an external collector.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from ..models import AgentStep, TraceSpan, new_id, stable_hash, utcnow

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, os.environ.get("ARR_LOG_LEVEL", level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-34s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


class TraceCollector:
    """Collects :class:`TraceSpan` objects and :class:`AgentStep` records."""

    def __init__(self, run_id: str = "", keep_steps: bool = True):
        self.run_id = run_id or new_id("run")
        self.spans: List[TraceSpan] = []
        self.steps: List[AgentStep] = []
        self.keep_steps = keep_steps
        self._step_index = 0

    # ------------------------------------------------------------------
    @contextmanager
    def span(
        self,
        stage: str,
        name: str = "",
        tags: Optional[List[str]] = None,
        capture_inputs: bool = True,
    ) -> Iterator[TraceSpan]:
        span = TraceSpan(
            span_id=new_id("spn"),
            run_id=self.run_id,
            stage=stage,
            name=name or stage,
            tags=tags or [],
        )
        start = time.perf_counter()
        try:
            yield span
            span.status = "ok"
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            span.end_time = utcnow()
            self.spans.append(span)

    # ------------------------------------------------------------------
    def step(
        self,
        stage: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        status: str = "ok",
        latency_ms: float = 0.0,
        tool_name: Optional[str] = None,
        error: Optional[str] = None,
        **meta: Any,
    ) -> AgentStep:
        step = AgentStep(
            run_id=self.run_id,
            index=self._step_index,
            stage=stage,
            action=action,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
            latency_ms=round(latency_ms, 3),
            tool_name=tool_name,
            error=error,
            action_signature=action_signature(stage, action, input_summary),
            meta=meta,
        )
        self._step_index += 1
        if self.keep_steps:
            self.steps.append(step)
        return step

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        by_stage: Dict[str, float] = {}
        for span in self.spans:
            by_stage[span.stage] = round(by_stage.get(span.stage, 0.0) + span.latency_ms, 3)
        return {
            "run_id": self.run_id,
            "spans": len(self.spans),
            "steps": len(self.steps),
            "latency_by_stage": by_stage,
            "total_latency_ms": round(sum(by_stage.values()), 3),
            "errors": [s.error for s in self.spans if s.error],
        }

    def merge(self, other: "TraceCollector") -> None:
        self.spans.extend(other.spans)
        self.steps.extend(other.steps)
        self._step_index = max(self._step_index, len(self.steps))


def action_signature(stage: str, action: str, payload: str = "") -> str:
    """Stable hash used to detect repeated agent actions."""
    return stable_hash(stage, action, payload)


class MetricsRegistry:
    """In-process counters used by the API/dashboard (cheap, no DB)."""

    def __init__(self) -> None:
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def snapshot(self) -> Dict[str, Any]:
        return {"counters": dict(self.counters), "gauges": dict(self.gauges)}


class LangfuseExporter:
    """Optional Langfuse exporter. Always a no-op when the package is absent."""

    def __init__(self, public_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self.enabled = bool(self.public_key and self.secret_key)
        self.client = None
        if self.enabled:  # pragma: no cover - optional integration
            try:
                from langfuse import Langfuse  # type: ignore

                self.client = Langfuse(public_key=self.public_key, secret_key=self.secret_key)
            except Exception:
                self.enabled = False

    def export_span(self, span: TraceSpan) -> bool:
        if not self.enabled or self.client is None:
            return False
        try:  # pragma: no cover - optional integration
            self.client.trace(
                name=span.name,
                trace_id=self.run_id if hasattr(self, "run_id") else span.run_id,
                metadata={"stage": span.stage, **span.metrics},
            )
            return True
        except Exception:
            return False


def trace_summary(spans: List[TraceSpan]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_stage": {}, "total_ms": 0.0, "errors": 0}
    for span in spans:
        entry = out["by_stage"].setdefault(span.stage, {"latency_ms": 0.0, "count": 0, "errors": 0})
        entry["latency_ms"] += span.latency_ms
        entry["count"] += 1
        if span.error:
            entry["errors"] += 1
            out["errors"] += 1
        out["total_ms"] += span.latency_ms
    return out