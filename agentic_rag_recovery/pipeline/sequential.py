"""Sequential (framework-free) runner.

The project supports LangGraph but does not depend on it: the identical stage
functions are executed by a plain state machine, which keeps the demo runnable
even if LangGraph is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..observability.trace import get_logger
from ..taxonomy import PIPELINE_STAGES, Stage
from .nodes import STAGE_SEQUENCE, run_stages_from
from .runtime import Runtime
from .state import PipelineState

logger = get_logger(__name__)


class SequentialPipelineRunner:
    """Minimal state machine: understand → decompose → retrieve → rank → assemble → generate → validate."""

    name = "sequential"

    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def run(self, state: PipelineState, start: Stage = Stage.QUERY_UNDERSTANDING) -> List[Any]:
        results = run_stages_from(start, state, self.runtime)
        self.runtime.trace.step(
            "pipeline",
            self.name,
            input_summary=state.query[:160],
            output_summary="; ".join(f"{r.stage.value}={'ok' if r.ok else 'error'}" for r in results),
            latency_ms=round(sum(r.latency_ms for r in results), 3),
            stages=[r.stage.value for r in results],
        )
        return results

    @staticmethod
    def stage_sequence() -> List[str]:
        return [s.value for s in STAGE_SEQUENCE]


class LangGraphPipelineRunner:
    """LangGraph-backed runner with the same observable behaviour.

    Falls back to the sequential runner if LangGraph is unavailable or the graph
    fails to compile, so behaviour is identical in both cases.
    """

    name = "langgraph"

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self._graph = None
        self._fallback = SequentialPipelineRunner(runtime)

    # ------------------------------------------------------------------
    def available(self) -> bool:
        try:
            import langgraph  # noqa: F401

            return True
        except Exception:
            return False

    def build(self):
        if self._graph is not None:
            return self._graph
        if not self.available():
            return None
        try:  # pragma: no cover - exercised only when langgraph is installed
            from langgraph.graph import END, START, StateGraph

            from .graph_state import LangGraphState

            graph = StateGraph(LangGraphState)
            graph.add_node("query_understanding", self._node_query_understanding)
            graph.add_node("decomposition", self._node_decomposition)
            graph.add_node("retrieval", self._node_retrieval)
            graph.add_node("ranking", self._node_ranking)
            graph.add_node("context_assembly", self._node_context_assembly)
            graph.add_node("generation", self._node_generation)
            graph.add_node("validation", self._node_validation)

            graph.add_edge(START, "query_understanding")
            graph.add_edge("query_understanding", "decomposition")
            graph.add_edge("decomposition", "retrieval")
            graph.add_edge("retrieval", "ranking")
            graph.add_edge("ranking", "context_assembly")
            graph.add_edge("context_assembly", "generation")
            graph.add_edge("generation", "validation")
            graph.add_edge("validation", END)
            self._graph = graph.compile()
            return self._graph
        except Exception as exc:  # pragma: no cover
            logger.warning("LangGraph unavailable (%s); using sequential runner", exc)
            return None

    # ------------------------------------------------------------------
    def _node(self, fn, state: PipelineState) -> Dict[str, Any]:
        result = fn(state, self.runtime)
        state.record_stage(result)
        return {"last_stage": result.stage.value, "ok": result.ok}

    def _node_query_understanding(self, state):  # pragma: no cover
        from .nodes import stage_query_understanding

        return self._node(stage_query_understanding, state)

    def _node_decomposition(self, state):  # pragma: no cover
        from .nodes import stage_decomposition

        return self._node(stage_decomposition, state)

    def _node_retrieval(self, state):  # pragma: no cover
        from .nodes import stage_retrieval as fn

        return self._node(lambda s, r: fn(s, r), state)

    def _node_ranking(self, state):  # pragma: no cover
        from .nodes import stage_ranking

        return self._node(stage_ranking, state)

    def _node_context_assembly(self, state):  # pragma: no cover
        from .nodes import stage_context_assembly

        return self._node(stage_context_assembly, state)

    def _node_generation(self, state):  # pragma: no cover
        from .nodes import stage_generation

        return self._node(stage_generation, state)

    def _node_validation(self, state):  # pragma: no cover
        from .nodes import stage_validation

        return self._node(stage_validation, state)

    def run(self, state: PipelineState, start: Stage = Stage.QUERY_UNDERSTANDING) -> List[Any]:  # pragma: no cover
        graph = self.build()
        if graph is None:
            return self._fallback.run(state, start)
        try:
            graph.invoke({"state": state}, config={"recursion_limit": 20})
            return list(state.stage_results)
        except Exception as exc:
            logger.warning("LangGraph execution failed (%s); using sequential runner", exc)
            return self._fallback.run(state, start)

    @staticmethod
    def stage_sequence() -> List[str]:
        return [s.value for s in STAGE_SEQUENCE]


def build_runner(runtime: Runtime, prefer: str = "auto") -> Any:
    """``auto`` uses LangGraph when installed, otherwise the sequential runner."""
    if prefer in {"sequential", "state_machine"}:
        return SequentialPipelineRunner(runtime)
    if prefer == "langgraph":
        return LangGraphPipelineRunner(runtime)
    graph_runner = LangGraphPipelineRunner(runtime)
    return graph_runner if graph_runner.available() else SequentialPipelineRunner(runtime)