"""LangGraph state schema (only imported when LangGraph is installed)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from .state import PipelineState


class LangGraphState(BaseModel):
    """Thin wrapper so the graph can carry the rich :class:`PipelineState`.

    ``model_config = ConfigDict(arbitrary_types_allowed=True)`` lets the real
    dataclass travel through the graph unchanged, so both runners observe the
    same state object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: Optional[PipelineState] = None
    last_stage: Optional[str] = None
    ok: bool = True
    notes: List[str] = []

    def pipeline_state(self) -> PipelineState:
        if self.state is None:
            raise RuntimeError("LangGraphState has no PipelineState attached")
        return self.state