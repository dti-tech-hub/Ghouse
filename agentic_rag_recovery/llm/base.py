"""Abstract LLM provider interface."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class LLMError(RuntimeError):
    """Base class for provider errors (maps to INFRASTRUCTURE_FAILURE)."""

    failure_subtype = "model_failure"


class ProviderTimeout(LLMError):
    failure_subtype = "timeout"


class LLMResponseError(LLMError):
    """Provider returned unparseable / schema-violating output."""

    failure_subtype = "malformed_response"


class MalformedResponseError(LLMResponseError):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CompletionResult:
    text: str = ""
    raw: Optional[Dict[str, Any]] = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    finish_reason: str = "stop"
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class LLMProvider(abc.ABC):
    """Base class every provider adapter implements."""

    name: str = "base"

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.options: Dict[str, Any] = kwargs
        self.temperature: float = float(kwargs.get("temperature", 0.0))
        self.max_tokens: int = int(kwargs.get("max_tokens", 512))
        self.timeout: float = float(kwargs.get("timeout", 30.0))
        self._call_count = 0

    # -- interface -------------------------------------------------------
    @abc.abstractmethod
    def _complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResult:
        """Provider specific completion call."""

    def complete(
        self, messages: Sequence[ChatMessage] | str, **kwargs: Any
    ) -> CompletionResult:
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        msgs = list(messages)
        started = time.perf_counter()
        self._call_count += 1
        result = self._complete(msgs, **kwargs)
        result.latency_ms = result.latency_ms or (time.perf_counter() - started) * 1000.0
        result.model = result.model or self.model
        result.provider = self.name
        if not result.usage.prompt_tokens:
            result.usage.prompt_tokens = sum(estimate_tokens(m.content) for m in msgs)
        return result

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> CompletionResult:
        messages = []
        if system:
            messages.append(ChatMessage("system", system))
        messages.append(ChatMessage("user", prompt))
        return self.complete(messages, **kwargs)

    def generate_json(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Return parsed JSON. Raises :class:`LLMResponseError` on failure."""
        import json
        import re

        result = self.generate(prompt, system=system, **kwargs)
        text = (result.text or "").strip()
        if not text:
            raise LLMResponseError("empty completion from provider")
        cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise LLMResponseError("no JSON object found in completion") from None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMResponseError(f"invalid JSON from provider: {exc}") from None
        result.raw = {"parsed": payload, "text": text}
        if isinstance(payload, dict):
            payload.setdefault("_usage", {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            })
            payload.setdefault("_latency_ms", result.latency_ms)
        return payload

    # -- helpers ---------------------------------------------------------
    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_call_count(self) -> None:
        self._call_count = 0

    def health(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model, "available": True, "calls": self._call_count}

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model!r}>"