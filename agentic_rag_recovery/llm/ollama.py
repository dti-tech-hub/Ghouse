"""Ollama (local models) provider adapter.

Uses the plain HTTP REST API so that no extra Python dependency is required.
Falls back cleanly with :class:`ProviderTimeout` / :class:`LLMError` when the
local daemon is not running - which the system treats as an infrastructure
failure (exactly the behaviour we want to be able to demo).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import httpx

from .base import ChatMessage, CompletionResult, LLMError, LLMProvider, ProviderTimeout, Usage, estimate_tokens


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ):
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, **kwargs)
        self.base_url = base_url.rstrip("/")

    def _complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResult:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"ollama timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unavailable at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"ollama returned invalid JSON: {exc}") from exc
        text = (data.get("message") or {}).get("content", "")
        return CompletionResult(
            text=text,
            raw=data,
            usage=Usage(
                prompt_tokens=int(data.get("prompt_eval_count") or sum(estimate_tokens(m.content) for m in messages)),
                completion_tokens=int(data.get("eval_count") or estimate_tokens(text)),
            ),
            model=data.get("model", self.model),
        )

    def health(self) -> Dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            models = [m.get("name") for m in (response.json().get("models") or [])] if response.is_success else []
            return {
                "provider": self.name,
                "model": self.model,
                "available": response.is_success,
                "models": models,
                "calls": self.call_count,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            return {"provider": self.name, "model": self.model, "available": False, "error": str(exc), "calls": self.call_count}

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0  # local models are free