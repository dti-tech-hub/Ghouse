"""OpenAI-compatible provider adapter.

Works with OpenAI itself and with every server that exposes
``POST {base_url}/chat/completions`` (vLLM, LM Studio, llama.cpp server,
Ollama's OpenAI shim, Together, Groq, OpenRouter, ...).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .base import ChatMessage, CompletionResult, LLMError, LLMProvider, ProviderTimeout, Usage, estimate_tokens

# USD per 1k tokens (prompt, completion). Only used for cost accounting.
PRICING: Dict[str, Any] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4.1-mini": (0.00040, 0.00160),
    "gpt-4.1": (0.00200, 0.00800),
    "llama-3.1-8b-instruct": (0.0, 0.0),
    "default": (0.0, 0.0),
}


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.response_format = response_format

    @property
    def is_local(self) -> bool:
        host = self.base_url.split("//")[-1].split(":")[0]
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"} or host.endswith(".local")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _complete(
        self, messages: Sequence[ChatMessage], json_mode: bool = False, **kwargs: Any
    ) -> CompletionResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if json_mode or self.response_format:
            payload["response_format"] = self.response_format or {"type": "json_object"}
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise LLMError(f"{self.name} HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{self.name} timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"{self.name} returned invalid JSON: {exc}") from exc

        try:
            choice = (data.get("choices") or [])[0]
            text = (choice.get("message") or {}).get("content") or ""
        except (IndexError, AttributeError) as exc:
            raise LLMError(f"{self.name} response missing choices: {exc}") from exc
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or sum(estimate_tokens(m.content) for m in messages))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
        pin, pout = PRICING.get(self.model, PRICING["default"])
        cost = (prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout
        return CompletionResult(
            text=text,
            raw=data,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
            cost=cost,
        )

    def generate_json(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        result = self.generate(prompt, system=system, json_mode=True, **kwargs)
        return _parse_json(result)

    def health(self) -> Dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/models", headers=self._headers(), timeout=5.0)
            ok = response.status_code < 400
            return {
                "provider": self.name,
                "model": self.model,
                "available": ok,
                "configured": bool(self.api_key) or self.is_local,
                "calls": self.call_count,
            }
        except Exception as exc:  # pragma: no cover - network dependent
            return {"provider": self.name, "model": self.model, "available": False, "error": str(exc), "calls": self.call_count}

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = PRICING.get(self.model, PRICING["default"])
        return (prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout


def _parse_json(result: CompletionResult) -> Dict[str, Any]:
    import json
    import re

    from .base import LLMResponseError

    text = (result.text or "").strip()
    if not text:
        raise LLMResponseError("empty completion from provider")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"invalid JSON from {result.provider}: {exc}") from None
    if not isinstance(payload, dict):
        raise LLMResponseError("diagnoser expected a JSON object")
    result.raw = {"parsed": payload, "text": text}
    return payload


__all__ = ["OpenAICompatibleProvider", "PRICING"]

# keep unused import lint-clean while documenting intent
_ = List