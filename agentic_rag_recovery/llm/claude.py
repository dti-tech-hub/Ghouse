"""Anthropic (Claude) provider adapter.

Implemented directly against the Messages REST API with ``httpx`` so no vendor
SDK is required.  Marked optional: the project is fully functional without it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import httpx

from .base import ChatMessage, CompletionResult, LLMError, LLMProvider, ProviderTimeout, Usage, estimate_tokens

ANTHROPIC_PRICING: Dict[str, Any] = {
    "claude-3-5-haiku-latest": (0.00080, 0.00400),
    "claude-3-5-sonnet-latest": (0.00300, 0.01500),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-sonnet-4-20250514": (0.00300, 0.01500),
    "default": (0.00300, 0.01500),
}
DEFAULT_VERSION = "2023-06-01"


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        base_url: str = "https://api.anthropic.com/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        api_version: str = DEFAULT_VERSION,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs: Any,
    ):
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = api_key or os.environ.get(api_key_env)
        self.api_version = api_version

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "anthropic-version": self.api_version}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _complete(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResult:
        system = "\n".join(m.content for m in messages if m.role == "system")
        chat = [m.to_dict() for m in messages if m.role != "system"]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system:
            payload["system"] = system
        try:
            response = httpx.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise LLMError(f"claude HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"claude timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"claude request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"claude returned invalid JSON: {exc}") from exc

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or sum(estimate_tokens(m.content) for m in messages))
        completion_tokens = int(usage.get("output_tokens") or estimate_tokens(text))
        pin, pout = ANTHROPIC_PRICING.get(self.model, ANTHROPIC_PRICING["default"])
        return CompletionResult(
            text=text,
            raw=data,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            model=data.get("model", self.model),
            finish_reason=data.get("stop_reason", "end_turn"),
            cost=(prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout,
        )

    def generate_json(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        import json
        import re

        from .base import LLMResponseError

        augmented_system = ((system or "") + "\nRespond with a single valid JSON object and nothing else.").strip()
        result = self.generate(prompt, system=augmented_system, **kwargs)
        cleaned = re.sub(r"^```(?:json)?|```$", "", (result.text or "").strip(), flags=re.MULTILINE).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"invalid JSON from claude: {exc}") from None
        if not isinstance(payload, dict):
            raise LLMResponseError("diagnoser expected a JSON object")
        result.raw = {"parsed": payload, "text": result.text}
        return payload

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "available": bool(self.api_key),
            "calls": self.call_count,
        }

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pin, pout = ANTHROPIC_PRICING.get(self.model, ANTHROPIC_PRICING["default"])
        return (prompt_tokens / 1000.0) * pin + (completion_tokens / 1000.0) * pout