"""Provider factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..settings import Settings, get_settings
from .base import LLMProvider
from .mock import MockProvider

PROVIDERS = ("mock", "ollama", "openai", "claude")


DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com/v1",
}


def build_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    **overrides: Any,
) -> LLMProvider:
    """Instantiate a provider from a name (``mock`` is the offline default)."""
    settings = settings or get_settings()
    name = (provider or settings.generation.provider or "mock").lower().strip()
    gen = settings.generation
    base_url = overrides.pop("base_url", None) or gen.base_url or ""
    kwargs: Dict[str, Any] = {
        "temperature": gen.temperature,
        "max_tokens": gen.max_tokens,
        "timeout": gen.request_timeout_s,
        "seed": settings.seed,
    }
    kwargs.update(overrides)

    if name in {"mock", "none", "offline", "demo"}:
        return MockProvider(model=model or gen.model, **{k: v for k, v in kwargs.items() if k != "model"})

    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(
            model=model or gen.model, base_url=base_url or DEFAULT_BASE_URLS["ollama"], **kwargs
        )

    if name in {"openai", "openai-compatible", "local", "vllm", "lmstudio"}:
        from .openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            model=model or gen.model,
            base_url=base_url or DEFAULT_BASE_URLS["openai"],
            api_key_env=gen.api_key_env,
            **kwargs,
        )

    if name in {"claude", "anthropic"}:
        from .claude import ClaudeProvider

        return ClaudeProvider(
            model=model or gen.model,
            base_url=base_url or DEFAULT_BASE_URLS["claude"],
            **kwargs,
        )

    raise ValueError(
        f"unknown LLM provider '{name}'. Available: {', '.join(PROVIDERS)}"
    )


def build_diagnoser_provider(
    settings: Optional[Settings] = None, **overrides: Any
) -> LLMProvider:
    """Provider used by the Tier-2 LLM diagnoser (defaults to generation)."""
    settings = settings or get_settings()
    provider = settings.diagnosis.llm_provider or settings.generation.provider
    model = settings.diagnosis.llm_model or settings.generation.model
    opts: Dict[str, Any] = {"temperature": settings.diagnosis.temperature}
    opts.update(overrides)
    return build_provider(provider, model, settings=settings, **opts)


def available_providers() -> Dict[str, bool]:
    """Probe which providers are actually usable in this environment."""
    out: Dict[str, bool] = {}
    for name in PROVIDERS:
        try:
            out[name] = bool(build_provider(name).health().get("available"))
        except Exception:
            out[name] = False
    return out