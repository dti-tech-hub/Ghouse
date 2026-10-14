"""Provider abstraction for LLMs.

Every component of the system talks to :class:`LLMProvider` and never to a
concrete SDK.  ``MockProvider`` is deterministic and offline so the whole
project is demonstrable without any API key.
"""

from .base import (
    ChatMessage,
    CompletionResult,
    LLMError,
    LLMProvider,
    LLMResponseError,
    ProviderTimeout,
    Usage,
)
from .factory import available_providers, build_diagnoser_provider, build_provider
from .mock import MockProvider

__all__ = [
    "LLMProvider",
    "CompletionResult",
    "ChatMessage",
    "Usage",
    "LLMError",
    "LLMResponseError",
    "ProviderTimeout",
    "MockProvider",
    "build_provider",
    "build_diagnoser_provider",
    "available_providers",
]