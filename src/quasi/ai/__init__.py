"""Embedded statistical AI assistant and provider adapters for quasi."""

from __future__ import annotations

from quasi.ai.adapters import (
    AnthropicAdapter,
    BaseLLMAdapter,
    GeminiAdapter,
    LLMResponse,
    MockAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from quasi.ai.router import (
    AIConfiguration,
    ask_assistant,
    configure,
    route_diagnosis_request,
    route_explanation_request,
)


def ask(query: str) -> str:
    """Ask an econometric design or code question to the quasi assistant."""
    return ask_assistant(query)


__all__ = [
    "configure",
    "ask",
    "ask_assistant",
    "route_explanation_request",
    "route_diagnosis_request",
    "AIConfiguration",
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "MockAdapter",
    "LLMResponse",
]