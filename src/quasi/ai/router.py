"""
Routing, provider configuration, and orchestration engine for quasi.ai.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal, Optional

from quasi.ai.adapters import (
    AnthropicAdapter,
    BaseLLMAdapter,
    GeminiAdapter,
    MockAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from quasi.ai.condenser import condense_metadata_payload
from quasi.ai.prompts import (
    AUDIENCE_INSTRUCTIONS,
    DIAGNOSTIC_TROUBLESHOOTING_SYSTEM_PROMPT,
    ECONOMETRIC_SYSTEM_GUARDRAILS,
    QA_ASSISTANT_SYSTEM_PROMPT,
)


class AIConfiguration:
    """Global configuration state for quasi.ai."""
    provider: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    offline: bool = False
    _adapter: Optional[BaseLLMAdapter] = None

    @classmethod
    def get_adapter(cls) -> BaseLLMAdapter:
        if cls.offline:
            return MockAdapter()

        if cls._adapter is not None:
            return cls._adapter

        # Heuristic auto-detection based on environment variables
        if os.getenv("OPENAI_API_KEY"):
            cls._adapter = OpenAIAdapter(model=cls.model or "gpt-4o-mini", api_key=cls.api_key, base_url=cls.base_url)
        elif os.getenv("ANTHROPIC_API_KEY"):
            cls._adapter = AnthropicAdapter(model=cls.model or "claude-sonnet-4-20250514", api_key=cls.api_key, base_url=cls.base_url)
        elif os.getenv("GEMINI_API_KEY"):
            cls._adapter = GeminiAdapter(model=cls.model or "gemini-2.5-flash", api_key=cls.api_key, base_url=cls.base_url)
        else:
            # Default to deterministic mock adapter if no key configured
            cls._adapter = MockAdapter()

        return cls._adapter


def configure(
    provider: Literal["openai", "anthropic", "gemini", "ollama", "mock"] = "mock",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    offline: bool = False,
) -> None:
    """
    Configures the AI provider and credentials for quasi.ai.

    Examples:
        >>> quasi.ai.configure(provider="openai", model="gpt-4o-mini")
        >>> quasi.ai.configure(provider="anthropic", model="claude-3-5-sonnet-20241022")
        >>> quasi.ai.configure(provider="ollama", model="llama3.1:8b", base_url="http://localhost:11434")
    """
    AIConfiguration.provider = provider
    AIConfiguration.model = model
    AIConfiguration.api_key = api_key
    AIConfiguration.base_url = base_url
    AIConfiguration.temperature = temperature
    AIConfiguration.offline = offline

    if offline:
        AIConfiguration._adapter = MockAdapter()
    elif provider == "openai":
        AIConfiguration._adapter = OpenAIAdapter(model=model or "gpt-4o-mini", api_key=api_key, base_url=base_url)
    elif provider == "anthropic":
        AIConfiguration._adapter = AnthropicAdapter(model=model or "claude-sonnet-4-20250514", api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        AIConfiguration._adapter = GeminiAdapter(model=model or "gemini-2.5-flash", api_key=api_key, base_url=base_url)
    elif provider == "ollama":
        AIConfiguration._adapter = OllamaAdapter(model=model or "llama3.1:8b", base_url=base_url)
    elif provider == "mock":
        AIConfiguration._adapter = MockAdapter()
    else:
        raise ValueError(
            f"Unknown AI provider {provider!r}; expected one of 'openai', 'anthropic', 'gemini', 'ollama', 'mock'"
        )


def route_explanation_request(
    metadata: Dict[str, Any],
    audience: Literal["academic", "stakeholder", "student"] = "academic",
) -> str:
    """Dispatches result explanation request to active adapter with econometric guardrails."""
    adapter = AIConfiguration.get_adapter()
    condensed = condense_metadata_payload(metadata)

    system_prompt = (
        ECONOMETRIC_SYSTEM_GUARDRAILS
        + "\n"
        + AUDIENCE_INSTRUCTIONS.get(audience, AUDIENCE_INSTRUCTIONS["academic"])
    )

    user_prompt = (
        f"Please synthesize and explain the following quasi-experimental findings for a {audience} audience.\n\n"
        f"Statistical Model Metadata (JSON):\n{json.dumps(condensed, indent=2, allow_nan=False)}"
    )

    response = adapter.generate(system_prompt, user_prompt, temperature=AIConfiguration.temperature)
    return str(response)


def route_diagnosis_request(metadata: Dict[str, Any]) -> str:
    """Dispatches diagnostic troubleshooting request when assumption gates flag warnings or failures."""
    adapter = AIConfiguration.get_adapter()
    condensed = condense_metadata_payload(metadata)

    diags = condensed.get("diagnostics", [])
    compromised = [d for d in diags if d.get("status") in ("WARN", "FAIL")]

    if not compromised:
        return (
            "No executed diagnostic reported a warning or failure. "
            "This does not establish identification or validate untested assumptions; "
            "review study design and diagnostic coverage separately."
        )

    system_prompt = (
        ECONOMETRIC_SYSTEM_GUARDRAILS
        + "\n"
        + DIAGNOSTIC_TROUBLESHOOTING_SYSTEM_PROMPT
    )

    user_prompt = (
        f"The model '{condensed.get('model_name')}' flagged the following assumption check warnings/failures:\n"
        f"Compromised Gates:\n{json.dumps(compromised, indent=2)}\n\n"
        f"Full Model Summary:\n{json.dumps(condensed, indent=2)}\n\n"
        "Please provide an econometric diagnosis and actionable next steps."
    )

    response = adapter.generate(system_prompt, user_prompt, temperature=AIConfiguration.temperature)
    return str(response)


def ask_assistant(query: str) -> str:
    """Answers conversational questions about study design, method selection, or quasi syntax."""
    adapter = AIConfiguration.get_adapter()
    response = adapter.generate(QA_ASSISTANT_SYSTEM_PROMPT, query, temperature=AIConfiguration.temperature)
    return str(response)