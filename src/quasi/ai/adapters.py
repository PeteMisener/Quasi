"""
Zero-dependency HTTP clients for LLM providers.
Supports OpenAI, Anthropic Claude, Google Gemini, Ollama, and Mock testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


def _validate_http_url(url: str) -> urllib.parse.ParseResult:
    """Validate that a configured adapter endpoint uses http/https with a host."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Unsupported or invalid adapter base_url {url!r}; must use http:// or https://")
    return parsed


def _is_loopback_host(url: str) -> bool:
    parsed = _validate_http_url(url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


@dataclass
class LLMResponse:
    """Standardized response container returned by any provider adapter."""
    content: str
    model: str
    provider: str
    raw: Dict[str, Any]

    def __str__(self) -> str:
        return self.content


class BaseLLMAdapter(ABC):
    """Abstract adapter interface for LLM completions."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        pass


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI, Azure, vLLM, and OpenAI-compatible API gateways."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        super().__init__(model=model, api_key=key, base_url=url)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        is_local = _is_loopback_host(self.base_url)
        if not self.api_key and not is_local:
            raise ValueError(
                "OpenAI API key not found. Configure via `quasi.ai.configure(provider='openai', api_key=...)` "
                "or set the OPENAI_API_KEY environment variable."
            )

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["choices"][0]["message"]["content"]
                return LLMResponse(content=text, model=self.model, provider="openai", raw=res_data)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error ({e.code}): {err_body}") from e


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        url = base_url or "https://api.anthropic.com/v1"
        super().__init__(model=model, api_key=key, base_url=url)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Configure via `quasi.ai.configure(provider='anthropic', api_key=...)` "
                "or set the ANTHROPIC_API_KEY environment variable."
            )

        endpoint = f"{self.base_url.rstrip('/')}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1200,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["content"][0]["text"]
                return LLMResponse(content=text, model=self.model, provider="anthropic", raw=res_data)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error ({e.code}): {err_body}") from e


class GeminiAdapter(BaseLLMAdapter):
    """Adapter for Google Gemini Developer API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        super().__init__(model=model, api_key=key, base_url=url)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Configure via `quasi.ai.configure(provider='gemini', api_key=...)` "
                "or set the GEMINI_API_KEY environment variable."
            )

        endpoint = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"temperature": temperature},
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return LLMResponse(content=text, model=self.model, provider="gemini", raw=res_data)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error ({e.code}): {err_body}") from e


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for local Ollama instances (air-gapped / secure on-premise execution)."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        super().__init__(model=model, api_key=None, base_url=url)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        _validate_http_url(self.base_url)
        endpoint = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["message"]["content"]
                return LLMResponse(content=text, model=self.model, provider="ollama", raw=res_data)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API error ({e.code}): {err_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to local Ollama service at '{self.base_url}'. "
                "Ensure Ollama is running (`ollama serve`)."
            ) from e


class MockAdapter(BaseLLMAdapter):
    """Deterministic mock adapter for unit tests and zero-network demonstrations."""

    def __init__(self, model: str = "mock-econometrician-v1") -> None:
        super().__init__(model=model, api_key="mock-key", base_url="mock://")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        # Generate simulated econometric interpretation
        if "DOCUMENTATION" in system_prompt.upper() or "documentation assistant" in system_prompt.lower():
            text = (
                "[quasi Assistant Guidance]\n"
                "To estimate this causal design, build an explicit DesignSpec and pass it to `quasi.estimate`:\n"
                "```python\n"
                "import quasi\n"
                "spec = quasi.PanelDesign(outcome='y', unit='unit', time='date', treatment='treated')\n"
                "result = quasi.estimate(df, design=spec)\n"
                "print(result.summary())\n"
                "result.validate()\n"
                "```\n"
                "This runs fail-closed data validation before invoking the selected estimator."
            )
        elif "DIAGNOSTIC" in system_prompt.upper() or "troubleshooter" in system_prompt.lower():
            text = (
                "[AI Diagnostic Report]\n"
                "• Identified Risk: Evaluation of diagnostic gates indicates potential assumption sensitivity.\n"
                "• Mechanism: If parallel trends or sorting tests exhibit marginal p-values, point estimates "
                "may confound genuine policy impact with underlying selection dynamics.\n"
                "• Recommended Actions:\n"
                "  1. Run `res.robustness()` to test specification stability across alternate bandwidths or windows.\n"
                "  2. For DiD: Run `res.sensitivity_analysis()` (Rambachan-Roth HonestDiD) to evaluate parallel-trends sensitivity.\n"
                "  3. For RDD: Run `res.donut_sensitivity()` to inspect sensitivity near the cutoff."
            )
        else:
            text = (
                "[AI Statistical Synthesis]\n"
                "The intervention yielded an estimated causal effect of the reported magnitude with bounded "
                "standard errors. Baseline diagnostics show no statistically detectable divergence prior to the "
                "intervention, providing empirical support for the underlying identification strategy within the "
                "observed study window."
            )

        return LLMResponse(content=text, model=self.model, provider="mock", raw={"mock": True})