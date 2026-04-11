"""Provider-agnostic LLM interface with auto-detection.

Auto-detection priority:
  1. CARL_LLM_BASE_URL + CARL_LLM_API_KEY (explicit override)
  2. ANTHROPIC_API_KEY → Anthropic native SDK
  3. OPENROUTER_API_KEY → OpenRouter (OpenAI-compatible)
  4. OPENAI_API_KEY → OpenAI-compatible
  5. Probe localhost:11434 → Ollama
  6. Probe localhost:1234 → LM Studio
"""
from __future__ import annotations

import json
import os
import urllib.request

__all__ = ["LLMProvider", "parse_llm_json"]


def _probe_local(port: int, timeout: float = 0.5) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=timeout)
        return True
    except Exception:
        return False


class LLMProvider:
    """Unified LLM. OpenAI-compatible by default, Anthropic native when detected."""

    def __init__(self, *, backend: str = "openai", base_url: str = "", api_key: str = "", model: str = ""):
        self._backend = backend
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._client = None

    @classmethod
    def auto(cls) -> LLMProvider:
        base_url = os.environ.get("CARL_LLM_BASE_URL")
        api_key = os.environ.get("CARL_LLM_API_KEY")
        model = os.environ.get("CARL_LLM_MODEL")

        if base_url:
            return cls(backend="openai", base_url=base_url, api_key=api_key or "not-needed", model=model or "")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            return cls(backend="anthropic", api_key=anthropic_key, model=model or "claude-sonnet-4-6")

        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            return cls(backend="openai", base_url="https://openrouter.ai/api/v1", api_key=or_key, model=model or "anthropic/claude-sonnet-4-6")

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return cls(backend="openai", api_key=openai_key, model=model or "gpt-4o")

        if _probe_local(11434):
            return cls(backend="openai", base_url="http://localhost:11434/v1", api_key="ollama", model=model or "qwen3.5:9b")
        if _probe_local(1234):
            return cls(backend="openai", base_url="http://localhost:1234/v1", api_key="lm-studio", model=model or "local")

        raise RuntimeError(
            "No LLM provider found. Set one of: CARL_LLM_BASE_URL, "
            "ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, "
            "or start a local model server (Ollama/LM Studio)."
        )

    @property
    def provider_name(self) -> str:
        if self._backend == "anthropic":
            return "Anthropic"
        if "openrouter" in self._base_url:
            return "OpenRouter"
        if "localhost" in self._base_url:
            return "Local"
        return "OpenAI"

    def complete(self, messages: list[dict], *, max_tokens: int = 8192) -> str:
        """Send completion. Returns content string."""
        if self._backend == "anthropic":
            return self._complete_anthropic(messages, max_tokens)
        return self._complete_openai(messages, max_tokens)

    def _complete_openai(self, messages: list[dict], max_tokens: int) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  # required for OpenAI-compatible providers")

        if self._client is None:
            kwargs: dict = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)

        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _complete_anthropic(self, messages: list[dict], max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic  # required for Anthropic provider")

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)

        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)

        params: dict = {"model": self._model, "messages": chat_messages, "max_tokens": max_tokens}
        if system:
            params["system"] = system

        resp = self._client.messages.create(**params)
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


def parse_llm_json(text: str) -> dict | None:
    """Parse JSON from LLM response, stripping markdown fences."""
    import json
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
