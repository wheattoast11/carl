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

import hashlib
import json
import os
import sqlite3
import time
import urllib.request

from carl_studio.settings import carl_home

__all__ = ["LLMProvider", "parse_llm_json", "resolve_provider"]

# ---------------------------------------------------------------------------
# Response cache (SQLite, best-effort)
# ---------------------------------------------------------------------------

_CACHE_DB = carl_home() / "llm_cache.db"
_CACHE_TTL_HOURS = 24


def _get_cache_db() -> sqlite3.Connection:
    """Open (and lazily create) the LLM response cache database."""
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def _cache_key(prompt: str, system: str, model: str, temperature: float) -> str:
    """Deterministic cache key from prompt parameters."""
    raw = f"{model}:{temperature}:{system}:{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    """Get cached response if within TTL."""
    try:
        conn = _get_cache_db()
        try:
            row = conn.execute(
                "SELECT response, created_at FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
            if row and (time.time() - row[1]) < _CACHE_TTL_HOURS * 3600:
                return row[0]
        finally:
            conn.close()
    except Exception:
        pass
    return None


def _cache_set(key: str, response: str, model: str) -> None:
    """Store response in cache."""
    try:
        conn = _get_cache_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response, model, created_at) VALUES (?, ?, ?, ?)",
                (key, response, model, time.time()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


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

    # -- public read-only properties ------------------------------------------

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def provider_name(self) -> str:
        if self._backend == "anthropic":
            return "Anthropic"
        if "openrouter" in self._base_url:
            return "OpenRouter"
        if "localhost" in self._base_url:
            return "Local"
        return "OpenAI"

    # -- core completion -----------------------------------------------------

    def complete(self, messages: list[dict], *, max_tokens: int = 8192) -> str:
        """Send completion. Returns content string."""
        if self._backend == "anthropic":
            return self._complete_anthropic(messages, max_tokens)
        return self._complete_openai(messages, max_tokens)

    # -- convenience wrappers with caching -----------------------------------

    @staticmethod
    def _build_messages(prompt: str, system: str) -> list[dict[str, str]]:
        """Build a messages list from a prompt and optional system string."""
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def cached_complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Complete with response caching.

        Same-prompt calls return the cached result within TTL.
        Cache failures are silently ignored — the call always succeeds
        (or raises from the underlying provider).
        """
        key = _cache_key(prompt, system, self._model, temperature)
        cached = _cache_get(key)
        if cached is not None:
            return cached

        result = self.complete(self._build_messages(prompt, system))
        try:
            _cache_set(key, result, self._model)
        except Exception:
            pass  # _cache_set catches internally; guard against mock replacement
        return result

    def batch_complete(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.0,
    ) -> list[str]:
        """Complete multiple prompts.

        Uses cache where available, sequential requests otherwise.
        """
        results: list[str] = []
        uncached_indices: list[int] = []

        # Check cache first
        for i, prompt in enumerate(prompts):
            key = _cache_key(prompt, system, self._model, temperature)
            cached = _cache_get(key)
            if cached is not None:
                results.append(cached)
            else:
                results.append("")  # placeholder
                uncached_indices.append(i)

        # Complete uncached prompts sequentially
        for i in uncached_indices:
            result = self.complete(self._build_messages(prompts[i], system))
            results[i] = result
            key = _cache_key(prompts[i], system, self._model, temperature)
            try:
                _cache_set(key, result, self._model)
            except Exception:
                pass

        return results

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


def resolve_provider() -> dict[str, str]:
    """Resolve the active LLM provider without constructing a client.

    Returns dict with keys: backend, model, base_url.
    Raises RuntimeError if no provider is available.
    """
    provider = LLMProvider.auto()
    return {
        "backend": provider.backend,
        "model": provider.model,
        "base_url": provider.base_url,
    }
