"""Tests for LLM response cache, cached_complete, batch_complete, resolve_provider."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.llm import (
    LLMProvider,
    _cache_get,
    _cache_key,
    _cache_set,
    _get_cache_db,
    resolve_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect the cache DB to a temp directory for every test."""
    db_path = tmp_path / "llm_cache.db"
    monkeypatch.setattr("carl_studio.llm._CACHE_DB", db_path)


def _make_provider(model: str = "test-model") -> LLMProvider:
    """Create a provider with a mocked client so no network calls are made."""
    p = LLMProvider(backend="openai", api_key="fake", model=model)
    return p


# ---------------------------------------------------------------------------
# _cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("hello", "sys", "m", 0.0)
        k2 = _cache_key("hello", "sys", "m", 0.0)
        assert k1 == k2

    def test_differs_on_prompt(self):
        k1 = _cache_key("a", "sys", "m", 0.0)
        k2 = _cache_key("b", "sys", "m", 0.0)
        assert k1 != k2

    def test_differs_on_system(self):
        k1 = _cache_key("p", "sys1", "m", 0.0)
        k2 = _cache_key("p", "sys2", "m", 0.0)
        assert k1 != k2

    def test_differs_on_model(self):
        k1 = _cache_key("p", "s", "m1", 0.0)
        k2 = _cache_key("p", "s", "m2", 0.0)
        assert k1 != k2

    def test_differs_on_temperature(self):
        k1 = _cache_key("p", "s", "m", 0.0)
        k2 = _cache_key("p", "s", "m", 0.7)
        assert k1 != k2

    def test_is_hex_sha256(self):
        k = _cache_key("p", "s", "m", 0.0)
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# _cache_get / _cache_set roundtrip
# ---------------------------------------------------------------------------

class TestCacheGetSet:
    def test_get_missing_returns_none(self):
        assert _cache_get("nonexistent") is None

    def test_roundtrip(self):
        _cache_set("k1", "response-text", "model-x")
        assert _cache_get("k1") == "response-text"

    def test_overwrite(self):
        _cache_set("k1", "v1", "m")
        _cache_set("k1", "v2", "m")
        assert _cache_get("k1") == "v2"

    def test_expired_entry_returns_none(self, monkeypatch):
        _cache_set("k1", "old", "m")
        # Advance time past TTL
        future = time.time() + 25 * 3600
        monkeypatch.setattr("carl_studio.llm.time.time", lambda: future)
        # Also need time.time in _cache_get to return future
        assert _cache_get("k1") is None

    def test_cache_failure_get_returns_none(self, monkeypatch):
        """If the DB is broken, _cache_get returns None instead of raising."""
        monkeypatch.setattr(
            "carl_studio.llm._get_cache_db",
            MagicMock(side_effect=Exception("boom")),
        )
        assert _cache_get("any") is None

    def test_cache_failure_set_does_not_raise(self, monkeypatch):
        """If the DB is broken, _cache_set silently swallows the error."""
        monkeypatch.setattr(
            "carl_studio.llm._get_cache_db",
            MagicMock(side_effect=Exception("boom")),
        )
        _cache_set("k", "v", "m")  # should not raise


# ---------------------------------------------------------------------------
# LLMProvider.cached_complete
# ---------------------------------------------------------------------------

class TestCachedComplete:
    def test_returns_completion_and_caches(self):
        provider = _make_provider()
        provider.complete = MagicMock(return_value="result-1")

        out1 = provider.cached_complete("prompt-a", system="sys")
        assert out1 == "result-1"
        assert provider.complete.call_count == 1

        # Second call should hit cache — complete() not called again
        out2 = provider.cached_complete("prompt-a", system="sys")
        assert out2 == "result-1"
        assert provider.complete.call_count == 1

    def test_different_prompts_not_shared(self):
        provider = _make_provider()
        provider.complete = MagicMock(side_effect=["r1", "r2"])

        assert provider.cached_complete("p1") == "r1"
        assert provider.cached_complete("p2") == "r2"
        assert provider.complete.call_count == 2

    def test_cache_failure_still_completes(self, monkeypatch):
        """If cache is broken, the call still goes through to the provider."""
        provider = _make_provider()
        provider.complete = MagicMock(return_value="live")
        monkeypatch.setattr("carl_studio.llm._cache_get", MagicMock(return_value=None))
        monkeypatch.setattr(
            "carl_studio.llm._cache_set", MagicMock(side_effect=Exception("boom"))
        )
        assert provider.cached_complete("p") == "live"


# ---------------------------------------------------------------------------
# LLMProvider.batch_complete
# ---------------------------------------------------------------------------

class TestBatchComplete:
    def test_all_uncached(self):
        provider = _make_provider()
        provider.complete = MagicMock(side_effect=["a", "b", "c"])

        results = provider.batch_complete(["p1", "p2", "p3"])
        assert results == ["a", "b", "c"]
        assert provider.complete.call_count == 3

    def test_mixed_cached_and_uncached(self):
        provider = _make_provider()
        # Pre-populate cache for p1
        key = _cache_key("p1", "", "test-model", 0.0)
        _cache_set(key, "cached-a", "test-model")

        provider.complete = MagicMock(side_effect=["fresh-b"])

        results = provider.batch_complete(["p1", "p2"])
        assert results == ["cached-a", "fresh-b"]
        assert provider.complete.call_count == 1

    def test_all_cached(self):
        provider = _make_provider()
        for prompt, resp in [("p1", "r1"), ("p2", "r2")]:
            key = _cache_key(prompt, "sys", "test-model", 0.0)
            _cache_set(key, resp, "test-model")

        provider.complete = MagicMock()

        results = provider.batch_complete(["p1", "p2"], system="sys")
        assert results == ["r1", "r2"]
        provider.complete.assert_not_called()

    def test_empty_prompts(self):
        provider = _make_provider()
        provider.complete = MagicMock()
        assert provider.batch_complete([]) == []
        provider.complete.assert_not_called()

    def test_results_are_cached_for_next_call(self):
        provider = _make_provider()
        provider.complete = MagicMock(side_effect=["fresh"])

        provider.batch_complete(["p1"])
        # Now calling again should hit cache
        provider.complete.reset_mock()
        provider.complete.side_effect = Exception("should not be called")

        results = provider.batch_complete(["p1"])
        assert results == ["fresh"]


# ---------------------------------------------------------------------------
# LLMProvider properties
# ---------------------------------------------------------------------------

class TestProviderProperties:
    def test_backend(self):
        p = LLMProvider(backend="anthropic", api_key="k", model="m")
        assert p.backend == "anthropic"

    def test_model(self):
        p = LLMProvider(backend="openai", model="gpt-4o")
        assert p.model == "gpt-4o"

    def test_base_url(self):
        p = LLMProvider(base_url="http://localhost:1234/v1")
        assert p.base_url == "http://localhost:1234/v1"

    def test_base_url_default_empty(self):
        p = LLMProvider()
        assert p.base_url == ""


# ---------------------------------------------------------------------------
# resolve_provider
# ---------------------------------------------------------------------------

class TestResolveProvider:
    def test_returns_expected_structure(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Clear any overrides that would take priority
        monkeypatch.delenv("CARL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CARL_LLM_MODEL", raising=False)

        result = resolve_provider()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"backend", "model", "base_url"}
        assert result["backend"] == "anthropic"
        assert result["model"] == "claude-sonnet-4-6"
        assert result["base_url"] == ""

    def test_openrouter(self, monkeypatch):
        monkeypatch.delenv("CARL_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CARL_LLM_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        result = resolve_provider()
        assert result["backend"] == "openai"
        assert "openrouter" in result["base_url"]

    def test_raises_when_no_provider(self, monkeypatch):
        for var in [
            "CARL_LLM_BASE_URL", "CARL_LLM_API_KEY", "CARL_LLM_MODEL",
            "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)
        # Patch out local probing to avoid hitting real ports
        monkeypatch.setattr("carl_studio.llm._probe_local", lambda port, timeout=0.5: False)

        with pytest.raises(RuntimeError, match="No LLM provider found"):
            resolve_provider()


# ---------------------------------------------------------------------------
# _build_messages helper
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_prompt_only(self):
        msgs = LLMProvider._build_messages("hello", "")
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_with_system(self):
        msgs = LLMProvider._build_messages("hello", "be helpful")
        assert msgs == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
