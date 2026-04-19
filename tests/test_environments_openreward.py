"""Tests for OpenRewardClient — PAID tier gating + mocked httpx transport."""

from __future__ import annotations

from typing import Any

import pytest

from carl_core.connection import (
    ConnectionPolicyError,
    ConnectionState,
    ConnectionUnavailableError,
    reset_registry,
)
from carl_core.tier import Tier

from carl_studio.environments import openreward as openreward_mod
from carl_studio.environments.openreward import (
    OPENREWARD_CONNECTION_SPEC,
    OpenRewardClient,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_conn_registry():  # pyright: ignore[reportUnusedFunction]
    reset_registry()
    yield
    reset_registry()


class _FakeAsyncResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class _FakeAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._score_payload = {"score": 0.87}

    async def post(self, path: str, json: dict[str, Any]) -> _FakeAsyncResponse:
        self.posts.append((path, json))
        if path == "/score":
            return _FakeAsyncResponse(self._score_payload)
        return _FakeAsyncResponse({}, status=404)

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Connection spec
# ---------------------------------------------------------------------------


class TestConnectionSpec:
    def test_is_metered(self):
        assert OPENREWARD_CONNECTION_SPEC.trust.value == "metered"

    def test_is_three_p_http(self):
        assert OPENREWARD_CONNECTION_SPEC.scope.value == "3p"
        assert OPENREWARD_CONNECTION_SPEC.transport.value == "http"

    def test_kind_model(self):
        assert OPENREWARD_CONNECTION_SPEC.kind.value == "model"


# ---------------------------------------------------------------------------
# Tier gate
# ---------------------------------------------------------------------------


class TestTierGate:
    @pytest.mark.asyncio
    async def test_free_tier_blocks_open(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.FREE)
        client = OpenRewardClient(api_key="sk-test")
        with pytest.raises(ConnectionPolicyError) as excinfo:
            await client.open()
        exc = excinfo.value
        assert exc.code == "carl.connection.policy.tier"
        assert exc.context["feature"] == "connection.openreward"
        assert exc.context["tier_required"] == "paid"

    @pytest.mark.asyncio
    async def test_paid_tier_allows_open(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        # Patch httpx.AsyncClient inside the module's lazy import target.
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = OpenRewardClient(api_key="sk-test", base_url="http://fake.local")
        await client.open()
        try:
            assert client.state == ConnectionState.READY
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Auth / transport plumbing
# ---------------------------------------------------------------------------


class TestAuth:
    @pytest.mark.asyncio
    async def test_missing_api_key_blocks_open(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        monkeypatch.delenv("OPENREWARD_API_KEY", raising=False)
        # Patch the transport even if we never reach it.
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = OpenRewardClient(api_key=None, base_url="http://fake.local")
        with pytest.raises(ConnectionUnavailableError):
            await client.open()

    @pytest.mark.asyncio
    async def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        monkeypatch.setenv("OPENREWARD_API_KEY", "sk-env")
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = OpenRewardClient(base_url="http://fake.local")
        await client.open()
        try:
            assert client.state == ConnectionState.READY
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_missing_base_url_blocks_open(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        client = OpenRewardClient(api_key="sk-test", base_url="")
        with pytest.raises(ConnectionUnavailableError):
            await client.open()

    def test_auth_headers_include_bearer(self):
        client = OpenRewardClient(api_key="sk-test")
        headers = client._auth_headers()  # pyright: ignore[reportPrivateUsage]
        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    @pytest.mark.asyncio
    async def test_score_round_trip(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = OpenRewardClient(api_key="sk-test", base_url="http://fake.local")
        await client.open()
        try:
            value = await client.score(
                "the answer is 42", reference="42", rubric="exact",
            )
            assert abs(value - 0.87) < 1e-9
            fake = client._client  # pyright: ignore[reportPrivateUsage]
            assert fake.posts
            path, body = fake.posts[0]
            assert path == "/score"
            assert body["response"] == "the answer is 42"
            assert body["reference"] == "42"
            assert body["rubric"] == "exact"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_score_clamps_to_unit_interval(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        import httpx

        class _OverClient(_FakeAsyncClient):
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                self._score_payload = {"score": 7.5}

        monkeypatch.setattr(httpx, "AsyncClient", _OverClient)
        client = OpenRewardClient(api_key="sk-test", base_url="http://fake.local")
        await client.open()
        try:
            assert await client.score("x") == 1.0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_score_before_open_raises(self):
        client = OpenRewardClient(api_key="sk-test", base_url="http://fake.local")
        with pytest.raises(ConnectionUnavailableError):
            await client.score("x")

    @pytest.mark.asyncio
    async def test_score_rejects_non_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(openreward_mod, "_resolve_tier", lambda: Tier.PAID)
        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
        client = OpenRewardClient(api_key="sk-test", base_url="http://fake.local")
        await client.open()
        try:
            with pytest.raises(TypeError):
                await client.score(42)  # type: ignore[arg-type]
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# _resolve_tier fallback semantics
# ---------------------------------------------------------------------------


class TestResolveTier:
    def test_returns_free_when_settings_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        # Force the import path inside _resolve_tier to blow up.
        def _boom(*a: Any, **k: Any) -> None:  # noqa: ARG001
            raise RuntimeError("settings unavailable")

        import carl_studio.tier as _tier_mod

        monkeypatch.setattr(_tier_mod, "detect_effective_tier", _boom)
        assert openreward_mod._resolve_tier() == Tier.FREE  # pyright: ignore[reportPrivateUsage]
