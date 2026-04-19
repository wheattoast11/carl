"""Tests for carl_core.connection.base — sync + async connection lifecycle."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from carl_core.connection.base import AsyncBaseConnection, BaseConnection
from carl_core.connection.errors import (
    ConnectionClosedError,
    ConnectionTransitionError,
    ConnectionUnavailableError,
)
from carl_core.connection.lifecycle import ConnectionState
from carl_core.connection.registry import reset_registry
from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)
from carl_core.interaction import ActionType, InteractionChain


_PUBLIC_SPEC = ConnectionSpec(
    name="demo.public",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.UTILITY,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
)


_AUTH_SPEC = ConnectionSpec(
    name="demo.auth",
    scope=ConnectionScope.THREE_P,
    kind=ConnectionKind.PROTOCOL,
    direction=ConnectionDirection.EGRESS,
    transport=ConnectionTransport.HTTP,
    trust=ConnectionTrust.AUTHENTICATED,
)


class _EchoConn(BaseConnection):
    """Minimal sync connection — records call counts for assertions."""

    spec = _PUBLIC_SPEC

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.connect_calls = 0
        self.handshake_calls = 0
        self.auth_calls = 0
        self.close_calls = 0

    def _connect(self) -> None:
        self.connect_calls += 1

    def _handshake(self) -> None:
        self.handshake_calls += 1

    def _authenticate(self) -> None:
        self.auth_calls += 1

    def _close(self) -> None:
        self.close_calls += 1


class _AuthConn(_EchoConn):
    spec = _AUTH_SPEC


class _NoSpecConn(BaseConnection):  # type: ignore[misc]
    """Declared without a class-level spec — __init__ must reject."""

    def _connect(self) -> None: ...
    def _close(self) -> None: ...


class _ConnectFails(BaseConnection):
    spec = _PUBLIC_SPEC

    def _connect(self) -> None:
        raise ConnectionUnavailableError(
            "transport down", context={"remote": "example.com"}
        )

    def _close(self) -> None: ...


class _CloseRaises(BaseConnection):
    spec = _PUBLIC_SPEC

    def _connect(self) -> None: ...

    def _close(self) -> None:
        raise RuntimeError("close explode")


@pytest.fixture(autouse=True)
def fresh_registry() -> None:
    reset_registry()


# -- synchronous ---------------------------------------------------------


def test_initial_state_is_init() -> None:
    conn = _EchoConn()
    assert conn.state == ConnectionState.INIT
    assert conn.connection_id.startswith("conn-")


def test_missing_spec_raises() -> None:
    with pytest.raises(TypeError, match="spec: ConnectionSpec"):
        _NoSpecConn()  # type: ignore[abstract]


def test_public_open_skips_authentication() -> None:
    conn = _EchoConn()
    conn.open()
    assert conn.state == ConnectionState.READY
    assert conn.connect_calls == 1
    assert conn.handshake_calls == 1
    assert conn.auth_calls == 0  # PUBLIC — auth skipped


def test_auth_open_runs_authentication() -> None:
    conn = _AuthConn()
    conn.open()
    assert conn.state == ConnectionState.READY
    assert conn.auth_calls == 1


def test_open_twice_raises_transition() -> None:
    conn = _EchoConn()
    conn.open()
    with pytest.raises(ConnectionTransitionError) as ei:
        conn.open()
    assert "ready" in ei.value.context["from"]


def test_close_idempotent() -> None:
    conn = _EchoConn()
    conn.open()
    conn.close()
    # A second close must not raise.
    conn.close()
    assert conn.state == ConnectionState.CLOSED
    # _close is called at least once; additional no-op closes don't retrigger it.
    assert conn.close_calls >= 1


def test_context_manager_opens_and_closes() -> None:
    with _EchoConn() as conn:
        assert conn.state == ConnectionState.READY
    assert conn.state == ConnectionState.CLOSED


def test_failed_open_lands_on_error() -> None:
    conn = _ConnectFails()
    with pytest.raises(ConnectionUnavailableError):
        conn.open()
    assert conn.state == ConnectionState.ERROR
    assert conn.last_error is not None


def test_close_after_error_reaches_closed() -> None:
    conn = _ConnectFails()
    with pytest.raises(ConnectionUnavailableError):
        conn.open()
    conn.close()
    assert conn.state == ConnectionState.CLOSED


def test_close_swallows_close_hook_error() -> None:
    conn = _CloseRaises()
    conn.open()
    # Must not raise even though _close raises.
    conn.close()
    assert conn.state == ConnectionState.CLOSED
    assert isinstance(conn.last_error, RuntimeError)


def test_require_ready_rejects_non_ready_states() -> None:
    conn = _EchoConn()
    with pytest.raises(ConnectionClosedError):
        conn.require_ready()
    conn.open()
    conn.require_ready()
    conn.close()
    with pytest.raises(ConnectionClosedError):
        conn.require_ready()


def test_transact_happy_path_returns_to_ready() -> None:
    conn = _EchoConn()
    conn.open()
    with conn.transact("echo"):
        assert conn.state == ConnectionState.TRANSACTING
    assert conn.state == ConnectionState.READY
    assert conn.stats.transactions_ok == 1


def test_transact_error_drops_to_degraded() -> None:
    conn = _EchoConn()
    conn.open()
    with pytest.raises(ValueError):
        with conn.transact("echo"):
            raise ValueError("nope")
    assert conn.state == ConnectionState.DEGRADED
    assert conn.stats.transactions_err == 1


def test_transact_requires_ready() -> None:
    conn = _EchoConn()
    with pytest.raises(ConnectionClosedError):
        with conn.transact("echo"):
            pass


def test_stats_snapshot_is_copy() -> None:
    conn = _EchoConn()
    conn.open()
    s1 = conn.stats
    assert s1.opens == 1
    # Mutating snapshot should not affect the underlying counter.
    s1.opens = 999
    assert conn.stats.opens == 1


def test_to_dict_has_expected_shape() -> None:
    conn = _EchoConn()
    conn.open()
    d = conn.to_dict()
    assert d["state"] == "ready"
    assert d["spec"]["name"] == "demo.public"
    assert d["opened_at"] is not None
    assert d["stats"]["opens"] == 1
    assert d["connection_id"] == conn.connection_id


def test_telemetry_records_on_chain() -> None:
    chain = InteractionChain()
    conn = _EchoConn(chain=chain)
    conn.open()
    with conn.transact("echo"):
        pass
    conn.close()
    names = [s.name for s in chain.steps]
    assert "connection.open" in names
    assert "connection.echo" in names
    assert "connection.close" in names
    # All steps are EXTERNAL-typed until we extend ActionType
    assert all(s.action == ActionType.EXTERNAL for s in chain.steps)


def test_explicit_connection_id_honored() -> None:
    conn = _EchoConn(connection_id="fixed-abc")
    assert conn.connection_id == "fixed-abc"


# -- asynchronous --------------------------------------------------------


class _AEcho(AsyncBaseConnection):
    spec = _PUBLIC_SPEC

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.connect_calls = 0
        self.close_calls = 0

    async def _connect(self) -> None:
        self.connect_calls += 1

    async def _close(self) -> None:
        self.close_calls += 1


def test_async_open_close_lifecycle() -> None:
    async def run() -> None:
        conn = _AEcho()
        await conn.open()
        assert conn.state == ConnectionState.READY
        await conn.close()
        assert conn.state == ConnectionState.CLOSED
    asyncio.run(run())


def test_async_context_manager() -> None:
    async def run() -> None:
        async with _AEcho() as conn:
            assert conn.state == ConnectionState.READY
        assert conn.state == ConnectionState.CLOSED
    asyncio.run(run())


def test_async_transact_happy_path() -> None:
    async def run() -> None:
        conn = _AEcho()
        await conn.open()
        async with conn.transact("op"):
            assert conn.state == ConnectionState.TRANSACTING
        assert conn.state == ConnectionState.READY
        assert conn.stats.transactions_ok == 1
        await conn.close()
    asyncio.run(run())


def test_async_transact_error_drops_to_degraded() -> None:
    async def run() -> None:
        conn = _AEcho()
        await conn.open()
        with pytest.raises(RuntimeError):
            async with conn.transact("op"):
                raise RuntimeError("boom")
        assert conn.state == ConnectionState.DEGRADED
        assert conn.stats.transactions_err == 1
        await conn.close()
    asyncio.run(run())
