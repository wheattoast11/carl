"""Tests for carl_core.connection.registry."""
from __future__ import annotations

import gc
from typing import Any

import pytest

from carl_core.connection.base import BaseConnection
from carl_core.connection.registry import (
    ConnectionRegistry,
    get_registry,
    reset_registry,
)
from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)


_SPEC = ConnectionSpec(
    name="demo.registry",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.UTILITY,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
)


class _Conn(BaseConnection):
    spec = _SPEC

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)

    def _connect(self) -> None: ...
    def _close(self) -> None: ...


@pytest.fixture(autouse=True)
def fresh_registry() -> None:
    reset_registry()


def test_get_registry_returns_singleton() -> None:
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_reset_registry_replaces_singleton() -> None:
    r1 = get_registry()
    reset_registry()
    r2 = get_registry()
    assert r1 is not r2


def test_register_on_open_unregister_on_close() -> None:
    conn = _Conn()
    assert get_registry().count() == 0
    conn.open()
    assert get_registry().count() == 1
    assert get_registry().get(conn.connection_id) is conn
    conn.close()
    assert get_registry().count() == 0


def test_live_returns_all_open_connections() -> None:
    a = _Conn()
    b = _Conn()
    a.open()
    b.open()
    live = get_registry().live()
    assert {c.connection_id for c in live} == {a.connection_id, b.connection_id}
    a.close()
    b.close()


def test_get_unknown_returns_none() -> None:
    assert get_registry().get("not-a-real-id") is None


def test_close_all_closes_sync_connections() -> None:
    a = _Conn()
    b = _Conn()
    a.open()
    b.open()
    assert get_registry().count() == 2
    n = get_registry().close_all()
    assert n == 2
    assert get_registry().count() == 0


def test_weakref_gc_removes_dropped_connection() -> None:
    """A connection that was opened then dropped by all refs and GC'd
    should disappear from the registry automatically."""
    conn = _Conn()
    conn.open()
    conn_id = conn.connection_id
    assert get_registry().get(conn_id) is conn
    # Close first, which does unregister — then drop all refs.
    conn.close()
    del conn
    gc.collect()
    assert get_registry().get(conn_id) is None


def test_to_dict_snapshot() -> None:
    a = _Conn()
    a.open()
    snap = get_registry().to_dict()
    assert len(snap) == 1
    assert snap[0]["connection_id"] == a.connection_id
    assert snap[0]["state"] == "ready"
    a.close()


def test_fresh_registry_isolated() -> None:
    """Explicit ConnectionRegistry() instances are independent from the global."""
    private = ConnectionRegistry()
    assert private.count() == 0
    conn = _Conn()
    conn.open()
    # global grew; private did not
    assert get_registry().count() == 1
    assert private.count() == 0
    conn.close()
