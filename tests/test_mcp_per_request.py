"""Tests for per-connection session state (H2, v0.7.1).

H2 removed the module-level ``_session`` dict from
``carl_studio.mcp.server`` and routed session state through the bound
:class:`~carl_studio.mcp.connection.MCPServerConnection`. These tests pin
the new contract end-to-end:

* The module-level global is gone.
* Each connection has an independent :class:`MCPSession`.
* Authentication writes flow to the bound connection, not a singleton.
* The FastMCP ``Context`` seam is honoured when present.
* :func:`extract_session` is the canonical ServerSession lookup helper
  used by both elicitation and sampling.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# The per-request machinery lives entirely in pure-Python modules; the
# optional ``mcp`` extra is only needed for the full integration path.
pytest.importorskip("mcp", reason="optional carl-studio[mcp] extra not installed")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeConn:
    """Minimal stand-in for MCPServerConnection.

    Mirrors the ``session`` property + setter contract without dragging in
    the full FSM lifecycle. Real MCPServerConnection instances behave the
    same way; this stub keeps the unit tests focused on ownership
    semantics.
    """

    def __init__(self) -> None:
        from carl_studio.mcp.session import MCPSession

        self._session_val = MCPSession()
        self.connection_id = "fake-conn"

    @property
    def session(self) -> Any:
        return self._session_val

    @session.setter
    def session(self, value: Any) -> None:
        self._session_val = value

    def record_tool_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _bind(conn: Any) -> None:
    import carl_studio.mcp.server as srv

    srv.bind_connection(conn)


def _unbind() -> None:
    import carl_studio.mcp.server as srv

    srv.bind_connection(None)


# ---------------------------------------------------------------------------
# Module-global removal
# ---------------------------------------------------------------------------


class TestNoModuleLevelSessionGlobal(unittest.TestCase):
    def test_no_session_global_attribute(self) -> None:
        """``server._session`` must not exist — H2 removed it in v0.7.1."""
        import carl_studio.mcp.server as srv

        assert not hasattr(srv, "_session"), (
            "carl_studio.mcp.server._session was removed in v0.7.1; "
            "session state now lives on MCPServerConnection."
        )


# ---------------------------------------------------------------------------
# MCPSession helpers
# ---------------------------------------------------------------------------


class TestMCPSessionHelpers(unittest.TestCase):
    def test_mcp_session_with_helper_returns_new_instance(self) -> None:
        """``MCPSession.with_`` must return a fresh frozen snapshot."""
        from carl_studio.mcp.session import MCPSession

        original = MCPSession()
        updated = original.with_(jwt="abc", tier="paid", user_id="u1")

        # New instance, original untouched (frozen).
        assert updated is not original
        assert original.jwt == ""
        assert original.tier == "free"
        assert updated.jwt == "abc"
        assert updated.tier == "paid"
        assert updated.user_id == "u1"

    def test_mcp_session_jwt_expires_at_defaults_none(self) -> None:
        """New ``jwt_expires_at`` field defaults to ``None``."""
        from carl_studio.mcp.session import MCPSession

        session = MCPSession()
        assert session.jwt_expires_at is None

        replaced = session.with_(jwt_expires_at=123.456)
        assert replaced.jwt_expires_at == 123.456

    def test_extract_session_canonical_helper_exported(self) -> None:
        """The canonical helper is re-exported from ``carl_studio.mcp``."""
        from carl_studio.mcp import extract_session as public
        from carl_studio.mcp.session import extract_session as canonical

        assert public is canonical


# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------


class TestConnectionSessionIsolation(unittest.TestCase):
    def test_two_connections_have_isolated_session(self) -> None:
        """Writing on conn A must not affect conn B's session."""
        from carl_studio.mcp.connection import MCPServerConnection

        conn_a = MCPServerConnection()
        conn_b = MCPServerConnection()

        conn_a.session = conn_a.session.with_(
            jwt="jwt-a", tier="paid", user_id="user-a"
        )

        # Isolation: B still sees the default empty session.
        assert conn_b.session.jwt == ""
        assert conn_b.session.tier == "free"
        assert conn_b.session.user_id == ""

        # A's write stuck on A.
        assert conn_a.session.jwt == "jwt-a"
        assert conn_a.session.tier == "paid"
        assert conn_a.session.user_id == "user-a"

    def test_bound_connection_session_is_writable(self) -> None:
        """The session setter must atomically replace the snapshot."""
        from carl_studio.mcp.connection import MCPServerConnection
        from carl_studio.mcp.session import MCPSession

        conn = MCPServerConnection()
        first = conn.session
        assert first.jwt == ""

        replacement = MCPSession(jwt="x", tier="paid", user_id="u")
        conn.session = replacement
        assert conn.session is replacement

        # Setter rejects non-MCPSession values.
        with pytest.raises(TypeError, match="MCPSession"):
            conn.session = "not-a-session"  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tools read/write the bound connection's session
# ---------------------------------------------------------------------------


class TestToolsUseBoundConnection(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        _unbind()

    async def test_authenticate_tool_writes_to_connection_session(self) -> None:
        """``authenticate`` must persist session state on the bound conn."""
        import json

        import carl_studio.mcp.server as srv

        conn = _FakeConn()
        _bind(conn)

        mock_profile = MagicMock()
        mock_profile.tier = "paid"
        mock_profile.user_id = "user-xyz"

        with patch(
            "carl_studio.camp.fetch_camp_profile", return_value=mock_profile
        ), patch("carl_studio.settings.CARLSettings.load") as mock_settings:
            mock_settings.return_value = MagicMock(
                supabase_url="https://example.supabase.co"
            )
            result = json.loads(await srv.authenticate("hdr.body.sig"))

        assert result["authenticated"] is True
        # The bound connection got the write — not a module global.
        assert conn.session.jwt == "hdr.body.sig"
        assert conn.session.tier == "paid"
        assert conn.session.user_id == "user-xyz"
        assert conn.session.authenticated_at is not None

    async def test_tool_reads_session_via_get_bound_connection(self) -> None:
        """``get_tier_status`` must reflect the bound connection's session."""
        import json

        import carl_studio.mcp.server as srv

        conn = _FakeConn()
        conn.session = conn.session.with_(
            jwt="jwt.paid", tier="paid", user_id="user-42"
        )
        _bind(conn)

        result = json.loads(await srv.get_tier_status())

        assert result["tier"] == "paid"
        assert result["authenticated"] is True
        assert result["user_id"] == "user-42"
        assert result["features"]["sync.cloud"] is True


# ---------------------------------------------------------------------------
# Context DI preference
# ---------------------------------------------------------------------------


class TestContextStatePreference(unittest.TestCase):
    def test_context_state_session_preferred_over_bound_connection(self) -> None:
        """When ``ctx.state.session`` is set, ``_get_session`` prefers it."""
        import carl_studio.mcp.server as srv
        from carl_studio.mcp.session import MCPSession

        # Bound connection has a paid session.
        bound = _FakeConn()
        bound.session = bound.session.with_(
            jwt="bound-jwt", tier="paid", user_id="bound-user"
        )
        _bind(bound)
        try:
            # Context state carries a different (free) session.
            ctx_session = MCPSession(
                jwt="ctx-jwt", tier="free", user_id="ctx-user"
            )
            ctx = MagicMock()
            ctx.state.session = ctx_session

            resolved = srv._get_session(ctx)
            assert resolved is ctx_session
            assert resolved.jwt == "ctx-jwt"
            assert resolved.tier == "free"
            assert resolved.user_id == "ctx-user"
        finally:
            _unbind()

    def test_get_session_falls_back_to_empty_when_no_connection(self) -> None:
        """Without ``ctx`` and without a bound conn, a fresh empty session."""
        import carl_studio.mcp.server as srv
        from carl_studio.mcp.session import MCPSession

        _unbind()
        result = srv._get_session()
        assert isinstance(result, MCPSession)
        assert result.jwt == ""
        assert result.tier == "free"
        assert result.user_id == ""

    def test_get_session_ignores_context_without_session_attr(self) -> None:
        """A bare Context without ``state.session`` falls back to bound conn."""
        import carl_studio.mcp.server as srv

        bound = _FakeConn()
        bound.session = bound.session.with_(jwt="bound", tier="paid")
        _bind(bound)
        try:
            ctx = MagicMock()
            # Simulate absence of ``state.session`` — return something that's
            # not an MCPSession so the isinstance() check fails.
            ctx.state.session = object()

            resolved = srv._get_session(ctx)
            # Because state.session was not an MCPSession, we fall through
            # to the bound connection.
            assert resolved is bound.session
        finally:
            _unbind()


# ---------------------------------------------------------------------------
# elicitation + sampling share the canonical helper
# ---------------------------------------------------------------------------


class TestElicitationSamplingShareHelper(unittest.TestCase):
    def test_elicitation_uses_shared_extract_session(self) -> None:
        """``elicitation._extract_session`` is an alias for the canonical helper."""
        from carl_studio.mcp import elicitation
        from carl_studio.mcp.session import extract_session

        assert elicitation._extract_session is extract_session

    def test_sampling_uses_shared_extract_session(self) -> None:
        from carl_studio.mcp import sampling
        from carl_studio.mcp.session import extract_session

        assert sampling._extract_session is extract_session


# ---------------------------------------------------------------------------
# Smoke: extract_session against a None connection
# ---------------------------------------------------------------------------


class TestExtractSessionContract(unittest.TestCase):
    def test_extract_session_none_returns_none(self) -> None:
        from carl_studio.mcp.session import extract_session

        assert extract_session(None) is None

    def test_extract_session_prefers_override_hook(self) -> None:
        """The ``_session_override`` attribute wins when present."""
        from carl_studio.mcp.session import extract_session

        stub = object()

        class _Conn:
            _session_override = stub
            fastmcp = None

        assert extract_session(_Conn()) is stub


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# Silence unused-import warnings in certain configurations.
_ = asyncio
