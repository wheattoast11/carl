"""Tests for the Phase 1 carl_core.connection port of the MCP surface.

Scope:
    * :class:`MCPServerConnection` spec/state/lifecycle round-trips.
    * :class:`MCPSession` typed view of the legacy ``_session`` dict.
    * ``bind_connection`` wires tool telemetry onto the chain.

Skips when the optional ``mcp`` package is not installed; we still test
the session dataclass round-trip because it has no ``mcp`` dependency.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pytest

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionState,
    ConnectionTransport,
    ConnectionTrust,
)
from carl_core.interaction import ActionType, InteractionChain

from carl_studio.mcp.session import MCPSession, session_from_dict, session_to_dict


# ---------------------------------------------------------------------------
# Session dataclass — no mcp dependency, always runs
# ---------------------------------------------------------------------------


class TestMCPSessionDataclass(unittest.TestCase):
    def test_unauthenticated_defaults(self) -> None:
        raw: dict[str, str] = {"jwt": "", "tier": "free", "user_id": ""}
        session = session_from_dict(raw)

        assert session.jwt == ""
        assert session.tier == "free"
        assert session.user_id == ""
        assert session.authenticated_at is None
        assert session.is_authenticated is False

    def test_authenticated_round_trip(self) -> None:
        epoch = 1_730_000_000
        raw: dict[str, str] = {
            "jwt": "hdr.body.sig",
            "tier": "paid",
            "user_id": "user-abc",
            "authenticated_at": str(epoch),
        }

        session = session_from_dict(raw)

        assert session.is_authenticated is True
        assert session.jwt == "hdr.body.sig"
        assert session.tier == "paid"
        assert session.user_id == "user-abc"
        assert session.authenticated_at is not None
        assert session.authenticated_at.tzinfo is timezone.utc
        assert int(session.authenticated_at.timestamp()) == epoch

        # Round-trip back to the legacy dict should be bit-equivalent.
        back = session_to_dict(session)
        assert back == raw

    def test_malformed_authenticated_at_degrades_to_none(self) -> None:
        raw: dict[str, str] = {
            "jwt": "hdr.body.sig",
            "tier": "paid",
            "user_id": "user-abc",
            "authenticated_at": "not-an-int",
        }

        session = session_from_dict(raw)
        assert session.authenticated_at is None
        # to_dict then renders the missing timestamp as empty string.
        back = session_to_dict(session)
        assert back["authenticated_at"] == ""

    def test_missing_keys_default_sanely(self) -> None:
        session = session_from_dict({})
        assert session.jwt == ""
        assert session.tier == "free"
        assert session.user_id == ""
        assert session.authenticated_at is None

    def test_session_is_frozen(self) -> None:
        session = session_from_dict({"jwt": "a", "tier": "paid", "user_id": "u"})
        with pytest.raises(Exception):  # FrozenInstanceError inherits from Exception
            session.jwt = "b"  # type: ignore[misc]

    def test_to_dict_preserves_datetime_epoch_exactly(self) -> None:
        # Datetime precision is whole seconds in the legacy format; make sure
        # sub-second fractions don't round away to a different epoch.
        dt = datetime(2026, 4, 19, 12, 0, 0, 500_000, tzinfo=timezone.utc)
        session = MCPSession(jwt="x", tier="paid", user_id="u", authenticated_at=dt)
        raw = session_to_dict(session)
        assert raw["authenticated_at"] == str(int(dt.timestamp()))


# ---------------------------------------------------------------------------
# Connection — requires the mcp package
# ---------------------------------------------------------------------------


pytest.importorskip("mcp", reason="optional carl-studio[mcp] extra not installed")

# Only import the connection module after the skip check; importing it
# indirectly touches carl_studio.mcp.__init__, which imports FastMCP.
from carl_studio.mcp.connection import MCPServerConnection  # noqa: E402


class TestMCPServerConnectionSpec(unittest.TestCase):
    def test_default_spec_attributes(self) -> None:
        conn = MCPServerConnection()
        spec = conn.spec

        assert spec.name == "carl.mcp.server"
        assert spec.scope == ConnectionScope.ONE_P
        assert spec.kind == ConnectionKind.PROTOCOL
        assert spec.direction == ConnectionDirection.INGRESS
        assert spec.transport == ConnectionTransport.STDIO
        assert spec.trust == ConnectionTrust.AUTHENTICATED
        assert spec.metadata["server_name"] == "carl-studio"
        assert spec.metadata["transport_choice"] == "stdio"

    def test_http_transport_override(self) -> None:
        conn = MCPServerConnection(transport="http")
        assert conn.spec.transport == ConnectionTransport.HTTP
        assert conn.transport_choice == "http"

    def test_invalid_transport_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported MCP transport"):
            MCPServerConnection(transport="ws")

    def test_initial_state_is_init(self) -> None:
        conn = MCPServerConnection()
        assert conn.state == ConnectionState.INIT
        assert conn.fastmcp is None


class TestMCPServerConnectionLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_open_close_transitions(self) -> None:
        conn = MCPServerConnection()
        assert conn.state == ConnectionState.INIT

        await conn.open()
        assert conn.state == ConnectionState.READY
        assert conn.fastmcp is not None

        await conn.close()
        assert conn.state == ConnectionState.CLOSED
        assert conn.fastmcp is None

    async def test_async_context_manager_and_transact(self) -> None:
        async with MCPServerConnection() as conn:
            assert conn.state == ConnectionState.READY
            async with conn.transact("ping"):
                assert conn.state == ConnectionState.TRANSACTING
            assert conn.state == ConnectionState.READY
        assert conn.state == ConnectionState.CLOSED

    async def test_transact_records_event_on_chain(self) -> None:
        chain = InteractionChain()
        async with MCPServerConnection(chain=chain) as conn:
            async with conn.transact("ping"):
                pass

        names = [s.name for s in chain.steps]
        # At minimum: connection.open, connection.ping, connection.close.
        assert "connection.open" in names
        assert "connection.ping" in names
        assert "connection.close" in names
        ping_step = next(s for s in chain.steps if s.name == "connection.ping")
        assert ping_step.success is True
        assert ping_step.duration_ms is not None
        assert ping_step.duration_ms >= 0

    async def test_session_property_owned_by_connection(self) -> None:
        """Each connection owns its :class:`MCPSession` (H2, v0.7.1)."""
        async with MCPServerConnection() as conn:
            snapshot = conn.session
            assert isinstance(snapshot, MCPSession)
            # Default snapshot is the empty / free-tier session.
            assert snapshot.tier == "free"
            assert snapshot.jwt == ""

            # Writing through the setter replaces the snapshot atomically.
            conn.session = conn.session.with_(jwt="j", tier="paid", user_id="u")
            assert conn.session.jwt == "j"
            assert conn.session.tier == "paid"
            assert conn.session.user_id == "u"


class TestBindConnectionToolTelemetry(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_records_connection_tool_event(self) -> None:
        """bind_connection + a tool invocation must record a
        ``connection.tool.<name>`` event on the chain.
        """
        from carl_studio.mcp import server as _server
        from carl_studio.mcp.server import bind_connection

        chain = InteractionChain()
        async with MCPServerConnection(chain=chain) as conn:
            bind_connection(conn)
            try:
                # list_backends is free, synchronous, no external deps.
                result = await _server.list_backends()
            finally:
                bind_connection(None)

        # Sanity: the tool still returned its normal JSON payload.
        assert '"backends"' in result

        # And the telemetry event landed on the chain.
        names = [s.name for s in chain.steps]
        assert "connection.tool.list_backends" in names
        evt = next(s for s in chain.steps if s.name == "connection.tool.list_backends")
        assert evt.success is True
        assert evt.action == ActionType.EXTERNAL

    async def test_tool_call_records_failure_event_on_exception(self) -> None:
        """Uncaught exceptions inside a tool must surface as success=False."""
        from carl_studio.mcp import server as _server
        from carl_studio.mcp.server import bind_connection

        async def _boom() -> str:
            raise RuntimeError("synthetic failure")

        chain = InteractionChain()
        async with MCPServerConnection(chain=chain) as conn:
            bind_connection(conn)
            try:
                # _run_tool is the module-private wrapper every tool body
                # goes through; the test exercises its failure path
                # directly because hitting it via a real tool would need
                # to stub out the tool's dependency chain.
                with pytest.raises(RuntimeError, match="synthetic failure"):
                    await _server._run_tool(  # pyright: ignore[reportPrivateUsage]
                        "unit_test_tool", _boom
                    )
            finally:
                bind_connection(None)

        evt = next(
            s for s in chain.steps if s.name == "connection.tool.unit_test_tool"
        )
        assert evt.success is False
        assert evt.input is not None

    async def test_unbound_tool_call_is_silent(self) -> None:
        """Without bind_connection, no telemetry is emitted (no crash, no event)."""
        from carl_studio.mcp import server as _server
        from carl_studio.mcp.server import bind_connection

        bind_connection(None)  # ensure detached
        # With no connection bound, calling a tool must still work.
        result = await _server.list_backends()
        assert '"backends"' in result


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
