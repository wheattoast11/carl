"""Tests for A2A connection wrappers over ``carl_core.connection``.

Covers:
  - :class:`ProtocolConnection` spec defaults
  - :class:`A2APeerConnection` spec derivation from :class:`CARLAgentCard`
  - Full state-machine round-trip (INIT -> ... -> READY -> TRANSACTING -> READY -> CLOSED)
  - InteractionChain emission for ``connection.open`` / ``connection.<op>``
    / ``connection.close``
  - :class:`A2AServerConnection` dispatch path (agent.card, message.send,
    tasks.get, unknown method)

Networking is *never* hit — we subclass :class:`A2APeerConnection` with an
in-memory ``_post_jsonrpc`` so httpx stays unexercised, which is exactly
what the hard constraints require (httpx is an optional dep for the a2a
extra).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionState,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
    reset_registry,
)
from carl_core.interaction import InteractionChain

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.bus import LocalBus
from carl_studio.a2a.connection import (
    A2APeerConnection,
    A2AServerConnection,
    ProtocolConnection,
)
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.spec import agent_card_to_spec


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_conn_registry():  # pyright: ignore[reportUnusedFunction]
    """Start each test with a clean global registry — the FSM leaves
    connections in the registry until ``close()`` runs, and we want one
    test's leftovers to never bleed into another."""
    reset_registry()
    yield
    reset_registry()


class _MockPeer(A2APeerConnection):
    """Peer subclass that intercepts networking.

    ``_responses`` is a dict keyed by JSON-RPC method name; each value is
    the raw body to hand back to the caller. Unmapped methods raise so
    tests don't silently pass against missing stubs.
    """

    def __init__(
        self,
        peer_card: CARLAgentCard,
        responses: dict[str, dict[str, Any]],
        *,
        chain: InteractionChain | None = None,
    ) -> None:
        super().__init__(peer_card, chain=chain)
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def _post_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((method, params))
        if method not in self._responses:
            raise AssertionError(f"unexpected method in mock: {method}")
        return self._responses[method]


def _peer_card(name: str = "mock-peer", endpoint: str = "http://example/a2a") -> CARLAgentCard:
    return CARLAgentCard(
        name=name,
        version="1.2.3",
        endpoint=endpoint,
        skills=["train", "eval"],
    )


# ---------------------------------------------------------------------------
# ProtocolConnection defaults
# ---------------------------------------------------------------------------


class TestProtocolConnectionSpec:
    def test_class_default_spec_is_protocol_bidirectional(self) -> None:
        spec = ProtocolConnection.spec
        assert spec.name == "a2a.protocol"
        assert spec.scope == ConnectionScope.THREE_P
        assert spec.kind == ConnectionKind.PROTOCOL
        assert spec.direction == ConnectionDirection.BIDIRECTIONAL
        assert spec.transport == ConnectionTransport.HTTP
        assert spec.trust == ConnectionTrust.SIGNED


# ---------------------------------------------------------------------------
# A2APeerConnection spec derivation + lifecycle
# ---------------------------------------------------------------------------


class TestPeerSpecDerivation:
    def test_spec_reflects_peer_card(self) -> None:
        card = _peer_card(name="partner-a", endpoint="https://partner.example/a2a")
        peer = A2APeerConnection(card)
        assert peer.spec.name == "a2a.peer.partner-a"
        assert peer.spec.endpoint == "https://partner.example/a2a"
        assert peer.spec.version == "1.2.3"
        assert peer.spec.metadata == {"tier": "free"}
        assert peer.spec.direction == ConnectionDirection.BIDIRECTIONAL
        assert peer.spec.trust == ConnectionTrust.SIGNED

    def test_initial_state_is_init(self) -> None:
        peer = A2APeerConnection(_peer_card())
        assert peer.state == ConnectionState.INIT


class TestPeerRoundTrip:
    def test_open_close_transitions(self) -> None:
        peer = _MockPeer(_peer_card(), responses={})

        async def run() -> list[ConnectionState]:
            observed: list[ConnectionState] = [peer.state]
            await peer.open()
            observed.append(peer.state)
            await peer.close()
            observed.append(peer.state)
            return observed

        observed = asyncio.run(run())
        assert observed == [
            ConnectionState.INIT,
            ConnectionState.READY,
            ConnectionState.CLOSED,
        ]

    def test_send_message_full_state_sequence(self) -> None:
        """Verify every FSM state we claim to traverse actually appears."""

        seen: list[ConnectionState] = []
        card = _peer_card()
        peer = _MockPeer(card, responses={"message.send": {"result": {"ok": True}}})

        original_transition = peer._transition  # pyright: ignore[reportPrivateUsage]

        def recording_transition(target: ConnectionState) -> None:
            original_transition(target)
            seen.append(target)

        peer._transition = recording_transition  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]

        async def run() -> dict[str, Any]:
            await peer.open()
            msg = A2AMessage.log(task_id="t-1", message="hello")
            body = await peer.send_message(msg)
            await peer.close()
            return body

        body = asyncio.run(run())
        assert body == {"result": {"ok": True}}
        # Expected order: CONNECTING -> AUTHENTICATING -> READY (open) ->
        # TRANSACTING -> READY (transact) -> CLOSING -> CLOSED (close).
        assert seen == [
            ConnectionState.CONNECTING,
            ConnectionState.AUTHENTICATING,
            ConnectionState.READY,
            ConnectionState.TRANSACTING,
            ConnectionState.READY,
            ConnectionState.CLOSING,
            ConnectionState.CLOSED,
        ]

    def test_interaction_chain_records_events(self) -> None:
        chain = InteractionChain()
        peer = _MockPeer(
            _peer_card(),
            responses={"message.send": {"result": {"ok": True}}},
            chain=chain,
        )

        async def run() -> None:
            await peer.open()
            await peer.send_message(A2AMessage.log(task_id="t-1", message="ok"))
            await peer.close()

        asyncio.run(run())

        names = [step.name for step in chain.steps]
        assert "connection.open" in names
        assert "connection.message.send" in names
        assert "connection.close" in names
        # And everything should have succeeded.
        assert all(step.success for step in chain.steps)

    def test_get_task_reconstructs_a2atask(self) -> None:
        peer = _MockPeer(
            _peer_card(),
            responses={
                "tasks.get": {
                    "result": {
                        "id": "t-42",
                        "status": {"state": "completed"},
                        "metadata": {
                            "skill": "eval",
                            "sender": "peer",
                            "receiver": "carl-studio",
                        },
                    }
                },
            },
        )

        async def run():
            await peer.open()
            task = await peer.get_task("t-42")
            await peer.close()
            return task

        task = asyncio.run(run())
        assert task.id == "t-42"
        assert task.skill == "eval"
        assert task.status.value == "done"

    def test_get_card_inverse_of_spec(self) -> None:
        card = _peer_card()
        peer = _MockPeer(
            card,
            responses={"agent.card": {"result": agent_card_to_spec(card)}},
        )

        async def run() -> CARLAgentCard:
            await peer.open()
            fetched = await peer.get_card()
            await peer.close()
            return fetched

        fetched = asyncio.run(run())
        assert fetched.name == card.name
        assert fetched.version == card.version
        assert set(fetched.skills) == set(card.skills)

    def test_malformed_result_raises_unavailable(self) -> None:
        """Peer returning ``{}`` (no ``result`` key) should surface as
        :class:`ConnectionUnavailableError`, not leak through as ``None``."""

        peer = _MockPeer(
            _peer_card(),
            responses={"tasks.get": {}},
        )

        async def run() -> None:
            await peer.open()
            try:
                await peer.get_task("missing")
            finally:
                await peer.close()

        with pytest.raises(ConnectionUnavailableError):
            asyncio.run(run())

    def test_non_http_endpoint_blocks_real_networking(self) -> None:
        """Guard: the real ``_post_jsonrpc`` must reject stdio endpoints
        rather than hand the URL to httpx."""

        card = CARLAgentCard(name="stdio-peer", endpoint="stdio")
        peer = A2APeerConnection(card)

        async def run() -> None:
            await peer.open()
            try:
                await peer.send_message(A2AMessage.log("t-1", "x"))
            finally:
                await peer.close()

        with pytest.raises(ConnectionUnavailableError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# A2AServerConnection dispatch
# ---------------------------------------------------------------------------


class TestServerDispatch:
    def test_agent_card_roundtrip(self, tmp_path: Path) -> None:
        chain = InteractionChain()
        bus = LocalBus(db_path=tmp_path / "a2a.db")
        card = CARLAgentCard(name="carl-local", version="0.4.1")
        server = A2AServerConnection(bus=bus, agent_card=card, chain=chain)

        async def run() -> dict[str, Any]:
            await server.open()
            body = await server.handle_request(
                {"jsonrpc": "2.0", "id": 7, "method": "agent.card", "params": {}},
            )
            await server.close()
            return body

        body = asyncio.run(run())
        assert body["id"] == 7
        assert body["result"]["name"] == "carl-local"
        assert any(step.name == "connection.agent.card" for step in chain.steps)
        bus.close()

    def test_message_send_inserts_into_bus(self, tmp_path: Path) -> None:
        bus = LocalBus(db_path=tmp_path / "a2a.db")
        server = A2AServerConnection(bus=bus)

        async def run() -> dict[str, Any]:
            await server.open()
            body = await server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": "req-1",
                    "method": "message.send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "train this"}],
                        },
                        "metadata": {"skill": "train"},
                    },
                },
            )
            await server.close()
            return body

        body = asyncio.run(run())
        assert body["id"] == "req-1"
        assert body["result"]["metadata"]["skill"] == "train"
        # Confirm the task actually landed in the bus.
        pending = bus.poll(limit=10)
        assert len(pending) == 1
        assert pending[0].skill == "train"
        bus.close()

    def test_tasks_get_missing_id_returns_invalid_params(
        self, tmp_path: Path
    ) -> None:
        bus = LocalBus(db_path=tmp_path / "a2a.db")
        server = A2AServerConnection(bus=bus)

        async def run() -> dict[str, Any]:
            await server.open()
            body = await server.handle_request(
                {"jsonrpc": "2.0", "id": 3, "method": "tasks.get", "params": {}},
            )
            await server.close()
            return body

        body = asyncio.run(run())
        assert "error" in body
        assert body["error"]["code"] == -32602  # INVALID_PARAMS
        bus.close()

    def test_unknown_method_returns_method_not_found(
        self, tmp_path: Path
    ) -> None:
        bus = LocalBus(db_path=tmp_path / "a2a.db")
        server = A2AServerConnection(bus=bus)

        async def run() -> dict[str, Any]:
            await server.open()
            body = await server.handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "unknown.method", "params": {}},
            )
            await server.close()
            return body

        body = asyncio.run(run())
        assert "error" in body
        assert body["error"]["code"] == -32601  # METHOD_NOT_FOUND
        bus.close()

    def test_injected_bus_is_not_closed(self, tmp_path: Path) -> None:
        """When the caller supplies a bus, ``A2AServerConnection.close``
        must not close it — the caller owns the lifecycle."""

        bus = LocalBus(db_path=tmp_path / "a2a.db")
        server = A2AServerConnection(bus=bus)

        async def run() -> None:
            await server.open()
            await server.close()

        asyncio.run(run())
        # bus should still be usable — the underlying connection should
        # still work even after the server closed.
        assert bus.pending_count() == 0
        bus.close()

    def test_default_bus_is_closed_when_owned(self) -> None:
        server = A2AServerConnection()

        async def run() -> None:
            await server.open()
            assert server._bus is not None  # pyright: ignore[reportPrivateUsage]
            await server.close()
            assert server._bus is None  # pyright: ignore[reportPrivateUsage]

        asyncio.run(run())


# ---------------------------------------------------------------------------
# ServerConnection spec
# ---------------------------------------------------------------------------


class TestServerSpec:
    def test_spec_is_1p_ingress(self) -> None:
        spec = A2AServerConnection.spec
        assert spec.scope == ConnectionScope.ONE_P
        assert spec.kind == ConnectionKind.PROTOCOL
        assert spec.direction == ConnectionDirection.INGRESS
        assert spec.transport == ConnectionTransport.HTTP
        assert spec.trust == ConnectionTrust.SIGNED
