"""A2A v1.0 conformance round-trip tests.

This suite stands up an :class:`A2AServerConnection` backed by an
in-memory :class:`LocalBus`, wires a mock :class:`A2APeerConnection`
that loops its requests back into the server's ``handle_request``, and
drives every one of the 11 JSON-RPC methods the spec mandates:

    agent/card, agent/getAuthenticatedExtendedCard,
    message/send, message/stream,
    tasks/get, tasks/subscribe, tasks/cancel,
    tasks/pushNotificationConfig/set / get / list / delete

Networking is never hit. Each method's request params, response shape,
and side-effects on the bus are asserted explicitly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from carl_core.connection import reset_registry

from carl_studio.a2a import (
    A2APeerConnection,
    A2AServerConnection,
    A2ATask,
    AgentIdentity,
    CARLAgentCard,
    LocalBus,
    parse_sse_events,
)
from carl_studio.a2a.connection import SUPPORTED_METHODS

# Type alias for the pair returned by the ``server_and_peer`` fixture.
ServerPeerPair = tuple["A2AServerConnection", "_LoopbackPeer"]


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    reset_registry()
    yield
    reset_registry()


@pytest.fixture(autouse=True)
def _isolate_keyring(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent cross-test contamination via the OS keyring."""
    from carl_studio.a2a import identity as identity_mod

    def _no_load() -> None:
        return None

    def _no_save(_pem: bytes) -> None:
        return None

    monkeypatch.setattr(identity_mod, "_try_keyring_load", _no_load)
    monkeypatch.setattr(identity_mod, "_try_keyring_save", _no_save)


@pytest.fixture
def tmp_bus(tmp_path: Path) -> Iterator[LocalBus]:
    bus = LocalBus(db_path=tmp_path / "a2a.db")
    yield bus
    bus.close()


@pytest.fixture
def keystore(tmp_path: Path) -> Path:
    d = tmp_path / "keys"
    d.mkdir(exist_ok=True)
    return d


class _LoopbackPeer(A2APeerConnection):
    """Peer that loops every JSON-RPC request back through the in-process
    server rather than hitting the wire.

    ``_post_jsonrpc`` is patched to build a request body and feed it to
    ``server.handle_request``. The return value is the JSON-RPC envelope
    the server produced, identical to what httpx would have returned.

    For streaming methods we feed ``server.handle_stream`` and collect
    the yielded SSE frames into memory; the base ``_stream_jsonrpc``
    hook is replaced to read from those frames via ``parse_sse_events``.
    """

    def __init__(
        self,
        peer_card: CARLAgentCard,
        server: A2AServerConnection,
    ) -> None:
        super().__init__(peer_card)
        self._server = server

    async def call(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Public helper — hits a method by its raw wire name, bypassing
        the outbound helper methods. Used by tests that exercise the
        slash-form aliases directly.
        """
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        return await self._server.handle_request(body)

    async def _post_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.call(method, params)

    async def _stream_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        # Collect every frame the server emits, then replay them through
        # the parser that `A2APeerConnection` ships with.
        frames: list[bytes] = []
        async for frame in self._server.handle_stream(body):
            frames.append(frame)

        async def _replay() -> AsyncIterator[bytes]:
            for f in frames:
                yield f

        async for event in parse_sse_events(_replay()):
            yield event


@pytest.fixture
def server_and_peer(
    tmp_bus: LocalBus, keystore: Path
) -> Iterator[tuple[A2AServerConnection, _LoopbackPeer]]:
    """Bring up a server with a bus + peer looped back to it."""
    # Pre-create the identity so agent/getAuthenticatedExtendedCard works.
    AgentIdentity.load(keystore_dir=keystore)
    # Redirect AgentIdentity.load() to our keystore during the whole
    # conformance run so server & peer share one identity.
    import carl_studio.a2a.identity as identity_mod

    original = identity_mod.AgentIdentity.load

    def _load_override(
        cls: type[AgentIdentity],
        keystore_dir: Path | None = None,
        **kw: Any,
    ) -> AgentIdentity:
        return original(keystore_dir=keystore, **kw)

    identity_mod.AgentIdentity.load = classmethod(_load_override)  # type: ignore[method-assign]

    server = A2AServerConnection(
        bus=tmp_bus,
        agent_card=CARLAgentCard(
            name="carl-test",
            version="1.0.0",
            skills=["train", "eval"],
            endpoint="http://test.example/a2a",
        ),
        advertise_streaming=True,
        advertise_push=True,
        advertise_identity=True,
    )

    peer_card = CARLAgentCard(
        name="remote-peer",
        version="1.0.0",
        endpoint="http://peer.example/a2a",
    )
    peer = _LoopbackPeer(peer_card, server)

    async def _setup():
        await server.open()
        await peer.open()

    asyncio.run(_setup())

    yield server, peer

    async def _teardown():
        await peer.close()
        await server.close()

    asyncio.run(_teardown())
    identity_mod.AgentIdentity.load = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Method surface.
# ---------------------------------------------------------------------------


def test_supported_methods_is_11_method_surface() -> None:
    assert len(SUPPORTED_METHODS) == 11
    # Spot-check the full spec surface is present.
    assert "agent.card" in SUPPORTED_METHODS
    assert "agent.getAuthenticatedExtendedCard" in SUPPORTED_METHODS
    assert "message.send" in SUPPORTED_METHODS
    assert "message.stream" in SUPPORTED_METHODS
    assert "tasks.get" in SUPPORTED_METHODS
    assert "tasks.subscribe" in SUPPORTED_METHODS
    assert "tasks.cancel" in SUPPORTED_METHODS
    assert "tasks.pushNotificationConfig.set" in SUPPORTED_METHODS
    assert "tasks.pushNotificationConfig.get" in SUPPORTED_METHODS
    assert "tasks.pushNotificationConfig.list" in SUPPORTED_METHODS
    assert "tasks.pushNotificationConfig.delete" in SUPPORTED_METHODS


# ---------------------------------------------------------------------------
# agent/card + agent/getAuthenticatedExtendedCard.
# ---------------------------------------------------------------------------


class TestAgentCardMethods:
    def test_agent_card_round_trip_declares_capabilities(self, server_and_peer: ServerPeerPair) -> None:
        _server, peer = server_and_peer

        async def run() -> CARLAgentCard:
            return await peer.get_card()

        card = asyncio.run(run())
        assert card.name == "carl-test"
        assert card.version == "1.0.0"

        # Direct inspection of the on-wire capabilities dict.
        async def run2() -> dict[str, Any]:
            body = await peer.call("agent.card", {})
            return body["result"]

        spec = asyncio.run(run2())
        assert spec["capabilities"]["streaming"] is True
        assert spec["capabilities"]["pushNotifications"] is True
        assert spec.get("securitySchemes") is not None

    def test_authenticated_extended_card_round_trip(self, server_and_peer: ServerPeerPair) -> None:
        _server, peer = server_and_peer

        async def run() -> CARLAgentCard:
            return await peer.get_authenticated_extended_card()

        verified = asyncio.run(run())
        assert isinstance(verified, CARLAgentCard)
        assert verified.name == "carl-test"

    def test_authenticated_extended_card_slash_form(self, server_and_peer: ServerPeerPair) -> None:
        _server, peer = server_and_peer

        async def run() -> dict[str, Any]:
            return await peer.call(
                "agent/getAuthenticatedExtendedCard", {}
            )

        body = asyncio.run(run())
        assert "result" in body
        assert "jws" in body["result"]
        assert body["result"]["alg"] == "ES256"


# ---------------------------------------------------------------------------
# message/send + message/stream.
# ---------------------------------------------------------------------------


class TestMessageMethods:
    def test_message_send(self, server_and_peer: ServerPeerPair) -> None:
        _server, peer = server_and_peer
        from carl_studio.a2a import A2AMessage

        async def run() -> dict[str, Any]:
            return await peer.send_message(
                A2AMessage(
                    id="m1",
                    task_id="",
                    type="progress",
                    payload={"skill": "train", "text": "hi"},
                    sender="user",
                )
            )

        body = asyncio.run(run())
        assert "result" in body
        assert body["result"]["status"]["state"] == "submitted"

    def test_message_stream_slash_form_dispatches_as_stream(
        self, server_and_peer: ServerPeerPair
    ) -> None:
        _server, peer = server_and_peer

        async def run() -> dict[str, Any]:
            # Use the slash-form wire name to confirm normalization works.
            return await peer.call(
                "message/stream",
                {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "hi"}],
                    },
                    "metadata": {"skill": "train"},
                },
            )

        body = asyncio.run(run())
        assert "result" in body
        assert body["result"]["stream"] is True
        assert body["result"]["task"]["status"]["state"] == "submitted"


# ---------------------------------------------------------------------------
# tasks/get + tasks/subscribe + tasks/cancel.
# ---------------------------------------------------------------------------


class TestTaskMethods:
    def test_tasks_get_returns_current_task(self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="get-1", skill="train"))

        async def run():
            return await peer.get_task("get-1")

        task = asyncio.run(run())
        assert task.id == "get-1"
        assert task.status.value == "pending"

    def test_tasks_subscribe_yields_status_events(
        self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus
    ) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="sub-1", skill="train"))
        tmp_bus.update(
            A2ATask(id="sub-1", skill="train").mark_running().mark_done({"ok": True})
        )

        async def run() -> list[dict[str, Any]]:
            events: list[dict[str, Any]] = []
            async for ev in peer.subscribe_to_task("sub-1"):
                events.append(ev)
            return events

        events = asyncio.run(run())
        assert events
        assert events[-1]["data"]["status"]["state"] == "completed"

    def test_tasks_cancel_transitions_to_cancelled(
        self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus
    ) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="cancel-me", skill="train"))

        async def run() -> bool:
            return await peer.cancel_task("cancel-me")

        ok = asyncio.run(run())
        assert ok is True

        task = tmp_bus.get("cancel-me")
        assert task is not None
        assert task.status.value == "cancelled"

    def test_tasks_cancel_slash_form_equivalent(self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="cancel-slash", skill="train"))

        async def run() -> dict[str, Any]:
            return await peer.call("tasks/cancel", {"id": "cancel-slash"})

        body = asyncio.run(run())
        assert body["result"]["cancelled"] is True

    def test_tasks_cancel_missing_returns_error(self, server_and_peer: ServerPeerPair) -> None:
        _server, peer = server_and_peer

        async def run() -> dict[str, Any]:
            return await peer.call("tasks.cancel", {"id": "ghost"})

        body = asyncio.run(run())
        assert "error" in body


# ---------------------------------------------------------------------------
# Push notification config CRUD.
# ---------------------------------------------------------------------------


class TestPushConfigMethods:
    def test_push_config_full_crud(self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="push-t", skill="train"))

        async def run():
            cfg = await peer.set_push_config(
                "push-t", "https://hook.example/a2a", token="s3cret"
            )
            assert cfg.config_id
            configs = await peer.list_push_configs("push-t")
            assert len(configs) == 1
            assert configs[0].config_id == cfg.config_id

            got = await peer.get_push_config(cfg.config_id)
            assert got.webhook_url == "https://hook.example/a2a"
            assert got.token == "s3cret"

            deleted = await peer.delete_push_config(cfg.config_id)
            assert deleted is True
            configs2 = await peer.list_push_configs("push-t")
            assert configs2 == []

        asyncio.run(run())

    def test_push_config_slash_form(self, server_and_peer: ServerPeerPair, tmp_bus: LocalBus) -> None:
        _server, peer = server_and_peer
        tmp_bus.post(A2ATask(id="push-s", skill="train"))

        async def run() -> dict[str, Any]:
            return await peer.call(
                "tasks/pushNotificationConfig/set",
                {"task_id": "push-s", "webhook_url": "https://h.example", "token": ""},
            )

        body = asyncio.run(run())
        assert "result" in body
        assert body["result"]["task_id"] == "push-s"


# ---------------------------------------------------------------------------
# Unknown method negative path.
# ---------------------------------------------------------------------------


def test_unknown_method_returns_method_not_found(server_and_peer: ServerPeerPair) -> None:
    _server, peer = server_and_peer

    async def run() -> dict[str, Any]:
        return await peer.call("no.such.method", {})

    body = asyncio.run(run())
    assert "error" in body
    assert body["error"]["code"] == -32601  # METHOD_NOT_FOUND
