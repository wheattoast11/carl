"""A2A peer and server connections built atop ``carl_core.connection``.

The A2A surface exposes two kinds of connections:

* :class:`A2APeerConnection` — EGRESS-capable bidirectional edge to a remote
  CARL (or A2A-compatible) agent. Used to send messages and query tasks on a
  peer. Transport is HTTP (JSON-RPC), trust is SIGNED because the A2A v1.0
  spec binds AgentCards with JWS; this module only declares the trust level,
  the signature check itself is Phase 2 (T1).

* :class:`A2AServerConnection` — INGRESS side of the same protocol: this is
  the connection that represents *our* side serving inbound A2A requests,
  backed by a :class:`~carl_studio.a2a.bus.LocalBus`.

Both derive from :class:`ProtocolConnection`, a thin async specialization of
:class:`~carl_core.connection.AsyncBaseConnection` with sensible defaults for
protocol-kind connections.

A2A v1.0 conformance adds streaming (``message/stream``, ``tasks/subscribe``),
push notification config CRUD, signed authenticated-extended-card delivery,
and task cancellation. Streaming yields a raw SSE byte stream (see
:mod:`carl_studio.a2a.streaming`); push delivery is routed through
:mod:`carl_studio.a2a.push` which owns its own circuit breaker.

Networking is intentionally lazy: ``httpx`` and ``anyio`` are optional
dependencies for the A2A extra. We import them inside the methods that
actually need the wire, so ``import carl_studio.a2a.connection`` stays cheap
and works even when the a2a extra is not installed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from carl_core.connection import (
    AsyncBaseConnection,
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
)

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.spec import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    agent_card_to_spec,
    task_to_jsonrpc_result,
    wrap_jsonrpc_error,
    wrap_jsonrpc_response,
)
from carl_studio.a2a.task import A2ATask, A2ATaskStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from carl_core.interaction import InteractionChain

    from carl_studio.a2a.bus import LocalBus
    from carl_studio.a2a.push import PushConfig, PushSubscriberStore


# ---------------------------------------------------------------------------
# Base protocol connection
# ---------------------------------------------------------------------------


class ProtocolConnection(AsyncBaseConnection):
    """Async base for every A2A-shaped connection.

    Subclasses must override :attr:`spec` (class- or instance-level) and may
    override the lifecycle hooks. The defaults implemented here are safe
    no-ops — fine for peer-to-peer models where ``open`` is really just a
    readiness declaration rather than a stateful handshake.
    """

    # Default spec — intentionally generic. Concrete subclasses override in
    # ``__init__`` with peer-specific identity (name, endpoint) before the
    # FSM ever transitions, so the class-level default only exists to satisfy
    # :class:`ConnectionBase`'s "must declare a spec" invariant.
    spec: ConnectionSpec = ConnectionSpec(
        name="a2a.protocol",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.PROTOCOL,
        direction=ConnectionDirection.BIDIRECTIONAL,
        transport=ConnectionTransport.HTTP,
        trust=ConnectionTrust.SIGNED,
    )

    async def _connect(self) -> None:
        """Default: no-op. Concrete transports can pre-warm a pool here."""

    async def _close(self) -> None:
        """Default: no-op. Concrete transports can tear down a pool here."""


# ---------------------------------------------------------------------------
# Outbound peer connection
# ---------------------------------------------------------------------------


# JSON-RPC method labels used both as transact op names (for telemetry keys
# like ``connection.message.send``) and as the wire ``method`` field.
#
# The A2A v1.0 spec's canonical wire names use slashes (``message/send``,
# ``tasks/get``). For telemetry and dispatch we accept both the slash form
# and the legacy dot form (``message.send``, ``tasks.get``) so pre-1.0
# callers keep working. The normalizing map ``_METHOD_ALIASES`` handles
# this centrally at dispatch time.
_METHOD_MESSAGE_SEND = "message.send"
_METHOD_TASKS_GET = "tasks.get"
_METHOD_AGENT_CARD = "agent.card"
_METHOD_MESSAGE_STREAM = "message.stream"
_METHOD_TASKS_SUBSCRIBE = "tasks.subscribe"
_METHOD_TASKS_CANCEL = "tasks.cancel"
_METHOD_PUSH_SET = "tasks.pushNotificationConfig.set"
_METHOD_PUSH_GET = "tasks.pushNotificationConfig.get"
_METHOD_PUSH_LIST = "tasks.pushNotificationConfig.list"
_METHOD_PUSH_DELETE = "tasks.pushNotificationConfig.delete"
_METHOD_EXTENDED_CARD = "agent.getAuthenticatedExtendedCard"

# Normalize A2A v1.0 slash notation to the dot notation used internally.
_METHOD_ALIASES: dict[str, str] = {
    "message/send": _METHOD_MESSAGE_SEND,
    "message/stream": _METHOD_MESSAGE_STREAM,
    "tasks/get": _METHOD_TASKS_GET,
    "tasks/subscribe": _METHOD_TASKS_SUBSCRIBE,
    "tasks/cancel": _METHOD_TASKS_CANCEL,
    "tasks/pushNotificationConfig/set": _METHOD_PUSH_SET,
    "tasks/pushNotificationConfig/get": _METHOD_PUSH_GET,
    "tasks/pushNotificationConfig/list": _METHOD_PUSH_LIST,
    "tasks/pushNotificationConfig/delete": _METHOD_PUSH_DELETE,
    "agent/card": _METHOD_AGENT_CARD,
    "agent/getAuthenticatedExtendedCard": _METHOD_EXTENDED_CARD,
}


def _normalize_method(method: str) -> str:
    """Return the canonical (dot-form) method label for a wire method string."""
    return _METHOD_ALIASES.get(method, method)


class A2APeerConnection(ProtocolConnection):
    """EGRESS edge to a single A2A peer.

    The spec is derived per-instance from the peer's :class:`CARLAgentCard`
    so routing / telemetry layers can distinguish peers without reaching
    into the underlying transport. Networking uses ``httpx`` behind a lazy
    import; if the library is not installed, any transact that tries to
    hit the wire raises :class:`ConnectionUnavailableError`.
    """

    def __init__(
        self,
        peer_card: CARLAgentCard,
        *,
        chain: InteractionChain | None = None,
        connection_id: str | None = None,
        http_timeout: float = 15.0,
    ) -> None:
        # Per-instance spec — override the class default so every transact
        # emits telemetry keyed on the peer identity, not the generic
        # ``a2a.protocol`` placeholder.
        self.spec = ConnectionSpec(
            name=f"a2a.peer.{peer_card.name}",
            scope=ConnectionScope.THREE_P,
            kind=ConnectionKind.PROTOCOL,
            direction=ConnectionDirection.BIDIRECTIONAL,
            transport=ConnectionTransport.HTTP,
            trust=ConnectionTrust.SIGNED,
            version=peer_card.version,
            endpoint=peer_card.endpoint,
            metadata={"tier": peer_card.tier},
        )
        super().__init__(chain=chain, connection_id=connection_id)
        self.peer_card = peer_card
        self._http_timeout = http_timeout

    # -- transports ------------------------------------------------------

    async def _post_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC request to the peer's endpoint and return the
        parsed JSON body. Raises :class:`ConnectionUnavailableError` when
        ``httpx`` is missing, the endpoint is not a URL, or the request
        fails. Subclasses may override for in-process or mocked transports.
        """
        endpoint = self.peer_card.endpoint
        if not endpoint or endpoint == "stdio" or not endpoint.startswith(("http://", "https://")):
            raise ConnectionUnavailableError(
                "peer endpoint is not an HTTP URL",
                context={
                    "spec_name": self.spec.name,
                    "endpoint": endpoint,
                },
            )

        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ConnectionUnavailableError(
                "httpx is required for A2A peer networking",
                context={"spec_name": self.spec.name, "hint": "pip install httpx"},
            ) from exc

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                body: Any = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via live net
            raise ConnectionUnavailableError(
                f"A2A peer request failed: {exc}",
                context={
                    "spec_name": self.spec.name,
                    "endpoint": endpoint,
                    "method": method,
                },
            ) from exc

        if not isinstance(body, dict):
            raise ConnectionUnavailableError(
                "A2A peer returned a non-object response",
                context={"spec_name": self.spec.name, "method": method},
            )
        typed_body: dict[str, Any] = dict(body)  # type: ignore[arg-type]
        return typed_body

    # -- user-facing operations ------------------------------------------

    async def send_message(self, message: A2AMessage) -> dict[str, Any]:
        """Dispatch an :class:`A2AMessage` via the peer's ``message/send``
        endpoint. Returns the raw JSON-RPC result body.
        """
        async with self.transact(_METHOD_MESSAGE_SEND):
            params = {
                "message": {
                    "role": message.sender,
                    "parts": [
                        {"kind": "text", "text": str(message.payload)},
                    ],
                },
                "metadata": {"task_id": message.task_id, "type": message.type},
            }
            return await self._post_jsonrpc(_METHOD_MESSAGE_SEND, params)

    async def get_task(self, task_id: str) -> A2ATask:
        """Fetch a task from the peer. Returns the reconstructed
        :class:`A2ATask`. Raises :class:`ConnectionUnavailableError` when
        the peer response is malformed.
        """
        async with self.transact(_METHOD_TASKS_GET):
            body = await self._post_jsonrpc(_METHOD_TASKS_GET, {"id": task_id})
            result_any: Any = body.get("result")
            if not isinstance(result_any, dict):
                raise ConnectionUnavailableError(
                    "A2A peer returned no task",
                    context={
                        "spec_name": self.spec.name,
                        "task_id": task_id,
                    },
                )
            result: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
            return _task_from_peer_result(result)

    async def get_card(self) -> CARLAgentCard:
        """Fetch the peer's AgentCard. Returns a freshly constructed
        :class:`CARLAgentCard`; the peer's original card is unchanged.
        """
        async with self.transact(_METHOD_AGENT_CARD):
            body = await self._post_jsonrpc(_METHOD_AGENT_CARD, {})
            result_any: Any = body.get("result")
            if not isinstance(result_any, dict):
                raise ConnectionUnavailableError(
                    "A2A peer returned no AgentCard",
                    context={"spec_name": self.spec.name},
                )
            result: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
            return _agent_card_from_spec(result)

    # -- A2A v1.0 conformance extensions --------------------------------

    def subscribe_to_task(
        self,
        task_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open an SSE subscription on the peer's ``tasks/subscribe`` method.

        Returns an async iterator of parsed events
        (``{"event": ..., "data": ...}``). Consumers iterate it directly:
        ``async for event in peer.subscribe_to_task(id): ...``.

        Declared sync (returns an async-generator object) rather than
        ``async def`` so the caller can start iterating without an
        intermediate ``await``.
        """
        return self._stream_jsonrpc(_METHOD_TASKS_SUBSCRIBE, {"id": task_id})

    def stream_message(
        self,
        message: A2AMessage,
    ) -> AsyncIterator[dict[str, Any]]:
        """Dispatch the peer's ``message/stream`` endpoint as an SSE stream.

        Returns an async-generator — iterate with ``async for``.
        """
        params = {
            "message": {
                "role": message.sender,
                "parts": [{"kind": "text", "text": str(message.payload)}],
            },
            "metadata": {"task_id": message.task_id, "type": message.type},
        }
        return self._stream_jsonrpc(_METHOD_MESSAGE_STREAM, params)

    async def set_push_config(
        self,
        task_id: str,
        webhook_url: str,
        token: str = "",
    ) -> PushConfig:
        """Register a push-notification webhook on the remote peer."""
        from carl_studio.a2a.push import PushConfig

        async with self.transact(_METHOD_PUSH_SET):
            body = await self._post_jsonrpc(
                _METHOD_PUSH_SET,
                {"task_id": task_id, "webhook_url": webhook_url, "token": token},
            )
            result_any: Any = body.get("result")
            if not isinstance(result_any, dict):
                raise ConnectionUnavailableError(
                    "peer did not return a push config",
                    context={"spec_name": self.spec.name, "task_id": task_id},
                )
            result: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
            return PushConfig(
                task_id=str(result.get("task_id", task_id)),
                webhook_url=str(result.get("webhook_url", webhook_url)),
                token=str(result.get("token", token)),
                created_at=str(result.get("created_at", "")),
                config_id=str(result.get("config_id", "")),
            )

    async def get_push_config(self, config_id: str) -> PushConfig:
        """Fetch a single push config by id from the peer."""
        from carl_studio.a2a.push import PushConfig

        async with self.transact(_METHOD_PUSH_GET):
            body = await self._post_jsonrpc(
                _METHOD_PUSH_GET, {"config_id": config_id}
            )
            result_any: Any = body.get("result")
            if not isinstance(result_any, dict):
                raise ConnectionUnavailableError(
                    "peer did not return a push config",
                    context={"spec_name": self.spec.name, "config_id": config_id},
                )
            result: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
            return PushConfig(
                task_id=str(result.get("task_id", "")),
                webhook_url=str(result.get("webhook_url", "")),
                token=str(result.get("token", "")),
                created_at=str(result.get("created_at", "")),
                config_id=str(result.get("config_id", config_id)),
            )

    async def list_push_configs(self, task_id: str) -> list[PushConfig]:
        """List push configs attached to a task on the peer."""
        from carl_studio.a2a.push import PushConfig

        async with self.transact(_METHOD_PUSH_LIST):
            body = await self._post_jsonrpc(_METHOD_PUSH_LIST, {"task_id": task_id})
            result_any: Any = body.get("result")
            if not isinstance(result_any, list):
                raise ConnectionUnavailableError(
                    "peer did not return a push config list",
                    context={"spec_name": self.spec.name, "task_id": task_id},
                )
            items: list[Any] = list(result_any)  # type: ignore[arg-type]
            out: list[PushConfig] = []
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                ed: dict[str, Any] = dict(entry)  # type: ignore[arg-type]
                out.append(
                    PushConfig(
                        task_id=str(ed.get("task_id", task_id)),
                        webhook_url=str(ed.get("webhook_url", "")),
                        token=str(ed.get("token", "")),
                        created_at=str(ed.get("created_at", "")),
                        config_id=str(ed.get("config_id", "")),
                    )
                )
            return out

    async def delete_push_config(self, config_id: str) -> bool:
        """Delete a push config on the peer. Returns True on success."""
        async with self.transact(_METHOD_PUSH_DELETE):
            body = await self._post_jsonrpc(
                _METHOD_PUSH_DELETE, {"config_id": config_id}
            )
            result_any: Any = body.get("result")
            if isinstance(result_any, dict):
                rd_del: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
                deleted_any: Any = rd_del.get("deleted")
                return bool(deleted_any)
            return bool(result_any)

    async def get_authenticated_extended_card(
        self,
        *,
        expected_kid: str | None = None,
    ) -> CARLAgentCard:
        """Fetch the peer's JWS-signed AgentCard and verify the signature.

        Uses :class:`AgentIdentity.load` to produce a verifier key —
        callers operating across trust domains should pre-load an
        identity whose ``public_key_pem`` matches the peer's published
        kid.
        """
        from carl_studio.a2a.identity import AgentIdentity

        async with self.transact(_METHOD_EXTENDED_CARD):
            body = await self._post_jsonrpc(_METHOD_EXTENDED_CARD, {})
            result_any: Any = body.get("result")
            if isinstance(result_any, dict):
                rd: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
                jws_any: Any = rd.get("jws")
                jws_token = str(jws_any) if jws_any is not None else ""
            elif isinstance(result_any, str):
                jws_token = result_any
            else:
                raise ConnectionUnavailableError(
                    "peer did not return a signed card",
                    context={"spec_name": self.spec.name},
                )
            if not jws_token:
                raise ConnectionUnavailableError(
                    "peer returned empty JWS token",
                    context={"spec_name": self.spec.name},
                )
            identity = AgentIdentity.load()
            return identity.verify_card(jws_token, expected_kid=expected_kid)

    async def cancel_task(self, task_id: str) -> bool:
        """Request task cancellation on the peer. Returns True on success."""
        async with self.transact(_METHOD_TASKS_CANCEL):
            body = await self._post_jsonrpc(_METHOD_TASKS_CANCEL, {"id": task_id})
            result_any: Any = body.get("result")
            if isinstance(result_any, dict):
                rd_cancel: dict[str, Any] = dict(result_any)  # type: ignore[arg-type]
                cancelled_any: Any = rd_cancel.get("cancelled")
                return bool(cancelled_any)
            return bool(result_any)

    # -- streaming transport --------------------------------------------

    async def _stream_jsonrpc(
        self,
        method: str,
        params: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """POST a JSON-RPC request and consume the SSE body as events.

        Lazy-imports httpx. Raises :class:`ConnectionUnavailableError` when
        the peer endpoint is not an HTTP URL or httpx is missing.

        Subclasses (mocks) may override to feed events from an in-memory
        source without hitting the wire.
        """
        endpoint = self.peer_card.endpoint
        if not endpoint or endpoint == "stdio" or not endpoint.startswith(("http://", "https://")):
            raise ConnectionUnavailableError(
                "peer endpoint is not an HTTP URL",
                context={"spec_name": self.spec.name, "endpoint": endpoint},
            )
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - no-extra path
            raise ConnectionUnavailableError(
                "httpx is required for A2A peer streaming",
                context={"spec_name": self.spec.name, "hint": "pip install httpx"},
            ) from exc

        from carl_studio.a2a.streaming import parse_sse_events

        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        async def _byte_stream() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk

        async for event in parse_sse_events(_byte_stream()):
            yield event


# ---------------------------------------------------------------------------
# Inbound server connection
# ---------------------------------------------------------------------------


class A2AServerConnection(AsyncBaseConnection):
    """INGRESS side of the A2A surface: our listener, backed by a
    :class:`LocalBus`.

    Owns the bus lifecycle when none is passed explicitly: ``_connect``
    materializes the default local SQLite bus, and ``_close`` closes it.
    When a bus is injected we leave its lifecycle untouched — the caller
    owns it.

    :meth:`handle_request` dispatches JSON-RPC method names to bus
    operations, wrapping each in ``self.transact(method)`` so every call
    emits ``connection.<method>`` telemetry.
    """

    spec: ConnectionSpec = ConnectionSpec(
        name="a2a.server",
        scope=ConnectionScope.ONE_P,
        kind=ConnectionKind.PROTOCOL,
        direction=ConnectionDirection.INGRESS,
        transport=ConnectionTransport.HTTP,
        trust=ConnectionTrust.SIGNED,
    )

    def __init__(
        self,
        *,
        bus: LocalBus | None = None,
        agent_card: CARLAgentCard | None = None,
        chain: InteractionChain | None = None,
        connection_id: str | None = None,
        advertise_streaming: bool = True,
        advertise_push: bool = True,
        advertise_identity: bool = False,
    ) -> None:
        super().__init__(chain=chain, connection_id=connection_id)
        self._bus: LocalBus | None = bus
        self._owns_bus: bool = bus is None
        self._agent_card = agent_card
        self._push_store: PushSubscriberStore | None = None
        self._advertise_streaming = advertise_streaming
        self._advertise_push = advertise_push
        self._advertise_identity = advertise_identity

    @property
    def bus(self) -> LocalBus:
        """Return the attached :class:`LocalBus`. Raises if not yet opened."""
        if self._bus is None:
            raise ConnectionUnavailableError(
                "A2A server has no bus attached (call open() first)",
                context={"spec_name": self.spec.name},
            )
        return self._bus

    @property
    def push_store(self) -> PushSubscriberStore:
        """Lazily-materialized :class:`PushSubscriberStore` bound to our bus."""
        if self._push_store is None:
            from carl_studio.a2a.push import PushSubscriberStore

            self._push_store = PushSubscriberStore(self.bus)
        return self._push_store

    async def _connect(self) -> None:
        """Materialize a default LocalBus if the caller didn't inject one."""
        if self._bus is None:
            from carl_studio.a2a.bus import LocalBus

            self._bus = LocalBus()
            self._owns_bus = True

    async def _close(self) -> None:
        """Close the bus if we own it. Injected buses are left alone."""
        if self._bus is not None and self._owns_bus:
            self._bus.close()
        self._bus = None
        self._push_store = None

    # -- request handling ------------------------------------------------

    async def handle_request(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a single JSON-RPC request body to the appropriate handler.

        Supports the full 11-method A2A v1.0 surface:

        * ``agent/card`` / ``agent.card``
        * ``agent/getAuthenticatedExtendedCard``
        * ``message/send`` / ``message.send``
        * ``message/stream`` / ``message.stream`` — handler returns a marker
          dict; consumers use :func:`handle_stream` to pull SSE frames.
        * ``tasks/get`` / ``tasks.get``
        * ``tasks/subscribe`` / ``tasks.subscribe`` — same streaming marker.
        * ``tasks/cancel`` / ``tasks.cancel``
        * ``tasks/pushNotificationConfig/{set,get,list,delete}``

        Unknown methods return ``METHOD_NOT_FOUND``. The method name (after
        normalization to dot form) becomes the :meth:`transact` op name so
        telemetry consistently shows ``connection.<method>``.
        """
        raw_id: Any = request_body.get("id")
        request_id: str | int | None
        if isinstance(raw_id, (str, int)):
            request_id = raw_id
        else:
            request_id = None
        method_any: Any = request_body.get("method")
        params_raw: Any = request_body.get("params") or {}
        if not isinstance(method_any, str):
            return wrap_jsonrpc_error(request_id, INVALID_PARAMS, "missing method")
        if not isinstance(params_raw, dict):
            return wrap_jsonrpc_error(request_id, INVALID_PARAMS, "params must be an object")
        params: dict[str, Any] = dict(params_raw)  # type: ignore[arg-type]
        method = _normalize_method(method_any)

        if method == _METHOD_AGENT_CARD:
            async with self.transact(_METHOD_AGENT_CARD):
                card = self._agent_card or CARLAgentCard()
                spec_dict = agent_card_to_spec(
                    card,
                    streaming=self._advertise_streaming,
                    push_notifications=self._advertise_push,
                    include_identity=self._advertise_identity,
                )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    spec_dict,
                )

        if method == _METHOD_EXTENDED_CARD:
            return await self._handle_extended_card(request_id)

        if method == _METHOD_MESSAGE_SEND:
            async with self.transact(_METHOD_MESSAGE_SEND):
                task = _task_from_message_send(params)
                try:
                    self.bus.post(task)
                except Exception as exc:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"bus.post failed: {exc}",
                    )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    task_to_jsonrpc_result(task),
                )

        if method == _METHOD_MESSAGE_STREAM:
            # Post the task, return a marker telling the caller to consume
            # ``handle_stream`` for the SSE body.
            async with self.transact(_METHOD_MESSAGE_STREAM):
                task = _task_from_message_send(params)
                try:
                    self.bus.post(task)
                except Exception as exc:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"bus.post failed: {exc}",
                    )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    {
                        "stream": True,
                        "task": task_to_jsonrpc_result(task),
                    },
                )

        if method == _METHOD_TASKS_GET:
            async with self.transact(_METHOD_TASKS_GET):
                task_id_any: Any = params.get("id")
                if not isinstance(task_id_any, str) or not task_id_any:
                    return wrap_jsonrpc_error(
                        request_id,
                        INVALID_PARAMS,
                        "id must be a non-empty string",
                    )
                try:
                    task = self.bus.get(task_id_any)
                except Exception as exc:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"bus.get failed: {exc}",
                    )
                if task is None:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"task not found: {task_id_any}",
                    )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    task_to_jsonrpc_result(task),
                )

        if method == _METHOD_TASKS_SUBSCRIBE:
            async with self.transact(_METHOD_TASKS_SUBSCRIBE):
                tid_any: Any = params.get("id")
                if not isinstance(tid_any, str) or not tid_any:
                    return wrap_jsonrpc_error(
                        request_id,
                        INVALID_PARAMS,
                        "id must be a non-empty string",
                    )
                task = self.bus.get(tid_any)
                if task is None:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"task not found: {tid_any}",
                    )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    {"stream": True, "task_id": tid_any},
                )

        if method == _METHOD_TASKS_CANCEL:
            async with self.transact(_METHOD_TASKS_CANCEL):
                tid_any2: Any = params.get("id")
                if not isinstance(tid_any2, str) or not tid_any2:
                    return wrap_jsonrpc_error(
                        request_id,
                        INVALID_PARAMS,
                        "id must be a non-empty string",
                    )
                existing = self.bus.get(tid_any2)
                if existing is None:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"task not found: {tid_any2}",
                    )
                try:
                    self.bus.cancel(tid_any2)
                except Exception as exc:
                    return wrap_jsonrpc_error(
                        request_id,
                        INTERNAL_ERROR,
                        f"cancel failed: {exc}",
                    )
                return wrap_jsonrpc_response(
                    request_id if request_id is not None else 0,
                    {"cancelled": True, "id": tid_any2},
                )

        if method == _METHOD_PUSH_SET:
            return await self._handle_push_set(request_id, params)
        if method == _METHOD_PUSH_GET:
            return await self._handle_push_get(request_id, params)
        if method == _METHOD_PUSH_LIST:
            return await self._handle_push_list(request_id, params)
        if method == _METHOD_PUSH_DELETE:
            return await self._handle_push_delete(request_id, params)

        return wrap_jsonrpc_error(
            request_id, METHOD_NOT_FOUND, f"unknown method: {method_any}"
        )

    async def handle_stream(
        self,
        request_body: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Yield raw SSE byte frames for a streaming JSON-RPC request.

        Used when the caller has detected that ``request_body.method`` is
        one of the streaming methods — typically by checking
        ``{"message/stream", "tasks/subscribe"}`` or their dot-form
        equivalents. Any non-streaming method raises ``ValueError``.
        """
        method_any: Any = request_body.get("method")
        params_raw: Any = request_body.get("params") or {}
        if not isinstance(method_any, str):
            raise ValueError("handle_stream: missing method")
        if not isinstance(params_raw, dict):
            raise ValueError("handle_stream: params must be an object")
        params: dict[str, Any] = dict(params_raw)  # type: ignore[arg-type]
        method = _normalize_method(method_any)

        from carl_studio.a2a.streaming import stream_message, stream_task_updates

        if method == _METHOD_TASKS_SUBSCRIBE:
            tid_any: Any = params.get("id")
            if not isinstance(tid_any, str) or not tid_any:
                raise ValueError("tasks/subscribe: id must be a non-empty string")
            async for frame in stream_task_updates(self, tid_any):
                yield frame
            return

        if method == _METHOD_MESSAGE_STREAM:
            msg_any: Any = params.get("message") or {}
            msg: dict[str, Any] = (
                dict(msg_any) if isinstance(msg_any, dict) else {}  # type: ignore[arg-type]
            )
            role_any: Any = msg.get("role", "user") or "user"
            parts_any: Any = msg.get("parts") or []
            text_bits: list[str] = []
            if isinstance(parts_any, list):
                parts_list: list[Any] = list(parts_any)  # type: ignore[arg-type]
                for p in parts_list:
                    if isinstance(p, dict):
                        pd: dict[str, Any] = dict(p)  # type: ignore[arg-type]
                        if pd.get("kind") == "text":
                            text_bits.append(str(pd.get("text", "")))
            meta_any: Any = params.get("metadata") or {}
            meta: dict[str, Any] = (
                dict(meta_any) if isinstance(meta_any, dict) else {}  # type: ignore[arg-type]
            )
            from uuid import uuid4

            message = A2AMessage(
                id=str(uuid4()),
                task_id=str(meta.get("task_id", "") or ""),
                type="progress",
                payload={"text": "".join(text_bits)},
                sender=str(role_any),
            )
            async for frame in stream_message(self, message):
                yield frame
            return

        raise ValueError(f"handle_stream: method not streamable: {method_any}")

    # -- push handlers -------------------------------------------------

    async def _handle_push_set(
        self,
        request_id: str | int | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        async with self.transact(_METHOD_PUSH_SET):
            task_id_any: Any = params.get("task_id") or params.get("taskId")
            url_any: Any = params.get("webhook_url") or params.get("webhookUrl")
            token_any: Any = params.get("token", "") or ""
            if not isinstance(task_id_any, str) or not task_id_any:
                return wrap_jsonrpc_error(
                    request_id, INVALID_PARAMS, "task_id must be a non-empty string"
                )
            if not isinstance(url_any, str) or not url_any:
                return wrap_jsonrpc_error(
                    request_id, INVALID_PARAMS, "webhook_url must be a non-empty string"
                )
            try:
                cfg = self.push_store.set(task_id_any, url_any, str(token_any))
            except ValueError as exc:
                return wrap_jsonrpc_error(request_id, INVALID_PARAMS, str(exc))
            except Exception as exc:
                return wrap_jsonrpc_error(
                    request_id, INTERNAL_ERROR, f"push.set failed: {exc}"
                )
            return wrap_jsonrpc_response(
                request_id if request_id is not None else 0,
                cfg.to_dict(),
            )

    async def _handle_push_get(
        self,
        request_id: str | int | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        async with self.transact(_METHOD_PUSH_GET):
            cid_any: Any = params.get("config_id") or params.get("configId")
            if not isinstance(cid_any, str) or not cid_any:
                return wrap_jsonrpc_error(
                    request_id, INVALID_PARAMS, "config_id must be a non-empty string"
                )
            cfg = self.push_store.get(cid_any)
            if cfg is None:
                return wrap_jsonrpc_error(
                    request_id, INTERNAL_ERROR, f"config not found: {cid_any}"
                )
            return wrap_jsonrpc_response(
                request_id if request_id is not None else 0,
                cfg.to_dict(),
            )

    async def _handle_push_list(
        self,
        request_id: str | int | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        async with self.transact(_METHOD_PUSH_LIST):
            tid_any: Any = params.get("task_id") or params.get("taskId")
            if not isinstance(tid_any, str) or not tid_any:
                return wrap_jsonrpc_error(
                    request_id, INVALID_PARAMS, "task_id must be a non-empty string"
                )
            configs = self.push_store.list(tid_any)
            return wrap_jsonrpc_response(
                request_id if request_id is not None else 0,
                [c.to_dict() for c in configs],
            )

    async def _handle_push_delete(
        self,
        request_id: str | int | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        async with self.transact(_METHOD_PUSH_DELETE):
            cid_any: Any = params.get("config_id") or params.get("configId")
            if not isinstance(cid_any, str) or not cid_any:
                return wrap_jsonrpc_error(
                    request_id, INVALID_PARAMS, "config_id must be a non-empty string"
                )
            deleted = self.push_store.delete(cid_any)
            return wrap_jsonrpc_response(
                request_id if request_id is not None else 0,
                {"deleted": deleted, "config_id": cid_any},
            )

    async def _handle_extended_card(
        self,
        request_id: str | int | None,
    ) -> dict[str, Any]:
        async with self.transact(_METHOD_EXTENDED_CARD):
            card = self._agent_card or CARLAgentCard()
            try:
                from carl_studio.a2a.identity import AgentIdentity

                identity = AgentIdentity.load()
                token = identity.sign_card(card)
            except ConnectionUnavailableError as exc:
                return wrap_jsonrpc_error(
                    request_id, INTERNAL_ERROR, f"identity unavailable: {exc}"
                )
            except Exception as exc:
                return wrap_jsonrpc_error(
                    request_id, INTERNAL_ERROR, f"sign_card failed: {exc}"
                )
            return wrap_jsonrpc_response(
                request_id if request_id is not None else 0,
                {"jws": token, "alg": identity.algorithm, "kid": identity.kid},
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_from_peer_result(result: dict[str, Any]) -> A2ATask:
    """Reconstruct an :class:`A2ATask` from an A2A tasks/get result payload.

    The spec's state vocabulary (submitted / working / completed / failed /
    canceled) is normalized back to CARL's internal statuses.
    """
    status_obj_any: Any = result.get("status") or {}
    status_obj: dict[str, Any] = (
        dict(status_obj_any) if isinstance(status_obj_any, dict) else {}  # type: ignore[arg-type]
    )
    state_val: Any = status_obj.get("state", "pending")
    state_str = str(state_val) if state_val is not None else "pending"
    inverse_status_map: dict[str, str] = {
        "submitted": "pending",
        "working": "running",
        "completed": "done",
        "failed": "failed",
        "canceled": "cancelled",
    }
    normalized = inverse_status_map.get(state_str, state_str)
    raw_meta_any: Any = result.get("metadata") or {}
    meta: dict[str, Any] = (
        dict(raw_meta_any) if isinstance(raw_meta_any, dict) else {}  # type: ignore[arg-type]
    )
    raw_id: Any = result.get("id")
    raw_skill: Any = meta.get("skill", "")
    raw_sender: Any = meta.get("sender", "")
    raw_receiver: Any = meta.get("receiver", "carl-studio")
    return A2ATask(
        id=str(raw_id) if raw_id is not None else "",
        skill=str(raw_skill) if raw_skill is not None else "",
        sender=str(raw_sender) if raw_sender is not None else "",
        receiver=str(raw_receiver) if raw_receiver else "carl-studio",
        status=A2ATaskStatus(normalized),
    )


def _agent_card_from_spec(spec: dict[str, Any]) -> CARLAgentCard:
    """Inverse of :func:`agent_card_to_spec`. Kept here (rather than in
    ``spec.py``) because it is only needed by the peer connection.
    """
    raw_skills: Any = spec.get("skills") or []
    skills: list[str] = []
    if isinstance(raw_skills, list):
        raw_list: list[Any] = list(raw_skills)  # type: ignore[arg-type]
        for entry in raw_list:
            if isinstance(entry, dict):
                entry_dict: dict[str, Any] = entry  # type: ignore[assignment]
                name_any: Any = entry_dict.get("name") or entry_dict.get("id")
                if isinstance(name_any, str):
                    skills.append(name_any)
            elif isinstance(entry, str):
                skills.append(entry)
    url_any: Any = spec.get("url")
    endpoint = url_any if isinstance(url_any, str) and url_any else "stdio"
    name_any2: Any = spec.get("name", "") or ""
    version_any: Any = spec.get("version", "0") or "0"
    return CARLAgentCard(
        name=str(name_any2),
        version=str(version_any),
        skills=skills,
        endpoint=endpoint,
    )


def _task_from_message_send(params: dict[str, Any]) -> A2ATask:
    """Build an :class:`A2ATask` from A2A ``message/send`` params.

    Reuses :func:`carl_studio.a2a.spec.message_send_to_task` for text
    extraction, then wraps the result in a freshly ID'd task.
    """
    from uuid import uuid4

    from carl_studio.a2a.spec import message_send_to_task

    kwargs: dict[str, Any] = message_send_to_task(params)
    raw_meta_any: Any = params.get("metadata") or {}
    meta: dict[str, Any] = (
        dict(raw_meta_any) if isinstance(raw_meta_any, dict) else {}  # type: ignore[arg-type]
    )
    skill_any: Any = kwargs.get("skill", "") or ""
    sender_any: Any = kwargs.get("sender", "") or ""
    inputs_any: Any = kwargs.get("inputs") or {}
    inputs_dict: dict[str, Any] = (
        dict(inputs_any) if isinstance(inputs_any, dict) else {}  # type: ignore[arg-type]
    )
    receiver_any: Any = meta.get("receiver", "carl-studio") or "carl-studio"
    return A2ATask(
        id=str(uuid4()),
        skill=str(skill_any),
        inputs=inputs_dict,
        sender=str(sender_any),
        receiver=str(receiver_any),
    )


__all__ = [
    "ProtocolConnection",
    "A2APeerConnection",
    "A2AServerConnection",
]


# Public symbols for tests / introspection. Callers should import from
# :mod:`carl_studio.a2a` (the package __init__) rather than reaching in
# directly; these are re-exports for convenience.
SUPPORTED_METHODS: frozenset[str] = frozenset(
    {
        _METHOD_AGENT_CARD,
        _METHOD_EXTENDED_CARD,
        _METHOD_MESSAGE_SEND,
        _METHOD_MESSAGE_STREAM,
        _METHOD_TASKS_GET,
        _METHOD_TASKS_SUBSCRIBE,
        _METHOD_TASKS_CANCEL,
        _METHOD_PUSH_SET,
        _METHOD_PUSH_GET,
        _METHOD_PUSH_LIST,
        _METHOD_PUSH_DELETE,
    }
)
