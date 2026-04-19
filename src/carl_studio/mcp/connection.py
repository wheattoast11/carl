"""MCPServerConnection — Connection-primitive facade around the FastMCP server.

This is the Phase 1 port: a thin :class:`AsyncBaseConnection` subclass that
wraps the existing ``carl_studio.mcp.server`` module so its lifetime,
transport choice, and tool-call telemetry line up with every other CARL
channel (training backends, A2A peers, x402 merchants, ...).

What's in scope for Phase 1
---------------------------
* A :class:`ConnectionSpec` declaring this as a 1P, PROTOCOL, INGRESS
  connection over STDIO (default) or HTTP (streamable).
* Lazy import of the optional :mod:`mcp` package; missing dep surfaces as
  :class:`ConnectionUnavailableError`.
* FSM-backed ``open`` / ``close`` via :class:`AsyncBaseConnection`.
* :meth:`run` entry point that brackets the FastMCP serve loop in a
  ``transact("serve")`` so DEGRADED/ERROR telemetry is emitted if the
  loop crashes.
* Read-only :attr:`session` snapshot bridging to ``server._session``.

What's deferred
---------------
* Multi-tenant session scoping (Phase 3 / T2).
* 2025-11 MCP tasks/elicitation/sampling (Phase 3 / T2).
* Moving the module-level ``_session`` dict onto the connection instance
  (Phase 3 / T2 — done once ``FastMCP`` exposes a per-request context).
"""

from __future__ import annotations

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
from carl_core.interaction import InteractionChain

from carl_studio.mcp.session import MCPSession, session_from_dict

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from mcp.server.fastmcp import FastMCP


_STDIO = "stdio"
_HTTP = "http"
_SUPPORTED_TRANSPORTS: frozenset[str] = frozenset({_STDIO, _HTTP})


def _transport_enum(transport: str) -> ConnectionTransport:
    """Map the string transport choice to the ConnectionSpec enum."""
    if transport == _STDIO:
        return ConnectionTransport.STDIO
    if transport == _HTTP:
        return ConnectionTransport.HTTP
    # _validate_transport already rejected this; kept for completeness.
    raise ValueError(f"unsupported transport: {transport!r}")


def _validate_transport(transport: str) -> str:
    """Normalize / validate the transport argument."""
    t = (transport or "").lower().strip()
    if t not in _SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"unsupported MCP transport {transport!r}; "
            f"expected one of {sorted(_SUPPORTED_TRANSPORTS)}",
        )
    return t


class MCPServerConnection(AsyncBaseConnection):
    """AsyncBaseConnection wrapper around the carl-studio FastMCP server.

    Usage::

        async with MCPServerConnection(transport="stdio") as conn:
            # tools are already registered at import time via the
            # @mcp.tool() decorators in server.py
            await conn.run()

    The underlying FastMCP instance is lazy-imported on ``_connect`` so
    that ``carl_studio.mcp`` stays importable when the ``mcp`` extra is
    not installed — merely constructing the connection does not drag in
    the dependency.
    """

    # Class-level ConnectionSpec placeholder. Instance ``__init__`` replaces
    # this with a per-instance spec reflecting the chosen transport. The
    # class attribute is required by ``ConnectionBase.__init__``; see the
    # ``hasattr(type(self), "spec")`` check in carl_core.connection.base.
    spec: ConnectionSpec = ConnectionSpec(
        name="carl.mcp.server",
        scope=ConnectionScope.ONE_P,
        kind=ConnectionKind.PROTOCOL,
        direction=ConnectionDirection.INGRESS,
        transport=ConnectionTransport.STDIO,
        trust=ConnectionTrust.AUTHENTICATED,
        version="1",
        endpoint=None,
        metadata={"server_name": "carl-studio"},
    )

    def __init__(
        self,
        *,
        transport: str = _STDIO,
        chain: InteractionChain | None = None,
        connection_id: str | None = None,
    ) -> None:
        normalized = _validate_transport(transport)
        self._transport_choice: str = normalized
        # Per-instance spec so the transport and endpoint metadata reflect
        # the caller's choice. ``ConnectionSpec`` is frozen, so we build a
        # fresh one rather than mutating the class default.
        self.spec = ConnectionSpec(
            name="carl.mcp.server",
            scope=ConnectionScope.ONE_P,
            kind=ConnectionKind.PROTOCOL,
            direction=ConnectionDirection.INGRESS,
            transport=_transport_enum(normalized),
            trust=ConnectionTrust.AUTHENTICATED,
            version="1",
            endpoint=None,
            metadata={"server_name": "carl-studio", "transport_choice": normalized},
        )
        self._fastmcp: FastMCP | None = None
        super().__init__(chain=chain, connection_id=connection_id)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def transport_choice(self) -> str:
        """The normalized transport name, ``"stdio"`` or ``"http"``."""
        return self._transport_choice

    @property
    def fastmcp(self) -> FastMCP | None:
        """The underlying FastMCP instance, or ``None`` before ``open``."""
        return self._fastmcp

    @property
    def session(self) -> MCPSession:
        """Typed, read-only view of the current ``server._session`` dict.

        Mutations still flow through the ``authenticate`` tool (which
        writes directly to ``server._session``). This property just
        reflects whatever that dict holds at the moment of the call.
        """
        # Lazy import — importing server.py at module scope would trigger
        # the FastMCP import chain even for callers that only touch the
        # session view.
        from carl_studio.mcp import server as _server

        # Phase 1: the legacy module-level state dict is still the source
        # of truth. Phase 3 (T2) inverts ownership so the connection holds
        # the session directly. Explicitly annotate + suppress until then.
        raw: dict[str, str] = _server._session  # pyright: ignore[reportPrivateUsage]
        return session_from_dict(raw)

    async def run(self) -> None:
        """Start the FastMCP serve loop wrapped in a ``transact("serve")``.

        The transact bracket gives us:
          * State transition READY -> TRANSACTING on entry.
          * Automatic DEGRADED transition if the loop raises.
          * Wall-clock duration + success/failure event on the chain.

        Blocks until the underlying FastMCP coroutine returns (stdio
        loop ends on EOF; http loop ends when the uvicorn server stops).
        """
        self.require_ready()
        fastmcp = self._fastmcp
        if fastmcp is None:
            # require_ready should have rejected this, but be explicit.
            raise ConnectionUnavailableError(
                "MCPServerConnection.run called before open()",
                context={
                    "connection_id": self.connection_id,
                    "spec_name": self.spec.name,
                },
            )
        async with self.transact("serve"):
            if self._transport_choice == _STDIO:
                await fastmcp.run_stdio_async()
            else:  # _HTTP
                await fastmcp.run_streamable_http_async()

    # ------------------------------------------------------------------
    # AsyncBaseConnection hooks
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Lazy-import ``mcp``; grab the module-level FastMCP singleton.

        The ``@mcp.tool()`` decorators in ``carl_studio.mcp.server``
        register all tools at module import time. Importing that module
        is therefore enough to arm the server — we don't need to rebuild
        the instance here.
        """
        try:
            from carl_studio.mcp import server as _server
        except ImportError as exc:
            raise ConnectionUnavailableError(
                "carl_studio.mcp.server import failed; "
                "install the 'mcp' extra: pip install carl-studio[mcp]",
                context={
                    "missing_package": "mcp",
                    "spec_name": self.spec.name,
                },
            ) from exc

        fastmcp = getattr(_server, "mcp", None)
        if fastmcp is None:
            raise ConnectionUnavailableError(
                "carl_studio.mcp.server.mcp attribute missing; "
                "module did not initialize a FastMCP instance",
                context={"spec_name": self.spec.name},
            )
        self._fastmcp = fastmcp

    async def _authenticate(self) -> None:
        """Phase 1 no-op.

        The spec declares ``AUTHENTICATED`` trust so the FSM correctly
        routes through an AUTHENTICATING state during ``open()``, but
        actual credential negotiation happens per-request via the
        ``authenticate`` tool in ``server.py``. Phase 3 will move that
        state from the module global onto the connection.
        """
        return None

    async def _close(self) -> None:
        """Drop the FastMCP reference. FastMCP holds no long-lived
        resources until ``run_*_async`` is entered; the serve loop owns
        its own teardown via the transport (stdio / uvicorn) so there is
        nothing to close explicitly here.
        """
        self._fastmcp = None

    # ------------------------------------------------------------------
    # Convenience for server.py's bind_connection / tool telemetry
    # ------------------------------------------------------------------

    def record_tool_event(
        self,
        tool_name: str,
        *,
        success: bool,
        duration_ms: float | None = None,
        **ctx: Any,
    ) -> None:
        """Expose ``_record_event`` for tool-call telemetry.

        ``server.py`` holds the @mcp.tool() functions and only gets a
        reference to this connection via ``bind_connection``; wiring
        through a public method keeps the contract small and avoids
        reaching into the private ``_record_event`` name.
        """
        self._record_event(
            f"connection.tool.{tool_name}",
            success=success,
            duration_ms=duration_ms,
            **ctx,
        )


__all__ = [
    "MCPServerConnection",
]
