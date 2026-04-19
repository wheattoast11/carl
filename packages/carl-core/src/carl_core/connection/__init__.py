"""carl_core.connection — the universal connection primitive.

Every inbound or outbound channel in the CARL stack (training backend, RL
environment, A2A peer, MCP server, x402 merchant, memory store, skill
provider) materializes as a :class:`ConnectionSpec` plus a
:class:`BaseConnection` (or :class:`AsyncBaseConnection`) subclass. This
module exports the primitive surface; concrete implementations live in
``carl_studio``.

Quick tour::

    from carl_core.connection import (
        ConnectionSpec, ConnectionScope, ConnectionKind,
        ConnectionDirection, ConnectionTransport, ConnectionTrust,
        BaseConnection,
    )

    class EchoConnection(BaseConnection):
        spec = ConnectionSpec(
            name="demo.echo",
            scope=ConnectionScope.ONE_P,
            kind=ConnectionKind.UTILITY,
            direction=ConnectionDirection.BIDIRECTIONAL,
            transport=ConnectionTransport.IN_PROCESS,
            trust=ConnectionTrust.PUBLIC,
        )

        def _connect(self): ...
        def _close(self): ...

    with EchoConnection() as conn:
        with conn.transact("echo"):
            ...
"""

from __future__ import annotations

from carl_core.connection.base import (
    AsyncBaseConnection,
    BaseConnection,
    ConnectionBase,
    ConnectionStats,
)
from carl_core.connection.errors import (
    CARLConnectionError,
    ConnectionAuthError,
    ConnectionClosedError,
    ConnectionPolicyError,
    ConnectionTransitionError,
    ConnectionUnavailableError,
)
from carl_core.connection.lifecycle import (
    VALID_TRANSITIONS,
    ConnectionState,
    can_transition,
)
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

__all__ = [
    # spec
    "ConnectionSpec",
    "ConnectionScope",
    "ConnectionKind",
    "ConnectionDirection",
    "ConnectionTransport",
    "ConnectionTrust",
    # lifecycle
    "ConnectionState",
    "VALID_TRANSITIONS",
    "can_transition",
    # errors
    "CARLConnectionError",
    "ConnectionUnavailableError",
    "ConnectionAuthError",
    "ConnectionTransitionError",
    "ConnectionClosedError",
    "ConnectionPolicyError",
    # base
    "BaseConnection",
    "AsyncBaseConnection",
    "ConnectionBase",
    "ConnectionStats",
    # registry
    "ConnectionRegistry",
    "get_registry",
    "reset_registry",
]
