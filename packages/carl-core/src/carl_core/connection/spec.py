"""ConnectionSpec — declarative metadata for any CARL connection.

Every inbound or outbound channel in the CARL stack (training backend, RL
environment, A2A peer, MCP server, x402 merchant, memory store, skill
provider) surfaces as a :class:`ConnectionSpec` plus a
:class:`BaseConnection` subclass that owns the transport/auth implementation.

The spec is pure data. No lifecycle, no policy, no side effects. It exists
so that routing/auth/tier/telemetry decisions can inspect a connection
without caring *how* the connection works.

Scope is the first axis: 1P vs 3P. 1P = terminals.tech-owned infrastructure
and bundled utilities; 3P = anything external (HuggingFace, Anthropic,
RunPod, a customer's webhook). Kind, Direction, Transport, Trust refine the
declaration without coupling consumers to a fixed vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConnectionScope(str, Enum):
    """Which party operates the far end of the connection.

    ONE_P
        terminals.tech-operated services and first-party utilities bundled
        with packages we publish. Implies a higher default trust floor: we
        know the schema and the SLO.
    THREE_P
        External services (other companies' APIs, open-source projects,
        user-supplied webhooks). Always treated as untrusted until the
        spec's :class:`ConnectionTrust` level says otherwise.
    """

    ONE_P = "1p"
    THREE_P = "3p"


class ConnectionKind(str, Enum):
    """Functional domain of what flows across the connection."""

    TRAINING = "training"          # UnifiedBackend adapters (TRL, Unsloth, Verl, ...)
    ENVIRONMENT = "environment"    # BaseEnvironment / OpenEnv runtimes
    PROTOCOL = "protocol"          # A2A peers, MCP servers/clients
    PAYMENT = "payment"            # x402, wallet, facilitator
    SKILL = "skill"                # skill marketplace / Agent Skills
    MEMORY = "memory"              # MemoryStore backends / vector DBs
    MODEL = "model"                # LLM inference endpoints
    UTILITY = "utility"            # misc 1P helpers


class ConnectionDirection(str, Enum):
    """Who initiates and who serves."""

    INGRESS = "ingress"              # the remote calls us
    EGRESS = "egress"                # we call the remote
    BIDIRECTIONAL = "bidirectional"  # either side may initiate


class ConnectionTransport(str, Enum):
    """Wire protocol. Kept coarse — specific framings live in metadata."""

    HTTP = "http"
    SSE = "sse"
    WEBSOCKET = "websocket"
    STDIO = "stdio"
    SUBPROCESS = "subprocess"
    GRPC = "grpc"
    IN_PROCESS = "inproc"


class ConnectionTrust(str, Enum):
    """Identity / auth stance. Ordered — stronger trust implies weaker trust
    is also acceptable (a SIGNED connection can carry PUBLIC operations).

    Use :data:`TRUST_RANK` or the :class:`Tier`-style comparison operators
    below when gating.
    """

    PUBLIC = "public"                 # no auth required
    AUTHENTICATED = "authenticated"   # shared secret / API key / bearer
    SIGNED = "signed"                 # cryptographic identity (JWS, mTLS)
    METERED = "metered"               # payment-gated (x402 or similar)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ConnectionTrust):
            raise TypeError(
                f"cannot compare ConnectionTrust with {type(other).__name__}",
            )
        return _TRUST_RANK[self] >= _TRUST_RANK[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, ConnectionTrust):
            raise TypeError(
                f"cannot compare ConnectionTrust with {type(other).__name__}",
            )
        return _TRUST_RANK[self] > _TRUST_RANK[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, ConnectionTrust):
            raise TypeError(
                f"cannot compare ConnectionTrust with {type(other).__name__}",
            )
        return _TRUST_RANK[self] <= _TRUST_RANK[other]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ConnectionTrust):
            raise TypeError(
                f"cannot compare ConnectionTrust with {type(other).__name__}",
            )
        return _TRUST_RANK[self] < _TRUST_RANK[other]


_TRUST_RANK: dict[ConnectionTrust, int] = {
    ConnectionTrust.PUBLIC: 0,
    ConnectionTrust.AUTHENTICATED: 1,
    ConnectionTrust.SIGNED: 2,
    ConnectionTrust.METERED: 3,
}


def _empty_metadata() -> dict[str, Any]:
    """Typed factory so pyright strict mode infers the metadata field
    cleanly. ``field(default_factory=dict)`` would infer ``dict[Unknown,
    Unknown]`` because the bare ``dict`` callable is ungeneric."""
    return {}


@dataclass(frozen=True)
class ConnectionSpec:
    """Declarative metadata for a connection.

    Pure data, no behaviour. Policy lives in :class:`BaseConnection` and its
    subclasses; this object is what routing/auth/tier/telemetry layers read
    without having to know the transport internals.

    Attributes
    ----------
    name
        Stable string identifier, e.g. ``"a2a.peer.my-partner"`` or
        ``"mcp.server.carl-studio"``. Used for telemetry correlation.
    scope
        1P (terminals.tech-owned) vs 3P (external). See :class:`ConnectionScope`.
    kind
        Functional domain — training, environment, protocol, ... See
        :class:`ConnectionKind`.
    direction
        Who initiates. See :class:`ConnectionDirection`.
    transport
        Wire protocol. See :class:`ConnectionTransport`.
    trust
        Minimum identity / auth level required. See :class:`ConnectionTrust`.
    version
        Free-form version string (e.g., the remote protocol version we
        negotiated). Default ``"0"``.
    endpoint
        Optional address (URL, pipe name, package dotted path). Null for
        in-process connections.
    metadata
        Arbitrary extra fields. Must be JSON-serializable values; we do not
        validate beyond that here.
    """

    name: str
    scope: ConnectionScope
    kind: ConnectionKind
    direction: ConnectionDirection
    transport: ConnectionTransport
    trust: ConnectionTrust
    version: str = "0"
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for logs / telemetry."""
        return {
            "name": self.name,
            "scope": self.scope.value,
            "kind": self.kind.value,
            "direction": self.direction.value,
            "transport": self.transport.value,
            "trust": self.trust.value,
            "version": self.version,
            "endpoint": self.endpoint,
            "metadata": {**self.metadata},
        }


__all__ = [
    "ConnectionScope",
    "ConnectionKind",
    "ConnectionDirection",
    "ConnectionTransport",
    "ConnectionTrust",
    "ConnectionSpec",
]
