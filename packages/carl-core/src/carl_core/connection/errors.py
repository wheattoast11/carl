"""Typed errors for carl_core.connection.

The class name ``CARLConnectionError`` is deliberately prefixed so we do not
shadow the stdlib ``ConnectionError`` — which is already used as a
retryable-exception type in :mod:`carl_core.retry`. Subclasses follow the
``carl.connection.<reason>`` code convention from
:mod:`carl_core.errors`.
"""

from __future__ import annotations

from carl_core.errors import CARLError


class CARLConnectionError(CARLError):
    """Base class for all carl-connection errors."""

    code = "carl.connection"


class ConnectionUnavailableError(CARLConnectionError):
    """Transport is unreachable or required package / binary is missing."""

    code = "carl.connection.unavailable"


class ConnectionAuthError(CARLConnectionError):
    """Identity / credential negotiation failed."""

    code = "carl.connection.auth"


class ConnectionTransitionError(CARLConnectionError):
    """An illegal FSM transition was attempted.

    Context always carries ``from``, ``to``, ``allowed``, and the connection
    identity so the traceback is actionable.
    """

    code = "carl.connection.transition"


class ConnectionClosedError(CARLConnectionError):
    """Operation attempted on a closed, errored, or not-yet-ready connection."""

    code = "carl.connection.closed"


class ConnectionPolicyError(CARLConnectionError):
    """Operation violated the connection's declared policy.

    Covers trust-level mismatches, tier gating, quota exhaustion, and other
    above-the-transport rules. Concrete subclasses may further refine via
    ``code`` suffixes like ``carl.connection.policy.tier``.
    """

    code = "carl.connection.policy"


__all__ = [
    "CARLConnectionError",
    "ConnectionUnavailableError",
    "ConnectionAuthError",
    "ConnectionTransitionError",
    "ConnectionClosedError",
    "ConnectionPolicyError",
]
