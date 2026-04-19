"""Connection lifecycle finite state machine.

The FSM models every connection from opening handshake to terminal close.
Transitions outside :data:`VALID_TRANSITIONS` raise
:class:`~carl_core.connection.errors.ConnectionTransitionError` — callers do
not have to defensively check state before every operation, because illegal
sequences are rejected at the boundary.

States
------
Non-terminal:
    INIT             created, not yet connecting
    CONNECTING       transport handshake in flight
    AUTHENTICATING   identity / credential negotiation
    READY            able to transact
    TRANSACTING      actively handling a request
    DEGRADED         circuit-breaker / probe failure; reads may still work
    CLOSING          graceful shutdown in progress

Terminal:
    CLOSED           clean exit
    ERROR            unrecoverable failure; may transition to CLOSED
"""

from __future__ import annotations

from enum import Enum


class ConnectionState(str, Enum):
    """States for a connection's lifecycle FSM."""

    INIT = "init"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    READY = "ready"
    TRANSACTING = "transacting"
    DEGRADED = "degraded"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        """``True`` if no outbound transitions are possible (CLOSED) or only
        a cleanup transition to CLOSED is allowed (ERROR).
        """
        return self in _TERMINAL_STATES


_TERMINAL_STATES: frozenset[ConnectionState] = frozenset(
    {ConnectionState.CLOSED, ConnectionState.ERROR}
)


# Allowed transitions. Any target not in the set for a given source state
# triggers ConnectionTransitionError.
#
# Design notes:
#   * INIT can skip straight to CLOSED (never opened).
#   * CONNECTING may skip AUTHENTICATING when trust is PUBLIC.
#   * READY <-> TRANSACTING is the hot loop during normal operation.
#   * DEGRADED permits self-recovery back to READY or graceful CLOSING.
#   * ERROR is one-way: only CLOSED can follow, so resources can be freed.
#   * CLOSED is a black hole; no outbound transitions.
VALID_TRANSITIONS: dict[ConnectionState, frozenset[ConnectionState]] = {
    ConnectionState.INIT: frozenset(
        {
            ConnectionState.CONNECTING,
            ConnectionState.CLOSED,
            ConnectionState.ERROR,
        }
    ),
    ConnectionState.CONNECTING: frozenset(
        {
            ConnectionState.AUTHENTICATING,
            ConnectionState.READY,
            ConnectionState.ERROR,
            ConnectionState.CLOSED,
        }
    ),
    ConnectionState.AUTHENTICATING: frozenset(
        {
            ConnectionState.READY,
            ConnectionState.ERROR,
            ConnectionState.CLOSED,
        }
    ),
    ConnectionState.READY: frozenset(
        {
            ConnectionState.TRANSACTING,
            ConnectionState.DEGRADED,
            ConnectionState.CLOSING,
            ConnectionState.CLOSED,
            ConnectionState.ERROR,
        }
    ),
    ConnectionState.TRANSACTING: frozenset(
        {
            ConnectionState.READY,
            ConnectionState.DEGRADED,
            ConnectionState.ERROR,
            ConnectionState.CLOSING,
            ConnectionState.CLOSED,
        }
    ),
    ConnectionState.DEGRADED: frozenset(
        {
            ConnectionState.READY,
            ConnectionState.CLOSING,
            ConnectionState.CLOSED,
            ConnectionState.ERROR,
        }
    ),
    ConnectionState.CLOSING: frozenset(
        {
            ConnectionState.CLOSED,
            ConnectionState.ERROR,
        }
    ),
    ConnectionState.CLOSED: frozenset(),
    ConnectionState.ERROR: frozenset({ConnectionState.CLOSED}),
}


def can_transition(from_state: ConnectionState, to_state: ConnectionState) -> bool:
    """Return True if ``from_state -> to_state`` is a valid FSM transition."""
    return to_state in VALID_TRANSITIONS.get(from_state, frozenset())


__all__ = [
    "ConnectionState",
    "VALID_TRANSITIONS",
    "can_transition",
]
