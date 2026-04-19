"""Property tests for the connection FSM.

We use hypothesis to enumerate transition paths and verify two invariants:

1. Every reachable target from a source state is explicitly listed in
   :data:`VALID_TRANSITIONS`.
2. Driving a concrete :class:`BaseConnection` through the same transitions
   via ``_transition`` never puts it into an inconsistent state (no torn
   reads, no forbidden targets accepted).
"""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from carl_core.connection.base import BaseConnection
from carl_core.connection.errors import ConnectionTransitionError
from carl_core.connection.lifecycle import (
    VALID_TRANSITIONS,
    ConnectionState,
    can_transition,
)
from carl_core.connection.registry import reset_registry
from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)


_SPEC = ConnectionSpec(
    name="demo.property",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.UTILITY,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
)


class _Conn(BaseConnection):
    spec = _SPEC

    def _connect(self) -> None: ...
    def _close(self) -> None: ...


_STATES = list(ConnectionState)


@given(
    src=st.sampled_from(_STATES),
    dst=st.sampled_from(_STATES),
)
def test_can_transition_matches_valid_transitions(
    src: ConnectionState, dst: ConnectionState
) -> None:
    assert can_transition(src, dst) == (dst in VALID_TRANSITIONS[src])


@given(
    src=st.sampled_from(_STATES),
    dst=st.sampled_from(_STATES),
)
def test_transition_accepts_valid_rejects_invalid(
    src: ConnectionState, dst: ConnectionState
) -> None:
    """Driving _transition should either succeed (if legal) or raise
    ConnectionTransitionError — never land silently on the wrong state."""
    reset_registry()
    conn = _Conn()
    # Force the source state directly for the test (bypass FSM to set it up).
    conn._state = src  # pyright: ignore[reportPrivateUsage]
    legal = dst in VALID_TRANSITIONS[src]
    if legal:
        conn._transition(dst)  # pyright: ignore[reportPrivateUsage]
        assert conn.state == dst
    else:
        try:
            conn._transition(dst)  # pyright: ignore[reportPrivateUsage]
        except ConnectionTransitionError:
            assert conn.state == src  # no partial state change
        else:
            # Should have raised — failure mode to catch.
            raise AssertionError(
                f"transition {src.value} -> {dst.value} was accepted "
                f"but is not in VALID_TRANSITIONS"
            )
