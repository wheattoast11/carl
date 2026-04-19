"""Tests for carl_core.connection.lifecycle — FSM correctness."""
from __future__ import annotations

from carl_core.connection.lifecycle import (
    VALID_TRANSITIONS,
    ConnectionState,
    can_transition,
)


def test_every_state_has_an_entry() -> None:
    """Every ConnectionState value has a VALID_TRANSITIONS entry
    (possibly empty). Prevents accidental dangling states on refactor."""
    for state in ConnectionState:
        assert state in VALID_TRANSITIONS


def test_transitions_reference_only_known_states() -> None:
    """Every target state in VALID_TRANSITIONS is itself a ConnectionState."""
    known = set(ConnectionState)
    for src, targets in VALID_TRANSITIONS.items():
        for target in targets:
            assert target in known, f"{src.value} -> {target} references unknown state"


def test_closed_is_terminal_with_no_outbound() -> None:
    assert VALID_TRANSITIONS[ConnectionState.CLOSED] == frozenset()
    assert ConnectionState.CLOSED.is_terminal


def test_error_is_terminal_but_can_close() -> None:
    """ERROR is terminal but cleanup can still flip to CLOSED for resource
    accounting. That's the only legal transition out of ERROR."""
    assert ConnectionState.ERROR.is_terminal
    assert VALID_TRANSITIONS[ConnectionState.ERROR] == frozenset(
        {ConnectionState.CLOSED}
    )


def test_init_can_go_to_connecting() -> None:
    assert can_transition(ConnectionState.INIT, ConnectionState.CONNECTING)


def test_init_cannot_jump_to_ready() -> None:
    assert not can_transition(ConnectionState.INIT, ConnectionState.READY)


def test_ready_to_transacting_and_back() -> None:
    """Hot-path loop: READY <-> TRANSACTING."""
    assert can_transition(ConnectionState.READY, ConnectionState.TRANSACTING)
    assert can_transition(ConnectionState.TRANSACTING, ConnectionState.READY)


def test_degraded_can_recover_or_close() -> None:
    assert can_transition(ConnectionState.DEGRADED, ConnectionState.READY)
    assert can_transition(ConnectionState.DEGRADED, ConnectionState.CLOSING)
    assert can_transition(ConnectionState.DEGRADED, ConnectionState.CLOSED)


def test_closed_has_no_way_out() -> None:
    for state in ConnectionState:
        assert not can_transition(ConnectionState.CLOSED, state)


def test_closing_can_only_reach_terminal() -> None:
    targets = VALID_TRANSITIONS[ConnectionState.CLOSING]
    assert targets.issubset({ConnectionState.CLOSED, ConnectionState.ERROR})


def test_public_trust_skip_auth_path_exists() -> None:
    """CONNECTING may go directly to READY (when trust=PUBLIC, we skip
    AUTHENTICATING)."""
    assert can_transition(ConnectionState.CONNECTING, ConnectionState.READY)
    assert can_transition(
        ConnectionState.CONNECTING, ConnectionState.AUTHENTICATING
    )


def test_no_self_loops_without_explicit_intent() -> None:
    """Self-loops are almost always a bug — none defined by default."""
    for state, targets in VALID_TRANSITIONS.items():
        assert state not in targets, f"{state.value} loops to itself"


def test_is_terminal_only_on_closed_and_error() -> None:
    terminals = {s for s in ConnectionState if s.is_terminal}
    assert terminals == {ConnectionState.CLOSED, ConnectionState.ERROR}
