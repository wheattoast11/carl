"""CARL agent FSM states and transition table."""
from __future__ import annotations

from enum import Enum


class AgentState(Enum):
    IDLE = "idle"               # Waiting for trigger or schedule
    OBSERVE = "observe"         # Reading metrics, probing environment
    HYPOTHESIZE = "hypothesize" # Forming hypothesis from observations
    EXECUTE = "execute"         # Running experiment (training job)
    GATE = "gate"               # Evaluating results against criteria
    PROMOTE = "promote"         # Pushing successful results
    SHADOW = "shadow"           # Running shadow env with new weights


# Valid state transitions
TRANSITIONS: dict[AgentState, list[AgentState]] = {
    AgentState.IDLE:        [AgentState.OBSERVE],
    AgentState.OBSERVE:     [AgentState.HYPOTHESIZE, AgentState.IDLE],
    AgentState.HYPOTHESIZE: [AgentState.EXECUTE, AgentState.IDLE],
    AgentState.EXECUTE:     [AgentState.GATE],
    AgentState.GATE:        [AgentState.PROMOTE, AgentState.HYPOTHESIZE],
    AgentState.PROMOTE:     [AgentState.SHADOW, AgentState.IDLE],
    AgentState.SHADOW:      [AgentState.OBSERVE, AgentState.IDLE],
}


def valid_transition(from_state: AgentState, to_state: AgentState) -> bool:
    """Check if a state transition is valid per the FSM."""
    return to_state in TRANSITIONS.get(from_state, [])
