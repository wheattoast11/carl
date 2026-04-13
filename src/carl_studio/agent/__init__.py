"""CARL Autonomous Agent — PAID tier agentic loop.

FSM: IDLE -> OBSERVE -> HYPOTHESIZE -> EXECUTE -> GATE -> PROMOTE -> SHADOW
"""
from __future__ import annotations

from carl_studio.agent.states import AgentState, TRANSITIONS, valid_transition
from carl_studio.agent.scheduler import Scheduler, Schedule
from carl_studio.agent.loop import CARLAgent, Observation, Hypothesis, AgentLog
from carl_studio.agent.tier_gate import requires_paid, get_tier, TierError

__all__ = [
    "AgentState",
    "TRANSITIONS",
    "valid_transition",
    "Scheduler",
    "Schedule",
    "CARLAgent",
    "Observation",
    "Hypothesis",
    "AgentLog",
    "requires_paid",
    "get_tier",
    "TierError",
]
