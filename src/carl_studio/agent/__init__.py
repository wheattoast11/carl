"""CARL Autonomous Agent — PAID tier agentic loop.

FSM: IDLE -> OBSERVE -> HYPOTHESIZE -> EXECUTE -> GATE -> PROMOTE -> SHADOW

The FSM autonomy class is :class:`AutonomyAgent`. It was previously named
``CARLAgent`` but was renamed to avoid collision with
:class:`carl_studio.chat_agent.CARLAgent`, which is the canonical Anthropic
chat loop used by every ``carl`` CLI entry point.

``carl_studio.agent.CARLAgent`` remains temporarily as a deprecated alias
accessed via :func:`__getattr__` so existing imports emit
:class:`DeprecationWarning` instead of breaking. The alias will be removed
in v0.7.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from carl_studio.agent.loop import AgentLog, AutonomyAgent, Hypothesis, Observation
from carl_studio.agent.scheduler import Schedule, Scheduler
from carl_studio.agent.states import TRANSITIONS, AgentState, valid_transition
from carl_studio.agent.tier_gate import get_tier, requires_paid

if TYPE_CHECKING:
    # Re-export under the deprecated name for static type checkers. At runtime
    # the alias is served by ``__getattr__`` below so a deprecation warning is
    # raised on access.
    CARLAgent = AutonomyAgent

__all__ = [
    "AgentLog",
    "AgentState",
    "AutonomyAgent",
    "Hypothesis",
    "Observation",
    "Schedule",
    "Scheduler",
    "TRANSITIONS",
    "get_tier",
    "requires_paid",
    "valid_transition",
]


def __getattr__(name: str) -> object:
    """Serve deprecated ``CARLAgent`` alias with a warning on access.

    The alias is intentionally not included in ``__all__`` so ``from
    carl_studio.agent import *`` does not re-introduce the collision. Direct
    imports (``from carl_studio.agent import CARLAgent``) still work, but emit
    a :class:`DeprecationWarning` at import time.
    """
    if name == "CARLAgent":
        warnings.warn(
            "carl_studio.agent.CARLAgent has been renamed to AutonomyAgent "
            "to avoid collision with carl_studio.chat_agent.CARLAgent. "
            "Update imports. Alias will be removed in v0.7.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AutonomyAgent
    raise AttributeError(f"module 'carl_studio.agent' has no attribute {name!r}")
