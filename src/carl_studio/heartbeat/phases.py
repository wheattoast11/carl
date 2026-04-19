"""Heartbeat phase ordering.

Each queued sticky note flows through every phase exactly once per cycle
(``INTERVIEW → EXPLORE → ... → AWAIT``). Phases are intentionally small and
named for the *intent* of the step rather than an implementation detail —
concrete behaviour is plugged in by :class:`~carl_studio.heartbeat.loop.HeartbeatLoop`
or a subclass thereof.
"""

from __future__ import annotations

from enum import Enum


class HeartbeatPhase(str, Enum):
    """Ordered stages of a single heartbeat cycle."""

    INTERVIEW = "interview"
    EXPLORE = "explore"
    RESEARCH = "research"
    PLAN = "plan"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    RECOMMEND = "recommend"
    AWAIT = "await"


ORDERED_PHASES: tuple[HeartbeatPhase, ...] = (
    HeartbeatPhase.INTERVIEW,
    HeartbeatPhase.EXPLORE,
    HeartbeatPhase.RESEARCH,
    HeartbeatPhase.PLAN,
    HeartbeatPhase.EXECUTE,
    HeartbeatPhase.EVALUATE,
    HeartbeatPhase.RECOMMEND,
    HeartbeatPhase.AWAIT,
)


__all__ = ["HeartbeatPhase", "ORDERED_PHASES"]
