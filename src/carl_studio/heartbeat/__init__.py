"""Heartbeat loop — background worker over the sticky-note queue.

The heartbeat drains :class:`~carl_studio.sticky.StickyQueue`, running every
claimed note through the ordered :class:`HeartbeatPhase` pipeline. Each cycle
records a ``HEARTBEAT_CYCLE`` pair on the shared
:class:`~carl_core.interaction.InteractionChain` so the operation is
auditable. :class:`HeartbeatConnection` wraps the loop as an
:class:`~carl_core.connection.AsyncBaseConnection` so it participates in the
global registry and shares lifecycle semantics with every other CARL channel.
"""

from __future__ import annotations

from carl_studio.heartbeat.connection import HeartbeatConnection
from carl_studio.heartbeat.loop import HeartbeatLoop
from carl_studio.heartbeat.phases import ORDERED_PHASES, HeartbeatPhase

__all__ = [
    "HeartbeatConnection",
    "HeartbeatLoop",
    "HeartbeatPhase",
    "ORDERED_PHASES",
]
