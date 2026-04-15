"""CARL Studio A2A — Agent-to-Agent protocol.

Based loosely on Google A2A spec (https://google.github.io/A2A/).
Adapted for CARL training workflows.

Local transport: SQLite message queue (no network needed).
Cloud transport: Supabase Realtime (PAID tier, stub for now).
"""

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.bus import LocalBus, SupabaseBus
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.task import A2ATask, A2ATaskStatus

__all__ = [
    "CARLAgentCard",
    "A2ATask",
    "A2ATaskStatus",
    "A2AMessage",
    "LocalBus",
    "SupabaseBus",
]
