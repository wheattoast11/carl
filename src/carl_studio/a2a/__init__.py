"""CARL Studio A2A — Agent-to-Agent protocol.

Based loosely on Google A2A spec (https://google.github.io/A2A/).
Adapted for CARL training workflows.

Local transport: SQLite message queue (no network needed).
Cloud transport: Supabase Realtime (PAID tier, stub for now).
"""

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.bus import LocalBus, SupabaseBus
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.spec import (
    agent_card_to_spec,
    message_send_to_task,
    task_to_jsonrpc_result,
    wrap_jsonrpc_error,
    wrap_jsonrpc_response,
)
from carl_studio.a2a.task import A2ATask, A2ATaskStatus

__all__ = [
    "CARLAgentCard",
    "A2ATask",
    "A2ATaskStatus",
    "A2AMessage",
    "LocalBus",
    "SupabaseBus",
    "agent_card_to_spec",
    "task_to_jsonrpc_result",
    "message_send_to_task",
    "wrap_jsonrpc_response",
    "wrap_jsonrpc_error",
]
