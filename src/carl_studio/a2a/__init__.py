"""CARL Studio A2A — Agent-to-Agent protocol.

Based on the Linux Foundation A2A v1.0 spec
(https://github.com/linuxfoundation/a2a). Adapted for CARL training
workflows.

Transport layers:

* Local: SQLite message queue (no network needed).
* Cloud: Supabase Realtime (PAID tier, stub for now).

The :mod:`carl_studio.a2a.connection` module provides
:class:`AsyncBaseConnection`-backed wrappers (:class:`A2APeerConnection`
for EGRESS, :class:`A2AServerConnection` for INGRESS) that route every
request through ``carl_core.connection``'s FSM / telemetry / registry.

A2A v1.0 additions (this module surface-level):

* :mod:`carl_studio.a2a.streaming` — SSE helpers for ``message/stream``
  and ``tasks/subscribe``.
* :mod:`carl_studio.a2a.push` — push-notification CRUD + delivery worker.
* :mod:`carl_studio.a2a.identity` — JWS-signed AgentCard identity for
  ``agent/getAuthenticatedExtendedCard``.
"""

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.bus import LocalBus, SupabaseBus
from carl_studio.a2a.connection import (
    A2APeerConnection,
    A2AServerConnection,
    ProtocolConnection,
)
from carl_studio.a2a.identity import AgentIdentity
from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.push import (
    PushConfig,
    PushDeliveryWorker,
    PushSubscriberStore,
)
from carl_studio.a2a.spec import (
    agent_card_to_spec,
    message_send_to_task,
    task_to_jsonrpc_result,
    wrap_jsonrpc_error,
    wrap_jsonrpc_response,
)
from carl_studio.a2a.streaming import (
    EVENT_ARTIFACT,
    EVENT_STATUS,
    parse_sse_events,
    stream_message,
    stream_task_updates,
)
from carl_studio.a2a.task import A2ATask, A2ATaskStatus

__all__ = [
    "CARLAgentCard",
    "A2ATask",
    "A2ATaskStatus",
    "A2AMessage",
    "LocalBus",
    "SupabaseBus",
    "ProtocolConnection",
    "A2APeerConnection",
    "A2AServerConnection",
    "agent_card_to_spec",
    "task_to_jsonrpc_result",
    "message_send_to_task",
    "wrap_jsonrpc_response",
    "wrap_jsonrpc_error",
    # v1.0 conformance additions
    "AgentIdentity",
    "PushConfig",
    "PushSubscriberStore",
    "PushDeliveryWorker",
    "EVENT_STATUS",
    "EVENT_ARTIFACT",
    "stream_task_updates",
    "stream_message",
    "parse_sse_events",
]
