"""Synchronous chat tools backed by a Session-owned async execution loop."""

from __future__ import annotations

import asyncio
import atexit
import threading
import uuid
from pathlib import Path
from typing import Any

from carl_studio.harness.runtime import HarnessRuntime
from carl_studio.harness.types import DelegationContext, DelegationRequest
from carl_studio.mcp.tasks import get_default_store
from carl_studio.session import Session


class HarnessBridge:
    """Keep delegated jobs alive across successive synchronous chat tool calls."""

    def __init__(self, session: Session, root: Path, *, allow_write: bool = False) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.runtime = HarnessRuntime(
            session,
            DelegationContext.local(owner=uuid.uuid4().hex, root=root, allow_write=allow_write),
            get_default_store(),
        )
        self.closed = False
        atexit.register(self.close)

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch through the same task operations exposed by MCP."""
        if name == "delegate_agent":
            future = asyncio.run_coroutine_threadsafe(
                self.runtime.submit(DelegationRequest.model_validate(arguments)), self.loop
            )
            return future.result(timeout=5)
        task_id = arguments["task_id"]
        if name == "tasks_get":
            return self.runtime.get(task_id)
        if name == "tasks_cancel":
            return asyncio.run_coroutine_threadsafe(self.runtime.cancel(task_id), self.loop).result(
                timeout=10
            )
        if name == "tasks_reply":
            return self.runtime.reply(
                task_id, arguments["request_id"], arguments.get("approve", False)
            )
        if name == "read_agent_result":
            return self.runtime.read_result(task_id)
        raise ValueError("Unknown harness operation")

    def close(self) -> None:
        """Reap owned executions before ending the bridge loop."""
        if self.closed:
            return
        self.closed = True
        try:
            asyncio.run_coroutine_threadsafe(self.runtime.close(), self.loop).result(timeout=15)
        finally:

            async def shutdown() -> None:
                await self.loop.shutdown_asyncgens()
                await self.loop.shutdown_default_executor()

            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(timeout=5)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            self.loop.close()
            atexit.unregister(self.close)
