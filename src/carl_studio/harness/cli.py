"""User-directed harness commands mounted on CARL's existing agent app."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from carl_core.errors import CARLError

from carl_studio.harness.adapters import list_harnesses
from carl_studio.harness.runtime import HarnessRuntime
from carl_studio.harness.types import DelegationContext, DelegationRequest
from carl_studio.mcp.tasks import MCPTaskStore
from carl_studio.session import Session


def register_commands(app: typer.Typer) -> None:
    """Use the existing agent command namespace for harness operations."""

    @app.command("harnesses")
    def harnesses() -> None:
        typer.echo(json.dumps({"harnesses": list_harnesses()}, indent=2))

    @app.command("delegate")
    def delegate(
        host: str,
        workdir: Annotated[Path, typer.Option("--workdir")] = Path("."),
        write: bool = typer.Option(False, "--write"),
        timeout: float = typer.Option(600, "--timeout"),
        model: str | None = typer.Option(None, "--model"),
    ) -> None:
        """Read an instruction from stdin, stream state, and await owned execution."""
        instruction = sys.stdin.read(262145)
        arguments = {
            "host": host,
            "instruction": instruction,
            "workdir": workdir,
            "write": write,
            "timeout_s": timeout,
            "model": model,
        }

        async def execute() -> None:
            session = Session()
            store = MCPTaskStore()
            context = DelegationContext.local(
                owner=uuid.uuid4().hex, root=workdir, allow_write=write
            )
            runtime = HarnessRuntime(session, context, store)
            try:
                handle = await runtime.submit(DelegationRequest.model_validate(arguments))
                task_id = handle["task_id"]
                typer.echo(json.dumps({"task_id": task_id, "status": handle["status"]}))
                while True:
                    state = runtime.get(task_id)
                    if state["status"] in {"completed", "failed", "cancelled"}:
                        typer.echo(json.dumps(state))
                        if state["status"] == "completed":
                            typer.echo(runtime.read_result(task_id))
                        else:
                            raise typer.Exit(1)
                        return
                    if state.get("pending_input"):
                        pending = state["pending_input"]
                        details = runtime.session.data_toolkit.read_text(
                            pending["details_ref"]["ref_id"], max_bytes=65536
                        )["text"]
                        typer.echo(details)
                        if sys.stdout.isatty():
                            allowed = await asyncio.to_thread(
                                typer.confirm, "Approve this native tool request?"
                            )
                        else:
                            allowed = False
                        runtime.reply(task_id, pending["request_id"], allowed)
                    await asyncio.sleep(0.1)
            finally:
                await runtime.close()
                store.close()

        try:
            asyncio.run(execute())
        except CARLError as exc:
            typer.echo(f"{exc.code}: {exc}", err=True)
            raise typer.Exit(1) from exc

    def operator_task(task_id: str) -> tuple[MCPTaskStore, Any]:
        store = MCPTaskStore()
        task = store.get(task_id)
        if task is None or task.tool_name != "delegate_agent":
            store.close()
            raise typer.BadParameter("Unknown delegated task")
        return store, task

    @app.command("cancel")
    def cancel(task_id: str) -> None:
        store, task = operator_task(task_id)
        try:
            if not task.is_terminal:
                store.update_metadata(task_id, {"cancel_requested": True})
            typer.echo(
                json.dumps({"task_id": task_id, "cancellation_requested": not task.is_terminal})
            )
        finally:
            store.close()

    @app.command("result")
    def result(task_id: str) -> None:
        store, task = operator_task(task_id)
        try:
            session = Session()
            runtime = HarnessRuntime(
                session,
                DelegationContext(
                    owner=task.metadata["owner"], root=Path(task.metadata["workdir"])
                ),
                store,
            )
            typer.echo(runtime.read_result(task_id))
            session.teardown()
        finally:
            store.close()

    @app.command("reply")
    def reply(
        task_id: str, request_id: str, approve: bool = typer.Option(False, "--approve")
    ) -> None:
        store, task = operator_task(task_id)
        try:
            pending = task.metadata.get("pending_input")
            if task.is_terminal or not pending or pending.get("request_id") != request_id:
                raise typer.BadParameter("Permission request is stale")
            if approve and pending.get("requires_write") and not task.metadata.get("write"):
                raise typer.BadParameter("Approval exceeds the task's grant")
            store.update_metadata(
                task_id,
                {
                    "pending_reply": {
                        "request_id": request_id,
                        "approve": approve,
                        "generation": task.metadata["generation"],
                    }
                },
            )
            typer.echo(json.dumps({"task_id": task_id, "reply_queued": True}))
        finally:
            store.close()
