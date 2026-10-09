"""Native delegation and era-portable host input on the existing MCP server."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, cast

from mcp.server.mcpserver import Context, Elicit, Resolve, Sample
from mcp.types import CreateMessageResult, SamplingMessage, TextContent
from pydantic import BaseModel, Field, ValidationError, create_model

from carl_studio.harness.types import DelegationContext, DelegationRequest


class WorkspaceGrant(BaseModel):
    """A caller's explicit permission for the selected workspace."""

    approved: bool
    directory: str


async def workspace_grant(
    ctx: Context, workdir: str = ".", write: bool = False
) -> WorkspaceGrant | Elicit[WorkspaceGrant]:
    """Resolve additional scope before a delegated process can start."""
    from carl_core.errors import CARLError
    from carl_core.safepath import safe_resolve

    from carl_studio.harness.runtime import get_runtime

    context = get_runtime(ctx).context
    try:
        directory = safe_resolve(workdir, context.root, must_exist=True)
        if not write or context.allow_write:
            return WorkspaceGrant(approved=True, directory=str(directory))
    except CARLError:
        pass
    raw = Path(workdir)
    path = (raw if raw.is_absolute() else context.root / raw).resolve()
    if not path.is_dir():
        raise CARLError("Workspace does not exist", code="carl.agent.workdir")
    bound = create_model(
        "BoundWorkspaceGrant",
        __base__=WorkspaceGrant,
        directory=(str, Field(default=str(path), pattern="^" + re.escape(str(path)) + "$")),
    )
    access = "read and edit" if write else "read"
    return Elicit(f"Allow this delegated agent to {access} files in {path}?", bound)


async def host_sample(prompt: str) -> Sample:
    """Ask the owning host to sample through the negotiated protocol."""
    return Sample(
        [SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
        max_tokens=1024,
    )


def register_harness_tools(server: Any) -> None:
    """Register on the existing MCP instance, without another dispatcher."""

    @server.tool()
    async def encode_data(data: dict[str, Any], ctx: Context) -> dict[str, Any]:
        from carl_studio.harness.runtime import get_runtime
        from carl_studio.mcp.server import _run_tool
        from carl_studio.training.preparation import run_in_worker
        async def body() -> dict[str, Any]:
            return await run_in_worker(lambda: get_runtime(ctx).session.semantic.encode_data(data))
        return await _run_tool("encode_data", body)

    @server.tool()
    async def interpret(record: dict[str, Any], ctx: Context) -> dict[str, Any]:
        from carl_studio.harness.runtime import get_runtime
        from carl_studio.mcp.server import _run_tool
        async def body() -> dict[str, Any]:
            return get_runtime(ctx).session.semantic.interpret(record)
        return await _run_tool("interpret", body)

    @server.tool()
    async def interpretation_feedback(interpretation_id: str, feedback_ref: str, confirmed: bool,
                                      ctx: Context, artifact_ref: str | None = None,
                                      correction: dict[str, Any] | None = None) -> dict[str, Any]:
        from carl_studio.harness.runtime import get_runtime
        from carl_studio.mcp.server import _run_tool
        async def body() -> dict[str, Any]:
            return get_runtime(ctx).session.semantic.interpretation_feedback(
                interpretation_id, feedback_ref, confirmed, artifact_ref, correction)
        return await _run_tool("interpretation_feedback", body)

    @server.tool()
    async def list_agent_harnesses() -> dict[str, Any]:
        from carl_studio.harness.adapters import list_harnesses

        return {"harnesses": list_harnesses()}

    @server.tool()
    async def delegate_agent(
        host: str,
        instruction: str,
        grant: Annotated[WorkspaceGrant, Resolve(workspace_grant)],
        ctx: Context,
        workdir: str = ".",
        write: bool = False,
        request_id: str | None = None,
        timeout_s: float = 600,
        model: str | None = None,
        native_session: str | None = None,
    ) -> dict[str, Any]:
        from carl_core.errors import CARLError

        from carl_studio.harness.runtime import get_runtime

        if not grant.approved:
            raise CARLError("Workspace permission declined", code="carl.agent.permission")
        runtime = get_runtime(ctx)

        directory = Path(grant.directory)
        context = DelegationContext(
            owner=runtime.context.owner,
            root=directory,
            allow_write=write,
            depth=runtime.context.depth,
        )
        arguments: dict[str, Any] = {
            "host": host,
            "instruction": instruction,
            "workdir": str(directory),
            "write": write,
            "timeout_s": timeout_s,
            "model": model,
            "native_session": native_session,
        }
        if request_id is not None:
            arguments["request_id"] = request_id
        try:
            request = DelegationRequest.model_validate(arguments)
        except ValidationError:
            raise CARLError("Invalid delegation request", code="carl.agent.validation") from None
        return await runtime.submit(request, approved_context=context)

    @server.tool()
    async def tasks_reply(
        task_id: str, request_id: str, ctx: Context, approve: bool = False
    ) -> dict[str, Any]:
        from carl_studio.harness.runtime import get_runtime

        return get_runtime(ctx).reply(task_id, request_id, approve)

    @server.tool()
    async def read_agent_result(task_id: str, ctx: Context) -> str:
        from carl_studio.harness.runtime import get_runtime

        return get_runtime(ctx).read_result(task_id)

    @server.tool()
    async def read_agent_input(task_id: str, ctx: Context) -> str:
        from carl_core.errors import CARLError

        from carl_studio.harness.runtime import get_runtime

        runtime = get_runtime(ctx)
        task = runtime.get(task_id)
        if "pending_input" not in task:
            raise CARLError("No pending input", code="carl.agent.stale_reply")
        ref = task["pending_input"]["details_ref"]["ref_id"]
        return runtime.session.data_toolkit.read_text(ref, max_bytes=65536)["text"]

    @server.tool()
    async def sample_host(
        prompt: str,
        answer: Annotated[CreateMessageResult, Resolve(host_sample)],
    ) -> dict[str, Any]:
        content: Any = answer.content
        blocks: list[Any] = cast(list[Any], content) if isinstance(content, list) else [content]
        return {
            "model": answer.model,
            "text": "".join(block.text for block in blocks if block.type == "text"),
        }
