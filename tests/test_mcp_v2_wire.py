"""Actual SDK dispatch and resolver checks in both protocol eras."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.server.mcpserver import Context
from mcp.types import CreateMessageResult, ElicitResult, TextContent


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_wire_schema_and_host_sampling(mode: str) -> None:
    from carl_studio.mcp.server import mcp

    async def sampling(ctx, params):
        return CreateMessageResult(
            role="assistant",
            model="offline-test",
            content=TextContent(type="text", text="sample answer"),
            stop_reason="endTurn",
        )

    async def exercise():
        async with Client(mcp, mode=mode, sampling_callback=sampling) as client:
            tools = (await client.list_tools()).tools
            metrics = next(tool for tool in tools if tool.name == "get_coherence_metrics")
            assert metrics.output_schema["type"] == "object"
            result = await client.call_tool("get_coherence_metrics", {"logits_summary": "{}"})
            assert not result.is_error
            assert result.structured_content["kappa"] > 0
            result = await client.call_tool("sample_host", {"prompt": "sample"})
            assert not result.is_error
            assert "sample answer" in result.content[0].text

    asyncio.run(exercise())


def test_modern_http_cannot_choose_its_caller_identity() -> None:
    import httpx2

    from carl_studio.harness.runtime import close_runtime, get_runtime
    from carl_studio.mcp.protocol import CARLMCPServer
    from carl_studio.mcp.server import bind_connection

    async def exercise():
        bind_connection(None)
        probe = CARLMCPServer("caller-boundary-test")

        @probe.tool()
        async def owner_probe(ctx: Context) -> str:
            return get_runtime(ctx).context.owner

        app = probe.streamable_http_app(json_response=True)
        try:
            async with (
                probe.session_manager.run(),
                httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app),
                    base_url="http://127.0.0.1:8000",
                ) as client,
            ):
                response = await client.post(
                    "/mcp",
                    headers={
                        "mcp-protocol-version": "2026-07-28",
                        "mcp-method": "tools/call",
                        "mcp-name": "owner_probe",
                        "mcp-session-id": "UNISSUED_CHOSEN_TOKEN",
                        "accept": "application/json, text/event-stream",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "owner_probe",
                            "arguments": {},
                            "_meta": {
                                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                                "io.modelcontextprotocol/clientCapabilities": {},
                            },
                        },
                    },
                )
                assert response.json()["result"]["isError"] is True
                assert "UNISSUED_CHOSEN_TOKEN" not in response.text
        finally:
            await close_runtime()

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_validation_error_logs_do_not_capture_instruction(mode: str, caplog) -> None:
    from carl_studio.harness.runtime import close_runtime
    from carl_studio.mcp.connection import MCPServerConnection
    from carl_studio.mcp.server import bind_connection, mcp

    async def exercise():
        bind_connection(MCPServerConnection())
        try:
            async with Client(mcp, mode=mode) as client:
                result = await client.call_tool(
                    "delegate_agent",
                    {
                        "host": "codex",
                        "instruction": "PRIVATE_CANARY_" + "x" * 262144,
                    },
                )
                assert result.is_error
                assert "PRIVATE_CANARY_" not in caplog.text
                assert "PRIVATE_CANARY_" not in str(result)
        finally:
            await close_runtime()
            bind_connection(None)

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_declined_scope_has_no_task_or_process(tmp_path: Path, monkeypatch, mode: str) -> None:
    from carl_studio.harness.runtime import close_runtime
    from carl_studio.mcp.connection import MCPServerConnection
    from carl_studio.mcp.server import bind_connection, mcp
    from carl_studio.mcp.tasks import MCPTaskStore, set_default_store

    store = MCPTaskStore(tmp_path / "mcp_tasks.db")
    set_default_store(store)
    monkeypatch.setenv("CARL_WORKSPACE_ROOT", str(tmp_path))

    async def decline(ctx, params):
        return ElicitResult(action="decline")

    async def exercise():
        connection = MCPServerConnection()
        bind_connection(connection)
        try:
            async with Client(mcp, mode=mode, elicitation_callback=decline) as client:
                result = await client.call_tool(
                    "delegate_agent",
                    {"host": "codex", "instruction": "do work", "workdir": str(tmp_path.parent)},
                )
                assert result.is_error
                assert store.list() == []
        finally:
            await close_runtime()
            bind_connection(None)
            set_default_store(None)
            store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_delegation_owner_and_relative_directory_survive_wire_requests(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    import json

    from test_harness_delegation import FIXTURE

    from carl_studio.harness.runtime import close_runtime
    from carl_studio.mcp.connection import MCPServerConnection
    from carl_studio.mcp.server import bind_connection, mcp
    from carl_studio.mcp.tasks import MCPTaskStore, set_default_store

    script = tmp_path / "native-fixture"
    script.write_text(FIXTURE)
    script.chmod(0o700)
    monkeypatch.setattr("carl_studio.harness.runtime.shutil.which", lambda name: str(script))
    monkeypatch.setenv("CARL_WORKSPACE_ROOT", str(tmp_path))
    store = MCPTaskStore(tmp_path / "mcp_tasks.db")
    set_default_store(store)
    bind_connection(MCPServerConnection())

    async def exercise():
        try:
            async with Client(mcp, mode=mode) as client:
                handle = await client.call_tool(
                    "delegate_agent", {"host": "codex", "instruction": "wait"}
                )
                assert not handle.is_error
                value = json.loads(handle.content[0].text)
                task_id = value["task_id"]
                state = await client.call_tool("tasks_get", {"task_id": task_id})
                assert not state.is_error
                task = store.get(task_id)
                assert task.metadata["workdir"] == str(tmp_path)
                await asyncio.sleep(0.2)
                cancelled = await client.call_tool("tasks_cancel", {"task_id": task_id})
                assert not cancelled.is_error
                assert json.loads(cancelled.content[0].text)["cancelled"] is True
                assert store.get(task_id).status == "cancelled"
        finally:
            await close_runtime()
            bind_connection(None)
            set_default_store(None)
            store.close()

    asyncio.run(exercise())
