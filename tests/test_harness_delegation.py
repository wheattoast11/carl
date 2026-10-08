"""Real child-process execution, task ownership, and cancellation witnesses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from carl_core.errors import CARLError

from carl_studio.harness.runtime import HarnessRuntime
from carl_studio.harness.types import DelegationContext, DelegationRequest
from carl_studio.mcp.tasks import MCPTaskStore
from carl_studio.session import Session

FIXTURE = """#!/usr/bin/env python3
import json,sys,time
def emit(value):
 print(json.dumps(value),flush=True)
host = sys.argv[1] if len(sys.argv)>1 else 'claude'
for line in sys.stdin:
 value=json.loads(line)
 if host=='app-server':
  method=value.get('method')
  if method=='initialize': emit({'id':0,'result':{}})
  elif method=='thread/start':
   emit({'id':1,'result':{'thread':{'id':'owned'}}})
   emit({'method':'mcpServer/startupStatus/updated','params':{'name':'carl','status':'ready'}})
  elif method=='turn/start':
   emit({'id':2,'result':{'turn':{'id':'turn'}}})
   text=value['params']['input'][0]['text']
   if text=='wait': continue
   if text=='approve':
    emit({'id':'approval','method':'item/fileChange/requestApproval',
          'params':{'threadId':'owned','turnId':'turn','itemId':'change'}})
   else:
    emit({'method':'item/agentMessage/delta','params':{'threadId':'owned','turnId':'turn','delta':'native answer'}})
    emit({'method':'turn/completed','params':{'threadId':'owned','turn':{'id':'turn','status':'completed'}}})
  elif method=='turn/steer':
   if value['params']['input'][0]['text']=='reject':
    emit({'id':value['id'],'error':{'code':-1,'message':'rejected'}})
   else: emit({'id':value['id'],'result':{'turnId':'turn'}})
  elif value.get('id')=='approval':
   emit({'method':'item/agentMessage/delta','params':{'threadId':'owned','turnId':'turn',
         'delta':'approved' if value['result']['decision']=='accept' else 'declined'}})
   emit({'method':'turn/completed','params':{'threadId':'owned','turn':{'id':'turn','status':'completed'}}})
  elif method=='turn/interrupt':
   emit({'id':99,'result':{}})
   emit({'method':'turn/completed','params':{'threadId':'owned','turn':{'id':'turn','status':'interrupted'}}})
 else:
  if value.get('type')=='control_request':
   if value['request']['subtype']=='initialize':
    emit({'type':'control_response','response':{'subtype':'success','request_id':'carl-init','response':{}}})
   elif value['request']['subtype']=='interrupt': sys.exit(0)
  elif value.get('type')=='user':
   if value['message']['content']=='wait': continue
   emit({'type':'assistant','message':{'content':[{'type':'thinking','thinking':'PRIVATE_REASONING'},
         {'type':'text','text':'native answer'}]}})
   emit({'type':'result','is_error':False,'result':'native answer'})
"""


@pytest.fixture
def fixture_host(tmp_path: Path) -> Path:
    path = tmp_path / "native-fixture"
    path.write_text(FIXTURE)
    path.chmod(0o700)
    return path


def runtime(tmp_path: Path, owner: str = "caller", write: bool = False) -> HarnessRuntime:
    return HarnessRuntime(
        Session(),
        DelegationContext(owner=owner, root=tmp_path, allow_write=write),
        MCPTaskStore(tmp_path / "mcp_tasks.db"),
    )


async def terminal(owner: HarnessRuntime, task_id: str) -> dict:
    for _ in range(100):
        task = owner.get(task_id)
        if task["status"] in {"completed", "failed", "cancelled"}:
            return task
        await asyncio.sleep(0.02)
    raise AssertionError("Task did not terminate")


@pytest.mark.parametrize("host", ["codex", "claude"])
def test_native_protocol_result_and_content_free_trace(
    tmp_path: Path, fixture_host: Path, host: str
) -> None:
    async def exercise():
        owner = runtime(tmp_path)
        try:
            request = DelegationRequest(host=host, instruction="PRIVATE_PROMPT", workdir=tmp_path)
            task = await owner.submit(request, executable=str(fixture_host))
            finished = await terminal(owner, task["task_id"])
            assert finished["status"] == "completed", finished
            assert owner.read_result(task["task_id"]) == "native answer"
            assert "PRIVATE_PROMPT" not in json.dumps(owner.session.chain.to_dict())
            assert "PRIVATE_REASONING" not in json.dumps(owner.session.chain.to_dict())
            assert "PRIVATE_PROMPT" not in (tmp_path / "mcp_tasks.db").read_bytes().decode(
                errors="ignore"
            )
            assert owner.session.subprocess_toolkit.list_processes() == []
            assert task["task_id"] not in owner.live
        finally:
            await owner.close()
            owner.store.close()

    asyncio.run(exercise())


def test_scope_and_depth_refuse_before_spawn(tmp_path: Path, fixture_host: Path) -> None:
    async def exercise():
        owner = runtime(tmp_path)
        try:
            with pytest.raises(CARLError):
                await owner.submit(
                    DelegationRequest(host="codex", instruction="x", workdir=tmp_path.parent),
                    executable=str(fixture_host),
                )
            with pytest.raises(CARLError, match="Write scope"):
                await owner.submit(
                    DelegationRequest(host="codex", instruction="x", workdir=tmp_path, write=True),
                    executable=str(fixture_host),
                )
            assert owner.session.subprocess_toolkit.list_processes() == []
            assert owner.store.list() == []
        finally:
            await owner.close()
            owner.store.close()

    asyncio.run(exercise())


def test_capacity_dedup_owner_and_cancel(tmp_path: Path, fixture_host: Path) -> None:
    async def exercise():
        owner, foreign = runtime(tmp_path), runtime(tmp_path, "foreign")
        try:
            request = DelegationRequest(
                host="codex", instruction="wait", workdir=tmp_path, request_id="one"
            )
            first = await owner.submit(request, executable=str(fixture_host))
            duplicate = await owner.submit(request, executable=str(fixture_host))
            assert duplicate["task_id"] == first["task_id"]
            second = await foreign.submit(
                request.model_copy(update={"request_id": "two"}), executable=str(fixture_host)
            )
            with pytest.raises(CARLError, match="Two delegates"):
                await owner.submit(
                    request.model_copy(update={"request_id": "three"}), executable=str(fixture_host)
                )
            with pytest.raises(CARLError, match="not owned"):
                foreign.get(first["task_id"])
            await asyncio.sleep(0.2)
            assert (await owner.cancel(first["task_id"]))["cancelled"]
            assert (await foreign.cancel(second["task_id"]))["cancelled"]
            assert owner.session.subprocess_toolkit.list_processes() == []
            assert foreign.session.subprocess_toolkit.list_processes() == []
        finally:
            await owner.close()
            await foreign.close()
            owner.store.close()
            foreign.store.close()

    asyncio.run(exercise())


def test_permission_reply_is_single_use_and_bound(tmp_path: Path, fixture_host: Path) -> None:
    async def exercise():
        owner = runtime(tmp_path, write=True)
        try:
            task = await owner.submit(
                DelegationRequest(
                    host="codex", instruction="approve", workdir=tmp_path, write=True
                ),
                executable=str(fixture_host),
            )
            for _ in range(100):
                state = owner.get(task["task_id"])
                if state.get("pending_input"):
                    break
                await asyncio.sleep(0.02)
            pending = state["pending_input"]
            with pytest.raises(CARLError, match="stale"):
                owner.reply(task["task_id"], "foreign-request", True)
            owner.reply(task["task_id"], pending["request_id"], True)
            with pytest.raises(CARLError):
                owner.reply(task["task_id"], pending["request_id"], True)
            assert (await terminal(owner, task["task_id"]))["status"] == "completed"
            assert owner.read_result(task["task_id"]) == "approved"
        finally:
            await owner.close()
            owner.store.close()

    asyncio.run(exercise())


def test_dead_owner_unstarted_reservations_release_capacity(tmp_path: Path) -> None:
    import time

    owner = runtime(tmp_path)
    try:
        for number in range(2):
            owner.store.create_delegation(
                {},
                {
                    "owner": "dead",
                    "request_id": str(number),
                    "owner_pid": 99999999,
                    "deadline": time.time() + 100,
                },
            )
        with owner.store._connect() as connection:
            connection.execute("UPDATE mcp_tasks SET metadata=json_set(metadata,'$.deadline',0)")
            connection.commit()
        task = owner.store.create_delegation(
            {},
            {"owner": "live", "request_id": "new", "owner_pid": 1, "deadline": time.time() + 100},
        )
        assert task.status == "pending"
        assert len(owner.store.list(status="failed")) == 2
    finally:
        owner.store.close()


def test_native_error_racing_cancel_keeps_cancelled(tmp_path: Path) -> None:
    async def exercise():
        owner = runtime(tmp_path)

        def error(task_id, live, binary):
            live.cancel.set()
            raise CARLError("Native abort error", code="carl.agent.native_failed")

        owner._drive = error
        try:
            task = await owner.submit(
                DelegationRequest(host="codex", instruction="x", workdir=tmp_path),
                executable="synthetic-unused",
            )
            assert (await terminal(owner, task["task_id"]))["status"] == "cancelled"
        finally:
            await owner.close()
            owner.store.close()

    asyncio.run(exercise())


def test_native_continuation_records_only_acknowledged_inputs(
    tmp_path: Path, fixture_host: Path
) -> None:
    async def exercise():
        owner = runtime(tmp_path)
        request = DelegationRequest(host="codex", instruction="wait", workdir=tmp_path)
        try:
            task = await owner.submit(request, executable=str(fixture_host))
            for _ in range(100):
                state = owner.get(task["task_id"])
                if state["metadata"].get("native_session"):
                    break
                await asyncio.sleep(0.02)
            continuation = request.model_copy(
                update={
                    "native_session": state["metadata"]["native_session"],
                    "request_id": "continuation",
                    "instruction": "reject",
                }
            )
            with pytest.raises(CARLError) as error:
                await owner.submit(continuation)
            assert error.value.code == "carl.agent.native_rejected"
            assert "continuations" not in owner.get(task["task_id"])["metadata"]
            accepted = continuation.model_copy(update={"instruction": "accepted"})
            await owner.submit(accepted)
            assert "continuation" in owner.get(task["task_id"])["metadata"]["continuations"]
            await owner.submit(accepted)
            assert (await owner.cancel(task["task_id"]))["cancelled"]
        finally:
            await owner.close()
            owner.store.close()

    asyncio.run(exercise())


def test_expired_command_escalation_never_reaches_caller_approval(tmp_path: Path) -> None:
    from carl_studio.harness.runtime import _LiveTask

    owner = runtime(tmp_path)
    request = DelegationRequest(host="codex", instruction="x", workdir=tmp_path)
    live = _LiveTask(request=request, context=owner.context, generation="test")
    try:
        assert (
            owner._approve(
                "none",
                live,
                "request",
                {
                    "kind": "item/commandExecution/requestApproval",
                    "cwd": str(tmp_path),
                    "command": "touch /outside/target",
                },
            )
            is False
        )
    finally:
        owner.store.close()


def test_metadata_migration_is_idempotent_across_open_stores(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "mcp_tasks.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE mcp_tasks (task_id TEXT PRIMARY KEY,tool_name TEXT,params_hash TEXT,"
            "status TEXT DEFAULT 'pending',submitted_at TEXT,completed_at TEXT,result TEXT,"
            "error TEXT,progress REAL DEFAULT 0)"
        )
    first, second = MCPTaskStore(path), MCPTaskStore(path)
    try:
        assert first.migrate_delegation() is True
        assert second.migrate_delegation() is False
        assert (
            second.create_delegation({}, {"owner": "test", "request_id": "one"}).status == "pending"
        )
    finally:
        first.close()
        second.close()


def test_concurrent_continuations_preserve_acknowledgements_and_dedup(tmp_path: Path) -> None:
    import time

    from carl_studio.harness.runtime import _LiveTask

    async def exercise():
        owner = runtime(tmp_path)
        request = DelegationRequest(host="codex", instruction="initial", workdir=tmp_path)
        sent = []

        class Adapter:
            def send_input(self, instruction):
                sent.append(instruction)
                time.sleep(0.03)

        task = owner.store.create_delegation(
            {},
            {
                "owner": "caller",
                "request_id": "initial",
                "native_session": "owned",
            },
        )
        owner.store.mark_running(task.task_id)
        owner.live[task.task_id] = _LiveTask(
            request=request,
            context=owner.context,
            generation="test",
            adapter=Adapter(),
            deadline=time.monotonic() + 30,
        )
        first = request.model_copy(
            update={"native_session": "owned", "request_id": "A", "instruction": "A"}
        )
        second = first.model_copy(update={"request_id": "B", "instruction": "B"})
        try:
            await asyncio.gather(owner.submit(first), owner.submit(second), owner.submit(first))
            assert sent == ["A", "B"]
            assert set(owner.store.get(task.task_id).metadata["continuations"]) == {"A", "B"}
            with pytest.raises(CARLError, match="reused"):
                await owner.submit(first.model_copy(update={"instruction": "changed"}))
            assert sent == ["A", "B"]
        finally:
            owner.live.clear()
            owner.store.close()

    asyncio.run(exercise())


def test_a2a_projection_exposes_progress_and_refuses_false_cancel(tmp_path: Path) -> None:
    from carl_studio.a2a.bus import LocalBus
    from carl_studio.a2a.spec import task_to_jsonrpc_result

    owner = runtime(tmp_path)
    bus = LocalBus(tmp_path / "a2a.db")
    try:
        task = owner.store.create_delegation({}, {"owner": "test", "request_id": "one"})
        owner.store.mark_running(task.task_id)
        owner.store.mark_progress(task.task_id, 0.75)
        projected = bus.get(task.task_id)
        assert projected.progress == 0.75
        assert task_to_jsonrpc_result(projected)["metadata"]["progress"] == 0.75
        with pytest.raises(CARLError, match="submitting MCP"):
            bus.cancel(task.task_id)
        assert owner.store.get(task.task_id).status == "running"
    finally:
        owner.store.close()


def test_chat_bridge_close_releases_its_loop_and_is_repeatable(tmp_path: Path) -> None:
    from carl_studio.harness.bridge import HarnessBridge
    from carl_studio.mcp.tasks import set_default_store

    store = MCPTaskStore(tmp_path / "mcp_tasks.db")
    set_default_store(store)
    bridge = HarnessBridge(Session(), tmp_path)
    try:
        bridge.close()
        bridge.close()
        assert not bridge.thread.is_alive()
        assert bridge.loop.is_closed()
    finally:
        bridge.close()
        set_default_store(None)
        store.close()
