"""Tests for the MCP 2025-11-25 async task surface.

Coverage:
    * :class:`MCPTaskStore` CRUD round-trips under an isolated temp DB.
    * ``@async_task`` decorator: background execution, cancellation,
      error capture, and params-hash determinism.
    * ``tasks_get`` / ``tasks_cancel`` tool factories return the shapes
      declared in :data:`OUTPUT_SCHEMAS`.
"""

from __future__ import annotations

import asyncio
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from carl_core.errors import CARLError
from carl_studio.mcp.tasks import (
    MCPTask,
    MCPTaskStore,
    async_task,
    build_tasks_cancel_tool,
    build_tasks_get_tool,
    set_default_store,
    wait_for_task,
)


# ---------------------------------------------------------------------------
# Store — CRUD round-trips (no FastMCP dep needed)
# ---------------------------------------------------------------------------


class TestMCPTaskStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = MCPTaskStore(db_path=Path(self._tmp.name) / "tasks.db")
        self.addCleanup(self.store.close)

    def test_create_returns_pending_task_with_params_hash(self) -> None:
        task = self.store.create("my_tool", {"config": "foo"})
        assert task.status == "pending"
        assert task.tool_name == "my_tool"
        assert task.params_hash  # non-empty sha256 hex
        assert len(task.params_hash) == 64
        assert task.task_id
        assert task.completed_at is None
        assert task.result is None
        assert task.error is None

    def test_create_rejects_empty_tool_name(self) -> None:
        with pytest.raises(ValueError):
            self.store.create("", {})

    def test_params_hash_is_deterministic(self) -> None:
        a = self.store.create("tool", {"x": 1, "y": 2})
        b = self.store.create("tool", {"y": 2, "x": 1})
        assert a.params_hash == b.params_hash

    def test_get_returns_task_after_create(self) -> None:
        original = self.store.create("tool", {"k": "v"})
        fetched = self.store.get(original.task_id)
        assert fetched is not None
        assert fetched.task_id == original.task_id
        assert fetched.tool_name == "tool"

    def test_get_unknown_returns_none(self) -> None:
        assert self.store.get("does-not-exist") is None

    def test_mark_running_transitions_status(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_running(task.task_id)
        reloaded = self.store.get(task.task_id)
        assert reloaded is not None
        assert reloaded.status == "running"

    def test_mark_completed_stores_result(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_completed(task.task_id, {"ok": True, "value": 42})
        done = self.store.get(task.task_id)
        assert done is not None
        assert done.status == "completed"
        assert done.result == {"ok": True, "value": 42}
        assert done.completed_at is not None
        assert done.progress == 1.0
        assert done.is_terminal

    def test_mark_failed_stores_carl_error_dict(self) -> None:
        task = self.store.create("tool", {})
        err = CARLError(
            "boom",
            code="carl.test.boom",
            context={"hint": "example"},
        )
        self.store.mark_failed(task.task_id, err)
        failed = self.store.get(task.task_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.error is not None
        assert failed.error.get("code") == "carl.test.boom"

    def test_mark_failed_with_plain_exception(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_failed(task.task_id, RuntimeError("kapow"))
        failed = self.store.get(task.task_id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.error is not None
        assert "kapow" in failed.error.get("message", "")

    def test_cancel_pending_task_returns_true(self) -> None:
        task = self.store.create("tool", {})
        assert self.store.cancel(task.task_id) is True
        cancelled = self.store.get(task.task_id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"
        assert cancelled.is_terminal

    def test_cancel_terminal_task_returns_false(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_completed(task.task_id, {"ok": True})
        assert self.store.cancel(task.task_id) is False

    def test_cancel_unknown_task_returns_false(self) -> None:
        assert self.store.cancel("missing") is False

    def test_list_filter_by_status(self) -> None:
        a = self.store.create("t1", {})
        b = self.store.create("t2", {})
        self.store.mark_completed(b.task_id, {"v": 1})
        pending = self.store.list(status="pending")
        done = self.store.list(status="completed")
        assert [t.task_id for t in pending] == [a.task_id]
        assert [t.task_id for t in done] == [b.task_id]

    def test_list_rejects_bad_status(self) -> None:
        with pytest.raises(ValueError):
            self.store.list(status="not-a-real-state")

    def test_list_rejects_zero_limit(self) -> None:
        with pytest.raises(ValueError):
            self.store.list(limit=0)

    def test_task_to_dict_matches_output_schema_shape(self) -> None:
        task = self.store.create("tool", {"k": "v"})
        payload = task.to_dict()
        for key in (
            "task_id",
            "tool_name",
            "params_hash",
            "status",
            "submitted_at",
            "completed_at",
            "result",
            "error",
            "progress",
        ):
            assert key in payload

    def test_progress_clamped(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_progress(task.task_id, 2.0)
        fetched = self.store.get(task.task_id)
        assert fetched is not None
        assert fetched.progress == 1.0
        self.store.mark_progress(task.task_id, -3.0)
        fetched2 = self.store.get(task.task_id)
        assert fetched2 is not None
        assert fetched2.progress == 0.0

    def test_dual_write_to_a2a_bus_when_present(self) -> None:
        # Create an a2a.db sibling so dual-write kicks in.
        a2a_path = Path(self._tmp.name) / "a2a.db"
        with sqlite3.connect(str(a2a_path)) as con:
            con.execute(
                "CREATE TABLE a2a_tasks (id TEXT PRIMARY KEY, skill TEXT, "
                "inputs TEXT, status TEXT, result TEXT, error TEXT, "
                "sender TEXT, receiver TEXT, priority INTEGER, "
                "created_at TEXT, updated_at TEXT, completed_at TEXT)"
            )
            con.commit()

        # Point the store at the same temp dir by shimming CARL_DIR.
        import carl_studio.mcp.tasks as _tasks

        original_default = _tasks._default_db_path  # pyright: ignore[reportPrivateUsage]
        original_a2a = _tasks._a2a_bus_path  # pyright: ignore[reportPrivateUsage]
        _tasks._default_db_path = lambda: Path(self._tmp.name) / "mcp.db"  # type: ignore[assignment] # pyright: ignore[reportPrivateUsage]
        _tasks._a2a_bus_path = lambda: a2a_path  # type: ignore[assignment] # pyright: ignore[reportPrivateUsage]
        try:
            store2 = MCPTaskStore()
            task = store2.create("dual", {"x": 1})
            store2.mark_completed(task.task_id, {"ok": True})
            # Verify the row landed in agent_tasks.
            with sqlite3.connect(str(a2a_path)) as con:
                row = con.execute(
                    "SELECT status, source FROM agent_tasks WHERE task_id = ?",
                    (task.task_id,),
                ).fetchone()
            assert row is not None
            assert row[0] == "completed"
            assert row[1] == "mcp"
            store2.close()
        finally:
            _tasks._default_db_path = original_default  # type: ignore[assignment] # pyright: ignore[reportPrivateUsage]
            _tasks._a2a_bus_path = original_a2a  # type: ignore[assignment] # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Decorator — @async_task
# ---------------------------------------------------------------------------


class TestAsyncTaskDecorator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.store = MCPTaskStore(db_path=Path(self._tmp.name) / "tasks.db")
        set_default_store(self.store)

    async def asyncTearDown(self) -> None:
        set_default_store(None)
        self.store.close()
        self._tmp.cleanup()

    async def test_decorator_returns_handle_immediately(self) -> None:
        @async_task("quick")
        async def body() -> dict[str, int]:
            return {"result": 1}

        handle = await body()
        assert handle["status"] == "pending"
        assert "task_id" in handle
        assert handle["tool_name"] == "quick"

    async def test_decorator_runs_body_in_background(self) -> None:
        @async_task("slow")
        async def body(x: int) -> dict[str, int]:
            await asyncio.sleep(0.01)
            return {"x": x, "doubled": x * 2}

        handle = await body(21)
        completed = await wait_for_task(handle["task_id"], timeout_s=2.0)
        assert completed.status == "completed"
        assert completed.result == {"x": 21, "doubled": 42}

    async def test_decorator_captures_carl_error(self) -> None:
        @async_task("boom")
        async def body() -> dict[str, int]:
            raise CARLError("nope", code="carl.test.nope")

        handle = await body()
        failed = await wait_for_task(handle["task_id"], timeout_s=2.0)
        assert failed.status == "failed"
        assert failed.error is not None
        assert failed.error.get("code") == "carl.test.nope"

    async def test_decorator_captures_generic_exception(self) -> None:
        @async_task("boom2")
        async def body() -> dict[str, int]:
            raise ValueError("argh")

        handle = await body()
        failed = await wait_for_task(handle["task_id"], timeout_s=2.0)
        assert failed.status == "failed"
        assert failed.error is not None

    async def test_concurrent_runs_isolated(self) -> None:
        @async_task("many")
        async def body(n: int) -> dict[str, int]:
            await asyncio.sleep(0.01)
            return {"n": n}

        handles = await asyncio.gather(*[body(i) for i in range(8)])
        task_ids = [h["task_id"] for h in handles]
        # All unique
        assert len(set(task_ids)) == 8
        # All complete
        results = await asyncio.gather(
            *[wait_for_task(tid, timeout_s=2.0) for tid in task_ids]
        )
        for i, task in enumerate(results):
            assert task.status == "completed"
            assert task.result == {"n": i}

    async def test_cancel_before_completion_writes_cancelled_status(self) -> None:
        @async_task("cancellable")
        async def body() -> dict[str, int]:
            await asyncio.sleep(0.05)
            return {"done": True}

        handle = await body()
        tid = handle["task_id"]
        # Cancel immediately.
        assert self.store.cancel(tid) is True
        # Wait for background to finish; status should stay cancelled.
        await asyncio.sleep(0.1)
        snap = self.store.get(tid)
        assert snap is not None
        assert snap.status == "cancelled"

    async def test_rejects_empty_tool_name(self) -> None:
        with pytest.raises(ValueError):
            async_task("")


# ---------------------------------------------------------------------------
# tasks_get / tasks_cancel tool factories
# ---------------------------------------------------------------------------


class TestTaskToolFactories(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.store = MCPTaskStore(db_path=Path(self._tmp.name) / "tasks.db")

    async def asyncTearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    async def test_tasks_get_returns_serialized_task(self) -> None:
        task = self.store.create("tool", {"x": 1})
        get_tool = build_tasks_get_tool(lambda: self.store)
        payload = await get_tool(task.task_id)
        assert payload["task_id"] == task.task_id
        assert payload["status"] == "pending"

    async def test_tasks_get_unknown_returns_error(self) -> None:
        get_tool = build_tasks_get_tool(lambda: self.store)
        payload = await get_tool("missing")
        assert payload.get("error") == "task_not_found"
        assert payload.get("task_id") == "missing"

    async def test_tasks_get_empty_task_id(self) -> None:
        get_tool = build_tasks_get_tool(lambda: self.store)
        payload = await get_tool("")
        assert "error" in payload

    async def test_tasks_cancel_pending_succeeds(self) -> None:
        task = self.store.create("tool", {})
        cancel_tool = build_tasks_cancel_tool(lambda: self.store)
        payload = await cancel_tool(task.task_id)
        assert payload["cancelled"] is True
        assert payload["task_id"] == task.task_id

    async def test_tasks_cancel_terminal_reports_already_terminal(self) -> None:
        task = self.store.create("tool", {})
        self.store.mark_completed(task.task_id, {"v": 1})
        cancel_tool = build_tasks_cancel_tool(lambda: self.store)
        payload = await cancel_tool(task.task_id)
        assert payload["cancelled"] is False
        assert payload["reason"] == "already_terminal"

    async def test_tasks_cancel_unknown(self) -> None:
        cancel_tool = build_tasks_cancel_tool(lambda: self.store)
        payload = await cancel_tool("nope")
        assert payload["cancelled"] is False
        assert payload["reason"] == "task_not_found"


# ---------------------------------------------------------------------------
# register_task_tools — FastMCP integration
# ---------------------------------------------------------------------------


class TestRegisterTaskTools(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered_on_fastmcp(self) -> None:
        pytest.importorskip("mcp")
        from mcp.server.fastmcp import FastMCP

        from carl_studio.mcp.tasks import register_task_tools

        m = FastMCP("test-register")
        register_task_tools(m)
        names = {t.name for t in await m.list_tools()}
        assert "tasks_get" in names
        assert "tasks_cancel" in names


# ---------------------------------------------------------------------------
# MCPTask dataclass
# ---------------------------------------------------------------------------


class TestMCPTaskDataclass(unittest.TestCase):
    def test_is_terminal(self) -> None:
        from datetime import datetime, timezone

        base = MCPTask(
            task_id="id",
            tool_name="t",
            params_hash="h",
            status="pending",
            submitted_at=datetime.now(timezone.utc),
        )
        assert base.is_terminal is False
        for terminal in ("completed", "failed", "cancelled"):
            t = MCPTask(
                task_id="id",
                tool_name="t",
                params_hash="h",
                status=terminal,
                submitted_at=datetime.now(timezone.utc),
            )
            assert t.is_terminal is True


if __name__ == "__main__":
    unittest.main()
