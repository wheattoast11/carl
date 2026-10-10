"""Prepared training reuses task custody without repeating its body."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from carl_core.errors import CARLError

from carl_studio.mcp import tasks

PLAN = "Eprep_" + "a" * 24


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tasks, "_a2a_bus_path", lambda: tmp_path / "absent-a2a.db")
    owner = tasks.MCPTaskStore(tmp_path / "tasks.db")
    tasks.set_default_store(owner)
    yield owner
    tasks.set_default_store(None)
    owner.close()


@pytest.mark.asyncio
async def test_live_and_terminal_replay_reuse_handle_and_body(store):
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    @tasks.async_task("submit_async_training", store=store)
    async def body(config_yaml: str, prepared_plan_id: str | None = None):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"phase": "complete", "acceptance": {"status": "rejected"}}

    first = await body("configured", PLAN)
    await asyncio.wait_for(started.wait(), timeout=2)
    second = await body(config_yaml="configured", prepared_plan_id=PLAN)
    assert second["task_id"] == first["task_id"]
    assert second["status"] == "running"
    assert second["submitted_at"] == first["submitted_at"]
    assert calls == 1
    release.set()
    finished = await tasks.wait_for_task(first["task_id"], timeout_s=2)
    third = await body("configured", prepared_plan_id=PLAN)
    assert third["task_id"] == first["task_id"] and third["status"] == "completed"
    assert finished.result["acceptance"]["status"] == "rejected"
    assert calls == 1


@pytest.mark.asyncio
async def test_changed_parameters_under_same_plan_are_refused(store):
    calls = 0

    @tasks.async_task("submit_async_training", store=store)
    async def body(config_yaml: str, prepared_plan_id: str | None = None):
        nonlocal calls
        calls += 1
        return {"phase": "complete"}

    first = await body("original", PLAN)
    await tasks.wait_for_task(first["task_id"], timeout_s=2)
    with pytest.raises(CARLError) as conflict:
        await body("changed", PLAN)
    assert conflict.value.code == "carl.tasks.request_conflict"
    assert len(store.list()) == 1 and calls == 1


@pytest.mark.asyncio
async def test_concurrent_prepared_requests_spawn_one_body(store):
    calls = 0

    @tasks.async_task("submit_async_training", store=store)
    async def body(config_yaml: str, prepared_plan_id: str | None = None):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"phase": "complete"}

    handles = await asyncio.gather(*[body("configured", PLAN) for _ in range(16)])
    assert len({handle["task_id"] for handle in handles}) == 1
    await tasks.wait_for_task(handles[0]["task_id"], timeout_s=2)
    assert calls == 1 and len(store.list()) == 1


def test_independent_sql_connections_have_one_reservation_winner(store):
    barrier = threading.Barrier(8)

    def reserve(_):
        independent = tasks.MCPTaskStore(store.path)
        try:
            barrier.wait(timeout=5)
            return independent.create_once(
                "submit_async_training",
                {"config_yaml": "configured", "prepared_plan_id": PLAN},
                request_id=PLAN,
            )
        finally:
            independent.close()

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(reserve, range(8)))
    assert sum(created for _, created in results) == 1
    assert len({task.task_id for task, _ in results}) == 1


@pytest.mark.asyncio
async def test_persisted_restart_reuses_terminal_without_execution(store):
    calls = 0

    @tasks.async_task("submit_async_training", store=store)
    async def initial(config_yaml: str, prepared_plan_id: str | None = None):
        return {"run_id": "retained"}

    first = await initial("configured", PLAN)
    await tasks.wait_for_task(first["task_id"], timeout_s=2)
    path = store.path
    store.close()
    restarted = tasks.MCPTaskStore(path)
    try:

        @tasks.async_task("submit_async_training", store=restarted)
        async def replay(config_yaml: str, prepared_plan_id: str | None = None):
            nonlocal calls
            calls += 1

        second = await replay("configured", PLAN)
        assert second["task_id"] == first["task_id"] and second["status"] == "completed"
        assert restarted.get(second["task_id"]).result == {"run_id": "retained"}
        assert calls == 0
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_pending_restart_reuses_custody_without_implicit_retry(store):
    task, created = store.create_once(
        "submit_async_training",
        {"config_yaml": "configured", "prepared_plan_id": PLAN},
        request_id=PLAN,
    )
    assert created
    store.close()
    restarted = tasks.MCPTaskStore(store.path)
    calls = 0
    try:

        @tasks.async_task("submit_async_training", store=restarted)
        async def replay(config_yaml: str, prepared_plan_id: str | None = None):
            nonlocal calls
            calls += 1

        handle = await replay("configured", PLAN)
        assert handle["task_id"] == task.task_id and handle["status"] == "pending"
        await asyncio.sleep(0)
        assert calls == 0
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_failed_task_replay_preserves_failure_without_resubmit(store):
    calls = 0

    @tasks.async_task("submit_async_training", store=store)
    async def body(config_yaml: str, prepared_plan_id: str | None = None):
        nonlocal calls
        calls += 1
        raise RuntimeError("fixture failure")

    first = await body("configured", PLAN)
    failed = await tasks.wait_for_task(first["task_id"], timeout_s=2)
    second = await body("configured", PLAN)
    assert failed.status == second["status"] == "failed"
    assert second["task_id"] == first["task_id"] and calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool, plan", [("submit_async_training", None), ("other_tool", PLAN)])
async def test_other_async_calls_keep_independent_handles(store, tool, plan):
    @tasks.async_task(tool, store=store)
    async def body(config_yaml: str, prepared_plan_id: str | None = None):
        return {"result": "independent"}

    first, second = await body("configured", plan), await body("configured", plan)
    assert first["task_id"] != second["task_id"]
    await asyncio.gather(
        *[tasks.wait_for_task(item["task_id"], timeout_s=2) for item in [first, second]]
    )
