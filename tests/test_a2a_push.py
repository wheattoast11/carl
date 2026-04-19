"""Tests for :mod:`carl_studio.a2a.push`.

Covers:

* :class:`PushSubscriberStore` CRUD round-trip over an in-memory
  :class:`LocalBus`.
* :func:`deliver` retry + circuit-breaker behaviour with a mocked HTTP
  client.
* :class:`PushDeliveryWorker` drain cycle firing deliveries once per
  ``updated_at`` transition.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from carl_core.connection import ConnectionUnavailableError
from carl_core.retry import CircuitBreaker, RetryPolicy

from carl_studio.a2a import (
    A2ATask,
    LocalBus,
    PushConfig,
    PushDeliveryWorker,
    PushSubscriberStore,
)
from carl_studio.a2a.push import deliver, get_default_breaker


@pytest.fixture
def tmp_bus(tmp_path: Path):
    bus = LocalBus(db_path=tmp_path / "push.db")
    yield bus
    bus.close()


@pytest.fixture
def store(tmp_bus: LocalBus) -> PushSubscriberStore:
    return PushSubscriberStore(tmp_bus)


# ---------------------------------------------------------------------------
# Mock HTTP client.
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    status_code: int = 200
    body: dict[str, Any] | None = None
    text: str = ""

    def json(self) -> Any:
        if self.body is None:
            raise ValueError("no json body")
        return self.body


@dataclass
class _MockHTTPClient:
    """Async mock matching the subset of httpx.AsyncClient we use.

    Stores every ``post`` call so tests can inspect URL, headers, payload.
    ``next_response`` is called on each post to produce a response — lets
    tests script failure / recovery sequences.
    """

    next_response: Any = None
    calls: list[dict[str, Any]] = field(default_factory=lambda: [])
    sequence: list[Any] | None = None

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> _FakeResponse:
        self.calls.append(
            {"url": url, "json": json, "headers": headers or {}, "timeout": timeout}
        )
        if self.sequence is not None:
            if not self.sequence:
                raise AssertionError("MockHTTPClient sequence exhausted")
            item = self.sequence.pop(0)
        else:
            item = self.next_response
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, _FakeResponse):
            return item
        return _FakeResponse(status_code=200, body={"ok": True})


# ---------------------------------------------------------------------------
# CRUD round-trip.
# ---------------------------------------------------------------------------


class TestPushSubscriberStore:
    def test_set_and_get(self, store: PushSubscriberStore) -> None:
        cfg = store.set("task-1", "https://hook.example/a2a", token="t0p-s3cret")
        assert isinstance(cfg, PushConfig)
        assert cfg.task_id == "task-1"
        assert cfg.webhook_url == "https://hook.example/a2a"
        assert cfg.token == "t0p-s3cret"
        assert cfg.config_id

        fetched = store.get(cfg.config_id)
        assert fetched is not None
        assert fetched.webhook_url == "https://hook.example/a2a"
        assert fetched.token == "t0p-s3cret"

    def test_list_returns_all_configs_for_task(self, store: PushSubscriberStore) -> None:
        c1 = store.set("task-L", "https://a.example", token="")
        c2 = store.set("task-L", "https://b.example", token="")
        c3 = store.set("task-other", "https://c.example", token="")

        listed = store.list("task-L")
        ids = {c.config_id for c in listed}
        assert c1.config_id in ids
        assert c2.config_id in ids
        assert c3.config_id not in ids

    def test_delete_returns_true_when_present(self, store: PushSubscriberStore) -> None:
        cfg = store.set("task-D", "https://del.example")
        assert store.delete(cfg.config_id) is True
        assert store.get(cfg.config_id) is None

    def test_delete_returns_false_when_missing(self, store: PushSubscriberStore) -> None:
        assert store.delete("never-existed") is False

    def test_set_rejects_empty_task_id(self, store: PushSubscriberStore) -> None:
        with pytest.raises(ValueError):
            store.set("", "https://h.example")

    def test_set_rejects_non_http_url(self, store: PushSubscriberStore) -> None:
        with pytest.raises(ValueError):
            store.set("task-X", "ftp://nope.example")

    def test_schema_is_idempotent_across_stores(self, tmp_bus: LocalBus) -> None:
        # Creating multiple stores on the same bus must not fail the schema.
        s1 = PushSubscriberStore(tmp_bus)
        s2 = PushSubscriberStore(tmp_bus)
        cfg = s1.set("task-A", "https://a.example")
        assert s2.get(cfg.config_id) is not None


# ---------------------------------------------------------------------------
# Delivery semantics.
# ---------------------------------------------------------------------------


class TestDeliver:
    def test_sends_bearer_token_header(self) -> None:
        client = _MockHTTPClient(next_response=_FakeResponse(body={"ok": True}))
        cfg = PushConfig(
            task_id="t",
            webhook_url="https://hook.example/a",
            token="abc123",
            config_id="cfg-1",
        )

        async def run() -> dict[str, Any]:
            breaker = CircuitBreaker(failure_threshold=3)
            return await deliver(
                cfg,
                {"taskId": "t"},
                breaker=breaker,
                http_client=client,
                policy=RetryPolicy(max_attempts=1, jitter=False),
            )

        result = asyncio.run(run())
        assert result["status"] == 200
        assert len(client.calls) == 1
        headers = client.calls[0]["headers"]
        assert headers.get("Authorization") == "Bearer abc123"
        assert headers.get("Content-Type") == "application/json"

    def test_retries_on_retryable_error(self) -> None:
        client = _MockHTTPClient(
            sequence=[
                ConnectionError("flaky"),
                ConnectionError("still flaky"),
                _FakeResponse(status_code=200, body={"ok": True}),
            ]
        )
        cfg = PushConfig(
            task_id="t",
            webhook_url="https://hook.example/a",
            token="",
            config_id="cfg-retry",
        )

        async def run() -> dict[str, Any]:
            breaker = CircuitBreaker(failure_threshold=10)
            return await deliver(
                cfg,
                {"taskId": "t"},
                breaker=breaker,
                http_client=client,
                policy=RetryPolicy(
                    max_attempts=3,
                    backoff_base=0.001,
                    jitter=False,
                ),
            )

        result = asyncio.run(run())
        assert result["status"] == 200
        assert len(client.calls) == 3

    def test_4xx_raises_connection_unavailable(self) -> None:
        client = _MockHTTPClient(next_response=_FakeResponse(status_code=403))
        cfg = PushConfig(
            task_id="t",
            webhook_url="https://hook.example/a",
            token="",
            config_id="cfg-403",
        )

        async def run() -> None:
            breaker = CircuitBreaker(failure_threshold=5)
            await deliver(
                cfg,
                {"x": 1},
                breaker=breaker,
                http_client=client,
                policy=RetryPolicy(max_attempts=1, jitter=False),
            )

        with pytest.raises(ConnectionUnavailableError):
            asyncio.run(run())

    def test_open_breaker_blocks_delivery(self) -> None:
        # Prime a breaker until it's open.
        breaker = CircuitBreaker(
            failure_threshold=1,
            reset_s=30.0,
            tracked_exceptions=(ConnectionError,),
        )
        try:
            with breaker:
                raise ConnectionError("trip")
        except ConnectionError:
            pass
        assert breaker.state == "open"

        cfg = PushConfig(
            task_id="t",
            webhook_url="https://hook.example/a",
            token="",
            config_id="cfg-open",
        )
        client = _MockHTTPClient(next_response=_FakeResponse(body={"ok": True}))

        async def run() -> dict[str, Any]:
            return await deliver(
                cfg,
                {"x": 1},
                breaker=breaker,
                http_client=client,
                policy=RetryPolicy(max_attempts=1, jitter=False),
            )

        with pytest.raises(ConnectionUnavailableError):
            asyncio.run(run())
        assert client.calls == []  # delivery blocked before wire

    def test_default_breaker_is_module_level_singleton(self) -> None:
        a = get_default_breaker()
        b = get_default_breaker()
        assert a is b


# ---------------------------------------------------------------------------
# Delivery worker.
# ---------------------------------------------------------------------------


class TestPushDeliveryWorker:
    def test_drain_delivers_on_new_update(self, tmp_bus: LocalBus) -> None:
        store = PushSubscriberStore(tmp_bus)
        task = A2ATask(id="w-1", skill="train")
        tmp_bus.post(task)
        store.set("w-1", "https://hook.example/a2a", token="tok")

        client = _MockHTTPClient(next_response=_FakeResponse(body={"ok": True}))
        worker = PushDeliveryWorker(
            tmp_bus,
            store=store,
            poll_interval=0.001,
            http_client=client,
        )

        async def run() -> None:
            await worker.drain()

        asyncio.run(run())
        assert worker.deliveries == 1
        assert len(client.calls) == 1
        assert client.calls[0]["headers"]["Authorization"] == "Bearer tok"
        payload = client.calls[0]["json"]
        assert payload["taskId"] == "w-1"

    def test_drain_skips_unchanged_tasks(self, tmp_bus: LocalBus) -> None:
        store = PushSubscriberStore(tmp_bus)
        task = A2ATask(id="w-2", skill="train")
        tmp_bus.post(task)
        store.set("w-2", "https://hook.example/a2a")

        client = _MockHTTPClient(next_response=_FakeResponse(body={"ok": True}))
        worker = PushDeliveryWorker(
            tmp_bus, store=store, poll_interval=0.001, http_client=client
        )

        async def run() -> None:
            await worker.drain()
            await worker.drain()
            await worker.drain()

        asyncio.run(run())
        # Delivered once on first sweep; subsequent sweeps see the same
        # updated_at and skip.
        assert worker.deliveries == 1

    def test_drain_refires_on_updated_at_change(self, tmp_bus: LocalBus) -> None:
        store = PushSubscriberStore(tmp_bus)
        task = A2ATask(id="w-3", skill="train")
        tmp_bus.post(task)
        store.set("w-3", "https://hook.example/a2a")

        client = _MockHTTPClient(next_response=_FakeResponse(body={"ok": True}))
        worker = PushDeliveryWorker(
            tmp_bus, store=store, poll_interval=0.001, http_client=client
        )

        async def run() -> None:
            await worker.drain()
            tmp_bus.update(task.mark_running())
            await worker.drain()
            tmp_bus.update(task.mark_running().mark_done({"score": 1.0}))
            await worker.drain()

        asyncio.run(run())
        assert worker.deliveries == 3

    def test_worker_records_failures_without_crashing(self, tmp_bus: LocalBus) -> None:
        store = PushSubscriberStore(tmp_bus)
        task = A2ATask(id="w-4", skill="train")
        tmp_bus.post(task)
        store.set("w-4", "https://hook.example/a2a")

        # 4xx body on every attempt — delivery raises, worker should
        # capture it and keep going.
        client = _MockHTTPClient(next_response=_FakeResponse(status_code=500))
        worker = PushDeliveryWorker(
            tmp_bus, store=store, poll_interval=0.001, http_client=client
        )

        async def run() -> None:
            await worker.drain()

        # Note: using the default retry policy would hit the breaker + sleep.
        # Worker calls ``deliver`` with default policy (5 attempts, 2s base).
        # We intercept via a short-circuit retry policy? Simpler: use a
        # zero-base policy by wrapping deliver with monkeypatching.
        #
        # Instead we rely on the on_error callback and a policy that fires
        # once. Patch module-level deliver? That's intrusive — just send a
        # failing request with a short retry policy by monkeypatching.
        import carl_studio.a2a.push as push_mod

        orig_deliver = push_mod.deliver

        async def _fast_deliver(
            cfg: PushConfig,
            payload: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            kwargs.setdefault(
                "policy",
                RetryPolicy(max_attempts=1, backoff_base=0.001, jitter=False),
            )
            return await orig_deliver(cfg, payload, **kwargs)

        push_mod.deliver = _fast_deliver  # type: ignore[assignment]
        try:
            asyncio.run(run())
        finally:
            push_mod.deliver = orig_deliver  # type: ignore[assignment]

        assert worker.deliveries == 0
        assert worker.failures >= 1
