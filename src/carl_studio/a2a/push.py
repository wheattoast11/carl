"""Push notification webhook CRUD + delivery for A2A v1.0.

Implements the ``tasks/pushNotificationConfig/{set,get,list,delete}`` surface
from the Linux Foundation A2A spec, plus the delivery worker that consumes
task updates from a :class:`~carl_studio.a2a.bus.LocalBus` and POSTs them to
registered webhooks.

Design notes
------------
* Storage is additive — we create a new ``a2a_push_subscribers`` SQLite
  table on the existing ``LocalBus`` database on first use, never touching
  the tables Phase 0 uses.
* Delivery uses ``carl_core.retry.async_retry`` + a shared
  ``CircuitBreaker`` so a misbehaving subscriber can't stall the whole bus.
* Tokens are opaque bearer secrets chosen by the caller — we store them
  as-is (no hashing) because the spec treats them as shared secrets between
  the registering client and the webhook endpoint, and rotating them is
  the client's responsibility.
* Basic per-agent push is FREE tier. Multi-tenant push relay (running push
  configs on behalf of other orgs) is PAID (``connection.push_relay``), but
  is explicitly out of scope for this PR.

Dependencies
------------
* ``anyio`` is lazy — the worker imports it inside its run loop.
* ``httpx`` is lazy — only needed at delivery time. When missing, delivery
  raises :class:`ConnectionUnavailableError` with a ``pip install httpx``
  hint so the worker can continue processing other tasks.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from carl_core.connection import ConnectionUnavailableError
from carl_core.retry import CircuitBreaker, RetryPolicy, async_retry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from carl_studio.a2a.bus import LocalBus

# ---------------------------------------------------------------------------
# Schema — additive. CREATE TABLE IF NOT EXISTS keeps this idempotent.
# ---------------------------------------------------------------------------

_PUSH_SCHEMA = """
CREATE TABLE IF NOT EXISTS a2a_push_subscribers (
    config_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    webhook_url TEXT NOT NULL,
    token TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_push_task ON a2a_push_subscribers (task_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Breaker shared by every delivery path unless the caller injects one.
# ---------------------------------------------------------------------------


_PUSH_BREAKER: CircuitBreaker = CircuitBreaker(
    failure_threshold=5,
    reset_s=30.0,
    tracked_exceptions=(
        ConnectionError,
        TimeoutError,
        IOError,
        ConnectionUnavailableError,
    ),
)


def get_default_breaker() -> CircuitBreaker:
    """Return the module-level circuit breaker used by :func:`deliver`."""
    return _PUSH_BREAKER


# ---------------------------------------------------------------------------
# PushConfig — pure data.
# ---------------------------------------------------------------------------


@dataclass
class PushConfig:
    """A registered push notification subscription.

    Attributes
    ----------
    task_id
        The task this config subscribes to. Per the spec one task may
        have multiple configs (e.g. one for the operator dashboard and
        one for a billing hook).
    webhook_url
        HTTPS URL the bus will POST to on each status / artifact update.
    token
        Opaque bearer token. Forwarded on the ``Authorization`` header as
        ``Bearer <token>``. Stored plaintext — callers rotate as needed.
    created_at
        ISO-8601 UTC timestamp set at registration time.
    config_id
        Stable UUID assigned at creation. Clients use this to delete /
        fetch the subscription.
    """

    task_id: str
    webhook_url: str
    token: str = ""
    created_at: str = field(default_factory=_utcnow)
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation for JSON-RPC responses."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Storage.
# ---------------------------------------------------------------------------


class PushSubscriberStore:
    """Persist :class:`PushConfig` records on top of :class:`LocalBus`.

    The store does not own the bus connection lifecycle — we reuse the
    bus's existing sqlite connection so transactions line up with the
    rest of the A2A state. The schema migration runs lazily on first use
    (and is idempotent across rebuilds).
    """

    def __init__(self, bus: LocalBus) -> None:
        self._bus = bus
        self._migrated = False

    def ensure_schema(self) -> None:
        """Create the ``a2a_push_subscribers`` table if needed. Idempotent."""
        if self._migrated:
            return
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            conn.executescript(_PUSH_SCHEMA)
            conn.commit()
        self._migrated = True

    # Backward-compatible alias; keep private name around for callers who
    # accessed the migration hook before it was part of the public API.
    _ensure_schema = ensure_schema

    def set(
        self,
        task_id: str,
        webhook_url: str,
        token: str = "",
        *,
        config_id: str | None = None,
    ) -> PushConfig:
        """Register (or replace) a push config. Returns the stored record."""
        if not task_id:
            raise ValueError("task_id must be a non-empty string")
        if not webhook_url:
            raise ValueError("webhook_url must be a non-empty string")
        if not webhook_url.startswith(("http://", "https://")):
            raise ValueError("webhook_url must be an http(s) URL")
        self._ensure_schema()
        cfg = PushConfig(
            task_id=task_id,
            webhook_url=webhook_url,
            token=token,
            config_id=config_id or str(uuid.uuid4()),
        )
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            # INSERT OR REPLACE so setting the same config_id twice is a
            # legal update rather than a PK violation — matches the spec's
            # "configure" semantics.
            conn.execute(
                """INSERT OR REPLACE INTO a2a_push_subscribers
                   (config_id, task_id, webhook_url, token, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (cfg.config_id, cfg.task_id, cfg.webhook_url, cfg.token, cfg.created_at),
            )
            conn.commit()
        return cfg

    def get(self, config_id: str) -> PushConfig | None:
        """Fetch a push config by its stable id. Returns None if missing."""
        self._ensure_schema()
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            row = conn.execute(
                "SELECT * FROM a2a_push_subscribers WHERE config_id = ?",
                (config_id,),
            ).fetchone()
        if row is None:
            return None
        return _config_from_row(row)

    def list(self, task_id: str) -> list[PushConfig]:
        """Return every push config registered for a given task."""
        self._ensure_schema()
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                """SELECT * FROM a2a_push_subscribers
                   WHERE task_id = ?
                   ORDER BY created_at ASC""",
                (task_id,),
            ).fetchall()
        return [_config_from_row(r) for r in rows]

    def delete(self, config_id: str) -> bool:
        """Remove a push config. Returns True iff a row was deleted."""
        self._ensure_schema()
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            cur = conn.execute(
                "DELETE FROM a2a_push_subscribers WHERE config_id = ?",
                (config_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_for_tasks(self, task_ids: list[str]) -> list[PushConfig]:
        """Bulk lookup for the delivery worker. Empty input -> empty output."""
        if not task_ids:
            return []
        self._ensure_schema()
        placeholders = ",".join("?" for _ in task_ids)
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                f"SELECT * FROM a2a_push_subscribers WHERE task_id IN ({placeholders})",
                tuple(task_ids),
            ).fetchall()
        return [_config_from_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Delivery.
# ---------------------------------------------------------------------------


async def deliver(
    config: PushConfig,
    payload: dict[str, Any],
    *,
    breaker: CircuitBreaker | None = None,
    http_client: Any | None = None,
    timeout: float = 10.0,
    policy: RetryPolicy | None = None,
) -> dict[str, Any]:
    """POST ``payload`` to ``config.webhook_url`` with retry + breaker.

    Parameters
    ----------
    config
        The :class:`PushConfig` identifying the webhook URL + token.
    payload
        JSON-serializable dict containing the event data (task snapshot,
        artifact summary, etc.). Must not contain binary values.
    breaker
        Optional :class:`CircuitBreaker`. When omitted the shared
        module-level breaker is used, so a misbehaving subscriber causes
        back-pressure across every delivery to every subscriber with the
        same failure pattern.
    http_client
        Optional async client that exposes ``post(url, json=..., headers=...,
        timeout=...)``. Used by tests to stub the wire. When ``None``, we
        materialize a fresh ``httpx.AsyncClient`` per call — inefficient at
        scale (the worker would pool one per task) but correct in the
        simple per-delivery API.
    timeout
        Request timeout in seconds. Default 10s.
    policy
        Optional retry policy. Defaults to 5 attempts with exponential
        backoff starting at 2s (per the spec's guidance for webhook
        endpoints).

    Returns
    -------
    dict
        ``{"status": <int>, "body": <decoded-body-or-text>}`` on success.

    Raises
    ------
    ConnectionUnavailableError
        Delivery failed after ``policy.max_attempts`` or the breaker was
        open at dispatch time.
    """
    cb = breaker or _PUSH_BREAKER
    retry_policy = policy or RetryPolicy(
        max_attempts=5,
        backoff_base=2.0,
        retryable=(
            ConnectionError,
            TimeoutError,
            IOError,
            ConnectionUnavailableError,
        ),
    )

    # Breaker gate: refuse to even try if open.
    try:
        cb.__enter__()
    except Exception as exc:
        raise ConnectionUnavailableError(
            "push delivery blocked by open circuit breaker",
            context={
                "config_id": config.config_id,
                "task_id": config.task_id,
                "webhook_url": config.webhook_url,
            },
        ) from exc
    try:
        result = await async_retry(
            lambda: _deliver_once(config, payload, http_client, timeout),
            policy=retry_policy,
        )
        cb.__exit__(None, None, None)
        return result
    except BaseException as exc:
        cb.__exit__(type(exc), exc, exc.__traceback__)
        if isinstance(exc, ConnectionUnavailableError):
            raise
        raise ConnectionUnavailableError(
            f"push delivery failed: {exc}",
            context={
                "config_id": config.config_id,
                "task_id": config.task_id,
                "webhook_url": config.webhook_url,
            },
        ) from exc


async def _deliver_once(
    config: PushConfig,
    payload: dict[str, Any],
    http_client: Any | None,
    timeout: float,
) -> dict[str, Any]:
    """Single attempt of a push delivery. Retry loop lives in :func:`deliver`."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.token:
        headers["Authorization"] = f"Bearer {config.token}"

    if http_client is None:
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - no-extra path
            raise ConnectionUnavailableError(
                "httpx is required for A2A push delivery",
                context={"hint": "pip install httpx", "task_id": config.task_id},
            ) from exc

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(config.webhook_url, json=payload, headers=headers)
            return _wrap_response(response)

    # Caller-supplied client: trust its lifecycle.
    response = await http_client.post(
        config.webhook_url, json=payload, headers=headers, timeout=timeout
    )
    return _wrap_response(response)


def _wrap_response(response: Any) -> dict[str, Any]:
    """Normalize an httpx-like response into our dict return shape."""
    status_any: Any = getattr(response, "status_code", 0)
    try:
        status = int(status_any)
    except (TypeError, ValueError):
        status = 0
    body: Any
    try:
        body = response.json()
    except Exception:
        body_text: Any = getattr(response, "text", "")
        body = str(body_text) if body_text is not None else ""
    if status >= 400:
        raise ConnectionUnavailableError(
            f"push webhook returned HTTP {status}",
            context={"status": status, "body": body},
        )
    return {"status": status, "body": body}


# ---------------------------------------------------------------------------
# Delivery worker.
# ---------------------------------------------------------------------------


class PushDeliveryWorker:
    """Drains bus task updates and fans out to registered webhooks.

    The worker runs as a single async task (``start``) that polls the bus at
    a configurable cadence. For every task that has moved since the last
    sweep, it builds a snapshot payload and calls :func:`deliver` for each
    :class:`PushConfig` attached to that task.

    Backpressure model: when the shared breaker is open, :func:`deliver`
    fails fast — the worker catches the exception, records it, and keeps
    going. A subscriber outage is visible in the worker's ``failures``
    counter rather than stalling the whole bus.

    The worker tracks progress via each task's ``updated_at`` timestamp;
    only tasks whose timestamp advances produce a new delivery.
    """

    def __init__(
        self,
        bus: LocalBus,
        store: PushSubscriberStore | None = None,
        *,
        poll_interval: float = 0.5,
        max_iterations: int | None = None,
        on_error: Callable[[BaseException, PushConfig], None] | None = None,
        http_client: Any | None = None,
    ) -> None:
        self._bus = bus
        self._store = store or PushSubscriberStore(bus)
        self._poll_interval = poll_interval
        self._max_iterations = max_iterations
        self._on_error = on_error
        self._http_client = http_client

        self._last_updates: dict[str, str] = {}
        self.deliveries: int = 0
        self.failures: int = 0

    async def run(self) -> None:
        """Consume bus updates until ``max_iterations`` or cancellation."""
        try:
            import anyio  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - no-extra path
            raise ConnectionUnavailableError(
                "anyio is required for A2A push worker",
                context={"hint": "pip install anyio"},
            ) from exc

        iteration = 0
        while True:
            iteration += 1
            if self._max_iterations is not None and iteration > self._max_iterations:
                return
            try:
                await self.drain()
            except anyio.get_cancelled_exc_class():
                return
            except Exception:
                # Keep the worker alive — one bad sweep shouldn't kill
                # every subscriber.
                self.failures += 1
            try:
                await anyio.sleep(self._poll_interval)
            except anyio.get_cancelled_exc_class():
                return

    async def drain(self) -> None:
        """One poll cycle. Visible for tests — tests call this directly."""
        configs = self._snapshot_configs()
        if not configs:
            return
        tasks_by_id = self._snapshot_tasks([c.task_id for c in configs])
        for cfg in configs:
            task = tasks_by_id.get(cfg.task_id)
            if task is None:
                continue
            prev = self._last_updates.get(cfg.task_id)
            if prev == task.updated_at:
                continue
            self._last_updates[cfg.task_id] = task.updated_at
            payload = {
                "taskId": task.id,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
                "updatedAt": task.updated_at,
            }
            try:
                await deliver(cfg, payload, http_client=self._http_client)
                self.deliveries += 1
            except BaseException as exc:
                self.failures += 1
                if self._on_error is not None:
                    try:
                        self._on_error(exc, cfg)
                    except Exception:
                        # Never let a logging callback crash the worker.
                        pass

    def _snapshot_configs(self) -> list[PushConfig]:
        self._store.ensure_schema()
        with self._bus._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                "SELECT * FROM a2a_push_subscribers ORDER BY created_at ASC"
            ).fetchall()
        return [_config_from_row(r) for r in rows]

    def _snapshot_tasks(self, task_ids: list[str]) -> dict[str, Any]:
        if not task_ids:
            return {}
        out: dict[str, Any] = {}
        for tid in task_ids:
            task = self._bus.get(tid)
            if task is not None:
                out[tid] = task
        return out


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _config_from_row(row: sqlite3.Row) -> PushConfig:
    """Deserialize a :class:`PushConfig` from a sqlite3.Row."""
    return PushConfig(
        config_id=row["config_id"],
        task_id=row["task_id"],
        webhook_url=row["webhook_url"],
        token=row["token"] or "",
        created_at=row["created_at"],
    )


__all__ = [
    "PushConfig",
    "PushSubscriberStore",
    "PushDeliveryWorker",
    "deliver",
    "get_default_breaker",
]
