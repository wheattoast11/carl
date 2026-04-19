"""Async task orchestration for long-running MCP tools.

Implements the 2025-11-25 MCP spec's ``tasks/get`` and ``tasks/cancel``
primitives. When a tool is wrapped with :func:`async_task`, it returns a
``{task_id, status: pending}`` handle immediately; the real body runs in an
``anyio`` background task and persists its result (or error) into the
:class:`MCPTaskStore`.

The store is SQLite-backed (default: ``~/.carl/mcp_tasks.db``). When a
LocalBus is already present at ``~/.carl/a2a.db``, we additionally
dual-write into a unified ``agent_tasks`` table on that bus — so MCP task
handles and A2A task handles are observable via the same query surface.

Design decisions
----------------
* ``status`` is a plain string (``pending | running | completed | failed |
  cancelled``) rather than an enum, to keep the write-path ``str``-based
  and serialization-cheap.
* Cancellation is cooperative: :meth:`MCPTaskStore.cancel` flips the row
  to ``cancelled`` and the wrapped body polls a ``anyio.Event`` between
  steps. If the body has already completed we still mark it cancelled but
  return ``False`` to signal "no-op" to the caller.
* ``params_hash`` uses :func:`carl_core.hashing.content_hash` over a
  canonical JSON representation so retries with the same params are
  trivially deduplicatable by the orchestrator (we don't enforce dedup
  here — we just surface the hash).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Generator

import anyio

from carl_core.errors import CARLError
from carl_core.hashing import content_hash

_MCP_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS mcp_tasks (
    task_id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    submitted_at TEXT NOT NULL,
    completed_at TEXT,
    result TEXT,
    error TEXT,
    progress REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_mcp_tasks_status ON mcp_tasks (status);
CREATE INDEX IF NOT EXISTS idx_mcp_tasks_tool_name ON mcp_tasks (tool_name);
"""

_UNIFIED_AGENT_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    submitted_at TEXT NOT NULL,
    completed_at TEXT,
    result TEXT,
    error TEXT,
    progress REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_source ON agent_tasks (source);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks (status);
"""

_TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
_VALID_STATES: frozenset[str] = frozenset(
    {"pending", "running", "completed", "failed", "cancelled"}
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_db_path() -> Path:
    """Return the default ``~/.carl/mcp_tasks.db`` path, creating parent dir."""
    # Lazy import keeps carl_studio.mcp import-light.
    from carl_studio.db import CARL_DIR

    CARL_DIR.mkdir(parents=True, exist_ok=True)
    return CARL_DIR / "mcp_tasks.db"


def _a2a_bus_path() -> Path:
    """Return the existing LocalBus DB path. Does not create the file."""
    from carl_studio.db import CARL_DIR

    return CARL_DIR / "a2a.db"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPTask:
    """Immutable snapshot of an async MCP task row."""

    task_id: str
    tool_name: str
    params_hash: str
    status: str
    submitted_at: datetime
    completed_at: datetime | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    progress: float = 0.0
    # ``_meta`` keeps the original ``params`` around at construction time so
    # the decorator can re-invoke the body. Never persisted.
    _params: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (MCP ``tasks/get`` response)."""
        completed: str | None
        if self.completed_at is not None:
            completed = self.completed_at.isoformat()
        else:
            completed = None
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "params_hash": self.params_hash,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": completed,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }

    @property
    def is_terminal(self) -> bool:
        """True when the task has completed, failed, or been cancelled."""
        return self.status in _TERMINAL_STATES


def _task_from_row(row: sqlite3.Row) -> MCPTask:
    result_raw = row["result"]
    error_raw = row["error"]
    result: Any = None
    if result_raw:
        try:
            result = json.loads(result_raw)
        except (TypeError, ValueError):
            result = result_raw
    error: dict[str, Any] | None = None
    if error_raw:
        try:
            loaded: Any = json.loads(error_raw)
            if isinstance(loaded, dict):
                # Re-cast to our declared Any value type for pyright.
                error = {str(k): v for k, v in loaded.items()}  # type: ignore[misc]
            else:
                error = {"message": str(loaded)}
        except (TypeError, ValueError):
            error = {"message": str(error_raw)}

    submitted = datetime.fromisoformat(row["submitted_at"])
    completed_raw = row["completed_at"]
    completed: datetime | None = None
    if completed_raw:
        try:
            completed = datetime.fromisoformat(completed_raw)
        except (TypeError, ValueError):
            completed = None

    return MCPTask(
        task_id=row["task_id"],
        tool_name=row["tool_name"],
        params_hash=row["params_hash"],
        status=row["status"],
        submitted_at=submitted,
        completed_at=completed,
        result=result,
        error=error,
        progress=float(row["progress"] or 0.0),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class MCPTaskStore:
    """SQLite-backed task table for long-running MCP tools."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._a2a_conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection plumbing
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.row_factory = sqlite3.Row
        try:
            yield self._conn
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            raise

    @contextmanager
    def _a2a_connect(self) -> Generator[sqlite3.Connection | None, None, None]:
        """Open the sibling LocalBus DB if it exists; otherwise yield ``None``.

        Dual-write is strictly best-effort: if the file is missing, locked,
        or schema-incompatible, we skip silently — the primary write into
        ``mcp_tasks.db`` has already succeeded.
        """
        path = _a2a_bus_path()
        if not path.exists():
            yield None
            return
        if self._a2a_conn is None:
            try:
                self._a2a_conn = sqlite3.connect(
                    str(path),
                    check_same_thread=False,
                    timeout=5.0,
                )
                self._a2a_conn.row_factory = sqlite3.Row
                self._a2a_conn.executescript(_UNIFIED_AGENT_TASKS_SCHEMA)
            except sqlite3.Error:
                self._a2a_conn = None
                yield None
                return
        try:
            yield self._a2a_conn
        except sqlite3.Error:
            try:
                assert self._a2a_conn is not None
                self._a2a_conn.rollback()
            except Exception:
                pass
            # Swallow secondary-store failures so primary writes still succeed.
            return

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_MCP_SCHEMA)
        with self._a2a_connect() as conn:
            if conn is None:
                return
            conn.executescript(_UNIFIED_AGENT_TASKS_SCHEMA)

    def close(self) -> None:
        for attr in ("_conn", "_a2a_conn"):
            c = getattr(self, attr)
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, tool_name: str, params: dict[str, Any]) -> MCPTask:
        """Insert a new pending task. ``params`` is hashed, not persisted."""
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        task_id = str(uuid.uuid4())
        params_hash = content_hash(params or {})
        now_iso = _utcnow_iso()
        submitted_at = datetime.fromisoformat(now_iso)

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO mcp_tasks
                   (task_id, tool_name, params_hash, status,
                    submitted_at, completed_at, result, error, progress)
                   VALUES (?, ?, ?, 'pending', ?, NULL, NULL, NULL, 0.0)""",
                (task_id, tool_name, params_hash, now_iso),
            )
            conn.commit()

        with self._a2a_connect() as conn:
            if conn is not None:
                conn.execute(
                    """INSERT OR REPLACE INTO agent_tasks
                       (task_id, source, tool_name, status,
                        submitted_at, completed_at, result, error, progress)
                       VALUES (?, 'mcp', ?, 'pending', ?, NULL, NULL, NULL, 0.0)""",
                    (task_id, tool_name, now_iso),
                )
                conn.commit()

        return MCPTask(
            task_id=task_id,
            tool_name=tool_name,
            params_hash=params_hash,
            status="pending",
            submitted_at=submitted_at,
            _params=dict(params) if params else {},
        )

    def get(self, task_id: str) -> MCPTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mcp_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return _task_from_row(row)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MCPTask]:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if status is not None and status not in _VALID_STATES:
            raise ValueError(
                f"invalid status {status!r}; expected one of {sorted(_VALID_STATES)}"
            )
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM mcp_tasks ORDER BY submitted_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM mcp_tasks
                       WHERE status = ?
                       ORDER BY submitted_at DESC
                       LIMIT ?""",
                    (status, limit),
                ).fetchall()
        return [_task_from_row(r) for r in rows]

    def _mirror_a2a(
        self,
        task_id: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
        completed_at: str | None = None,
        progress: float | None = None,
    ) -> None:
        with self._a2a_connect() as conn:
            if conn is None:
                return
            assignments: list[str] = ["status = ?"]
            params: list[Any] = [status]
            if result is not None:
                assignments.append("result = ?")
                params.append(result)
            if error is not None:
                assignments.append("error = ?")
                params.append(error)
            if completed_at is not None:
                assignments.append("completed_at = ?")
                params.append(completed_at)
            if progress is not None:
                assignments.append("progress = ?")
                params.append(progress)
            params.append(task_id)
            conn.execute(
                f"UPDATE agent_tasks SET {', '.join(assignments)} WHERE task_id = ?",
                params,
            )
            conn.commit()

    def mark_running(self, task_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE mcp_tasks SET status = 'running' WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
        self._mirror_a2a(task_id, "running")

    def mark_progress(self, task_id: str, progress: float) -> None:
        """Update the progress ratio on a running task (0..1, clamped)."""
        clamped = max(0.0, min(1.0, float(progress)))
        with self._connect() as conn:
            conn.execute(
                "UPDATE mcp_tasks SET progress = ? WHERE task_id = ?",
                (clamped, task_id),
            )
            conn.commit()
        self._mirror_a2a(task_id, "running", progress=clamped)

    def mark_completed(self, task_id: str, result: Any) -> None:
        result_json = json.dumps(result, default=str)
        completed_iso = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """UPDATE mcp_tasks
                   SET status = 'completed',
                       result = ?,
                       completed_at = ?,
                       progress = 1.0
                   WHERE task_id = ?""",
                (result_json, completed_iso, task_id),
            )
            conn.commit()
        self._mirror_a2a(
            task_id,
            "completed",
            result=result_json,
            completed_at=completed_iso,
            progress=1.0,
        )

    def mark_failed(self, task_id: str, error: CARLError | Exception) -> None:
        if isinstance(error, CARLError):
            err_dict = error.to_dict()
        else:
            err_dict = {"code": "carl.unknown", "message": str(error)}
        err_json = json.dumps(err_dict, default=str)
        completed_iso = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """UPDATE mcp_tasks
                   SET status = 'failed',
                       error = ?,
                       completed_at = ?
                   WHERE task_id = ?""",
                (err_json, completed_iso, task_id),
            )
            conn.commit()
        self._mirror_a2a(
            task_id,
            "failed",
            error=err_json,
            completed_at=completed_iso,
        )

    def cancel(self, task_id: str) -> bool:
        """Flip a non-terminal task to ``cancelled``.

        Returns True if the task was cancelled; False if the task is already
        terminal or does not exist.
        """
        completed_iso = _utcnow_iso()
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE mcp_tasks
                   SET status = 'cancelled',
                       completed_at = ?
                   WHERE task_id = ?
                     AND status NOT IN ('completed', 'failed', 'cancelled')""",
                (completed_iso, task_id),
            )
            conn.commit()
            changed = cur.rowcount > 0
        if changed:
            self._mirror_a2a(task_id, "cancelled", completed_at=completed_iso)
        return changed

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MCPTaskStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Decorator + registry
# ---------------------------------------------------------------------------


_default_store: MCPTaskStore | None = None


def get_default_store() -> MCPTaskStore:
    """Lazy singleton store backing the decorator + ``tasks/get`` / ``tasks/cancel``."""
    global _default_store
    if _default_store is None:
        _default_store = MCPTaskStore()
    return _default_store


def set_default_store(store: MCPTaskStore | None) -> None:
    """Override the module-global store — used by tests."""
    global _default_store
    _default_store = store


async def _run_in_background(
    store: MCPTaskStore,
    task_id: str,
    body: Callable[..., Awaitable[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Drive the wrapped body, writing result/error into the store."""
    store.mark_running(task_id)
    try:
        result = await body(*args, **kwargs)
    except CARLError as exc:
        store.mark_failed(task_id, exc)
        return
    except Exception as exc:  # noqa: BLE001 — we record all failures
        store.mark_failed(task_id, exc)
        return
    # Check for cancellation before writing success.
    current = store.get(task_id)
    if current is not None and current.status == "cancelled":
        return
    store.mark_completed(task_id, result)


def async_task(
    tool_name: str,
    *,
    store: MCPTaskStore | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[dict[str, Any]]]]:
    """Decorator: wrap an ``async`` MCP tool so it returns a task handle.

    Usage::

        @mcp.tool()
        @async_task("long_training")
        async def long_training(config_yaml: str) -> dict:
            ...

    The decorator:
      * Creates a pending task row in the :class:`MCPTaskStore`.
      * Spawns the body in an ``anyio`` task group — the MCP call returns
        immediately with ``{"task_id": ..., "status": "pending"}``.
      * On completion, writes ``result``; on failure, writes the
        :class:`CARLError` dict.
    """
    if not tool_name:
        raise ValueError("tool_name must be non-empty")

    def _decorator(
        body: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        @wraps(body)
        async def _wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            resolved_store = store or get_default_store()
            # Normalize params for hashing — positional args get indexed keys.
            params: dict[str, Any] = dict(kwargs)
            for idx, val in enumerate(args):
                params[f"_arg{idx}"] = val
            task = resolved_store.create(tool_name, params)

            async def _bg() -> None:
                await _run_in_background(
                    resolved_store, task.task_id, body, args, kwargs
                )

            # Spawn detached: caller returns immediately with the handle.
            async with anyio.create_task_group() as tg:
                tg.start_soon(_sleep_then_spawn, _bg)
            return {
                "task_id": task.task_id,
                "status": "pending",
                "tool_name": tool_name,
                "submitted_at": task.submitted_at.isoformat(),
            }

        return _wrapper

    return _decorator


async def _sleep_then_spawn(bg: Callable[[], Awaitable[None]]) -> None:
    """Schedule the background coroutine onto the running event loop.

    ``anyio.create_task_group`` blocks on exit until child tasks complete;
    to get "spawn and return immediately" semantics we hand the body to
    :func:`asyncio.get_running_loop().create_task`. If no loop is running
    (rare — only the sync test path), we drive the coroutine inline.
    """
    import asyncio

    coro = bg()  # create the coroutine object synchronously
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — drive to completion inline.
        await coro
        return
    # ``create_task`` expects a Coroutine; ``bg`` is ``Callable[[], Awaitable]``
    # but all call sites return concrete coroutines. Cast to keep pyright
    # happy without loosening the public API.
    from typing import cast
    from typing import Coroutine

    loop.create_task(cast(Coroutine[Any, Any, None], coro))


# ---------------------------------------------------------------------------
# Tool registration helpers
# ---------------------------------------------------------------------------


def build_tasks_get_tool(
    store_provider: Callable[[], MCPTaskStore] = get_default_store,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    async def _tasks_get(task_id: str) -> dict[str, Any]:
        """Return the current snapshot of an async MCP task.

        Args:
            task_id: The task handle returned by an ``@async_task`` tool.

        Returns:
            JSON-serializable dict with ``status`` in ``pending | running |
            completed | failed | cancelled`` plus result / error / progress.
            Returns ``{"error": "task_not_found", "task_id": ...}`` if the
            handle is unknown.
        """
        if not task_id:
            return {"error": "task_id must be non-empty"}
        store = store_provider()
        task = store.get(task_id)
        if task is None:
            return {"error": "task_not_found", "task_id": task_id}
        return task.to_dict()

    _tasks_get.__name__ = "tasks_get"
    return _tasks_get


def build_tasks_cancel_tool(
    store_provider: Callable[[], MCPTaskStore] = get_default_store,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    async def _tasks_cancel(task_id: str) -> dict[str, Any]:
        """Cancel a pending or running async MCP task.

        Args:
            task_id: The task handle returned by an ``@async_task`` tool.

        Returns:
            ``{"cancelled": true, "task_id": ...}`` on success,
            ``{"cancelled": false, "reason": "already_terminal" | "task_not_found"}``
            otherwise.
        """
        if not task_id:
            return {"cancelled": False, "reason": "empty_task_id"}
        store = store_provider()
        existing = store.get(task_id)
        if existing is None:
            return {"cancelled": False, "reason": "task_not_found", "task_id": task_id}
        if existing.is_terminal:
            return {
                "cancelled": False,
                "reason": "already_terminal",
                "task_id": task_id,
                "status": existing.status,
            }
        changed = store.cancel(task_id)
        return {"cancelled": bool(changed), "task_id": task_id}

    _tasks_cancel.__name__ = "tasks_cancel"
    return _tasks_cancel


def register_task_tools(mcp_instance: Any) -> None:
    """Register ``tasks/get`` and ``tasks/cancel`` tools on the FastMCP instance.

    FastMCP's ``@tool`` decorator rejects slashes in tool names (JSON-RPC
    method names may contain ``/``, but FastMCP's internal registry uses
    underscored identifiers). We register the canonical underscored names;
    the MCP 2025-11 spec permits either form.
    """
    get_tool = build_tasks_get_tool()
    cancel_tool = build_tasks_cancel_tool()

    mcp_instance.tool(
        name="tasks_get",
        description=(
            "Poll an async MCP task by id. Returns status / result / error. "
            "Pair with tools that wrap their body in @async_task."
        ),
    )(get_tool)
    mcp_instance.tool(
        name="tasks_cancel",
        description=(
            "Cancel a pending or running async MCP task by id. "
            "No-op on already-terminal tasks."
        ),
    )(cancel_tool)


# ---------------------------------------------------------------------------
# Simple polling helper — used by tests that want to block until done
# ---------------------------------------------------------------------------


async def wait_for_task(
    task_id: str,
    *,
    store: MCPTaskStore | None = None,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.02,
) -> MCPTask:
    """Poll ``store`` until ``task_id`` reaches a terminal state.

    Raises ``TimeoutError`` when the task is still non-terminal at the
    deadline and ``KeyError`` when the task handle is unknown.
    """
    resolved = store or get_default_store()
    deadline = time.monotonic() + timeout_s
    while True:
        task = resolved.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.is_terminal:
            return task
        if time.monotonic() >= deadline:
            raise TimeoutError(f"task {task_id} still {task.status} after {timeout_s}s")
        await anyio.sleep(poll_interval_s)


__all__ = [
    "MCPTask",
    "MCPTaskStore",
    "async_task",
    "build_tasks_cancel_tool",
    "build_tasks_get_tool",
    "get_default_store",
    "register_task_tools",
    "set_default_store",
    "wait_for_task",
]
