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
* The task tool cancels an owned worker and waits for cleanup before recording
  ``cancelled``. Running tasks without live worker custody require reconciliation.
  Store-only cancellation remains available for pending work.
* Prepared training uses its plan ID as a persistent request identity.
  Replays reuse the task handle; changed parameters under that identity fail.
  Other tools retain independent task handles and expose ``params_hash``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sqlite3
import threading
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

logger = logging.getLogger(__name__)

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
    progress REAL NOT NULL DEFAULT 0.0,
    metadata TEXT NOT NULL DEFAULT '{}'
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
    metadata: dict[str, Any] = field(default_factory=dict)
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
        result = {
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
        if self.metadata:
            result["metadata"] = self.metadata
        return result

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
        metadata=json.loads(row["metadata"]) if "metadata" in row.keys() else {},  # noqa: SIM118 (sqlite3.Row checks values)
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
        self._lock = threading.RLock()
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
        with self._lock:
            try:
                yield self._conn
            except Exception:
                self._conn.rollback()
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
            self._has_metadata = any(
                row[1] == "metadata" for row in conn.execute("PRAGMA table_info(mcp_tasks)")
            )
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

    @property
    def path(self) -> Path:
        """The owning task database's location."""
        return self._path

    def migrate_delegation(self) -> bool:
        """Add delegation metadata after explicit operator approval."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._has_metadata = any(
                row[1] == "metadata" for row in conn.execute("PRAGMA table_info(mcp_tasks)")
            )
            if self._has_metadata:
                conn.commit()
                return False
            conn.execute("ALTER TABLE mcp_tasks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            conn.commit()
            self._has_metadata = True
        return True

    def create_delegation(self, params: dict[str, Any], metadata: dict[str, Any]) -> MCPTask:
        """Atomically reserve a shared delegate slot and deduplicate a request."""
        with self._connect() as connection:
            self._has_metadata = any(
                row[1] == "metadata" for row in connection.execute("PRAGMA table_info(mcp_tasks)")
            )
        if not self._has_metadata:
            raise CARLError("Run carl plugin migrate --apply", code="carl.tasks.migration_required")
        digest = content_hash(params)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM mcp_tasks WHERE tool_name = 'delegate_agent' AND status IN ('pending','running')"
            ).fetchall()
            for row in rows:
                if row["status"] in _TERMINAL_STATES:
                    continue
                info = json.loads(row["metadata"])
                pid = info.get("owner_pid")
                alive = True
                if isinstance(pid, int):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        alive = False
                    except PermissionError:
                        alive = True
                if (
                    not alive
                    and not info.get("spawn_started")
                    and not info.get("native_pid")
                    and info.get("deadline", float("inf")) < time.time()
                ):
                    conn.execute(
                        "UPDATE mcp_tasks SET status='failed',completed_at=?,error=? "
                        "WHERE task_id=? AND status IN ('pending','running')",
                        (
                            _utcnow_iso(),
                            json.dumps(
                                {
                                    "code": "carl.agent.interrupted",
                                    "message": "Owner exited before execution started",
                                }
                            ),
                            row["task_id"],
                        ),
                    )
            row = conn.execute(
                "SELECT * FROM mcp_tasks WHERE tool_name='delegate_agent' "
                "AND json_extract(metadata,'$.owner')=? AND json_extract(metadata,'$.request_id')=? LIMIT 1",
                (metadata["owner"], metadata["request_id"]),
            ).fetchone()
            if row is not None:
                previous = _task_from_row(row)
                if previous.params_hash != digest:
                    raise CARLError("Request ID reused", code="carl.agent.request_conflict")
                conn.commit()
                return previous
            active = conn.execute(
                "SELECT COUNT(*) FROM mcp_tasks WHERE tool_name='delegate_agent' "
                "AND status IN ('pending','running')"
            ).fetchone()[0]
            if active >= 2:
                raise CARLError("Two delegates are already active", code="carl.agent.capacity")
            task_id, now = str(uuid.uuid4()), _utcnow_iso()
            conn.execute(
                "INSERT INTO mcp_tasks (task_id,tool_name,params_hash,submitted_at,metadata) "
                "VALUES (?,'delegate_agent',?,?,?)",
                (task_id, digest, now, json.dumps(metadata)),
            )
            conn.commit()
        return self.get(task_id)  # type: ignore[return-value]

    def update_metadata(self, task_id: str, changes: dict[str, Any]) -> None:
        """Merge metadata while preserving terminal disposition."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM mcp_tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is not None and row["status"] not in _TERMINAL_STATES:
                metadata = json.loads(row["metadata"])
                metadata.update(changes)
                conn.execute(
                    "UPDATE mcp_tasks SET metadata=? WHERE task_id=?",
                    (json.dumps(metadata), task_id),
                )
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, tool_name: str, params: dict[str, Any]) -> MCPTask:
        """Insert a new pending task. ``params`` is hashed, not persisted."""
        task, _ = self._create_pending(tool_name, params)
        return task

    def create_once(
        self, tool_name: str, params: dict[str, Any], *, request_id: str
    ) -> tuple[MCPTask, bool]:
        """Reserve a scoped request once; return its task and whether it was created."""
        if type(request_id) is not str or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a nonempty string of at most 128 characters")
        return self._create_pending(tool_name, params, request_id=request_id)

    def _create_pending(
        self, tool_name: str, params: dict[str, Any], *, request_id: str | None = None
    ) -> tuple[MCPTask, bool]:
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        task_id = str(
            uuid.uuid4()
            if request_id is None
            else uuid.uuid5(
                uuid.NAMESPACE_URL, content_hash({"tool_name": tool_name, "request_id": request_id})
            )
        )
        params_hash = content_hash(params or {})
        now_iso = _utcnow_iso()
        submitted_at = datetime.fromisoformat(now_iso)

        with self._connect() as conn:
            if request_id is not None:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT * FROM mcp_tasks WHERE task_id=?", (task_id,)).fetchone()
                if row is not None:
                    previous = _task_from_row(row)
                    if previous.tool_name != tool_name or previous.params_hash != params_hash:
                        raise CARLError(
                            "Request identity reused with different parameters",
                            code="carl.tasks.request_conflict",
                        )
                    conn.commit()
                    return previous, False
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

        task = MCPTask(
            task_id=task_id,
            tool_name=tool_name,
            params_hash=params_hash,
            status="pending",
            submitted_at=submitted_at,
            _params=dict(params) if params else {},
        )
        return task, True

    @staticmethod
    def _operation_task_id(operation_id: str) -> str:
        if type(operation_id) is not str or not operation_id or len(operation_id) > 128:
            raise ValueError("Invalid operation identity")
        return str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash({"operation_id": operation_id})))

    @staticmethod
    def _operation_record(
        record: dict[str, Any], operation_id: str, generation: int
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("Operation record must be a JSON mapping")
        payload = dict(record)
        if payload.get("operation_id", operation_id) != operation_id:
            raise ValueError("Operation record identity changed")
        payload.update(operation_id=operation_id, generation=generation)
        return json.loads(json.dumps(payload, allow_nan=False))

    def load_operation(self, operation_id: str) -> dict[str, Any] | None:
        """Read the operation's retained JSON record without executing its action."""
        task = self.get(self._operation_task_id(operation_id))
        if task is None:
            return None
        if task.tool_name != "operation_state" or not isinstance(task.result, dict):
            raise CARLError("Operation state is invalid", code="carl.tasks.operation_conflict")
        return dict(task.result)

    @staticmethod
    def _operation_process_start(pid: int) -> str | None:
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text()
            return raw[raw.rfind(")") + 2 :].split()[19]
        except FileNotFoundError:
            return None

    @classmethod
    def _operation_owner_alive(cls, claim: dict[str, Any]) -> bool:
        start = claim.get("owner_start_ticks")
        pid = claim.get("owner_pid")
        if type(pid) is not int or type(start) is not str:
            return True
        try:
            return cls._operation_process_start(pid) == start
        except (OSError, IndexError):
            return True

    def reconcile_operation(
        self,
        operation_id: str,
        expected_generation: int,
        claimed_record: dict[str, Any],
        action_ref: str,
        *,
        previous_token: str | None = None,
    ) -> str:
        """Recover observation custody after owner death or an explicit token handoff."""
        return self.claim_operation(
            operation_id,
            expected_generation,
            claimed_record,
            action_ref,
            reconcile_only=True,
            previous_token=previous_token,
        )

    def claim_operation(
        self,
        operation_id: str,
        expected_generation: int,
        claimed_record: dict[str, Any],
        action_ref: str,
        *,
        reconcile_only: bool = False,
        previous_token: str | None = None,
    ) -> str:
        """Fence one action by generation; the caller supplies a content-free JSON record."""
        task_id = self._operation_task_id(operation_id)
        if type(expected_generation) is not int or expected_generation < 0:
            raise ValueError("Invalid operation generation")
        if type(action_ref) is not str or not action_ref or len(action_ref) > 256:
            raise ValueError("Invalid operation action reference")
        generation = expected_generation + 1
        payload = self._operation_record(claimed_record, operation_id, generation)
        params_hash = content_hash({"operation_id": operation_id})
        token = task_id + "." + uuid.uuid4().hex
        with self._connect() as conn:
            if not self._has_metadata:
                raise CARLError(
                    "Run carl plugin migrate --apply", code="carl.tasks.migration_required"
                )
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM mcp_tasks WHERE task_id=?", (task_id,)).fetchone()
            metadata: dict[str, Any] = {"operation_id": operation_id, "generation": 0}
            if row is not None:
                metadata = json.loads(row["metadata"])
                if (
                    row["tool_name"] != "operation_state"
                    or row["params_hash"] != params_hash
                    or metadata.get("operation_id") != operation_id
                    or row["status"] in _TERMINAL_STATES
                ):
                    raise CARLError(
                        "Operation binding changed", code="carl.tasks.operation_conflict"
                    )
            if metadata.get("generation") != expected_generation:
                raise CARLError(
                    "Operation claim requires reconciliation", code="carl.tasks.operation_conflict"
                )
            previous = metadata.get("inflight")
            if previous:
                if not reconcile_only or (
                    previous_token != previous.get("token")
                    if previous_token is not None
                    else self._operation_owner_alive(previous)
                ):
                    raise CARLError(
                        "Operation claim requires reconciliation",
                        code="carl.tasks.operation_conflict",
                    )
            elif previous_token is not None:
                raise CARLError(
                    "Operation handoff token is stale", code="carl.tasks.operation_conflict"
                )
            metadata.update(
                generation=generation,
                inflight={
                    "token": token,
                    "action_ref": action_ref,
                    "generation": generation,
                    "mode": "reconcile-only" if reconcile_only else "execute",
                    "owner_pid": os.getpid(),
                    "owner_start_ticks": self._operation_process_start(os.getpid()),
                },
            )
            if row is None:
                conn.execute(
                    "INSERT INTO mcp_tasks (task_id,tool_name,params_hash,status,submitted_at,result,metadata) "
                    "VALUES (?,'operation_state',?,'running',?,?,?)",
                    (
                        task_id,
                        params_hash,
                        _utcnow_iso(),
                        json.dumps(payload),
                        json.dumps(metadata),
                    ),
                )
            else:
                conn.execute(
                    "UPDATE mcp_tasks SET status='running',result=?,metadata=? WHERE task_id=?",
                    (json.dumps(payload), json.dumps(metadata), task_id),
                )
            conn.commit()
        return token

    def commit_operation(self, token: str, record: dict[str, Any]) -> dict[str, Any]:
        """Commit the claimed action once; identical terminal replay returns its record."""
        task_id, separator, nonce = token.partition(".")
        if not separator or not task_id or not nonce:
            raise ValueError("Invalid operation token")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM mcp_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None or row["tool_name"] != "operation_state":
                raise CARLError("Operation token is unknown", code="carl.tasks.operation_conflict")
            metadata = json.loads(row["metadata"])
            operation_id = metadata["operation_id"]
            if row["params_hash"] != content_hash({"operation_id": operation_id}):
                raise CARLError("Operation binding changed", code="carl.tasks.operation_conflict")
            payload = self._operation_record(record, operation_id, metadata["generation"])
            record_hash = content_hash(payload)
            previous = metadata.get("last_commit", {})
            if previous.get("token") == token:
                if previous.get("record_hash") != record_hash:
                    raise CARLError(
                        "Operation commit replay changed", code="carl.tasks.operation_conflict"
                    )
                conn.commit()
                return json.loads(row["result"])
            inflight = metadata.get("inflight", {})
            if (
                inflight.get("token") != token
                or inflight.get("generation") != metadata["generation"]
                or row["status"] != "running"
            ):
                raise CARLError("Operation claim is stale", code="carl.tasks.operation_conflict")
            metadata.pop("inflight")
            metadata["last_commit"] = {"token": token, "record_hash": record_hash}
            conn.execute(
                "UPDATE mcp_tasks SET status='pending',result=?,metadata=? WHERE task_id=?",
                (json.dumps(payload), json.dumps(metadata), task_id),
            )
            conn.commit()
        return payload

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
            raise ValueError(f"invalid status {status!r}; expected one of {sorted(_VALID_STATES)}")
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
        primary = self.get(task_id)
        if primary is None or primary.status != status:
            return
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
                "UPDATE mcp_tasks SET status = 'running' WHERE task_id = ? AND status='pending'",
                (task_id,),
            )
            conn.commit()
        self._mirror_a2a(task_id, "running")

    def mark_progress(self, task_id: str, progress: float) -> None:
        """Update the progress ratio on a running task (0..1, clamped)."""
        clamped = max(0.0, min(1.0, float(progress)))
        with self._connect() as conn:
            conn.execute(
                "UPDATE mcp_tasks SET progress = ? WHERE task_id = ? AND status IN ('pending','running')",
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
                   WHERE task_id = ? AND status IN ('pending','running')""",
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
                   WHERE task_id = ? AND status IN ('pending','running')""",
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
_live_tasks: dict[str, Any] = {}


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
    current = store.get(task_id)
    if current is None or current.status != "running":
        return
    try:
        result = await body(*args, **kwargs)
    except asyncio.CancelledError:
        store.cancel(task_id)
        raise
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
            created = True
            if tool_name == "submit_async_training":
                bound = inspect.signature(body).bind(*args, **kwargs)
                prepared_plan_id = bound.arguments.get("prepared_plan_id")
                if prepared_plan_id is not None:
                    bound.apply_defaults()
                    task, created = resolved_store.create_once(
                        tool_name, dict(bound.arguments), request_id=prepared_plan_id
                    )
                else:
                    task = resolved_store.create(tool_name, params)
            else:
                task = resolved_store.create(tool_name, params)

            async def _bg() -> None:
                import asyncio

                _live_tasks[task.task_id] = asyncio.current_task()
                try:
                    await _run_in_background(resolved_store, task.task_id, body, args, kwargs)
                finally:
                    _live_tasks.pop(task.task_id, None)

            # Spawn detached: caller returns immediately with the handle.
            if created:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_sleep_then_spawn, _bg)
            return {
                "task_id": task.task_id,
                "status": task.status,
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
        if task.tool_name == "delegate_agent":
            from carl_studio.harness.runtime import get_runtime

            return get_runtime().get(task_id)
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
        if existing.tool_name == "delegate_agent":
            from carl_studio.harness.runtime import get_runtime

            return await get_runtime().cancel(task_id)
        if existing.is_terminal:
            return {
                "cancelled": False,
                "reason": "already_terminal",
                "task_id": task_id,
                "status": existing.status,
            }
        worker = _live_tasks.get(task_id)
        if worker is not None:
            if not worker.cancelling():
                worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            stopped = store.get(task_id)
            if stopped is not None and stopped.status == "cancelled":
                return {"cancelled": True, "task_id": task_id}
            if stopped is not None and stopped.status == "failed":
                return {
                    "cancelled": False,
                    "task_id": task_id,
                    "reason": "execution_not_confirmed",
                    "status": stopped.status,
                }
        elif existing.status == "running":
            return {
                "cancelled": False,
                "task_id": task_id,
                "reason": "execution_not_owned",
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
    from mcp.server.mcpserver import Context

    legacy_get = build_tasks_get_tool()
    legacy_cancel = build_tasks_cancel_tool()

    async def tasks_get(task_id: str, ctx: Any) -> dict[str, Any]:
        task = get_default_store().get(task_id)
        if task is not None and task.tool_name == "delegate_agent":
            from carl_studio.harness.runtime import get_runtime

            return get_runtime(ctx).get(task_id)
        return await legacy_get(task_id)

    async def tasks_cancel(task_id: str, ctx: Any) -> dict[str, Any]:
        task = get_default_store().get(task_id)
        if task is not None and task.tool_name == "delegate_agent":
            from carl_studio.harness.runtime import get_runtime

            return await get_runtime(ctx).cancel(task_id)
        return await legacy_cancel(task_id)

    for function in (tasks_get, tasks_cancel):
        function.__annotations__["ctx"] = Context
        mcp_instance.tool()(function)


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
