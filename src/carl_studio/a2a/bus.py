"""A2A message bus implementations.

LocalBus: SQLite-backed, polling-based, offline-first. No external dependencies.
SupabaseBus: stub — raises NotImplementedError (PAID tier, future work).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from carl_studio.a2a.message import A2AMessage
from carl_studio.a2a.task import A2ATask, A2ATaskStatus

_A2A_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS a2a_tasks (
    id TEXT PRIMARY KEY,
    skill TEXT NOT NULL,
    inputs TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    sender TEXT NOT NULL DEFAULT '',
    receiver TEXT NOT NULL DEFAULT 'carl-studio',
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_status ON a2a_tasks (status);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_receiver ON a2a_tasks (receiver);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_receiver_status ON a2a_tasks (receiver, status);

CREATE TABLE IF NOT EXISTS a2a_messages (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES a2a_tasks(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    sender TEXT NOT NULL DEFAULT 'carl-studio',
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_messages_task ON a2a_messages (task_id);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_from_row(row: sqlite3.Row) -> A2ATask:
    """Deserialize an A2ATask from a sqlite3.Row."""
    return A2ATask(
        id=row["id"],
        skill=row["skill"],
        inputs=json.loads(row["inputs"] or "{}"),
        status=A2ATaskStatus(row["status"]),
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        sender=row["sender"] or "",
        receiver=row["receiver"] or "carl-studio",
        priority=row["priority"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _message_from_row(row: sqlite3.Row) -> A2AMessage:
    """Deserialize an A2AMessage from a sqlite3.Row."""
    return A2AMessage(
        id=row["id"],
        task_id=row["task_id"],
        type=row["type"],
        payload=json.loads(row["payload"] or "{}"),
        sender=row["sender"] or "carl-studio",
        timestamp=row["timestamp"],
    )


class LocalBus:
    """SQLite-backed A2A bus.

    Polling-based — no network, no threads required. Safe for concurrent
    readers (WAL mode). Write operations use immediate transactions to
    prevent writer starvation.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        from carl_studio.db import CARL_DIR

        self._path = db_path or (CARL_DIR / "a2a.db")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_A2A_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection with row_factory set.

        Reuses a single persistent connection per LocalBus instance.
        Rolls back on exception to keep the connection usable.
        """
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

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ─── Task operations ────────────────────────────────────────────────────

    def post(self, task: A2ATask) -> str:
        """Insert a task into the queue.

        Returns the task id. Raises ValueError if task id already exists.
        """
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO a2a_tasks
                       (id, skill, inputs, status, result, error, sender, receiver,
                        priority, created_at, updated_at, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        task.id,
                        task.skill,
                        json.dumps(task.inputs),
                        task.status.value,
                        json.dumps(task.result) if task.result is not None else None,
                        task.error,
                        task.sender,
                        task.receiver,
                        task.priority,
                        task.created_at,
                        task.updated_at,
                        task.completed_at,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Task id already exists: {task.id}") from exc
        return task.id

    def poll(
        self,
        receiver: str = "carl-studio",
        status: str = "pending",
        limit: int = 10,
    ) -> list[A2ATask]:
        """Return up to ``limit`` tasks for ``receiver`` in the given ``status``.

        Ordered by priority DESC, then created_at ASC (highest priority first,
        FIFO within same priority).
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM a2a_tasks
                   WHERE receiver = ? AND status = ?
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (receiver, status, limit),
            ).fetchall()
        return [_task_from_row(r) for r in rows]

    def get(self, task_id: str) -> A2ATask | None:
        """Retrieve a single task by id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM a2a_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return _task_from_row(row)

    def update(self, task: A2ATask) -> None:
        """Persist the current state of a task (all mutable fields)."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE a2a_tasks
                   SET status = ?, result = ?, error = ?, updated_at = ?,
                       completed_at = ?, priority = ?
                   WHERE id = ?""",
                (
                    task.status.value,
                    json.dumps(task.result) if task.result is not None else None,
                    task.error,
                    task.updated_at,
                    task.completed_at,
                    task.priority,
                    task.id,
                ),
            )
            conn.commit()

    def cancel(self, task_id: str) -> None:
        """Set a task's status to CANCELLED.

        No-op if the task is already in a terminal state or does not exist.
        """
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """UPDATE a2a_tasks
                   SET status = 'cancelled', updated_at = ?, completed_at = ?
                   WHERE id = ? AND status NOT IN ('done', 'failed', 'cancelled')""",
                (now, now, task_id),
            )
            conn.commit()

    def pending_count(self, receiver: str = "carl-studio") -> int:
        """Count pending tasks for a given receiver."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM a2a_tasks WHERE receiver = ? AND status = 'pending'",
                (receiver,),
            ).fetchone()
        return int(row["n"]) if row else 0

    # ─── Message operations ──────────────────────────────────────────────────

    def publish_message(self, msg: A2AMessage) -> None:
        """Insert a message associated with a task.

        Raises ValueError if the referenced task_id does not exist.
        """
        with self._connect() as conn:
            # Verify the task exists (foreign key enforcement may be PRAGMA-gated)
            task_row = conn.execute(
                "SELECT id FROM a2a_tasks WHERE id = ?",
                (msg.task_id,),
            ).fetchone()
            if task_row is None:
                raise ValueError(
                    f"Cannot publish message: task '{msg.task_id}' not found."
                )
            try:
                conn.execute(
                    """INSERT INTO a2a_messages (id, task_id, type, payload, sender, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        msg.id,
                        msg.task_id,
                        msg.type,
                        json.dumps(msg.payload),
                        msg.sender,
                        msg.timestamp,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Message id already exists: {msg.id}") from exc

    def get_messages(self, task_id: str) -> list[A2AMessage]:
        """Return all messages for a task, ordered by timestamp ASC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM a2a_messages WHERE task_id = ? ORDER BY timestamp ASC",
                (task_id,),
            ).fetchall()
        return [_message_from_row(r) for r in rows]

    # ─── Context manager support ─────────────────────────────────────────────

    def __enter__(self) -> "LocalBus":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SupabaseBus:
    """Supabase Realtime A2A bus (PAID tier).

    Stub implementation — raises NotImplementedError on every method.
    Will be wired to Supabase Realtime channels in a future release.
    """

    def post(self, task: A2ATask) -> str:
        raise NotImplementedError(
            "SupabaseBus requires PAID tier and a Supabase connection. "
            "Use LocalBus for offline operation."
        )

    def poll(
        self,
        receiver: str = "carl-studio",
        status: str = "pending",
        limit: int = 10,
    ) -> list[A2ATask]:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def get(self, task_id: str) -> A2ATask | None:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def update(self, task: A2ATask) -> None:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def publish_message(self, msg: A2AMessage) -> None:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def get_messages(self, task_id: str) -> list[A2AMessage]:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def cancel(self, task_id: str) -> None:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")

    def pending_count(self, receiver: str = "carl-studio") -> int:
        raise NotImplementedError("SupabaseBus not yet implemented. Use LocalBus.")
