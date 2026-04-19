"""Sticky-note persistent priority queue.

User-appended work items awaiting processing by the CARL agent. Backed by
the ``sticky_notes`` table in ``carl.db``. This module is intentionally
independent of any CLI wiring or heartbeat loop — those are separate layers
that consume this API.

Status lifecycle: ``queued -> processing -> done | archived``. ``dequeue()``
atomically claims the highest-priority queued note (priority DESC, then
created_at ASC) and flips it to ``processing``; ``complete()`` records a
result and flips it to ``done``; ``archive()`` is a soft delete.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from carl_studio.db import LocalDB

StickyStatus = Literal["queued", "processing", "done", "archived"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_id() -> str:
    return f"sn-{uuid.uuid4().hex[:12]}"


class StickyNote(BaseModel):
    """A single queued work item."""

    id: str = Field(default_factory=_gen_id)
    content: str
    status: StickyStatus = "queued"
    priority: int = 5
    session_id: str | None = None
    jit_context: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    created_at: str = Field(default_factory=_now_iso)
    started_at: str | None = None
    completed_at: str | None = None


class StickyQueue:
    """Persistent priority queue for user-appended sticky notes."""

    def __init__(self, db: LocalDB) -> None:
        self._db = db

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Thin wrapper around ``LocalDB._connect()`` for local scoping."""
        # LocalDB intentionally exposes _connect as the shared SQLite handle;
        # we isolate the access here so callers in this module read naturally.
        with self._db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
            yield conn

    def append(
        self,
        content: str,
        *,
        priority: int = 5,
        session_id: str | None = None,
        jit_context: Mapping[str, Any] | None = None,
    ) -> StickyNote:
        """Enqueue a new note. Returns the created ``StickyNote``."""
        if not content or not content.strip():
            raise ValueError("StickyQueue.append: content must be a non-empty string")

        note = StickyNote(
            content=content,
            priority=priority,
            session_id=session_id,
            jit_context=dict(jit_context) if jit_context else None,
        )
        self._write(note)
        return note

    def dequeue(self) -> StickyNote | None:
        """Atomically claim the highest-priority queued note.

        Transitions status ``queued -> processing`` and sets ``started_at``.
        Returns ``None`` if the queue is empty.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM sticky_notes WHERE status = 'queued' "
                "ORDER BY priority DESC, created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            note_id = cast(str, row[0])
            now = _now_iso()
            cursor = conn.execute(
                "UPDATE sticky_notes SET status = 'processing', started_at = ? "
                "WHERE id = ? AND status = 'queued'",
                (now, note_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                # Lost a race with another claimer; treat as empty.
                return None
        return self.get(note_id)

    def complete(self, note_id: str, result: Mapping[str, Any]) -> None:
        """Record a result dict and mark the note as ``done``."""
        if not note_id:
            raise ValueError("StickyQueue.complete: note_id must be a non-empty string")
        payload = json.dumps(dict(result))
        with self._conn() as conn:
            conn.execute(
                "UPDATE sticky_notes SET status = 'done', result = ?, completed_at = ? "
                "WHERE id = ?",
                (payload, _now_iso(), note_id),
            )
            conn.commit()

    def archive(self, note_id: str) -> None:
        """Soft-delete a note (transition to ``archived``)."""
        if not note_id:
            raise ValueError("StickyQueue.archive: note_id must be a non-empty string")
        with self._conn() as conn:
            conn.execute(
                "UPDATE sticky_notes SET status = 'archived' WHERE id = ?",
                (note_id,),
            )
            conn.commit()

    def get(self, note_id: str) -> StickyNote | None:
        """Fetch a single note by id, or ``None`` if missing."""
        if not note_id:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, content, status, priority, session_id, jit_context, result, "
                "created_at, started_at, completed_at FROM sticky_notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        return _row_to_note(row) if row else None

    def status(
        self,
        *,
        limit: int = 20,
        status: StickyStatus | None = None,
    ) -> list[StickyNote]:
        """List notes ordered by priority DESC, then created_at DESC.

        ``status`` is an optional filter; ``limit`` caps the result set.
        """
        if limit <= 0:
            raise ValueError("StickyQueue.status: limit must be a positive int")

        query = (
            "SELECT id, content, status, priority, session_id, jit_context, result, "
            "created_at, started_at, completed_at FROM sticky_notes"
        )
        args: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            args = (status,)
        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        args = (*args, limit)

        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
        return [_row_to_note(r) for r in rows if r]

    def _write(self, note: StickyNote) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sticky_notes "
                "(id, content, status, priority, session_id, jit_context, result, "
                "created_at, started_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    note.id,
                    note.content,
                    note.status,
                    note.priority,
                    note.session_id,
                    json.dumps(note.jit_context) if note.jit_context is not None else None,
                    json.dumps(note.result) if note.result is not None else None,
                    note.created_at,
                    note.started_at,
                    note.completed_at,
                ),
            )
            conn.commit()


def _row_to_note(row: Any) -> StickyNote:
    """Rehydrate a ``StickyNote`` from a SQLite row (sqlite3.Row or tuple)."""
    jit_raw = row[5]
    result_raw = row[6]
    return StickyNote(
        id=row[0],
        content=row[1],
        status=row[2],
        priority=row[3],
        session_id=row[4],
        jit_context=json.loads(jit_raw) if jit_raw else None,
        result=json.loads(result_raw) if result_raw else None,
        created_at=row[7],
        started_at=row[8],
        completed_at=row[9],
    )
