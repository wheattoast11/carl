"""Tests for carl_studio.sticky (sticky-note queue foundation, ARC-004 Phase-1)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from carl_studio.db import LocalDB
from carl_studio.sticky import StickyNote, StickyQueue


@pytest.fixture()
def queue(tmp_path: Path) -> StickyQueue:
    """Isolated LocalDB + StickyQueue per test, rooted in tmp_path."""
    db = LocalDB(tmp_path / "carl.db")
    return StickyQueue(db)


def test_append_returns_queued_note(queue: StickyQueue) -> None:
    note = queue.append("train tiny cartpole", priority=7, session_id="sess-1")

    assert isinstance(note, StickyNote)
    assert note.content == "train tiny cartpole"
    assert note.status == "queued"
    assert note.priority == 7
    assert note.session_id == "sess-1"
    assert note.id.startswith("sn-") and len(note.id) == len("sn-") + 12
    assert note.started_at is None
    assert note.completed_at is None
    assert note.result is None

    # Round-trip through the DB.
    fetched = queue.get(note.id)
    assert fetched is not None
    assert fetched.id == note.id
    assert fetched.status == "queued"


def test_dequeue_returns_highest_priority_first(queue: StickyQueue) -> None:
    low = queue.append("low-priority task", priority=1)
    # Ensure distinct created_at ordering for the tie-break path.
    time.sleep(0.01)
    high = queue.append("urgent task", priority=9)
    time.sleep(0.01)
    mid_a = queue.append("mid task a", priority=5)
    time.sleep(0.01)
    mid_b = queue.append("mid task b", priority=5)

    first = queue.dequeue()
    assert first is not None
    assert first.id == high.id
    assert first.status == "processing"
    assert first.started_at is not None

    # Among equal priorities, oldest created_at wins.
    second = queue.dequeue()
    third = queue.dequeue()
    fourth = queue.dequeue()
    assert second is not None and third is not None and fourth is not None
    assert second.id == mid_a.id
    assert third.id == mid_b.id
    assert fourth.id == low.id

    # Queue is now drained; dequeue is idempotent-empty.
    assert queue.dequeue() is None


def test_dequeue_empty_returns_none(queue: StickyQueue) -> None:
    assert queue.dequeue() is None


def test_complete_transitions_status_and_records_result(queue: StickyQueue) -> None:
    note = queue.append("compute digest", priority=3)
    claimed = queue.dequeue()
    assert claimed is not None and claimed.id == note.id

    queue.complete(note.id, {"digest": "abc123", "elapsed_ms": 42})

    done = queue.get(note.id)
    assert done is not None
    assert done.status == "done"
    assert done.result == {"digest": "abc123", "elapsed_ms": 42}
    assert done.completed_at is not None
    # started_at survives the complete() transition.
    assert done.started_at is not None


def test_archive_transitions_status(queue: StickyQueue) -> None:
    note = queue.append("abandon me", priority=2)
    queue.archive(note.id)

    archived = queue.get(note.id)
    assert archived is not None
    assert archived.status == "archived"

    # Archived notes are invisible to dequeue.
    assert queue.dequeue() is None


def test_status_filters_and_orders(queue: StickyQueue) -> None:
    a = queue.append("a", priority=1)
    time.sleep(0.01)
    b = queue.append("b", priority=5)
    time.sleep(0.01)
    c = queue.append("c", priority=5)
    time.sleep(0.01)
    d = queue.append("d", priority=9)

    # Drain two into processing.
    claimed = queue.dequeue()  # d (priority 9)
    assert claimed is not None and claimed.id == d.id
    claimed2 = queue.dequeue()  # b (priority 5, older than c)
    assert claimed2 is not None and claimed2.id == b.id

    # Default: all statuses, priority DESC then created_at ASC.
    full = queue.status()
    # In SQLite ORDER BY priority DESC, created_at ASC is what test behavior currently produces
    # Note: b is older than c, so it gets dequeued first.
    assert [n.id for n in full] == [d.id, b.id, c.id, a.id]

    # Filter to queued: only c and a remain queued, priority DESC.
    queued_only = queue.status(status="queued")
    assert [n.id for n in queued_only] == [c.id, a.id]
    assert all(n.status == "queued" for n in queued_only)

    # Filter to processing: b and d, priority DESC.
    processing_only = queue.status(status="processing")
    assert {n.id for n in processing_only} == {b.id, d.id}
    assert all(n.status == "processing" for n in processing_only)

    # Limit clamps result size.
    limited = queue.status(limit=1)
    assert len(limited) == 1
    assert limited[0].id == d.id


def test_append_rejects_empty_content(queue: StickyQueue) -> None:
    with pytest.raises(ValueError):
        queue.append("", priority=5)
    with pytest.raises(ValueError):
        queue.append("   ", priority=5)


def test_jit_context_round_trips(queue: StickyQueue) -> None:
    ctx = {"focus": "cartpole-v1", "budget_usd": 0.25, "tags": ["rl", "tiny"]}
    note = queue.append("warm up", priority=4, jit_context=ctx)
    fetched = queue.get(note.id)
    assert fetched is not None
    assert fetched.jit_context == ctx
