"""Tests for the E2 sticky-note reclaim paths (``requeue`` + ``reclaim_stale``).

These cover the crash-recovery contract described in the v0.6.1 hardening
spec — a row left in ``processing`` by a dead process must not wedge the
queue. Two entry points:

* :meth:`StickyQueue.requeue` — single-row, used by the heartbeat loop on
  exception.
* :meth:`StickyQueue.reclaim_stale` — bulk, used by the daemon on boot and
  periodically during operation.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from carl_studio.db import LocalDB
from carl_studio.sticky import StickyQueue


@pytest.fixture()
def queue(tmp_path: Path) -> StickyQueue:
    """Isolated :class:`StickyQueue` rooted in ``tmp_path``."""
    db = LocalDB(tmp_path / "carl.db")
    return StickyQueue(db)


# ---------------------------------------------------------------------------
# StickyQueue.requeue
# ---------------------------------------------------------------------------


def test_requeue_flips_processing_row_back_to_queued(queue: StickyQueue) -> None:
    """A ``processing`` row transitions back to ``queued`` and ``started_at``
    is cleared so a subsequent :meth:`dequeue` re-stamps it cleanly."""
    note = queue.append("train a classifier", priority=5)
    claimed = queue.dequeue()
    assert claimed is not None and claimed.id == note.id
    assert claimed.status == "processing"
    assert claimed.started_at is not None

    flipped = queue.requeue(note.id)
    assert flipped is True

    after = queue.get(note.id)
    assert after is not None
    assert after.status == "queued"
    assert after.started_at is None


def test_requeue_is_noop_on_done_note(queue: StickyQueue) -> None:
    """``requeue`` never resurrects a ``done`` note — returns ``False``."""
    note = queue.append("ship it", priority=5)
    claimed = queue.dequeue()
    assert claimed is not None
    queue.complete(note.id, {"ok": True})

    flipped = queue.requeue(note.id)
    assert flipped is False

    still_done = queue.get(note.id)
    assert still_done is not None
    assert still_done.status == "done"


def test_requeue_is_noop_on_queued_note(queue: StickyQueue) -> None:
    """Idempotent on ``queued`` rows — prevents double-flips from races."""
    note = queue.append("already queued", priority=5)
    flipped = queue.requeue(note.id)
    assert flipped is False
    still = queue.get(note.id)
    assert still is not None and still.status == "queued"


def test_requeue_is_noop_on_missing_id(queue: StickyQueue) -> None:
    """Unknown ids do not raise — returns ``False``."""
    assert queue.requeue("sn-doesnotexist") is False


def test_requeue_rejects_empty_id(queue: StickyQueue) -> None:
    with pytest.raises(ValueError):
        queue.requeue("")
    with pytest.raises(ValueError):
        queue.requeue("   ")


def test_requeued_note_is_next_in_line(queue: StickyQueue) -> None:
    """A requeued priority-9 note wins the next :meth:`dequeue`."""
    urgent = queue.append("urgent work", priority=9)
    claimed = queue.dequeue()
    assert claimed is not None and claimed.id == urgent.id

    queue.requeue(urgent.id)

    next_claim = queue.dequeue()
    assert next_claim is not None
    assert next_claim.id == urgent.id


# ---------------------------------------------------------------------------
# StickyQueue.reclaim_stale
# ---------------------------------------------------------------------------


def _force_started_at(queue: StickyQueue, note_id: str, age_seconds: int) -> None:
    """Test helper — rewrite a row's ``started_at`` to simulate elapsed time.

    We can't wait real seconds in a unit test so we bypass the API and
    mutate the row directly. This is the only place test code pokes at the
    DB outside the public API.
    """
    past = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    past_str = past.strftime("%Y-%m-%dT%H:%M:%SZ")
    with queue._db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "UPDATE sticky_notes SET started_at = ? WHERE id = ?",
            (past_str, note_id),
        )
        conn.commit()


def test_reclaim_stale_empty_queue_returns_zero(queue: StickyQueue) -> None:
    assert queue.reclaim_stale() == 0


def test_reclaim_stale_flips_old_rows_only(queue: StickyQueue) -> None:
    """Only rows older than ``max_age_seconds`` flip — younger rows survive."""
    old_note = queue.append("old work", priority=5)
    young_note = queue.append("young work", priority=5)

    old_claim = queue.dequeue()
    assert old_claim is not None and old_claim.id == old_note.id
    young_claim = queue.dequeue()
    assert young_claim is not None and young_claim.id == young_note.id

    _force_started_at(queue, old_note.id, age_seconds=3600)

    reclaimed = queue.reclaim_stale(max_age_seconds=600)
    assert reclaimed == 1

    after_old = queue.get(old_note.id)
    after_young = queue.get(young_note.id)
    assert after_old is not None and after_old.status == "queued"
    assert after_old.started_at is None
    assert after_young is not None and after_young.status == "processing"


def test_reclaim_stale_zero_max_age_flips_all_processing(queue: StickyQueue) -> None:
    """``max_age_seconds=0`` is the "reclaim everything now" knob."""
    a = queue.append("a", priority=5)
    b = queue.append("b", priority=5)
    queue.dequeue()
    queue.dequeue()

    reclaimed = queue.reclaim_stale(max_age_seconds=0)
    assert reclaimed == 2
    assert (queue.get(a.id) or a).status == "queued"
    assert (queue.get(b.id) or b).status == "queued"


def test_reclaim_stale_ignores_done_and_queued_rows(queue: StickyQueue) -> None:
    """Only ``processing`` rows are candidates."""
    done_note = queue.append("done task", priority=5)
    queued_note = queue.append("queued task", priority=5)

    done_claim = queue.dequeue()
    assert done_claim is not None and done_claim.id == done_note.id
    queue.complete(done_note.id, {"ok": True})

    reclaimed = queue.reclaim_stale(max_age_seconds=0)
    assert reclaimed == 0
    assert (queue.get(done_note.id) or done_note).status == "done"
    assert (queue.get(queued_note.id) or queued_note).status == "queued"


def test_reclaim_stale_flips_null_started_at_defensively(queue: StickyQueue) -> None:
    """Rows in ``processing`` with ``started_at IS NULL`` are also reclaimed.

    The schema in :meth:`StickyQueue.dequeue` sets both in one statement, but
    the SQL in :meth:`reclaim_stale` defensively handles a NULL ``started_at``
    so a row wedged by future divergence doesn't stay stuck forever.
    """
    note = queue.append("pathological", priority=5)
    with queue._db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "UPDATE sticky_notes SET status = 'processing', started_at = NULL "
            "WHERE id = ?",
            (note.id,),
        )
        conn.commit()

    reclaimed = queue.reclaim_stale(max_age_seconds=600)
    assert reclaimed == 1
    assert (queue.get(note.id) or note).status == "queued"


def test_reclaim_stale_rejects_negative_max_age(queue: StickyQueue) -> None:
    with pytest.raises(ValueError):
        queue.reclaim_stale(max_age_seconds=-1)


# ---------------------------------------------------------------------------
# StickyQueue.oldest_processing_age_seconds — doctor surface helper
# ---------------------------------------------------------------------------


def test_oldest_processing_age_none_when_empty(queue: StickyQueue) -> None:
    assert queue.oldest_processing_age_seconds() is None


def test_oldest_processing_age_none_when_no_processing(queue: StickyQueue) -> None:
    queue.append("queued only", priority=5)
    assert queue.oldest_processing_age_seconds() is None


def test_oldest_processing_age_reports_oldest(queue: StickyQueue) -> None:
    old = queue.append("old", priority=5)
    queue.append("young", priority=5)
    queue.dequeue()
    queue.dequeue()
    _force_started_at(queue, old.id, age_seconds=3600)
    age = queue.oldest_processing_age_seconds()
    assert age is not None
    # Allow a wide window; what matters is we picked the ~1h-old row,
    # not the fresh one.
    assert age >= 3500.0
