"""Tests for the E3 ``LocalDB.maintenance`` path.

Covers three responsibilities: retention sweep on ``sticky_notes``, WAL
checkpoint, and the optional ``VACUUM``. Uses ``tmp_path`` throughout so
nothing in ``~/.carl`` is touched.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from carl_studio.db import LocalDB
from carl_studio.sticky import StickyQueue


@pytest.fixture()
def db(tmp_path: Path) -> LocalDB:
    """Fresh LocalDB rooted in ``tmp_path``."""
    return LocalDB(tmp_path / "carl.db")


def _force_completed_at(db: LocalDB, note_id: str, age_days: int) -> None:
    """Rewrite a row's ``completed_at`` to simulate an aged archive."""
    past = datetime.now(timezone.utc) - timedelta(days=age_days)
    past_str = past.strftime("%Y-%m-%dT%H:%M:%SZ")
    with db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "UPDATE sticky_notes SET completed_at = ? WHERE id = ?",
            (past_str, note_id),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Retention sweep
# ---------------------------------------------------------------------------


def test_maintenance_empty_db_is_safe(db: LocalDB) -> None:
    """Maintenance never raises on an empty DB."""
    stats = db.maintenance()
    assert stats["notes_deleted"] == 0
    wal = stats.get("wal_checkpoint")
    assert isinstance(wal, dict)
    assert set(wal.keys()) == {"busy", "log_pages", "checkpointed"}
    assert stats["vacuumed"] is False


def test_maintenance_retention_deletes_old_archived_notes(db: LocalDB) -> None:
    """Archived rows older than ``retention_days`` are pruned."""
    q = StickyQueue(db)
    old = q.append("old archived", priority=5)
    young = q.append("young archived", priority=5)
    q.archive(old.id)
    q.archive(young.id)

    # Both ``archive`` calls leave ``completed_at`` NULL on these rows
    # because ``archive`` does not stamp it. Force ``completed_at`` for
    # determinism — the old row will be 60 days old, the young one 1 day.
    _force_completed_at(db, old.id, age_days=60)
    _force_completed_at(db, young.id, age_days=1)

    stats = db.maintenance(retention_days=30)
    assert stats["notes_deleted"] == 1
    assert q.get(old.id) is None
    assert q.get(young.id) is not None


def test_maintenance_retention_zero_skips_sweep(db: LocalDB) -> None:
    """``retention_days=0`` disables the sweep entirely."""
    q = StickyQueue(db)
    ancient = q.append("ancient", priority=5)
    q.archive(ancient.id)
    _force_completed_at(db, ancient.id, age_days=9999)

    stats = db.maintenance(retention_days=0)
    assert stats["notes_deleted"] == 0
    assert q.get(ancient.id) is not None


def test_maintenance_retention_ignores_non_archived(db: LocalDB) -> None:
    """``done`` / ``queued`` / ``processing`` rows are never swept."""
    q = StickyQueue(db)
    done = q.append("still here", priority=5)
    q.dequeue()
    q.complete(done.id, {"ok": True})
    _force_completed_at(db, done.id, age_days=9999)

    stats = db.maintenance(retention_days=30)
    assert stats["notes_deleted"] == 0
    assert q.get(done.id) is not None


def test_maintenance_uses_created_at_when_completed_at_null(db: LocalDB) -> None:
    """COALESCE to ``created_at`` — archived legacy rows with NULL
    ``completed_at`` still age out rather than surviving forever."""
    q = StickyQueue(db)
    note = q.append("legacy archived", priority=5)
    q.archive(note.id)
    # Intentionally leave ``completed_at`` NULL; force ``created_at`` back.
    past = datetime.now(timezone.utc) - timedelta(days=60)
    past_str = past.strftime("%Y-%m-%dT%H:%M:%SZ")
    with db._connect() as conn:  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "UPDATE sticky_notes SET completed_at = NULL, created_at = ? "
            "WHERE id = ?",
            (past_str, note.id),
        )
        conn.commit()

    stats = db.maintenance(retention_days=30)
    assert stats["notes_deleted"] == 1


def test_maintenance_rejects_negative_retention(db: LocalDB) -> None:
    with pytest.raises(ValueError):
        db.maintenance(retention_days=-1)


# ---------------------------------------------------------------------------
# WAL checkpoint
# ---------------------------------------------------------------------------


def test_maintenance_returns_wal_checkpoint_metadata(db: LocalDB) -> None:
    """The checkpoint row shape is preserved through the dict wrapper."""
    stats = db.maintenance()
    wal = stats["wal_checkpoint"]
    assert isinstance(wal, dict)
    # On a fresh WAL DB all three ints are populated. We don't assert
    # specific values because SQLite version differences affect the shape
    # of log pages / checkpointed — ``busy`` is the one we care about for
    # the success signal and it should always be ``0`` here.
    busy = wal.get("busy")
    if busy is not None:
        assert isinstance(busy, int)


def test_maintenance_checkpoint_truncates_wal(db: LocalDB, tmp_path: Path) -> None:
    """After maintenance with ``TRUNCATE`` the WAL file is at or near zero.

    Pre-populate with a handful of writes to grow the WAL, then confirm
    the file shrinks (or stays at the initial small size SQLite keeps
    while WAL is active). On macOS the WAL can be unlinked or reduced to
    a header — we tolerate either.
    """
    # Write enough rows to guarantee a non-trivial WAL.
    q = StickyQueue(db)
    for i in range(50):
        q.append(f"write {i}", priority=5)
    wal_path = tmp_path / "carl.db-wal"
    pre_size = wal_path.stat().st_size if wal_path.exists() else 0
    stats = db.maintenance()
    assert stats["wal_checkpoint"] is not None
    post_size = wal_path.stat().st_size if wal_path.exists() else 0
    # TRUNCATE should at worst keep it the same; typically shrinks.
    assert post_size <= pre_size


# ---------------------------------------------------------------------------
# VACUUM
# ---------------------------------------------------------------------------


def test_maintenance_vacuum_off_by_default(db: LocalDB) -> None:
    stats = db.maintenance()
    assert stats["vacuumed"] is False


def test_maintenance_vacuum_opt_in_runs_successfully(db: LocalDB) -> None:
    """``vacuum=True`` executes ``VACUUM`` without raising."""
    q = StickyQueue(db)
    for i in range(20):
        note = q.append(f"work {i}", priority=5)
        q.archive(note.id)

    stats = db.maintenance(retention_days=0, vacuum=True)
    assert stats["vacuumed"] is True


def test_maintenance_vacuum_after_retention_works(db: LocalDB) -> None:
    """VACUUM runs cleanly after a retention sweep committed freed pages."""
    q = StickyQueue(db)
    for i in range(10):
        note = q.append(f"{i}", priority=5)
        q.archive(note.id)
        _force_completed_at(db, note.id, age_days=90)

    stats = db.maintenance(retention_days=30, vacuum=True)
    assert stats["notes_deleted"] == 10
    assert stats["vacuumed"] is True
