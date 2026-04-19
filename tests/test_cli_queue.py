"""Tests for `carl queue ...` CLI (ARC-007)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import carl_studio.db as db_mod
from carl_studio.cli import app
from carl_studio.db import LocalDB
from carl_studio.sticky import StickyQueue


runner = CliRunner()


@pytest.fixture()
def isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the process-wide CARL dir + DB path into ``tmp_path``.

    Mirrors the pattern used by the other CLI tests (see tests/test_cli.py).
    """
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")
    return tmp_path / "carl.db"


def test_queue_add_appends_note(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "add", "train cartpole", "-p", "7"])
    assert result.exit_code == 0, result.output
    assert "queued" in result.output
    assert "priority=7" in result.output

    notes = StickyQueue(LocalDB()).status(limit=10)
    assert len(notes) == 1
    assert notes[0].content == "train cartpole"
    assert notes[0].priority == 7
    assert notes[0].status == "queued"


def test_queue_add_rejects_empty_content(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "add", "   "])
    assert result.exit_code == 1
    assert "must not be empty" in result.output


def test_queue_add_rejects_out_of_range_priority(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "add", "hello", "-p", "0"])
    assert result.exit_code == 1
    assert "priority must be in 1..10" in result.output

    result = runner.invoke(app, ["queue", "add", "hello", "-p", "11"])
    assert result.exit_code == 1
    assert "priority must be in 1..10" in result.output


def test_queue_list_empty_prints_info(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "list"])
    assert result.exit_code == 0
    assert "queue empty" in result.output


def test_queue_list_with_notes_renders_table(isolated_db: Path) -> None:
    q = StickyQueue(LocalDB())
    q.append("alpha", priority=9)
    q.append("bravo", priority=3)

    result = runner.invoke(app, ["queue", "list"])
    assert result.exit_code == 0
    # Highest priority first — table header + two note bodies.
    assert "alpha" in result.output
    assert "bravo" in result.output
    # Priority DESC ordering — alpha (9) precedes bravo (3).
    assert result.output.index("alpha") < result.output.index("bravo")


def test_queue_list_rejects_invalid_status(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "list", "--status", "garbage"])
    assert result.exit_code != 0
    assert "status must be one of" in result.output


def test_queue_list_filters_by_status(isolated_db: Path) -> None:
    q = StickyQueue(LocalDB())
    queued = q.append("still waiting", priority=5)
    claimed = q.dequeue()
    assert claimed is not None
    # One note stays queued after dequeue.
    q.append("next up", priority=2)

    result = runner.invoke(app, ["queue", "list", "--status", "processing"])
    assert result.exit_code == 0
    assert claimed.id in result.output
    assert "next up" not in result.output
    assert queued.id == claimed.id  # sanity — same note transitioned


def test_queue_status_shows_bucket_counts(isolated_db: Path) -> None:
    q = StickyQueue(LocalDB())
    q.append("one", priority=5)
    q.append("two", priority=5)
    claimed = q.dequeue()
    assert claimed is not None
    q.complete(claimed.id, {"ok": True})
    q.append("three", priority=5)

    result = runner.invoke(app, ["queue", "status"])
    assert result.exit_code == 0
    # Expect queued=2, processing=0, done=1, archived=0.
    assert "queued" in result.output
    assert "processing" in result.output
    assert "done" in result.output
    assert "archived" in result.output
    # At least one "pending note(s) awaiting Carl" line because queued > 0.
    assert "pending note(s) awaiting Carl" in result.output


def test_queue_status_empty_reports_no_pending(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "status"])
    assert result.exit_code == 0
    assert "No pending work" in result.output


def test_queue_clear_done_archives_only_done(isolated_db: Path) -> None:
    q = StickyQueue(LocalDB())
    a = q.append("a", priority=9)
    b = q.append("b", priority=5)
    # Drain a, complete it; b stays queued.
    claimed = q.dequeue()
    assert claimed is not None and claimed.id == a.id
    q.complete(a.id, {"ok": True})

    result = runner.invoke(app, ["queue", "clear"])  # default: --done
    assert result.exit_code == 0
    assert "archived 1 done note(s)" in result.output

    # a is now archived, b untouched.
    a_after = q.get(a.id)
    b_after = q.get(b.id)
    assert a_after is not None and a_after.status == "archived"
    assert b_after is not None and b_after.status == "queued"


def test_queue_clear_all_archives_non_archived(isolated_db: Path) -> None:
    q = StickyQueue(LocalDB())
    a = q.append("a", priority=9)
    b = q.append("b", priority=5)
    # a in processing, b queued — both non-archived.
    claimed = q.dequeue()
    assert claimed is not None and claimed.id == a.id

    # Pre-archive a separate note so we can confirm it is not re-processed.
    c = q.append("already gone", priority=1)
    q.archive(c.id)

    result = runner.invoke(app, ["queue", "clear", "--all"])
    assert result.exit_code == 0
    assert "archived 2 non-archived note(s)" in result.output

    for note_id, expected in ((a.id, "archived"), (b.id, "archived"), (c.id, "archived")):
        note = q.get(note_id)
        assert note is not None and note.status == expected


def test_queue_clear_done_no_op_on_empty_queue(isolated_db: Path) -> None:
    result = runner.invoke(app, ["queue", "clear"])
    assert result.exit_code == 0
    assert "archived 0 done note(s)" in result.output
