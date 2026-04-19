"""Tests for ``carl_studio.db.LocalDB`` lifecycle correctness.

Covers:

* **REV-002** — ``_connect()`` commits on clean exit, rolls back on
  exception without swallowing the original failure, and serializes
  access across threads.
* **REV-008** — if ``_init_schema()`` raises the half-open connection
  is torn down so subsequent ``LocalDB`` instances can start clean.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from carl_studio.db import LocalDB


# ---------------------------------------------------------------------------
# REV-002 — commit on clean exit, rollback on exception
# ---------------------------------------------------------------------------


def test_commit_on_clean_exit(tmp_path: Path) -> None:
    """A write through ``_connect()`` is visible without an explicit commit.

    Before REV-002 callers had to remember ``conn.commit()`` inside every
    ``with self._connect()`` block. This test proves the context manager
    now commits for them: we insert a config row through the private
    context manager *without* a trailing ``conn.commit()`` call, close
    the DB, reopen it from a fresh ``LocalDB``, and assert the row
    survived the round trip.
    """
    db = LocalDB(tmp_path / "commit.db")
    with db._connect() as conn:
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?)",
            ("rev002-commit", "yes"),
        )
        # Deliberately no explicit ``conn.commit()`` — REV-002 makes the
        # context manager responsible.
    db.close()

    db2 = LocalDB(tmp_path / "commit.db")
    try:
        assert db2.get_config("rev002-commit") == "yes"
    finally:
        db2.close()


def test_rollback_on_exception(tmp_path: Path) -> None:
    """A raised exception inside ``_connect()`` must be re-raised verbatim.

    Pre-fix the rollback itself could raise and clobber the original
    exception — callers would see a ``sqlite3.ProgrammingError`` about a
    closed connection instead of the real bug. We assert the *original*
    ``RuntimeError`` propagates and that state written during the
    transaction was rolled back (not persisted).
    """
    db = LocalDB(tmp_path / "rollback.db")
    sentinel = RuntimeError("rev002-rollback-marker")

    with pytest.raises(RuntimeError, match="rev002-rollback-marker"):
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                ("rollback-key", "should-disappear"),
            )
            raise sentinel

    # Row must not have been committed because the context manager
    # called rollback on the exception path.
    assert db.get_config("rollback-key") is None
    db.close()


def test_original_exception_preserved_even_if_rollback_fails(
    tmp_path: Path,
) -> None:
    """If ``rollback()`` itself raises, keep the caller's exception alive.

    We wrap the real sqlite connection in a proxy whose ``rollback``
    explodes. The fix in REV-002 must swallow that secondary exception
    and re-raise the caller's ``RuntimeError`` — not
    ``sqlite3.OperationalError`` from the swallowed rollback.
    """
    db = LocalDB(tmp_path / "rollback-fail.db")

    class _PoisonedRollbackConn:
        """Delegates to the real connection but blows up on ``rollback``.

        Every other attribute proxies through so the context-manager
        body can still execute normally — only the cleanup path
        is poisoned.
        """

        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real
            self.rollback_called = False

        def rollback(self) -> None:
            self.rollback_called = True
            raise sqlite3.OperationalError("rollback exploded (simulated)")

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

    real_conn = db._conn
    assert real_conn is not None
    poisoned = _PoisonedRollbackConn(real_conn)
    db._conn = poisoned  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="caller-error"):
        with db._connect() as _conn:
            raise RuntimeError("caller-error")

    assert poisoned.rollback_called, (
        "_connect() must have attempted rollback on the exception path"
    )

    # Restore the real connection so ``db.close()`` works cleanly.
    db._conn = real_conn
    db.close()


# ---------------------------------------------------------------------------
# REV-002 — threading safety
# ---------------------------------------------------------------------------


def test_concurrent_writes_serialize(tmp_path: Path) -> None:
    """Two threads writing through the same ``LocalDB`` do not corrupt state.

    sqlite3.Connection is not thread-safe; prior to REV-002 two threads
    racing on the shared handle would occasionally raise
    ``ProgrammingError`` or lose writes. With the ``threading.Lock``
    in ``_connect()`` plus ``check_same_thread=False`` on the handle
    all writes must succeed and every row must be readable.
    """
    db = LocalDB(tmp_path / "concurrent.db")
    errors: list[BaseException] = []
    n_per_thread = 40

    def _writer(prefix: str) -> None:
        try:
            for i in range(n_per_thread):
                db.set_config(f"{prefix}:{i}", f"value-{i}")
        except BaseException as exc:  # noqa: BLE001 — tests accumulate failures
            errors.append(exc)

    threads = [
        threading.Thread(target=_writer, args=(f"thread-{k}",))
        for k in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"thread errors: {errors!r}"

    # All rows must be visible and correct.
    for k in range(4):
        for i in range(n_per_thread):
            assert db.get_config(f"thread-{k}:{i}") == f"value-{i}"

    db.close()


def test_lock_blocks_concurrent_access(tmp_path: Path) -> None:
    """While one thread holds ``_connect()``, a second caller must wait.

    Proves the lock is actually serializing — not merely present. We
    open the context manager from thread A and sleep briefly; thread B
    attempts to acquire the same LocalDB concurrently. Thread B must
    not return until thread A releases.
    """
    db = LocalDB(tmp_path / "lock.db")
    entered_a = threading.Event()
    release_a = threading.Event()
    b_started_at: list[float] = []
    errors: list[BaseException] = []

    def _thread_a() -> None:
        try:
            with db._connect() as conn:
                conn.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    ("thread-a", "held"),
                )
                entered_a.set()
                # Hold the lock until the main thread explicitly releases us.
                release_a.wait(timeout=5.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _thread_b() -> None:
        try:
            entered_a.wait(timeout=5.0)
            t0 = time.monotonic()
            with db._connect() as conn:
                b_started_at.append(time.monotonic() - t0)
                conn.execute(
                    "INSERT INTO config (key, value) VALUES (?, ?)",
                    ("thread-b", "after"),
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ta = threading.Thread(target=_thread_a)
    tb = threading.Thread(target=_thread_b)
    ta.start()
    tb.start()

    # Give thread B time to block on the lock, then let A finish.
    time.sleep(0.15)
    release_a.set()

    ta.join(timeout=5.0)
    tb.join(timeout=5.0)

    assert not ta.is_alive() and not tb.is_alive()
    assert not errors, f"thread errors: {errors!r}"
    # B started after A's sleep — the lock forced it to wait.
    assert b_started_at, "thread B never acquired the lock"
    assert b_started_at[0] >= 0.1, (
        f"thread B acquired lock too fast ({b_started_at[0]:.3f}s) — "
        "serialization not working"
    )
    assert db.get_config("thread-a") == "held"
    assert db.get_config("thread-b") == "after"
    db.close()


# ---------------------------------------------------------------------------
# REV-008 — init_schema failure must tear down the connection
# ---------------------------------------------------------------------------


def test_init_schema_failure_cleans_up_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``_init_schema`` raises, ``self._conn`` must be ``None`` for cleanup.

    We replace ``sqlite3.connect`` so the *first* invocation returns a
    proxy whose ``executescript`` explodes. Subsequent calls fall
    through to the real ``sqlite3.connect``.

    Assertions:
      1. The original exception propagates intact.
      2. No dangling connection lingers on the first ``LocalDB`` — a
         subsequent, non-failing construction succeeds, proving we
         actually tore down the broken handle.
    """
    import carl_studio.db as db_module

    db_path = tmp_path / "broken.db"
    real_connect = sqlite3.connect
    calls = {"n": 0}
    closed_proxies: list[sqlite3.Connection] = []

    class _FirstConnectExplodes:
        """Proxy that fails on ``executescript`` exactly once."""

        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def executescript(self, _script: str) -> None:
            raise sqlite3.OperationalError("simulated init failure (REV-008)")

        def close(self) -> None:
            closed_proxies.append(self._real)
            self._real.close()

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

        def __setattr__(self, name: str, value: object) -> None:
            if name == "_real":
                super().__setattr__(name, value)
            else:
                setattr(self._real, name, value)

    def _fake_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        calls["n"] += 1
        real = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        if calls["n"] == 1:
            return _FirstConnectExplodes(real)  # type: ignore[return-value]
        return real

    monkeypatch.setattr(db_module.sqlite3, "connect", _fake_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated init failure"):
        LocalDB(db_path)

    # REV-008: the first, broken handle must have been closed by
    # ``self.close()`` in ``__init__``'s ``except`` branch.
    assert closed_proxies, (
        "LocalDB.__init__ must call self.close() on bootstrap failure "
        "so the leaked handle is released"
    )

    # Second construction uses the real sqlite3.connect and must succeed.
    db2 = LocalDB(db_path)
    try:
        db2.set_config("post-recovery", "ok")
        assert db2.get_config("post-recovery") == "ok"
    finally:
        db2.close()


def test_close_tolerates_already_closed_connection(tmp_path: Path) -> None:
    """``close()`` must not raise when called on an already-closed handle.

    Guards the secondary failure path in REV-008: if the connection was
    closed out from under us (e.g. by the exception cleanup in
    ``__init__``) a subsequent ``db.close()`` must still be a no-op.
    """
    db = LocalDB(tmp_path / "double-close.db")
    db.close()
    # Second close on a cleared handle is a no-op.
    db.close()
    # Forcibly poke a stale handle to prove resilience.
    dead = sqlite3.connect(str(tmp_path / "double-close.db"))
    dead.close()
    db._conn = dead  # type: ignore[assignment]
    db.close()  # Must not raise.
