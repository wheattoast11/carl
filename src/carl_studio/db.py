"""SQLite persistence layer for carl-studio CLI.

Manages ~/.carl/carl.db with WAL mode for concurrent reads.
Zero external dependencies — uses Python stdlib sqlite3.

Usage:
    from platform.cli.db import LocalDB

    db = LocalDB()                    # Opens/creates ~/.carl/carl.db
    db.insert_run(run_dict)           # Store a run locally
    db.get_unsynced('runs')           # Get entities pending push
    db.mark_synced(run_id, remote_id) # Mark as synced after push
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, TypedDict
from uuid import uuid4

from carl_core import now_iso


class WalCheckpointResult(TypedDict):
    """Shape of the ``wal_checkpoint`` field returned by :meth:`LocalDB.maintenance`.

    Each slot corresponds to the equivalent column in the tuple returned by
    ``PRAGMA wal_checkpoint`` — ``None`` values mean SQLite returned no row
    (WAL not active, in-memory DB, etc.) or a non-integer where an int was
    expected.
    """

    busy: int | None
    log_pages: int | None
    checkpointed: int | None


class MaintenanceResult(TypedDict):
    """Return type of :meth:`LocalDB.maintenance`."""

    notes_deleted: int
    wal_checkpoint: WalCheckpointResult
    vacuumed: bool

CARL_DIR = Path.home() / ".carl"
DB_PATH = CARL_DIR / "carl.db"
_RUN_JSON_COLUMNS: frozenset[str] = frozenset({"config", "result"})

#: Default number of days to retain ``archived`` sticky-note rows before the
#: maintenance sweep prunes them. Hoisted here (rather than in
#: ``carl_studio.sticky``) because this module does not import ``sticky`` —
#: the reverse *is* true, so ``sticky`` can (and does) re-export this
#: symbol as ``DEFAULT_RETENTION_DAYS``. Every call site — the CLI, the
#: heartbeat daemon, and ``StickyQueue.maintenance`` — pulls from the same
#: constant so drift between surfaces is impossible (R2-005).
DEFAULT_RETENTION_DAYS: int = 30

# SQLite schema — executed on first connect
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS auth (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    hardware TEXT,
    config TEXT NOT NULL,
    result TEXT,
    started_at TEXT,
    completed_at TEXT,
    synced INTEGER NOT NULL DEFAULT 0,
    remote_id TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    content_hash TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics (run_id, step);

CREATE TABLE IF NOT EXISTS gates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    passed INTEGER NOT NULL,
    criteria TEXT NOT NULL,
    results TEXT NOT NULL,
    decided_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL CHECK (operation IN ('push', 'pull')),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'syncing', 'synced', 'failed')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_sync_pending ON sync_queue (status) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS contracts (
    id TEXT PRIMARY KEY,
    parties TEXT NOT NULL,
    terms_hash TEXT NOT NULL,
    terms_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    signed_at TEXT,
    witness_hash TEXT,
    chain TEXT,
    envelope TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS sticky_notes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','processing','done','archived')),
    priority INTEGER NOT NULL DEFAULT 5,
    session_id TEXT,
    jit_context TEXT,
    result TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sticky_status ON sticky_notes (status, priority DESC, created_at);
"""


def content_hash(entity: dict, exclude: set[str] | None = None) -> str:
    """Deterministic content-addressable hash for sync comparison."""
    exclude = exclude or {
        "version",
        "content_hash",
        "created_at",
        "updated_at",
        "synced",
        "remote_id",
    }
    canonical = {k: v for k, v in sorted(entity.items()) if k not in exclude}
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _encode_run_value(key: str, value: object) -> object:
    """Serialize structured run fields for SQLite storage."""
    if key in _RUN_JSON_COLUMNS and value is not None and not isinstance(value, str):
        return json.dumps(value)
    if key == "synced":
        return int(bool(value))
    return value


def _decode_run_row(row: dict) -> dict:
    """Decode structured run fields after reading from SQLite."""
    decoded = dict(row)
    for key in _RUN_JSON_COLUMNS:
        value = decoded.get(key)
        if isinstance(value, str):
            try:
                decoded[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
    if "synced" in decoded:
        decoded["synced"] = bool(decoded["synced"])
    return decoded


class LocalDB:
    """SQLite persistence for carl-studio CLI."""

    def __init__(self, db_path: Path | str | None = None):
        self.path = Path(db_path) if db_path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        # Guards the shared single-connection hot path (REV-002).
        # Acquired inside ``_connect()`` so every caller serializes on the
        # same sqlite handle — sqlite3 Connection objects are *not*
        # thread-safe when shared across threads.
        self._lock = threading.Lock()
        try:
            self._init_schema()
        except Exception:
            # REV-008: If schema bootstrap blows up (corrupted db, disk
            # full, readonly fs, …) we must not leave a half-open
            # connection attached to ``self._conn`` — the next
            # ``LocalDB(path)`` instantiation would inherit a broken
            # connection. Tear it down so subsequent instances can
            # start clean, then re-raise the original failure.
            self.close()
            raise

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Public thread-safe context manager over the shared SQLite handle.

        Use this from modules outside ``carl_studio.db`` (e.g. ``sticky.py``,
        heartbeat loop) so the access is not a private-attribute reach. The
        underlying lock + commit/rollback semantics are identical to the
        internal ``_connect``.
        """
        with self._connect() as conn:
            yield conn

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        # Serialize access across threads — sqlite3.Connection is not
        # thread-safe by default, and we deliberately reuse a single
        # connection per LocalDB instance. The lock is re-entrant-safe
        # only for single-threaded reentry via the same thread because
        # ``threading.Lock`` is non-reentrant; all current call sites
        # are non-nested, which keeps this correct and simple.
        with self._lock:
            if self._conn is None:
                # ``check_same_thread=False`` is required because the
                # single shared connection may legitimately be reached
                # from multiple threads (heartbeat worker, CLI main,
                # MCP tool coroutines). The ``self._lock`` above
                # serializes every call so sqlite never sees
                # concurrent use — we pay only the safety check, not
                # the actual race.
                self._conn = sqlite3.connect(
                    str(self.path),
                    check_same_thread=False,
                )
                self._conn.row_factory = sqlite3.Row
            try:
                yield self._conn
                # REV-002: commit on clean exit so write-side callers no
                # longer have to remember an explicit ``conn.commit()``.
                # Existing sites that already call ``conn.commit()``
                # remain correct — sqlite treats the trailing commit as
                # a no-op when no transaction is open.
                self._conn.commit()
            except Exception:
                # REV-002: if rollback itself raises (e.g. closed conn
                # during shutdown) swallow it so we preserve the
                # *original* exception for the caller.
                try:
                    self._conn.rollback()
                except Exception:
                    pass
                raise

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                # Closing an already-broken connection must not mask
                # the original fault (REV-008) — best-effort only.
                pass
            self._conn = None

    # ─── Maintenance ─────────────────────────────────────────

    def maintenance(
        self,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        vacuum: bool = False,
    ) -> MaintenanceResult:
        """Run periodic DB maintenance.

        Performs three actions, in order:

        1. **Retention sweep** — deletes ``sticky_notes`` rows with
           ``status='archived'`` whose ``completed_at`` (or ``created_at``
           fallback when ``completed_at`` is ``NULL``) is older than the
           cutoff. Heartbeat load is dominated by this table, so it is the
           primary driver of unbounded growth. ``retention_days=0`` disables
           the sweep.
        2. **WAL checkpoint** — truncates ``carl.db-wal`` back to zero bytes
           by running ``PRAGMA wal_checkpoint(TRUNCATE)``. Without this the
           WAL file grows unboundedly under heartbeat-driven writes; manual
           checkpointing is the SQLite-recommended mitigation.
        3. **VACUUM** — optional, off by default because it takes a full-
           database exclusive lock. Call with ``vacuum=True`` during a
           low-traffic maintenance window or from a user-initiated ``carl db
           maintenance --vacuum`` command.

        Parameters
        ----------
        retention_days
            Maximum age (days) for ``archived`` sticky notes. Rows older
            than the cutoff are deleted. Set to ``0`` to skip the retention
            sweep. Must be non-negative.
        vacuum
            When ``True``, run ``VACUUM`` after the checkpoint to reclaim
            freed pages. Off by default — ``VACUUM`` is expensive and takes
            an exclusive lock, so it should be opt-in.

        Returns
        -------
        dict[str, object]
            ``notes_deleted``: int count of deleted sticky rows.
            ``wal_checkpoint``: dict with ``busy`` / ``log_pages`` /
            ``checkpointed`` from ``PRAGMA wal_checkpoint``. Each is either
            an int (happy path) or ``None`` (SQLite returned no row, which
            only happens when WAL is not active, e.g. memory-backed DBs).
            ``vacuumed``: bool — whether ``VACUUM`` actually ran.

        Raises
        ------
        ValueError
            If ``retention_days`` is negative.
        """
        if retention_days < 0:
            raise ValueError(
                "LocalDB.maintenance: retention_days must be non-negative",
            )

        deleted = 0
        checkpoint_row: tuple[object, ...] | None = None

        with self._connect() as conn:
            if retention_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
                cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
                # ``completed_at`` is the honest bounce-date for archived
                # rows; when it's missing (edge case: archived before
                # completion, or legacy inserts that pre-date the field
                # being populated), fall back to ``created_at`` so old rows
                # still age out rather than surviving forever.
                cursor = conn.execute(
                    "DELETE FROM sticky_notes WHERE status = 'archived' "
                    "AND COALESCE(completed_at, created_at) < ?",
                    (cutoff_str,),
                )
                deleted = int(cursor.rowcount)

            # WAL checkpoint — returns (busy, log_pages, checkpointed).
            # ``TRUNCATE`` is the aggressive mode: it truncates the WAL file
            # to zero bytes after writing all frames back to the main db.
            try:
                checkpoint_row = conn.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)",
                ).fetchone()
            except sqlite3.DatabaseError:
                # WAL may be inactive (e.g. ``:memory:`` DBs or very old
                # SQLite). Degrade gracefully rather than crash the
                # maintenance tick.
                checkpoint_row = None

            conn.commit()

            if vacuum:
                # VACUUM cannot run inside a transaction — we committed
                # above, so we're clear. It takes an exclusive lock for
                # its duration.
                conn.execute("VACUUM")

        def _slot(idx: int) -> int | None:
            if checkpoint_row is None or len(checkpoint_row) <= idx:
                return None
            val = checkpoint_row[idx]
            if isinstance(val, int):
                return val
            return None

        wal: WalCheckpointResult = {
            "busy": _slot(0),
            "log_pages": _slot(1),
            "checkpointed": _slot(2),
        }
        return {
            "notes_deleted": deleted,
            "wal_checkpoint": wal,
            "vacuumed": bool(vacuum),
        }

    # ─── Config ──────────────────────────────────────────────

    def get_config(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
                (key, value, now_iso(), value, now_iso()),
            )
            conn.commit()

    # ─── Auth ────────────────────────────────────────────────

    def get_auth(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM auth WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] and row["expires_at"] < now_iso():
                return None  # expired
            return row["value"]

    def set_auth(self, key: str, value: str, ttl_hours: int = 24) -> None:
        expires = None
        if ttl_hours > 0:
            expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO auth (key, value, expires_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = ?, expires_at = ?",
                (key, value, expires, value, expires),
            )
            conn.commit()

    def clear_auth(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth")
            conn.commit()

    # ─── Runs ────────────────────────────────────────────────

    def insert_run(self, run: dict) -> str:
        run_id = run.get("id", str(uuid4()))
        run["id"] = run_id
        run["content_hash"] = content_hash(run)

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (id, model_id, mode, status, hardware, config,
                   result, started_at, completed_at, synced, remote_id, version, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run["model_id"],
                    run["mode"],
                    run.get("status", "pending"),
                    run.get("hardware"),
                    _encode_run_value("config", run.get("config", {})),
                    _encode_run_value("result", run.get("result")),
                    run.get("started_at"),
                    run.get("completed_at"),
                    _encode_run_value("synced", run.get("synced", 0)),
                    run.get("remote_id"),
                    run.get("version", 1),
                    run["content_hash"],
                ),
            )
            conn.commit()
        return run_id

    def update_run(self, run_id: str, updates: dict) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return

            current = dict(row)
            current = _decode_run_row(current)
            current.update(updates)
            current["version"] = current.get("version", 1) + 1
            current["content_hash"] = content_hash(current)

            serialized_updates = {
                key: _encode_run_value(key, value) for key, value in updates.items()
            }

            set_clauses = ", ".join(f"{k} = ?" for k in serialized_updates)
            set_clauses += ", version = ?, content_hash = ?"
            values = list(serialized_updates.values()) + [
                current["version"],
                current["content_hash"],
            ]

            conn.execute(
                f"UPDATE runs SET {set_clauses} WHERE id = ?",
                values + [run_id],
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return _decode_run_row(dict(row)) if row else None

    def list_runs(self, limit: int = 20, status: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [_decode_run_row(dict(r)) for r in rows]

    # ─── Metrics ─────────────────────────────────────────────

    def insert_metric(self, run_id: str, step: int, data: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, step, data) VALUES (?, ?, ?)",
                (run_id, step, json.dumps(data)),
            )
            conn.commit()

    def insert_metrics_batch(self, run_id: str, metrics: list[tuple[int, dict]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO metrics (run_id, step, data) VALUES (?, ?, ?)",
                [(run_id, step, json.dumps(data)) for step, data in metrics],
            )
            conn.commit()

    def get_metrics(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step, data, created_at FROM metrics WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()
            return [
                {"step": r["step"], "data": json.loads(r["data"]), "created_at": r["created_at"]}
                for r in rows
            ]

    # ─── Gates ───────────────────────────────────────────────

    def insert_gate(self, run_id: str, passed: bool, criteria: dict, results: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO gates (run_id, passed, criteria, results) VALUES (?, ?, ?, ?)",
                (run_id, int(passed), json.dumps(criteria), json.dumps(results)),
            )
            conn.commit()

    def get_gates(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM gates WHERE run_id = ? ORDER BY decided_at",
                (run_id,),
            ).fetchall()
            return [
                {
                    "run_id": r["run_id"],
                    "passed": bool(r["passed"]),
                    "criteria": json.loads(r["criteria"]),
                    "results": json.loads(r["results"]),
                    "decided_at": r["decided_at"],
                }
                for r in rows
            ]

    # ─── Sync ────────────────────────────────────────────────

    _ALLOWED_SYNC_TYPES: frozenset[str] = frozenset({"runs"})

    def get_unsynced(self, entity_type: str) -> list[dict]:
        if entity_type not in self._ALLOWED_SYNC_TYPES:
            raise ValueError(
                f"Unknown sync entity type: {entity_type!r}. "
                f"Allowed: {sorted(self._ALLOWED_SYNC_TYPES)}"
            )
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {entity_type} WHERE synced = 0",  # noqa: S608 — whitelisted above
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_synced(self, entity_id: str, remote_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET synced = 1, remote_id = ? WHERE id = ?",
                (remote_id, entity_id),
            )
            conn.commit()

    def enqueue_sync(self, operation: str, entity_type: str, entity_id: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue (operation, entity_type, entity_id, payload)
                   VALUES (?, ?, ?, ?)""",
                (operation, entity_type, entity_id, json.dumps(payload)),
            )
            conn.commit()

    def get_pending_sync(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM sync_queue WHERE status = 'pending'
                   ORDER BY created_at LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_sync_status(self, sync_id: int, status: str, error: str | None = None) -> None:
        with self._connect() as conn:
            if error:
                conn.execute(
                    "UPDATE sync_queue SET status = ?, error = ?, retry_count = retry_count + 1 WHERE id = ?",
                    (status, error, sync_id),
                )
            else:
                conn.execute(
                    "UPDATE sync_queue SET status = ? WHERE id = ?",
                    (status, sync_id),
                )
            conn.commit()

    # ─── Contracts ──────────────────────────────────────────────

    def insert_contract(self, contract: dict[str, str | None]) -> str:
        """Insert a contract and return its id."""
        cid = str(contract.get("id") or uuid4())
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO contracts
                   (id, parties, terms_hash, terms_url, status,
                    signed_at, witness_hash, chain, envelope)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cid,
                    contract.get("parties", "[]"),
                    contract.get("terms_hash", ""),
                    contract.get("terms_url", ""),
                    contract.get("status", "draft"),
                    contract.get("signed_at"),
                    contract.get("witness_hash"),
                    contract.get("chain"),
                    contract.get("envelope"),
                ),
            )
            conn.commit()
        return cid

    def update_contract(self, contract_id: str, updates: dict[str, str | None]) -> None:
        """Update specific fields on a contract."""
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [contract_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE contracts SET {set_clause} WHERE id = ?",  # noqa: S608
                values,
            )
            conn.commit()

    def get_contract(self, contract_id: str) -> dict[str, str | None] | None:
        """Get a single contract by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contracts WHERE id = ?", (contract_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_contracts(self, limit: int = 20) -> list[dict[str, str | None]]:
        """List contracts ordered by creation date."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM contracts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
