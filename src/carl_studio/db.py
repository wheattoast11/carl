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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator
from uuid import uuid4

CARL_DIR = Path.home() / ".carl"
DB_PATH = CARL_DIR / "carl.db"

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
"""


def content_hash(entity: dict, exclude: set[str] | None = None) -> str:
    """Deterministic content-addressable hash for sync comparison."""
    exclude = exclude or {"version", "content_hash", "created_at", "updated_at",
                          "synced", "remote_id"}
    canonical = {k: v for k, v in sorted(entity.items()) if k not in exclude}
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LocalDB:
    """SQLite persistence for carl-studio CLI."""

    def __init__(self, db_path: Path | str | None = None):
        self.path = Path(db_path) if db_path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Config ──────────────────────────────────────────────

    def get_config(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_config(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
                (key, value, _now(), value, _now()),
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
            if row["expires_at"] and row["expires_at"] < _now():
                return None  # expired
            return row["value"]

    def set_auth(self, key: str, value: str, ttl_hours: int = 24) -> None:
        expires = datetime.now(timezone.utc).replace(
            hour=datetime.now(timezone.utc).hour + ttl_hours
        ).strftime("%Y-%m-%dT%H:%M:%SZ") if ttl_hours > 0 else None

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
                   result, started_at, completed_at, version, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run["model_id"],
                    run["mode"],
                    run.get("status", "pending"),
                    run.get("hardware"),
                    json.dumps(run.get("config", {})),
                    json.dumps(run.get("result")) if run.get("result") else None,
                    run.get("started_at"),
                    run.get("completed_at"),
                    run.get("version", 1),
                    run["content_hash"],
                ),
            )
            conn.commit()
        return run_id

    def update_run(self, run_id: str, updates: dict) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                return

            current = dict(row)
            current.update(updates)
            current["version"] = current.get("version", 1) + 1
            current["content_hash"] = content_hash(current)

            set_clauses = ", ".join(f"{k} = ?" for k in updates)
            set_clauses += ", version = ?, content_hash = ?"
            values = list(updates.values()) + [current["version"], current["content_hash"]]

            conn.execute(
                f"UPDATE runs SET {set_clauses} WHERE id = ?",
                values + [run_id],
            )
            conn.commit()

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

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
            return [dict(r) for r in rows]

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

    def get_unsynced(self, entity_type: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {entity_type} WHERE synced = 0",
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_synced(self, entity_id: str, remote_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET synced = 1, remote_id = ? WHERE id = ?",
                (remote_id, entity_id),
            )
            conn.commit()

    def enqueue_sync(self, operation: str, entity_type: str,
                     entity_id: str, payload: dict) -> None:
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

    def update_sync_status(self, sync_id: int, status: str,
                           error: str | None = None) -> None:
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
