"""Traffic logger for shadow replay.

Logs production inference requests to SQLite for later replay
through the shadow adapter. Same offline-first pattern as
carl-studio's LocalDB.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TrafficEntry:
    """One logged inference request."""

    id: int
    timestamp: str
    request: dict
    response: dict | None
    latency_ms: float


class TrafficLogger:
    """Log production inference requests for shadow replay.

    Uses SQLite with WAL mode for concurrent read/write safety.
    """

    def __init__(self, db_path: str = ".carl/traffic.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS traffic_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT,
            latency_ms REAL
        )"""
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON traffic_log(timestamp)"
        )
        self.conn.commit()

    def log(
        self,
        request: dict,
        response: dict | None = None,
        latency_ms: float = 0.0,
    ) -> int:
        """Log an inference request.

        Args:
            request: The inference request payload.
            response: The inference response payload, if available.
            latency_ms: Request latency in milliseconds.

        Returns:
            The auto-incremented entry ID.

        Raises:
            TypeError: If request is not a dict.
        """
        if not isinstance(request, dict):
            raise TypeError(f"request must be a dict, got {type(request).__name__}")

        cursor = self.conn.execute(
            "INSERT INTO traffic_log (timestamp, request_json, response_json, latency_ms) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(request),
                json.dumps(response) if response else None,
                latency_ms,
            ),
        )
        self.conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT failed: no lastrowid returned")
        return row_id

    def window(self, start: datetime, end: datetime) -> list[TrafficEntry]:
        """Retrieve traffic entries in a time window for shadow replay.

        Args:
            start: Window start (inclusive).
            end: Window end (inclusive).

        Returns:
            List of TrafficEntry objects ordered by timestamp.
        """
        rows = self.conn.execute(
            "SELECT id, timestamp, request_json, response_json, latency_ms "
            "FROM traffic_log WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [
            TrafficEntry(
                id=row[0],
                timestamp=row[1],
                request=json.loads(row[2]),
                response=json.loads(row[3]) if row[3] else None,
                latency_ms=row[4],
            )
            for row in rows
        ]

    def count(self) -> int:
        """Total entries in the log."""
        return self.conn.execute("SELECT COUNT(*) FROM traffic_log").fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except (TypeError, AttributeError):
            pass
