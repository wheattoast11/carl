"""Scheduler for autonomous agent actions. SQLite-backed persistence."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Schedule:
    """A scheduled agent action."""

    id: int
    cron_expr: str        # "daily", "hourly", "every_6h", or "HH:MM" for daily at time
    action: str           # "observe" | "train" | "eval" | "shadow"
    config: dict          # Action-specific config
    enabled: bool = True
    last_fired: str | None = None


class Scheduler:
    """Manages scheduled agent actions. Uses SQLite for persistence."""

    def __init__(self, db_path: str = ".carl/scheduler.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute("""CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cron_expr TEXT NOT NULL,
            action TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            enabled INTEGER NOT NULL DEFAULT 1,
            last_fired TEXT
        )""")
        self.conn.commit()

    def add(self, cron_expr: str, action: str, config: dict | None = None) -> int:
        """Add a new schedule. Returns schedule ID."""
        cursor = self.conn.execute(
            "INSERT INTO schedules (cron_expr, action, config_json) VALUES (?, ?, ?)",
            (cron_expr, action, json.dumps(config or {})),
        )
        self.conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT failed: no lastrowid returned")
        return row_id

    def remove(self, schedule_id: int) -> bool:
        """Remove a schedule. Returns True if found."""
        cursor = self.conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def enable(self, schedule_id: int) -> bool:
        """Enable a schedule. Returns True if found."""
        cursor = self.conn.execute(
            "UPDATE schedules SET enabled = 1 WHERE id = ?", (schedule_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def disable(self, schedule_id: int) -> bool:
        """Disable a schedule. Returns True if found."""
        cursor = self.conn.execute(
            "UPDATE schedules SET enabled = 0 WHERE id = ?", (schedule_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def list_all(self) -> list[Schedule]:
        """List all schedules."""
        rows = self.conn.execute(
            "SELECT id, cron_expr, action, config_json, enabled, last_fired "
            "FROM schedules"
        ).fetchall()
        return [
            Schedule(
                id=r[0],
                cron_expr=r[1],
                action=r[2],
                config=json.loads(r[3]),
                enabled=bool(r[4]),
                last_fired=r[5],
            )
            for r in rows
        ]

    def check_due(self) -> Schedule | None:
        """Return the first enabled schedule that is due to fire, or None.

        Simplified scheduling:
        - "hourly": fires if last_fired is None or > 1 hour ago
        - "every_6h": fires if last_fired is None or > 6 hours ago
        - "daily": fires if last_fired is None or > 24 hours ago
        - "HH:MM": fires once daily at that UTC time
        """
        now = datetime.now(timezone.utc)
        rows = self.conn.execute(
            "SELECT id, cron_expr, action, config_json, enabled, last_fired "
            "FROM schedules WHERE enabled = 1"
        ).fetchall()
        enabled_schedules = [
            Schedule(id=r[0], cron_expr=r[1], action=r[2],
                     config=json.loads(r[3]), enabled=bool(r[4]), last_fired=r[5])
            for r in rows
        ]
        for schedule in enabled_schedules:
            if self._is_due(schedule, now):
                self.conn.execute(
                    "UPDATE schedules SET last_fired = ? WHERE id = ?",
                    (now.isoformat(), schedule.id),
                )
                self.conn.commit()
                schedule.last_fired = now.isoformat()
                return schedule
        return None

    def _is_due(self, schedule: Schedule, now: datetime) -> bool:
        """Check if a schedule should fire now."""
        if schedule.last_fired is None:
            return True  # Never fired — fire immediately

        last = datetime.fromisoformat(schedule.last_fired)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed_hours = (now - last).total_seconds() / 3600

        if schedule.cron_expr == "hourly":
            return elapsed_hours >= 1.0
        elif schedule.cron_expr == "every_6h":
            return elapsed_hours >= 6.0
        elif schedule.cron_expr == "daily":
            return elapsed_hours >= 24.0
        elif ":" in schedule.cron_expr:
            # "HH:MM" format — fire once daily at that UTC time
            try:
                hour, minute = map(int, schedule.cron_expr.split(":"))
                if now.hour == hour and now.minute >= minute and elapsed_hours >= 23.0:
                    return True
            except ValueError:
                pass
        return False

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.conn.close()
        except (TypeError, AttributeError):
            pass
