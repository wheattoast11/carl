"""SkillRunner — resolves, executes, records, and awards CARL skills.

Manages its own ``skill_runs`` table via ``_ensure_schema()`` on init.
Does NOT touch LocalDB's schema — appends to the same sqlite file but
manages an isolated table.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from carl_studio.db import DB_PATH
from carl_studio.skills.base import BaseSkill, SkillResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT NOT NULL,
    run_id TEXT,
    inputs TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '{}',
    badge_earned INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_skill_runs_name ON skill_runs (skill_name);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SkillRegistry:
    """Name -> BaseSkill mapping. Thin wrapper, no persistence."""

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[BaseSkill]:
        return list(self._skills.values())


class SkillRunner:
    """Execute skills, record results in SQLite, award badges via CampConsole."""

    def __init__(self, db_path: Path | None = None) -> None:
        path = db_path or DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._registry = SkillRegistry()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ─── Registry delegation ──────────────────────────────────────────────

    def register(self, skill: BaseSkill) -> None:
        self._registry.register(skill)

    def get(self, name: str) -> BaseSkill | None:
        return self._registry.get(name)

    def list_skills(self) -> list[BaseSkill]:
        return self._registry.list_skills()

    # ─── Execution ────────────────────────────────────────────────────────

    def run(
        self,
        name: str,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Execute a skill by name, record result, award badge if earned.

        Returns SkillResult. If skill not found, returns a failure SkillResult
        rather than raising so callers get a consistent return type.
        """
        skill = self._registry.get(name)
        if skill is None:
            result = SkillResult(
                skill_name=name,
                success=False,
                message=f"Skill '{name}' not found. Register it first.",
            )
            self._record(name, run_id, kwargs, result)
            return result

        started = _now()
        try:
            result = skill.execute(**kwargs)
        except Exception as exc:
            result = SkillResult(
                skill_name=name,
                success=False,
                message=f"Skill execution raised: {exc}",
            )

        completed = _now()
        self._record(name, run_id, kwargs, result, started, completed)

        if result.badge_earned:
            try:
                from carl_studio.console import get_console
                get_console().badge_award(skill.badge, result.message)
            except Exception:
                pass  # Badge award is cosmetic — never block on it

        return result

    def _record(
        self,
        skill_name: str,
        run_id: str | None,
        inputs: dict[str, Any],
        result: SkillResult,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        now = _now()
        try:
            self._conn.execute(
                """INSERT INTO skill_runs
                   (skill_name, run_id, inputs, result, badge_earned, success, message,
                    started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    skill_name,
                    run_id,
                    json.dumps(inputs, default=str),
                    result.model_dump_json(),
                    int(result.badge_earned),
                    int(result.success),
                    result.message,
                    started_at or now,
                    completed_at or now,
                ),
            )
            self._conn.commit()
        except Exception:
            # DB recording is best-effort; never break skill execution
            pass

    # ─── History ──────────────────────────────────────────────────────────

    def get_history(
        self,
        skill_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent skill runs, newest first."""
        if skill_name:
            rows = self._conn.execute(
                """SELECT * FROM skill_runs
                   WHERE skill_name = ?
                   ORDER BY id DESC LIMIT ?""",
                (skill_name, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM skill_runs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def badges_earned(self) -> list[str]:
        """Return distinct skill names where at least one run earned a badge."""
        rows = self._conn.execute(
            "SELECT DISTINCT skill_name FROM skill_runs WHERE badge_earned = 1"
            " ORDER BY skill_name"
        ).fetchall()
        return [r["skill_name"] for r in rows]
