"""Carlito/Carlitas -- small specialized agents trained and spawned from CARL.

A carlito is what you get when a CurriculumTrack graduates. It packages a
trained model lineage with earned skills, a domain persona, and deployment
readiness into a named, registry-tracked entity.

Carlitos can later be published to the marketplace for multiplayer composition.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

from pydantic import BaseModel, Field

_CARL_DIR = Path.home() / ".carl"
_DB_PATH = _CARL_DIR / "carl.db"

_CARLITO_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS carlitos (
    name TEXT PRIMARY KEY,
    parent_model TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    persona TEXT NOT NULL DEFAULT '',
    skills TEXT NOT NULL DEFAULT '[]',
    environment_spec TEXT NOT NULL DEFAULT '{}',
    training_config_snapshot TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'incubating',
    curriculum_model_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    metadata TEXT NOT NULL DEFAULT '{}'
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CarlitoStatus(str, Enum):
    """Lifecycle status of a carlito."""

    INCUBATING = "incubating"
    TRAINING = "training"
    GRADUATED = "graduated"
    DEPLOYED = "deployed"
    DORMANT = "dormant"


class CarlitoSpec(BaseModel):
    """Specification for a carlito agent."""

    name: str
    parent_model: str
    domain: str = ""
    persona: str = ""
    skills: list[str] = Field(default_factory=list)
    environment_spec: dict[str, Any] = Field(default_factory=dict)
    training_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    status: CarlitoStatus = CarlitoStatus.INCUBATING
    curriculum_model_id: str = ""
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CarlitoRegistry:
    """SQLite-backed registry for carlitos."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CARLITO_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        pass  # connections are per-call

    def save(self, spec: CarlitoSpec) -> None:
        """Upsert a carlito spec by name."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO carlitos
                   (name, parent_model, domain, persona, skills,
                    environment_spec, training_config_snapshot, status,
                    curriculum_model_id, created_at, updated_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                    parent_model=excluded.parent_model,
                    domain=excluded.domain,
                    persona=excluded.persona,
                    skills=excluded.skills,
                    environment_spec=excluded.environment_spec,
                    training_config_snapshot=excluded.training_config_snapshot,
                    status=excluded.status,
                    curriculum_model_id=excluded.curriculum_model_id,
                    updated_at=excluded.updated_at,
                    metadata=excluded.metadata
                """,
                (
                    spec.name,
                    spec.parent_model,
                    spec.domain,
                    spec.persona,
                    json.dumps(spec.skills),
                    json.dumps(spec.environment_spec),
                    json.dumps(spec.training_config_snapshot),
                    spec.status.value,
                    spec.curriculum_model_id,
                    spec.created_at,
                    spec.updated_at,
                    json.dumps(spec.metadata),
                ),
            )

    def load(self, name: str) -> CarlitoSpec | None:
        """Load a carlito by name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM carlitos WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_spec(row)

    def list_all(self, status: CarlitoStatus | None = None) -> list[CarlitoSpec]:
        """List all carlitos, optionally filtered by status."""
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM carlitos WHERE status = ? ORDER BY updated_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM carlitos ORDER BY updated_at DESC"
                ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def retire(self, name: str) -> bool:
        """Set a carlito's status to DORMANT. Returns True if found."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE carlitos SET status = ?, updated_at = ? WHERE name = ?",
                (CarlitoStatus.DORMANT.value, _now_iso(), name),
            )
        return cursor.rowcount > 0

    def delete(self, name: str) -> bool:
        """Permanently delete a carlito. Returns True if found."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM carlitos WHERE name = ?", (name,))
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_spec(row: sqlite3.Row) -> CarlitoSpec:
        return CarlitoSpec(
            name=row["name"],
            parent_model=row["parent_model"],
            domain=row["domain"],
            persona=row["persona"],
            skills=json.loads(row["skills"]),
            environment_spec=json.loads(row["environment_spec"]),
            training_config_snapshot=json.loads(row["training_config_snapshot"]),
            status=CarlitoStatus(row["status"]),
            curriculum_model_id=row["curriculum_model_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )


def _domain_capabilities(domain: str) -> list[str]:
    """Map domain to default capability list for agent card."""
    base = ["train", "eval", "observe"]
    domain_caps: dict[str, list[str]] = {
        "coding": ["push", "bundle", "bench"],
        "math": ["bench", "align"],
        "research": ["learn", "align", "bench"],
    }
    return base + domain_caps.get(domain.lower(), [])


class CarlitoSpawner:
    """Produces a deployable agent card from a graduated CurriculumTrack + CarlitoSpec."""

    def __init__(self, registry: CarlitoRegistry | None = None) -> None:
        self._registry = registry

    def spawn(self, spec: CarlitoSpec, track: Any) -> Any:
        """Produce an agent card from a graduated track.

        Raises ValueError if track is not in GRADUATED or DEPLOYED phase.
        """
        from carl_studio.curriculum import CurriculumPhase

        if track.phase not in (CurriculumPhase.GRADUATED, CurriculumPhase.DEPLOYED):
            raise ValueError(
                f"Cannot spawn carlito '{spec.name}': curriculum track "
                f"'{track.model_id}' is in phase '{track.phase.value}', "
                f"must be 'graduated' or 'deployed'."
            )

        from carl_studio.a2a.agent_card import CARLAgentCard

        card = CARLAgentCard(
            name=spec.name,
            tier="free",
            capabilities=_domain_capabilities(spec.domain),
            skills=list(spec.skills),
            metadata={
                "parent_model": spec.parent_model,
                "domain": spec.domain,
                "persona": spec.persona,
                "curriculum_model_id": spec.curriculum_model_id,
                "curriculum_version": getattr(track, "version", 1),
            },
        )

        spec_updated = spec.model_copy(
            update={
                "status": CarlitoStatus.DEPLOYED,
                "updated_at": _now_iso(),
            }
        )
        if self._registry is not None:
            self._registry.save(spec_updated)

        return card

    @staticmethod
    def from_graduated_track(
        name: str,
        track: Any,
        domain: str = "",
        persona: str = "",
        skills: list[str] | None = None,
    ) -> CarlitoSpec:
        """Create a CarlitoSpec from a graduated CurriculumTrack."""
        from carl_studio.curriculum import CurriculumPhase

        status = (
            CarlitoStatus.GRADUATED
            if track.phase == CurriculumPhase.GRADUATED
            else CarlitoStatus.INCUBATING
        )

        return CarlitoSpec(
            name=name,
            parent_model=track.model_id,
            domain=domain,
            persona=persona,
            skills=skills or [],
            status=status,
            curriculum_model_id=track.model_id,
        )
