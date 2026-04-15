"""CARL curriculum tracking -- the academic career path of a carlito.

Tracks which phase a model (carlito) is in, gates advancement to the next phase,
and records milestones. The curriculum is the structural spine of the training loop.

Phases (Summer Camp -> Academic):
  ENROLLED   -> Undergrad  -- model selected, not yet trained
  DRILLING   -> PhD track  -- SFT or GRPO training in progress
  EVALUATED  -> Qualifying -- eval gate pending or in progress
  GRADUATED  -> Post-doc   -- gate PASS, ready for deployment
  DEPLOYED   -> Professor  -- pushed to Hub, serving inference
  TTT_ACTIVE -> Research   -- live TTT (SLOT/LoRA) active on inference

Each transition requires a gate:
  ENROLLED -> DRILLING:   carl.yaml configured + dataset validated
  DRILLING -> EVALUATED:  training job completed (any status)
  EVALUATED -> GRADUATED: gate PASS (task_completion >= 0.80, format >= 0.95)
  GRADUATED -> DEPLOYED:  carl push succeeds (model on Hub)
  DEPLOYED -> TTT_ACTIVE: carl infer --live started with --ttt flag
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


class CurriculumPhase(str, Enum):
    """Phase in the CARL training curriculum."""

    ENROLLED = "enrolled"
    DRILLING = "drilling"
    EVALUATED = "evaluated"
    GRADUATED = "graduated"
    DEPLOYED = "deployed"
    TTT_ACTIVE = "ttt_active"


# Valid transitions -- FSM must be closed.
# Every phase appears as a key; every target appears as a CurriculumPhase value.
_TRANSITIONS: dict[CurriculumPhase, set[CurriculumPhase]] = {
    CurriculumPhase.ENROLLED: {CurriculumPhase.DRILLING},
    CurriculumPhase.DRILLING: {CurriculumPhase.EVALUATED},
    CurriculumPhase.EVALUATED: {CurriculumPhase.GRADUATED, CurriculumPhase.DRILLING},
    CurriculumPhase.GRADUATED: {CurriculumPhase.DEPLOYED},
    CurriculumPhase.DEPLOYED: {CurriculumPhase.TTT_ACTIVE, CurriculumPhase.DRILLING},
    CurriculumPhase.TTT_ACTIVE: {CurriculumPhase.DRILLING},
}


def verify_fsm_closure() -> bool:
    """Verify the FSM is closed: every phase has transitions, every target is valid.

    Returns True if closed. Raises ValueError with details if not.
    """
    all_phases = set(CurriculumPhase)

    # Every phase must have at least one outgoing transition
    missing_sources = all_phases - set(_TRANSITIONS.keys())
    if missing_sources:
        raise ValueError(f"Phases without outgoing transitions: {missing_sources}")

    # Every target must be a valid phase
    all_targets: set[CurriculumPhase] = set()
    for targets in _TRANSITIONS.values():
        all_targets.update(targets)
    invalid_targets = all_targets - all_phases
    if invalid_targets:
        raise ValueError(f"Transition targets not in enum: {invalid_targets}")

    return True


class Milestone(BaseModel):
    """A recorded curriculum event."""

    phase: CurriculumPhase
    event: str  # e.g. "gate_pass", "training_started", "pushed_to_hub"
    detail: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CurriculumTrack(BaseModel):
    """The curriculum state for a single carlito (model lineage)."""

    model_id: str  # e.g. "il-terminals-carl-omni9b-v12"
    phase: CurriculumPhase = CurriculumPhase.ENROLLED
    milestones: list[Milestone] = Field(default_factory=list)
    version: int = 1  # increments each full cycle

    def can_advance(self, to: CurriculumPhase) -> bool:
        """Check if transition is valid per FSM."""
        return to in _TRANSITIONS.get(self.phase, set())

    def advance(
        self,
        to: CurriculumPhase,
        event: str = "",
        detail: str = "",
    ) -> CurriculumTrack:
        """Advance to a new phase. Raises ValueError if transition is invalid."""
        if not self.can_advance(to):
            valid = _TRANSITIONS.get(self.phase, set())
            raise ValueError(
                f"Cannot transition from {self.phase.value} to {to.value}. "
                f"Valid: {sorted(v.value for v in valid)}"
            )
        ms = Milestone(
            phase=to,
            event=event or f"advanced_to_{to.value}",
            detail=detail,
        )
        new_milestones = list(self.milestones) + [ms]
        new_version = self.version + 1 if to == CurriculumPhase.ENROLLED else self.version
        return self.model_copy(
            update={
                "phase": to,
                "milestones": new_milestones,
                "version": new_version,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return a display-friendly summary."""
        return {
            "model_id": self.model_id,
            "phase": self.phase.value,
            "version": self.version,
            "milestones": len(self.milestones),
            "last_event": self.milestones[-1].event if self.milestones else "none",
        }


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_CARL_DIR = Path.home() / ".carl"
_DB_PATH = _CARL_DIR / "carl.db"

_CURRICULUM_SCHEMA = """
CREATE TABLE IF NOT EXISTS curriculum (
    model_id TEXT PRIMARY KEY,
    phase TEXT NOT NULL DEFAULT 'enrolled',
    version INTEGER NOT NULL DEFAULT 1,
    milestones TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


class CurriculumStore:
    """SQLite persistence for curriculum tracks.

    Uses ~/.carl/carl.db (same file as LocalDB, own table via _ensure_schema).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path else _DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_CURRICULUM_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
        try:
            yield self._conn
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def save(self, track: CurriculumTrack) -> None:
        """Persist a curriculum track (upsert)."""
        milestones_json = json.dumps(
            [m.model_dump(mode="json") for m in track.milestones]
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO curriculum (model_id, phase, version, milestones, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET "
                "phase = excluded.phase, version = excluded.version, "
                "milestones = excluded.milestones, updated_at = excluded.updated_at",
                (track.model_id, track.phase.value, track.version, milestones_json, now),
            )
            conn.commit()

    def load(self, model_id: str) -> CurriculumTrack | None:
        """Load a curriculum track by model_id. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_id, phase, version, milestones FROM curriculum WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_track(row)

    def list_tracks(self) -> list[CurriculumTrack]:
        """List all curriculum tracks, ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_id, phase, version, milestones FROM curriculum ORDER BY updated_at DESC"
            ).fetchall()
            return [self._row_to_track(r) for r in rows]

    def current(self) -> CurriculumTrack | None:
        """Get the most recently updated track."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_id, phase, version, milestones FROM curriculum ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return self._row_to_track(row)

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> CurriculumTrack:
        """Convert a database row to a CurriculumTrack."""
        milestones_raw: list[dict[str, Any]] = json.loads(row["milestones"])
        milestones = [Milestone(**m) for m in milestones_raw]
        return CurriculumTrack(
            model_id=row["model_id"],
            phase=CurriculumPhase(row["phase"]),
            version=row["version"],
            milestones=milestones,
        )
