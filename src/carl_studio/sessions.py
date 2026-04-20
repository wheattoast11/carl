"""Session persistence for CARLAgent — v0.12 extraction from chat_agent.

:class:`SessionStore` persists and resumes CARLAgent sessions under
``~/.carl/sessions/``. Corrupt files (bad JSON, schema mismatch,
validation failure) are moved to a quarantine subdirectory rather
than deleted, so a user can recover state if needed.

Extracted from ``chat_agent.py`` in v0.12.0 as part of the god-class
decomposition initiated in v0.7.0. The public shape (``SessionStore``
class + ``_SESSION_SCHEMA_VERSION`` constant) is preserved; callers
that imported from ``carl_studio.chat_agent`` continue to work via a
compatibility re-export.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError


logger = logging.getLogger("carl_studio.sessions")


# Schema versioning — sessions carry this in their persisted JSON so
# loaders can reject mismatched formats rather than crash on Pydantic.
_SESSION_SCHEMA_VERSION = "1"

_SESSIONS_DIR = Path.home() / ".carl" / "sessions"
_QUARANTINE_SUBDIR = ".quarantine"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quarantine_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class SessionStore:
    """Save and resume CARLAgent sessions at ``~/.carl/sessions/``.

    Corrupted session files (bad JSON, schema mismatch, or validation error)
    are moved to ``~/.carl/sessions/.quarantine/<id>-<timestamp>.json`` and
    :meth:`load` returns ``None`` so callers fall through to a fresh session.
    """

    def __init__(self, sessions_dir: Path | str | None = None) -> None:
        self._dir = Path(sessions_dir) if sessions_dir else _SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_dir = self._dir / _QUARANTINE_SUBDIR
        # Stashed by _quarantine_path so load_session can surface a visible
        # warning to the user. None means "load did not quarantine".
        self._last_quarantine_dest: Path | None = None

    # -- helpers -----------------------------------------------------------

    def _quarantine_path(self, session_id: str, path: Path) -> Path:
        """Move ``path`` into the quarantine directory. Returns the destination.

        Never raises — quarantine is best-effort. On OS failure we log a
        warning and the caller still sees a None-load result. The last
        quarantine destination is stashed on the instance so
        :meth:`CARLAgent.load_session` can surface it to the UI.
        """
        try:
            self._quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = self._quarantine_dir / f"{session_id}-{_quarantine_stamp()}.json"
            shutil.move(str(path), str(dest))
            logger.warning(
                "Quarantined corrupted session %s -> %s", session_id, dest,
            )
            self._last_quarantine_dest = dest
            return dest
        except OSError as exc:
            logger.warning("Failed to quarantine %s: %s", path, exc)
            self._last_quarantine_dest = path
            return path

    # -- public API --------------------------------------------------------

    def save(self, session_id: str, state: dict[str, Any]) -> Path:
        """Persist session state to JSON."""
        state["updated_at"] = _now_iso()
        if "created_at" not in state:
            state["created_at"] = state["updated_at"]
        state["schema_version"] = _SESSION_SCHEMA_VERSION
        path = self._dir / f"{session_id}.json"
        # Serialize sets in knowledge entries
        knowledge = state.get("knowledge", [])
        serializable_knowledge: list[dict[str, Any]] = []
        for entry in knowledge:
            entry_copy = dict(entry)
            if "words" in entry_copy and isinstance(entry_copy["words"], set):
                entry_copy["words"] = sorted(entry_copy["words"])
            serializable_knowledge.append(entry_copy)
        state["knowledge"] = serializable_knowledge
        path.write_text(json.dumps(state, indent=2, default=str))
        return path

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load session state from JSON.

        Returns ``None`` when the file is missing, malformed, or of an
        unsupported schema version. Corrupted files are moved to the
        quarantine subdirectory; the destination path is stashed on
        :attr:`_last_quarantine_dest` for the caller to surface.
        """
        self._last_quarantine_dest = None
        path = self._dir / f"{session_id}.json"
        if not path.is_file():
            return None

        try:
            raw = path.read_text()
        except OSError as exc:
            logger.warning("Could not read session %s: %s", session_id, exc)
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Session %s has invalid JSON: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None
        except Exception as exc:
            logger.warning("Session %s load failed: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None

        if not isinstance(data, dict):
            logger.warning("Session %s is not a JSON object", session_id)
            self._quarantine_path(session_id, path)
            return None

        schema_version = data.get("schema_version")
        # Unstamped (legacy) sessions default to schema 1 for forward-compat.
        if schema_version is not None and schema_version != _SESSION_SCHEMA_VERSION:
            logger.warning(
                "Session %s has schema_version=%s; expected %s — quarantining",
                session_id, schema_version, _SESSION_SCHEMA_VERSION,
            )
            self._quarantine_path(session_id, path)
            return None

        try:
            # Deserialize word lists back to sets
            for entry in data.get("knowledge", []):
                if isinstance(entry, dict) and "words" in entry and isinstance(entry["words"], list):
                    entry["words"] = set(entry["words"])
        except (TypeError, PydanticValidationError) as exc:
            logger.warning("Session %s knowledge deserialize failed: %s", session_id, exc)
            self._quarantine_path(session_id, path)
            return None

        return data

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List sessions, most recent first. Skips quarantined / corrupt files."""
        sessions: list[dict[str, Any]] = []
        try:
            candidates = sorted(
                self._dir.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            logger.warning("Could not enumerate sessions: %s", exc)
            return []

        for p in candidates:
            # Skip entries inside the quarantine directory. Path.glob("*.json")
            # on the top-level session dir never recurses, but guard anyway.
            try:
                if _QUARANTINE_SUBDIR in p.parts:
                    continue
            except Exception:
                continue
            if len(sessions) >= limit:
                break
            try:
                data = json.loads(p.read_text())
                if not isinstance(data, dict):
                    continue
                sessions.append({
                    "id": p.stem,
                    "title": data.get("title", ""),
                    "model": data.get("model", ""),
                    "turn_count": data.get("turn_count", 0),
                    "total_cost_usd": data.get("total_cost_usd", 0.0),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, OSError):
                # Don't fail the whole list because one file is bad.
                continue
            except Exception:
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if found."""
        path = self._dir / f"{session_id}.json"
        if path.is_file():
            path.unlink()
            return True
        return False


__all__ = [
    "SessionStore",
]
