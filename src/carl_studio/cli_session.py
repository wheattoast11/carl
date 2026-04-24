"""CliSession — v0.18 Track D per-project CLI session primitive.

A :class:`CliSession` is a durable, project-local CLI-level session record:
it groups runs, events, and intents across a user's interactive work on a
specific project. Sessions persist as JSON under
``<project_root>/.carl/sessions/<id>.json`` and the id of the currently-
active session is tracked in ``<project_root>/.carl/sessions/current.txt``.

Naming — deliberately ``CliSession`` (NOT ``Session``) because
``carl_studio.session.Session`` is the v0.17 handle-runtime bundle
(chain + vaults + toolkits). These two concerns are orthogonal:

  * ``Session`` (handle-runtime)   — tool execution surface, ephemeral.
  * ``CliSession`` (this module)   — durable CLI-level marker, persisted.

Both may coexist: a single process can wrap its chain in a handle-runtime
``Session`` while simultaneously recording its lifecycle in a
``CliSession``.

Wire alignment — the JSON schema maps 1:1 to carl.camp migration 025's
``sessions`` table (``id``, ``intent``, ``metadata``, ``started_at``,
``completed_at``, ``status``) so a future sync round-trip is shape-
compatible. The sync path is gated on
``CARL_CAMP_SESSIONS_SYNC=1`` and is a deliberate stub in v0.18
(platform route pending per ``docs/platform-parity-reply-2026-04-22.md`` Q4).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from carl_core.errors import ValidationError


__all__ = [
    "CliSession",
    "SessionStatus",
    "SESSION_SCHEMA_VERSION",
]


logger = logging.getLogger("carl_studio.cli_session")


SESSION_SCHEMA_VERSION = 1

SessionStatus = Literal["active", "completed", "abandoned"]

_VALID_STATUSES: frozenset[str] = frozenset({"active", "completed", "abandoned"})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sessions_dir(project_root: Path) -> Path:
    """Resolve ``<project_root>/.carl/sessions``; never creates."""
    return project_root / ".carl" / "sessions"


def _session_file(project_root: Path, session_id: str) -> Path:
    return _sessions_dir(project_root) / f"{session_id}.json"


def _current_file(project_root: Path) -> Path:
    return _sessions_dir(project_root) / "current.txt"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    """ISO-8601 with explicit UTC 'Z' suffix if naive is somehow passed."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CliSession(BaseModel):
    """Frozen, persisted per-project CLI session record.

    Fields mirror carl.camp migration 025's ``sessions`` table columns.
    The ``metadata`` dict is free-form JSONB on the platform side; treat
    it as opaque here and serialize as-is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="uuid4 session identifier")
    project_root: Path = Field(..., description="Absolute path to the project this session belongs to.")
    started_at: datetime = Field(..., description="Session start time (UTC).")
    intent: str | None = Field(
        default=None,
        description="Optional human-authored intent string (maps to migration 025 'intent' column).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form JSONB payload. Opaque to the CLI.",
    )
    status: SessionStatus = Field(
        default="active",
        description="active | completed | abandoned.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Set when status transitions out of 'active'.",
    )
    schema_version: int = Field(
        default=SESSION_SCHEMA_VERSION,
        description="Persisted-shape version. Bump if the on-disk JSON shape changes.",
    )

    @field_serializer("project_root")
    def _serialize_project_root(self, value: Path) -> str:
        return str(value)

    @field_serializer("started_at")
    def _serialize_started_at(self, value: datetime) -> str:
        return _iso(value)

    @field_serializer("completed_at")
    def _serialize_completed_at(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return _iso(value)

    # -- constructors -------------------------------------------------------

    @classmethod
    def start(
        cls,
        project_root: Path,
        intent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CliSession:
        """Mint a new session, persist it, and mark it as current.

        Raises :class:`ValidationError` on invalid project_root (non-dir
        or missing parent) or metadata shape. The ``.carl/sessions``
        directory is created if missing.
        """
        resolved = _validate_project_root(project_root)
        meta = _coerce_metadata(metadata)
        session = cls(
            id=str(uuid.uuid4()),
            project_root=resolved,
            started_at=_now_utc(),
            intent=intent,
            metadata=meta,
            status="active",
            completed_at=None,
        )
        session.save()
        _write_current(resolved, session.id)
        _maybe_sync_to_camp(session)
        return session

    # -- persistence --------------------------------------------------------

    def save(self) -> None:
        """Write this session to ``<project_root>/.carl/sessions/<id>.json``.

        The containing directory is created if missing. Writes are best-
        effort atomic: write to a ``.tmp`` sibling, then ``os.replace``.
        """
        target = _session_file(self.project_root, self.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)

    # -- state transitions --------------------------------------------------

    def complete(self) -> CliSession:
        """Return a new session instance marked completed.

        Persists the new state and emits the sync stub. The returned
        instance replaces this one for the caller's purposes.
        """
        updated = self.model_copy(
            update={"status": "completed", "completed_at": _now_utc()},
        )
        updated.save()
        _maybe_sync_to_camp(updated)
        return updated

    def abandon(self) -> CliSession:
        """Return a new session instance marked abandoned."""
        updated = self.model_copy(
            update={"status": "abandoned", "completed_at": _now_utc()},
        )
        updated.save()
        _maybe_sync_to_camp(updated)
        return updated


# ---------------------------------------------------------------------------
# Module-level helpers (exported)
# ---------------------------------------------------------------------------


def start(
    project_root: Path,
    intent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CliSession:
    """Module-level alias for :meth:`CliSession.start`."""
    return CliSession.start(project_root, intent=intent, metadata=metadata)


def load(session_id: str, project_root: Path) -> CliSession:
    """Load a session by id from ``<project_root>/.carl/sessions/<id>.json``.

    Raises :class:`ValidationError` with ``code='carl.cli_session.not_found'``
    when the file is missing, ``'carl.cli_session.corrupt'`` when the JSON
    cannot be decoded or fails schema validation, and
    ``'carl.cli_session.schema_mismatch'`` when the stored
    ``schema_version`` differs from the current one.
    """
    if not session_id or not _is_valid_uuid(session_id):
        raise ValidationError(
            f"invalid session id: {session_id!r}",
            code="carl.cli_session.invalid_id",
            context={"session_id": session_id},
        )

    resolved = _validate_project_root(project_root)
    path = _session_file(resolved, session_id)
    if not path.is_file():
        raise ValidationError(
            f"session {session_id!r} not found under {resolved}",
            code="carl.cli_session.not_found",
            context={"session_id": session_id, "project_root": str(resolved)},
        )

    try:
        raw_any: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"session {session_id!r} is corrupt: {exc}",
            code="carl.cli_session.corrupt",
            context={"session_id": session_id, "path": str(path)},
            cause=exc,
        ) from exc

    if not isinstance(raw_any, dict):
        raise ValidationError(
            f"session {session_id!r} payload is not a JSON object",
            code="carl.cli_session.corrupt",
            context={"session_id": session_id, "path": str(path)},
        )

    raw = cast(dict[str, Any], raw_any)
    stored_version: Any = raw.get("schema_version", 1)
    if stored_version != SESSION_SCHEMA_VERSION:
        raise ValidationError(
            f"session {session_id!r} schema_version={stored_version} "
            f"(expected {SESSION_SCHEMA_VERSION})",
            code="carl.cli_session.schema_mismatch",
            context={
                "session_id": session_id,
                "stored": stored_version,
                "expected": SESSION_SCHEMA_VERSION,
            },
        )

    try:
        return CliSession.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface all Pydantic failures identically
        raise ValidationError(
            f"session {session_id!r} failed schema validation: {exc}",
            code="carl.cli_session.corrupt",
            context={"session_id": session_id, "path": str(path)},
            cause=exc,
        ) from exc


def save(session: CliSession) -> None:
    """Module-level alias for :meth:`CliSession.save`."""
    session.save()


def list_sessions(project_root: Path) -> list[CliSession]:
    """Enumerate every session under ``<project_root>/.carl/sessions/``.

    Corrupt files are logged at WARNING and skipped (``list`` is a read
    path; callers use :func:`load` for strict semantics). Returns sessions
    sorted by ``started_at`` descending (newest first).
    """
    resolved = _validate_project_root(project_root)
    root = _sessions_dir(resolved)
    if not root.is_dir():
        return []

    sessions: list[CliSession] = []
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix != ".json":
            continue
        session_id = child.stem
        if not _is_valid_uuid(session_id):
            continue
        try:
            sessions.append(load(session_id, resolved))
        except ValidationError as exc:
            logger.warning(
                "skipping corrupt session file %s: %s", child, exc,
            )
            continue

    sessions.sort(key=lambda s: s.started_at, reverse=True)
    return sessions


def current(project_root: Path) -> CliSession | None:
    """Return the currently-active session for ``project_root``, or None.

    Reads ``<project_root>/.carl/sessions/current.txt`` and resolves
    the recorded id. If the file is absent, malformed, or points at a
    session that no longer exists, returns ``None`` (not a raise) so
    CLI surfaces can treat "no current session" as a normal state.
    """
    resolved = _validate_project_root(project_root)
    pointer = _current_file(resolved)
    if not pointer.is_file():
        return None
    try:
        session_id = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not session_id or not _is_valid_uuid(session_id):
        return None
    try:
        return load(session_id, resolved)
    except ValidationError:
        return None


def set_current(project_root: Path, session_id: str) -> None:
    """Mark ``session_id`` as the current session for ``project_root``.

    Validates the id and the presence of the session file before writing
    the pointer. Raises :class:`ValidationError` when the session is
    unknown.
    """
    resolved = _validate_project_root(project_root)
    # Load validates existence + schema.
    load(session_id, resolved)
    _write_current(resolved, session_id)


# ---------------------------------------------------------------------------
# carl.camp sync stub (future-proof)
# ---------------------------------------------------------------------------


def _maybe_sync_to_camp(session: CliSession) -> None:
    """Sync to carl.camp — stub. Platform endpoint pending v0.18.1.

    Per ``docs/platform-parity-reply-2026-04-22.md`` Q4, the v0.18 CLI
    ships the local surface only: the platform route
    (``POST /api/sessions``) is not yet public. This function is the
    single hook where the real HTTP call will land when the route exists.

    Behaviour:

      * ``CARL_CAMP_SESSIONS_SYNC`` unset / any value other than ``"1"``
        — noop. This is the v0.18 default.
      * ``CARL_CAMP_SESSIONS_SYNC=1`` — emit a structured TODO at
        WARNING level so sync-opted-in users see a single-line hint that
        the platform hook is still pending.

    The function **never** raises and **never** performs network I/O.
    """
    if os.environ.get("CARL_CAMP_SESSIONS_SYNC") != "1":
        return
    logger.warning(
        "TODO(sync not wired; platform endpoint pending v0.18.1 coordination): "
        "would sync session=%s status=%s project=%s",
        session.id,
        session.status,
        session.project_root,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_project_root(project_root: Path) -> Path:
    """Return the resolved project_root or raise ValidationError.

    The directory is allowed to exist or not — sessions may be minted
    before any carl.yaml is written. We only require that the *parent*
    exists so we can create ``<project_root>/.carl/sessions`` without
    accidentally fabricating an arbitrary path deep in the filesystem.
    """
    # Defence in depth — the signature types as Path, but callers crossing
    # typed boundaries (dicts, env parses, flow-op dispatch) can still
    # hand us a str. Runtime guard is kept intentionally; pyright strict
    # sees the narrow Path signature and flags the isinstance as
    # unnecessary, which is the source-level lie we cover for here.
    unchecked: object = project_root
    if not isinstance(unchecked, Path):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValidationError(
            f"project_root must be a pathlib.Path, got {type(unchecked).__name__}",
            code="carl.cli_session.invalid_project_root",
            context={"type": type(unchecked).__name__},
        )
    resolved = project_root.expanduser()
    try:
        resolved = resolved.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValidationError(
            f"project_root cannot be resolved: {project_root}",
            code="carl.cli_session.invalid_project_root",
            context={"project_root": str(project_root)},
            cause=exc,
        ) from exc
    if not resolved.parent.exists():
        raise ValidationError(
            f"project_root parent does not exist: {resolved.parent}",
            code="carl.cli_session.invalid_project_root",
            context={"project_root": str(resolved)},
        )
    return resolved


def _coerce_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return a shallow-copied, JSON-safe metadata dict."""
    if metadata is None:
        return {}
    # Defence in depth — callers crossing typed boundaries can still hand
    # us a list or scalar. Pyright strict sees the narrow dict signature
    # and flags the isinstance as unnecessary; we keep the guard anyway
    # because the runtime cost is zero and the failure mode ("list
    # silently saved as a wrapped-in-dict key") is debugging hell.
    unchecked: object = metadata
    if not isinstance(unchecked, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValidationError(
            f"metadata must be a dict, got {type(unchecked).__name__}",
            code="carl.cli_session.invalid_metadata",
            context={"type": type(unchecked).__name__},
        )
    # JSON round-trip to catch non-serializable values at mint-time rather
    # than at save-time (where we'd leak a half-written .tmp on disk).
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"metadata is not JSON-serializable: {exc}",
            code="carl.cli_session.invalid_metadata",
            cause=exc,
        ) from exc
    return dict(metadata)


def _is_valid_uuid(candidate: str) -> bool:
    try:
        uuid.UUID(candidate)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _write_current(project_root: Path, session_id: str) -> None:
    """Write ``session_id`` atomically to ``current.txt``."""
    target = _current_file(project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".txt.tmp")
    tmp.write_text(session_id, encoding="utf-8")
    os.replace(tmp, target)
