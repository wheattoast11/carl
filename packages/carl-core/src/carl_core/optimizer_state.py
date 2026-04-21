"""OptimizerStateStore — durable Adam (m, v) across sessions.

Why this exists: the heartbeat theorem (see ``carl_core.heartbeat``)
requires the Adam moments to *persist* across restarts. Without that,
every session starts from zero momentum and the "self-referential
standing wave" is broken every launch. With persistence, Adam's (m, v)
state acts as the system's own phase accumulator across real wall-clock
time — the substrate of Tej's emergent heartbeat.

Design is deliberately parallel to ``MemoryStore``:

    <root>/<session_id>/<module_id>.npz

Concurrency and crash safety:

* Writes use tempfile + atomic rename (``Path.replace``) — a kill mid-write
  leaves either the prior good file or the new good file, never a partial.
* Corrupted files are detected on load (via ``np.load`` exceptions), renamed
  to ``<name>.quarantine_<ts>`` so the operator can forensic them, and
  ``load()`` returns ``None`` so the caller can cold-start.
* Per-session+module locks serialise concurrent saves of the same slot.

This module is torch-free. Adam moments are plain numpy arrays; downstream
callers (the trainer, the heartbeat loop, inference-time adapters) pull
them out via `load()` and keep them in whatever runtime representation
they need.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carl_core.errors import ValidationError
from carl_core.hashing import content_hash

__all__ = [
    "DEFAULT_OPT_STATE_DIR",
    "AdamMoments",
    "OptimizerStateStore",
]


DEFAULT_OPT_STATE_DIR: Path = Path.home() / ".carl" / "optimizer_states"


# Conservative character whitelist for filesystem-safe ids.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_id(name: str, *, kind: str) -> str:
    """Reject ids with path separators, dots-only, or anything outside the whitelist."""
    if not name:
        raise ValidationError(
            f"{kind} must be a non-empty string",
            code="carl.optimizer_state.bad_id",
            context={"kind": kind, "got": repr(name)},
        )
    if name in {".", ".."}:
        raise ValidationError(
            f"{kind} may not be '.' or '..'",
            code="carl.optimizer_state.bad_id",
            context={"kind": kind, "got": name},
        )
    if not _SAFE_ID.match(name):
        raise ValidationError(
            f"{kind} must match [A-Za-z0-9_.-]+",
            code="carl.optimizer_state.bad_id",
            context={"kind": kind, "got": name},
        )
    return name


@dataclass
class AdamMoments:
    """Serializable Adam moment tuple + provenance.

    Pair of numpy arrays (m, v) plus the step counter and the betas they
    were accumulated against. `hash()` returns a stable fingerprint so
    callers can diff persisted states without rehashing the raw npz.
    """

    m: NDArray[np.float64]
    v: NDArray[np.float64]
    step: int
    beta1: float
    beta2: float
    created_at: float
    last_updated_at: float

    def __post_init__(self) -> None:
        if self.m.shape != self.v.shape:
            raise ValidationError(
                "AdamMoments m and v must have identical shape",
                code="carl.optimizer_state.shape_mismatch",
                context={"m": list(self.m.shape), "v": list(self.v.shape)},
            )
        if self.step < 0:
            raise ValidationError(
                "AdamMoments.step must be non-negative",
                code="carl.optimizer_state.bad_step",
                context={"step": self.step},
            )
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValidationError(
                "AdamMoments betas must be in [0, 1)",
                code="carl.optimizer_state.bad_betas",
                context={"beta1": self.beta1, "beta2": self.beta2},
            )

    def hash(self) -> str:
        """Stable hash of the moment payload + metadata.

        Uses a rounded version of the arrays so numeric jitter below
        float64 precision doesn't change the digest.
        """
        return content_hash(
            {
                "m": [round(float(x), 15) for x in self.m.reshape(-1)],
                "v": [round(float(x), 15) for x in self.v.reshape(-1)],
                "shape": list(self.m.shape),
                "step": int(self.step),
                "beta1": float(self.beta1),
                "beta2": float(self.beta2),
                "created_at": float(self.created_at),
                "last_updated_at": float(self.last_updated_at),
            }
        )


class OptimizerStateStore:
    """Per-session, per-module Adam (m, v) persistence.

    Parallel to ``MemoryStore`` but tuned for binary numpy state instead of
    JSONL text. Files land at
    ``<root>/<session_id>/<module_id>.npz`` with one ``.npz`` per
    (session, module) pair.

    This store is safe to share across threads in the same process via
    the internal RLock. It is also safe to share across processes: the
    atomic-rename pattern guarantees reads never observe a partial file.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = Path(root) if root is not None else DEFAULT_OPT_STATE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock: threading.RLock = threading.RLock()

    # -- path helpers -------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        _validate_id(session_id, kind="session_id")
        return self.root / session_id

    def _file_path(self, session_id: str, module_id: str) -> Path:
        _validate_id(module_id, kind="module_id")
        return self._session_dir(session_id) / f"{module_id}.npz"

    # -- public API ---------------------------------------------------------

    def save(
        self,
        session_id: str,
        module_id: str,
        moments: AdamMoments,
    ) -> Path:
        """Persist `moments` to ``<root>/<session>/<module>.npz`` atomically.

        Returns the final path. Creates the session directory on demand.
        Raises ``ValidationError`` on bad ids and re-raises any OS-level
        errors unchanged.
        """
        return self.atomic_save(session_id, module_id, moments)

    def atomic_save(
        self,
        session_id: str,
        module_id: str,
        moments: AdamMoments,
    ) -> Path:
        """Tempfile + fsync + rename.

        Guarantees: the target file either exists with its prior contents,
        or exists with the new contents. No partial writes are ever visible
        to a concurrent reader.
        """
        final_path = self._file_path(session_id, module_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # ``np.savez`` appends ``.npz`` to string/path arguments if the
            # name doesn't already end in ``.npz``. We write through an open
            # file handle to bypass that behaviour and keep the exact
            # temp-name we chose — so the rename target stays stable.
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=f".{module_id}.",
                suffix=".tmp",
                dir=str(final_path.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                # `fdopen` binds the fd into a binary-write file object.
                # `np.savez` accepts a file-like target and writes the zip
                # archive to that exact handle — no auto-suffixing.
                with os.fdopen(tmp_fd, "wb") as fh:
                    np.savez(
                        fh,
                        m=moments.m,
                        v=moments.v,
                        step=np.array(moments.step, dtype=np.int64),
                        beta1=np.array(moments.beta1, dtype=np.float64),
                        beta2=np.array(moments.beta2, dtype=np.float64),
                        created_at=np.array(
                            moments.created_at, dtype=np.float64
                        ),
                        last_updated_at=np.array(
                            moments.last_updated_at, dtype=np.float64
                        ),
                    )
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        # Best-effort: not every fs supports fsync; atomic
                        # rename still gives us crash safety.
                        pass
                # Atomic rename into place.
                tmp_path.replace(final_path)
            except BaseException:
                # Clean up the temp file on any failure, then re-raise. If
                # the fd was already consumed by fdopen above, the file will
                # still exist under `tmp_path` and want unlinking.
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                raise
        return final_path

    def load(
        self,
        session_id: str,
        module_id: str,
    ) -> AdamMoments | None:
        """Read the persisted moments, or return None.

        On file-not-found: returns None.
        On corruption: quarantines the file under a ``.quarantine_<ts>``
        sibling and returns None. The operator can recover or discard.
        """
        path = self._file_path(session_id, module_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                with np.load(path, allow_pickle=False) as data:
                    # Access every required field — any missing key or bad
                    # dtype triggers the quarantine branch below.
                    m = np.asarray(data["m"], dtype=np.float64)
                    v = np.asarray(data["v"], dtype=np.float64)
                    step = int(np.asarray(data["step"]).item())
                    beta1 = float(np.asarray(data["beta1"]).item())
                    beta2 = float(np.asarray(data["beta2"]).item())
                    created_at = float(
                        np.asarray(data["created_at"]).item()
                    )
                    last_updated_at = float(
                        np.asarray(data["last_updated_at"]).item()
                    )
            except Exception:
                # numpy raises ``zipfile.BadZipFile`` (an ``Exception``
                # subclass) on a corrupt ``.npz``, plus every narrower
                # archive / decode failure surfaces as ``OSError`` /
                # ``ValueError`` / ``KeyError`` / ``TypeError`` /
                # ``EOFError`` — all of which are ``Exception`` subclasses.
                # We catch the base so the loader never crashes: any
                # failure quarantines the file and returns ``None`` so the
                # caller can cold-start.
                self.quarantine(session_id, module_id)
                return None
        try:
            return AdamMoments(
                m=m,
                v=v,
                step=step,
                beta1=beta1,
                beta2=beta2,
                created_at=created_at,
                last_updated_at=last_updated_at,
            )
        except ValidationError:
            # Internally-consistent but semantically bad — treat as corrupt.
            self.quarantine(session_id, module_id)
            return None

    def list_sessions(self) -> list[str]:
        """Sorted list of session ids currently present under root."""
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def list_modules(self, session_id: str) -> list[str]:
        """Sorted module ids in a given session. Empty if session absent."""
        d = self._session_dir(session_id)
        if not d.exists():
            return []
        return sorted(
            p.stem for p in d.iterdir() if p.is_file() and p.suffix == ".npz"
        )

    def delete_session(self, session_id: str) -> int:
        """Remove every file in a session directory. Returns count removed.

        The directory itself is also removed (if empty). Non-npz files
        inside the session directory (quarantine artefacts, etc.) are also
        deleted so delete_session is a true 'forget this session'.
        """
        d = self._session_dir(session_id)
        if not d.exists():
            return 0
        removed = 0
        with self._lock:
            for p in list(d.iterdir()):
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                    elif p.is_dir():
                        # Nested dirs aren't part of the schema — ignore.
                        continue
                except OSError:
                    continue
            try:
                d.rmdir()
            except OSError:
                # Non-empty (residual nested dir) — leave it for the operator.
                pass
        return removed

    def quarantine(
        self,
        session_id: str,
        module_id: str,
    ) -> Path | None:
        """Move a corrupt file out of the way. Returns the quarantine path or None."""
        path = self._file_path(session_id, module_id)
        if not path.exists():
            return None
        stamp = int(time.time_ns())
        q = path.with_name(f"{path.name}.quarantine_{stamp}")
        try:
            path.rename(q)
            return q
        except OSError:
            return None

    # -- convenience -------------------------------------------------------

    def save_arrays(
        self,
        session_id: str,
        module_id: str,
        m: NDArray[np.float64],
        v: NDArray[np.float64],
        *,
        step: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Path:
        """Shortcut: build an :class:`AdamMoments` from raw arrays + save it."""
        now = time.time()
        existing = self.load(session_id, module_id)
        created = existing.created_at if existing is not None else now
        moments = AdamMoments(
            m=np.asarray(m, dtype=np.float64),
            v=np.asarray(v, dtype=np.float64),
            step=int(step),
            beta1=float(beta1),
            beta2=float(beta2),
            created_at=created,
            last_updated_at=now,
        )
        return self.atomic_save(session_id, module_id, moments)

    def has(self, session_id: str, module_id: str) -> bool:
        """Check existence without loading."""
        return self._file_path(session_id, module_id).exists()

    def summary(self) -> dict[str, Any]:
        """Cheap overview: session count, total modules, root path."""
        sessions = self.list_sessions()
        total_modules = sum(len(self.list_modules(s)) for s in sessions)
        return {
            "root": str(self.root),
            "sessions": sessions,
            "session_count": len(sessions),
            "module_count": total_modules,
        }
