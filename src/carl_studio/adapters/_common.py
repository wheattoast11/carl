"""Internal helpers shared by adapter implementations.

These utilities are NOT part of the public API. They handle run-id
generation, on-disk state files, and process lifecycle so each adapter
stays under ~200 lines of glue.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import AdapterError, BackendJob, BackendStatus


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(os.environ.get("CARL_HOME", Path.home() / ".carl")) / "adapters"


def state_dir(backend: str) -> Path:
    """Return the per-backend state directory, creating it if needed."""
    override = os.environ.get("CARL_ADAPTER_STATE_DIR")
    root = Path(override) if override else _DEFAULT_ROOT
    path = root / backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_id(prefix: str) -> str:
    """Generate a short, collision-resistant run id."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Job state file (one JSON blob per run_id)
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    """Persisted adapter state for a single run."""

    run_id: str
    backend: str
    status: str = BackendStatus.PENDING
    submitted_at: str | None = None
    completed_at: str | None = None
    pid: int | None = None
    log_path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_job(self) -> BackendJob:
        def _parse(ts: str | None) -> datetime | None:
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return None

        return BackendJob(
            run_id=self.run_id,
            backend=self.backend,
            status=self.status,
            submitted_at=_parse(self.submitted_at),
            completed_at=_parse(self.completed_at),
            metrics=dict(self.metrics),
            logs_url=None,
            raw={**self.raw, "pid": self.pid, "log_path": self.log_path},
        )


def _state_path(backend: str, run_id: str) -> Path:
    return state_dir(backend) / f"{run_id}.json"


def save_state(state: JobState) -> None:
    """Persist ``state`` atomically to disk."""
    path = _state_path(state.backend, state.run_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_state(backend: str, run_id: str) -> JobState:
    """Load persisted state for ``run_id`` under ``backend``."""
    path = _state_path(backend, run_id)
    if not path.exists():
        raise AdapterError(
            f"unknown {backend} run: {run_id}",
            code="carl.adapter.status",
            context={"run_id": run_id, "backend": backend},
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(
            f"state file for {run_id} is unreadable: {exc}",
            code="carl.adapter.status",
            context={"run_id": run_id, "backend": backend, "path": str(path)},
            cause=exc,
        ) from exc
    return JobState(**data)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------


def refresh_pid_status(state: JobState) -> JobState:
    """If ``state.pid`` is set and the process is done, mark state terminal.

    Mutates and returns ``state``. Safe to call repeatedly.
    """
    if BackendStatus.is_terminal(state.status):
        return state
    if state.pid is None:
        return state
    try:
        pid_gone = not _pid_alive(state.pid)
    except Exception:
        # If we cannot probe, leave status unchanged.
        return state
    if pid_gone:
        state.status = BackendStatus.COMPLETED
        state.completed_at = state.completed_at or now_iso()
    else:
        state.status = BackendStatus.RUNNING
    return state


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists, owned by someone else (should not happen, but honest).
        return True
    return True


def cancel_pid(state: JobState) -> bool:
    """Send SIGTERM to ``state.pid`` if it is still running.

    Returns True if the signal was sent (or the process was already gone).
    """
    if state.pid is None:
        return False
    try:
        if not _pid_alive(state.pid):
            state.status = BackendStatus.CANCELED
            state.completed_at = state.completed_at or now_iso()
            return True
        os.kill(state.pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already gone — idempotent success.
        pass
    except PermissionError as exc:
        raise AdapterError(
            f"cannot cancel pid {state.pid}: permission denied",
            code="carl.adapter.cancel",
            context={"pid": state.pid, "run_id": state.run_id},
            cause=exc,
        ) from exc
    state.status = BackendStatus.CANCELED
    state.completed_at = state.completed_at or now_iso()
    return True


def spawn(cmd: list[str], *, log_path: Path, cwd: Path | None = None,
          stdin: str | None = None, env: dict[str, str] | None = None) -> int:
    """Spawn a detached subprocess, redirecting stdout+stderr to ``log_path``.

    Returns the child PID. The parent does NOT wait — callers read pid from
    state and use :func:`refresh_pid_status` to observe liveness.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise AdapterError(
            f"command not found: {cmd[0]!r}",
            code="carl.adapter.submit",
            context={"cmd": cmd},
            cause=exc,
        ) from exc
    except OSError as exc:
        log_file.close()
        raise AdapterError(
            f"failed to spawn {cmd[0]!r}: {exc}",
            code="carl.adapter.submit",
            context={"cmd": cmd},
            cause=exc,
        ) from exc

    if stdin is not None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(stdin.encode("utf-8"))
            proc.stdin.close()
        except OSError:
            # The child may have closed stdin already; not fatal.
            pass

    return proc.pid


def tail_log(log_path: Path | None, tail: int) -> list[str]:
    """Return the last ``tail`` lines of ``log_path`` (bytes→utf-8, lossy)."""
    if tail < 1:
        raise AdapterError(
            "tail must be >= 1",
            code="carl.adapter.logs",
            context={"tail": tail},
        )
    if log_path is None or not Path(log_path).exists():
        return []
    try:
        raw = Path(log_path).read_bytes()
    except OSError as exc:
        raise AdapterError(
            f"cannot read log: {exc}",
            code="carl.adapter.logs",
            context={"log_path": str(log_path)},
            cause=exc,
        ) from exc
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-tail:]


# ---------------------------------------------------------------------------
# Unavailability helper
# ---------------------------------------------------------------------------


def unavailable(backend: str, *, hint: str) -> AdapterError:
    """Build a consistent AdapterError for missing-backend cases."""
    return AdapterError(
        f"backend {backend!r} is not available on this machine",
        code="carl.adapter.unavailable",
        context={"backend": backend, "install_hint": hint},
    )
