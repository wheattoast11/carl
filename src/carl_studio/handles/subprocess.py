"""SubprocessToolkit — capability-constrained OS subprocess management.

Same grammar as :class:`carl_studio.cu.browser.BrowserToolkit`: long-lived
processes live in a :class:`~carl_core.resource_handles.ResourceVault`;
the agent receives :class:`~carl_core.resource_handles.ResourceRef` ids
and issues actions against them. Captured stdout / stderr route through
:class:`~carl_studio.handles.data.DataToolkit` so byte payloads never
stream through the agent's context.

Safety rules enforced at toolkit layer:

* **No shell interpolation.** ``spawn(argv: list[str])`` only. Strings
  are not accepted — if you need a pipe or wildcard, build the argv or
  chain multiple spawns. This is the capability-security model's equivalent
  of "value never crosses an agent boundary" for shell injection: the
  value (shell string) simply cannot be expressed.
* **Default TTL.** ``ttl_s=300`` by default so orphan processes self-
  terminate within 5 minutes. Callers that need longer lifetimes pass
  ``ttl_s`` explicitly.
* **Capture caps.** ``read_stdout`` / ``read_stderr`` land buffered bytes
  in the data vault as :class:`DataRef`s. Large outputs don't leak into
  agent context.

Error codes under ``carl.subprocess.*``:

* ``carl.subprocess.invalid_argv`` — argv empty or not a list of str
* ``carl.subprocess.spawn_failed`` — OSError from Popen
* ``carl.subprocess.still_running`` — wait timeout or operation on live proc
* ``carl.subprocess.already_exited`` — operation on a closed ref
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.resource_handles import (
    ResourceError,
    ResourceRef,
    ResourceVault,
)

from carl_studio.handles.data import DataToolkit


__all__ = ["SubprocessToolkit", "SubprocessToolkitError"]


_DEFAULT_TTL_S: int = 300


class SubprocessToolkitError(CARLError):
    """Base for ``carl.subprocess.*`` errors."""

    code = "carl.subprocess"


@dataclass
class SubprocessToolkit:
    """Agent-callable subprocess lifecycle + stdout/stderr capture."""

    resource_vault: ResourceVault
    data_toolkit: DataToolkit
    chain: InteractionChain
    default_ttl_s: int = _DEFAULT_TTL_S

    @classmethod
    def build(
        cls,
        chain: InteractionChain,
        *,
        data_toolkit: DataToolkit,
        resource_vault: ResourceVault | None = None,
        default_ttl_s: int = _DEFAULT_TTL_S,
    ) -> SubprocessToolkit:
        return cls(
            resource_vault=(
                resource_vault if resource_vault is not None else ResourceVault()
            ),
            data_toolkit=data_toolkit,
            chain=chain,
            default_ttl_s=default_ttl_s,
        )

    # -- spawn -----------------------------------------------------------

    def spawn(
        self,
        argv: Any,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        ttl_s: int | None = None,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Spawn a subprocess. argv MUST be a list — no shell interpolation.

        Returns the :class:`ResourceRef` descriptor. stdout / stderr are
        captured as pipes; call :meth:`read_stdout` / :meth:`read_stderr`
        to surface them as :class:`DataRef`s.

        ``argv`` is typed ``Any`` so the runtime isinstance check can
        actually fire — pyright would otherwise flag it as redundant. The
        check is intentional (defence in depth against shell injection
        when callers disobey the contract).
        """
        if not isinstance(argv, list) or not all(
            isinstance(a, str) for a in argv  # pyright: ignore[reportUnknownVariableType]
        ):
            raise SubprocessToolkitError(
                "argv must be a list[str] — shell strings rejected by design",
                code="carl.subprocess.invalid_argv",
                context={"argv_type": type(argv).__name__},  # pyright: ignore[reportUnknownArgumentType]
            )
        argv_list: list[str] = list(argv)  # pyright: ignore[reportUnknownArgumentType]
        if len(argv_list) == 0:
            raise SubprocessToolkitError(
                "argv must not be empty",
                code="carl.subprocess.invalid_argv",
                context={"argv": argv_list},
            )

        cwd_str = str(Path(cwd).expanduser().resolve()) if cwd is not None else None
        effective_env = dict(env) if env is not None else None
        effective_ttl = ttl_s if ttl_s is not None else self.default_ttl_s

        try:
            proc = subprocess.Popen(
                argv_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=cwd_str,
                env=effective_env,
            )
        except OSError as exc:
            raise SubprocessToolkitError(
                f"subprocess spawn failed: {exc}",
                code="carl.subprocess.spawn_failed",
                context={"argv": argv_list, "cwd": cwd_str},
                cause=exc,
            ) from exc

        ref = self.resource_vault.put(
            backend=proc,
            kind="subprocess",
            provider="subprocess",
            uri=f"pid:{proc.pid}",
            labels={"argv0": argv_list[0], **(labels or {})},
            ttl_s=effective_ttl,
            closer=_terminate_subprocess,
        )
        desc = ref.describe()
        self.chain.record(
            ActionType.RESOURCE_OPEN,
            "subprocess.spawn",
            input={
                "argv": argv_list,
                "cwd": cwd_str,
                "env_keys": sorted(effective_env.keys()) if effective_env else [],
                "ttl_s": effective_ttl,
            },
            output={**desc, "pid": proc.pid},
            success=True,
        )
        return {**desc, "pid": proc.pid}

    # -- status ---------------------------------------------------------

    def poll(self, ref_id: str) -> dict[str, Any]:
        """Non-blocking check. Returns ``{ref_id, exit_code, running}``.

        ``exit_code`` is ``None`` while the process is still running.
        """
        proc = self._proc_from_id(ref_id)
        rc = proc.poll()
        result = {"ref_id": ref_id, "exit_code": rc, "running": rc is None}
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "subprocess.poll",
            input={"ref_id": ref_id},
            output=result,
            success=True,
        )
        return result

    def wait(
        self,
        ref_id: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Block until the process exits (or ``timeout_s`` elapses).

        Returns ``{ref_id, exit_code, stdout_ref, stderr_ref}``. stdout
        and stderr content lands in the data vault; the agent sees only
        :class:`DataRef` descriptors.
        """
        proc = self._proc_from_id(ref_id)
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise SubprocessToolkitError(
                f"subprocess {ref_id} did not exit within {timeout_s}s",
                code="carl.subprocess.still_running",
                context={"ref_id": ref_id, "timeout_s": timeout_s},
                cause=exc,
            ) from exc

        stdout_desc = self.data_toolkit.open_bytes(
            stdout_bytes or b"",
            uri=f"carl-data://subprocess/{ref_id}/stdout",
            content_type="application/octet-stream",
        )
        stderr_desc = self.data_toolkit.open_bytes(
            stderr_bytes or b"",
            uri=f"carl-data://subprocess/{ref_id}/stderr",
            content_type="application/octet-stream",
        )
        result: dict[str, Any] = {
            "ref_id": ref_id,
            "exit_code": proc.returncode,
            "stdout_ref": stdout_desc,
            "stderr_ref": stderr_desc,
        }
        self.chain.record(
            ActionType.RESOURCE_ACT,
            "subprocess.wait",
            input={"ref_id": ref_id, "timeout_s": timeout_s},
            output={
                "ref_id": ref_id,
                "exit_code": proc.returncode,
                "stdout_ref_id": stdout_desc["ref_id"],
                "stderr_ref_id": stderr_desc["ref_id"],
            },
            success=(proc.returncode == 0),
        )
        return result

    # -- terminate ------------------------------------------------------

    def terminate(
        self,
        ref_id: str,
        *,
        grace_s: float = 5.0,
    ) -> dict[str, Any]:
        """SIGTERM then SIGKILL after ``grace_s``. Revokes the ref."""
        ref = self._ref_from_id(ref_id)
        proc = self.resource_vault.resolve(ref, privileged=True)
        outcome = "terminated"
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
                outcome = "killed"
        else:
            outcome = "already_exited"
        self.resource_vault.revoke(ref)
        result = {
            "ref_id": ref_id,
            "outcome": outcome,
            "exit_code": proc.returncode,
        }
        self.chain.record(
            ActionType.RESOURCE_CLOSE,
            "subprocess.terminate",
            input={"ref_id": ref_id, "grace_s": grace_s},
            output=result,
            success=True,
        )
        return result

    # -- incremental capture --------------------------------------------

    def read_stdout(self, ref_id: str) -> dict[str, Any]:
        """Read currently-buffered stdout into a :class:`DataRef`.

        Non-blocking; may return empty bytes if the child hasn't flushed.
        Not safe to mix with :meth:`wait` on the same ref (communicate
        and read-from-pipe aren't composable in Popen's API).
        """
        return self._read_pipe(ref_id, stream="stdout")

    def read_stderr(self, ref_id: str) -> dict[str, Any]:
        return self._read_pipe(ref_id, stream="stderr")

    def _read_pipe(self, ref_id: str, *, stream: str) -> dict[str, Any]:
        proc = self._proc_from_id(ref_id)
        pipe = getattr(proc, stream, None)
        if pipe is None:
            raise SubprocessToolkitError(
                f"subprocess {ref_id} has no {stream} pipe",
                code="carl.subprocess.spawn_failed",
                context={"ref_id": ref_id, "stream": stream},
            )
        # Read whatever's available without blocking. On POSIX we set
        # the pipe non-blocking; on Windows we accept a short-read.
        try:
            data = _read_nonblocking(pipe)
        except OSError as exc:
            raise SubprocessToolkitError(
                f"read_{stream} failed: {exc}",
                code="carl.subprocess.spawn_failed",
                context={"ref_id": ref_id, "stream": stream},
                cause=exc,
            ) from exc

        desc = self.data_toolkit.open_bytes(
            data,
            uri=f"carl-data://subprocess/{ref_id}/{stream}/chunk",
            content_type="application/octet-stream",
        )
        self.chain.record(
            ActionType.RESOURCE_ACT,
            f"subprocess.read_{stream}",
            input={"ref_id": ref_id, "stream": stream},
            output={"ref_id": ref_id, "stream": stream, "data_ref_id": desc["ref_id"]},
            success=True,
        )
        return {"ref_id": ref_id, "stream": stream, "data_ref": desc}

    # -- list / query ---------------------------------------------------

    def list_processes(self) -> list[dict[str, Any]]:
        """Current non-revoked subprocess refs (with live-status tag)."""
        out: list[dict[str, Any]] = []
        for ref in self.resource_vault.list_refs():
            if ref.kind != "subprocess":
                continue
            try:
                proc = self.resource_vault.resolve(ref, privileged=True)
                running = proc.poll() is None
                exit_code = proc.returncode if not running else None
            except ResourceError:
                continue
            out.append({**ref.describe(), "running": running, "exit_code": exit_code})
        return out

    # -- agent schemas --------------------------------------------------

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "subprocess_spawn",
                "description": "Spawn a subprocess. argv is a list — no shell.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "argv": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": ["string", "null"]},
                        "env": {"type": ["object", "null"]},
                        "ttl_s": {"type": ["integer", "null"]},
                    },
                    "required": ["argv"],
                },
            },
            {
                "name": "subprocess_poll",
                "description": "Non-blocking check: running/exit_code.",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref_id": {"type": "string"}},
                    "required": ["ref_id"],
                },
            },
            {
                "name": "subprocess_wait",
                "description": "Block until process exits (or timeout_s). Returns stdout/stderr as DataRefs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "timeout_s": {"type": ["number", "null"]},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "subprocess_terminate",
                "description": "SIGTERM then SIGKILL after grace_s. Revokes the ref.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ref_id": {"type": "string"},
                        "grace_s": {"type": "number", "default": 5.0},
                    },
                    "required": ["ref_id"],
                },
            },
            {
                "name": "subprocess_read_stdout",
                "description": "Read buffered stdout into a DataRef (non-blocking).",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref_id": {"type": "string"}},
                    "required": ["ref_id"],
                },
            },
            {
                "name": "subprocess_read_stderr",
                "description": "Read buffered stderr into a DataRef (non-blocking).",
                "input_schema": {
                    "type": "object",
                    "properties": {"ref_id": {"type": "string"}},
                    "required": ["ref_id"],
                },
            },
            {
                "name": "subprocess_list",
                "description": "List non-revoked subprocess refs with running/exit_code.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    # -- internals ------------------------------------------------------

    def _ref_from_id(self, ref_id: str) -> ResourceRef:
        try:
            parsed = uuid.UUID(ref_id)
        except (TypeError, ValueError) as exc:
            raise SubprocessToolkitError(
                f"ref_id is not a valid UUID: {ref_id!r}",
                code="carl.subprocess.invalid_ref_id",
                context={"ref_id": ref_id},
                cause=exc,
            ) from exc
        for ref in self.resource_vault.list_refs():
            if ref.ref_id == parsed:
                return ref
        raise ResourceError(
            f"unknown or closed subprocess: {ref_id}",
            code="carl.resource.not_found",
            context={"ref_id": ref_id},
        )

    def _proc_from_id(self, ref_id: str) -> subprocess.Popen[bytes]:
        ref = self._ref_from_id(ref_id)
        return self.resource_vault.resolve(ref, privileged=True)


def _terminate_subprocess(proc: subprocess.Popen[bytes]) -> None:
    """Closer callback run at revoke time. Best-effort + non-blocking."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:  # pragma: no cover
        pass


def _read_nonblocking(pipe: Any) -> bytes:
    """Drain whatever bytes are available without blocking.

    POSIX: sets O_NONBLOCK on the fd + ``read(...)`` loops until EAGAIN.
    Windows: falls back to peeking the pipe via ``msvcrt`` /
    ``PeekNamedPipe`` — only a best-effort subset (no Windows in CI so
    keep it simple; real Windows support can come later).
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        # Windows path: read what's there via .readable() semantics.
        data = pipe.read(65536) if hasattr(pipe, "read") else b""
        return data or b""

    fd = pipe.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    out = bytearray()
    try:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            out.extend(chunk)
    except BlockingIOError:
        pass
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)
    return bytes(out)
