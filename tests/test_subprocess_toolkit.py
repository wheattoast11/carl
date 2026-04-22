"""Tests for carl_studio.handles.subprocess.SubprocessToolkit.

Uses real ``subprocess.Popen`` against ``/bin/echo`` + ``/bin/sh -c`` to
exercise the lifecycle end-to-end. Does NOT mock subprocess — that would
validate the wrapper only. These are integration tests against real child
processes; they're fast because the children are trivial.
"""

from __future__ import annotations

import base64
import shutil
import sys
import time
from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.handles.data import DataToolkit
from carl_studio.handles.subprocess import (
    SubprocessToolkit,
    SubprocessToolkitError,
)


def _build_toolkit() -> tuple[InteractionChain, SubprocessToolkit]:
    chain = InteractionChain()
    data = DataToolkit.build(chain)
    tk = SubprocessToolkit.build(chain, data_toolkit=data)
    return chain, tk


# ---------------------------------------------------------------------------
# argv validation
# ---------------------------------------------------------------------------


def test_spawn_rejects_string_argv() -> None:
    _, tk = _build_toolkit()
    with pytest.raises(SubprocessToolkitError) as exc:
        tk.spawn("echo hi")  # type: ignore[arg-type]
    assert exc.value.code == "carl.subprocess.invalid_argv"


def test_spawn_rejects_empty_argv() -> None:
    _, tk = _build_toolkit()
    with pytest.raises(SubprocessToolkitError) as exc:
        tk.spawn([])
    assert exc.value.code == "carl.subprocess.invalid_argv"


def test_spawn_rejects_non_str_elements() -> None:
    _, tk = _build_toolkit()
    with pytest.raises(SubprocessToolkitError) as exc:
        tk.spawn(["echo", 123])  # type: ignore[list-item]
    assert exc.value.code == "carl.subprocess.invalid_argv"


# ---------------------------------------------------------------------------
# happy-path spawn / wait
# ---------------------------------------------------------------------------


def _python_exe() -> str:
    return sys.executable


def test_spawn_and_wait_captures_stdout() -> None:
    chain, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "print('hello world')"])
    assert "ref_id" in desc
    assert "pid" in desc

    result = tk.wait(desc["ref_id"], timeout_s=10.0)
    assert result["exit_code"] == 0
    assert "stdout_ref" in result
    # Read the stdout DataRef back
    stdout_bytes = base64.b64decode(
        tk.data_toolkit.read(result["stdout_ref"]["ref_id"], length=1024)["bytes_b64"]
    )
    assert b"hello world" in stdout_bytes


def test_spawn_emits_resource_open_step() -> None:
    chain, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "pass"])
    tk.wait(desc["ref_id"], timeout_s=5.0)
    opens = chain.by_action(ActionType.RESOURCE_OPEN)
    assert any(s.name == "subprocess.spawn" for s in opens)


def test_wait_captures_stderr() -> None:
    _, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "import sys; sys.stderr.write('oops\\n')"])
    result = tk.wait(desc["ref_id"], timeout_s=5.0)
    stderr_bytes = base64.b64decode(
        tk.data_toolkit.read(result["stderr_ref"]["ref_id"], length=1024)["bytes_b64"]
    )
    assert b"oops" in stderr_bytes


def test_nonzero_exit_code_recorded_as_failure() -> None:
    chain, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "import sys; sys.exit(42)"])
    result = tk.wait(desc["ref_id"], timeout_s=5.0)
    assert result["exit_code"] == 42
    waits = [s for s in chain.by_action(ActionType.RESOURCE_ACT) if s.name == "subprocess.wait"]
    assert waits[-1].success is False


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------


def test_poll_running_then_completed() -> None:
    _, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "import time; time.sleep(0.3)"])
    rid = desc["ref_id"]
    status = tk.poll(rid)
    # May already have finished on very fast systems; both branches valid
    assert "running" in status
    # Drain
    tk.wait(rid, timeout_s=5.0)
    final = tk.poll(rid)
    assert final["running"] is False
    assert final["exit_code"] == 0


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------


def test_terminate_kills_long_running_process() -> None:
    chain, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "import time; time.sleep(30)"])
    # Give the child a moment to actually start
    time.sleep(0.1)
    result = tk.terminate(desc["ref_id"], grace_s=2.0)
    assert result["outcome"] in ("terminated", "killed")
    assert result["exit_code"] is not None
    closes = chain.by_action(ActionType.RESOURCE_CLOSE)
    assert any(s.name == "subprocess.terminate" for s in closes)


def test_terminate_on_already_exited_process() -> None:
    _, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "pass"])
    tk.wait(desc["ref_id"], timeout_s=5.0)
    # Terminating an already-completed process is a no-op with outcome=already_exited
    result = tk.terminate(desc["ref_id"])
    assert result["outcome"] == "already_exited"


# ---------------------------------------------------------------------------
# read_stdout / read_stderr streaming
# ---------------------------------------------------------------------------


def test_read_stdout_returns_dataref(tmp_path: Any) -> None:
    _, tk = _build_toolkit()
    # Slow-writing child so we can read mid-flight
    desc = tk.spawn(
        [
            _python_exe(),
            "-c",
            "import sys; sys.stdout.write('chunk\\n'); sys.stdout.flush()",
        ]
    )
    # Wait a beat for the child to flush
    time.sleep(0.2)
    result = tk.read_stdout(desc["ref_id"])
    assert "data_ref" in result
    # Clean up
    tk.terminate(desc["ref_id"])


# ---------------------------------------------------------------------------
# list_processes
# ---------------------------------------------------------------------------


def test_list_processes_reports_live() -> None:
    _, tk = _build_toolkit()
    long_lived = tk.spawn(
        [_python_exe(), "-c", "import time; time.sleep(10)"]
    )
    try:
        listed = tk.list_processes()
        pids = [p["uri"] for p in listed]
        assert any(long_lived["pid"] == int(u.split("pid:")[1]) for u in pids)
    finally:
        tk.terminate(long_lived["ref_id"])


# ---------------------------------------------------------------------------
# TTL auto-close
# ---------------------------------------------------------------------------


def test_ttl_self_revoke_on_resolve() -> None:
    # ttl_s=1 means the ref is expired after 1 second; subsequent lookup
    # filters the ref out of list_refs, which manifests as not_found (the
    # caller's perspective: the ref is gone). Both error codes are valid
    # ways to say "the ref is no longer addressable."
    _, tk = _build_toolkit()
    desc = tk.spawn([_python_exe(), "-c", "import time; time.sleep(5)"], ttl_s=1)
    try:
        time.sleep(1.1)
        with pytest.raises(Exception) as exc:
            tk.poll(desc["ref_id"])
        code = str(getattr(exc.value, "code", ""))
        assert "expired" in code or "not_found" in code
    finally:
        # Explicit cleanup — the background python sleep(5) is still alive
        # behind an expired/revoked ref; terminate via PID directly.
        import os
        import signal

        try:
            os.kill(desc["pid"], signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass


# ---------------------------------------------------------------------------
# Bad path: spawn of nonexistent executable
# ---------------------------------------------------------------------------


def test_spawn_nonexistent_executable_raises() -> None:
    _, tk = _build_toolkit()
    with pytest.raises(SubprocessToolkitError) as exc:
        tk.spawn(["/nonexistent/path/to/exe"])
    assert exc.value.code == "carl.subprocess.spawn_failed"


# ---------------------------------------------------------------------------
# tool_schemas
# ---------------------------------------------------------------------------


def test_tool_schemas_cover_methods() -> None:
    _, tk = _build_toolkit()
    names = {s["name"] for s in tk.tool_schemas()}
    assert {
        "subprocess_spawn",
        "subprocess_poll",
        "subprocess_wait",
        "subprocess_terminate",
        "subprocess_read_stdout",
        "subprocess_read_stderr",
        "subprocess_list",
    } <= names


@pytest.mark.skipif(shutil.which("sh") is None, reason="/bin/sh required")
def test_cwd_is_respected(tmp_path: Any) -> None:
    marker = tmp_path / "marker.txt"
    marker.write_text("from-cwd")
    _, tk = _build_toolkit()
    desc = tk.spawn(
        [_python_exe(), "-c", "import pathlib; print(pathlib.Path('marker.txt').read_text())"],
        cwd=tmp_path,
    )
    result = tk.wait(desc["ref_id"], timeout_s=5.0)
    assert result["exit_code"] == 0
    stdout_bytes = base64.b64decode(
        tk.data_toolkit.read(result["stdout_ref"]["ref_id"], length=1024)["bytes_b64"]
    )
    assert b"from-cwd" in stdout_bytes
