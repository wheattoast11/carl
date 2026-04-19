"""Coding sandbox environment -- real subprocess execution, file I/O, binary reward.

This is the reference CARL environment. TRL creates one instance per generation,
auto-discovers public methods as tools, handles the multi-turn loop.

Mirrors the standalone CodingSandboxEnv in zero-rl-pipeline/phase2/coding_sandbox_env.py
but inherits from carl_studio's BaseEnvironment for turn history and CARL analysis.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)

from carl_studio.environments.connection import EnvironmentConnection
from carl_studio.environments.protocol import EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import register_environment

_EXEC_TIMEOUT = 10  # seconds


@register_environment
class CodingSandboxEnv(EnvironmentConnection):
    """Real coding sandbox -- subprocess execution, tempdir isolation, binary reward.

    Each instance gets an isolated temp directory. The model reads files,
    writes code, and executes it. The reward is whether the code runs
    without error.

    Instrumentation attributes (read by reward functions and callbacks):
        _tool_call_count: Total tool invocations this episode.
        _tool_failure_count: Tool calls that returned errors.
        _execution_attempted: Whether execute_code or run_shell was called.
        _execution_succeeded: Whether the last execution had exit code 0.
    """

    spec = EnvironmentSpec(
        lane=EnvironmentLane.CODE,
        name="python-sandbox",
        tools=("read_file", "write_file", "execute_code", "run_shell"),
        max_turns=8,
        reward_type="binary",
        system_prompt=(
            "You are a coding agent. You solve programming tasks by reading files, "
            "writing code, and executing it.\n\n"
            "Use the available tools to interact with the sandbox environment. "
            "Write code, run it, check the output, fix errors. "
            "The task is complete when your code executes successfully.\n\n"
            "Do NOT explain what you're doing. Just act."
        ),
        dataset_columns=("task_description",),
    )

    connection_spec = ConnectionSpec(
        name="carl.env.code",
        scope=ConnectionScope.ONE_P,
        kind=ConnectionKind.ENVIRONMENT,
        direction=ConnectionDirection.BIDIRECTIONAL,
        transport=ConnectionTransport.IN_PROCESS,
        trust=ConnectionTrust.PUBLIC,
        metadata={"lane": "code", "sandbox": "python-subprocess"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.workdir: str | None = None
        self._execution_attempted: bool = False
        self._execution_succeeded: bool = False
        self._tool_call_count: int = 0
        self._tool_failure_count: int = 0

    def reset(self, **kwargs: Any) -> str | None:
        """Create fresh sandbox for a new coding task.

        Args:
            **kwargs: Dataset columns. Expected: task_description (str),
                initial_files (dict, optional).
        """
        super().reset(**kwargs)
        if self.workdir and os.path.exists(self.workdir):
            shutil.rmtree(self.workdir, ignore_errors=True)

        self.workdir = tempfile.mkdtemp(prefix="carl_sandbox_")
        self._execution_attempted = False
        self._execution_succeeded = False
        self._tool_call_count = 0
        self._tool_failure_count = 0

        initial_files = kwargs.get("initial_files", {})
        if isinstance(initial_files, dict):
            for fname, content in initial_files.items():
                fpath = os.path.join(self.workdir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w") as f:
                    f.write(str(content))

        task = kwargs.get("task_description", "")
        return task if task else None

    def _safe_path(self, path: str) -> str:
        """Resolve path within sandbox. Prevents directory traversal."""
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized -- call reset() first")
        if os.path.isabs(path):
            path = path.lstrip("/")
        resolved = os.path.normpath(os.path.join(self.workdir, path))
        if not (resolved == self.workdir or resolved.startswith(self.workdir + os.sep)):
            raise ValueError(f"Path escapes sandbox: {path}")
        return resolved

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: File path relative to sandbox root.

        Returns:
            The file contents, or an error message if the file does not exist.
        """
        self._tool_call_count += 1
        try:
            fpath = self._safe_path(path)
            with open(fpath) as f:
                result = f.read()
        except FileNotFoundError:
            self._tool_failure_count += 1
            result = f"Error: File not found: {path}"
        except ValueError as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
        self._record_turn("read_file", {"path": path}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox.

        Args:
            path: File path relative to sandbox root.
            content: The content to write.

        Returns:
            Confirmation message with bytes written.
        """
        self._tool_call_count += 1
        try:
            fpath = self._safe_path(path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as f:
                f.write(content)
            result = f"Written {len(content)} bytes to {path}"
        except ValueError as e:
            self._tool_failure_count += 1
            result = f"Error: {e}"
        self._record_turn("write_file", {"path": path}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def execute_code(self, code: str) -> str:
        """Execute Python code in the sandbox.

        Args:
            code: Python source code to execute.

        Returns:
            The stdout/stderr output from execution.
        """
        self._tool_call_count += 1
        self._execution_attempted = True
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized -- call reset() first")
        fpath = os.path.join(self.workdir, "_exec.py")
        with open(fpath, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["python", fpath],
                capture_output=True, text=True, timeout=_EXEC_TIMEOUT,
                cwd=self.workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if proc.returncode == 0:
                self._execution_succeeded = True
                self.reward = 1.0
                result = proc.stdout if proc.stdout.strip() else "(executed successfully, no output)"
            else:
                self._execution_succeeded = False
                self._tool_failure_count += 1
                self.reward = 0.0
                result = f"{proc.stdout}\nError (exit {proc.returncode}):\n{proc.stderr}"
        except subprocess.TimeoutExpired:
            self._execution_succeeded = False
            self._tool_failure_count += 1
            self.reward = 0.0
            result = f"Error: Execution timed out ({_EXEC_TIMEOUT}s limit)"
        except Exception as e:
            self._execution_succeeded = False
            self._tool_failure_count += 1
            self.reward = 0.0
            result = f"Error: {e}"
        self._record_turn("execute_code", {"code_length": len(code)}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def run_shell(self, command: str) -> str:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command to execute.

        Returns:
            The command output (stdout + stderr).
        """
        self._tool_call_count += 1
        self._execution_attempted = True
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized -- call reset() first")
        try:
            proc = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=_EXEC_TIMEOUT,
                cwd=self.workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if proc.returncode == 0:
                self._execution_succeeded = True
                self.reward = 1.0
                result = proc.stdout if proc.stdout.strip() else "(completed, no output)"
            else:
                self._execution_succeeded = False
                self._tool_failure_count += 1
                self.reward = 0.0
                result = f"{proc.stdout}\n(exit {proc.returncode})\n{proc.stderr}"
        except subprocess.TimeoutExpired:
            self._execution_succeeded = False
            self._tool_failure_count += 1
            self.reward = 0.0
            result = f"Error: Command timed out ({_EXEC_TIMEOUT}s limit)"
        except Exception as e:
            self._execution_succeeded = False
            self._tool_failure_count += 1
            self.reward = 0.0
            result = f"Error: {e}"
        self._record_turn("run_shell", {"command": command}, result)  # pyright: ignore[reportPrivateUsage]
        return result

    def __del__(self) -> None:
        try:
            if self.workdir and os.path.exists(self.workdir):
                shutil.rmtree(self.workdir, ignore_errors=True)
        except (TypeError, AttributeError):
            pass
        # Delegate to EnvironmentConnection.__del__ so the underlying
        # connection adapter is closed cleanly on shutdown.
        try:
            super().__del__()
        except (TypeError, AttributeError):
            pass
