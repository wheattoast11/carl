"""Coding sandbox environment — real subprocess execution, file I/O, binary reward.

This is the reference CARL environment. TRL creates one instance per generation,
auto-discovers public methods as tools, handles the multi-turn loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec
from carl_studio.environments.registry import register_environment


@register_environment
class CodingSandboxEnv(BaseEnvironment):
    """Real coding sandbox — subprocess execution, tempdir isolation, binary reward."""

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

    def __init__(self) -> None:
        super().__init__()
        self.workdir: str | None = None
        self._execution_attempted: bool = False
        self._execution_succeeded: bool = False

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
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized — call reset() first")
        if os.path.isabs(path):
            path = path.lstrip("/")
        resolved = os.path.normpath(os.path.join(self.workdir, path))
        if not resolved.startswith(self.workdir):
            raise ValueError(f"Path escapes sandbox: {path}")
        return resolved

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: File path relative to sandbox root.

        Returns:
            The file contents.
        """
        try:
            result = open(self._safe_path(path)).read()
        except FileNotFoundError:
            result = f"Error: File not found: {path}"
        except ValueError as e:
            result = f"Error: {e}"
        self._record_turn("read_file", {"path": path}, result)
        return result

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file in the sandbox.

        Args:
            path: File path relative to sandbox root.
            content: The content to write.

        Returns:
            Confirmation message.
        """
        try:
            fpath = self._safe_path(path)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as f:
                f.write(content)
            result = f"Written {len(content)} bytes to {path}"
        except ValueError as e:
            result = f"Error: {e}"
        self._record_turn("write_file", {"path": path}, result)
        return result

    def execute_code(self, code: str) -> str:
        """Execute Python code in the sandbox.

        Args:
            code: Python source code to execute.

        Returns:
            Execution output (stdout and stderr).
        """
        self._execution_attempted = True
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized — call reset() first")
        fpath = os.path.join(self.workdir, "_exec.py")
        with open(fpath, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["python", fpath],
                capture_output=True, text=True, timeout=10,
                cwd=self.workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if proc.returncode == 0:
                self._execution_succeeded = True
                self.reward = 1.0
                result = proc.stdout if proc.stdout.strip() else "(executed successfully, no output)"
            else:
                self._execution_succeeded = False
                self.reward = 0.0
                result = f"{proc.stdout}\nError (exit {proc.returncode}):\n{proc.stderr}"
        except subprocess.TimeoutExpired:
            self._execution_succeeded = False
            self.reward = 0.0
            result = "Error: Execution timed out (10s limit)"
        except Exception as e:
            self._execution_succeeded = False
            self.reward = 0.0
            result = f"Error: {e}"
        self._record_turn("execute_code", {"code_length": len(code)}, result)
        return result

    def run_shell(self, command: str) -> str:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command to execute.

        Returns:
            Command output.
        """
        self._execution_attempted = True
        if self.workdir is None:
            raise RuntimeError("Sandbox not initialized — call reset() first")
        try:
            proc = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=10,
                cwd=self.workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if proc.returncode == 0:
                self._execution_succeeded = True
                self.reward = 1.0
                result = proc.stdout if proc.stdout.strip() else "(completed, no output)"
            else:
                self._execution_succeeded = False
                self.reward = 0.0
                result = f"{proc.stdout}\n(exit {proc.returncode})\n{proc.stderr}"
        except subprocess.TimeoutExpired:
            self._execution_succeeded = False
            self.reward = 0.0
            result = "Error: Command timed out (10s limit)"
        except Exception as e:
            self._execution_succeeded = False
            self.reward = 0.0
            result = f"Error: {e}"
        self._record_turn("run_shell", {"command": command}, result)
        return result

    def __del__(self) -> None:
        try:
            if self.workdir and os.path.exists(self.workdir):
                shutil.rmtree(self.workdir, ignore_errors=True)
        except (TypeError, AttributeError):
            pass
