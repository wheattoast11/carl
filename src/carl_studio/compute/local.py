"""Local compute backend — direct GPU execution."""
from __future__ import annotations

import os
import subprocess
import tempfile


class LocalBackend:
    """Run training scripts locally via subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._script_path: str | None = None

    @property
    def name(self) -> str:
        return "local"

    async def provision(self, hardware: str, timeout: int) -> str:
        return "local"

    async def execute(self, script: str, **kwargs: object) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        self._script_path = script_path

        self._process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ},
        )
        return f"local-{self._process.pid}"

    async def status(self, job_id: str) -> str:
        if self._process is None:
            return "unknown"
        poll = self._process.poll()
        if poll is None:
            return "running"
        return "completed" if poll == 0 else "error"

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        if self._process and self._process.stdout:
            lines = []
            for line in self._process.stdout:
                lines.append(line.rstrip())
                if len(lines) > tail:
                    lines.pop(0)
            return lines
        return []

    async def stop(self, job_id: str) -> None:
        if self._process:
            self._process.terminate()

    async def teardown(self) -> None:
        if self._process:
            self._process.kill()
            self._process = None
        if self._script_path and os.path.exists(self._script_path):
            os.unlink(self._script_path)
            self._script_path = None
