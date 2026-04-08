"""Compute backend protocol — all backends implement this."""

from __future__ import annotations

from typing import Protocol, AsyncIterator, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """Abstraction over compute providers. Same config runs anywhere."""

    @property
    def name(self) -> str: ...

    async def provision(self, hardware: str, timeout: int) -> str:
        """Provision compute. Returns a session/pod ID."""
        ...

    async def execute(self, script: str, **kwargs: object) -> str:
        """Execute a training script on provisioned compute. Returns job ID."""
        ...

    async def status(self, job_id: str) -> str:
        """Get job status: running, completed, error."""
        ...

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        """Get recent log lines."""
        ...

    async def stop(self, job_id: str) -> None:
        """Cancel a running job."""
        ...

    async def teardown(self) -> None:
        """Release provisioned compute."""
        ...
