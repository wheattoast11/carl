"""Prime Intellect compute backend — GPU marketplace."""
from __future__ import annotations

import os
import subprocess


class PrimeBackend:
    """Provision GPUs via Prime Intellect marketplace."""

    def __init__(self) -> None:
        self._pod_id: str | None = None

    @property
    def name(self) -> str:
        return "prime"

    async def provision(self, hardware: str, timeout: int) -> str:
        gpu_map = {
            "l4x1": "L4",
            "a10g-large": "A10G",
            "a100-large": "A100_80GB",
            "h100": "H100_80GB",
        }
        gpu_type = gpu_map.get(hardware, "H100_80GB")

        # Use prime CLI — it handles auth via stored API key
        result = subprocess.run(
            ["prime", "pods", "create", "--gpu-type", gpu_type, "--name", "carl-studio"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"prime pods create failed: {result.stderr}")

        # Parse pod ID from output
        self._pod_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
        return self._pod_id

    async def execute(self, script: str, **kwargs: object) -> str:
        if not self._pod_id:
            raise RuntimeError("Must provision() before execute()")
        return self._pod_id

    async def status(self, job_id: str) -> str:
        result = subprocess.run(
            ["prime", "pods", "status", job_id],
            capture_output=True, text=True, timeout=30,
        )
        if "running" in result.stdout.lower():
            return "running"
        elif "terminated" in result.stdout.lower():
            return "completed"
        return "unknown"

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        return [f"Prime pod {job_id} — SSH via: prime pods ssh {job_id}"]

    async def stop(self, job_id: str) -> None:
        subprocess.run(["prime", "pods", "terminate", job_id], timeout=30)

    async def teardown(self) -> None:
        if self._pod_id:
            subprocess.run(["prime", "pods", "terminate", self._pod_id], timeout=30)
            self._pod_id = None
