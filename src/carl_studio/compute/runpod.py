"""RunPod compute backend — pod-based GPU provisioning."""
from __future__ import annotations

import os


class RunPodBackend:
    """Create RunPod GPU pods, execute training scripts via SSH."""

    def __init__(self) -> None:
        self._pod_id: str | None = None

    @property
    def name(self) -> str:
        return "runpod"

    async def provision(self, hardware: str, timeout: int) -> str:
        import runpod

        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod.api_key:
            raise ValueError("RUNPOD_API_KEY environment variable required")

        gpu_map = {
            "l4x1": "NVIDIA L4",
            "a10g-large": "NVIDIA A10G",
            "a100-large": "NVIDIA A100 80GB",
            "h100": "NVIDIA H100 80GB HBM3",
        }
        gpu_type = gpu_map.get(hardware, hardware)

        pod = runpod.create_pod(
            name="carl-studio-training",
            image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            gpu_type_id=gpu_type,
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=20,
            ports="22/tcp",
            env={
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            },
        )
        self._pod_id = pod["id"]
        return self._pod_id

    async def execute(self, script: str, **kwargs: object) -> str:
        if not self._pod_id:
            raise RuntimeError("Must provision() before execute()")
        # For now, return pod_id as job_id — actual SSH exec is Phase 2.6
        return self._pod_id

    async def status(self, job_id: str) -> str:
        import runpod

        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        pod = runpod.get_pod(job_id)
        status_map = {
            "RUNNING": "running",
            "EXITED": "completed",
            "ERROR": "error",
        }
        return status_map.get(pod.get("desiredStatus", ""), "unknown")

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        return [f"RunPod pod {job_id} — SSH into pod for logs"]

    async def stop(self, job_id: str) -> None:
        import runpod

        runpod.api_key = os.getenv("RUNPOD_API_KEY")
        runpod.stop_pod(job_id)

    async def teardown(self) -> None:
        if self._pod_id:
            import runpod

            runpod.api_key = os.getenv("RUNPOD_API_KEY")
            runpod.terminate_pod(self._pod_id)
            self._pod_id = None
