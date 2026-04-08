"""HuggingFace Jobs compute backend."""
from __future__ import annotations

import os
from carl_studio.compute.protocol import ComputeBackend


class HFJobsBackend:
    """Submit training jobs via huggingface_hub.run_uv_job()."""

    @property
    def name(self) -> str:
        return "hf_jobs"

    async def provision(self, hardware: str, timeout: int) -> str:
        # HF Jobs doesn't require pre-provisioning
        return "hf_jobs_implicit"

    async def execute(self, script: str, **kwargs: object) -> str:
        from huggingface_hub import HfApi, get_token

        api = HfApi()
        hardware = kwargs.get("hardware", "l4x1")
        timeout = kwargs.get("timeout", 14400)

        # Write script to temp file (run_uv_job needs file path)
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            job = api.run_uv_job(
                script=script_path,
                flavor=hardware,
                timeout=timeout,
                env={"PYTHONUNBUFFERED": "1"},
                secrets={"HF_TOKEN": get_token()},
            )
            return job.id
        finally:
            os.unlink(script_path)

    async def status(self, job_id: str) -> str:
        from huggingface_hub import HfApi

        j = HfApi().inspect_job(job_id=job_id)
        return j.status.stage.lower()

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        from huggingface_hub import HfApi

        raw = list(HfApi().fetch_job_logs(job_id=job_id))
        return [str(e)[:300] for e in raw[-tail:]]

    async def stop(self, job_id: str) -> None:
        from huggingface_hub import HfApi

        HfApi().cancel_job(job_id=job_id)

    async def teardown(self) -> None:
        pass  # No persistent resources
