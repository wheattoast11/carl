"""Tinker compute backend — managed training API."""
from __future__ import annotations

import os


class TinkerBackend:
    """Train via Tinker's managed API — forward/backward runs remotely."""

    def __init__(self) -> None:
        self._training_client = None
        self._session_id: str | None = None

    @property
    def name(self) -> str:
        return "tinker"

    async def provision(self, hardware: str, timeout: int) -> str:
        # Tinker doesn't need explicit provisioning — it's implicit in create_*_client
        return "tinker_implicit"

    async def execute(self, script: str, **kwargs: object) -> str:
        """For Tinker, we don't execute scripts — we use the API directly.
        This method creates the training client. The actual training loop
        is driven by CARLTrainer calling forward_backward() on this client.
        """
        import tinker

        model_id = kwargs.get("model_id", "")
        lora_rank = kwargs.get("lora_rank", 64)

        service = tinker.ServiceClient()
        self._training_client = service.create_lora_training_client(
            base_model=model_id,
            rank=lora_rank,
        )
        self._session_id = getattr(self._training_client, "session_id", "tinker-session")
        return str(self._session_id)

    async def status(self, job_id: str) -> str:
        if self._training_client is None:
            return "unknown"
        return "running"  # Tinker training is driven by the caller

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        return [f"Tinker session {job_id} — training driven by local loop"]

    async def stop(self, job_id: str) -> None:
        self._training_client = None

    async def teardown(self) -> None:
        self._training_client = None
        self._session_id = None

    @property
    def training_client(self):
        """Access the Tinker training client for forward_backward() calls."""
        return self._training_client
