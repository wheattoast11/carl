"""TRL adapter.

carl-studio already ships its own TRL-based trainer at
``carl_studio.training.trainer.CARLTrainer``. The TRL adapter is a thin
wrapper that validates the carl config and delegates to the in-process
trainer. It is always reported as available.
"""

from __future__ import annotations

from typing import Any

from .protocol import AdapterError, BackendJob, BackendStatus
from ._common import (
    JobState,
    load_state,
    logs_common,
    new_run_id,
    now_iso,
    save_state,
    status_common,
)


class TRLAdapter:
    """Default adapter that routes to carl-studio's built-in CARLTrainer.

    The actual training is executed by the existing in-process trainer; the
    adapter records a :class:`JobState` so the same ``carl run show`` +
    ``carl run logs`` surface works across backends.
    """

    name = "trl"

    def available(self) -> bool:
        """TRL is a first-party dep declared in ``carl-studio[training]``.

        We report True unconditionally — the trainer will surface a clean
        ImportError with an install hint if the extras weren't installed.
        """
        return True

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        if not isinstance(carl_config, dict):
            raise AdapterError(
                "carl_config must be a dict",
                code="carl.adapter.translation",
                context={"backend": self.name, "type": type(carl_config).__name__},
            )

        run_id = new_run_id("trl")
        state = JobState(
            run_id=run_id,
            backend=self.name,
            status=BackendStatus.PENDING,
            submitted_at=now_iso(),
            config=dict(carl_config),
            raw={"engine": "carl_studio.training.trainer.CARLTrainer"},
        )
        save_state(state)
        return state.to_job()

    def status(self, run_id: str) -> BackendJob:
        return status_common(self.name, run_id)

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        return logs_common(self.name, run_id, tail=tail)

    def cancel(self, run_id: str) -> bool:
        """Mark the run canceled. In-process cancellation of CARLTrainer
        requires cooperation from the caller (anyio cancel scope); we record
        the intent in state so a supervising loop can honor it.
        """
        state = load_state(self.name, run_id)
        if BackendStatus.is_terminal(state.status):
            return False
        state.status = BackendStatus.CANCELED
        state.completed_at = now_iso()
        save_state(state)
        return True

    # ------------------------------------------------------------------
    # Extended — used by ``carl train --backend trl`` to actually run
    # ------------------------------------------------------------------

    def run_sync(self, carl_config: dict[str, Any]) -> BackendJob:
        """Submit and block until the CARLTrainer finishes.

        This is the entry point used by the CLI when ``--backend trl`` is
        selected (the default). It lazy-imports the trainer to keep this
        module import-time lightweight.
        """
        from carl_studio.types.config import TrainingConfig

        try:
            training_config = TrainingConfig(**carl_config)
        except Exception as exc:
            raise AdapterError(
                f"invalid carl config for TRL backend: {exc}",
                code="carl.adapter.translation",
                context={"backend": self.name},
                cause=exc,
            ) from exc

        job = self.submit(carl_config)
        state = load_state(self.name, job.run_id)
        state.status = BackendStatus.RUNNING
        save_state(state)

        try:
            import anyio

            from carl_studio.training.trainer import CARLTrainer

            trainer = CARLTrainer(training_config)
            run = anyio.run(trainer.train)
        except ImportError as exc:
            state.status = BackendStatus.FAILED
            state.completed_at = now_iso()
            state.raw["error"] = f"training extras missing: {exc}"
            save_state(state)
            raise AdapterError(
                "training extras not installed — install: pip install 'carl-studio[training]'",
                code="carl.adapter.unavailable",
                context={"backend": self.name, "install_hint": "carl-studio[training]"},
                cause=exc,
            ) from exc
        except Exception as exc:
            state.status = BackendStatus.FAILED
            state.completed_at = now_iso()
            state.raw["error"] = str(exc)
            save_state(state)
            raise AdapterError(
                f"TRL training failed: {exc}",
                code="carl.adapter.submit",
                context={"backend": self.name, "run_id": job.run_id},
                cause=exc,
            ) from exc

        state.status = BackendStatus.COMPLETED
        state.completed_at = now_iso()
        state.raw["local_run_id"] = getattr(run, "id", None)
        state.raw["hub_job_id"] = getattr(run, "hub_job_id", None)
        save_state(state)
        return state.to_job()
