"""Tinker adapter.

Thinking Machines' Tinker exposes a Python API for fine-tuning runs. The
public API surface is still evolving; this adapter validates availability
and translates carl.yaml into a Tinker-shaped dict, but raises a clear
error from ``submit()`` pointing at the install+API docs rather than
guessing at an unstable API.

Once the Tinker API stabilizes, swap the ``NotImplementedError`` branch
in ``submit()`` for a real call — translation + state machinery is
already wired up.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .protocol import AdapterError, BackendJob, BackendStatus
from ._common import (
    JobState,
    cancel_pid,
    load_state,
    new_run_id,
    now_iso,
    refresh_pid_status,
    save_state,
    state_dir,
    tail_log,
    unavailable,
)


_INSTALL_HINT = (
    "pip install tinker  (see https://thinkingmachines.ai/tinker for current install)"
)


def translate_config(carl_config: dict[str, Any]) -> dict[str, Any]:
    """Map a carl.yaml dict to a tinker-style training spec.

    The shape here tracks the documented public surface: a
    ``TrainingClient`` consumes ``model``, ``dataset``, ``method``, and
    hyperparams. Unknown fields are preserved verbatim under ``extras``.
    """
    if not isinstance(carl_config, dict):
        raise AdapterError(
            "carl_config must be a dict",
            code="carl.adapter.translation",
            context={"backend": "tinker", "type": type(carl_config).__name__},
        )

    try:
        model = str(carl_config["base_model"]).strip()
    except KeyError as exc:
        raise AdapterError(
            "carl config is missing 'base_model'",
            code="carl.adapter.translation",
            context={"backend": "tinker"},
            cause=exc,
        ) from exc

    try:
        dataset = str(carl_config["dataset_repo"]).strip()
    except KeyError as exc:
        raise AdapterError(
            "carl config is missing 'dataset_repo'",
            code="carl.adapter.translation",
            context={"backend": "tinker"},
            cause=exc,
        ) from exc

    method = str(carl_config.get("method", "sft")).lower().strip()
    if method not in {"sft", "dpo", "grpo"}:
        raise AdapterError(
            f"unsupported method for tinker: {method!r}",
            code="carl.adapter.translation",
            context={"backend": "tinker", "method": method},
        )

    lora = carl_config.get("lora") or {}
    return {
        "model": model,
        "dataset": dataset,
        "split": str(carl_config.get("dataset_split", "train")),
        "method": method,
        "hyperparameters": {
            "learning_rate": float(carl_config.get("learning_rate", 2e-5)),
            "batch_size": int(carl_config.get("per_device_train_batch_size", 1)),
            "gradient_accumulation_steps": int(
                carl_config.get("gradient_accumulation_steps", 8)
            ),
            "epochs": int(carl_config.get("num_train_epochs", 3)),
            "max_steps": int(carl_config.get("max_steps", -1)),
            "seed": int(carl_config.get("seed", 42)),
        },
        "adapter": {
            "type": "lora",
            "r": int(lora.get("r", 64)),
            "alpha": int(lora.get("alpha", 128)),
            "dropout": float(lora.get("dropout", 0.05)),
        },
        "output": {
            "push_to_hub": bool(carl_config.get("push_to_hub", True)),
            "repo_id": str(carl_config.get("output_repo", "")) or None,
        },
        "extras": {
            k: v
            for k, v in carl_config.items()
            if k
            not in {
                "base_model",
                "dataset_repo",
                "dataset_split",
                "method",
                "learning_rate",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "max_steps",
                "seed",
                "lora",
                "push_to_hub",
                "output_repo",
            }
        },
    }


class TinkerAdapter:
    name = "tinker"

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("tinker") is not None
        except (ImportError, ValueError):
            return False

    def translate(self, carl_config: dict[str, Any]) -> dict[str, Any]:
        return translate_config(carl_config)

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        translated = translate_config(carl_config)

        if not self.available():
            raise unavailable(self.name, hint=_INSTALL_HINT)

        # Tinker's Python API is still stabilizing. We persist state and a
        # clear marker so operators know the run was accepted at the
        # adapter layer but needs a manual kick via the Tinker SDK.
        run_id = new_run_id("tinker")
        run_dir = state_dir(self.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "run.log"
        log_path.write_text(
            "tinker adapter: translation recorded; submission to the Tinker "
            "API is not yet wired. Update carl_studio.adapters.tinker_adapter "
            "once the API surface stabilizes.\n",
            encoding="utf-8",
        )

        state = JobState(
            run_id=run_id,
            backend=self.name,
            status=BackendStatus.PENDING,
            submitted_at=now_iso(),
            log_path=str(log_path),
            config=dict(carl_config),
            raw={"translated": translated, "note": "tinker API wire-up pending"},
        )
        save_state(state)

        raise AdapterError(
            "tinker submission path is not yet implemented — translation "
            "was recorded under " + str(run_dir),
            code="carl.adapter.submit",
            context={
                "backend": self.name,
                "run_id": run_id,
                "run_dir": str(run_dir),
            },
        )

    def status(self, run_id: str) -> BackendJob:
        state = load_state(self.name, run_id)
        state = refresh_pid_status(state)
        save_state(state)
        return state.to_job()

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        state = load_state(self.name, run_id)
        return tail_log(
            Path(state.log_path) if state.log_path else None,
            tail,
        )

    def cancel(self, run_id: str) -> bool:
        state = load_state(self.name, run_id)
        if BackendStatus.is_terminal(state.status):
            return False
        cancelled = cancel_pid(state)
        if not cancelled:
            state.status = BackendStatus.CANCELED
            state.completed_at = now_iso()
        save_state(state)
        return True
