"""Atropos adapter.

Nous Research's Atropos is a multi-environment RL training stack with its
own CLI. This adapter shells out to ``atropos`` with a translated config
JSON piped on stdin.
"""

from __future__ import annotations

import json
import shutil
from typing import Any

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)

from .connection import TrainingConnection
from .protocol import AdapterError, BackendJob, BackendStatus
from ._common import (
    JobState,
    cancel_common,
    logs_common,
    new_run_id,
    now_iso,
    require_str,
    save_state,
    spawn,
    state_dir,
    status_common,
    unavailable,
)


_BINARY = "atropos"
_INSTALL_HINT = "pip install atropos"


def translate_config(carl_config: dict[str, Any]) -> dict[str, Any]:
    """Map a carl.yaml dict to an atropos-style training spec."""
    if not isinstance(carl_config, dict):
        raise AdapterError(
            "carl_config must be a dict",
            code="carl.adapter.translation",
            context={"backend": "atropos", "type": type(carl_config).__name__},
        )

    model = require_str(carl_config, "base_model", backend="atropos")

    method = str(carl_config.get("method", "grpo")).lower().strip()
    # Atropos is primarily RL-oriented; SFT is still accepted as a warmup path.
    if method not in {"sft", "grpo", "dpo"}:
        raise AdapterError(
            f"unsupported method for atropos: {method!r}",
            code="carl.adapter.translation",
            context={"backend": "atropos", "method": method},
        )

    dataset = str(carl_config.get("dataset_repo", "")).strip()

    lora = carl_config.get("lora") or {}
    return {
        "model": model,
        "algorithm": method,
        "dataset": dataset or None,
        "environments": list(carl_config.get("environments", [])),
        "num_generations": int(carl_config.get("num_generations", 8)),
        "max_completion_length": int(carl_config.get("max_completion_length", 512)),
        "temperature": float(carl_config.get("grpo_temperature", 1.0)),
        "learning_rate": float(carl_config.get("learning_rate", 2e-5)),
        "max_steps": int(carl_config.get("max_steps", 1000)),
        "batch_size": int(carl_config.get("per_device_train_batch_size", 1)),
        "seed": int(carl_config.get("seed", 42)),
        "lora": {
            "r": int(lora.get("r", 64)),
            "alpha": int(lora.get("alpha", 128)),
            "dropout": float(lora.get("dropout", 0.05)),
        },
        "hub_repo_id": str(carl_config.get("output_repo", "")) or None,
    }


class AtroposAdapter(TrainingConnection):
    spec = ConnectionSpec(
        name="carl.training.atropos",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.TRAINING,
        direction=ConnectionDirection.EGRESS,
        transport=ConnectionTransport.SUBPROCESS,
        trust=ConnectionTrust.PUBLIC,
    )

    name = "atropos"

    def available(self) -> bool:
        try:
            return shutil.which(_BINARY) is not None
        except Exception:
            return False

    def translate(self, carl_config: dict[str, Any]) -> dict[str, Any]:
        return translate_config(carl_config)

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        translated = translate_config(carl_config)

        if not self.available():
            raise unavailable(self.name, hint=_INSTALL_HINT)

        binary = shutil.which(_BINARY)
        if binary is None:
            raise unavailable(self.name, hint=_INSTALL_HINT)

        run_id = new_run_id("atropos")
        run_dir = state_dir(self.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"
        log_path = run_dir / "run.log"
        payload = json.dumps(translated, indent=2)
        cfg_path.write_text(payload, encoding="utf-8")

        pid = spawn(
            [binary, "train", "--config", str(cfg_path)],
            log_path=log_path,
        )

        state = JobState(
            run_id=run_id,
            backend=self.name,
            status=BackendStatus.RUNNING,
            submitted_at=now_iso(),
            pid=pid,
            log_path=str(log_path),
            config=dict(carl_config),
            raw={"config_path": str(cfg_path), "translated": translated},
        )
        save_state(state)
        return state.to_job()

    def status(self, run_id: str) -> BackendJob:
        return status_common(self.name, run_id)

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        return logs_common(self.name, run_id, tail=tail)

    def cancel(self, run_id: str) -> bool:
        return cancel_common(self.name, run_id)
