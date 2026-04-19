"""Tinker adapter.

Thinking Machines' Tinker exposes a Python API for fine-tuning runs. The
public API surface is still evolving — this adapter validates availability
and translates ``carl.yaml`` into a Tinker-shaped dict, but ``submit()``
deliberately raises a clear :class:`AdapterError` with code
``carl.adapter.tinker_not_implemented`` rather than guessing at an
unstable API.

Scaffolding policy: we do NOT persist any stub state from ``submit()``.
Persisting state that nobody can observe via ``status()``/``logs()`` (the
run_id is never returned to the caller) creates zombie files and misleads
operators. Once Tinker's API stabilizes, flip ``submit()`` to a real
implementation — translation plumbing is already wired up and callers of
``translate()`` continue to work.
"""

from __future__ import annotations

import importlib.util
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
from .protocol import AdapterError, BackendJob
from ._common import (
    cancel_common,
    logs_common,
    require_str,
    status_common,
)


_INSTALL_HINT = (
    "pip install tinker  (see https://thinkingmachines.ai/tinker for current install)"
)

_DOCS_URL = "https://thinkingmachines.ai/tinker"


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

    model = require_str(carl_config, "base_model", backend="tinker")
    dataset = require_str(carl_config, "dataset_repo", backend="tinker")

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


class TinkerAdapter(TrainingConnection):
    spec = ConnectionSpec(
        name="carl.training.tinker",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.TRAINING,
        direction=ConnectionDirection.EGRESS,
        transport=ConnectionTransport.HTTP,
        trust=ConnectionTrust.AUTHENTICATED,
    )

    """Scaffolded Tinker adapter.

    Availability is derived from whether the ``tinker`` module is importable
    (the Python package is installed). ``submit()`` always raises
    :class:`AdapterError` with code ``carl.adapter.tinker_not_implemented``
    — translation is wired up and available via :meth:`translate`, but the
    submission path will remain scaffolded until Tinker's Python API
    stabilizes.
    """

    name = "tinker"

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("tinker") is not None
        except (ImportError, ValueError):
            return False

    def translate(self, carl_config: dict[str, Any]) -> dict[str, Any]:
        return translate_config(carl_config)

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        # Validate translation first so the caller still gets a translation
        # error on malformed configs even though we never submit.
        translated = translate_config(carl_config)
        raise AdapterError(
            "Tinker adapter is scaffolded but not yet implemented — "
            f"see {_DOCS_URL} for the current Python API. Use "
            "TinkerAdapter().translate(carl_config) to render the spec "
            "without submitting.",
            code="carl.adapter.tinker_not_implemented",
            context={
                "backend": self.name,
                "docs_url": _DOCS_URL,
                "install_hint": _INSTALL_HINT,
                "translated_keys": sorted(translated.keys()),
            },
        )

    def status(self, run_id: str) -> BackendJob:
        return status_common(self.name, run_id)

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        return logs_common(self.name, run_id, tail=tail)

    def cancel(self, run_id: str) -> bool:
        return cancel_common(self.name, run_id)
