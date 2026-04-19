"""Axolotl adapter.

Translates ``carl.yaml`` into axolotl's YAML config schema and shells out
to ``axolotl train`` (or ``accelerate launch`` wrapping it). Axolotl reads
config from stdin when the path argument is ``-``.
"""

from __future__ import annotations

import shutil
from typing import Any

import yaml

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


_BINARY = "axolotl"
_INSTALL_HINT = "pip install axolotl"


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

_RL_TRAINER_MAP = {
    "sft": None,  # plain SFT, no rl section
    "dpo": "dpo",
    "grpo": "grpo",
    "kto": "kto",
    "orpo": "orpo",
}


def translate_config(carl_config: dict[str, Any]) -> dict[str, Any]:
    """Map a carl.yaml dict to an axolotl YAML-shaped dict.

    The output dict matches axolotl's schema (see
    https://axolotl.ai/configuration.html) so callers can ``yaml.safe_dump``
    it directly into a file or stdin.
    """
    if not isinstance(carl_config, dict):
        raise AdapterError(
            "carl_config must be a dict",
            code="carl.adapter.translation",
            context={"backend": "axolotl", "type": type(carl_config).__name__},
        )

    base_model = require_str(carl_config, "base_model", backend="axolotl")
    dataset = require_str(carl_config, "dataset_repo", backend="axolotl")

    method = str(carl_config.get("method", "sft")).lower().strip()
    if method not in _RL_TRAINER_MAP:
        raise AdapterError(
            f"unsupported method for axolotl: {method!r}",
            code="carl.adapter.translation",
            context={"backend": "axolotl", "method": method},
        )

    quant = carl_config.get("quantization") or {}
    lora = carl_config.get("lora") or {}

    out: dict[str, Any] = {
        "base_model": base_model,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "datasets": [
            {
                "path": dataset,
                "type": "alpaca" if method == "sft" else "chat_template",
                "split": str(carl_config.get("dataset_split", "train")),
            }
        ],
        "output_dir": f"./outputs/{carl_config.get('run_name', 'carl-axolotl')}",
        "sequence_len": int(carl_config.get("max_length", 512)),
        "sample_packing": False,
        "pad_to_sequence_len": True,
        # ---- adapter ----
        "adapter": "lora",
        "lora_r": int(lora.get("r", 64)),
        "lora_alpha": int(lora.get("alpha", 128)),
        "lora_dropout": float(lora.get("dropout", 0.05)),
        "lora_target_modules": list(
            lora.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
        # ---- quantization ----
        "load_in_8bit": bool(quant.get("load_in_8bit", True))
        and not bool(quant.get("load_in_4bit", False)),
        "load_in_4bit": bool(quant.get("load_in_4bit", False)),
        # ---- trainer ----
        "num_epochs": int(carl_config.get("num_train_epochs", 3)),
        "max_steps": int(carl_config.get("max_steps", -1)),
        "micro_batch_size": int(carl_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(carl_config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(carl_config.get("learning_rate", 2e-5)),
        "warmup_ratio": float(carl_config.get("warmup_ratio", 0.1)),
        "weight_decay": float(carl_config.get("weight_decay", 0.01)),
        "max_grad_norm": float(carl_config.get("max_grad_norm", 1.0)),
        "lr_scheduler": str(carl_config.get("lr_scheduler_type", "cosine")),
        "bf16": bool(carl_config.get("bf16", True)),
        "seed": int(carl_config.get("seed", 42)),
        "hub_model_id": str(carl_config.get("output_repo", "")) or None,
        "hub_strategy": str(carl_config.get("hub_strategy", "every_save")),
    }

    rl_name = _RL_TRAINER_MAP[method]
    if rl_name is not None:
        out["rl"] = rl_name
        if method == "grpo":
            out["num_generations"] = int(carl_config.get("num_generations", 8))
            out["max_completion_length"] = int(carl_config.get("max_completion_length", 512))
            out["beta"] = float(carl_config.get("beta", 0.0))
            out["temperature"] = float(carl_config.get("grpo_temperature", 2.0))

    # Strip keys with None values so the YAML is clean.
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AxolotlAdapter(TrainingConnection):
    spec = ConnectionSpec(
        name="carl.training.axolotl",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.TRAINING,
        direction=ConnectionDirection.EGRESS,
        transport=ConnectionTransport.SUBPROCESS,
        trust=ConnectionTrust.PUBLIC,
    )

    name = "axolotl"

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

        run_id = new_run_id("axolotl")
        run_dir = state_dir(self.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.yaml"
        log_path = run_dir / "run.log"

        yaml_text = yaml.safe_dump(translated, sort_keys=False)
        cfg_path.write_text(yaml_text, encoding="utf-8")

        binary = shutil.which(_BINARY)
        if binary is None:
            raise unavailable(self.name, hint=_INSTALL_HINT)

        # Axolotl: `axolotl train -` reads YAML from stdin. We pipe the
        # translated YAML so the child process needs no config file.
        pid = spawn(
            [binary, "train", "-"],
            log_path=log_path,
            stdin=yaml_text,
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
