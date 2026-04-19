"""Unsloth adapter.

Unsloth provides its own fast-training stack with 4-bit quantization and
patched trainers. We translate ``carl.yaml`` into a short Python entrypoint
that constructs ``FastLanguageModel`` + the appropriate TRL trainer, write
it to a temp script, and launch it with the current Python interpreter.

Import of ``unsloth`` is lazy — ``available()`` uses ``importlib.util``
only.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from textwrap import dedent
from typing import Any

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


_SUPPORTED_METHODS: frozenset[str] = frozenset({"sft", "grpo"})


# ---------------------------------------------------------------------------
# Translation: carl.yaml dict -> unsloth entrypoint script
# ---------------------------------------------------------------------------


def translate_config(carl_config: dict[str, Any]) -> dict[str, Any]:
    """Map a carl.yaml dict to unsloth entrypoint parameters.

    Raises:
        AdapterError: when required fields are missing.
    """
    if not isinstance(carl_config, dict):
        raise AdapterError(
            "carl_config must be a dict",
            code="carl.adapter.translation",
            context={"backend": "unsloth", "type": type(carl_config).__name__},
        )

    model = require_str(carl_config, "base_model", backend="unsloth")
    dataset = require_str(carl_config, "dataset_repo", backend="unsloth")

    method = str(carl_config.get("method", "sft")).lower().strip()
    if method not in _SUPPORTED_METHODS:
        raise AdapterError(
            f"unsupported method for unsloth: {method!r}",
            code="carl.adapter.translation",
            context={
                "backend": "unsloth",
                "method": method,
                "supported": sorted(_SUPPORTED_METHODS),
            },
        )

    quant = carl_config.get("quantization") or {}
    lora = carl_config.get("lora") or {}

    # Mutually exclusive quantization selection. Precedence:
    #   1. explicit load_in_4bit=True wins,
    #   2. else load_in_8bit (default True) if truthy,
    #   3. else neither (full-precision).
    if quant.get("load_in_4bit"):
        load_in_4bit, load_in_8bit = True, False
    elif quant.get("load_in_8bit", True):
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    translated: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "dataset_split": str(carl_config.get("dataset_split", "train")),
        "method": method,
        "max_seq_length": int(carl_config.get("max_length", 512)),
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "lora_r": int(lora.get("r", 64)),
        "lora_alpha": int(lora.get("alpha", 128)),
        "lora_dropout": float(lora.get("dropout", 0.05)),
        "target_modules": list(
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
        "learning_rate": float(carl_config.get("learning_rate", 2e-5)),
        "num_train_epochs": int(carl_config.get("num_train_epochs", 3)),
        "max_steps": int(carl_config.get("max_steps", -1)),
        "per_device_train_batch_size": int(carl_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(carl_config.get("gradient_accumulation_steps", 8)),
        "seed": int(carl_config.get("seed", 42)),
        "output_repo": str(carl_config.get("output_repo", "")),
        "run_name": str(carl_config.get("run_name", "carl-unsloth")),
        "fast_inference": bool(carl_config.get("fast_inference", False)),
    }
    return translated


# ---------------------------------------------------------------------------
# Entrypoint template
# ---------------------------------------------------------------------------

_ENTRYPOINT_TEMPLATE = dedent(
    '''\
    """Auto-generated Unsloth training entrypoint (carl-studio adapter).

    This file is created by UnslothAdapter.submit() and executed as a
    subprocess. It is self-contained so the spawned process does not depend
    on carl-studio being importable in its environment.
    """
    import json
    import os
    import sys

    CFG = json.loads(os.environ["CARL_UNSLOTH_CONFIG"])

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print("unsloth is not installed in this env:", exc, file=sys.stderr)
        sys.exit(2)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CFG["model"],
        max_seq_length=CFG["max_seq_length"],
        load_in_4bit=CFG["load_in_4bit"],
        load_in_8bit=CFG["load_in_8bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["target_modules"],
    )

    from datasets import load_dataset
    ds = load_dataset(CFG["dataset"], split=CFG["dataset_split"])

    if CFG["method"] == "sft":
        from trl import SFTTrainer, SFTConfig
        args = SFTConfig(
            output_dir="outputs",
            num_train_epochs=CFG["num_train_epochs"],
            max_steps=CFG["max_steps"],
            per_device_train_batch_size=CFG["per_device_train_batch_size"],
            gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
            learning_rate=CFG["learning_rate"],
            seed=CFG["seed"],
            run_name=CFG["run_name"],
        )
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=args)
    elif CFG["method"] == "grpo":
        from trl import GRPOTrainer, GRPOConfig
        args = GRPOConfig(
            output_dir="outputs",
            num_train_epochs=CFG["num_train_epochs"],
            max_steps=CFG["max_steps"],
            per_device_train_batch_size=CFG["per_device_train_batch_size"],
            gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
            learning_rate=CFG["learning_rate"],
            seed=CFG["seed"],
            run_name=CFG["run_name"],
        )
        trainer = GRPOTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=args)
    else:
        print(f"unsloth adapter: method {CFG['method']} not implemented", file=sys.stderr)
        sys.exit(3)

    trainer.train()

    if CFG.get("output_repo"):
        try:
            model.push_to_hub(CFG["output_repo"])
        except Exception as exc:
            print(f"push_to_hub failed: {exc}", file=sys.stderr)
    '''
)


class UnslothAdapter:
    """Adapter that launches Unsloth via a generated Python entrypoint."""

    name = "unsloth"
    _INSTALL_HINT = "pip install unsloth"

    def available(self) -> bool:
        try:
            return importlib.util.find_spec("unsloth") is not None
        except (ImportError, ValueError):
            return False

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        translated = translate_config(carl_config)

        if not self.available():
            raise unavailable(self.name, hint=self._INSTALL_HINT)

        run_id = new_run_id("unsloth")
        run_dir = state_dir(self.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / "train.py"
        log_path = run_dir / "run.log"
        script_path.write_text(_ENTRYPOINT_TEMPLATE, encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        import json as _json

        env["CARL_UNSLOTH_CONFIG"] = _json.dumps(translated)

        pid = spawn(
            [sys.executable, str(script_path)],
            log_path=log_path,
            env=env,
        )

        state = JobState(
            run_id=run_id,
            backend=self.name,
            status=BackendStatus.RUNNING,
            submitted_at=now_iso(),
            pid=pid,
            log_path=str(log_path),
            config=dict(carl_config),
            raw={"translated": translated, "script_path": str(script_path)},
        )
        save_state(state)
        return state.to_job()

    def translate(self, carl_config: dict[str, Any]) -> dict[str, Any]:
        """Public helper: translate carl.yaml to the unsloth param dict.

        Useful for ``--dry-run`` surfaces so the CLI can display the
        translated config without submitting a job.
        """
        return translate_config(carl_config)

    def status(self, run_id: str) -> BackendJob:
        return status_common(self.name, run_id)

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        return logs_common(self.name, run_id, tail=tail)

    def cancel(self, run_id: str) -> bool:
        return cancel_common(self.name, run_id)
