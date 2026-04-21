"""Slime translator.

Sole owner of the ``carl.yaml`` → slime CLI-arg surface. The adapter
delegates here so that upstream arg drift (slime is actively developed
by THUDM / Z.ai) becomes a single-file fix.

Slime's runtime expects three logical argument groups (documented in
``slime/utils/arguments.py`` upstream):

    * Megatron-LM args — tensor/pipeline parallelism, training loop knobs.
    * SGLang args (prefixed ``--sglang-``) — rollout engine config.
    * Slime args — the orchestration layer (sync/async, rollout batch,
      reward hook, advantage estimator, etc.).

The translator produces a typed :class:`SlimeArgs` record. Callers use
:meth:`SlimeArgs.to_cli_args` to flatten it to a subprocess-ready list
or :meth:`SlimeArgs.to_dict` for telemetry / dry-run output.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from .protocol import AdapterError


_SUPPORTED_METHODS: frozenset[str] = frozenset({"sft", "grpo", "distill"})
_SUPPORTED_MODES: frozenset[str] = frozenset({"sync", "async"})
_SUPPORTED_ADVANTAGE: frozenset[str] = frozenset({"grpo", "reinforce", "gae"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class SlimeArgs(BaseModel):
    """Translated slime launch configuration.

    Three dicts of ``name → value`` plus a free-form ``extra`` passthrough
    for advanced users who need to inject flags the translator does not
    know about. Everything stays JSON-serializable so dry-run and state
    files round-trip cleanly.

    **Public JSON Schema.** :meth:`json_schema` returns the Pydantic v2
    JSON Schema of this model. carl.camp's ``POST /api/train/slime/submit``
    consumes the same schema for server-side validation so the schema
    definition has exactly one source of truth (v0.16 commitment).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    megatron: dict[str, Any] = Field(default_factory=dict)
    sglang: dict[str, Any] = Field(default_factory=dict)
    slime: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "megatron": dict(self.megatron),
            "sglang": dict(self.sglang),
            "slime": dict(self.slime),
            "extra": dict(self.extra),
        }

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return the Pydantic v2 JSON Schema for this model.

        Public surface for carl.camp's server-side config validation.
        The schema is a pure data dict — safe to serialize, publish,
        or cache.
        """
        return cls.model_json_schema()

    def to_cli_args(self) -> list[str]:
        """Flatten to a subprocess-ready argument list.

        Conventions matching slime's ``slime/utils/arguments.py``:

          * Megatron flags: ``--<key>`` (snake → kebab).
          * SGLang flags: ``--sglang-<key>`` (snake → kebab).
          * Slime flags: ``--<key>`` (snake → kebab).
          * Boolean True → bare flag, False → omitted entirely (slime
            uses store_true throughout).
          * List values → space-separated ``nargs='+'`` form.

        The ``extra`` dict is appended verbatim (already-CLI key/value
        pairs) so users can bypass the translator when needed.
        """
        parts: list[str] = []
        parts.extend(_emit_group(self.megatron, prefix=""))
        parts.extend(_emit_group(self.sglang, prefix="sglang-"))
        parts.extend(_emit_group(self.slime, prefix=""))
        parts.extend(_emit_group(self.extra, prefix=""))
        return parts


def _emit_group(group: dict[str, Any], *, prefix: str) -> list[str]:
    out: list[str] = []
    for key, value in group.items():
        flag = f"--{prefix}{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                out.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            seq = cast("list[Any] | tuple[Any, ...]", value)
            if not seq:
                continue
            out.append(flag)
            for item in seq:
                out.append(str(item))
            continue
        out.append(flag)
        out.append(str(value))
    return out


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate_config(carl_config: dict[str, Any]) -> SlimeArgs:
    """Translate a ``carl.yaml`` dict into a :class:`SlimeArgs` record.

    Required keys:
      * ``base_model`` — HF model id or local path.
      * ``dataset_repo`` — HF dataset id or local path (slime ``--prompt-data``).
      * ``method`` — one of ``sft`` / ``grpo`` / ``distill``.

    Optional keys honoured (with defaults):
      * ``mode`` — ``sync`` (default) or ``async``.
      * ``max_length`` — rollout response length cap.
      * ``num_generations`` — samples per prompt (GRPO only).
      * ``grpo_temperature`` — rollout temperature.
      * ``per_device_train_batch_size`` / ``gradient_accumulation_steps``.
      * ``num_train_epochs`` / ``max_steps`` / ``learning_rate`` / ``seed``.
      * ``slime.tensor_parallel`` / ``slime.pipeline_parallel`` /
        ``slime.expert_parallel`` — Megatron parallelism dimensions.
      * ``slime.rollout_engine_tp`` — SGLang tensor parallelism.
      * ``slime.advantage_estimator`` — ``grpo`` (default) / ``reinforce`` / ``gae``.
      * ``slime.disaggregated`` — bool, turns on PD disaggregation.
      * ``slime.reward_fn`` — dotted path to a user reward callable
        (slime consumes this via its custom-reward hook).
      * ``slime.extra_args`` — dict of free-form overrides appended verbatim.
      * ``output_repo`` / ``run_name`` — standard carl fields.

    Raises:
        AdapterError: with code ``"carl.adapter.translation"`` on any
            validation failure, ``"carl.adapter.missing_required"`` when
            a required key is absent.
    """
    base_model = _require_str(carl_config, "base_model")
    dataset = _require_str(carl_config, "dataset_repo")
    method = str(carl_config.get("method", "grpo")).lower().strip()
    if method not in _SUPPORTED_METHODS:
        raise AdapterError(
            f"unsupported method for slime: {method!r}",
            code="carl.adapter.translation",
            context={
                "backend": "slime",
                "method": method,
                "supported": sorted(_SUPPORTED_METHODS),
            },
        )

    mode = str(carl_config.get("mode", "sync")).lower().strip()
    if mode not in _SUPPORTED_MODES:
        raise AdapterError(
            f"unsupported slime mode: {mode!r}",
            code="carl.adapter.translation",
            context={
                "backend": "slime",
                "mode": mode,
                "supported": sorted(_SUPPORTED_MODES),
            },
        )

    raw_slime_block = carl_config.get("slime")
    if isinstance(raw_slime_block, dict):
        slime_block: dict[str, Any] = cast(dict[str, Any], raw_slime_block)
    else:
        slime_block = {}

    advantage = str(slime_block.get("advantage_estimator", "grpo")).lower().strip()
    if advantage not in _SUPPORTED_ADVANTAGE:
        raise AdapterError(
            f"unsupported slime advantage estimator: {advantage!r}",
            code="carl.adapter.translation",
            context={
                "backend": "slime",
                "advantage_estimator": advantage,
                "supported": sorted(_SUPPORTED_ADVANTAGE),
            },
        )

    tp = int(slime_block.get("tensor_parallel", 1))
    pp = int(slime_block.get("pipeline_parallel", 1))
    ep = int(slime_block.get("expert_parallel", 1))
    rollout_tp = int(slime_block.get("rollout_engine_tp", tp))

    micro_batch = int(carl_config.get("per_device_train_batch_size", 1))
    grad_accum = int(carl_config.get("gradient_accumulation_steps", 1))
    global_batch = int(
        carl_config.get("global_batch_size", max(1, micro_batch * grad_accum))
    )

    megatron: dict[str, Any] = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "seq_length": int(carl_config.get("max_length", 2048)),
        "micro_batch_size": micro_batch,
        "global_batch_size": global_batch,
        "lr": float(carl_config.get("learning_rate", 1e-6)),
        "seed": int(carl_config.get("seed", 42)),
        "bf16": bool(carl_config.get("bf16", True)),
    }
    if "max_steps" in carl_config:
        steps = int(carl_config["max_steps"])
        if steps > 0:
            megatron["train_iters"] = steps

    sglang: dict[str, Any] = {
        "model_path": base_model,
        "tp_size": rollout_tp,
        "dtype": str(carl_config.get("sglang_dtype", "bfloat16")),
    }

    slime_args: dict[str, Any] = {
        "model": base_model,
        "prompt_data": dataset,
        "prompt_split": str(carl_config.get("dataset_split", "train")),
        "run_name": str(carl_config.get("run_name", "carl-slime")),
        "output_dir": str(
            carl_config.get(
                "output_dir", f"./outputs/{carl_config.get('run_name', 'carl-slime')}"
            )
        ),
        "mode": mode,
        "advantage_estimator": advantage,
        "rollout_max_response_len": int(
            carl_config.get("max_completion_length", carl_config.get("max_length", 2048))
        ),
        "rollout_temperature": float(carl_config.get("grpo_temperature", 1.0)),
        "rollout_batch_size": int(slime_block.get("rollout_batch_size", 32)),
        "n_samples_per_prompt": int(carl_config.get("num_generations", 8)),
        "num_rollout": int(slime_block.get("num_rollout", 1000)),
    }

    if bool(slime_block.get("disaggregated", False)):
        slime_args["disaggregated"] = True

    num_epochs_raw = carl_config.get("num_train_epochs")
    if num_epochs_raw is not None:
        slime_args["num_epochs"] = int(num_epochs_raw)

    output_repo_raw = carl_config.get("output_repo")
    if output_repo_raw:
        slime_args["hub_model_id"] = str(output_repo_raw)

    reward_fn_raw = slime_block.get("reward_fn")
    if reward_fn_raw:
        slime_args["custom_reward_fn"] = str(reward_fn_raw)

    reward_model_raw = slime_block.get("reward_model")
    if reward_model_raw:
        slime_args["reward_model"] = str(reward_model_raw)

    raw_extra = slime_block.get("extra_args")
    if isinstance(raw_extra, dict):
        extra: dict[str, Any] = cast(dict[str, Any], raw_extra)
    else:
        extra = {}

    return SlimeArgs(
        megatron=megatron,
        sglang=sglang,
        slime=slime_args,
        extra=extra,
    )


def _require_str(cfg: dict[str, Any], key: str) -> str:
    try:
        raw = cfg[key]
    except KeyError as exc:
        raise AdapterError(
            f"carl config is missing {key!r}",
            code="carl.adapter.missing_required",
            context={"backend": "slime", "key": key},
            cause=exc,
        ) from exc
    value = str(raw).strip()
    if not value:
        raise AdapterError(
            f"carl config {key!r} is empty",
            code="carl.adapter.translation",
            context={"backend": "slime", "key": key},
        )
    return value


__all__ = ["SlimeArgs", "translate_config"]
