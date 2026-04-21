"""Tests for the slime adapter.

The real slime / Megatron / SGLang stack is not installed in CI. These
tests exercise:

  * translation of ``carl.yaml`` → :class:`SlimeArgs` and its CLI form,
  * availability detection with patched ``importlib.util.find_spec``,
  * :class:`AdapterError` code values on every documented failure path,
  * registry inclusion (``get_adapter("slime")`` / ``list_adapters``).

Every test isolates adapter state via the autouse ``CARL_ADAPTER_STATE_DIR``
fixture so ``submit`` does not pollute the real ``~/.carl``.
"""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import pytest

from carl_studio.adapters import (
    AdapterError,
    BackendStatus,
    SlimeAdapter,
    get_adapter,
    list_adapters,
)
from carl_studio.adapters.slime_adapter import (
    resolve_entry,
    resolve_launch_cmd,
)
from carl_studio.adapters.slime_translator import (
    SlimeArgs,
    translate_config,
)


def _none_spec(_name: str) -> ModuleSpec | None:
    return None


def _present_spec(_name: str) -> ModuleSpec | None:
    # find_spec() callers only check ``is None``. A bare ModuleSpec with a
    # made-up loader satisfies both "truthy" and typing — ModuleType is the
    # wrong type for this API.
    return ModuleSpec(_name, loader=None)


def _boom_spec(_name: str) -> ModuleSpec | None:
    raise ValueError("find_spec exploded")


_SAMPLE_CFG: dict[str, Any] = {
    "run_name": "slime-demo",
    "base_model": "Qwen/Qwen3-7B",
    "output_repo": "acme/demo-slime",
    "dataset_repo": "acme/grpo-prompts",
    "dataset_split": "train",
    "method": "grpo",
    "mode": "sync",
    "max_length": 4096,
    "max_completion_length": 1024,
    "num_generations": 8,
    "grpo_temperature": 1.2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-6,
    "num_train_epochs": 1,
    "max_steps": 1000,
    "seed": 7,
    "bf16": True,
    "slime": {
        "tensor_parallel": 4,
        "pipeline_parallel": 2,
        "expert_parallel": 1,
        "rollout_engine_tp": 2,
        "advantage_estimator": "grpo",
        "rollout_batch_size": 64,
        "num_rollout": 250,
        "reward_fn": "my.module:custom_reward",
        "extra_args": {"debug_dump_trajectories": True},
    },
}


@pytest.fixture(autouse=True)
def _adapters_state_tmp(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CARL_ADAPTER_STATE_DIR", str(tmp_path / "adapters"))


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def test_translate_config_returns_slime_args_with_all_three_groups() -> None:
    args = translate_config(_SAMPLE_CFG)

    assert isinstance(args, SlimeArgs)
    assert args.megatron["tensor_model_parallel_size"] == 4
    assert args.megatron["pipeline_model_parallel_size"] == 2
    assert args.megatron["seq_length"] == 4096
    assert args.megatron["micro_batch_size"] == 1
    assert args.megatron["global_batch_size"] == 16
    assert args.megatron["train_iters"] == 1000
    assert args.megatron["bf16"] is True

    assert args.sglang["model_path"] == "Qwen/Qwen3-7B"
    assert args.sglang["tp_size"] == 2
    assert args.sglang["dtype"] == "bfloat16"

    assert args.slime["model"] == "Qwen/Qwen3-7B"
    assert args.slime["prompt_data"] == "acme/grpo-prompts"
    assert args.slime["mode"] == "sync"
    assert args.slime["advantage_estimator"] == "grpo"
    assert args.slime["rollout_max_response_len"] == 1024
    assert args.slime["n_samples_per_prompt"] == 8
    assert args.slime["num_rollout"] == 250
    assert args.slime["rollout_batch_size"] == 64
    assert args.slime["hub_model_id"] == "acme/demo-slime"
    assert args.slime["custom_reward_fn"] == "my.module:custom_reward"
    assert args.slime["num_epochs"] == 1

    assert args.extra == {"debug_dump_trajectories": True}


def test_translate_config_cli_args_emit_prefixes_and_booleans() -> None:
    args = translate_config(_SAMPLE_CFG)
    cli = args.to_cli_args()

    assert "--tensor-model-parallel-size" in cli
    assert "--sglang-model-path" in cli
    assert "--sglang-tp-size" in cli
    assert "--mode" in cli
    assert "--bf16" in cli  # True boolean → bare flag
    # bf16 must NOT have a following "True" literal
    idx = cli.index("--bf16")
    assert idx == len(cli) - 1 or not cli[idx + 1].startswith("True")

    # Extra passthrough flag
    assert "--debug-dump-trajectories" in cli


def test_translate_config_requires_base_model() -> None:
    bad = dict(_SAMPLE_CFG)
    bad.pop("base_model")
    with pytest.raises(AdapterError) as excinfo:
        translate_config(bad)
    assert excinfo.value.code == "carl.adapter.missing_required"
    assert excinfo.value.context["backend"] == "slime"
    assert excinfo.value.context["key"] == "base_model"


def test_translate_config_requires_dataset_repo() -> None:
    bad = dict(_SAMPLE_CFG)
    bad.pop("dataset_repo")
    with pytest.raises(AdapterError) as excinfo:
        translate_config(bad)
    assert excinfo.value.code == "carl.adapter.missing_required"
    assert excinfo.value.context["key"] == "dataset_repo"


def test_translate_config_rejects_unsupported_method() -> None:
    bad = dict(_SAMPLE_CFG)
    bad["method"] = "dpo"  # slime does not expose DPO
    with pytest.raises(AdapterError) as excinfo:
        translate_config(bad)
    assert excinfo.value.code == "carl.adapter.translation"
    assert "dpo" in excinfo.value.context["method"]


def test_translate_config_rejects_unsupported_mode() -> None:
    bad = dict(_SAMPLE_CFG)
    bad["mode"] = "hybrid"
    with pytest.raises(AdapterError) as excinfo:
        translate_config(bad)
    assert excinfo.value.code == "carl.adapter.translation"


def test_translate_config_rejects_unsupported_advantage_estimator() -> None:
    bad = dict(_SAMPLE_CFG)
    bad_slime = dict(bad["slime"])
    bad_slime["advantage_estimator"] = "dpo"  # not in set
    bad["slime"] = bad_slime
    with pytest.raises(AdapterError) as excinfo:
        translate_config(bad)
    assert excinfo.value.code == "carl.adapter.translation"


def test_translate_config_disaggregated_emits_flag() -> None:
    cfg = dict(_SAMPLE_CFG)
    cfg_slime = dict(cfg["slime"])
    cfg_slime["disaggregated"] = True
    cfg["slime"] = cfg_slime

    args = translate_config(cfg)
    assert args.slime["disaggregated"] is True
    assert "--disaggregated" in args.to_cli_args()


def test_translate_config_missing_slime_block_uses_defaults() -> None:
    cfg = {
        "base_model": "Qwen/Qwen3-7B",
        "dataset_repo": "acme/prompts",
        "method": "sft",
    }
    args = translate_config(cfg)
    assert args.slime["mode"] == "sync"
    assert args.slime["advantage_estimator"] == "grpo"
    assert args.extra == {}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_available_false_when_deps_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure nothing mentioning slime/sglang/megatron ghost-imports.
    for mod in ("slime", "sglang", "megatron", "megatron.core"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    adapter = SlimeAdapter()

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.importlib.util.find_spec",
        _none_spec,
    )

    assert adapter.available() is False
    missing = set(adapter.missing_dependencies())
    assert {"slime", "sglang", "megatron-lm"}.issubset(missing)


def test_available_true_when_all_deps_findable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlimeAdapter()

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.importlib.util.find_spec",
        _present_spec,
    )

    assert adapter.available() is True
    assert adapter.missing_dependencies() == []


def test_available_never_raises_on_find_spec_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlimeAdapter()

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.importlib.util.find_spec",
        _boom_spec,
    )

    # Never raises, always returns bool.
    assert adapter.available() is False
    assert isinstance(adapter.missing_dependencies(), list)


# ---------------------------------------------------------------------------
# Submit — exercising the unavailable path without spawning real processes
# ---------------------------------------------------------------------------


def test_submit_raises_unavailable_when_deps_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlimeAdapter()
    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.importlib.util.find_spec",
        _none_spec,
    )

    with pytest.raises(AdapterError) as excinfo:
        adapter.submit(_SAMPLE_CFG)

    err = excinfo.value
    assert err.code == "carl.adapter.unavailable"
    assert err.context["backend"] == "slime"
    assert "install_hint" in err.context
    assert "slime" in err.context["missing"]


def test_submit_translates_before_availability_check() -> None:
    """A broken carl.yaml should raise a translation error even if slime
    is not installed — translation happens first."""
    adapter = SlimeAdapter()
    broken = dict(_SAMPLE_CFG)
    broken.pop("base_model")

    with pytest.raises(AdapterError) as excinfo:
        adapter.submit(broken)
    assert excinfo.value.code == "carl.adapter.missing_required"


def test_submit_writes_state_and_returns_job_when_deps_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SlimeAdapter()

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.importlib.util.find_spec",
        _present_spec,
    )

    spawned: dict[str, Any] = {}

    def _fake_spawn(cmd: list[str], **kwargs: Any) -> int:
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        return 424242

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.spawn",
        _fake_spawn,
    )

    job = adapter.submit(_SAMPLE_CFG)

    assert job.backend == "slime"
    assert job.status == BackendStatus.RUNNING
    assert job.run_id.startswith("slime-")
    assert "translated" in job.raw
    assert job.raw["entry"] == "slime.cli:main"
    # ``cmd`` ends with the generated launcher script path.
    cmd_recorded: list[str] = spawned["cmd"]
    assert cmd_recorded[-1].endswith("train.py")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_exposes_slime_adapter() -> None:
    adapter = get_adapter("slime")
    assert isinstance(adapter, SlimeAdapter)
    assert adapter.name == "slime"


def test_list_adapters_includes_slime_entry() -> None:
    entries = list_adapters()
    names = {e["name"] for e in entries}
    assert "slime" in names
    for entry in entries:
        if entry["name"] == "slime":
            assert isinstance(entry["available"], bool)


# ---------------------------------------------------------------------------
# Launcher resolution
# ---------------------------------------------------------------------------


def test_launch_cmd_falls_back_to_python_when_torchrun_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If torchrun is not on PATH, the launcher should degrade to ``python script``
    rather than surface a cryptic FileNotFoundError downstream.
    """

    def _no_torchrun(_name: str) -> str | None:
        return None

    monkeypatch.setattr(
        "carl_studio.adapters.slime_adapter.shutil.which", _no_torchrun
    )

    cmd = resolve_launch_cmd(_SAMPLE_CFG, script_path="/tmp/train.py")
    assert cmd[-1] == "/tmp/train.py"
    # Fallback path: sys.executable + script, no torchrun in the middle.
    assert cmd[0].endswith("python") or cmd[0].endswith("python3") or "python" in cmd[0]


def test_launch_cmd_honors_launcher_cmd_override() -> None:
    cfg = dict(_SAMPLE_CFG)
    cfg_slime = dict(cfg["slime"])
    cfg_slime["launcher_cmd"] = ["bash", "-lc", "my-wrapper"]
    cfg["slime"] = cfg_slime

    cmd = resolve_launch_cmd(cfg, script_path="/tmp/train.py")
    assert cmd == ["bash", "-lc", "my-wrapper", "/tmp/train.py"]


def test_entry_resolution_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLIME_ENTRY", "slime.recipes.glm:main")
    # No explicit slime.entry in config — env should win.
    cfg = {k: v for k, v in _SAMPLE_CFG.items() if k != "slime"}
    assert resolve_entry(cfg) == "slime.recipes.glm:main"


def test_entry_resolution_config_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLIME_ENTRY", "slime.recipes.glm:main")
    cfg = dict(_SAMPLE_CFG)
    cfg_slime = dict(cfg["slime"])
    cfg_slime["entry"] = "my.module:go"
    cfg["slime"] = cfg_slime
    assert resolve_entry(cfg) == "my.module:go"


def test_entry_resolution_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLIME_ENTRY", raising=False)
    assert resolve_entry({}) == "slime.cli:main"
