"""Tests for the UnifiedBackend adapter layer.

These tests exercise the registry, protocol contract, each built-in
adapter's translation logic, availability detection with patched
``shutil.which`` / ``importlib.util.find_spec``, and the CLI surfaces
(``carl train --backend`` and ``carl lab backends``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from typer.testing import CliRunner

from carl_core.errors import ConfigError

from carl_studio.adapters import (
    AdapterError,
    AtroposAdapter,
    AxolotlAdapter,
    BackendJob,
    BackendStatus,
    TinkerAdapter,
    TRLAdapter,
    UnslothAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)
from carl_studio.adapters.axolotl_adapter import translate_config as axolotl_translate
from carl_studio.adapters.tinker_adapter import translate_config as tinker_translate
from carl_studio.adapters.unsloth_adapter import translate_config as unsloth_translate


_SAMPLE_CARL_CFG: dict[str, Any] = {
    "run_name": "demo-run",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "output_repo": "acme/demo",
    "dataset_repo": "acme/sft-data",
    "dataset_split": "train",
    "method": "grpo",
    "num_train_epochs": 2,
    "max_steps": 500,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1.5e-5,
    "max_length": 1024,
    "num_generations": 8,
    "max_completion_length": 256,
    "grpo_temperature": 1.7,
    "lora": {"r": 32, "alpha": 64, "dropout": 0.07},
    "quantization": {"load_in_8bit": False, "load_in_4bit": True},
    "seed": 123,
}


@pytest.fixture(autouse=True)
def _adapters_state_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect adapter state to a temporary directory per test."""
    monkeypatch.setenv("CARL_ADAPTER_STATE_DIR", str(tmp_path / "adapters"))


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def test_get_adapter_returns_registered_instance_by_name():
    adapter = get_adapter("trl")
    assert isinstance(adapter, TRLAdapter)
    assert adapter.name == "trl"


def test_get_adapter_unknown_raises_config_error():
    with pytest.raises(ConfigError) as excinfo:
        get_adapter("does-not-exist")

    err = excinfo.value
    assert err.code == "carl.adapter.unknown"
    assert "known" in err.context
    assert "trl" in err.context["known"]


def test_list_adapters_returns_name_available_dicts():
    entries = list_adapters()

    names = {e["name"] for e in entries}
    # All five built-ins must be present.
    assert {"trl", "unsloth", "axolotl", "tinker", "atropos"}.issubset(names)

    for entry in entries:
        assert set(entry.keys()) == {"name", "available"}
        assert isinstance(entry["available"], bool)


def test_register_adapter_rejects_nameless():
    class _NoName:
        name = ""

        def available(self) -> bool:
            return False

    with pytest.raises(ConfigError):
        register_adapter(_NoName())  # type: ignore[arg-type]


def test_list_adapters_coerces_raises_to_false():
    """available() that raises should not break list_adapters()."""

    class _Explodes:
        name = "explodes-adapter"

        def available(self) -> bool:
            raise RuntimeError("boom")

        def submit(self, cfg):  # pragma: no cover
            raise NotImplementedError

        def status(self, run_id):  # pragma: no cover
            raise NotImplementedError

        def logs(self, run_id, *, tail=100):  # pragma: no cover
            raise NotImplementedError

        def cancel(self, run_id):  # pragma: no cover
            raise NotImplementedError

    register_adapter(_Explodes())
    try:
        entries = {e["name"]: e for e in list_adapters()}
        assert entries["explodes-adapter"]["available"] is False
    finally:
        from carl_studio.adapters.registry import _unregister

        _unregister("explodes-adapter")


# ---------------------------------------------------------------------------
# 2. Availability
# ---------------------------------------------------------------------------


def test_trl_adapter_always_available():
    assert TRLAdapter().available() is True


def test_unsloth_available_false_when_spec_missing():
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=None,
    ):
        assert UnslothAdapter().available() is False


def test_tinker_available_false_when_spec_missing():
    with patch(
        "carl_studio.adapters.tinker_adapter.importlib.util.find_spec",
        return_value=None,
    ):
        assert TinkerAdapter().available() is False


def test_axolotl_available_false_when_binary_missing():
    with patch("carl_studio.adapters.axolotl_adapter.shutil.which", return_value=None):
        assert AxolotlAdapter().available() is False


def test_atropos_available_false_when_binary_missing():
    with patch("carl_studio.adapters.atropos_adapter.shutil.which", return_value=None):
        assert AtroposAdapter().available() is False


def test_unsloth_available_true_with_fake_spec():
    fake_spec = object()
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=fake_spec,
    ):
        assert UnslothAdapter().available() is True


# ---------------------------------------------------------------------------
# 3. Unavailable submit paths
# ---------------------------------------------------------------------------


def test_unsloth_submit_when_unavailable_raises_adapter_unavailable():
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=None,
    ):
        with pytest.raises(AdapterError) as excinfo:
            UnslothAdapter().submit(_SAMPLE_CARL_CFG)
    assert excinfo.value.code == "carl.adapter.unavailable"
    assert "install_hint" in excinfo.value.context


def test_axolotl_submit_when_unavailable_raises_adapter_unavailable():
    with patch("carl_studio.adapters.axolotl_adapter.shutil.which", return_value=None):
        with pytest.raises(AdapterError) as excinfo:
            AxolotlAdapter().submit(_SAMPLE_CARL_CFG)
    assert excinfo.value.code == "carl.adapter.unavailable"


def test_atropos_submit_when_unavailable_raises_adapter_unavailable():
    with patch("carl_studio.adapters.atropos_adapter.shutil.which", return_value=None):
        with pytest.raises(AdapterError) as excinfo:
            AtroposAdapter().submit(_SAMPLE_CARL_CFG)
    assert excinfo.value.code == "carl.adapter.unavailable"


def test_tinker_submit_raises_not_implemented_cleanly(tmp_path: Path):
    """Tinker's submission path is deliberately scaffolded.

    Option B (current policy): ``submit()`` raises
    ``carl.adapter.tinker_not_implemented`` immediately after validating
    translation. We must NOT persist any JobState for a run the caller
    cannot observe (no run_id is returned on failure).
    """
    # Spy on save_state to catch any accidental state persistence.
    from carl_studio.adapters import _common as common

    saved: list[str] = []
    original_save = common.save_state

    def _capture(state):
        saved.append(state.run_id)
        return original_save(state)

    with patch.object(common, "save_state", side_effect=_capture):
        with pytest.raises(AdapterError) as excinfo:
            TinkerAdapter().submit(_SAMPLE_CARL_CFG)

    assert excinfo.value.code == "carl.adapter.tinker_not_implemented"
    assert "docs_url" in excinfo.value.context
    assert excinfo.value.context["backend"] == "tinker"
    # No state may have been persisted — zombie state files are forbidden.
    assert saved == []
    # And the on-disk state dir for tinker must be empty (or non-existent).
    tinker_state = Path(
        __import__("os").environ["CARL_ADAPTER_STATE_DIR"]
    ) / "tinker"
    if tinker_state.exists():
        assert list(tinker_state.glob("*.json")) == []
        assert list(tinker_state.glob("*/run.log")) == []


def test_tinker_submit_still_validates_translation():
    """Even though submission raises not_implemented, translation errors
    still surface before the not-implemented error so operators see the
    real problem first."""
    bad = {"base_model": "x"}  # missing dataset_repo
    with pytest.raises(AdapterError) as excinfo:
        TinkerAdapter().submit(bad)
    assert excinfo.value.code in {
        "carl.adapter.translation",
        "carl.adapter.missing_required",
    }


# ---------------------------------------------------------------------------
# 4. Translation
# ---------------------------------------------------------------------------


def test_axolotl_translation_maps_grpo_lora_to_schema_keys():
    out = axolotl_translate(_SAMPLE_CARL_CFG)

    # Core identity
    assert out["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert out["datasets"][0]["path"] == "acme/sft-data"
    assert out["datasets"][0]["split"] == "train"

    # LoRA mapping
    assert out["adapter"] == "lora"
    assert out["lora_r"] == 32
    assert out["lora_alpha"] == 64
    assert out["lora_dropout"] == pytest.approx(0.07)

    # Quantization: 4bit only
    assert out["load_in_4bit"] is True
    assert out["load_in_8bit"] is False

    # GRPO specifics
    assert out["rl"] == "grpo"
    assert out["num_generations"] == 8
    assert out["max_completion_length"] == 256
    assert out["temperature"] == pytest.approx(1.7)

    # Hyperparameters
    assert out["micro_batch_size"] == 2
    assert out["gradient_accumulation_steps"] == 4
    assert out["learning_rate"] == pytest.approx(1.5e-5)
    assert out["sequence_len"] == 1024
    assert out["seed"] == 123

    # YAML-roundtrippable
    assert yaml.safe_load(yaml.safe_dump(out)) == out


def test_axolotl_translation_missing_required_raises_translation_error():
    bad = {"base_model": "x"}  # no dataset_repo
    with pytest.raises(AdapterError) as excinfo:
        axolotl_translate(bad)
    # The shared ``require_str`` helper surfaces missing-required failures
    # with a distinct code so callers can branch on it. The older
    # ``carl.adapter.translation`` code is still used for shape/type errors.
    assert excinfo.value.code == "carl.adapter.missing_required"
    assert excinfo.value.context.get("key") == "dataset_repo"
    assert excinfo.value.context.get("backend") == "axolotl"


def test_axolotl_translation_rejects_unknown_method():
    cfg = dict(_SAMPLE_CARL_CFG, method="ppo")
    with pytest.raises(AdapterError) as excinfo:
        axolotl_translate(cfg)
    assert excinfo.value.code == "carl.adapter.translation"


def test_unsloth_translation_4bit_config():
    cfg = dict(
        _SAMPLE_CARL_CFG,
        method="sft",
        quantization={"load_in_8bit": False, "load_in_4bit": True},
        lora={"r": 16, "alpha": 32, "dropout": 0.0},
    )
    out = unsloth_translate(cfg)

    assert out["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert out["method"] == "sft"
    assert out["load_in_4bit"] is True
    assert out["load_in_8bit"] is False
    assert out["lora_r"] == 16
    assert out["lora_alpha"] == 32
    assert out["lora_dropout"] == 0.0
    assert out["max_seq_length"] == 1024
    assert out["num_train_epochs"] == 2


def test_unsloth_translation_default_8bit_when_unspecified():
    cfg = {k: v for k, v in _SAMPLE_CARL_CFG.items() if k != "quantization"}
    cfg["method"] = "sft"
    out = unsloth_translate(cfg)
    # Without quantization section, default is 8-bit on, 4-bit off.
    assert out["load_in_8bit"] is True
    assert out["load_in_4bit"] is False


def test_unsloth_translation_rejects_unknown_method():
    cfg = dict(_SAMPLE_CARL_CFG, method="xrl")
    with pytest.raises(AdapterError) as excinfo:
        unsloth_translate(cfg)
    assert excinfo.value.code == "carl.adapter.translation"


@pytest.mark.parametrize("method", ["dpo", "kto", "orpo"])
def test_unsloth_rejects_unsupported_method(method: str):
    """dpo/kto/orpo are validated-but-unimplemented at translation time.

    The entrypoint template only handles sft and grpo, so we fail fast at
    translation rather than spawning a subprocess that sys.exit(3)s.
    """
    cfg = dict(_SAMPLE_CARL_CFG, method=method)

    # Via translate_config:
    with pytest.raises(AdapterError) as excinfo:
        unsloth_translate(cfg)
    assert excinfo.value.code == "carl.adapter.translation"
    assert excinfo.value.context.get("method") == method
    assert excinfo.value.context.get("supported") == ["grpo", "sft"]

    # Via UnslothAdapter.submit: availability is mocked True so we verify
    # the error surfaces BEFORE spawning anything.
    fake_spec = object()
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=fake_spec,
    ):
        with patch("carl_studio.adapters.unsloth_adapter.spawn") as spawn_mock:
            with pytest.raises(AdapterError) as excinfo2:
                UnslothAdapter().submit(cfg)
            assert spawn_mock.called is False
    assert excinfo2.value.code == "carl.adapter.translation"


def test_unsloth_quantization_4bit_only():
    """load_in_4bit=True wins — 8bit is forced off even if also requested."""
    cfg = dict(
        _SAMPLE_CARL_CFG,
        method="sft",
        quantization={"load_in_4bit": True, "load_in_8bit": True},
    )
    out = unsloth_translate(cfg)
    assert out["load_in_4bit"] is True
    assert out["load_in_8bit"] is False


def test_unsloth_quantization_8bit_default():
    """Empty/unspecified quantization section defaults to 8-bit on only."""
    cfg = dict(_SAMPLE_CARL_CFG, method="sft")
    cfg.pop("quantization", None)
    out = unsloth_translate(cfg)
    assert out["load_in_4bit"] is False
    assert out["load_in_8bit"] is True


def test_unsloth_quantization_neither():
    """Explicit load_in_8bit=False, no 4bit -> full precision (both False)."""
    cfg = dict(
        _SAMPLE_CARL_CFG,
        method="sft",
        quantization={"load_in_8bit": False, "load_in_4bit": False},
    )
    out = unsloth_translate(cfg)
    assert out["load_in_4bit"] is False
    assert out["load_in_8bit"] is False


def test_tinker_translation_preserves_extras():
    cfg = dict(_SAMPLE_CARL_CFG, vlm_mode=True, some_custom_key="abc")
    out = tinker_translate(cfg)
    assert out["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert out["method"] == "grpo"
    assert out["adapter"]["r"] == 32
    assert out["extras"]["vlm_mode"] is True
    assert out["extras"]["some_custom_key"] == "abc"


# ---------------------------------------------------------------------------
# 5. BackendJob + state round-trip
# ---------------------------------------------------------------------------


def test_trl_submit_produces_pending_backend_job():
    job = TRLAdapter().submit(_SAMPLE_CARL_CFG)
    assert isinstance(job, BackendJob)
    assert job.backend == "trl"
    assert job.status == BackendStatus.PENDING
    assert job.run_id.startswith("trl-")


def test_trl_status_loads_persisted_state():
    adapter = TRLAdapter()
    job = adapter.submit(_SAMPLE_CARL_CFG)
    reloaded = adapter.status(job.run_id)
    assert reloaded.run_id == job.run_id
    assert reloaded.backend == "trl"


def test_status_unknown_run_raises_adapter_error(tmp_path: Path):
    adapter = TRLAdapter()
    with pytest.raises(AdapterError) as excinfo:
        adapter.status("trl-does-not-exist")
    assert excinfo.value.code == "carl.adapter.status"


def test_backend_status_helpers():
    assert BackendStatus.is_terminal(BackendStatus.COMPLETED)
    assert BackendStatus.is_terminal(BackendStatus.FAILED)
    assert not BackendStatus.is_terminal(BackendStatus.RUNNING)
    assert BackendStatus.is_known("running")
    assert not BackendStatus.is_known("marinating")


# ---------------------------------------------------------------------------
# 6. CLI surfaces
# ---------------------------------------------------------------------------


def _cli_app():
    # Import inside the function so the stub-friendly conftest gets a chance
    # to run before carl_studio.cli is pulled in.
    from carl_studio.cli import app

    return app


def test_cli_train_with_unknown_backend_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    config = tmp_path / "carl.yaml"
    config.write_text(
        "\n".join(
            [
                "base_model: acme/model",
                "method: sft",
                "dataset_repo: acme/data",
                "output_repo: acme/out",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".carl").mkdir()
    result = runner.invoke(
        _cli_app(), ["train", "--config", str(config), "--backend", "nope"]
    )
    assert result.exit_code == 2
    assert "unknown training backend" in result.output


def test_cli_train_unsloth_dry_run_translates_without_submitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    config = tmp_path / "carl.yaml"
    config.write_text(
        "\n".join(
            [
                "base_model: acme/model",
                "method: sft",
                "dataset_repo: acme/data",
                "output_repo: acme/out",
                "max_length: 2048",
                "lora:",
                "  r: 8",
                "  alpha: 16",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".carl").mkdir()

    # Pretend unsloth is installed so availability is True.
    fake_spec = object()
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=fake_spec,
    ):
        result = runner.invoke(
            _cli_app(),
            ["train", "--config", str(config), "--backend", "unsloth", "--dry-run"],
        )

    assert result.exit_code == 0, result.output
    assert "Backend Dispatch" in result.output
    assert "unsloth" in result.output
    assert "\"max_seq_length\": 2048" in result.output
    assert "\"lora_r\": 8" in result.output


def test_cli_lab_backends_lists_entries_as_json():
    runner = CliRunner()
    result = runner.invoke(_cli_app(), ["lab", "backends", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    names = {e["name"] for e in data}
    assert {"trl", "unsloth", "axolotl", "tinker", "atropos"}.issubset(names)


def test_cli_lab_backends_table_output_contains_all():
    runner = CliRunner()
    result = runner.invoke(_cli_app(), ["lab", "backends"])
    assert result.exit_code == 0
    for name in ("trl", "unsloth", "axolotl", "tinker", "atropos"):
        assert name in result.output


def test_cli_train_unavailable_backend_errors_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = CliRunner()
    config = tmp_path / "carl.yaml"
    config.write_text(
        "\n".join(
            [
                "base_model: acme/model",
                "method: sft",
                "dataset_repo: acme/data",
                "output_repo: acme/out",
            ]
        )
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".carl").mkdir()
    # Force unsloth to appear absent.
    with patch(
        "carl_studio.adapters.unsloth_adapter.importlib.util.find_spec",
        return_value=None,
    ):
        result = runner.invoke(
            _cli_app(),
            ["train", "--config", str(config), "--backend", "unsloth"],
        )
    assert result.exit_code == 1
    assert "not installed" in result.output or "unavailable" in result.output
