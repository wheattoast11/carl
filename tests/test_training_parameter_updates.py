"""Independent optimizer controls for content-free parameter update witnesses."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carl_studio.training.callbacks import ParameterUpdateCallback
from carl_studio.training.trainer import CARLTrainer
from carl_studio.types.config import ComputeTarget, TrainingConfig, TrainingMethod


class Parameter:
    def __init__(self, values: list[float], *, trainable: bool = True) -> None:
        self.values = np.array(values, dtype=np.float32)
        self.requires_grad = trainable

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def numel(self) -> int:
        return int(self.values.size)

    def detach(self) -> Parameter:
        return self

    def contiguous(self) -> Parameter:
        return self

    def reshape(self, *shape: int) -> Parameter:
        result = Parameter([])
        result.values = self.values.reshape(*shape)
        return result

    def view(self, dtype: Any) -> Parameter:
        assert self.requires_grad
        result = Parameter([])
        result.values = self.values.view(dtype)
        return result

    def cpu(self) -> Parameter:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return self.values


class Model:
    def __init__(self) -> None:
        self.adapter = Parameter([1.0, 2.0])
        self.frozen = Parameter([99.0], trainable=False)
        self.peft_config = {"default": SimpleNamespace(peft_type="LORA")}

    def named_parameters(self) -> list[tuple[str, Parameter]]:
        return [("adapter.lora_B.default.weight", self.adapter), ("base.weight", self.frozen)]


class Optimizer:
    def __init__(self, model: Model, *, learning_rate: float, noop: bool) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.noop = noop
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        if not self.noop:
            self.model.adapter.values -= self.learning_rate * np.array([1.0, -1.0])


class Fit:
    def __init__(self, output: Path, *, learning_rate: float = 0.1, noop: bool = False) -> None:
        self.model = Model()
        self.args = SimpleNamespace(device=SimpleNamespace(type="cpu"), output_dir=str(output))
        self.callbacks: list[Any] = []
        self.optimizer = Optimizer(self.model, learning_rate=learning_rate, noop=noop)
        self.state = SimpleNamespace(is_world_process_zero=True, global_step=0)
        self.is_fsdp_enabled = False
        self.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=0))
        )

    def add_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def train(self, *, resume_from_checkpoint: str | None = None) -> None:
        if resume_from_checkpoint:
            self.model.adapter.values[:] = [10.0, 20.0]
            self.state.global_step = 7
        control = SimpleNamespace()
        for callback in self.callbacks:
            if hasattr(callback, "on_train_begin"):
                callback.on_train_begin(self.args, self.state, control, model=self.model)
        self.optimizer.step()
        self.state.global_step += 1
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(self.args, self.state, control, model=self.model)


class Subject(CARLTrainer):
    async def fit(self, fit: Fit, resume: str | None = None) -> None:
        await self._fit(fit, resume)

    def retain(self, fit: Any, output: str) -> None:
        self._retain_checkpoint(fit, output)

    def tokenizer(self, value: Any) -> None:
        self._tokenizer = value


class SnapshotProbe(ParameterUpdateCallback):
    @property
    def before(self) -> dict[str, dict[str, Any]] | None:
        return self._before


@pytest.fixture(autouse=True)
def tensor_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def isfinite(parameter: Parameter) -> np.ndarray[Any, Any]:
        return np.isfinite(parameter.values)

    torch = ModuleType("torch")
    torch.uint8 = np.uint8  # type: ignore[attr-defined]
    torch.isfinite = isfinite  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


@pytest.fixture(scope="module")
def cpu_tensor_backend() -> ModuleType:
    return pytest.importorskip("torch")


def subject() -> Subject:
    return Subject(
        TrainingConfig(
            run_name="parameter-update-test",
            base_model="unused",
            output_repo="test/unused",
            dataset_repo="unused",
            method=TrainingMethod.SFT,
            compute_target=ComputeTarget.LOCAL,
            push_to_hub=False,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "learning_rate,noop,changed", [(0.1, False, True), (0.1, True, False), (0.0, False, False)]
)
async def test_optimizer_update_witness(
    tmp_path: Path, learning_rate: float, noop: bool, changed: bool
) -> None:
    owner = subject()
    fit = Fit(tmp_path, learning_rate=learning_rate, noop=noop)
    initial = hashlib.sha256(fit.model.adapter.values.tobytes()).hexdigest()
    await owner.fit(fit)
    assert fit.optimizer.steps == 1
    assert owner.run.resource_usage["trainable_parameter_tensors"] == 1
    assert owner.run.resource_usage["updated_parameter_tensors"] == int(changed)
    assert owner.run.resource_usage["updated_parameter_modules"] == int(changed)
    artifact = owner.run.artifacts[0]
    payload = Path(artifact.path).read_bytes()
    assert artifact.checksum == hashlib.sha256(payload).hexdigest()
    evidence = json.loads(payload)
    assert evidence["schema"] == "carl.parameter-updates/v1"
    assert evidence["before"]["adapter.lora_B.default.weight"]["sha256"] == initial
    assert (
        evidence["after"]["adapter.lora_B.default.weight"]["sha256"]
        == hashlib.sha256(fit.model.adapter.values.tobytes()).hexdigest()
    )
    assert evidence["changed_parameters"] == (["adapter.lora_B.default.weight"] if changed else [])
    assert (evidence["before_sha256"] != evidence["after_sha256"]) == changed
    assert "base.weight" not in evidence["before"]
    assert evidence["start_step"] == 0 and evidence["end_step"] == 1


@pytest.mark.asyncio
async def test_resume_loading_precedes_initial_witness(tmp_path: Path) -> None:
    owner = subject()
    fit = Fit(tmp_path, learning_rate=0)
    await owner.fit(fit, "checkpoint-7")
    evidence = json.loads(Path(owner.run.artifacts[0].path).read_bytes())
    resumed = np.array([10.0, 20.0], dtype=np.float32)
    assert (
        evidence["before"]["adapter.lora_B.default.weight"]["sha256"]
        == hashlib.sha256(resumed.tobytes()).hexdigest()
    )
    assert evidence["start_step"] == 7 and evidence["end_step"] == 8
    assert evidence["changed_parameters"] == []


@pytest.mark.asyncio
async def test_non_main_rank_has_no_parameter_artifact(tmp_path: Path) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    fit.state.is_world_process_zero = False
    await owner.fit(fit)
    assert owner.run.artifacts == []
    assert owner.run.resource_usage == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_parameter_identity_change_refuses_witness(tmp_path: Path) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    callback = ParameterUpdateCallback(owner.run)
    callback.on_train_begin(fit.args, fit.state, None, model=fit.model)
    fit.model.adapter.values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="identities changed"):
        callback.on_train_end(fit.args, fit.state, None, model=fit.model)
    assert owner.run.artifacts == []


@pytest.mark.parametrize("finite", [False, True])
def test_missing_or_nonfinite_trainable_population_refuses(tmp_path: Path, finite: bool) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    fit.model.adapter.values[0] = np.nan
    fit.model.adapter.requires_grad = not finite
    with pytest.raises(ValueError, match="no trainable|Nonfinite"):
        ParameterUpdateCallback(owner.run).on_train_begin(
            fit.args, fit.state, None, model=fit.model
        )
    assert owner.run.artifacts == []


@pytest.mark.parametrize(
    "dtype_name,expected", [("float32", b"\x00\x00\x80\x3f"), ("bfloat16", b"\x80\x3f")]
)
def test_scalar_tensor_bytes_preserve_dtype(
    monkeypatch: pytest.MonkeyPatch,
    cpu_tensor_backend: ModuleType,
    dtype_name: str,
    expected: bytes,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", cpu_tensor_backend)
    import torch as cpu_torch

    dtype = getattr(cpu_torch, dtype_name)
    parameter = cpu_torch.tensor(1.0, dtype=dtype, device="cpu", requires_grad=True)
    model = SimpleNamespace(named_parameters=lambda: [("adapter.lora_B.default.weight", parameter)])
    run = subject().run
    callback = SnapshotProbe(run)
    callback.on_train_begin(
        None, SimpleNamespace(is_world_process_zero=True, global_step=0), None, model=model
    )
    assert callback.before is not None
    assert callback.before["adapter.lora_B.default.weight"]["shape"] == []
    assert (
        callback.before["adapter.lora_B.default.weight"]["sha256"]
        == hashlib.sha256(expected).hexdigest()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sharding", ["fsdp", "zero3"])
async def test_sharded_parameters_refuse_before_training(tmp_path: Path, sharding: str) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    if sharding == "fsdp":
        fit.is_fsdp_enabled = True
    else:
        fit.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3))
        )
    with pytest.raises(ValueError, match="unsharded parameters"):
        await owner.fit(fit)
    assert fit.optimizer.steps == 0
    assert owner.run.artifacts == []


def test_trainable_base_weights_are_not_copied(tmp_path: Path) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    fit.model.frozen.requires_grad = True
    fit.model.frozen.values[0] = np.nan
    callback = ParameterUpdateCallback(owner.run)
    callback.on_train_begin(fit.args, fit.state, None, model=fit.model)
    callback.on_train_end(fit.args, fit.state, None, model=fit.model)
    evidence = json.loads(Path(owner.run.artifacts[0].path).read_bytes())
    assert set(evidence["before"]) == {"adapter.lora_B.default.weight"}


@pytest.mark.asyncio
async def test_interrupted_training_has_no_completed_witness(tmp_path: Path) -> None:
    class InterruptedFit(Fit):
        def train(self, *, resume_from_checkpoint: str | None = None) -> None:
            for callback in self.callbacks:
                if hasattr(callback, "on_train_begin"):
                    callback.on_train_begin(self.args, self.state, None, model=self.model)
            self.optimizer.step()
            raise RuntimeError("interrupted training")

    owner = subject()
    fit = InterruptedFit(tmp_path)
    with pytest.raises(RuntimeError, match="interrupted training"):
        await owner.fit(fit)
    assert fit.optimizer.steps == 1
    assert owner.run.artifacts == []
    assert owner.run.resource_usage == {}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_tampered_existing_parameter_artifact_refuses(tmp_path: Path) -> None:
    owner = subject()
    await owner.fit(Fit(tmp_path, learning_rate=0))
    artifact = owner.run.artifacts[0]
    Path(artifact.path).write_text("changed")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        await owner.fit(Fit(tmp_path, learning_rate=0))
    assert len(owner.run.artifacts) == 1


@pytest.mark.asyncio
async def test_report_and_tokenizer_files_are_not_checkpoint_weights(tmp_path: Path) -> None:
    owner = subject()
    await owner.fit(Fit(tmp_path))

    def save_tokenizer(output: str) -> None:
        (Path(output) / "tokenizer.json").write_text("{}")

    owner.tokenizer(SimpleNamespace(save_pretrained=save_tokenizer))

    def skipped_save(output: str) -> None:
        pass

    fit = SimpleNamespace(save_model=skipped_save, state=SimpleNamespace(global_step=1))
    with pytest.raises(ValueError, match="did not produce model weights"):
        owner.retain(fit, str(tmp_path))
    assert owner.run.checkpoint is None
    assert len(owner.run.artifacts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_checkpoint_weight_files_remain_independent_of_report(
    tmp_path: Path, empty: bool
) -> None:
    owner = subject()
    fit = Fit(tmp_path)
    await owner.fit(fit)

    def saved_model(output: str) -> None:
        header = json.dumps(
            {"adapter.lora_B.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        ).encode()
        payload = (
            b""
            if empty
            else struct.pack("<Q", len(header)) + header + fit.model.adapter.values.tobytes()
        )
        (Path(output) / "adapter_model.safetensors").write_bytes(payload)

    def saved_tokenizer(output: str) -> None:
        (Path(output) / "tokenizer.json").write_text("{}")

    owner.tokenizer(SimpleNamespace(save_pretrained=saved_tokenizer))
    checkpoint = SimpleNamespace(save_model=saved_model, state=SimpleNamespace(global_step=1))
    if empty:
        with pytest.raises(ValueError, match="did not produce model weights"):
            owner.retain(checkpoint, str(tmp_path))
        assert owner.run.checkpoint is None
    else:
        owner.retain(checkpoint, str(tmp_path))
        assert owner.run.checkpoint == str(tmp_path)
        assert owner.run.current_step == 1
        assert [artifact.artifact_type for artifact in owner.run.artifacts] == [
            "report",
            "checkpoint",
        ]


def test_preparation_binds_parameter_callback_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from carl_core.errors import ValidationError

    from carl_studio.experiment.manager import ExperimentManager
    from carl_studio.training import preparation

    owners: tuple[Path, ...] = preparation.implementation_sources()
    callback = next(path for path in owners if path.name == "callbacks.py")
    copied_callback = tmp_path / "callbacks.py"
    copied_callback.write_bytes(callback.read_bytes())
    monkeypatch.setattr(
        preparation,
        "implementation_sources",
        lambda: tuple(copied_callback if path == callback else path for path in owners),
    )

    def available_dependency(name: str) -> SimpleNamespace:
        return SimpleNamespace()

    monkeypatch.setattr(preparation.importlib.util, "find_spec", available_dependency)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"declared-test-fixture")
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "eval.jsonl"
    train.write_text(
        json.dumps({"id": "train", "prompt": "first task", "expected_output": "a"}) + "\n"
    )
    heldout.write_text(
        json.dumps({"id": "heldout", "prompt": "different task", "expected_output": "b"}) + "\n"
    )
    config = TrainingConfig(
        run_name="callback-source-test",
        base_model=str(model),
        output_repo="test/unused",
        dataset_repo=str(train),
        eval_dataset_repo=str(heldout),
        method=TrainingMethod.SFT,
        compute_target=ComputeTarget.LOCAL,
        max_steps=1,
        push_to_hub=False,
    )
    prepared = preparation.prepare_training(
        config, project_root=tmp_path, manager=ExperimentManager(tmp_path / "experiments")
    )
    assert prepared.ready
    assert any(source.path == str(copied_callback) for source in prepared.sources)
    preparation.validate_preparation(prepared)
    copied_callback.write_bytes(copied_callback.read_bytes() + b"\n# changed source\n")
    with pytest.raises(ValidationError, match="prepared source changed"):
        preparation.validate_preparation(prepared)
