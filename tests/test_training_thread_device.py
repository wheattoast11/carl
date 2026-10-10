"""CPU mocks for CUDA device custody across the training executor boundary."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from carl_studio.training.preparation import run_in_worker
from carl_studio.training.trainer import CARLTrainer
from carl_studio.types.config import ComputeTarget, TrainingConfig, TrainingMethod


class ThreadLocalCuda:
    """Model CUDA's thread-local current device without touching hardware."""

    def __init__(self) -> None:
        self.local = threading.local()
        self.bindings: list[tuple[int, int]] = []

    def device_count(self) -> int:
        return 2

    def current_device(self) -> int:
        return getattr(self.local, "device", 0)

    def set_device(self, index: int) -> None:
        if not 0 <= index < self.device_count():
            raise ValueError("Invalid mock CUDA device")
        self.local.device = index
        self.bindings.append((threading.get_ident(), index))


class FitProbe:
    def __init__(self, cuda: ThreadLocalCuda, device: SimpleNamespace) -> None:
        self.cuda = cuda
        self.args = SimpleNamespace(device=device)
        self.calls: list[tuple[int, int | None, dict[str, Any]]] = []
        self.callback: Any = None

    def add_callback(self, callback: Any) -> None:
        self.callback = callback

    def train(self, **kwargs: Any) -> None:
        current = self.cuda.current_device() if self.args.device.type == "cuda" else None
        self.calls.append((threading.get_ident(), current, kwargs))


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> ThreadLocalCuda:
    value = ThreadLocalCuda()
    torch = ModuleType("torch")
    torch.cuda = value  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    return value


class Subject(CARLTrainer):
    async def fit(self, fit: FitProbe, resume: bool | str | None) -> None:
        await self._fit(fit, resume)


def trainer() -> Subject:
    return Subject(
        TrainingConfig(
            run_name="thread-device-test",
            base_model="unused",
            output_repo="test/unused",
            dataset_repo="unused",
            method=TrainingMethod.SFT,
            compute_target=ComputeTarget.LOCAL,
            push_to_hub=False,
        )
    )


@pytest.mark.asyncio
async def test_main_thread_device_does_not_carry_into_executor(cuda: ThreadLocalCuda) -> None:
    cuda.set_device(1)
    executor_device = await run_in_worker(cuda.current_device)
    assert cuda.current_device() == 1
    assert executor_device == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [None, True, False, "checkpoint-7"])
async def test_cuda_rank_is_bound_in_training_executor(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch, resume: bool | str | None
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    cuda.set_device(1)
    main_thread = threading.get_ident()
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=1))
    await trainer().fit(fit, resume)
    worker_thread, worker_device, kwargs = fit.calls[0]
    assert worker_thread != main_thread
    assert worker_device == 1
    assert cuda.bindings[-1] == (worker_thread, 1)
    assert cuda.current_device() == 1
    assert kwargs == ({} if resume is None else {"resume_from_checkpoint": resume})


@pytest.mark.asyncio
@pytest.mark.parametrize("rank", ["1", "invalid"])
async def test_cpu_training_ignores_cuda_rank_and_preserves_resume(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch, rank: str
) -> None:
    monkeypatch.setenv("LOCAL_RANK", rank)
    fit = FitProbe(cuda, SimpleNamespace(type="cpu", index=None))
    await trainer().fit(fit, "checkpoint-9")
    assert fit.calls[0][1:] == (None, {"resume_from_checkpoint": "checkpoint-9"})
    assert cuda.bindings == []


@pytest.mark.asyncio
async def test_indexed_cuda_without_rank_binds_declared_device(cuda: ThreadLocalCuda) -> None:
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=1))
    await trainer().fit(fit, None)
    assert fit.calls[0][1] == 1


@pytest.mark.asyncio
async def test_unindexed_cuda_uses_declared_rank(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=None))
    await trainer().fit(fit, None)
    assert fit.calls[0][1] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rank", ["invalid", "-1", "1.5"])
async def test_invalid_cuda_rank_refuses_before_training(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch, rank: str
) -> None:
    monkeypatch.setenv("LOCAL_RANK", rank)
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=None))
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        await trainer().fit(fit, None)
    assert fit.calls == []
    assert cuda.bindings == []


@pytest.mark.asyncio
async def test_cuda_rank_and_trainer_device_mismatch_refuses(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=1))
    with pytest.raises(ValueError, match="does not match LOCAL_RANK"):
        await trainer().fit(fit, None)
    assert fit.calls == []
    assert cuda.bindings == []


@pytest.mark.asyncio
async def test_out_of_range_rank_refuses_before_training(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "2")
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=None))
    with pytest.raises(ValueError, match="visible device index"):
        await trainer().fit(fit, None)
    assert fit.calls == []
    assert cuda.bindings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [None, -1, 2])
async def test_unavailable_cuda_device_refuses_before_training(
    cuda: ThreadLocalCuda, index: int | None
) -> None:
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=index))
    with pytest.raises(ValueError, match="visible device index"):
        await trainer().fit(fit, None)
    assert fit.calls == []
    assert cuda.bindings == []


@pytest.mark.asyncio
async def test_failed_cuda_binding_refuses_before_training(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ignored_binding(index: int) -> None:
        pass

    monkeypatch.setattr(cuda, "set_device", ignored_binding)
    fit = FitProbe(cuda, SimpleNamespace(type="cuda", index=1))
    with pytest.raises(RuntimeError, match="CUDA training device binding failed"):
        await trainer().fit(fit, None)
    assert fit.calls == []


@pytest.mark.asyncio
async def test_cuda_cancellation_preserves_worker_cleanup(
    cuda: ThreadLocalCuda, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    started = threading.Event()
    ended = threading.Event()

    class BlockingFit(FitProbe):
        def train(self, **kwargs: Any) -> None:
            assert self.cuda.current_device() == 1
            started.set()
            control = SimpleNamespace(should_training_stop=False, should_save=False)
            while not control.should_training_stop:
                self.callback.on_step_end(None, None, control)
                time.sleep(0.001)
            assert control.should_save
            time.sleep(0.01)
            ended.set()

    task = asyncio.create_task(
        trainer().fit(BlockingFit(cuda, SimpleNamespace(type="cuda", index=1)), None)
    )
    try:
        async with asyncio.timeout(3):
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert ended.is_set()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
