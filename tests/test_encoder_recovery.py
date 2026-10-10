"""Completed optimizer custody survives independent evaluator failures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from carl_core.errors import ValidationError

from carl_studio.training import encoder
from carl_studio.types.config import TrainingConfig
from carl_studio.types.run import RunPhase, TrainingRun


class Owner:
    def __init__(self, root: Path) -> None:
        self.base_dir = root
        (root / "plan").mkdir()
        self.recorded: TrainingRun | None = None

    def load_training_result(self, plan_id: str) -> TrainingRun | None:
        return self.recorded.model_copy(deep=True) if self.recorded else None

    def save_training_result(self, plan_id: str, run: TrainingRun) -> None:
        self.recorded = run.model_copy(deep=True)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    config = TrainingConfig(
        run_name="recovery",
        base_model="local/model",
        output_repo="local/out",
        method="sft",
        dataset_repo="local/data",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, value in {
        "encoder_state.pt": "weights",
        "trainer_state.json": json.dumps({"status": "complete"}),
        "measurements.json": "{}",
    }.items():
        (checkpoint / name).write_text(value)
    run = TrainingRun(
        id="run",
        config=config,
        phase=RunPhase.COMPLETE,
        checkpoint=str(checkpoint),
        optimizer_phase="complete",
        evaluation_phase="pending",
        activation_phase="pending",
        current_step=128,
    )
    run.completion_custody = encoder.completion_custody(run)
    owner = Owner(tmp_path)
    owner.save_training_result("plan", run)
    prepared = SimpleNamespace(plan_id="plan", config=config.model_dump(mode="json"))
    monkeypatch.setattr(encoder, "validate", lambda *args: None)
    return run, prepared, owner, checkpoint


def test_failed_evaluation_recovers_once_without_optimizer(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, prepared, owner, _ = case
    attempts = []

    def evaluate(*args: Any) -> dict[str, object]:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Evaluator interrupted")
        return {"status": "accepted", "reasons": []}

    monkeypatch.setattr(encoder, "evaluate", evaluate)
    with pytest.raises(ValidationError):
        encoder.finish_evaluation(run, prepared, owner)
    assert owner.recorded.optimizer_phase == "complete"
    assert owner.recorded.evaluation_phase == "failed"
    recovered = encoder.finish_evaluation(run, prepared, owner)
    assert recovered.phase == RunPhase.COMPLETE
    assert recovered.current_step == 128
    assert recovered.id == "run"
    assert recovered.evaluation_phase == "complete"
    assert encoder.finish_evaluation(run, prepared, owner) == recovered
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "filename", ["encoder_state.pt", "trainer_state.json", "measurements.json"]
)
def test_changed_completion_bytes_refuse_recovery(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    run, prepared, owner, checkpoint = case
    monkeypatch.setattr(encoder, "evaluate", lambda *args: pytest.fail("Evaluator executed"))
    (checkpoint / filename).write_text("changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        encoder.finish_evaluation(run, prepared, owner)


def test_changed_evaluator_binding_refuses_recovery(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, prepared, owner, _ = case

    def changed(*args: Any) -> None:
        raise ValueError("Callable source bytes changed")

    monkeypatch.setattr(encoder, "validate", changed)
    monkeypatch.setattr(encoder, "evaluate", lambda *args: pytest.fail("Evaluator executed"))
    with pytest.raises(ValueError, match="Callable"):
        encoder.finish_evaluation(run, prepared, owner)


def test_artifacts_changed_by_evaluator_cannot_be_accepted(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    run, prepared, owner, checkpoint = case

    def changed(*args: Any) -> dict[str, object]:
        (checkpoint / "measurements.json").write_text('{"changed":true}')
        return {"status": "accepted", "reasons": []}

    monkeypatch.setattr(encoder, "evaluate", changed)
    with pytest.raises(ValidationError):
        encoder.finish_evaluation(run, prepared, owner)
    assert owner.recorded.evaluation_phase == "failed"
    assert owner.recorded.activation_phase != "complete"


def test_legacy_run_has_no_invented_optimizer_completion() -> None:
    config = TrainingConfig(
        run_name="legacy",
        base_model="local/model",
        output_repo="local/out",
        method="sft",
        dataset_repo="local/data",
    )
    run = TrainingRun(id="legacy", config=config)
    assert run.optimizer_phase is None
    assert run.evaluation_phase is None
    assert run.activation_phase is None
    assert run.completion_custody == {}


def test_repeated_submit_recovers_evaluation_without_constructing_trainer(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from carl_studio.training import trainer

    run, prepared, owner, _ = case
    run.phase = RunPhase.FAILED
    run.evaluation_phase = "failed"
    owner.save_training_result("plan", run)
    monkeypatch.setattr(
        trainer, "CARLTrainer", lambda *args, **kwargs: pytest.fail("Optimizer constructed")
    )
    calls = []

    def evaluate(*args: Any) -> dict[str, object]:
        calls.append(1)
        return {"status": "accepted", "reasons": []}

    monkeypatch.setattr(encoder, "evaluate", evaluate)
    first = asyncio.run(encoder.submit(run.config, prepared, owner))
    second = asyncio.run(encoder.submit(run.config, prepared, owner))
    assert first.id == second.id == "run"
    assert first.current_step == second.current_step == 128
    assert first.evaluation_phase == second.evaluation_phase == "complete"
    assert len(calls) == 1
