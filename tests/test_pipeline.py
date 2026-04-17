"""Tests for SendItPipeline — no GPU required.

Covers:
  * Dry-run plan generation for single-stage and full SFT->GRPO pipelines.
  * Validation failure surfaces as FAILED TrainingRun.
  * skip_credits + resume_from_checkpoint propagate to underlying trainers.
  * Post-SFT failure aborts without running GRPO.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from carl_studio.training.pipeline import (
    PipelineEvent,
    PipelinePlan,
    PipelineStage,
    SendItPipeline,
)
from carl_studio.types.config import TrainingConfig
from carl_studio.types.run import RunPhase, TrainingRun


def _make_config(**overrides: Any) -> TrainingConfig:
    defaults: dict[str, Any] = {
        "run_name": "test-pipeline",
        "base_model": "Qwen/Qwen3-8B",
        "output_repo": "test/test",
        "method": "sft",
        "dataset_repo": "trl-lib/Capybara",
        "compute_target": "local",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def test_plan_single_stage() -> None:
    pipeline = SendItPipeline(_make_config(method="sft"))
    plan = pipeline.plan()
    assert isinstance(plan, PipelinePlan)
    stage_names = [s[0] for s in plan.stages]
    assert "validate" in stage_names
    assert "sft" in stage_names
    assert "push" in stage_names
    assert plan.config_summary["Method"] == "sft"


def test_plan_grpo_full_pipeline() -> None:
    pipeline = SendItPipeline(_make_config(method="grpo", compute_target="l4x1"))
    plan = pipeline.plan()
    stage_names = [s[0] for s in plan.stages]
    # Full pipeline: validate → sft → sft_gate → grpo → grpo_gate → push
    for needed in ["validate", "sft", "sft_gate", "grpo", "grpo_gate", "push"]:
        assert needed in stage_names, f"missing stage {needed}"


# ---------------------------------------------------------------------------
# Validation failure path
# ---------------------------------------------------------------------------


def test_pipeline_validation_failure_missing_token() -> None:
    """Remote target + no HF token → FAILED TrainingRun with clear message."""
    pipeline = SendItPipeline(_make_config(compute_target="l4x1"))

    # Force no HF token discoverable.
    with patch("huggingface_hub.get_token", return_value=None), patch(
        "os.environ.get", return_value=None
    ):
        run = asyncio.run(pipeline.run())

    assert run.phase == RunPhase.FAILED
    assert "HF auth" in (run.error_message or "")


def test_pipeline_validation_failure_missing_torch_local() -> None:
    """Local target + no torch installed → validation failure."""
    pipeline = SendItPipeline(_make_config(compute_target="local"))

    real_import = __import__

    def stub_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("torch not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=stub_import):
        run = asyncio.run(pipeline.run())

    assert run.phase == RunPhase.FAILED
    assert "torch" in (run.error_message or "")


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


def test_pipeline_emits_validate_events() -> None:
    events: list[PipelineEvent] = []
    pipeline = SendItPipeline(
        _make_config(compute_target="l4x1"),
        on_event=events.append,
    )

    # Force validation failure so we stop early.
    with patch("huggingface_hub.get_token", return_value=None), patch(
        "os.environ.get", return_value=None
    ):
        asyncio.run(pipeline.run())

    stages = [e.stage for e in events]
    assert PipelineStage.VALIDATE in stages
    assert PipelineStage.FAILED in stages


# ---------------------------------------------------------------------------
# Flag propagation
# ---------------------------------------------------------------------------


def test_skip_credits_propagates_to_trainer_single_stage() -> None:
    """SendItPipeline(skip_credits=True) → CARLTrainer(skip_credits=True)."""
    pipeline = SendItPipeline(
        _make_config(method="sft", compute_target="l4x1"),
        skip_credits=True,
        resume_from_checkpoint="auto",
    )

    # Bypass validation + instantiate a mock trainer that records the args.
    with patch.object(pipeline, "_validate", return_value=[]), patch(
        "carl_studio.training.trainer.CARLTrainer"
    ) as mock_trainer_cls:
        fake_run = TrainingRun(
            id="fake", config=pipeline.config, phase=RunPhase.COMPLETE
        )
        mock_trainer = MagicMock()
        mock_trainer.train_and_watch = AsyncMock(return_value=fake_run)
        mock_trainer.is_remote = True
        mock_trainer_cls.return_value = mock_trainer

        with patch.object(pipeline, "_push_model", new=AsyncMock()):
            asyncio.run(pipeline.run())

    mock_trainer_cls.assert_called_once()
    _args, kwargs = mock_trainer_cls.call_args
    assert kwargs["skip_credits"] is True
    assert kwargs["resume_from_checkpoint"] == "auto"


def test_skip_credits_propagates_to_both_trainers_in_full_pipeline() -> None:
    """Full pipeline: both SFT and GRPO trainers receive skip_credits/resume args."""
    pipeline = SendItPipeline(
        _make_config(method="grpo", compute_target="l4x1"),
        skip_credits=True,
        resume_from_checkpoint=True,
    )

    constructed_kwargs: list[dict[str, Any]] = []

    def fake_trainer_cls(cfg: Any, **kwargs: Any) -> Any:
        constructed_kwargs.append(dict(kwargs))
        fake_run = TrainingRun(
            id=f"fake-{len(constructed_kwargs)}",
            config=cfg,
            phase=RunPhase.COMPLETE,
            hub_job_id=None,
        )
        trainer = MagicMock()
        trainer.train = AsyncMock(return_value=fake_run)
        trainer.is_remote = False  # Skip the watch() call.
        return trainer

    with patch.object(pipeline, "_validate", return_value=[]), patch(
        "carl_studio.training.trainer.CARLTrainer", side_effect=fake_trainer_cls
    ), patch.object(pipeline, "_check_gate", return_value=True), patch.object(
        pipeline, "_push_model", new=AsyncMock()
    ):
        asyncio.run(pipeline.run())

    assert len(constructed_kwargs) == 2, "expected SFT + GRPO trainer creations"
    for kwargs in constructed_kwargs:
        assert kwargs.get("skip_credits") is True
        assert kwargs.get("resume_from_checkpoint") is True


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------


def test_sft_failure_aborts_before_grpo() -> None:
    """If SFT returns FAILED, GRPO trainer is never constructed."""
    pipeline = SendItPipeline(
        _make_config(method="grpo", compute_target="l4x1"),
    )

    created: list[str] = []

    def fake_trainer_cls(cfg: Any, **_kwargs: Any) -> Any:
        created.append(cfg.method.value)
        fail_run = TrainingRun(
            id="sft-fail",
            config=cfg,
            phase=RunPhase.FAILED,
            error_message="synthetic sft failure",
        )
        trainer = MagicMock()
        trainer.train = AsyncMock(return_value=fail_run)
        trainer.is_remote = False
        return trainer

    with patch.object(pipeline, "_validate", return_value=[]), patch(
        "carl_studio.training.trainer.CARLTrainer", side_effect=fake_trainer_cls
    ):
        run = asyncio.run(pipeline.run())

    # Only SFT trainer was constructed.
    assert created == ["sft"]
    assert run.phase == RunPhase.FAILED
    assert "sft" in (run.error_message or "").lower()


def test_sft_gate_failure_aborts_before_grpo() -> None:
    pipeline = SendItPipeline(_make_config(method="grpo", compute_target="l4x1"))
    created: list[str] = []

    def fake_trainer_cls(cfg: Any, **_kwargs: Any) -> Any:
        created.append(cfg.method.value)
        ok_run = TrainingRun(id="sft-ok", config=cfg, phase=RunPhase.COMPLETE)
        trainer = MagicMock()
        trainer.train = AsyncMock(return_value=ok_run)
        trainer.is_remote = False
        return trainer

    with patch.object(pipeline, "_validate", return_value=[]), patch(
        "carl_studio.training.trainer.CARLTrainer", side_effect=fake_trainer_cls
    ), patch.object(pipeline, "_check_gate", return_value=False):
        run = asyncio.run(pipeline.run())

    assert created == ["sft"], "GRPO should not run when SFT gate fails"
    assert run.phase == RunPhase.FAILED
    assert "SFT eval gate failed" in (run.error_message or "")
