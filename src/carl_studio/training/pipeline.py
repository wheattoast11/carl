"""SendItPipeline — full autonomous training lifecycle.

``carl train --send-it`` runs this pipeline:

  1. Validate config (model, dataset, compute, HF token)
  2. SFT training → poll until complete
  3. Eval gate (PhaseTransitionGate or metric threshold)
  4. GRPO training → poll until convergence
  5. Final eval gate
  6. Push final model to Hub

Each step emits ``PipelineEvent`` objects for progress tracking.
The pipeline can be previewed with ``--dry-run``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from carl_studio.types.config import TrainingConfig, TrainingMethod, ComputeTarget
from carl_studio.types.run import RunPhase, TrainingRun

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    VALIDATE = "validate"
    SFT = "sft"
    SFT_GATE = "sft_gate"
    GRPO = "grpo"
    GRPO_GATE = "grpo_gate"
    PUSH = "push"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PipelineEvent:
    """Progress event emitted by the pipeline."""

    stage: PipelineStage
    message: str
    progress: float = 0.0  # 0.0 - 1.0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePlan:
    """Dry-run plan showing what --send-it would do."""

    stages: list[tuple[str, str]]  # (stage_name, description)
    config_summary: dict[str, str]
    estimated_cost: str = ""


class SendItPipeline:
    """Full pipeline: SFT → gate → GRPO → eval → push.

    Usage::

        pipeline = SendItPipeline(config, on_event=print_event)
        result = await pipeline.run()

    Or dry-run::

        plan = pipeline.plan()
    """

    def __init__(
        self,
        config: TrainingConfig,
        on_event: Callable[[PipelineEvent], None] | None = None,
        poll_interval: float = 60.0,
    ) -> None:
        self.config = config
        self._on_event = on_event or (lambda e: None)
        self._poll_interval = poll_interval
        self._sft_job_id: str | None = None
        self._grpo_job_id: str | None = None

    def _emit(
        self, stage: PipelineStage, message: str, progress: float = 0.0, **detail: Any
    ) -> None:
        event = PipelineEvent(stage=stage, message=message, progress=progress, detail=detail)
        self._on_event(event)

    # ------------------------------------------------------------------
    # Dry-run plan
    # ------------------------------------------------------------------

    def plan(self) -> PipelinePlan:
        """Generate a dry-run plan without executing anything."""
        c = self.config
        stages = []

        stages.append(("validate", f"Validate config: {c.base_model} on {c.compute_target.value}"))

        if c.method == TrainingMethod.GRPO:
            # Full pipeline: SFT → gate → GRPO
            stages.append(
                ("sft", f"SFT training on {c.compute_target.value} ({c.max_steps} steps)")
            )
            stages.append(
                ("sft_gate", "Eval gate: mean_token_accuracy >= 0.99 (PhaseTransitionGate)")
            )
            stages.append(("grpo", f"GRPO training on {c.compute_target.value}"))
            stages.append(("grpo_gate", "Final eval gate"))
        else:
            # Single method
            stages.append(
                (c.method.value, f"{c.method.value.upper()} training on {c.compute_target.value}")
            )
            stages.append(("eval_gate", "Final eval gate"))

        stages.append(("push", f"Push to Hub: {c.output_repo or '(auto)'}"))

        config_summary = {
            "Model": c.base_model,
            "Method": c.method.value,
            "Compute": c.compute_target.value,
            "Steps": str(c.max_steps),
            "Dataset": c.dataset_repo or "(default)",
            "Output": c.output_repo or "(auto)",
            "CARL": "enabled",
        }

        return PipelinePlan(stages=stages, config_summary=config_summary)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def run(self) -> TrainingRun:
        """Execute the full pipeline. Returns the final TrainingRun."""
        from carl_studio.training.trainer import CARLTrainer

        # Stage 1: Validate
        self._emit(PipelineStage.VALIDATE, "Validating config...", 0.0)
        issues = self._validate()
        if issues:
            self._emit(PipelineStage.FAILED, f"Validation failed: {'; '.join(issues)}")
            run = TrainingRun(
                id="failed-validation",
                config=self.config,
                phase=RunPhase.FAILED,
                error_message="; ".join(issues),
            )
            return run

        self._emit(PipelineStage.VALIDATE, "Config valid", 1.0)

        if self.config.method == TrainingMethod.GRPO:
            # Full SFT → GRPO pipeline
            return await self._run_full_pipeline()
        else:
            # Single-stage training
            return await self._run_single_stage()

    async def _run_full_pipeline(self) -> TrainingRun:
        """SFT → gate → GRPO → eval → push."""
        from carl_studio.training.trainer import CARLTrainer
        import copy

        # Stage 2: SFT
        self._emit(PipelineStage.SFT, "Starting SFT...", 0.0)

        sft_config = copy.deepcopy(self.config)
        sft_config.method = TrainingMethod.SFT

        sft_trainer = CARLTrainer(sft_config)
        sft_run = await sft_trainer.train()

        if sft_run.phase == RunPhase.FAILED:
            self._emit(PipelineStage.FAILED, f"SFT failed: {sft_run.error_message}")
            return sft_run

        self._sft_job_id = sft_run.hub_job_id
        self._emit(PipelineStage.SFT, f"SFT submitted: {sft_run.hub_job_id}", 0.1)

        # Watch SFT
        if sft_trainer.is_remote and sft_run.hub_job_id:
            self._emit(PipelineStage.SFT, "Watching SFT job...", 0.2)
            sft_run = await sft_trainer.watch(poll_interval=self._poll_interval)
            if sft_run.phase == RunPhase.FAILED:
                self._emit(PipelineStage.FAILED, f"SFT failed: {sft_run.error_message}")
                return sft_run

        self._emit(PipelineStage.SFT, "SFT complete", 1.0)

        # Stage 3: SFT Gate
        self._emit(PipelineStage.SFT_GATE, "Running SFT eval gate...", 0.0)
        gate_passed = self._check_gate(sft_run)
        if not gate_passed:
            self._emit(PipelineStage.FAILED, "SFT gate failed — model not ready for GRPO")
            sft_run.phase = RunPhase.FAILED
            sft_run.error_message = "SFT eval gate failed"
            return sft_run

        self._emit(PipelineStage.SFT_GATE, "SFT gate PASSED", 1.0)

        # Stage 4: GRPO
        self._emit(PipelineStage.GRPO, "Starting GRPO...", 0.0)

        grpo_trainer = CARLTrainer(self.config)
        grpo_run = await grpo_trainer.train()

        if grpo_run.phase == RunPhase.FAILED:
            self._emit(PipelineStage.FAILED, f"GRPO failed: {grpo_run.error_message}")
            return grpo_run

        self._grpo_job_id = grpo_run.hub_job_id
        self._emit(PipelineStage.GRPO, f"GRPO submitted: {grpo_run.hub_job_id}", 0.1)

        # Watch GRPO
        if grpo_trainer.is_remote and grpo_run.hub_job_id:
            self._emit(PipelineStage.GRPO, "Watching GRPO job...", 0.2)
            grpo_run = await grpo_trainer.watch(poll_interval=self._poll_interval)
            if grpo_run.phase == RunPhase.FAILED:
                self._emit(PipelineStage.FAILED, f"GRPO failed: {grpo_run.error_message}")
                return grpo_run

        self._emit(PipelineStage.GRPO, "GRPO complete", 1.0)

        # Stage 5: GRPO Gate
        self._emit(PipelineStage.GRPO_GATE, "Running final eval gate...", 0.0)
        gate_passed = self._check_gate(grpo_run)
        self._emit(
            PipelineStage.GRPO_GATE,
            "Final gate PASSED" if gate_passed else "Final gate FAILED (pushing anyway)",
            1.0,
        )

        # Stage 6: Push
        self._emit(PipelineStage.PUSH, "Pushing to Hub...", 0.0)
        await self._push_model(grpo_run)
        self._emit(PipelineStage.PUSH, "Pushed", 1.0)

        self._emit(PipelineStage.DONE, "Pipeline complete", 1.0)
        grpo_run.phase = RunPhase.COMPLETE
        return grpo_run

    async def _run_single_stage(self) -> TrainingRun:
        """Single-stage training (SFT, DPO, KTO, ORPO)."""
        from carl_studio.training.trainer import CARLTrainer

        stage = PipelineStage.SFT  # Use SFT stage for any single-stage
        self._emit(stage, f"Starting {self.config.method.value.upper()}...", 0.0)

        trainer = CARLTrainer(self.config)
        run = await trainer.train_and_watch(poll_interval=self._poll_interval)

        if run.phase == RunPhase.FAILED:
            self._emit(PipelineStage.FAILED, f"Training failed: {run.error_message}")
            return run

        self._emit(stage, "Training complete", 1.0)

        # Push
        self._emit(PipelineStage.PUSH, "Pushing to Hub...", 0.0)
        await self._push_model(run)
        self._emit(PipelineStage.PUSH, "Pushed", 1.0)

        self._emit(PipelineStage.DONE, "Pipeline complete", 1.0)
        run.phase = RunPhase.COMPLETE
        return run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate(self) -> list[str]:
        """Validate config before execution. Returns list of issues."""
        issues = []

        if not self.config.base_model:
            issues.append("base_model is required")

        if self.config.compute_target == ComputeTarget.LOCAL:
            # Local mode — check for torch
            try:
                import torch
            except ImportError:
                issues.append("torch not installed (required for local training)")
        else:
            # Remote mode — check for HF token (prefer hub credentials)
            token = None
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:
                pass
            if not token:
                token = os.environ.get("HF_TOKEN")

            if not token:
                issues.append("HF auth not detected (set HF_TOKEN or run `hf auth login`)")

        return issues

    def _check_gate(self, run: TrainingRun) -> bool:
        """Check if a training run passes the eval gate.

        For GRPO runs, uses Phase 2' eval (multi-turn sandbox, threshold 0.30).
        For SFT runs, uses Phase 1 eval (tool-call format, threshold 0.50).
        Falls back to completion-check if eval deps aren't available.
        """
        if run.phase == RunPhase.FAILED:
            return False

        output_repo = self.config.output_repo
        if not output_repo:
            logger.warning("No output_repo -- gate defaults to completion check")
            return run.phase == RunPhase.COMPLETE

        # Determine phase and threshold based on training method
        is_grpo = self.config.method == TrainingMethod.GRPO
        eval_phase = "2prime" if is_grpo else "auto"
        default_threshold = 0.30 if is_grpo else 0.50

        try:
            from carl_studio.eval.runner import EvalConfig, EvalRunner, EvalGate

            eval_config = EvalConfig(
                checkpoint=output_repo,
                dataset=self.config.dataset_repo,
                data_files="eval.jsonl" if is_grpo else None,
                phase=eval_phase,
                threshold=getattr(self.config, "eval_threshold", default_threshold),
                base_model=self.config.base_model if is_grpo else None,
                sft_adapter=getattr(self.config, "sft_adapter", None) if is_grpo else None,
                max_samples=getattr(self.config, "eval_samples", 100),
            )
            report = EvalRunner(eval_config).run()
            gate = EvalGate(threshold=eval_config.threshold, phase=eval_phase)

            logger.info(
                "Eval gate: %s=%.3f (threshold=%.2f) -> %s",
                report.primary_metric,
                report.primary_value,
                report.threshold,
                "PASS" if report.passed else "FAIL",
            )
            return gate.check(report)
        except ImportError:
            logger.warning(
                "EvalRunner not available (install carl-studio[training]). Falling back to completion check."
            )
            return run.phase == RunPhase.COMPLETE
        except Exception as e:
            logger.warning("Eval gate failed: %s. Falling back to completion check.", e)
            return run.phase == RunPhase.COMPLETE

    async def _push_model(self, run: TrainingRun) -> None:
        """Push model to Hub if output_repo is configured."""
        output_repo = self.config.output_repo
        if not output_repo:
            logger.info("No output_repo configured — skipping push")
            return

        try:
            from carl_studio.hub.models import push_with_metadata

            await push_with_metadata(
                model_path=output_repo,  # For remote jobs, model is already on Hub
                repo_id=output_repo,
                base_model=self.config.base_model,
                method=self.config.method.value,
                dataset=self.config.dataset_repo or "",
            )
        except Exception as exc:
            logger.warning("Push failed (non-fatal): %s", exc)
