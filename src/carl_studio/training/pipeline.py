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
        *,
        skip_credits: bool = False,
        resume_from_checkpoint: bool | str | None = None,
        task_reward_funcs: list[Any] | None = None,
        task_reward_weights: list[float] | None = None,
        task_reward_stages: list[str] | None = None,
        evaluator: Any = None,
        primary_metric: str | None = None,
    ) -> None:
        self.config = config
        self._on_event = on_event or (lambda e: None)
        self._poll_interval = poll_interval
        self.skip_credits = bool(skip_credits)
        self.resume_from_checkpoint = resume_from_checkpoint
        self._sft_job_id: str | None = None
        self._grpo_job_id: str | None = None
        self._trainer_bindings = (
            {
                "task_reward_funcs": task_reward_funcs,
                "task_reward_weights": task_reward_weights,
                "task_reward_stages": task_reward_stages,
            }
            if task_reward_funcs is not None
            else {}
        )
        self._evaluator = evaluator
        self._primary_metric = primary_metric
        self.active_run: TrainingRun | None = None

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

        if c.push_to_hub:
            stages.append(("push", f"Push to Hub after evaluation: {c.output_repo}"))

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
        sft_config.push_to_hub = (
            self.config.compute_target != ComputeTarget.LOCAL and self.config.push_to_hub
        )
        sft_config.hub_private = True
        if self.config.output_dir is not None:
            sft_config.output_dir = self.config.output_dir / "sft"

        sft_trainer = CARLTrainer(
            sft_config,
            skip_credits=self.skip_credits,
            resume_from_checkpoint=self.resume_from_checkpoint,
            **self._trainer_bindings,
        )
        self.active_run = sft_trainer.run
        sft_run = await sft_trainer.train()
        self.active_run = sft_run

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
        gate_passed = await self._evaluate_gate(sft_run)
        if not gate_passed:
            self._emit(PipelineStage.FAILED, "SFT gate failed — model not ready for GRPO")
            sft_run.phase = RunPhase.FAILED
            sft_run.error_message = "SFT eval gate failed"
            return sft_run

        self._emit(PipelineStage.SFT_GATE, "SFT gate PASSED", 1.0)

        # Stage 4: GRPO
        self._emit(PipelineStage.GRPO, "Starting GRPO...", 0.0)

        grpo_config = self.config.model_copy(deep=True)
        grpo_config.starting_adapters = [
            *self.config.starting_adapters,
            *([self.config.sft_adapter] if self.config.sft_adapter else []),
        ]
        grpo_config.sft_adapter = sft_run.checkpoint
        grpo_config.push_to_hub = (
            self.config.compute_target != ComputeTarget.LOCAL and self.config.push_to_hub
        )
        grpo_config.hub_private = True
        if self.config.output_dir is not None:
            grpo_config.output_dir = self.config.output_dir / "grpo"
        grpo_trainer = CARLTrainer(
            grpo_config,
            skip_credits=self.skip_credits,
            resume_from_checkpoint=self.resume_from_checkpoint,
            **self._trainer_bindings,
        )
        self.active_run = grpo_trainer.run
        grpo_run = await grpo_trainer.train()
        self.active_run = grpo_run
        grpo_run.artifacts = [*sft_run.artifacts, *grpo_run.artifacts]

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
        gate_passed = await self._evaluate_gate(grpo_run)
        self._emit(
            PipelineStage.GRPO_GATE,
            "Final gate PASSED" if gate_passed else "Final gate FAILED",
            1.0,
        )
        if not gate_passed:
            grpo_run.phase = RunPhase.FAILED
            grpo_run.error_message = "Final eval gate failed"
            return grpo_run

        # Stage 6: Push
        if not await self._publish_accepted(grpo_run):
            return grpo_run

        self._emit(PipelineStage.DONE, "Pipeline complete", 1.0)
        grpo_run.phase = RunPhase.COMPLETE
        return grpo_run

    async def _run_single_stage(self) -> TrainingRun:
        """Single-stage training (SFT, DPO, KTO, ORPO)."""
        from carl_studio.training.trainer import CARLTrainer

        stage = PipelineStage.SFT  # Use SFT stage for any single-stage
        self._emit(stage, f"Starting {self.config.method.value.upper()}...", 0.0)

        stage_config = self.config.model_copy(deep=True)
        stage_config.push_to_hub = (
            self.config.compute_target != ComputeTarget.LOCAL and self.config.push_to_hub
        )
        stage_config.hub_private = True
        trainer = CARLTrainer(
            stage_config,
            skip_credits=self.skip_credits,
            resume_from_checkpoint=self.resume_from_checkpoint,
            **self._trainer_bindings,
        )
        self.active_run = trainer.run
        run = await trainer.train_and_watch(poll_interval=self._poll_interval)
        self.active_run = run

        if run.phase == RunPhase.FAILED:
            self._emit(PipelineStage.FAILED, f"Training failed: {run.error_message}")
            return run

        self._emit(stage, "Training complete", 1.0)

        if not await self._evaluate_gate(run):
            run.phase = RunPhase.FAILED
            run.error_message = "Final eval gate failed"
            self._emit(PipelineStage.FAILED, run.error_message)
            return run

        # Push
        if not await self._publish_accepted(run):
            return run

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

    async def _evaluate_gate(self, run: TrainingRun) -> bool:
        from carl_studio.training.preparation import run_in_worker

        return await run_in_worker(lambda: self._check_gate(run))

    def _check_gate(self, run: TrainingRun) -> bool:
        """Check if a training run passes the eval gate.

        For GRPO runs, uses Phase 2' eval (multi-turn sandbox, threshold 0.30).
        For SFT runs, uses Phase 1 eval (tool-call format, threshold 0.50).
        Missing evaluation or artifacts cannot satisfy the gate.
        """
        if run.phase == RunPhase.FAILED:
            return False

        checkpoint = run.checkpoint
        if not checkpoint:
            logger.warning("No stage checkpoint available for evaluation")
            return False

        # Determine phase and threshold based on training method
        is_grpo = run.config.method == TrainingMethod.GRPO
        eval_phase = "1" if self._evaluator is not None else ("2prime" if is_grpo else "auto")
        default_threshold = 0.30 if is_grpo else 0.50

        try:
            from carl_studio.eval.runner import EvalConfig, EvalRunner, EvalGate

            eval_config = EvalConfig(
                checkpoint=checkpoint,
                dataset=run.config.eval_dataset_repo or run.config.dataset_repo,
                dataset_split=run.config.eval_split,
                phase=eval_phase,
                threshold=(
                    run.config.goal.threshold
                    if run.config.goal
                    else getattr(self.config, "eval_threshold", default_threshold)
                ),
                metric_direction=run.config.goal.direction if run.config.goal else "higher",
                base_model=run.config.base_model,
                sft_adapter=run.config.sft_adapter,
                starting_adapters=run.config.starting_adapters,
                tokenizer_source=run.config.tokenizer_source,
                max_samples=(
                    run.config.goal.max_eval_samples
                    if run.config.goal
                    else getattr(self.config, "eval_samples", 100)
                ),
                max_new_tokens=min(max(run.config.max_completion_length, 64), 512),
                **(
                    {
                        "coherence_phi_floor": run.config.goal.coherence_phi_floor,
                        "discontinuity_min": run.config.goal.discontinuity_min,
                        "discontinuity_max": run.config.goal.discontinuity_max,
                    }
                    if run.config.goal
                    else {}
                ),
            )
            runner_args = (
                {"evaluator": self._evaluator, "primary_metric": self._primary_metric}
                if self._evaluator is not None
                else {}
            )
            report = EvalRunner(eval_config, **runner_args).run()
            gate = EvalGate(
                threshold=eval_config.threshold,
                phase=eval_phase,
                config=eval_config,
            )
            gate_result = gate.check(report)

            logger.info(
                "Eval gate: %s=%.3f (threshold=%.2f) -> %s | %s",
                report.primary_metric,
                report.primary_value,
                report.threshold,
                "PASS" if gate_result else "FAIL",
                report.gate_reason or "(no reason recorded)",
            )
            return gate_result
        except ImportError:
            logger.warning(
                "EvalRunner not available (install carl-studio[training]); evaluation failed"
            )
            return False
        except Exception as e:
            logger.warning(
                "Eval gate failed: %s. Treating as FAIL.",
                type(e).__name__ if run.config.goal is not None else e,
            )
            return False

    async def _push_model(self, run: TrainingRun) -> None:
        """Push model to Hub if output_repo is configured."""
        output_repo = self.config.output_repo
        if not self.config.push_to_hub or not output_repo:
            logger.info("No output_repo configured — skipping push")
            return

        from carl_studio.hub.models import push_with_metadata

        if not run.checkpoint:
            raise ValueError("Publication requires a completed checkpoint")
        if self.config.compute_target != ComputeTarget.LOCAL:
            raise ValueError(
                "Remote candidate publication requires explicit artifact reconciliation"
            )
        await push_with_metadata(
            model_path=run.checkpoint,
            repo_id=output_repo,
            base_model=self.config.base_model,
            method=self.config.method.value,
            dataset=self.config.dataset_repo or "",
        )

    async def _publish_accepted(self, run: TrainingRun) -> bool:
        if not self.config.push_to_hub:
            return True
        if self.config.compute_target != ComputeTarget.LOCAL:
            self._emit(
                PipelineStage.PUSH,
                "Evaluated private remote checkpoint retained; public publication is separate",
                1.0,
            )
            return True
        self._emit(PipelineStage.PUSH, "Publishing evaluated artifact", 0.0)
        try:
            await self._push_model(run)
        except Exception:
            run.phase = RunPhase.FAILED
            run.error_message = "Artifact publication failed"
            self._emit(PipelineStage.FAILED, run.error_message)
            return False
        self._emit(PipelineStage.PUSH, "Published", 1.0)
        return True


async def submit_training(
    config: TrainingConfig,
    *,
    prepared_plan_id: str | None = None,
    manager: Any = None,
    skip_credits: bool = False,
    resume_from_checkpoint: bool | str | None = None,
) -> TrainingRun:
    """Shared CLI/MCP submission through the current trainer and adapter owners."""
    from carl_studio.training.trainer import CARLTrainer

    if prepared_plan_id is None:
        if config.goal is not None or config.method == TrainingMethod.ENCODER:
            from carl_core.errors import ValidationError

            raise ValidationError(
                "Prepare and review the goal before training", code="carl.preparation.required"
            )
        if config.adapter != "trl":
            from carl_studio.adapters.registry import get_adapter

            job = get_adapter(config.adapter).submit(config.model_dump(mode="json"))
            return TrainingRun(id=job.run_id, config=config, phase=RunPhase.PROVISIONING)
        if config.pipeline:
            return await SendItPipeline(
                config, skip_credits=skip_credits, resume_from_checkpoint=resume_from_checkpoint
            ).run()
        return await CARLTrainer(
            config, skip_credits=skip_credits, resume_from_checkpoint=resume_from_checkpoint
        ).train()

    import asyncio
    import time
    from functools import partial
    from pathlib import Path

    from carl_core.errors import ValidationError

    from carl_studio.eval.runner import EvalConfig, EvalRunner
    from carl_studio.experiment.types import Witness
    from carl_studio.training.acceptance import compare_candidate, goal_progress
    from carl_studio.training.preparation import (
        default_manager,
        load_preparation,
        resolve_callable,
        run_in_worker,
        validate_preparation,
    )
    from carl_studio.types.preparation import EvaluationMeasurement

    owner = manager or default_manager()
    prepared = load_preparation(prepared_plan_id, owner)
    if config.method == TrainingMethod.ENCODER:
        from carl_studio.training.encoder import submit
        return await submit(config, prepared, owner)
    await run_in_worker(partial(validate_preparation, prepared, config))
    existing = owner.load_training_result(prepared.plan_id)
    if existing is not None and not (
        resume_from_checkpoint and existing.phase in {RunPhase.PAUSED, RunPhase.FAILED}
    ):
        return existing
    config = TrainingConfig.model_validate(prepared.config)
    if resume_from_checkpoint is not None:
        supplied_resume = str(Path(resume_from_checkpoint).resolve())
        if supplied_resume != config.resume_from_checkpoint:
            raise ValidationError(
                "Resume checkpoint does not match prepared inputs", code="carl.preparation.resume"
            )
    root = Path(prepared.project_root)
    eval_source = next(source for source in prepared.sources if source.kind == "eval_data")
    owner.claim_training(prepared.plan_id, resume=bool(resume_from_checkpoint))
    started = time.monotonic()
    run = TrainingRun(id=prepared.plan_id, config=config)
    pipeline: SendItPipeline | None = None

    async def read_measurement(
        checkpoint: str, *, candidate: bool, stage_config: TrainingConfig | None = None
    ) -> EvaluationMeasurement:
        measured_config = stage_config or config
        compare_base = not candidate and config.comparison_baseline == "base"
        sft_adapter = None if compare_base else measured_config.sft_adapter
        starting_adapters = [] if compare_base else measured_config.starting_adapters
        evaluation = EvalConfig(
            checkpoint=checkpoint,
            base_model=config.base_model if candidate or sft_adapter else None,
            sft_adapter=sft_adapter,
            starting_adapters=starting_adapters,
            tokenizer_source=measured_config.tokenizer_source,
            dataset=config.eval_dataset_repo or config.dataset_repo,
            dataset_split=config.eval_split,
            phase="1",
            threshold=prepared.goal.threshold,
            metric_direction=prepared.goal.direction,
            max_samples=prepared.goal.max_eval_samples,
            max_new_tokens=min(max(config.max_completion_length, 64), 512),
            coherence_phi_floor=prepared.goal.coherence_phi_floor,
            discontinuity_min=prepared.goal.discontinuity_min,
            discontinuity_max=prepared.goal.discontinuity_max,
        )
        runner = EvalRunner(
            evaluation, evaluator=evaluator, primary_metric=prepared.goal.primary_metric
        )
        report = await run_in_worker(runner.run)
        values = {}
        for policy in prepared.goal.policies:
            result = policies[policy.id](runner.last_completions, runner.last_samples)
            if not isinstance(result, dict) or policy.id not in result:
                raise ValidationError(
                    "A policy did not report its declared metric", code="carl.preparation.policy"
                )
            values[policy.id] = float(result[policy.id])
        if report.n_samples != len(prepared.eval_sample_ids):
            raise ValidationError(
                "The evaluation population changed", code="carl.preparation.population"
            )
        return EvaluationMeasurement(
            checkpoint=checkpoint,
            dataset_sha256=eval_source.sha256,
            sample_ids=prepared.eval_sample_ids,
            primary_metric=report.primary_metric,
            primary_value=report.primary_value,
            metrics=report.metrics,
            coherence=report.coherence,
            coherence_source="full_logits" if report.coherence else "unavailable",
            policy_results=values,
            generation={
                "max_new_tokens": evaluation.max_new_tokens,
                "seed": config.seed,
                "sampling": "greedy",
                "phase": evaluation.phase,
            },
        )

    try:
        evaluator = resolve_callable(prepared.goal.evaluator, root)
        rewards = [
            resolve_callable(binding.reference, root, reward=True) for binding in prepared.goal.rewards
        ]
        policies = {
            policy.id: resolve_callable(policy.reference, root) for policy in prepared.goal.policies
        }
        baseline = await read_measurement(config.base_model, candidate=False)
        await run_in_worker(partial(validate_preparation, prepared))
        trainer_args = {
            "skip_credits": skip_credits,
            "resume_from_checkpoint": resume_from_checkpoint or config.resume_from_checkpoint,
            "task_reward_funcs": rewards,
            "task_reward_weights": [binding.weight for binding in prepared.goal.rewards],
            "task_reward_stages": [binding.stage for binding in prepared.goal.rewards],
        }
        if config.pipeline:
            pipeline = SendItPipeline(
                config,
                evaluator=evaluator,
                primary_metric=prepared.goal.primary_metric,
                **trainer_args,
            )
            owner.start(prepared.plan_id, run.id)
            run = await pipeline.run()
        else:
            trainer = CARLTrainer(config, **trainer_args)
            run = trainer.run
            owner.start(prepared.plan_id, run.id)
            run = await trainer.train()
        if run.phase != RunPhase.COMPLETE or not run.checkpoint:
            return run
        candidate = await read_measurement(run.checkpoint, candidate=True, stage_config=run.config)
        await run_in_worker(partial(validate_preparation, prepared))
        run.acceptance = compare_candidate(prepared.goal, baseline, candidate)
        progress, target, improved = goal_progress(
            prepared.goal, baseline.primary_value, candidate.primary_value
        )
        goal_passed = target and improved
        owner.add_witness(
            prepared.plan_id,
            Witness(
                prediction_id="P_goal",
                observed_value=progress,
                observed_at=run.id,
                supports=goal_passed,
                detail={"baseline": baseline.primary_value, "candidate": candidate.primary_value},
            ),
        )
        if candidate.coherence is not None and "phi_mean" in candidate.coherence:
            owner.add_witness(
                prepared.plan_id,
                Witness(
                    prediction_id="P_coherence",
                    observed_value=candidate.coherence["phi_mean"],
                    observed_at=run.id,
                    supports=run.acceptance.coherence_passed,
                ),
            )
        for policy in prepared.goal.policies:
            if policy.id in candidate.policy_results:
                owner.add_witness(
                    prepared.plan_id,
                    Witness(
                        prediction_id=f"P_policy_{policy.id}",
                        observed_value=candidate.policy_results[policy.id],
                        observed_at=run.id,
                        supports=candidate.policy_results[policy.id] >= policy.threshold,
                    ),
                )
        owner.judge(prepared.plan_id)
        return run
    except asyncio.CancelledError:
        if pipeline is not None and pipeline.active_run is not None:
            run = pipeline.active_run
        run.phase = RunPhase.PAUSED
        raise
    except Exception:  # noqa: BLE001 - declared callables share this sanitized boundary
        run.phase = RunPhase.FAILED
        run.error_message = "Prepared training or evaluation failed"
        raise ValidationError(run.error_message, code="carl.preparation.execution") from None
    finally:
        run.resource_usage["elapsed_seconds"] = time.monotonic() - started
        owner.save_training_result(prepared.plan_id, run)
