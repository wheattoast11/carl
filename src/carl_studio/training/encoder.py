"""Encoder branch of the existing preparation and training lifecycle."""

from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from carl_core.errors import CARLError, ValidationError
from carl_core.hashing import content_hash

from carl_studio.experiment.types import Hypothesis, Prediction, PredictionComparator
from carl_studio.semantic.learning import EncoderExample, validate_splits
from carl_studio.semantic.local import invoke, materialize_windows, source_identity
from carl_studio.semantic.types import ExecutionBinding
from carl_studio.session import Session
from carl_studio.types.config import ComputeTarget, TrainingConfig
from carl_studio.types.preparation import (
    ReadinessIssue,
    SourceBinding,
    TrainingGoal,
    TrainingPreparation,
)
from carl_studio.types.run import RunPhase, TrainingRun


def implementation_sources() -> list[Path]:
    from carl_core import encoder, encoder_settings
    from carl_encoders import worker

    semantic = Path(__file__).parent.parent / "semantic"
    package = Path(worker.__file__).parent
    from carl_studio.training.preparation import implementation_sources

    return [
        Path(__file__).resolve(),
        *sorted(semantic.glob("*.py")),
        *sorted(package.glob("*.py")),
        Path(encoder.__file__).resolve(),
        Path(encoder_settings.__file__).resolve(),
        *implementation_sources(),
    ]


def groups(config: TrainingConfig) -> dict[str, list[EncoderExample]]:
    from carl_studio.training.preparation import read_samples

    if config.encoder is None:
        raise ValueError("Encoder settings required")
    result = {
        name: [EncoderExample.model_validate(row) for row in read_samples(Path(path))]
        for name, path in (
            ("train", config.dataset_repo),
            ("validation", config.encoder.validation_dataset),
            ("test", config.eval_dataset_repo or ""),
        )
    }
    validate_splits(result, config.encoder.cutoff)
    return result


def carrier_binding(config: TrainingConfig) -> dict[str, Any]:
    """Bind reusable embeddings independently of the ranking-head recipe."""
    if config.encoder is None or config.encoder.execution is None:
        raise ValueError("Prepared encoder execution required")
    producer = config.encoder.execution.model_copy(
        update={
            "trainable_modules": (),
            "dependencies": {
                key: value
                for key, value in config.encoder.execution.dependencies.items()
                if not key.startswith("carl.encoder.fit.")
            },
        }
    )
    return {
        "model": source_identity(Path(config.base_model))[0],
        "execution": producer.model_dump(mode="json"),
        "processed_tokens": config.encoder.processed_tokens,
        "encoding_batch_size": config.encoder.encode_batch_size,
    }


def cache_embeddings(
    config: TrainingConfig, prepared: TrainingPreparation, manifest: Path
) -> dict[str, Any]:
    """Materialize reusable carriers through the prepared worker and artifact owners."""
    from carl_studio.settings import carl_home

    validate(prepared, config)
    bound = TrainingConfig.model_validate(prepared.config)
    settings = bound.encoder
    if settings is None or settings.mode != "frozen_heads":
        raise ValueError("Carrier materialization requires frozen encoder settings")
    rows = {
        name: [row.model_dump(mode="json") for row in items]
        for name, items in groups(bound).items()
    }
    if any(
        part["modality"] not in {"text", "structured"}
        for items in rows.values()
        for row in items
        for sample in (row["query"], row["positive"], *row["negatives"])
        for part in sample["parts"]
    ):
        raise ValueError("Cached carrier preparation currently requires text or structured inputs")
    with Session() as session:
        result = invoke(
            session,
            Path(settings.interpreter),
            "cache",
            {
                "model": bound.base_model,
                "groups": rows,
                "settings": settings.model_dump(mode="json"),
                "cache_binding": carrier_binding(bound),
                "manifest": str(manifest.resolve()),
                "artifact_dir": str(carl_home() / "artifacts" / "semantic"),
            },
            timeout=settings.runtime_s,
        )
        validate(prepared)
        reference = session.data_vault.open_file(manifest, content_type="application/json")
        reference = reference.model_copy(update={"sha256": result["sha256"]})
        session.data_vault.restore_file(reference)
        result["artifact"] = reference.describe()
    return result


def prepare(
    config: TrainingConfig,
    root: Path,
    *,
    manager: Any = None,
    config_path: Path | None = None,
    persist: bool = True,
) -> TrainingPreparation:
    """Bind offline data, worker environment and model without executing learning."""
    from carl_studio.training.preparation import (
        default_manager,
        file_hash,
        model_sources,
        preparation_identity,
    )

    config = config.model_copy(deep=True)
    owner = manager or default_manager()
    issues: list[ReadinessIssue] = []
    sources: list[SourceBinding] = []

    def issue(code: str, message: str) -> None:
        issues.append(
            ReadinessIssue(
                code=code, message=message, action="Provide matching local encoder inputs"
            )
        )

    if config.encoder is None:
        raise ValidationError("Encoder settings required", code="carl.encoder.config")
    from carl_studio.adapters.registry import get_capabilities

    try:
        available_methods = get_capabilities(config.adapter).get("encoder_local_prepared", [])
        if not isinstance(available_methods, list) or config.encoder.mode not in available_methods:
            issue("encoder_backend", "Selected backend has no local prepared encoder integration")
    except CARLError:
        issue("encoder_backend", "Selected training backend is unregistered")
    if config.compute_target != ComputeTarget.LOCAL or config.pipeline or config.push_to_hub:
        issue("encoder_effects", "Encoder pilots require local compute and separate output")
    if config.max_steps != config.encoder.optimizer_steps or config.seed != config.encoder.seed:
        issue("encoder_budget", "Top-level steps and seed must match encoder pilot settings")
    for field in ("base_model", "dataset_repo", "eval_dataset_repo", "output_dir"):
        value = getattr(config, field)
        if value:
            path = Path(value)
            setattr(config, field, str((path if path.is_absolute() else root / path).resolve()))
    settings = config.encoder
    validation = Path(settings.validation_dataset)
    validation = (validation if validation.is_absolute() else root / validation).resolve()
    config.encoder = settings.model_copy(update={"validation_dataset": str(validation)})
    settings = config.encoder
    populations: dict[str, list[EncoderExample]] = {}
    try:
        populations = groups(config)
    except (ValueError, OSError):
        issue(
            "encoder_data", "Encoder examples, grouped splits or temporal cutoff failed admission"
        )
    required_slices: set[str] = set()
    for rows in populations.values():
        for row in rows:
            media = {
                part.modality
                for sample in (row.query, row.positive, *row.negatives)
                for part in sample.parts
            } & {"image", "audio", "video"}
            for modality in media | {"multimodal"} if media else {"text"}:
                required_slices.update(
                    modality + ":" + str(d)
                    for d in ((256, 512, 768) if media else (128, 256, 512, 768))
                )
    if not required_slices <= set(settings.required_slices):
        issue(
            "encoder_retention",
            "Required per-rung and per-modality retention slices are undeclared",
        )
    for kind, path in (
        ("train_data", config.dataset_repo),
        ("eval_data", config.eval_dataset_repo),
        ("callable", str(validation)),
    ):
        if path and Path(path).is_file():
            sources.append(
                SourceBinding(
                    kind=cast(Literal["train_data", "eval_data", "callable"], kind),
                    path=str(path),
                    sha256=file_hash(Path(path)),
                )
            )
    if config_path is not None:
        sources.append(
            SourceBinding(
                kind="config", path=str(config_path.resolve()), sha256=file_hash(config_path)
            )
        )
    for source in model_sources(config, root):
        sources.append(SourceBinding(kind="model", path=str(source), sha256=file_hash(source)))
    if not Path(config.base_model).is_dir() or not sources:
        issue("encoder_source", "Local encoder weights unavailable")
    if config.output_dir is None:
        issue("encoder_output", "Choose a candidate output directory")
    elif (
        Path(config.output_dir).resolve() == Path(config.base_model).resolve()
        or Path(config.base_model).resolve() in Path(config.output_dir).resolve().parents
    ):
        issue("encoder_output", "Candidate output overlaps base weights")
    from urllib.parse import unquote, urlparse

    with Session() as session:
        for rows in populations.values():
            for row in rows:
                for ref in row.source_artifacts.values():
                    try:
                        session.data_vault.restore_file(ref)
                        path = Path(unquote(urlparse(ref.uri).path))
                        sources.append(
                            SourceBinding(kind="artifact", path=str(path), sha256=file_hash(path))
                        )
                    except (ValueError, OSError, RuntimeError, CARLError):
                        issue("encoder_artifact", "Media source descriptor failed restoration")
    if (
        config.output_dir is not None
        and (Path(config.output_dir) / "trainer_state.json").exists()
        and not config.resume_from_checkpoint
    ):
        issue("encoder_output", "Candidate output already contains an execution checkpoint")
    for source in implementation_sources():
        sources.append(SourceBinding(kind="callable", path=str(source), sha256=file_hash(source)))
    try:
        _, processor = source_identity(Path(config.base_model))
        with Session() as session:
            operation = "qualify" if settings.mode == "adapter" else "metadata"
            request = {"model": config.base_model} if operation == "qualify" else None
            metadata = invoke(session, Path(settings.interpreter), operation, request)
            if settings.mode == "adapter":
                metadata["dependencies"].update(
                    invoke(session, Path(settings.interpreter), "fit_metadata")
                )
        execution = ExecutionBinding(**metadata, processor_sha256=processor)
        if (
            execution.dependencies.get("transformers") != "5.19.0"
            or execution.dependencies.get("sentence-transformers") != "6.1.0"
        ):
            issue(
                "encoder_environment",
                "Selected environment differs from qualified encoder dependencies",
            )
        if settings.mode == "adapter" and (
            not execution.trainable_modules or execution.dependencies.get("peft") == "unavailable"
        ):
            issue("encoder_peft", "Exact trainable PEFT targets unavailable")
        config.encoder = settings.model_copy(update={"execution": execution})
    except (ValueError, OSError, RuntimeError, CARLError):
        issue("encoder_environment", "Isolated interpreter or model qualification failed")
    if config.encoder and (config.encoder.embedding_cache or config.encoder.baseline_cache):
        from carl_encoders.artifacts import inputs, load_carriers

        try:
            if config.encoder.embedding_cache and config.encoder.mode != "frozen_heads":
                raise ValueError("Adapter learning requires differentiable fresh features")
            field = "embedding_cache" if config.encoder.embedding_cache else "baseline_cache"
            manifest = Path(getattr(config.encoder, field)).resolve()
            config.encoder = config.encoder.model_copy(update={field: str(manifest)})
            request = {
                "settings": config.encoder.model_dump(mode="json"),
                "groups": {
                    name: [row.model_dump(mode="json") for row in rows]
                    for name, rows in groups(config).items()
                },
            }
            _, document = load_carriers(manifest, carrier_binding(config), set(inputs(request)))
            sources.append(
                SourceBinding(kind="artifact", path=str(manifest), sha256=file_hash(manifest))
            )
            for entry in document["entries"].values():
                sources.append(
                    SourceBinding(kind="artifact", path=entry["path"], sha256=entry["sha256"])
                )
        except (ValueError, OSError, KeyError, TypeError):
            issue("encoder_cache", "Bound raw carrier artifacts are unavailable or changed")
    if settings.action_evaluator:
        from carl_studio.training.preparation import callable_sources

        try:
            for path in callable_sources(settings.action_evaluator, root):
                sources.append(
                    SourceBinding(kind="callable", path=str(path), sha256=file_hash(path))
                )
        except (ValueError, OSError):
            issue("encoder_evaluator", "Independent action evaluator cannot be bound")
    if config.resume_from_checkpoint:
        resume = Path(config.resume_from_checkpoint).resolve()
        config.resume_from_checkpoint = str(resume)
        try:
            state = json.loads((resume / "trainer_state.json").read_text())
            if state["status"] != "stopped":
                issue("encoder_resume", "Resume requires stopped execution")
            stopped_owner = None
            for record_path in owner.base_dir.glob("Eprep_*/experiment.json"):
                recorded = owner.load_training_result(record_path.parent.name)
                if recorded is not None and recorded.id == state.get("run_id"):
                    stopped_owner = recorded
                    break
            if (
                stopped_owner is None
                or stopped_owner.phase not in {RunPhase.PAUSED, RunPhase.FAILED}
                or stopped_owner.checkpoint != str(resume)
            ):
                issue(
                    "encoder_resume_custody",
                    "Recorded execution is not stopped at this exact checkpoint",
                )
            if config.output_dir is not None and Path(config.output_dir).resolve() == resume:
                issue(
                    "encoder_resume_output",
                    "Resume must preserve the original checkpoint in a separate candidate directory",
                )
            for path in sorted(resume.glob("*")):
                if path.is_file():
                    sources.append(
                        SourceBinding(kind="resume", path=str(path), sha256=file_hash(path))
                    )
        except (OSError, ValueError, KeyError):
            issue("encoder_resume", "Exact stopped checkpoint unavailable")
    goal = config.goal or TrainingGoal(
        description="Improve held-out encoder decision accuracy", threshold=0.8, min_delta=0.05
    )
    config.goal = goal
    preparation = TrainingPreparation(
        plan_id="",
        project_root=str(root),
        config=config.model_dump(mode="json"),
        goal=goal,
        hypothesis=Hypothesis(
            id="H_encoder",
            title="Encoder decision improvement",
            observation="Explicit feedback episodes",
            statement=goal.description,
            predictions=[
                Prediction(
                    id="P_encoder",
                    claim="Held-out action progress",
                    metric="action_accuracy",
                    comparator=PredictionComparator.GT,
                    threshold=0.05,
                )
            ],
        ),
        sources=sources,
        issues=issues,
        train_samples=len(populations.get("train", [])),
        eval_samples=len(populations.get("test", [])),
        eval_sample_ids=[row.id for row in populations.get("test", [])],
        capabilities={
            "encoder": True,
            "mode": settings.mode,
            "execution": config.encoder.execution.model_dump(mode="json")
            if config.encoder.execution
            else {},
        },
        effects=[
            "Train local encoder " + settings.mode,
            "Write local encoder checkpoint and measurements",
        ]
        + (
            ["Activate accepted encoder for " + settings.activate_workspace]
            if settings.activate_workspace
            else []
        ),
    )
    preparation = preparation.model_copy(update={"plan_id": preparation_identity(preparation)})
    if persist:
        owner.save_preparation(preparation)
    return preparation


def validate(prepared: TrainingPreparation, config: TrainingConfig | None = None) -> None:
    from carl_studio.training.preparation import file_hash, model_sources, preparation_identity

    if not prepared.ready or preparation_identity(prepared) != prepared.plan_id:
        raise ValidationError("Encoder preparation not ready", code="carl.encoder.not_ready")
    bound = TrainingConfig.model_validate(prepared.config)
    if {str(path) for path in model_sources(bound, Path(prepared.project_root))} != {
        source.path for source in prepared.sources if source.kind == "model"
    }:
        raise ValidationError("Encoder manifest changed", code="carl.encoder.stale")
    from carl_studio.training.preparation import callable_sources

    expected_sources = {str(path.resolve()) for path in implementation_sources()} | {
        str(Path(bound.encoder.validation_dataset).resolve()) if bound.encoder else ""
    }
    if bound.encoder and bound.encoder.action_evaluator:
        expected_sources.update(
            str(path)
            for path in callable_sources(
                bound.encoder.action_evaluator, Path(prepared.project_root)
            )
        )
    if expected_sources != {
        source.path for source in prepared.sources if source.kind == "callable"
    }:
        raise ValidationError("Encoder implementation manifest changed", code="carl.encoder.stale")
    for source in prepared.sources:
        path = Path(source.path)
        if not path.is_file() or file_hash(path) != source.sha256:
            raise ValidationError("Encoder source bytes changed", code="carl.encoder.stale")
    if bound.encoder is None or bound.encoder.execution is None:
        raise ValidationError("Encoder execution unbound", code="carl.encoder.stale")
    with Session() as session:
        current = invoke(session, Path(bound.encoder.interpreter), "metadata")
        if bound.encoder.mode == "adapter":
            current["dependencies"].update(
                invoke(session, Path(bound.encoder.interpreter), "fit_metadata")
            )
    for key in ("interpreter", "interpreter_sha256", "dependencies"):
        if current[key] != getattr(bound.encoder.execution, key):
            raise ValidationError("Encoder environment changed", code="carl.encoder.stale")
    groups(bound)
    if config is not None:
        candidate = config.model_copy(deep=True)
        root = Path(prepared.project_root)
        for field in (
            "base_model",
            "dataset_repo",
            "eval_dataset_repo",
            "output_dir",
            "resume_from_checkpoint",
        ):
            value = getattr(candidate, field)
            if value:
                path = Path(value)
                setattr(
                    candidate, field, str((path if path.is_absolute() else root / path).resolve())
                )
        if candidate.encoder:
            candidate.encoder = candidate.encoder.model_copy(
                update={
                    "execution": bound.encoder.execution,
                    "validation_dataset": str(
                        (root / candidate.encoder.validation_dataset).resolve()
                    ),
                }
            )
        candidate.goal = candidate.goal or bound.goal
        if candidate.model_dump(mode="json") != bound.model_dump(mode="json"):
            raise ValidationError("Encoder configuration changed", code="carl.encoder.mismatch")


async def train(trainer: Any) -> TrainingRun:
    """Run isolated learning before any causal language model loader."""
    from carl_studio.training.preparation import run_in_worker

    config = trainer.config
    if config.encoder is None or config.encoder.execution is None or config.output_dir is None:
        raise ValidationError(
            "Prepared encoder execution required", code="carl.encoder.preparation"
        )
    started = time.monotonic()
    deadline = started + config.encoder.runtime_s
    data = groups(config)
    wire_groups = {
        name: [row.model_dump(mode="json") for row in rows] for name, rows in data.items()
    }
    from urllib.parse import unquote, urlparse

    def prepare_media() -> None:
        with Session() as media_session:
            for rows in wire_groups.values():
                if trainer._cancel_requested.is_set():
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Encoder preprocessing exceeded the pilot runtime")
                for row in rows:
                    for descriptor in row["source_artifacts"].values():
                        from carl_core.data_handles import DataRef

                        media_session.data_vault.restore_file(DataRef.model_validate(descriptor))
                    for sample in (row["query"], row["positive"], *row["negatives"]):
                        for part in sample["parts"]:
                            if part["source_ref"]:
                                part["path"] = unquote(
                                    urlparse(
                                        str(row["source_artifacts"][part["source_ref"]]["uri"])
                                    ).path
                                )
                        materialize_windows(media_session, sample, deadline=deadline)

    await run_in_worker(prepare_media, on_cancel=trainer._cancel_requested.set)
    request = {
        "model": config.base_model,
        "run_id": trainer.run.id,
        "started": started,
        "settings": config.encoder.model_dump(mode="json"),
        "groups": wire_groups,
        "output": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "optimizer": config.model_dump(
            include={"weight_decay", "warmup_ratio", "lr_scheduler_type", "max_grad_norm"}
        ),
        "binding": content_hash(
            {
                "model": source_identity(Path(config.base_model)),
                "data": {
                    name: [row.model_dump(mode="json") for row in rows]
                    for name, rows in data.items()
                },
                "settings": config.encoder.model_dump(mode="json"),
                "learning_rate": config.learning_rate,
                "optimizer": config.model_dump(
                    include={"weight_decay", "warmup_ratio", "lr_scheduler_type", "max_grad_norm"}
                ),
            }
        ),
        "resume": config.resume_from_checkpoint,
    }
    if config.encoder.embedding_cache or config.encoder.baseline_cache:
        request["cache_binding"] = carrier_binding(config)
    trainer.run.checkpoint = str(config.output_dir)
    trainer.run.phase = RunPhase.TRAINING
    with Session(chain=trainer.chain) if trainer.chain is not None else Session() as session:
        cancel_path = Path(config.output_dir) / "cancel"

        def cancel() -> None:
            trainer._cancel_requested.set()
            cancel_path.parent.mkdir(parents=True, exist_ok=True)
            cancel_path.touch()

        result = await run_in_worker(
            partial(
                invoke,
                session,
                Path(config.encoder.interpreter),
                "fit",
                request,
                timeout=max(0.1, deadline - time.monotonic()),
            ),
            on_cancel=cancel,
        )
    trainer.run.checkpoint = result["checkpoint"]
    trainer.run.current_step = result["steps"]
    trainer.run.total_steps = config.encoder.optimizer_steps
    trainer.run.phase = RunPhase.COMPLETE if result["status"] == "complete" else RunPhase.PAUSED
    trainer.run.resource_usage = {
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "peak_memory_bytes": result.get("peak_memory_bytes", 0),
    }
    for name in ("counters", "timings"):
        trainer.run.resource_usage.update(
            {key: float(value) for key, value in result.get(name, {}).items()}
        )
    trainer.run.resource_usage["model_loads"] = float(result.get("model_loads", 1))
    if result.get("history"):
        trainer.run.loss = result["history"][-1]["loss"]
    measurements = Path(config.output_dir, "measurements.json")
    if not measurements.exists():
        import os
        import tempfile

        descriptor, temporary_name = tempfile.mkstemp(dir=config.output_dir, suffix=".json")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(result, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, measurements)
        finally:
            temporary.unlink(missing_ok=True)
    return trainer.run


def completion_custody(run: TrainingRun) -> dict[str, str]:
    """Bind completed optimizer artifacts before independent evaluation starts."""
    from carl_studio.training.preparation import file_hash

    if run.checkpoint is None:
        raise ValueError("Completed encoder checkpoint unavailable")
    root = Path(run.checkpoint)
    state = json.loads((root / "trainer_state.json").read_text())
    if state.get("status") != "complete":
        raise ValueError("Evaluation recovery requires completed optimizer custody")
    return {
        str((root / name).resolve()): file_hash(root / name)
        for name in ("encoder_state.pt", "trainer_state.json", "measurements.json")
    }


def finish_evaluation(run: TrainingRun, prepared: TrainingPreparation, owner: Any) -> TrainingRun:
    """Recover evaluation and activation without repeating optimizer effects."""
    import fcntl

    with (owner.base_dir / prepared.plan_id / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recorded = owner.load_training_result(prepared.plan_id)
        if recorded is not None:
            run = recorded
        if run.optimizer_phase != "complete" or not run.completion_custody:
            raise ValueError("Completed optimizer custody is required")
        validate(prepared)
        if completion_custody(run) != run.completion_custody:
            raise ValueError("Completed encoder artifact bytes changed")
        if run.evaluation_phase == "complete" and run.activation_phase in {
            "complete",
            "not_requested",
        }:
            return run
        config = TrainingConfig.model_validate(prepared.config)
        try:
            if run.evaluation_phase != "complete":
                run.evaluation_phase = "pending"
                owner.save_training_result(prepared.plan_id, run)
                run.representation_acceptance = evaluate(config, prepared)
                validate(prepared)
                if completion_custody(run) != run.completion_custody:
                    raise ValueError("Completed encoder artifacts changed during evaluation")
                run.evaluation_phase = "complete"
                owner.save_training_result(prepared.plan_id, run)
            if (
                config.encoder is not None
                and config.encoder.activate_workspace
                and run.representation_acceptance is not None
                and run.representation_acceptance.get("status") == "accepted"
            ):
                run.activation_phase = "pending"
                owner.save_training_result(prepared.plan_id, run)
                activate(prepared, run)
                run.activation_phase = "complete"
            else:
                run.activation_phase = "not_requested"
            run.phase = RunPhase.COMPLETE
            run.error_message = None
            return run
        except Exception:  # noqa: BLE001 - persist custody and sanitize evaluator errors
            if run.evaluation_phase != "complete":
                run.evaluation_phase = "failed"
            else:
                run.activation_phase = "failed"
            run.phase = RunPhase.FAILED
            run.error_message = "Prepared encoder evaluation or activation failed"
            raise ValidationError(run.error_message, code="carl.encoder.evaluation") from None
        finally:
            owner.save_training_result(prepared.plan_id, run)


async def submit(config: TrainingConfig, prepared: TrainingPreparation, owner: Any) -> TrainingRun:
    """Reuse recorded optimizer effects while recovering independent evaluation."""
    from carl_studio.training.preparation import run_in_worker
    from carl_studio.training.trainer import CARLTrainer

    await run_in_worker(partial(validate, prepared, config))
    recorded = owner.load_training_result(prepared.plan_id)
    if recorded is not None and not config.resume_from_checkpoint:
        if recorded.optimizer_phase == "complete" and recorded.completion_custody:
            return await run_in_worker(partial(finish_evaluation, recorded, prepared, owner))
        return recorded
    owner.claim_training(prepared.plan_id, resume=bool(config.resume_from_checkpoint))
    bound = TrainingConfig.model_validate(prepared.config)
    trainer = CARLTrainer(bound, skip_credits=True)
    trainer.run.optimizer_phase = "pending"
    owner.start(prepared.plan_id, trainer.run.id)
    started = time.monotonic()
    try:
        result = await trainer.train()
        if result.phase == RunPhase.COMPLETE:
            result.completion_custody = completion_custody(result)
            result.optimizer_phase = "complete"
            result.evaluation_phase = "pending"
            result.activation_phase = "pending"
            owner.save_training_result(prepared.plan_id, result)
            return await run_in_worker(partial(finish_evaluation, result, prepared, owner))
        result.optimizer_phase = "stopped" if result.phase == RunPhase.PAUSED else "failed"
        return result
    except Exception:  # noqa: BLE001 - persist execution custody before propagating failure
        failed = trainer.run
        if failed.optimizer_phase != "complete":
            failed.optimizer_phase = "failed"
        failed.phase = RunPhase.FAILED
        failed.error_message = "Prepared encoder training or evaluation failed"
        raise ValidationError(failed.error_message, code="carl.encoder.execution") from None
    finally:
        trainer.run.resource_usage["elapsed_seconds"] = time.monotonic() - started
        # Recovery reloads its recorded object; preserve its evaluation evidence.
        final = owner.load_training_result(prepared.plan_id)
        if final is not None and final.optimizer_phase == "complete":
            final.resource_usage.update(trainer.run.resource_usage)
            owner.save_training_result(prepared.plan_id, final)
        else:
            owner.save_training_result(prepared.plan_id, trainer.run)


def evaluate(config: TrainingConfig, prepared: TrainingPreparation) -> dict[str, object]:
    """Run the independently bound action evaluator without text generation."""
    from carl_studio.semantic.learning import RepresentationMeasurement, representation_acceptance
    from carl_studio.training.preparation import resolve_callable

    settings = config.encoder
    if settings is None or settings.action_evaluator is None or config.output_dir is None:
        return {
            "status": "inconclusive",
            "reasons": ["Independent action evaluator is not supplied"],
        }
    evidence = json.loads(Path(config.output_dir, "measurements.json").read_text())
    updated = evidence["updated_parameters"]
    required = ["ranker."]
    if getattr(settings, "head_layout", "shared") == "per_rung":
        rungs = sorted({int(name.rsplit(":", 1)[1]) for name in settings.required_slices})
        required = [f"ranker.{dimension}." for dimension in rungs]
    if getattr(settings, "relation_weight", 1.0) > 0:
        required.append("relation.")
    if settings.mode == "adapter":
        required.append("encoder.")
    if any(not any(name.startswith(prefix) for name in updated) for prefix in required):
        return {
            "status": "rejected",
            "reasons": ["Required head or encoder parameter updates were not measured"],
        }
    rows = groups(config)["test"]
    by_id = {row.id: row for row in rows}
    evaluator = resolve_callable(settings.action_evaluator, Path(prepared.project_root))
    measurements: list[RepresentationMeasurement] = []
    for name in ("baseline", "candidate"):
        measured = evidence[name]
        completions: list[str] = []
        for prediction in measured["predictions"]:
            row = by_id[prediction["id"]]
            selected = next(
                (
                    value
                    for value in (row.positive, *row.negatives)
                    if value.event_id == prediction["selected_event_id"]
                ),
                None,
            )
            completions.append(
                "" if selected is None else "".join(part.text or "" for part in selected.parts)
            )
        metrics = evaluator(
            completions, [{"id": row.id, "expected_output": row.action_label} for row in rows]
        )
        if "task_success_rate" not in metrics:
            return {"status": "inconclusive", "reasons": ["Evaluator omitted action accuracy"]}
        measurements.append(
            RepresentationMeasurement(
                population=content_hash([row.id for row in rows]),
                processing=settings.execution.processor_sha256 if settings.execution else "unbound",
                tasks=content_hash([row.model_dump(mode="json") for row in rows]),
                budget=content_hash(settings.model_dump(mode="json")),
                action_accuracy=metrics["task_success_rate"],
                dimensions=measured["dimensions"],
                finite=measured["finite"],
                positive_negative_margin=measured["positive_negative_margin"],
                variance=measured["variance"],
                slices=measured["slices"],
                policies={
                    key: (
                        metrics[key] is True
                        or (
                            type(metrics[key]) in {int, float}
                            and math.isfinite(metrics[key])
                            and metrics[key] >= 1
                        )
                    )
                    for key in settings.required_policies
                    if key in metrics
                },
                resources_passed=evidence["peak_memory_bytes"] <= settings.memory_gib * 1024**3
                and evidence["elapsed_seconds"] <= settings.runtime_s,
                updated_parameters=tuple(evidence["updated_parameters"])
                if name == "candidate"
                else (),
            )
        )
    return representation_acceptance(
        measurements[0],
        measurements[1],
        set(settings.required_slices),
        set(settings.required_policies),
    )


def activate(prepared: TrainingPreparation, result: TrainingRun) -> str:
    """Atomically bind accepted workspace generations and retain their predecessor."""
    from carl_studio.db import LocalDB
    from carl_studio.semantic.local import source_identity_checkpoint

    config = TrainingConfig.model_validate(prepared.config)
    settings = config.encoder
    if (
        settings is None
        or not settings.activate_workspace
        or not result.checkpoint
        or not result.representation_acceptance
        or result.representation_acceptance.get("status") != "accepted"
    ):
        raise ValueError("Activation requires accepted measurements and declared workspace effects")
    workspace = str(Path(settings.activate_workspace).resolve())
    if "Activate accepted encoder for " + settings.activate_workspace not in prepared.effects:
        raise ValueError("Local activation was not declared")
    checkpoint = Path(result.checkpoint)
    generation = {
        "model": config.base_model,
        "checkpoint": str(checkpoint),
        "processor": settings.execution.processor_sha256 if settings.execution else None,
        "checkpoint_sha256": source_identity_checkpoint(checkpoint),
        "model_artifact_sha256": source_identity(Path(config.base_model))[0],
        "execution": settings.execution.model_dump(mode="json") if settings.execution else {},
        "interpreter": settings.interpreter,
        "plan_id": prepared.plan_id,
    }
    key = "encoder-workspace:" + content_hash(workspace)
    owner = LocalDB()
    with owner.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        prior = row["value"] if row else None
        if prior and json.loads(prior).get("current") == generation:
            return content_hash(generation)
        value = json.dumps(
            {"current": generation, "predecessor": json.loads(prior) if prior else None}
        )
        connection.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
    return content_hash(generation)
