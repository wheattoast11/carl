"""Prepare and bind experiments through existing data and experiment owners."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import types
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from carl_core.errors import CARLError, ValidationError
from carl_core.hashing import content_hash
from carl_core.safepath import safe_resolve

from carl_studio.data.types import Domain, Modality, UnifiedSample, Verification
from carl_studio.experiment.manager import ExperimentManager
from carl_studio.experiment.types import Hypothesis, Prediction, PredictionComparator
from carl_studio.types.config import ComputeTarget, TrainingConfig, TrainingMethod
from carl_studio.types.preparation import (
    ReadinessIssue,
    SourceBinding,
    TrainingGoal,
    TrainingPreparation,
)

RewardCallable = Callable[..., list[float]]
EvaluatorCallable = Callable[[list[str], list[dict[str, Any]]], dict[str, float]]
_REFERENCE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_PLAN_ID = re.compile(r"^Eprep_[0-9a-f]{24}$")
_SECRET_FILES = {".env", "credentials", "token", "id_rsa", "id_ed25519"}
_IMPORT_LOCK = RLock()
_CREDENTIAL_FIELDS = frozenset(
    {
        "api_key",
        "hf_token",
        "huggingface_token",
        "hugging_face_hub_token",
        "hub_token",
        "hub_api_token",
        "access_token",
        "authorization",
        "password",
        "client_secret",
        "token",
        "secret",
        "bearer_token",
        "private_key",
        "credentials",
    }
)


async def run_in_worker[ResultT](
    operation: Callable[[], ResultT], *, on_cancel: Callable[[], None] | None = None
) -> ResultT:
    """Run blocking work with caller context and wait for cancellation cleanup."""
    import asyncio
    from contextvars import copy_context

    future = asyncio.get_running_loop().run_in_executor(None, copy_context().run, operation)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
        future.result()
        raise


def implementation_sources() -> tuple[Path, ...]:
    library = Path(__file__).parent
    return tuple(
        path.resolve()
        for path in (
            library / "preparation.py",
            library / "pipeline.py",
            library / "trainer.py",
            library / "acceptance.py",
            library.parent / "eval" / "runner.py",
        )
    )


def model_sources(config: TrainingConfig, root: Path) -> list[Path]:
    paths: set[Path] = set()
    for value in (
        config.base_model,
        config.tokenizer_source,
        config.sft_adapter,
        *config.starting_adapters,
    ):
        if not value:
            continue
        directory = Path(value)
        directory = directory if directory.is_absolute() else root / directory
        if directory.is_dir():
            paths.update(path.absolute() for path in directory.rglob("*") if path.is_file())
    return sorted(paths)


def callable_sources(reference: str, root: Path) -> list[Path]:
    """Bind local helper and package files, including their membership."""
    source = callable_source(reference, root)
    if reference == "verification":
        return [source] if source is not None else []
    package = reference.split(":")[0].split(".")[0]
    pending = [source] if source is not None else []
    paths: set[Path] = set(root.glob("*.py"))
    if (root / package).is_dir():
        paths.update((root / package).rglob("*.py"))
    pending.extend(paths)
    walked: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in walked:
            continue
        walked.add(path)
        if len(walked) > 256:
            raise ValidationError(
                "Binding source closure exceeds its budget", code="carl.preparation.binding"
            )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                top = name.split(".")[0]
                if (root / top).is_dir():
                    pending.extend((root / top).rglob("*.py"))
                elif (root / (top + ".py")).is_file():
                    pending.append(root / (top + ".py"))
    return sorted(path.absolute() for path in walked)


def file_hash(path: Path) -> str:
    """Hash source bytes without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hash(path: Path, cache_dir: Path) -> str:
    """Reuse byte digests only while file identity and change timestamps match."""
    resolved = path.resolve()
    stat = resolved.stat()
    identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    entry = cache_dir / (content_hash(str(resolved)) + ".json")
    try:
        cached = json.loads(entry.read_text())
        value = cached.get("sha256", "")
        if cached.get("identity") == identity and re.fullmatch(r"[0-9a-f]{64}", value):
            return value
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    value = file_hash(resolved)
    after = resolved.stat()
    if identity != [
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ]:
        raise ValidationError("Checkpoint changed during binding", code="carl.preparation.stale")
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=cache_dir, prefix=".hash-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"identity": identity, "sha256": value}, stream)
        os.replace(temporary, entry)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def default_manager() -> ExperimentManager:
    """Use CARL's existing private experiment directory."""
    return ExperimentManager(Path.home() / ".carl" / "experiments")


def _target(sample: dict[str, Any]) -> str | None:
    verification = sample.get("verification")
    if isinstance(verification, dict):
        typed_verification = cast(dict[str, Any], verification)
        if typed_verification.get("expected_output") is not None:
            return str(typed_verification["expected_output"])
    for key in ("expected_output", "answer", "golden_solution", "completion"):
        if sample.get(key) is not None:
            return str(sample[key])
    return None


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        messages = cast(list[dict[str, Any]], completion)
        return str(messages[-1].get("content", ""))
    raise ValidationError("Unsupported completion shape", code="carl.preparation.completion")


def verification_evaluator(
    completions: list[str], samples: list[dict[str, Any]]
) -> dict[str, float]:
    """Check exact declared targets; missing targets are not successful tasks."""
    if not samples or len(completions) != len(samples):
        raise ValidationError("Evaluation population mismatch", code="carl.preparation.population")
    targets = [_target(sample) for sample in samples]
    if any(target is None for target in targets):
        raise ValidationError("A verification target is required", code="carl.preparation.grader")
    correct = sum(text.strip() == str(target).strip() for text, target in zip(completions, targets))
    return {"task_success_rate": correct / len(samples)}


def verification_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Use the dataset's verification targets as a TRL task reward."""
    targets = kwargs.get("verification")
    if targets is not None:
        rows = [{"verification": value} for value in targets]
    else:
        targets = kwargs.get("expected_output", kwargs.get("answer"))
        if targets is None:
            raise ValidationError("Task reward needs targets", code="carl.preparation.grader")
        rows = [{"expected_output": value} for value in targets]
    if len(rows) != len(completions) or any(_target(row) is None for row in cast(list[Any], rows)):
        raise ValidationError("Task reward population mismatch", code="carl.preparation.population")
    return [
        float(_text(text).strip() == str(_target(row)).strip())
        for text, row in zip(completions, rows)
    ]


def checked_reward(function: RewardCallable) -> RewardCallable:
    """Reject missing, nonfinite and misaligned task rewards."""
    if not callable(function):
        raise ValidationError("Task reward must be callable", code="carl.preparation.reward")

    def score(completions: list[Any], **kwargs: Any) -> list[float]:
        try:
            values = [float(value) for value in function(completions=completions, **kwargs)]
        except Exception as exc:
            raise ValidationError("Task reward failed", code="carl.preparation.reward") from exc
        if len(values) != len(completions) or any(not math.isfinite(value) for value in values):
            raise ValidationError("Invalid task reward result", code="carl.preparation.reward")
        return values

    score.__name__ = getattr(function, "__name__", "task_reward")
    return score


def callable_source(reference: str, root: Path) -> Path | None:
    """Inspect a project callable declaration without executing its module."""
    if reference == "verification":
        return Path(__file__).resolve()
    if not _REFERENCE.fullmatch(reference):
        raise ValidationError("Use module:function for a binding", code="carl.preparation.binding")
    module, name = reference.split(":")
    path = safe_resolve(module.replace(".", "/") + ".py", root)
    if not path.is_file():
        raise ValidationError(
            "Binding source is not in the project", code="carl.preparation.binding"
        )
    tree = ast.parse(path.read_text())
    if not any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body):
        raise ValidationError("Binding function is not declared", code="carl.preparation.binding")
    return path


def resolve_callable(reference: str, root: Path, *, reward: bool = False) -> Callable[..., Any]:
    """Load a bound project function only at the authorized execution boundary."""
    if reference == "verification":
        return verification_reward if reward else verification_evaluator
    path = callable_source(reference, root)
    assert path is not None
    module_name, name = reference.split(":")
    names = {
        str(source.relative_to(root).with_suffix("")).replace("/", ".").removesuffix(".__init__")
        for source in callable_sources(reference, root)
    }
    loaded: dict[str, types.ModuleType] = {}

    def in_project(operation: Callable[[], Any]) -> Any:
        with _IMPORT_LOCK:
            previous_path = list(sys.path)
            previous = {key: sys.modules[key] for key in names if key in sys.modules}
            for key in names:
                sys.modules.pop(key, None)
            sys.modules.update(loaded)
            sys.path.insert(0, str(root))
            try:
                return operation()
            finally:
                loaded.update({key: sys.modules[key] for key in names if key in sys.modules})
                for key in names:
                    sys.modules.pop(key, None)
                sys.modules.update(previous)
                sys.path[:] = previous_path

    function = in_project(lambda: getattr(importlib.import_module(module_name), name))
    if not callable(function):
        raise ValidationError("Binding is not callable", code="carl.preparation.binding")

    def invoke(*args: Any, **kwargs: Any) -> Any:
        return in_project(lambda: function(*args, **kwargs))

    invoke.__name__ = name
    return invoke


def read_samples(path: Path) -> list[dict[str, Any]]:
    """Read bounded local JSON/JSONL data without a provider or model call."""
    if path.name in _SECRET_FILES or path.suffix not in {".json", ".jsonl"}:
        raise ValidationError("Use a JSON or JSONL dataset", code="carl.preparation.dataset")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValidationError("Prepare a bounded dataset shard", code="carl.preparation.dataset")
    try:
        text = path.read_text()
        rows: Any = (
            json.loads(text)
            if path.suffix == ".json"
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
    except (OSError, ValueError) as exc:
        raise ValidationError("Dataset cannot be parsed", code="carl.preparation.dataset") from exc
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in cast(list[Any], rows))
    ):
        raise ValidationError(
            "Dataset must contain nonempty records", code="carl.preparation.dataset"
        )
    return cast(list[dict[str, Any]], rows)


def normalize_samples(rows: list[dict[str, Any]], source: str) -> list[UnifiedSample]:
    """Keep task identity, verification and provenance together."""
    normalized: list[UnifiedSample] = []
    for row in rows:
        prompt = row.get("prompt", row.get("problem_statement", row.get("question")))
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if not isinstance(prompt, list) or not prompt:
            raise ValidationError("Dataset needs task prompts", code="carl.preparation.dataset")
        verification = cast(dict[str, Any], row.get("verification") or {})
        target = _target(row)
        if target is not None:
            verification = {**verification, "expected_output": target}
        normalized.append(
            UnifiedSample(
                id=str(row.get("id") or content_hash(prompt)),
                prompt=cast(list[dict[str, Any]], prompt),
                problem_statement=str(row.get("problem_statement") or row.get("question") or ""),
                domain=row.get("domain", Domain.INSTRUCTION),
                modality=row.get("modality", Modality.TEXT),
                verification=Verification.model_validate(verification),
                golden_solution=target,
                source=str(row.get("source", source)),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"prompt", "verification", "id", "source"}
                },
            )
        )
    return normalized


def discover_variants(root: Path) -> dict[str, str]:
    """Expose the two requested Tinny modes without promoting an ablation."""
    result: dict[str, str] = {}
    for name, relative in (
        ("Tinny alone", "carl/configs/tinny-single-agent.yaml"),
        ("Tinny with TCG", "carl/configs/tinny-tcg-lead.yaml"),
    ):
        if (root / relative).is_file():
            result[name] = relative
    return result


def project_root_for_config(path: Path) -> Path:
    """Use existing project discovery, then the nearest Git worktree."""
    from carl_studio.project_context import current

    directory = path.resolve().parent
    context = current(directory)
    if context is not None:
        return context.root
    for parent in (directory, *directory.parents):
        if (parent / ".git").exists():
            return parent
    return directory


def preparation_identity(preparation: TrainingPreparation) -> str:
    """Bind configuration, source bytes, goals and the held-out population."""
    return (
        "Eprep_"
        + content_hash(
            {
                "root": preparation.project_root,
                "config": preparation.config,
                "goal": preparation.goal.model_dump(mode="json"),
                "sources": [source.model_dump() for source in preparation.sources],
                "eval_ids": preparation.eval_sample_ids,
                "capabilities": preparation.capabilities,
                "issues": [issue.model_dump() for issue in preparation.issues],
                "train_samples": preparation.train_samples,
                "eval_samples": preparation.eval_samples,
                "effects": preparation.effects,
            }
        )[:24]
    )


def prepare_training(
    config: TrainingConfig,
    *,
    project_root: Path | None = None,
    config_path: Path | None = None,
    manager: ExperimentManager | None = None,
    persist: bool = True,
) -> TrainingPreparation:
    """Prepare local inputs and trusted bindings; never submit, install or publish."""
    if any(
        name.lower().replace("-", "_") in _CREDENTIAL_FIELDS
        or name.lower().endswith(("_api_key", "_password", "_client_secret"))
        for name in config.extra_args
    ):
        raise ValidationError(
            "Literal credentials are not training settings; use the configured credential owner",
            code="carl.preparation.credentials",
        )
    root = (project_root or Path.cwd()).resolve()
    if config.method == TrainingMethod.ENCODER:
        from carl_studio.training.encoder import prepare
        return prepare(config, root, manager=manager, config_path=config_path, persist=persist)
    goal = config.goal or TrainingGoal()
    config = config.model_copy(deep=True, update={"goal": goal, "push_to_hub": False})
    issues: list[ReadinessIssue] = []
    sources: list[SourceBinding] = []
    if config_path is not None:
        config_path = config_path.resolve()
        sources.append(
            SourceBinding(kind="config", path=str(config_path), sha256=file_hash(config_path))
        )
    populations: dict[str, list[UnifiedSample]] = {}

    def issue(code: str, message: str, action: str) -> None:
        issues.append(ReadinessIssue(code=code, message=message, action=action))

    from carl_studio.adapters.registry import get_capabilities

    try:
        capabilities = get_capabilities(config.adapter)
    except CARLError:
        capabilities = {}
        issue(
            "backend",
            "The selected backend is not registered",
            "Choose a registered training backend",
        )
    if not capabilities.get("task_rewards") or not capabilities.get("artifacts"):
        issue(
            "backend_hooks",
            "The selected backend has no qualified goal binding",
            "Bind reward, evaluation and artifact hooks for this backend",
        )
    if config.compute_target != ComputeTarget.LOCAL:
        issue(
            "remote_preparation",
            "Remote goal inputs need approved artifact custody",
            "Prepare local inputs or provide a qualified remote integration",
        )
    if config.method not in {TrainingMethod.SFT, TrainingMethod.GRPO}:
        issue(
            "method",
            "The selected method is not supported by this integration",
            "Use a qualified training method",
        )
    if not 0.0 <= goal.threshold <= 1.0:
        issue(
            "metric_scale",
            "The current evaluation gate uses a metric in [0, 1]",
            "Normalize the primary metric and its threshold to [0, 1]",
        )
    if config.goal and {
        "max_steps",
        "output_dir",
        "push_to_hub",
        "hub_model_id",
        "max_length",
        "num_generations",
    } & set(config.extra_args):
        issue(
            "runtime_overrides",
            "Extra arguments override prepared resource or effect limits",
            "Put those settings in their declared configuration fields",
        )
    if config.max_steps < 1:
        issue(
            "step_budget",
            "Training needs a finite step budget",
            "Set max_steps for a bounded pilot",
        )
    if config.method == TrainingMethod.GRPO and config.max_steps <= config.cascade.carl_start:
        issue(
            "reward_budget",
            "The step budget ends before the coherence reward starts",
            "Allow active coherence steps inside the pilot budget",
        )
    for dependency in ("torch", "transformers", "trl", "datasets", "peft"):
        try:
            available = importlib.util.find_spec(dependency) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            issue(
                "dependency",
                "Training dependencies are incomplete",
                "Install carl-studio[training] in the selected environment",
            )
            break

    model = Path(config.base_model)
    model = model if model.is_absolute() else root / model
    if model.is_dir() and (model / "config.json").is_file():
        config.base_model = str(model.resolve())
        files = [
            p
            for p in sorted(model.rglob("*"))
            if p.is_file()
            and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".py"}
        ]
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            issue("model_weights", "The local model has no weights", "Supply the source checkpoint")
    else:
        issue(
            "model_source",
            "The model is not bound to local source bytes",
            "Fetch the selected model revision with approval, then prepare its local checkpoint",
        )
    if config.tokenizer_source:
        tokenizer = Path(config.tokenizer_source)
        tokenizer = tokenizer if tokenizer.is_absolute() else root / tokenizer
        if not tokenizer.is_dir():
            issue(
                "tokenizer",
                "The tokenizer source is not bound locally",
                "Prepare the selected tokenizer alongside the model",
            )
        else:
            config.tokenizer_source = str(tokenizer.resolve())
    if config.sft_adapter:
        adapter = Path(config.sft_adapter)
        adapter = adapter if adapter.is_absolute() else root / adapter
        if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
            issue(
                "sft_adapter",
                "The starting adapter is not bound locally",
                "Prepare the selected SFT adapter before the run",
            )
        else:
            config.sft_adapter = str(adapter.resolve())
    cache_dir = (manager or default_manager()).base_dir / "source-hashes"

    def bind_model_source(path: Path) -> SourceBinding:
        digest = (
            checkpoint_hash(path, cache_dir)
            if persist and path.suffix in {".safetensors", ".bin"}
            else file_hash(path)
        )
        return SourceBinding(kind="model", path=str(path), sha256=digest)

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="carl-source") as workers:
        sources.extend(workers.map(bind_model_source, model_sources(config, root)))
    normalized_adapters: list[str] = []
    for value in config.starting_adapters:
        adapter = Path(value)
        adapter = adapter if adapter.is_absolute() else root / adapter
        if not adapter.is_dir():
            issue(
                "starting_adapter",
                "A starting adapter is unavailable",
                "Prepare each adapter in the declared order",
            )
        normalized_adapters.append(str(adapter.resolve()))
    config.starting_adapters = normalized_adapters
    if config.output_dir is not None:
        output = config.output_dir
        output = (output if output.is_absolute() else root / output).resolve()
        config.output_dir = output
        input_directories = {path.parent.resolve() for path in model_sources(config, root)}
        if any(output == source or source in output.parents for source in input_directories):
            issue(
                "output_collision",
                "The candidate output overlaps the starting model",
                "Choose a separate candidate directory",
            )
    if config.resume_from_checkpoint:
        resume = Path(config.resume_from_checkpoint)
        resume = resume if resume.is_absolute() else root / resume
        if not resume.is_dir() or not (resume / "trainer_state.json").is_file():
            issue(
                "resume",
                "Resume needs a recorded trainer checkpoint",
                "Select the checkpoint from the previous run",
            )
        else:
            config.resume_from_checkpoint = str(resume.resolve())
            for path in sorted(resume.rglob("*")):
                if path.is_file():
                    sources.append(
                        SourceBinding(
                            kind="resume", path=str(path.absolute()), sha256=file_hash(path)
                        )
                    )

    for kind, name in (
        ("train_data", config.dataset_repo),
        ("eval_data", config.eval_dataset_repo or ""),
    ):
        path = Path(name)
        if not path.is_absolute():
            candidates = ([config_path.resolve().parent / path] if config_path else []) + [
                root / path
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), root / path)
        if not name or not path.is_file():
            issue(
                kind,
                "A local dataset is required for this preparation",
                "Provide task examples and a separate held-out JSONL file",
            )
            continue
        try:
            samples = normalize_samples(read_samples(path), str(path.resolve()))
            populations[kind] = samples
            sources.append(
                SourceBinding(
                    kind=cast(Literal["train_data", "eval_data"], kind),
                    path=str(path.resolve()),
                    sha256=file_hash(path),
                )
            )
            if kind == "train_data":
                config.dataset_repo = str(path.resolve())
            else:
                config.eval_dataset_repo = str(path.resolve())
            if goal.evaluator == "verification" and any(
                sample.verification.expected_output is None for sample in samples
            ):
                issue(
                    "verification",
                    "Task examples need executable grading or declared targets",
                    "Supply expected outputs or a project evaluator",
                )
        except (ValueError, ValidationError, OSError):
            issue(
                kind, "Dataset preparation failed", "Check the JSON records and verification fields"
            )

    train = populations.get("train_data", [])
    heldout = populations.get("eval_data", [])
    for population in (train, heldout):
        if len({sample.id for sample in population}) != len(population):
            issue(
                "duplicate_ids", "Sample identities are duplicated", "Assign stable unique task IDs"
            )
    if {sample.id for sample in train} & {sample.id for sample in heldout} or {
        content_hash(sample.prompt) for sample in train
    } & {content_hash(sample.prompt) for sample in heldout}:
        issue(
            "split_overlap",
            "Training and held-out tasks overlap",
            "Split by task identity before preparation",
        )

    references = [
        goal.evaluator,
        *[binding.reference for binding in goal.rewards],
        *[policy.reference for policy in goal.policies],
    ]
    for path in implementation_sources():
        sources.append(
            SourceBinding(kind="callable", path=str(path.resolve()), sha256=file_hash(path))
        )
    for reference in dict.fromkeys(references):
        try:
            for path in callable_sources(reference, root):
                sources.append(
                    SourceBinding(kind="callable", path=str(path), sha256=file_hash(path))
                )
        except (ValidationError, OSError, SyntaxError):
            issue(
                "binding",
                "A required callable cannot be bound",
                "Provide a declared project module:function binding",
            )

    comparison = PredictionComparator.GT
    hypothesis = Hypothesis(
        id="H_training_goal",
        title="Prepared model improvement",
        observation="Caller-declared training goal",
        statement=goal.description,
        predictions=[
            Prediction(
                id="P_goal",
                claim="Held-out goal progress",
                metric=goal.primary_metric,
                comparator=comparison,
                threshold=goal.min_delta,
            ),
            Prediction(
                id="P_coherence",
                claim="Declared coherence floor",
                metric="phi_mean",
                comparator=PredictionComparator.GE,
                threshold=goal.coherence_phi_floor,
            ),
            *[
                Prediction(
                    id=f"P_policy_{policy.id}",
                    claim="Declared policy check",
                    metric=policy.id,
                    comparator=PredictionComparator.GE,
                    threshold=policy.threshold,
                )
                for policy in goal.policies
            ],
        ],
    )
    preparation = TrainingPreparation(
        plan_id="",
        project_root=str(root),
        config=config.model_dump(mode="json"),
        goal=goal,
        hypothesis=hypothesis,
        sources=sources,
        train_samples=len(train),
        eval_samples=min(len(heldout), goal.max_eval_samples),
        eval_sample_ids=[sample.id for sample in heldout[: goal.max_eval_samples]],
        issues=issues,
        variants=discover_variants(root),
        capabilities=capabilities,
        effects=[
            "Train the selected local model",
            "Write owner-readable model and evaluation artifacts",
        ],
    )
    preparation = preparation.model_copy(update={"plan_id": preparation_identity(preparation)})
    if persist:
        (manager or default_manager()).save_preparation(preparation)
    return preparation


def load_preparation(plan_id: str, manager: ExperimentManager | None = None) -> TrainingPreparation:
    """Load a matching owner-recorded preparation through ExperimentManager."""
    if not _PLAN_ID.fullmatch(plan_id):
        raise ValidationError("Invalid prepared experiment ID", code="carl.preparation.id")
    preparation = (manager or default_manager()).load_preparation(plan_id)
    if preparation_identity(preparation) != preparation.plan_id:
        raise ValidationError("Prepared inputs changed", code="carl.preparation.stale")
    return preparation


def validate_preparation(
    preparation: TrainingPreparation, config: TrainingConfig | None = None
) -> None:
    """Refuse stale or incomplete inputs before constructing the trainer."""
    if TrainingConfig.model_validate(preparation.config).method == TrainingMethod.ENCODER:
        from carl_studio.training.encoder import validate
        return validate(preparation, config)
    if not preparation.ready or preparation_identity(preparation) != preparation.plan_id:
        raise ValidationError("The experiment is not ready", code="carl.preparation.not_ready")
    bound = TrainingConfig.model_validate(preparation.config)
    root = Path(preparation.project_root)
    if {str(path) for path in model_sources(bound, root)} != {
        source.path for source in preparation.sources if source.kind == "model"
    }:
        raise ValidationError("The model source manifest changed", code="carl.preparation.stale")
    if bound.resume_from_checkpoint:
        resume = Path(bound.resume_from_checkpoint)
        if {str(path.absolute()) for path in resume.rglob("*") if path.is_file()} != {
            source.path for source in preparation.sources if source.kind == "resume"
        }:
            raise ValidationError(
                "The resume source manifest changed", code="carl.preparation.stale"
            )
    expected_callables = {str(path) for path in implementation_sources()}
    for reference in [
        preparation.goal.evaluator,
        *[binding.reference for binding in preparation.goal.rewards],
        *[policy.reference for policy in preparation.goal.policies],
    ]:
        expected_callables.update(str(path) for path in callable_sources(reference, root))
    if expected_callables != {
        source.path for source in preparation.sources if source.kind == "callable"
    }:
        raise ValidationError("The callable source manifest changed", code="carl.preparation.stale")
    for source in preparation.sources:
        path = Path(source.path)
        if not path.is_file() or file_hash(path) != source.sha256:
            raise ValidationError("A prepared source changed", code="carl.preparation.stale")
    if config is not None:
        expected = TrainingConfig.model_validate(preparation.config)
        candidate = config.model_copy(
            deep=True, update={"push_to_hub": False, "goal": config.goal or preparation.goal}
        )
        for field in (
            "base_model",
            "dataset_repo",
            "eval_dataset_repo",
            "tokenizer_source",
            "sft_adapter",
            "resume_from_checkpoint",
        ):
            value = getattr(candidate, field)
            if value:
                path = Path(value)
                resolved = path if path.is_absolute() else root / path
                if field in {"dataset_repo", "eval_dataset_repo"} and not path.is_absolute():
                    configs = [
                        Path(source.path).parent / path
                        for source in preparation.sources
                        if source.kind == "config"
                    ]
                    resolved = next(
                        (item for item in [*configs, root / path] if item.is_file()), resolved
                    )
                if resolved.exists():
                    setattr(candidate, field, str(resolved.resolve()))
        candidate.starting_adapters = [
            str((Path(value) if Path(value).is_absolute() else root / value).resolve())
            for value in candidate.starting_adapters
        ]
        if candidate.output_dir is not None:
            output = candidate.output_dir
            candidate.output_dir = (output if output.is_absolute() else root / output).resolve()
        if candidate.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValidationError(
                "Configuration does not match the preparation", code="carl.preparation.mismatch"
            )


def render_preparation(preparation: TrainingPreparation) -> str:
    """Show the next useful decision; leave training internals in details."""
    lines = [
        "Ready to review a bounded training run"
        if preparation.ready
        else "Preparation needs input",
        f"Experiment: {preparation.plan_id}",
        f"Goal: {preparation.goal.description}",
        f"Data: {preparation.train_samples} training, {preparation.eval_samples} held-out tasks",
    ]
    config = TrainingConfig.model_validate(preparation.config)
    stages = 2 if config.pipeline and config.method == TrainingMethod.GRPO else 1
    steps = config.max_steps * stages
    lines.append(f"Budget: at most {steps} training steps across {stages} stage(s)")
    for issue in preparation.issues:
        lines.append(f"{issue.message}. {issue.action}.")
    if preparation.ready:
        lines.extend(
            [
                "Execution requires approval of the selected inputs and resource limits.",
                "Acceptance requires goal, policy and coherence checks; publication is separate.",
            ]
        )
    return "\n".join(lines)
