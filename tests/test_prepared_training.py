"""Prepared inputs, real experiment persistence and acceptance counterexamples."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from carl_core.errors import ValidationError

from carl_studio.eval.runner import EvalRunner
from carl_studio.experiment.manager import ExperimentManager
from carl_studio.training.acceptance import compare_candidate
from carl_studio.training.pipeline import submit_training
from carl_studio.training.preparation import (
    checked_reward,
    load_preparation,
    prepare_training,
    validate_preparation,
    verification_evaluator,
    verification_reward,
)
from carl_studio.types.config import TrainingConfig
from carl_studio.types.preparation import EvaluationMeasurement, PolicyCheck, TrainingGoal
from carl_studio.types.run import RunPhase, TrainingRun


@pytest.fixture
def inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TrainingConfig, ExperimentManager]:
    import importlib.util

    original = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            SimpleNamespace()
            if name in {"torch", "transformers", "trl", "datasets", "peft"}
            else original(name, *args)
        ),
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"vocab_size":16}')
    (model / "model.safetensors").write_bytes(b"source-fixture")
    for name, prefix in (("train", "train"), ("eval", "heldout")):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"{prefix}{i}",
                        "prompt": f"{prefix} task {i}",
                        "expected_output": answer,
                    }
                )
                for i, answer in enumerate(("A", "B"))
            )
        )
    config = TrainingConfig(
        run_name="fixture",
        base_model=str(model),
        output_repo="fixture/output",
        dataset_repo=str(tmp_path / "train.jsonl"),
        eval_dataset_repo=str(tmp_path / "eval.jsonl"),
        method="grpo",
        compute_target="local",
        max_steps=64,
        cascade={"carl_start": 0},
        goal=TrainingGoal(),
    )
    return config, ExperimentManager(tmp_path / "experiments")


def measurement(value: float, **overrides: object) -> EvaluationMeasurement:
    return EvaluationMeasurement.model_validate(
        {
            "checkpoint": "candidate",
            "dataset_sha256": "a" * 64,
            "sample_ids": ["a", "b"],
            "primary_metric": "task_success_rate",
            "primary_value": value,
            "metrics": {"task_success_rate": value},
            "coherence": {"phi_mean": 0.4, "discontinuity_score": 0.5},
            "coherence_source": "full_logits",
            "generation": {"sampling": "greedy", "max_new_tokens": 128},
            **overrides,
        }
    )


def test_preparation_is_repeatable_and_retains_owner_record(inputs, tmp_path):
    config, owner = inputs
    first = prepare_training(config, project_root=tmp_path, manager=owner)
    second = prepare_training(config, project_root=tmp_path, manager=owner)
    assert first.ready and second.plan_id == first.plan_id
    assert first.train_samples == first.eval_samples == 2
    assert config.push_to_hub is True
    assert first.config["push_to_hub"] is False
    loaded = load_preparation(first.plan_id, owner)
    validate_preparation(loaded, config)
    assert owner.load(first.plan_id).hypothesis.predictions
    assert (owner.base_dir / first.plan_id).stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("missing", "eval_data"),
        ("overlap", "split_overlap"),
        ("unbounded", "step_budget"),
        ("no_grader", "verification"),
        ("early_reward", "reward_budget"),
    ],
)
def test_readiness_counterexamples(inputs, tmp_path, mutation, code):
    config, owner = inputs
    if mutation == "missing":
        config.eval_dataset_repo = str(tmp_path / "absent.jsonl")
    elif mutation == "overlap":
        Path(config.eval_dataset_repo).write_bytes(Path(config.dataset_repo).read_bytes())
    elif mutation == "unbounded":
        config.max_steps = -1
    elif mutation == "early_reward":
        config.max_steps = 1
        config.cascade.carl_start = 50
    else:
        Path(config.eval_dataset_repo).write_text('{"id":"new","prompt":"new task"}')
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert not prepared.ready
    assert code in {issue.code for issue in prepared.issues}


def test_tinny_exposes_two_modes(inputs, tmp_path):
    config, owner = inputs
    folder = tmp_path / "carl" / "configs"
    folder.mkdir(parents=True)
    for name in ("single-agent", "tcg-lead", "tcg-universal"):
        (folder / f"tinny-{name}.yaml").write_text("{}")
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert set(prepared.variants) == {"Tinny alone", "Tinny with TCG"}


def test_stale_input_refused_before_trainer(inputs, tmp_path, monkeypatch):
    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    Path(config.dataset_repo).write_text('{"prompt":"changed"}')
    trainer = MagicMock()
    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", trainer)
    with pytest.raises(ValidationError, match="source changed"):
        asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
    trainer.assert_not_called()


def test_changed_config_refused(inputs, tmp_path):
    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    with pytest.raises(ValidationError, match="does not match"):
        validate_preparation(prepared, config.model_copy(update={"max_steps": 65}))


def test_literal_credentials_refused_before_preparation_persistence(inputs, tmp_path):
    config, owner = inputs
    config.extra_args = {"api_key": "INVENTED_PRIVATE_CANARY"}
    with pytest.raises(ValidationError) as error:
        prepare_training(config, project_root=tmp_path, manager=owner)
    assert "INVENTED_PRIVATE_CANARY" not in str(error.value)
    assert not list(owner.base_dir.glob("*/preparation.json"))
    config.extra_args = {"max_prompt_tokens": "1024"}
    assert prepare_training(config, project_root=tmp_path, manager=owner).ready


def test_callable_import_error_is_sanitized_and_recorded(inputs, tmp_path, monkeypatch, caplog):
    config, owner = inputs
    (tmp_path / "rules.py").write_text(
        "raise RuntimeError('INVENTED_IMPORT_CANARY')\n"
        "def grade(completions, samples):\n return {'task_success_rate': 1.0}\n"
    )
    config.goal = TrainingGoal(evaluator="rules:grade")
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    trainer = MagicMock()
    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", trainer)
    with pytest.raises(ValidationError) as error:
        asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
    assert "INVENTED_IMPORT_CANARY" not in str(error.value)
    assert "INVENTED_IMPORT_CANARY" not in caplog.text
    result = owner.load_training_result(prepared.plan_id)
    assert result.phase == RunPhase.FAILED
    assert "INVENTED_IMPORT_CANARY" not in result.model_dump_json()
    trainer.assert_not_called()


def test_prepared_dry_run_does_not_execute_training(inputs, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from carl_studio.cli import app

    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    path = tmp_path / "carl.yaml"
    path.write_text(config.model_dump_json())
    submit = AsyncMock(
        return_value=TrainingRun(id="fixture", config=config, phase=RunPhase.COMPLETE)
    )
    monkeypatch.setattr("carl_studio.training.pipeline.submit_training", submit)
    monkeypatch.setattr("carl_studio.training.preparation.default_manager", lambda: owner)
    result = CliRunner().invoke(
        app, ["train", "--config", str(path), "--prepared-plan", prepared.plan_id, "--dry-run"]
    )
    submit.assert_not_awaited()
    assert result.exit_code == 0, result.output
    assert prepared.plan_id in result.output


def test_prepared_send_it_keeps_existing_autonomy_gate(inputs, tmp_path, monkeypatch):
    import importlib

    from typer.testing import CliRunner

    from carl_studio.cli import app

    config, owner = inputs
    config.pipeline = True
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    path = tmp_path / "carl.yaml"
    path.write_text(config.model_dump_json())
    submit = AsyncMock()
    monkeypatch.setattr("carl_studio.training.pipeline.submit_training", submit)
    tier = importlib.import_module("carl_studio.tier")
    monkeypatch.setattr(tier, "check_tier", lambda _: (False, None, None))
    monkeypatch.setattr(tier, "tier_message", lambda _: "Autonomy requires CARL Paid")
    result = CliRunner().invoke(
        app, ["train", "--config", str(path), "--prepared-plan", prepared.plan_id, "--send-it"]
    )
    submit.assert_not_awaited()
    assert result.exit_code == 1
    assert "Autonomy requires" in result.output


def test_modified_ready_state_cannot_remove_a_hold(inputs, tmp_path):
    config, owner = inputs
    config.max_steps = -1
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    with pytest.raises(ValidationError):
        validate_preparation(prepared.model_copy(update={"issues": []}))


def test_task_reward_requires_actual_verification():
    long_wrong = "An incorrect answer with enough characters to trigger the former length shortcut"
    assert verification_reward([long_wrong, "A"], expected_output=["A", "A"]) == [0.0, 1.0]
    with pytest.raises(ValidationError):
        verification_reward([long_wrong])
    assert (
        verification_evaluator(["wrong", "A"], [{"answer": "A"}, {"answer": "A"}])[
            "task_success_rate"
        ]
        == 0.5
    )


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [1.0, 2.0]])
def test_malformed_reward_rejected(values):
    with pytest.raises(ValidationError):
        checked_reward(lambda **kwargs: values)(completions=["A"])


@pytest.mark.parametrize("value,status", [(0.9, "accepted"), (0.5, "rejected"), (0.2, "rejected")])
def test_goal_progress_and_unchanged_control(value, status):
    assert compare_candidate(TrainingGoal(), measurement(0.5), measurement(value)).status == status


def test_high_coherence_does_not_accept_wrong_task():
    outcome = compare_candidate(TrainingGoal(), measurement(0.1), measurement(0.2))
    assert outcome.coherence_passed
    assert outcome.status == "rejected"


def test_missing_coherence_and_changed_populations_are_inconclusive():
    base = measurement(0.5)
    assert (
        compare_candidate(
            TrainingGoal(), base, measurement(0.9, coherence=None, coherence_source="unavailable")
        ).status
        == "inconclusive"
    )
    assert (
        compare_candidate(TrainingGoal(), base, measurement(0.9, sample_ids=["other"])).status
        == "inconclusive"
    )
    assert (
        compare_candidate(TrainingGoal(), base, measurement(0.9, dataset_sha256="b" * 64)).status
        == "inconclusive"
    )


def test_policy_results_are_independent():
    goal = TrainingGoal(policies=[PolicyCheck(id="privacy", reference="rules:privacy")])
    assert compare_candidate(goal, measurement(0.5), measurement(0.9)).status == "inconclusive"
    assert (
        compare_candidate(
            goal, measurement(0.5), measurement(0.9, policy_results={"privacy": 0.0})
        ).status
        == "rejected"
    )
    assert (
        compare_candidate(
            goal, measurement(0.5), measurement(0.9, policy_results={"privacy": 1.0})
        ).status
        == "accepted"
    )


def test_real_owner_loop_and_replay_do_not_repeat_training(inputs, tmp_path, monkeypatch):
    config, owner = inputs
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}")
    config.tokenizer_source = str(tokenizer)
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    trainer = MagicMock()
    trainer.run = TrainingRun(
        id="executed-fixture",
        config=TrainingConfig.model_validate(prepared.config),
        phase=RunPhase.COMPLETE,
        checkpoint=str(candidate_dir),
    )
    trainer.train = AsyncMock(return_value=trainer.run)
    factory = MagicMock(return_value=trainer)
    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", factory)
    selected_tokenizers = []

    def load_model(runner):
        selected_tokenizers.append(runner.config.tokenizer_source)
        return object(), object()

    monkeypatch.setattr(EvalRunner, "_load_model_simple", load_model)
    monkeypatch.setattr(
        EvalRunner,
        "_generate_single_turn",
        lambda self, model, tokenizer, samples: (
            ["wrong"] * len(samples)
            if self.config.checkpoint == config.base_model
            else [sample["expected_output"] for sample in samples]
        ),
    )
    monkeypatch.setattr(
        EvalRunner,
        "_compute_coherence",
        lambda *args: {"phi_mean": 0.4, "discontinuity_score": 0.5},
    )
    run = asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
    assert run.acceptance.status == "accepted"
    assert run.acceptance.baseline.primary_value == 0.0
    assert run.acceptance.candidate.primary_value == 1.0
    assert owner.load(prepared.plan_id).judgment.verdict.value == "realized"
    repeated = asyncio.run(
        submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner)
    )
    assert repeated.id == run.id
    assert trainer.train.await_count == 1
    assert selected_tokenizers == [str(tokenizer), str(tokenizer)]
    reward = factory.call_args.kwargs["task_reward_funcs"][0]
    assert reward(["A", "wrong"], expected_output=["A", "A"]) == [1.0, 0.0]


def test_evaluator_missing_primary_and_nonfinite_metrics_fail():
    from carl_studio.eval.runner import EvalConfig

    runner = EvalRunner(
        EvalConfig(checkpoint="fixture"), evaluator=lambda *_: {"score": float("nan")}
    )
    with pytest.raises(ValueError, match="invalid measurements"):
        runner._compute_metrics(["A"], [{"answer": "A"}])


def test_added_model_file_invalidates_preparation(inputs, tmp_path):
    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    (Path(config.base_model) / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValidationError, match="manifest changed"):
        validate_preparation(prepared)


def test_chat_template_is_bound(inputs, tmp_path):
    config, owner = inputs
    template = Path(config.base_model) / "chat_template.jinja"
    template.write_text("original")
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    template.write_text("changed")
    with pytest.raises(ValidationError, match="source changed"):
        validate_preparation(prepared)


def test_relative_adapter_and_resume_inputs_match(inputs, tmp_path):
    config, owner = inputs
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    (resume / "trainer_state.json").write_text("{}")
    config.sft_adapter = "adapter"
    config.resume_from_checkpoint = "checkpoint"
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    validate_preparation(prepared, config)


def test_relative_output_uses_selected_project_and_protects_starting_adapter(inputs, tmp_path):
    config, owner = inputs
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    config.sft_adapter = "adapter"
    config.output_dir = Path("candidate")
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert prepared.config["output_dir"] == str(tmp_path / "candidate")
    validate_preparation(prepared, config)
    config.output_dir = Path("adapter")
    assert "output_collision" in {
        issue.code
        for issue in prepare_training(config, project_root=tmp_path, manager=owner).issues
    }


def test_readiness_declares_metric_scale_and_total_pipeline_budget(inputs, tmp_path):
    from carl_studio.training.preparation import render_preparation

    config, owner = inputs
    config.pipeline = True
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert "128 training steps across 2" in render_preparation(prepared)
    config.goal = config.goal.model_copy(update={"threshold": 3.0})
    assert "metric_scale" in {
        issue.code
        for issue in prepare_training(config, project_root=tmp_path, manager=owner).issues
    }


def test_unbound_resume_is_refused(inputs, tmp_path, monkeypatch):
    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    trainer = MagicMock()
    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", trainer)
    with pytest.raises(ValidationError, match="Resume checkpoint"):
        asyncio.run(
            submit_training(
                config,
                prepared_plan_id=prepared.plan_id,
                manager=owner,
                resume_from_checkpoint=str(tmp_path / "other"),
            )
        )
    trainer.assert_not_called()


def test_added_resume_file_invalidates_preparation(inputs, tmp_path):
    config, owner = inputs
    resume = tmp_path / "checkpoint"
    resume.mkdir()
    (resume / "trainer_state.json").write_text("{}")
    config.resume_from_checkpoint = str(resume)
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    (resume / "optimizer.pt").write_bytes(b"changed state")
    with pytest.raises(ValidationError, match="manifest changed"):
        validate_preparation(prepared)


def test_evaluation_preserves_ordered_adapter_lineage(monkeypatch):
    import peft
    import transformers

    from carl_studio.eval.runner import EvalConfig

    events = []
    tokenizers = []
    model = SimpleNamespace(device=SimpleNamespace(type="cpu"), eval=lambda: None)

    def load_base(path, **kwargs):
        events.append(path)
        return model

    def load_adapter(current, path, **kwargs):
        assert current is model
        events.append(path)
        return SimpleNamespace(merge_and_unload=lambda: model)

    def load_tokenizer(path):
        tokenizers.append(path)
        return SimpleNamespace(pad_token="pad")

    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=load_base)
    )
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=load_tokenizer),
    )
    monkeypatch.setattr(peft, "PeftModel", SimpleNamespace(from_pretrained=load_adapter))
    runner = EvalRunner(
        EvalConfig(
            checkpoint="GRPO-C",
            base_model="base",
            starting_adapters=["original-A"],
            sft_adapter="SFT-B",
            tokenizer_source="selected-tokenizer",
            phase="1",
            device="cpu",
        )
    )
    runner._load_model_simple()
    assert events == ["base", "original-A", "SFT-B", "GRPO-C"]
    assert tokenizers == ["selected-tokenizer"]


@pytest.mark.asyncio
async def test_pipeline_cancellation_retains_actual_stage(inputs, tmp_path, monkeypatch):
    config, owner = inputs
    config.pipeline = True
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    started = asyncio.Event()

    class Trainer:
        def __init__(self, stage, **kwargs):
            self.run = TrainingRun(
                id="actual-SFT", config=stage, current_step=7, checkpoint="checkpoint-7"
            )
            self.is_remote = False

        async def train(self):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", Trainer)
    monkeypatch.setattr(
        EvalRunner,
        "run",
        lambda *_: SimpleNamespace(
            n_samples=2,
            primary_metric="task_success_rate",
            primary_value=0.0,
            metrics={"task_success_rate": 0.0},
            coherence={"phi_mean": 0.4, "discontinuity_score": 0.5},
        ),
    )
    task = asyncio.create_task(
        submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner)
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = owner.load_training_result(prepared.plan_id)
    assert result.id == "actual-SFT"
    assert result.config.method.value == "sft"
    assert result.current_step == 7
    assert result.checkpoint == "checkpoint-7"
    assert result.phase == RunPhase.PAUSED


@pytest.mark.asyncio
async def test_stage_evaluation_keeps_event_loop_and_cancellation_responsive(inputs, monkeypatch):
    import threading

    from carl_studio.training.pipeline import SendItPipeline

    config, _ = inputs
    pipeline = SendItPipeline(config)
    started = threading.Event()
    release = threading.Event()
    ended = threading.Event()

    def gate(_):
        started.set()
        release.wait(timeout=2)
        ended.set()
        return True

    monkeypatch.setattr(pipeline, "_check_gate", gate)
    task = asyncio.create_task(pipeline._evaluate_gate(TrainingRun(id="stage", config=config)))
    while not started.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended.is_set()


@pytest.mark.parametrize("value,expected", [(0.2, True), (0.8, False)])
def test_lower_goal_gate_matches_acceptance(value, expected):
    from carl_studio.eval.runner import EvalConfig, EvalGate, EvalReport

    config = EvalConfig(checkpoint="candidate", metric_direction="lower", threshold=0.3)
    report = EvalReport(
        checkpoint="candidate",
        phase="1",
        n_samples=2,
        metrics={"error_rate": value},
        primary_metric="error_rate",
        primary_value=value,
        threshold=0.3,
        passed=False,
        coherence={"phi_mean": 0.4, "discontinuity_score": 0.5},
    )
    assert EvalGate(threshold=config.threshold, config=config).check(report) is expected
    goal = TrainingGoal(primary_metric="error_rate", direction="lower", threshold=0.3)
    baseline = measurement(0.5, primary_metric="error_rate")
    candidate = measurement(value, primary_metric="error_rate")
    assert (compare_candidate(goal, baseline, candidate).status == "accepted") is expected


def test_local_helper_and_package_membership_are_bound(inputs, tmp_path):
    config, owner = inputs
    (tmp_path / "rules.py").write_text(
        "from helpers import score\ndef grade(completions, samples):\n return score(completions, samples)\n"
    )
    (tmp_path / "helpers.py").write_text(
        "def score(completions, samples):\n return {'task_success_rate': 0.0}\n"
    )
    config.goal = TrainingGoal(evaluator="rules:grade")
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert prepared.ready
    (tmp_path / "helpers.py").write_text(
        "def score(completions, samples):\n return {'task_success_rate': 1.0}\n"
    )
    with pytest.raises(ValidationError, match="source changed"):
        validate_preparation(prepared)


def test_project_imports_work_outside_project_and_support_dataclasses(tmp_path, monkeypatch):
    from carl_studio.training.preparation import resolve_callable

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text("VALUE = 0.75\n")
    (package / "rules.py").write_text(
        "from dataclasses import dataclass\nfrom .helpers import VALUE\n@dataclass\nclass Result:\n value: float = VALUE\ndef grade(completions, samples):\n return {'task_success_rate': Result().value}\n"
    )
    monkeypatch.chdir(tmp_path.parent)
    function = resolve_callable("pkg.rules:grade", tmp_path)
    assert function(["A"], [{}]) == {"task_success_rate": 0.75}


def test_partial_preparation_creation_recovers(inputs, tmp_path, monkeypatch):
    config, owner = inputs
    original = owner._save
    monkeypatch.setattr(owner, "_save", MagicMock(side_effect=RuntimeError("interrupted")))
    with pytest.raises(RuntimeError):
        prepare_training(config, project_root=tmp_path, manager=owner)
    monkeypatch.setattr(owner, "_save", original)
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert owner.load(prepared.plan_id).id == prepared.plan_id


def test_resume_claim_requires_stopped_execution(inputs, tmp_path):
    config, owner = inputs
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    owner.claim_training(prepared.plan_id)
    with pytest.raises(ValueError, match="reconciliation"):
        owner.claim_training(prepared.plan_id, resume=True)
    owner.save_training_result(
        prepared.plan_id, TrainingRun(id="paused", config=config, phase=RunPhase.PAUSED)
    )
    owner.claim_training(prepared.plan_id, resume=True)
    assert len(list((owner.base_dir / prepared.plan_id).glob("execution.*.claim"))) == 1


@pytest.mark.asyncio
async def test_trainer_cancellation_waits_for_worker_cleanup(inputs):
    import threading
    import time

    from carl_studio.training.trainer import CARLTrainer

    config, _ = inputs
    started = threading.Event()
    ended = threading.Event()
    trainer = CARLTrainer(config)

    class Fit:
        def add_callback(self, callback):
            self.callback = callback

        def train(self):
            started.set()
            control = SimpleNamespace(should_training_stop=False, should_save=False)
            while not control.should_training_stop:
                self.callback.on_step_end(None, None, control)
                time.sleep(0.005)
            assert control.should_save
            time.sleep(0.03)
            ended.set()

    task = asyncio.create_task(trainer._fit(Fit(), None))
    while not started.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended.is_set()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_worker_cleanup(inputs):
    import threading
    import time

    from carl_studio.training.trainer import CARLTrainer

    config, _ = inputs
    ended = threading.Event()
    cleaning = threading.Event()
    trainer = CARLTrainer(config)

    class Fit:
        def add_callback(self, callback):
            self.callback = callback

        def train(self):
            control = SimpleNamespace(should_training_stop=False, should_save=False)
            while not control.should_training_stop:
                self.callback.on_step_end(None, None, control)
                time.sleep(0.005)
            assert control.should_save
            cleaning.set()
            time.sleep(0.05)
            ended.set()

    task = asyncio.create_task(trainer._fit(Fit(), None))
    await asyncio.sleep(0.01)
    task.cancel()
    while not cleaning.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ended.is_set()


def test_coherence_failure_after_twentieth_sample_is_not_hidden(monkeypatch):
    import torch

    from carl_studio.eval.runner import EvalConfig

    calls = []

    class Model:
        config = SimpleNamespace(vocab_size=3)
        device = torch.device("cpu")
        training = False

        def eval(self):
            return self

        def __call__(self, **kwargs):
            calls.append(1)
            if len(calls) == 21:
                raise RuntimeError("private canary")
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 3.0], [3.0, 1.0, 0.0]]]))

    def tokenize(*args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    runner = EvalRunner(EvalConfig(checkpoint="fixture"))
    assert runner._compute_coherence(Model(), tokenize, ["answer"] * 21) is None
    assert len(calls) == 21


@pytest.mark.parametrize("location", ["config", "root"])
def test_dataset_resolution_from_nested_config(inputs, tmp_path, location, monkeypatch):
    config, owner = inputs
    folder = tmp_path / "configs"
    folder.mkdir()
    config_path = folder / "nested.yaml"
    config_path.write_text("{}")
    if location == "config":
        for name in ("train", "eval"):
            (folder / f"{name}.jsonl").write_bytes((tmp_path / f"{name}.jsonl").read_bytes())
    config.dataset_repo = "train.jsonl"
    config.eval_dataset_repo = "eval.jsonl"
    for cwd in (tmp_path, folder):
        monkeypatch.chdir(cwd)
        prepared = prepare_training(
            config, project_root=tmp_path, config_path=config_path, manager=owner
        )
        assert prepared.ready
        expected = folder if location == "config" else tmp_path
        assert prepared.config["dataset_repo"] == str(expected / "train.jsonl")
        validate_preparation(prepared, config)


def test_checkpoint_hash_cache_reuses_bytes_and_detects_same_size_change(tmp_path, monkeypatch):
    from carl_studio.training import preparation

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"first")
    cache = tmp_path / "cache"
    original = preparation.file_hash
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(preparation, "file_hash", counted)
    first = preparation.checkpoint_hash(shard, cache)
    assert preparation.checkpoint_hash(shard, cache) == first
    assert len(calls) == 1
    before = shard.stat()
    shard.write_bytes(b"other")
    import os

    os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert preparation.checkpoint_hash(shard, cache) != first
    assert len(calls) == 2


def test_model_index_cannot_hide_shard_mutation(inputs, tmp_path):
    config, owner = inputs
    model = Path(config.base_model)
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    (model / "model.safetensors").write_bytes(b"changed-fixture")
    with pytest.raises(ValidationError, match="source changed"):
        validate_preparation(prepared)


@pytest.mark.parametrize("baseline,status", [("starting", "rejected"), ("base", "accepted")])
def test_staged_candidate_compares_with_explicit_prepared_baseline(
    inputs, tmp_path, monkeypatch, baseline, status
):
    config, owner = inputs
    adapter = tmp_path / "sft-stage"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"bound-SFT-fixture")
    config.sft_adapter = str(adapter)
    config.comparison_baseline = baseline
    prepared = prepare_training(config, project_root=tmp_path, manager=owner)
    assert prepared.ready
    assert any(
        source.path == str(adapter / "adapter_model.safetensors") for source in prepared.sources
    )
    candidate = tmp_path / "grpo-candidate"
    candidate.mkdir()
    trainer = MagicMock()
    trainer.run = TrainingRun(
        id="staged-fixture",
        config=TrainingConfig.model_validate(prepared.config),
        phase=RunPhase.COMPLETE,
        checkpoint=str(candidate),
    )
    trainer.train = AsyncMock(return_value=trainer.run)
    factory = MagicMock(return_value=trainer)
    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", factory)
    observed = []

    def load(runner):
        observed.append(runner.config)
        return object(), object()

    def generate(runner, model, tokenizer, samples):
        competent = runner.config.checkpoint != config.base_model or runner.config.sft_adapter
        return [sample["expected_output"] if competent else "wrong" for sample in samples]

    monkeypatch.setattr(EvalRunner, "_load_model_simple", load)
    monkeypatch.setattr(EvalRunner, "_generate_single_turn", generate)
    monkeypatch.setattr(
        EvalRunner,
        "_compute_coherence",
        lambda *args: {"phi_mean": 0.4, "discontinuity_score": 0.5},
    )
    result = asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
    assert result.acceptance.status == status
    assert result.acceptance.candidate.primary_value == 1.0
    assert result.acceptance.baseline.primary_value == (1.0 if baseline == "starting" else 0.0)
    assert observed[0].sft_adapter == (str(adapter) if baseline == "starting" else None)
    assert observed[1].sft_adapter == str(adapter)
    assert factory.call_args.args[0].sft_adapter == str(adapter)
    replay = asyncio.run(submit_training(config, prepared_plan_id=prepared.plan_id, manager=owner))
    assert replay.id == result.id
    assert trainer.train.await_count == 1
    config.comparison_baseline = "base" if baseline == "starting" else "starting"
    with pytest.raises(ValidationError):
        validate_preparation(prepared, config)
