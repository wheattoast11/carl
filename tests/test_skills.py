"""Tests for the CARL Studio Skills system.

No GPU, no network, no heavy ML deps required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ─── Conftest already stubs carl_studio; add skills subpackages here ─────────

import sys
import types

def _ensure_skills_loaded() -> None:
    """Load skills subpackages using importlib, same pattern as conftest."""
    import importlib.util

    # Register bare subpackage stubs so child imports can resolve parents.
    for name, path in {
        "carl_studio.skills": "src/carl_studio/skills",
        "carl_studio.skills.builtins": "src/carl_studio/skills/builtins",
    }.items():
        if name not in sys.modules:
            sub = types.ModuleType(name)
            sub.__path__ = [path]  # type: ignore[attr-defined]
            sys.modules[name] = sub

    # Now load the real __init__.py files so exports are available.
    _init_modules = {
        "carl_studio.skills.base": "src/carl_studio/skills/base.py",
        "carl_studio.skills.runner": "src/carl_studio/skills/runner.py",
        "carl_studio.skills": "src/carl_studio/skills/__init__.py",
        "carl_studio.skills.builtins.observer": "src/carl_studio/skills/builtins/observer.py",
        "carl_studio.skills.builtins.grader": "src/carl_studio/skills/builtins/grader.py",
        "carl_studio.skills.builtins.trainer": "src/carl_studio/skills/builtins/trainer.py",
        "carl_studio.skills.builtins.synthesizer": "src/carl_studio/skills/builtins/synthesizer.py",
        "carl_studio.skills.builtins.deployer": "src/carl_studio/skills/builtins/deployer.py",
        "carl_studio.skills.builtins": "src/carl_studio/skills/builtins/__init__.py",
        "carl_studio.skills._cli": "src/carl_studio/skills/_cli.py",
    }
    for name, path in _init_modules.items():
        existing = sys.modules.get(name)
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                if existing is None:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[name] = mod
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                else:
                    spec.loader.exec_module(existing)  # type: ignore[union-attr]
        except Exception:
            pass  # Some modules have optional deps — that's fine


_ensure_skills_loaded()

from carl_studio.skills.base import BaseSkill, SkillResult, SkillRun
from carl_studio.skills.runner import SkillRegistry, SkillRunner
from carl_studio.skills.builtins.observer import ObserverSkill
from carl_studio.skills.builtins.grader import GraderSkill
from carl_studio.skills.builtins.trainer import TrainerSkill
from carl_studio.skills.builtins.synthesizer import SynthesizerSkill
from carl_studio.skills.builtins.deployer import DeployerSkill
from carl_studio.skills.builtins import BUILTIN_SKILLS


# ─── Test fixtures ────────────────────────────────────────────────────────────

class _PassSkill(BaseSkill):
    name = "test_pass"
    badge = "Test Pass Badge"
    description = "Always succeeds and awards a badge."
    requires_tier = "free"

    def execute(self, **kwargs: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            success=True,
            badge_earned=True,
            message="All good",
            metrics={"score": 1.0},
        )


class _FailSkill(BaseSkill):
    name = "test_fail"
    badge = "Test Fail Badge"
    description = "Always fails without awarding a badge."
    requires_tier = "free"

    def execute(self, **kwargs: Any) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            success=False,
            badge_earned=False,
            message="Expected failure",
        )


class _RaisingSkill(BaseSkill):
    name = "test_raise"
    badge = "Raise Badge"
    description = "Raises an exception during execute."
    requires_tier = "free"

    def execute(self, **kwargs: Any) -> SkillResult:
        raise RuntimeError("unexpected crash")


@pytest.fixture()
def tmp_runner(tmp_path: Path) -> SkillRunner:
    """SkillRunner backed by a temp SQLite DB, cleaned up after test."""
    db = tmp_path / "test_carl.db"
    runner = SkillRunner(db_path=db)
    yield runner
    runner.close()


# ─── BaseSkill contract ───────────────────────────────────────────────────────

@pytest.mark.parametrize("skill", BUILTIN_SKILLS)
def test_builtin_has_required_class_vars(skill: BaseSkill) -> None:
    assert isinstance(skill.name, str) and skill.name, f"{type(skill).__name__}.name must be non-empty str"
    assert isinstance(skill.badge, str) and skill.badge, f"{type(skill).__name__}.badge must be non-empty str"
    assert isinstance(skill.description, str) and skill.description, f"{type(skill).__name__}.description must be non-empty str"
    assert skill.requires_tier in ("free", "paid"), f"{type(skill).__name__}.requires_tier must be 'free' or 'paid'"


@pytest.mark.parametrize("skill", BUILTIN_SKILLS)
def test_builtin_to_mcp_schema(skill: BaseSkill) -> None:
    schema = skill.to_mcp_schema()
    assert schema["name"] == f"skill_{skill.name}"
    assert skill.badge in schema["description"]
    assert "inputSchema" in schema


def test_skill_result_defaults() -> None:
    r = SkillResult(skill_name="x", success=True)
    assert not r.badge_earned
    assert r.metrics == {}
    assert r.artifact == {}
    assert r.message == ""


def test_skill_run_model() -> None:
    run = SkillRun(id=1, skill_name="observer", success=True, started_at="2026-04-12T00:00:00Z")
    assert run.id == 1
    assert run.badge_earned is False


# ─── SkillRegistry ────────────────────────────────────────────────────────────

def test_registry_register_and_get() -> None:
    reg = SkillRegistry()
    skill = _PassSkill()
    reg.register(skill)
    assert reg.get("test_pass") is skill


def test_registry_get_missing_returns_none() -> None:
    reg = SkillRegistry()
    assert reg.get("nonexistent") is None


def test_registry_list_skills() -> None:
    reg = SkillRegistry()
    reg.register(_PassSkill())
    reg.register(_FailSkill())
    names = {s.name for s in reg.list_skills()}
    assert names == {"test_pass", "test_fail"}


# ─── SkillRunner.register / get ──────────────────────────────────────────────

def test_runner_register_and_get(tmp_runner: SkillRunner) -> None:
    skill = _PassSkill()
    tmp_runner.register(skill)
    assert tmp_runner.get("test_pass") is skill


def test_runner_get_unregistered(tmp_runner: SkillRunner) -> None:
    assert tmp_runner.get("ghost") is None


# ─── SkillRunner.run — success path ──────────────────────────────────────────

def test_runner_run_success_returns_result(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    result = tmp_runner.run("test_pass")
    assert result.success is True
    assert result.badge_earned is True
    assert result.skill_name == "test_pass"
    assert result.message == "All good"


def test_runner_run_records_to_db(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.run("test_pass")
    history = tmp_runner.get_history(skill_name="test_pass")
    assert len(history) == 1
    assert history[0]["badge_earned"] == 1
    assert history[0]["success"] == 1


# ─── SkillRunner.run — failure path ──────────────────────────────────────────

def test_runner_run_failure_no_badge(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_FailSkill())
    result = tmp_runner.run("test_fail")
    assert result.success is False
    assert result.badge_earned is False


def test_runner_run_failure_recorded(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_FailSkill())
    tmp_runner.run("test_fail")
    history = tmp_runner.get_history(skill_name="test_fail")
    assert len(history) == 1
    assert history[0]["badge_earned"] == 0
    assert history[0]["success"] == 0


# ─── SkillRunner.run — missing skill ─────────────────────────────────────────

def test_runner_run_missing_skill_returns_failure(tmp_runner: SkillRunner) -> None:
    result = tmp_runner.run("does_not_exist")
    assert result.success is False
    assert "not found" in result.message


# ─── SkillRunner.run — exception inside skill ────────────────────────────────

def test_runner_run_exception_caught(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_RaisingSkill())
    result = tmp_runner.run("test_raise")
    assert result.success is False
    assert "unexpected crash" in result.message


# ─── SkillRunner.get_history ─────────────────────────────────────────────────

def test_runner_get_history_all(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.register(_FailSkill())
    tmp_runner.run("test_pass")
    tmp_runner.run("test_fail")
    history = tmp_runner.get_history()
    assert len(history) == 2


def test_runner_get_history_filtered(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.register(_FailSkill())
    tmp_runner.run("test_pass")
    tmp_runner.run("test_fail")
    history = tmp_runner.get_history(skill_name="test_pass")
    assert all(r["skill_name"] == "test_pass" for r in history)


def test_runner_get_history_respects_limit(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    for _ in range(5):
        tmp_runner.run("test_pass")
    history = tmp_runner.get_history(limit=3)
    assert len(history) == 3


def test_runner_get_history_returns_list_of_dicts(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.run("test_pass")
    history = tmp_runner.get_history()
    assert isinstance(history, list)
    assert isinstance(history[0], dict)
    assert "skill_name" in history[0]


# ─── SkillRunner.badges_earned ───────────────────────────────────────────────

def test_runner_badges_earned_empty(tmp_runner: SkillRunner) -> None:
    assert tmp_runner.badges_earned() == []


def test_runner_badges_earned_after_pass(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.run("test_pass")
    earned = tmp_runner.badges_earned()
    assert "test_pass" in earned


def test_runner_badges_earned_excludes_failures(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_FailSkill())
    tmp_runner.run("test_fail")
    earned = tmp_runner.badges_earned()
    assert "test_fail" not in earned


def test_runner_badges_earned_deduplicated(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    for _ in range(3):
        tmp_runner.run("test_pass")
    earned = tmp_runner.badges_earned()
    assert earned.count("test_pass") == 1


# ─── SkillRunner.run with run_id ─────────────────────────────────────────────

def test_runner_run_with_run_id(tmp_runner: SkillRunner) -> None:
    tmp_runner.register(_PassSkill())
    tmp_runner.run("test_pass", run_id="abc-123")
    history = tmp_runner.get_history(skill_name="test_pass")
    assert history[0]["run_id"] == "abc-123"


# ─── ObserverSkill graceful failure ──────────────────────────────────────────

def test_observer_no_source_returns_failure() -> None:
    skill = ObserverSkill()
    result = skill.execute()
    assert result.success is False
    assert "url" in result.message.lower() or "file" in result.message.lower()


def test_observer_missing_file_returns_failure() -> None:
    skill = ObserverSkill()
    result = skill.execute(file="/tmp/definitely_does_not_exist_carl_test.jsonl")
    # Either no data found or an error — must not raise
    assert result.success is False


def test_observer_valid_jsonl(tmp_path: Path) -> None:
    """Smoke test with a minimal JSONL file."""
    log = tmp_path / "metrics.jsonl"
    frames = [
        {"step": i, "phi": 0.5, "loss": 1.0, "reward_mean": 0.8}
        for i in range(5)
    ]
    log.write_text("\n".join(json.dumps(f) for f in frames))
    skill = ObserverSkill()
    result = skill.execute(file=str(log))
    assert result.success is True
    assert result.metrics["n_frames"] == 5
    assert "health" in result.metrics


# ─── GraderSkill graceful failure (no training deps) ─────────────────────────

def test_grader_without_eval_runner_returns_failure() -> None:
    """If EvalRunner is missing or raises, GraderSkill returns success=False."""
    skill = GraderSkill()
    # Pass a checkpoint that will fail since we're not in a GPU environment
    result = skill.execute(checkpoint="nonexistent/model")
    # Must not raise — either ImportError or runtime error caught
    assert isinstance(result, SkillResult)
    assert result.success is False


# ─── SynthesizerSkill graceful failure ───────────────────────────────────────

def test_synthesizer_error_returns_failure() -> None:
    skill = SynthesizerSkill()
    result = skill.execute(source="/nonexistent/path")
    assert isinstance(result, SkillResult)
    assert result.success is False
    assert result.message  # has an error message


# ─── DeployerSkill graceful failure ──────────────────────────────────────────

def test_deployer_missing_deps_returns_failure() -> None:
    skill = DeployerSkill()
    result = skill.execute(model_path="/nonexistent", repo_id="user/model")
    assert isinstance(result, SkillResult)
    assert result.success is False


# ─── CLI tests ───────────────────────────────────────────────────────────────

from typer.testing import CliRunner as TyperRunner
from carl_studio.skills._cli import skills_app

_cli_runner = TyperRunner()


def test_skill_list_command() -> None:
    result = _cli_runner.invoke(skills_app, ["list"])
    assert result.exit_code == 0
    # All built-in skill names should appear in output
    for skill in BUILTIN_SKILLS:
        assert skill.name in result.output


def test_skill_run_missing_skill_exits_nonzero() -> None:
    result = _cli_runner.invoke(skills_app, ["run", "nonexistent_skill_xyz"])
    assert result.exit_code != 0


def test_skill_run_invalid_json_exits_nonzero() -> None:
    result = _cli_runner.invoke(skills_app, ["run", "observer", "--inputs", "not-valid-json"])
    assert result.exit_code != 0


def test_skill_badges_command_no_badges() -> None:
    result = _cli_runner.invoke(skills_app, ["badges"])
    assert result.exit_code == 0
    # When no badges, should say so gracefully
    assert result.output  # non-empty output


def test_skill_history_command_empty() -> None:
    result = _cli_runner.invoke(skills_app, ["history"])
    assert result.exit_code == 0


def test_skill_history_command_with_filter() -> None:
    result = _cli_runner.invoke(skills_app, ["history", "--skill", "observer"])
    assert result.exit_code == 0


def test_skill_list_shows_badges_column() -> None:
    result = _cli_runner.invoke(skills_app, ["list"])
    assert result.exit_code == 0
    assert "Badge" in result.output
    assert "Tier" in result.output
