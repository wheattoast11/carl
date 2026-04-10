"""CLI regression tests for end-user workflows."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from carl_studio.cli import app


runner = CliRunner()


class DummyTrackioSource:
    def __init__(self, space: str, project: str = "", run: str = "", token: str | None = None):
        self.space = space
        self.project = project
        self.run = run
        self.token = token
        self.resolved_project = project or "demo-project"
        self.resolved_run = run or "demo-run"

    def poll(self):
        return []


def test_train_missing_fields_has_actionable_error():
    result = runner.invoke(
        app, ["train", "--model", "Qwen/Qwen3.5-9B", "--method", "grpo", "--compute", "a100"]
    )
    assert result.exit_code == 1
    assert "Training config is incomplete or invalid." in result.output
    assert "Missing required fields: dataset_repo, output_repo" in result.output
    assert "a100-large" in result.output
    assert "ValidationError" not in result.output


def test_train_accepts_compute_alias_and_default_run_name(tmp_path: Path):
    config = tmp_path / "carl.yaml"
    config.write_text(
        "\n".join(
            [
                "base_model: Tesslate/OmniCoder-9B",
                "method: grpo",
                "dataset_repo: org/data",
                "output_repo: org/model",
                "compute_target: a100",
            ]
        )
    )

    result = runner.invoke(app, ["train", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0
    assert "a100-large" in result.output
    assert "org/model" in result.output


def test_observe_live_forwards_normalized_space_project_and_run(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_app(**kwargs):
        calls.update(kwargs)

    fake_module = types.ModuleType("carl_studio.observe.app")
    fake_module.run_app = fake_run_app
    monkeypatch.setitem(sys.modules, "carl_studio.observe.app", fake_module)

    result = runner.invoke(
        app,
        [
            "observe",
            "--live",
            "--url",
            "https://huggingface.co/spaces/wheattoast11/trackio",
            "--project",
            "demo-project",
            "--run",
            "demo-run",
        ],
    )

    assert result.exit_code == 0
    assert calls["source"] == "trackio"
    assert calls["space"] == "wheattoast11-trackio"
    assert calls["project"] == "demo-project"
    assert calls["run"] == "demo-run"


def test_observe_reports_trackio_failures(monkeypatch):
    class FailingTrackioSource(DummyTrackioSource):
        def poll(self):
            raise RuntimeError("Trackio API unreachable")

    monkeypatch.setattr("carl_studio.observe.data_source.TrackioSource", FailingTrackioSource)

    result = runner.invoke(
        app, ["observe", "--url", "https://wheattoast11-trackio.hf.space/", "--run", "demo-run"]
    )
    assert result.exit_code == 1
    assert "Trackio API unreachable" in result.output
    assert "No data found" not in result.output


def test_observe_one_shot_forwards_project_and_run(monkeypatch):
    monkeypatch.setattr("carl_studio.observe.data_source.TrackioSource", DummyTrackioSource)

    result = runner.invoke(
        app,
        [
            "observe",
            "--url",
            "https://wheattoast11-trackio.hf.space/",
            "--project",
            "demo-project",
            "--run",
            "demo-run",
        ],
    )

    assert result.exit_code == 0
    assert "project: demo-project" in result.output
    assert "run: demo-run" in result.output


def test_observe_live_missing_tui_extra_shows_install_hint(monkeypatch):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "carl_studio.observe.app":
            raise ImportError("No module named 'textual'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = runner.invoke(
        app, ["observe", "--live", "--url", "https://wheattoast11-trackio.hf.space/"]
    )
    assert result.exit_code == 1
    assert "install carl-studio[tui]" in result.output


def test_observe_diagnose_missing_extra_shows_install_hint(monkeypatch, tmp_path: Path):
    log_file = tmp_path / "train.jsonl"
    log_file.write_text(json.dumps({"step": 1, "phi": 0.5, "loss": 1.0, "reward_mean": 0.1}) + "\n")

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = runner.invoke(
        app,
        [
            "observe",
            "--file",
            str(log_file),
            "--diagnose",
            "--api-key",
            "dummy",
        ],
    )
    assert result.exit_code == 0
    assert "install 'carl-studio[observe]'" in result.output


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["status", "job-123"], "install carl-studio[hf]"),
        (["logs", "job-123"], "install carl-studio[hf]"),
        (["stop", "job-123"], "install carl-studio[hf]"),
    ],
)
def test_hf_commands_missing_extra_show_install_hint(monkeypatch, command, expected):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = runner.invoke(app, command)
    assert result.exit_code == 1
    assert expected in result.output


def test_push_forwards_arguments(monkeypatch, tmp_path: Path):
    artifact_dir = tmp_path / "adapter"
    artifact_dir.mkdir()
    (artifact_dir / "adapter_model.safetensors").write_text("weights")

    mock_push = Mock(return_value="https://huggingface.co/org/model")
    monkeypatch.setattr("carl_studio.hub.models.push_with_metadata", mock_push)
    monkeypatch.setattr("anyio.run", lambda func, *args: func(*args))

    result = runner.invoke(
        app,
        [
            "push",
            str(artifact_dir),
            "--repo",
            "org/model",
            "--base-model",
            "Tesslate/OmniCoder-9B",
            "--method",
            "grpo",
            "--dataset",
            "org/data",
            "--private",
        ],
    )

    assert result.exit_code == 0
    mock_push.assert_called_once_with(
        str(artifact_dir),
        "org/model",
        "Tesslate/OmniCoder-9B",
        "grpo",
        "org/data",
        None,
        True,
    )
    assert "Published: https://huggingface.co/org/model" in result.output


def test_push_missing_path_errors():
    result = runner.invoke(app, ["push", "/tmp/does-not-exist", "--repo", "org/model"])
    assert result.exit_code == 1
    assert "Model path not found" in result.output
