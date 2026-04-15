"""CLI regression tests for end-user workflows."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

import carl_studio.db as db_mod
import carl_studio.settings as settings_mod
import carl_studio.theme as theme_mod
import carl_studio.tier as tier_mod
from carl_studio.cli import app
from carl_studio.settings import CARLSettings
from carl_studio.types.config import ComputeTarget


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


def test_start_inventory_lists_curated_command_map(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr(settings_mod, "_find_local_config", lambda: None)
    monkeypatch.setattr(theme_mod, "THEME_FILE", tmp_path / "theme.yaml")

    result = runner.invoke(app, ["start", "--inventory"])

    assert result.exit_code == 0
    assert "Current Command Inventory" in result.output
    assert "Core Workbench" in result.output
    assert "Camp Platform" in result.output
    assert "Advanced and Lab" in result.output


def test_root_help_emphasizes_camp_and_lab_namespaces():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "camp" in result.output
    assert "lab" in result.output
    assert "login" not in result.output
    assert "subscription" not in result.output
    assert "\n│ status" not in result.output


def test_doctor_json_reports_project_blocker(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings_mod, "GLOBAL_CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr(settings_mod, "_find_local_config", lambda: None)
    monkeypatch.setattr(theme_mod, "THEME_FILE", tmp_path / "theme.yaml")

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["guided_workbench"] is False
    assert payload["blocking_issues"] == ["Create a project file with 'carl project init'"]


def test_camp_and_lab_help_expose_grouped_aliases():
    camp_result = runner.invoke(app, ["camp", "--help"])
    assert camp_result.exit_code == 0
    assert "login" in camp_result.output
    assert "sync" in camp_result.output
    assert "marketplace" in camp_result.output

    lab_result = runner.invoke(app, ["lab", "--help"])
    assert lab_result.exit_code == 0
    assert "bench" in lab_result.output
    assert "golf" in lab_result.output
    assert "admin" in lab_result.output


def test_observe_live_uses_settings_trackio_url_and_poll(monkeypatch):
    calls: dict[str, object] = {}

    def fake_run_app(**kwargs):
        calls.update(kwargs)

    fake_module = types.ModuleType("carl_studio.observe.app")
    fake_module.run_app = fake_run_app
    monkeypatch.setitem(sys.modules, "carl_studio.observe.app", fake_module)

    settings = CARLSettings()
    settings.trackio_url = "https://huggingface.co/spaces/wheattoast11/trackio"
    settings.observe_defaults.default_source = "trackio"
    settings.observe_defaults.default_poll_interval = 7.0
    monkeypatch.setattr(CARLSettings, "load", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        tier_mod, "check_tier", lambda feature: (True, settings.tier, settings.tier)
    )

    result = runner.invoke(app, ["observe", "--live", "--run", "demo-run"])

    assert result.exit_code == 0
    assert calls["source"] == "trackio"
    assert calls["space"] == "wheattoast11-trackio"
    assert calls["run"] == "demo-run"
    assert calls["poll"] == 7.0


def test_project_init_uses_settings_defaults_noninteractive(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    settings = CARLSettings()
    settings.default_model = "Qwen/Qwen3.5-9B"
    settings.default_compute = ComputeTarget.A100_LARGE
    settings.hub_namespace = "demo-org"
    settings.trackio_url = "https://trackio.example"
    monkeypatch.setattr(CARLSettings, "load", classmethod(lambda cls: settings))

    result = runner.invoke(app, ["project", "init", "--name", "legal-reviewer", "--no-interactive"])

    assert result.exit_code == 0
    assert "Project saved to carl.yaml" in result.output
    payload = yaml.safe_load((tmp_path / "carl.yaml").read_text())
    assert payload["base_model"] == "Qwen/Qwen3.5-9B"
    assert payload["compute_target"] == "a100-large"
    assert payload["output_repo"] == "demo-org/legal-reviewer"
    assert payload["tracking_url"] == "https://trackio.example"


def test_train_persists_local_run_and_prints_run_guidance(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")

    fake_run = types.SimpleNamespace(
        id="run-123",
        phase=types.SimpleNamespace(value="training"),
        hub_job_id="job-123",
        error_message=None,
        checkpoint_steps=[],
    )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        async def train(self):
            return fake_run

    monkeypatch.setattr("carl_studio.training.trainer.CARLTrainer", FakeTrainer)
    monkeypatch.setattr("anyio.run", lambda func, *args, **kwargs: fake_run)

    result = runner.invoke(
        app,
        [
            "train",
            "--model",
            "Qwen/Qwen3.5-9B",
            "--method",
            "grpo",
            "--dataset",
            "org/data",
            "--output-repo",
            "org/model",
            "--compute",
            "a100",
        ],
    )

    assert result.exit_code == 0
    assert "Monitor via local run: carl run status run-123" in result.output
    assert "Run details: carl run show run-123" in result.output

    from carl_studio.db import LocalDB

    stored = LocalDB().get_run("run-123")
    assert stored is not None
    assert stored["remote_id"] == "job-123"
    assert stored["mode"] == "train:grpo"


def test_run_status_resolves_local_run_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")

    from carl_studio.db import LocalDB

    LocalDB().insert_run(
        {
            "id": "run-abc",
            "model_id": "org/model",
            "mode": "train:grpo",
            "status": "training",
            "hardware": "a100-large",
            "config": {"output_repo": "org/model"},
            "result": {"hub_job_id": "job-abc"},
            "remote_id": "job-abc",
            "started_at": "2026-04-14T00:00:00Z",
        }
    )

    fake_hf = types.ModuleType("huggingface_hub")

    class FakeStatus:
        stage = "running"
        message = "warming up"

    class FakeJob:
        status = FakeStatus()
        flavor = "a100-large"

    class FakeApi:
        def inspect_job(self, job_id: str):
            assert job_id == "job-abc"
            return FakeJob()

    fake_hf.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    result = runner.invoke(app, ["run", "status", "run-abc"])

    assert result.exit_code == 0
    assert "run-abc" in result.output
    assert "job-abc" in result.output
    assert "warming up" in result.output

    updated = LocalDB().get_run("run-abc")
    assert updated is not None
    assert updated["status"] == "training"
    assert updated["result"]["remote_status"] == "running"


def test_run_list_and_show_json(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(db_mod, "CARL_DIR", tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "carl.db")

    from carl_studio.db import LocalDB

    LocalDB().insert_run(
        {
            "id": "run-json",
            "model_id": "org/model",
            "mode": "pipeline",
            "status": "training",
            "hardware": "l40sx1",
            "config": {"output_repo": "org/model"},
            "result": {"hub_job_id": "job-json"},
            "remote_id": "job-json",
            "started_at": "2026-04-14T00:00:00Z",
        }
    )

    list_result = runner.invoke(app, ["run", "list", "--json"])
    assert list_result.exit_code == 0
    payload = json.loads(list_result.output)
    assert payload[0]["id"] == "run-json"
    assert payload[0]["remote_id"] == "job-json"

    show_result = runner.invoke(app, ["run", "show", "run-json", "--json"])
    assert show_result.exit_code == 0
    show_payload = json.loads(show_result.output)
    assert show_payload["id"] == "run-json"
    assert show_payload["result"]["hub_job_id"] == "job-json"
