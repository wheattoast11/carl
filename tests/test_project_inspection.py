"""Project inspection accepts current and legacy adapter field names."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from carl_studio.cli.project_data import project_app
from carl_studio.project import load_project


@pytest.mark.parametrize("field", ["adapter", "backend"])
def test_project_show_resolves_adapter_schema(tmp_path, field):
    config = tmp_path / "project.yaml"
    config.write_text(f"{field}: unsloth\ncompute_target: l4x1\ncompute_backend: hf_jobs\n")
    assert load_project(config).adapter == "unsloth"
    result = CliRunner().invoke(project_app, ["show", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "unsloth" in result.output
    assert "hf_jobs" in result.output


def test_current_adapter_wins_over_legacy_backend(tmp_path):
    config = tmp_path / "project.yaml"
    config.write_text("adapter: trl\nbackend: unsloth\n")
    project = load_project(config)
    assert project.adapter == "trl"
    assert "backend" not in project.model_dump()
