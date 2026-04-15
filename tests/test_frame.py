"""Tests for carl_studio.frame -- WorkFrame model, persistence, decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from carl_studio.frame import WorkFrame


class TestWorkFrame:
    def test_empty_frame_inactive(self) -> None:
        frame = WorkFrame()
        assert frame.active is False

    def test_frame_with_domain_is_active(self) -> None:
        frame = WorkFrame(domain="saas_sales")
        assert frame.active is True

    def test_attention_query_includes_all_set_dims(self) -> None:
        frame = WorkFrame(
            domain="pharma",
            function="drug_rollout",
            role="manager",
            objectives=["maximize adoption"],
            entities=["territory", "rep", "HCP"],
            metrics=["coverage", "scripts"],
        )
        q = frame.attention_query()
        assert "Domain: pharma" in q
        assert "Function: drug_rollout" in q
        assert "Role perspective: manager" in q
        assert "maximize adoption" in q
        assert "territory" in q
        assert "coverage" in q

    def test_attention_query_empty_frame(self) -> None:
        frame = WorkFrame()
        assert frame.attention_query() == ""

    def test_decompose_returns_lanes(self) -> None:
        frame = WorkFrame(
            domain="saas",
            function="quota_setting",
            objectives=["hit target", "retain sellers"],
            metrics=["attainment", "churn"],
        )
        lanes = frame.decompose()
        assert "domain:saas" in lanes
        assert "function:quota_setting" in lanes
        assert "objective:hit target" in lanes
        assert "metric:attainment" in lanes

    def test_decompose_empty_returns_general(self) -> None:
        assert WorkFrame().decompose() == ["general"]

    def test_entity_patterns(self) -> None:
        frame = WorkFrame(entities=["Account", "Seller", "Territory"])
        patterns = frame.entity_patterns()
        assert "account" in patterns
        assert "seller" in patterns


class TestWorkFramePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        frame = WorkFrame(
            domain="logistics",
            function="route_planning",
            role="analyst",
            objectives=["minimize cost", "maximize coverage"],
            entities=["depot", "vehicle", "route"],
        )
        path = frame.save(tmp_path / "frame.yaml")
        loaded = WorkFrame.load(path)
        assert loaded.domain == "logistics"
        assert loaded.function == "route_planning"
        assert loaded.objectives == ["minimize cost", "maximize coverage"]
        assert loaded.entities == ["depot", "vehicle", "route"]

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        frame = WorkFrame.load(tmp_path / "nope.yaml")
        assert frame.active is False

    def test_from_project_with_frame_key(self, tmp_path: Path) -> None:
        import yaml

        config = {
            "base_model": "test/model",
            "frame": {
                "domain": "healthcare",
                "function": "vaccine_rollout",
                "role": "exec",
            },
        }
        config_path = tmp_path / "carl.yaml"
        config_path.write_text(yaml.dump(config))
        frame = WorkFrame.from_project(str(config_path))
        assert frame.domain == "healthcare"
        assert frame.function == "vaccine_rollout"


class TestFrameCLI:
    def _make_app(self) -> typer.Typer:
        from carl_studio.cli.frame import frame_app

        app = typer.Typer()
        app.add_typer(frame_app, name="frame")
        return app

    def test_set_and_show(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        runner = CliRunner()
        app = self._make_app()

        result = runner.invoke(app, [
            "frame", "set",
            "--domain", "saas_sales",
            "--function", "territory_planning",
            "--role", "analyst",
            "--goal", "maximize coverage",
        ])
        assert result.exit_code == 0
        assert "Frame saved" in result.output

        result = runner.invoke(app, ["frame", "show", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["domain"] == "saas_sales"
        assert data["function"] == "territory_planning"

    def test_clear(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        runner = CliRunner()
        app = self._make_app()

        runner.invoke(app, ["frame", "set", "--domain", "test"])
        result = runner.invoke(app, ["frame", "clear"])
        assert result.exit_code == 0
        assert "cleared" in result.output

    def test_set_no_dims_errors(self) -> None:
        runner = CliRunner()
        app = self._make_app()
        result = runner.invoke(app, ["frame", "set"])
        assert result.exit_code == 1
