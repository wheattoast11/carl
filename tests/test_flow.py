"""Tests for carl flow — slash-prefixed op chain execution."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.cli import flow as flow_mod
from carl_studio.cli import operations as ops_mod
from carl_studio.cli.apps import app


def _ensure_wired() -> None:
    import carl_studio.cli.wiring  # noqa: F401


class TestParseChain:
    def test_single_op_no_args(self) -> None:
        assert flow_mod.parse_chain("/doctor") == [("doctor", [])]

    def test_single_op_with_args(self) -> None:
        assert flow_mod.parse_chain("/echo hello world") == [("echo", ["hello", "world"])]

    def test_multiple_ops(self) -> None:
        result = flow_mod.parse_chain("/echo hi /echo bye")
        assert result == [("echo", ["hi"]), ("echo", ["bye"])]

    def test_quoted_argument(self) -> None:
        result = flow_mod.parse_chain('/ask "train a small model"')
        assert result == [("ask", ["train a small model"])]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            flow_mod.parse_chain("")

    def test_arg_before_any_op_raises(self) -> None:
        with pytest.raises(ValueError):
            flow_mod.parse_chain("bareword /echo x")

    def test_lone_slash_is_treated_as_arg(self) -> None:
        # A single "/" is not a valid op name (len <= 1)
        with pytest.raises(ValueError):
            flow_mod.parse_chain("/ hello")


class TestOperationsRegistry:
    def test_has_core_ops(self) -> None:
        ops = ops_mod.list_operations()
        assert "doctor" in ops
        assert "echo" in ops
        assert "freshness" in ops

    def test_get_unknown_returns_none(self) -> None:
        assert ops_mod.get_operation("does-not-exist") is None

    def test_echo_records_step(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["echo"](chain, ["hello", "world"])
        assert len(chain) == 1
        step = chain.last()
        assert step is not None
        assert step.action == ActionType.CLI_CMD
        assert step.name == "echo"
        assert step.input == {"args": ["hello", "world"]}
        assert step.output == {"message": "hello world"}


class TestFlowCommand:
    def test_dry_run_renders_plan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo foo /echo bar", "--dry-run"])
        assert result.exit_code == 0
        assert "/echo foo" in result.output
        assert "/echo bar" in result.output
        # No trace file written on dry-run
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_list_ops(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "--list"])
        assert result.exit_code == 0
        for name in ops_mod.list_operations():
            assert f"/{name}" in result.output

    def test_missing_chain_errors(self) -> None:
        _ensure_wired()
        runner = CliRunner()
        result = runner.invoke(app, ["flow"])
        assert result.exit_code != 0

    def test_execute_persists_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo ping"])
        assert result.exit_code == 0
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "ping" in content
        assert "echo" in content

    def test_no_persist_skips_disk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/echo x", "--no-persist"])
        assert result.exit_code == 0
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_unknown_op_continues_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ensure_wired()
        monkeypatch.setattr(flow_mod, "INTERACTIONS_DIR", tmp_path)
        runner = CliRunner()
        result = runner.invoke(app, ["flow", "/nope /echo survived"])
        assert result.exit_code == 0
        # Trace records the unknown op as failed AND the echo as success
        files = list(tmp_path.glob("*.jsonl"))
        content = files[0].read_text()
        assert "unknown:/nope" in content
        assert "survived" in content
