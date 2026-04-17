"""Tests for the one-agent CLI surface:
  - Bare `carl` routes to chat via the default callback
  - `carl ask "<prompt>"` runs a single agent turn
"""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from carl_studio.cli.apps import app


def _register_routing() -> None:
    """Ensure wiring has registered the default callback + ask command."""
    # Importing wiring triggers registration on `app`.
    import carl_studio.cli.wiring  # noqa: F401


class TestDefaultCallback:
    def test_bare_carl_calls_chat(self) -> None:
        _register_routing()
        runner = CliRunner()
        with patch("carl_studio.cli.chat.chat_cmd") as fake_chat:
            result = runner.invoke(app, [])
        # Callback fires when no subcommand is invoked
        assert fake_chat.called, result.output

    def test_subcommand_skips_default(self) -> None:
        _register_routing()
        runner = CliRunner()
        with patch("carl_studio.cli.chat.chat_cmd") as fake_chat:
            # --help is a valid "command" in Typer's eyes; use a real subcommand
            runner.invoke(app, ["init", "--help"])
        assert not fake_chat.called


class TestAskCommand:
    def test_ask_registered(self) -> None:
        _register_routing()
        runner = CliRunner()
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Ask Carl a single question" in result.output

    def test_ask_invokes_run_one_shot(self) -> None:
        _register_routing()
        runner = CliRunner()
        with patch("carl_studio.cli.chat.run_one_shot_agent") as fake_run:
            runner.invoke(app, ["ask", "train a small model"])
        assert fake_run.called
        args, kwargs = fake_run.call_args
        assert args[0] == "train a small model"
