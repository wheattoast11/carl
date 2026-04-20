"""Tests for the one-agent CLI surface (F1 — two canonical surfaces):

- Bare ``carl`` prints help + a 1-line nudge (no auto-route to chat).
- ``carl ask "<prompt>"`` runs a single agent turn.
- ``carl chat`` is the interactive loop.

The former bare-routing-to-chat behaviour was retired in v0.7 so new
users discover both entry points explicitly instead of landing in the
interactive loop by accident.
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
    def test_bare_carl_prints_help_and_nudge(self) -> None:
        """Bare ``carl`` must show help and point users at chat/ask."""
        _register_routing()
        runner = CliRunner()
        with patch("carl_studio.cli.chat.chat_cmd") as fake_chat:
            result = runner.invoke(app, [])
        # F1 — bare `carl` no longer auto-routes to chat. The callback
        # still fires (to render help + nudge) but chat_cmd MUST NOT run.
        assert not fake_chat.called, result.output
        assert result.exit_code == 0
        assert "carl chat" in result.output
        assert "carl ask" in result.output

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
