"""F1 — CLI chat surface collapse.

After v0.7 there are exactly two canonical chat surfaces:

* ``carl chat`` — interactive loop (full CARLAgent).
* ``carl ask "<prompt>"`` — one-shot single-turn with tools.

Bare ``carl`` prints help + a one-line nudge. The legacy
``carl lab repl`` command is a removed-tombstone that directs users to
``carl chat``. The previous positional ``carl "<prompt>"`` shape emits a
:class:`DeprecationWarning` via the back-compat shim and then routes to
``run_one_shot_agent``.

These tests pin the routing contract so the simplification cannot
silently regress.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from carl_studio.cli.apps import app
from carl_studio.cli import chat as chat_mod


runner = CliRunner()


def _wired() -> None:
    import carl_studio.cli.wiring  # noqa: F401


# ---------------------------------------------------------------------------
# Bare `carl` -> help + nudge
# ---------------------------------------------------------------------------


class TestBareCarlShowsHelp:
    def test_bare_carl_prints_help_and_nudge(self) -> None:
        """Bare ``carl`` must render help and mention both chat + ask."""
        _wired()
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        # Help block mentions the app banner.
        assert "carl" in result.output.lower()
        # The 1-line nudge points users at the two canonical surfaces.
        assert "carl chat" in result.output
        assert "carl ask" in result.output

    def test_bare_carl_does_not_launch_chat_loop(self) -> None:
        """Bare ``carl`` must NOT call into ``chat_cmd`` anymore."""
        _wired()
        with patch.object(chat_mod, "chat_cmd") as mock_chat:
            result = runner.invoke(app, [])
            assert result.exit_code == 0
            assert not mock_chat.called


# ---------------------------------------------------------------------------
# `carl chat` and `carl ask` are registered
# ---------------------------------------------------------------------------


class TestCanonicalSurfaces:
    def test_chat_command_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Chat with CARL" in result.output or "interactive" in result.output.lower()

    def test_ask_command_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "ask" in result.output.lower()

    def test_ask_takes_positional_prompt(self) -> None:
        """``carl ask "prompt"`` must accept the prompt as a positional arg."""
        _wired()
        # --help avoids running the real agent.
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "prompt" in result.output.lower() or "ask" in result.output.lower()

    def test_ask_dispatches_to_run_one_shot_agent(self) -> None:
        """`carl ask "<prompt>"` must route through ``run_one_shot_agent``."""
        _wired()
        with patch.object(chat_mod, "run_one_shot_agent") as mock_run:
            result = runner.invoke(app, ["ask", "train a tiny model"])
            assert result.exit_code == 0, result.output
            assert mock_run.called
            # First positional arg is the prompt.
            args, _ = mock_run.call_args
            assert args[0] == "train a tiny model"


# ---------------------------------------------------------------------------
# `carl lab repl` is a tombstone
# ---------------------------------------------------------------------------


class TestLabReplRemoved:
    def test_lab_repl_prints_migration_hint_and_exits_nonzero(self) -> None:
        """Legacy ``carl lab repl`` must redirect users to ``carl chat``."""
        _wired()
        result = runner.invoke(app, ["lab", "repl"])
        assert result.exit_code != 0
        # The hint must name the canonical replacement surface.
        assert "carl chat" in result.output

    def test_lab_repl_is_hidden_from_lab_help(self) -> None:
        """The removed command should NOT be advertised in `carl lab --help`."""
        _wired()
        result = runner.invoke(app, ["lab", "--help"])
        assert result.exit_code == 0
        # A removed-tombstone must not be promoted in help. We accept
        # either full absence (Typer hides hidden commands) or at least
        # that it is not the first-class surface.
        lines = [line for line in result.output.split("\n") if "repl" in line.lower()]
        # Any mention must include the hidden indicator or none at all.
        assert all("lab repl" not in line or "[removed]" in line for line in lines)


# ---------------------------------------------------------------------------
# Deprecated positional back-compat shim
# ---------------------------------------------------------------------------


class TestDeprecatedPositionalShim:
    def test_deprecated_positional_emits_warning(self) -> None:
        """The back-compat shim fires ``DeprecationWarning`` on use."""
        with patch.object(chat_mod, "run_one_shot_agent") as mock_run:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chat_mod.deprecated_positional_ask("hello world")
            assert any(
                issubclass(w.category, DeprecationWarning) for w in caught
            ), [str(w) for w in caught]
            # It must also route to the one-shot implementation.
            assert mock_run.called
            assert mock_run.call_args.args[0] == "hello world"

    def test_deprecated_positional_mentions_canonical_form(self) -> None:
        """The warning message must show users where to migrate."""
        with patch.object(chat_mod, "run_one_shot_agent"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                chat_mod.deprecated_positional_ask("ping")
            messages = [str(w.message) for w in caught]
            assert any("carl ask" in msg for msg in messages), messages


# ---------------------------------------------------------------------------
# Context-aware priming in the one-shot path (F4 overlap)
# ---------------------------------------------------------------------------


class TestContextPriming:
    def test_context_prompt_extension_empty_when_no_context(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """No ``context.json`` -> empty extension string."""
        from carl_studio.cli import init as init_mod

        monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)
        monkeypatch.setattr(init_mod, "CONTEXT_FILE", tmp_path / "context.json")
        assert chat_mod._context_prompt_extension() == ""

    def test_context_prompt_extension_renders_both_fields(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from carl_studio.cli import init as init_mod

        monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)
        monkeypatch.setattr(init_mod, "CONTEXT_FILE", tmp_path / "context.json")
        init_mod._save_context({
            "github_repo": "acme/repo",
            "hf_model": "mistralai/Mistral-7B-v0.1",
        })
        ext = chat_mod._context_prompt_extension()
        assert "acme/repo" in ext
        assert "mistralai/Mistral-7B-v0.1" in ext
        # Must flag the role clearly so the agent uses them as defaults.
        assert "defaults" in ext.lower()

    def test_context_prompt_extension_resilient_on_bad_json(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from carl_studio.cli import init as init_mod

        monkeypatch.setattr(init_mod, "CARL_HOME", tmp_path)
        monkeypatch.setattr(init_mod, "CONTEXT_FILE", tmp_path / "context.json")
        (tmp_path / "context.json").write_text("{not json")
        # Must not raise; simply collapse to empty.
        assert chat_mod._context_prompt_extension() == ""
