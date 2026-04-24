"""Tests for the v0.18 Track A unified entry-point router.

The router — ``carl_studio.cli.entry.route`` — owns the decision ladder
that picks between the three entry modes of ``carl``:

1. bare ``carl``         → interactive REPL
2. ``carl "<prompt>"``   → REPL with pre-submitted first turn
3. ``carl -p "<prompt>"`` → one-shot ``ask`` path
4. ``carl <verb>``       → fall through to Typer dispatch
5. first-run on TTY      → wizard, then clean exit (no chain into chat)
6. empty argv on non-TTY → help (piped script path)
7. trusted project gate  → bare interactive entry requires trust pre-check

Every test stubs all handlers so neither the real agent nor the
real onboarding wizard fire; the router's contract is "which handler
gets called, and is the invocation claimed?" not "does the handler do
the right thing".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from carl_studio.cli import entry as entry_mod
from carl_studio.cli.apps import app


runner = CliRunner()


def _wired() -> None:
    import carl_studio.cli.wiring as _wiring  # noqa: F401

    assert _wiring is not None


class _HandlerRecorder:
    """Record invocations so tests can assert which branch fired."""

    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.ask_calls: list[str] = []
        self.init_calls = 0
        self.help_calls = 0
        self.trust_calls = 0
        self.trust_result = True

    def chat(self, *, initial_message: str | None = None) -> None:
        self.chat_calls.append({"initial_message": initial_message})

    def ask(self, prompt: str) -> None:
        self.ask_calls.append(prompt)

    def init(self) -> None:
        self.init_calls += 1

    def help(self) -> None:
        self.help_calls += 1

    def trust_precheck(self) -> bool:
        self.trust_calls += 1
        return self.trust_result

    def route_kwargs(self) -> dict[str, Any]:
        return {
            "chat": self.chat,
            "ask": self.ask,
            "init": self.init,
            "show_help": self.help,
            "trust_precheck": self.trust_precheck,
        }

    def inject(self) -> tuple[Any, Any, Any, Any, Any]:
        return self.chat, self.ask, self.init, self.help, self.trust_precheck


@pytest.fixture
def recorder() -> _HandlerRecorder:
    return _HandlerRecorder()


@pytest.fixture
def initialized_marker(tmp_path: Path) -> Path:
    marker = tmp_path / ".initialized"
    marker.touch()
    return marker


@pytest.fixture
def missing_marker(tmp_path: Path) -> Path:
    return tmp_path / ".initialized"


def _run(
    argv: list[str],
    rec: _HandlerRecorder,
    *,
    marker: Path,
    tty: bool = True,
) -> bool:
    return entry_mod.route(
        argv,
        **rec.route_kwargs(),
        first_run_marker=marker,
        stdin_is_tty=lambda: tty,
    )


class TestHelpAndVersionFallThrough:
    def test_help_flag_falls_through(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["--help"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0
        assert recorder.chat_calls == []
        assert recorder.ask_calls == []
        assert recorder.init_calls == 0
        assert recorder.help_calls == 0

    def test_short_help_flag_falls_through(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["-h"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_version_flag_falls_through(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["--version"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_help_flag_wins_over_first_run_gate(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run(["--help"], recorder, marker=missing_marker)
        assert handled is False
        assert recorder.init_calls == 0
        assert recorder.trust_calls == 0


class TestFirstRunGate:
    def test_missing_marker_on_tty_runs_init_without_trust(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run([], recorder, marker=missing_marker, tty=True)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 1
        assert recorder.chat_calls == []
        assert recorder.ask_calls == []
        assert recorder.help_calls == 0

    def test_first_run_tty_print_path_skips_trust(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run(["-p", "hello"], recorder, marker=missing_marker, tty=True)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 1
        assert recorder.ask_calls == []

    def test_first_run_tty_help_path_skips_trust(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run(["--help"], recorder, marker=missing_marker, tty=True)
        assert handled is False
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 0

    def test_first_run_tty_positional_prompt_still_runs_init_before_trust(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run(["hello world"], recorder, marker=missing_marker, tty=True)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 1
        assert recorder.chat_calls == []

    def test_missing_marker_non_tty_skips_init_and_trust(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run([], recorder, marker=missing_marker, tty=False)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 0
        assert recorder.help_calls == 1

    def test_first_run_gate_fires_before_print_flag(
        self, recorder: _HandlerRecorder, missing_marker: Path
    ) -> None:
        handled = _run(["-p", "hello"], recorder, marker=missing_marker, tty=True)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 1
        assert recorder.ask_calls == []


class TestPrintFlagRouting:
    def test_short_print_flag_routes_to_ask_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["-p", "train a model"], recorder, marker=initialized_marker)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.ask_calls == ["train a model"]

    def test_long_print_flag_routes_to_ask_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["--print", "what is CARL?"], recorder, marker=initialized_marker)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.ask_calls == ["what is CARL?"]

    def test_equals_form_print_flag(self, recorder: _HandlerRecorder, initialized_marker: Path) -> None:
        handled = _run(["--print=inline prompt"], recorder, marker=initialized_marker)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.ask_calls == ["inline prompt"]

    def test_empty_print_falls_through(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["-p", ""], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0
        assert recorder.ask_calls == []


class TestBareInvocation:
    def test_empty_argv_on_tty_opens_chat_after_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run([], recorder, marker=initialized_marker, tty=True)
        assert handled is True
        assert recorder.trust_calls == 1
        assert recorder.chat_calls == [{"initial_message": None}]
        assert recorder.help_calls == 0

    def test_empty_argv_non_tty_prints_help_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run([], recorder, marker=initialized_marker, tty=False)
        assert handled is True
        assert recorder.trust_calls == 0
        assert recorder.help_calls == 1
        assert recorder.chat_calls == []

    def test_trust_decline_stops_bare_entry(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        recorder.trust_result = False
        with pytest.raises(typer.Exit) as exc_info:
            _run([], recorder, marker=initialized_marker, tty=True)
        assert exc_info.value.exit_code == 1
        assert recorder.trust_calls == 1
        assert recorder.chat_calls == []
        assert recorder.ask_calls == []
        assert recorder.init_calls == 0
        assert recorder.help_calls == 0


class TestPositionalPrompt:
    def test_positional_prompt_opens_chat_with_initial_message_after_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["hello world"], recorder, marker=initialized_marker)
        assert handled is True
        assert recorder.trust_calls == 1
        assert recorder.chat_calls == [{"initial_message": "hello world"}]
        assert recorder.ask_calls == []

    def test_positional_prompt_non_tty_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["hello"], recorder, marker=initialized_marker, tty=False)
        assert handled is False
        assert recorder.trust_calls == 0
        assert recorder.chat_calls == []

    def test_trust_decline_stops_positional_prompt(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        recorder.trust_result = False
        with pytest.raises(typer.Exit) as exc_info:
            _run(["hello world"], recorder, marker=initialized_marker, tty=True)
        assert exc_info.value.exit_code == 1
        assert recorder.trust_calls == 1
        assert recorder.chat_calls == []
        assert recorder.ask_calls == []

    def test_registered_subcommand_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["chat"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_flag_starting_token_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["--foo"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_numeric_token_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["12345"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_path_like_token_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["./thing.yaml"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_multi_token_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["unknown_verb", "foo"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0


class TestSubcommandFallThrough:
    def test_doctor_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["doctor"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_doctor_with_args_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["doctor", "--verbose"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0

    def test_init_subcommand_falls_through_without_trust(
        self, recorder: _HandlerRecorder, initialized_marker: Path
    ) -> None:
        handled = _run(["init"], recorder, marker=initialized_marker)
        assert handled is False
        assert recorder.trust_calls == 0
        assert recorder.init_calls == 0


class TestRegistrySync:
    def test_registered_subcommands_match_wiring(self) -> None:
        _wired()

        live: set[str] = set()
        for c in app.registered_commands:
            if isinstance(c.name, str):
                live.add(c.name)
        for g in app.registered_groups:
            name = g.typer_instance.info.name if g.typer_instance else None
            if isinstance(name, str):
                live.add(name)
            if isinstance(g.name, str):
                live.add(g.name)

        missing = live - entry_mod.REGISTERED_SUBCOMMANDS
        assert missing == set(), (
            f"REGISTERED_SUBCOMMANDS is missing live verbs: {sorted(missing)}. "
            "Update carl_studio/cli/entry.py::REGISTERED_SUBCOMMANDS."
        )


class TestChatCmdInitialMessageKwarg:
    def test_chat_cmd_initial_message_is_not_a_typer_flag(self) -> None:
        _wired()
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--initial-message" not in result.output
        assert "--initial_message" not in result.output
        assert "--session" in result.output


class TestPublicSurface:
    def test_route_exposes_canonical_public_api(self) -> None:
        assert callable(entry_mod.route)
        assert callable(entry_mod.route_from_sys_argv)
        assert isinstance(entry_mod.REGISTERED_SUBCOMMANDS, frozenset)
        assert "trust" in entry_mod.REGISTERED_SUBCOMMANDS
        assert "trust_precheck" in entry_mod.route.__code__.co_varnames

    def test_route_from_sys_argv_delegates(
        self,
        recorder: _HandlerRecorder,
        initialized_marker: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys as real_sys

        monkeypatch.setattr(real_sys, "argv", ["carl", "hello world"])
        monkeypatch.setattr(entry_mod, "_default_first_run_marker", lambda: initialized_marker)
        monkeypatch.setattr(entry_mod, "_default_stdin_is_tty", lambda: True)
        monkeypatch.setattr(entry_mod, "_resolve_default_handlers", recorder.inject)

        handled = entry_mod.route_from_sys_argv()
        assert handled is True
        assert recorder.trust_calls == 1
        assert recorder.chat_calls == [{"initial_message": "hello world"}]
