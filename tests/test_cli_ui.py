"""Tests for carl_studio.cli.ui — the arrow-key + typer-fallback facade.

Coverage:

- ``Choice`` normalization from plain strings
- Arrow-key questionary path (mocked via ``_use_modern_ui``)
- ``typer.prompt`` fallback path (stdin/stdout non-TTY)
- Ctrl-C raises ``typer.Abort`` in both paths
- First-is-default convention
- Validation loop re-prompts on invalid input
- Password masking routed through ``text(secret=True)``
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer

from carl_studio.cli import ui


# ---------------------------------------------------------------------------
# Choice normalization
# ---------------------------------------------------------------------------


def test_select_normalizes_plain_strings() -> None:
    """Passing strings promotes them to Choice(value=s)."""
    with patch.object(ui, "_use_modern_ui", return_value=False):
        # stdin non-TTY → fallback. Simulate "press Enter for default".
        with patch("typer.prompt", return_value="1"):
            result = ui.select("pick", ["a", "b", "c"])
    assert result == "a"


# ---------------------------------------------------------------------------
# Fallback path (non-TTY)
# ---------------------------------------------------------------------------


def test_select_fallback_first_is_default() -> None:
    """Empty answer → default index (first option)."""
    with patch.object(ui, "_use_modern_ui", return_value=False):
        # typer.prompt with default="1" returns "1" when empty-Enter is pressed
        with patch("typer.prompt", return_value="1"):
            result = ui.select(
                "pick",
                [ui.Choice(value="apple"), ui.Choice(value="pear")],
                default=0,
            )
    assert result == "apple"


def test_select_fallback_second_option() -> None:
    """User types 2 → returns second option."""
    with patch.object(ui, "_use_modern_ui", return_value=False):
        with patch("typer.prompt", return_value="2"):
            result = ui.select("pick", ["a", "b", "c"])
    assert result == "b"


def test_select_fallback_rejects_out_of_range() -> None:
    """Invalid input re-prompts until valid."""
    with patch.object(ui, "_use_modern_ui", return_value=False):
        answers = iter(["9", "0", "foo", "2"])
        def _next(*_a: object, **_k: object) -> str:
            return next(answers)

        with patch("typer.prompt", side_effect=_next):
            result = ui.select("pick", ["a", "b", "c"])
    assert result == "b"


# ---------------------------------------------------------------------------
# questionary path (mocked)
# ---------------------------------------------------------------------------


def test_select_questionary_returns_value() -> None:
    """Modern UI path: questionary.select(...).ask() → answer."""
    fake_q = MagicMock()
    fake_q.select.return_value.ask.return_value = "apple"
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
    ):
        result = ui.select("pick", ["apple", "pear"])
    assert result == "apple"


def test_select_questionary_ctrl_c_raises_abort() -> None:
    """questionary returning None (Ctrl-C) → typer.Abort."""
    fake_q = MagicMock()
    fake_q.select.return_value.ask.return_value = None
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
        pytest.raises(typer.Abort),
    ):
        ui.select("pick", ["a"])


def test_select_questionary_passes_badge_and_hint() -> None:
    """Badge + hint get rendered into the display label."""
    fake_q = MagicMock()
    fake_q.select.return_value.ask.return_value = "a"
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
    ):
        ui.select(
            "pick",
            [ui.Choice(value="a", label="Apple", badge="recommended", hint="fastest")],
        )
    # The Choice constructor in questionary was called with title containing
    # our rendered label. Inspect last call.
    choice_call = fake_q.Choice.call_args
    assert choice_call is not None
    title = choice_call.kwargs.get("title", "") or (choice_call.args[0] if choice_call.args else "")
    assert "Apple" in title
    assert "recommended" in title
    assert "fastest" in title


# ---------------------------------------------------------------------------
# confirm
# ---------------------------------------------------------------------------


def test_confirm_fallback_routes_to_typer() -> None:
    with (
        patch.object(ui, "_use_modern_ui", return_value=False),
        patch("typer.confirm", return_value=True) as tc,
    ):
        assert ui.confirm("proceed?", default=True) is True
    tc.assert_called_once()
    assert tc.call_args.kwargs.get("default") is True


def test_confirm_questionary_returns_bool() -> None:
    fake_q = MagicMock()
    fake_q.confirm.return_value.ask.return_value = False
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
    ):
        assert ui.confirm("proceed?") is False


def test_confirm_questionary_ctrl_c_raises_abort() -> None:
    fake_q = MagicMock()
    fake_q.confirm.return_value.ask.return_value = None
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
        pytest.raises(typer.Abort),
    ):
        ui.confirm("proceed?")


# ---------------------------------------------------------------------------
# text / secret
# ---------------------------------------------------------------------------


def test_text_fallback_default() -> None:
    with (
        patch.object(ui, "_use_modern_ui", return_value=False),
        patch("typer.prompt", return_value="hello"),
    ):
        assert ui.text("say", default="hello") == "hello"


def test_text_questionary_secret_routes_password() -> None:
    fake_q = MagicMock()
    fake_q.password.return_value.ask.return_value = "hunter2"
    with (
        patch.object(ui, "_use_modern_ui", return_value=True),
        patch.object(ui, "_questionary", return_value=fake_q),
    ):
        assert ui.text("key", secret=True) == "hunter2"
    fake_q.password.assert_called_once()
    fake_q.text.assert_not_called()


def test_text_validate_loops_on_reject() -> None:
    """Fallback path: validate returning a string re-prompts until True."""
    answers = iter(["short", "ok-value"])

    def _validate(v: str) -> bool | str:
        return True if len(v) > 5 else "too short"

    def _next_text(*_a: object, **_k: object) -> str:
        return next(answers)

    with (
        patch.object(ui, "_use_modern_ui", return_value=False),
        patch("typer.prompt", side_effect=_next_text),
    ):
        assert ui.text("key", validate=_validate) == "ok-value"


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------


def test_path_fallback_empty_allowed_when_not_must_exist(tmp_path: object) -> None:
    del tmp_path
    with (
        patch.object(ui, "_use_modern_ui", return_value=False),
        patch("typer.prompt", return_value=""),
    ):
        assert ui.path("location") == ""


def test_path_must_exist_reprompts_on_missing(tmp_path: object) -> None:
    del tmp_path
    answers = iter(["/nonexistent-carl-test", "/tmp"])

    def _next_path(*_a: object, **_k: object) -> str:
        return next(answers)

    with (
        patch.object(ui, "_use_modern_ui", return_value=False),
        patch("typer.prompt", side_effect=_next_path),
    ):
        assert ui.path("location", must_exist=True) == "/tmp"
