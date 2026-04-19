"""Tests for ``carl_studio.cli.intro`` — the env-baked intro helper.

Covers render_intro (message content + key coverage) and
parse_intro_selection (single-letter, full-word, free-form).
"""
from __future__ import annotations

from typing import Any

import pytest

from carl_studio.cli.intro import (
    INTRO_MOVES,
    INTRO_TITLE,
    parse_intro_selection,
    render_intro,
)


class _FakeConsole:
    """Minimal CampConsole stub — records method calls for assertions."""

    def __init__(self) -> None:
        self.rule_calls: list[str] = []
        self.info_calls: list[str] = []
        self.kv_calls: list[tuple[str, Any]] = []

    def rule(self, title: str = "") -> None:
        self.rule_calls.append(title)

    def info(self, message: str) -> None:
        self.info_calls.append(message)

    def kv(self, key: str, value: Any, key_width: int = 12) -> None:
        self.kv_calls.append((key, value))


class TestRenderIntro:
    def test_render_intro_lists_all_moves(self) -> None:
        """Every move in INTRO_MOVES must appear as a kv line."""
        c = _FakeConsole()
        render_intro(c)

        # Four moves -> four kv calls.
        assert len(c.kv_calls) == len(INTRO_MOVES) == 4

        # Each key is present in the rendered kv lines.
        rendered_keys = [k for k, _ in c.kv_calls]
        for key, label, _desc in INTRO_MOVES:
            assert any(f"[{key}]" in line for line in rendered_keys)
            assert any(label in line for line in rendered_keys)

    def test_render_intro_prints_title(self) -> None:
        c = _FakeConsole()
        render_intro(c)
        # Title must be emitted in one of the info() calls.
        assert any(INTRO_TITLE == msg for msg in c.info_calls)

    def test_render_intro_brackets_output_with_rules(self) -> None:
        """The greeting should be framed between two rule lines."""
        c = _FakeConsole()
        render_intro(c)
        # Exactly two rule() calls — one open, one close.
        assert len(c.rule_calls) == 2

    def test_render_intro_includes_free_form_hint(self) -> None:
        c = _FakeConsole()
        render_intro(c)
        joined = " ".join(c.info_calls)
        assert "anything" in joined.lower() or "free" in joined.lower()


class TestParseSelection:
    @pytest.mark.parametrize("raw", ["e", "t", "v", "s"])
    def test_parse_selection_single_letter(self, raw: str) -> None:
        assert parse_intro_selection(raw) == raw

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("explore", "e"),
            ("train", "t"),
            ("evaluate", "v"),
            ("sticky", "s"),
        ],
    )
    def test_parse_selection_full_word(self, raw: str, expected: str) -> None:
        assert parse_intro_selection(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            " ",
            "help",
            "train a gsm8k model",  # freeform despite containing a keyword
            "please explore my repo",
            "42",
            "quit",
        ],
    )
    def test_parse_selection_free_form_returns_none(self, raw: str) -> None:
        assert parse_intro_selection(raw) is None

    def test_parse_selection_case_insensitive(self) -> None:
        assert parse_intro_selection("EXPLORE") == "e"
        assert parse_intro_selection("  Train  ") == "t"
        assert parse_intro_selection("V") == "v"

    def test_parse_selection_strips_whitespace(self) -> None:
        assert parse_intro_selection("  s  ") == "s"
        assert parse_intro_selection("\tevaluate\n") == "v"


class TestRenderIntroRealConsole:
    """Drive render_intro through a real Rich console — pins UAT-052.

    UAT-052 regression: Rich interpreted ``[e]`` / ``[t]`` / ``[v]`` / ``[s]``
    as style tags and silently dropped them, so end-users saw unlabelled
    ``explore: …`` lines. The FakeConsole stub does not go through Rich,
    so the original tests missed it. This class asserts that the literal
    shortcut markers appear in the captured output via the real pipeline.
    """

    def test_shortcut_keys_visible_in_rendered_output(self) -> None:
        from io import StringIO
        from carl_studio.console import CampConsole

        c = CampConsole()
        buf = StringIO()
        # CampConsole wraps a rich.Console; redirect its file to a buffer.
        c._console.file = buf  # type: ignore[attr-defined]
        render_intro(c)
        output = buf.getvalue()
        for key, _label, _desc in INTRO_MOVES:
            marker = f"[{key}]"
            assert marker in output, (
                f"shortcut {marker!r} not visible in rendered intro — "
                "Rich markup may be consuming it. See UAT-052."
            )


class TestIntroConstants:
    def test_intro_moves_shape(self) -> None:
        """Each move tuple must be (key, label, description) — all non-empty."""
        for entry in INTRO_MOVES:
            assert len(entry) == 3
            key, label, desc = entry
            assert key and len(key) == 1
            assert label and label.isalpha()
            assert desc and len(desc) > 5

    def test_intro_moves_keys_unique(self) -> None:
        keys = [k for k, _, _ in INTRO_MOVES]
        assert len(keys) == len(set(keys))

    def test_intro_moves_labels_unique(self) -> None:
        labels = [label for _, label, _ in INTRO_MOVES]
        assert len(labels) == len(set(labels))
