"""Tests for ``carl_studio.cli.operations`` — SIMP-006 + SIMP-010.

Covers:
- ``parse_flags`` shared flag parser (``--flag value`` / ``--flag=value`` /
  boolean flags / aliases / missing-value errors / mixed positional).
- Prompt-template macros (``ask`` / ``chat`` / ``train`` / ``review`` /
  ``simplify`` / ``ship``) produced by ``_prompt_op``.
"""
from __future__ import annotations

from typing import Any

import pytest

from carl_core.interaction import InteractionChain

from carl_studio.cli import operations as ops_mod


# ---------------------------------------------------------------------------
# parse_flags — SIMP-006
# ---------------------------------------------------------------------------


class TestParseFlagsEqualsSyntax:
    def test_string_flag(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["--mode=fast"],
            {"mode": (str, "default")},
        )
        assert flags["mode"] == "fast"
        assert rest == []

    def test_int_flag(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--rank=128"],
            {"rank": (int, 64)},
        )
        assert flags["rank"] == 128
        assert isinstance(flags["rank"], int)

    def test_float_flag(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--threshold=0.75"],
            {"threshold": (float, 0.3)},
        )
        assert flags["threshold"] == pytest.approx(0.75)


class TestParseFlagsSpaceSyntax:
    def test_string_flag(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["--mode", "fast"],
            {"mode": (str, "default")},
        )
        assert flags["mode"] == "fast"
        assert rest == []

    def test_int_flag(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--rank", "128"],
            {"rank": (int, 64)},
        )
        assert flags["rank"] == 128

    def test_short_alias_resolves_to_canonical(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["-m", "fast"],
            {"mode": (str, "default")},
            aliases={"m": "mode"},
        )
        assert flags["mode"] == "fast"


class TestParseFlagsBoolFlag:
    def test_bool_flag_presence_sets_true(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--quick"],
            {"quick": (bool, False)},
        )
        assert flags["quick"] is True

    def test_bool_flag_absence_keeps_default(self) -> None:
        flags, _ = ops_mod.parse_flags(
            [],
            {"quick": (bool, False)},
        )
        assert flags["quick"] is False

    def test_bool_equals_false(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--quick=false"],
            {"quick": (bool, True)},
        )
        assert flags["quick"] is False

    def test_bool_equals_true(self) -> None:
        flags, _ = ops_mod.parse_flags(
            ["--quick=yes"],
            {"quick": (bool, False)},
        )
        assert flags["quick"] is True


class TestParseFlagsMixedAndPositional:
    def test_positional_before_flag(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["org/model-a", "--mode", "fast"],
            {"mode": (str, "default")},
        )
        assert flags["mode"] == "fast"
        assert rest == ["org/model-a"]

    def test_positional_between_flags(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["--mode", "fast", "foo", "--quick", "bar"],
            {"mode": (str, "default"), "quick": (bool, False)},
        )
        assert flags["mode"] == "fast"
        assert flags["quick"] is True
        assert rest == ["foo", "bar"]

    def test_unknown_flag_passes_through_as_positional(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["--unknown", "value"],
            {"mode": (str, "default")},
        )
        assert flags["mode"] == "default"
        # Unknown flag is left verbatim; value becomes trailing positional.
        assert "--unknown" in rest

    def test_multiple_flags_parsed(self) -> None:
        flags, rest = ops_mod.parse_flags(
            ["--adapter", "org/ckpt", "--phase", "2prime", "extra-pos"],
            {
                "adapter": (str, ""),
                "phase": (str, "auto"),
                "threshold": (float, 0.3),
            },
        )
        assert flags["adapter"] == "org/ckpt"
        assert flags["phase"] == "2prime"
        assert flags["threshold"] == pytest.approx(0.3)
        assert rest == ["extra-pos"]


class TestParseFlagsMissingValueRaises:
    def test_non_bool_flag_without_value(self) -> None:
        with pytest.raises(ValueError, match="requires a value"):
            ops_mod.parse_flags(
                ["--mode"],
                {"mode": (str, "default")},
            )


# ---------------------------------------------------------------------------
# Prompt-template macros — SIMP-010
# ---------------------------------------------------------------------------


class TestPromptOpTemplates:
    """Assert that positional args format into each op's template."""

    @pytest.fixture(autouse=True)
    def _capture_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.captured_prompts: list[str] = []

        def _fake_run(prompt: str) -> None:
            self.captured_prompts.append(prompt)

        # Patch at the import path the prompt ops use inside their ``_do``.
        monkeypatch.setattr("carl_studio.cli.chat.run_one_shot_agent", _fake_run)

    def test_ask_templates_args(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["ask"](chain, ["what", "is", "coherence"])
        assert self.captured_prompts == ["what is coherence"]

    def test_train_templates_args(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["train"](chain, ["a", "small", "model"])
        assert "a small model" in self.captured_prompts[0]
        assert "training run" in self.captured_prompts[0].lower()

    def test_review_templates_target(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["review"](chain, ["src/foo.py"])
        assert "src/foo.py" in self.captured_prompts[0]
        assert "Review" in self.captured_prompts[0]

    def test_simplify_templates_target(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["simplify"](chain, ["src/bar.py"])
        assert "src/bar.py" in self.captured_prompts[0]
        assert "Simplify" in self.captured_prompts[0]

    def test_ship_without_args_does_not_include_extra(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["ship"](chain, [])
        assert "Extra instructions" not in self.captured_prompts[0]

    def test_ship_with_args_appends_extra(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["ship"](chain, ["no-push"])
        assert "Extra instructions: no-push" in self.captured_prompts[0]

    def test_review_fallback_used_when_no_args(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["review"](chain, [])
        assert "the most recent change" in self.captured_prompts[0]

    def test_ask_fallback_used_when_no_args(self) -> None:
        chain = InteractionChain()
        ops_mod.OPERATIONS["ask"](chain, [])
        assert self.captured_prompts == ["hello"]


class TestPromptOpCallsRunOneShotAgent:
    """Explicitly assert the dispatch target and argument passing."""

    def test_calls_run_one_shot_agent_with_exact_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, ...]] = []

        def _fake(prompt: str) -> None:
            calls.append((prompt,))

        monkeypatch.setattr("carl_studio.cli.chat.run_one_shot_agent", _fake)

        chain = InteractionChain()
        ops_mod.OPERATIONS["ask"](chain, ["exact", "payload"])
        assert calls == [("exact payload",)]
        step = chain.last()
        assert step is not None
        assert step.success is True
        assert step.name == "ask"


class TestPromptOpChatFallbackToRepl:
    """``chat`` with no args opens the REPL, not the one-shot agent."""

    def test_chat_no_args_calls_chat_cmd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: dict[str, bool] = {"chat_cmd": False, "one_shot": False}

        def _fake_chat_cmd(*a: Any, **k: Any) -> None:
            called["chat_cmd"] = True

        def _fake_one_shot(prompt: str) -> None:
            called["one_shot"] = True

        monkeypatch.setattr("carl_studio.cli.chat.chat_cmd", _fake_chat_cmd)
        monkeypatch.setattr("carl_studio.cli.chat.run_one_shot_agent", _fake_one_shot)

        chain = InteractionChain()
        ops_mod.OPERATIONS["chat"](chain, [])
        assert called["chat_cmd"] is True
        assert called["one_shot"] is False

    def test_chat_with_args_calls_one_shot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: dict[str, bool] = {"chat_cmd": False, "one_shot": False}
        captured: list[str] = []

        def _fake_chat_cmd(*a: Any, **k: Any) -> None:
            called["chat_cmd"] = True

        def _fake_one_shot(prompt: str) -> None:
            called["one_shot"] = True
            captured.append(prompt)

        monkeypatch.setattr("carl_studio.cli.chat.chat_cmd", _fake_chat_cmd)
        monkeypatch.setattr("carl_studio.cli.chat.run_one_shot_agent", _fake_one_shot)

        chain = InteractionChain()
        ops_mod.OPERATIONS["chat"](chain, ["hello"])
        assert called["chat_cmd"] is False
        assert called["one_shot"] is True
        assert "hello" in captured[0]


# ---------------------------------------------------------------------------
# Descriptions — JRN-007
# ---------------------------------------------------------------------------


class TestOperationDescriptions:
    def test_every_op_has_description(self) -> None:
        for name in ops_mod.list_operations():
            desc = ops_mod.get_description(name)
            assert desc, f"op /{name} is missing a description"

    def test_get_description_unknown_returns_empty_string(self) -> None:
        assert ops_mod.get_description("does-not-exist") == ""

    def test_descriptions_are_one_liners(self) -> None:
        for name in ops_mod.list_operations():
            desc = ops_mod.get_description(name)
            assert "\n" not in desc, f"/{name} description spans multiple lines"
            assert len(desc) <= 120, f"/{name} description is too long ({len(desc)} chars)"
