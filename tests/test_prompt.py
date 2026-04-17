"""Tests for cli/prompt.py — the in-flow require() helper."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import typer

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.cli.prompt import KNOWN_KEYS, RequireSpec, known_keys, require


class TestKnownKeys:
    def test_returns_copy(self) -> None:
        keys = known_keys()
        assert "ANTHROPIC_API_KEY" in keys
        assert "HF_TOKEN" in keys
        assert "OPENROUTER_API_KEY" in keys
        # Mutating the returned dict should not affect the registry
        keys["NEW"] = RequireSpec("NEW", env_var="NEW", signup_url="")
        assert "NEW" not in KNOWN_KEYS

    def test_anthropic_spec_points_to_console(self) -> None:
        spec = KNOWN_KEYS["ANTHROPIC_API_KEY"]
        assert "console.anthropic.com" in spec.signup_url


class TestRequireEnvHit:
    def test_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        result = require("ANTHROPIC_API_KEY")
        assert result == "sk-from-env"

    def test_env_hit_records_gate_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        chain = InteractionChain()
        require("ANTHROPIC_API_KEY", chain=chain)
        assert len(chain) == 1
        step = chain.last()
        assert step is not None
        assert step.action == ActionType.GATE
        assert step.name == "require:ANTHROPIC_API_KEY"
        assert step.input == {"resolved_via": "env"}
        assert step.success is True


class TestRequireStoredHit:
    def test_settings_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        class FakeSettings:
            anthropic_api_key = "sk-from-settings"

        with patch("carl_studio.settings.CARLSettings", return_value=FakeSettings()):
            result = require("ANTHROPIC_API_KEY")
        assert result == "sk-from-settings"


class TestRequirePromptFallback:
    def test_inline_prompt_saves_and_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with (
            patch("carl_studio.settings.CARLSettings", side_effect=Exception("no settings")),
            patch("carl_studio.db.LocalDB", side_effect=Exception("no db")),
        ):
            result = require(
                "ANTHROPIC_API_KEY",
                prompt_fn=lambda *a, **kw: "sk-pasted",
                confirm_fn=lambda *a, **kw: False,
            )
        assert result == "sk-pasted"

    def test_empty_prompt_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with (
            patch("carl_studio.settings.CARLSettings", side_effect=Exception),
            patch("carl_studio.db.LocalDB", side_effect=Exception),
            pytest.raises(typer.Abort),
        ):
            require(
                "ANTHROPIC_API_KEY",
                prompt_fn=lambda *a, **kw: "",
                confirm_fn=lambda *a, **kw: False,
            )

    def test_prompt_records_success_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        chain = InteractionChain()
        with (
            patch("carl_studio.settings.CARLSettings", side_effect=Exception),
            patch("carl_studio.db.LocalDB", side_effect=Exception),
        ):
            require(
                "ANTHROPIC_API_KEY",
                chain=chain,
                prompt_fn=lambda *a, **kw: "sk-pasted",
                confirm_fn=lambda *a, **kw: False,
            )
        step = chain.last()
        assert step is not None
        assert step.input == {"resolved_via": "prompt"}
        assert step.success is True

    def test_abort_records_failure_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        chain = InteractionChain()
        with (
            patch("carl_studio.settings.CARLSettings", side_effect=Exception),
            patch("carl_studio.db.LocalDB", side_effect=Exception),
            pytest.raises(typer.Abort),
        ):
            require(
                "ANTHROPIC_API_KEY",
                chain=chain,
                prompt_fn=lambda *a, **kw: "",
                confirm_fn=lambda *a, **kw: False,
            )
        step = chain.last()
        assert step is not None
        assert step.success is False
        assert step.input == {"resolved_via": "aborted"}


class TestAdHocSpec:
    def test_ad_hoc_key_uses_provided_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUSTOM_XYZ", "hello")
        result = require("CUSTOM_XYZ", env_var="CUSTOM_XYZ", signup_url="https://example.com")
        assert result == "hello"
