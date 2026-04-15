"""Tests for carl_studio.consent -- ConsentState, ConsentManager."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import typer
from typer.testing import CliRunner

from carl_studio.consent import (
    CONSENT_KEYS,
    ConsentError,
    ConsentFlag,
    ConsentManager,
    ConsentState,
    consent_state_from_profile,
)
from carl_studio.cli.consent import consent_app


class FakeDB:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


runner = CliRunner()
_app = typer.Typer()
_app.add_typer(consent_app, name="consent")


class TestConsentFlag:
    def test_defaults_to_off(self) -> None:
        flag = ConsentFlag()
        assert flag.enabled is False
        assert flag.changed_at is None

    def test_enabled_with_timestamp(self) -> None:
        flag = ConsentFlag(enabled=True, changed_at="2026-04-15T00:00:00Z")
        assert flag.enabled is True
        assert flag.changed_at == "2026-04-15T00:00:00Z"


class TestConsentState:
    def test_all_off_by_default(self) -> None:
        state = ConsentState()
        for key in CONSENT_KEYS:
            assert getattr(state, key).enabled is False

    def test_model_dump_roundtrip(self) -> None:
        state = ConsentState(
            observability=ConsentFlag(enabled=True, changed_at="2026-01-01T00:00:00Z")
        )
        dumped = state.model_dump_json()
        restored = ConsentState.model_validate_json(dumped)
        assert restored.observability.enabled is True


class TestConsentManager:
    def test_load_empty_returns_defaults(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        state = mgr.load()
        assert state.observability.enabled is False
        assert state.telemetry.enabled is False

    def test_save_and_load_roundtrip(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        state = ConsentState(
            telemetry=ConsentFlag(enabled=True, changed_at="2026-04-15T00:00:00Z")
        )
        mgr.save(state)
        loaded = mgr.load()
        assert loaded.telemetry.enabled is True
        assert loaded.telemetry.changed_at == "2026-04-15T00:00:00Z"

    def test_update_sets_flag_and_timestamp(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        state = mgr.update("observability", True)
        assert state.observability.enabled is True
        assert state.observability.changed_at is not None

    def test_update_invalid_key_raises(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        try:
            mgr.update("nonexistent", True)
            assert False, "Should have raised"
        except ConsentError as exc:
            assert "nonexistent" in str(exc)

    def test_all_off_resets_everything(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("observability", True)
        state = mgr.all_off()
        assert state.observability.enabled is False
        assert state.observability.changed_at is not None  # tracks the reset

    def test_sync_with_profile_local_wins(self) -> None:
        """Local consent state is authoritative over remote profile."""
        db = FakeDB()
        mgr = ConsentManager(db=db)
        # User explicitly disabled observability
        mgr.update("observability", False)

        class FakeProfile:
            observability_opt_in = True
            telemetry_opt_in = False
            usage_tracking_enabled = False
            contract_witnessing = False

        state = mgr.sync_with_profile(FakeProfile())
        # Local explicit off should win over remote True
        assert state.observability.enabled is False

    def test_sync_with_profile_adopts_remote_when_never_set(self) -> None:
        """If user never set a flag, adopt remote value."""
        db = FakeDB()
        mgr = ConsentManager(db=db)

        class FakeProfile:
            observability_opt_in = True
            telemetry_opt_in = False
            usage_tracking_enabled = False
            contract_witnessing = True

        state = mgr.sync_with_profile(FakeProfile())
        assert state.observability.enabled is True
        assert state.contract_witnessing.enabled is True
        assert state.telemetry.enabled is False


class TestConsentFromProfile:
    def test_projection_maps_flags(self) -> None:
        class FakeProfile:
            observability_opt_in = True
            telemetry_opt_in = True
            usage_tracking_enabled = False
            contract_witnessing = True

        state = consent_state_from_profile(FakeProfile())
        assert state.observability.enabled is True
        assert state.telemetry.enabled is True
        assert state.usage_analytics.enabled is False
        assert state.contract_witnessing.enabled is True


class TestConsentShowCLI:
    def test_show_json(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("telemetry", True)

        with patch("carl_studio.cli.consent.ConsentManager", return_value=mgr):
            result = runner.invoke(_app, ["consent", "show", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["telemetry"]["enabled"] is True

    def test_show_text(self) -> None:
        db = FakeDB()
        with patch("carl_studio.cli.consent.ConsentManager", return_value=ConsentManager(db=db)):
            result = runner.invoke(_app, ["consent", "show"])
        assert result.exit_code == 0
        assert "off" in result.output


class TestConsentUpdateCLI:
    def test_enable(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        with patch("carl_studio.cli.consent.ConsentManager", return_value=mgr):
            result = runner.invoke(_app, ["consent", "update", "observability", "--enable"])
        assert result.exit_code == 0
        assert mgr.load().observability.enabled is True

    def test_invalid_key(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        with patch("carl_studio.cli.consent.ConsentManager", return_value=mgr):
            result = runner.invoke(_app, ["consent", "update", "bogus", "--enable"])
        assert result.exit_code == 1


class TestConsentResetCLI:
    def test_resets_with_force(self) -> None:
        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("telemetry", True)
        with patch("carl_studio.cli.consent.ConsentManager", return_value=mgr):
            result = runner.invoke(_app, ["consent", "reset", "--force"])
        assert result.exit_code == 0
        assert mgr.load().telemetry.enabled is False
