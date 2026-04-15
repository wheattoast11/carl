"""Tests for carl_studio.infer_cli and carl_studio.curriculum_cli."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from carl_studio.curriculum import CurriculumPhase, CurriculumStore, CurriculumTrack

# Build a minimal Typer app that includes infer + curriculum commands
import typer

_test_app = typer.Typer()


def _register_test_app() -> typer.Typer:
    """Register infer and curriculum commands on a test app."""
    from carl_studio.infer_cli import infer_cmd
    from carl_studio.curriculum_cli import curriculum_app

    _test_app.command("infer")(infer_cmd)
    _test_app.add_typer(curriculum_app, name="curriculum")
    return _test_app


_app = _register_test_app()
runner = CliRunner()


# ---------------------------------------------------------------------------
# Infer CLI
# ---------------------------------------------------------------------------


class TestInferCmd:
    def test_no_model_shows_error(self) -> None:
        """Running infer with no model and no carl.yaml should error."""
        with patch("carl_studio.infer_cli._resolve_model_adapter", return_value=("", "")):
            result = runner.invoke(_app, ["infer", "--model", ""])
            # Should fail because no model could be resolved
            # (default_model may resolve from settings, so we patch the resolver)
            assert result.exit_code != 0 or "No model specified" in result.output

    def test_invalid_ttt_mode_shows_error(self) -> None:
        result = runner.invoke(_app, ["infer", "--model", "test/model", "--ttt", "invalid"])
        assert result.exit_code == 1
        assert "Invalid --ttt mode" in result.output

    def test_ttt_slot_without_admin_shows_message(self) -> None:
        """TTT SLOT without admin or terminals-runtime should show guidance."""
        from carl_studio.infer_cli import _check_ttt_availability

        # Tier gate passes (PAID), but admin is False and terminals_runtime absent
        with patch("carl_studio.tier.check_tier", return_value=(True, MagicMock(), MagicMock())), \
             patch("carl_studio.admin.is_admin", return_value=False), \
             patch.dict(sys.modules, {"terminals_runtime": None}):
            available, msg = _check_ttt_availability("slot")
            assert not available
            assert "terminals-runtime" in msg or "admin" in msg

    def test_ttt_none_always_available(self) -> None:
        from carl_studio.infer_cli import _check_ttt_availability

        available, msg = _check_ttt_availability("none")
        assert available is True
        assert msg == ""

    def test_resolve_model_adapter_from_args(self) -> None:
        from carl_studio.infer_cli import _resolve_model_adapter

        model, adapter = _resolve_model_adapter("test/model", "test/adapter")
        assert model == "test/model"
        assert adapter == "test/adapter"


# ---------------------------------------------------------------------------
# Curriculum CLI
# ---------------------------------------------------------------------------


class TestCurriculumCLI:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> CurriculumStore:
        db = tmp_path / "test_carl.db"
        s = CurriculumStore(db_path=db)
        yield s
        s.close()

    def test_curriculum_show_no_tracks(self, tmp_path: Path) -> None:
        """Show with empty DB should indicate no carlitos enrolled."""
        db = tmp_path / "test.db"
        with patch("carl_studio.curriculum_cli._get_store", return_value=CurriculumStore(db_path=db)):
            result = runner.invoke(_app, ["curriculum", "show"])
            assert result.exit_code == 0
            assert "No carlitos enrolled" in result.output

    def test_curriculum_enroll(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(_app, ["curriculum", "enroll", "test-model"])
            assert result.exit_code == 0
            assert "Enrolled" in result.output

            # Verify persisted
            track = store.load("test-model")
            assert track is not None
            assert track.phase == CurriculumPhase.ENROLLED

    def test_curriculum_advance_valid(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        store.save(CurriculumTrack(model_id="test-model"))

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(
                _app, ["curriculum", "advance", "test-model", "drilling", "--event", "training_started"]
            )
            assert result.exit_code == 0
            assert "drilling" in result.output

            track = store.load("test-model")
            assert track is not None
            assert track.phase == CurriculumPhase.DRILLING

    def test_curriculum_advance_invalid(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        store.save(CurriculumTrack(model_id="test-model"))

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(
                _app, ["curriculum", "advance", "test-model", "graduated"]
            )
            assert result.exit_code == 1
            assert "Cannot transition" in result.output

    def test_curriculum_advance_nonexistent_model(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(
                _app, ["curriculum", "advance", "ghost", "drilling"]
            )
            assert result.exit_code == 1
            assert "No carlito found" in result.output

    def test_curriculum_list_empty(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(_app, ["curriculum", "list"])
            assert result.exit_code == 0
            assert "No carlitos enrolled" in result.output

    def test_curriculum_list_with_tracks(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        store.save(CurriculumTrack(model_id="model-a"))
        store.save(CurriculumTrack(model_id="model-b"))

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(_app, ["curriculum", "list"])
            assert result.exit_code == 0
            assert "model-a" in result.output
            assert "model-b" in result.output

    def test_curriculum_show_specific_model(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        track = CurriculumTrack(model_id="specific-model")
        track = track.advance(CurriculumPhase.DRILLING, event="started")
        store.save(track)

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(_app, ["curriculum", "show", "specific-model"])
            assert result.exit_code == 0
            assert "specific-model" in result.output
            assert "drilling" in result.output

    def test_curriculum_advance_invalid_phase_string(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        store.save(CurriculumTrack(model_id="test-model"))

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(
                _app, ["curriculum", "advance", "test-model", "nonexistent_phase"]
            )
            assert result.exit_code == 1
            assert "Invalid phase" in result.output

    def test_curriculum_enroll_duplicate(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        store = CurriculumStore(db_path=db)
        store.save(CurriculumTrack(model_id="dup"))

        with patch("carl_studio.curriculum_cli._get_store", return_value=store):
            result = runner.invoke(_app, ["curriculum", "enroll", "dup"])
            assert result.exit_code == 0
            assert "already enrolled" in result.output
