"""Regression tests for ``carl push`` — the first-class HF ship verb."""
from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from carl_studio.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_db_stub(row: dict[str, Any] | None):
    """Return a Mock class that LocalDB() -> instance with get_run -> row."""
    instance = Mock()
    instance.get_run.return_value = row

    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_run(self, run_id: str) -> dict[str, Any] | None:
            return row

    return _Stub


def _install_hf_stub(monkeypatch: pytest.MonkeyPatch, *, api_mock: Any):
    """Make `from huggingface_hub import HfApi, get_token` succeed.

    The stub module is injected via ``sys.modules`` and via a targeted
    ``builtins.__import__`` override so the top-level
    ``import huggingface_hub`` (if any) plus the ``from
    huggingface_hub import HfApi, get_token`` inside ``push_cmd`` both
    resolve to the fake.
    """
    import sys
    import types

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = Mock(return_value=api_mock)
    fake.get_token = Mock(return_value="hf_test_token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


# ---------------------------------------------------------------------------
# Missing HF extra
# ---------------------------------------------------------------------------


def test_push_missing_hf_extra_errors_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``huggingface_hub`` is absent, the command should exit 1 with a
    clear install hint, not crash with a bare ImportError traceback."""
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()

    # Force the `from huggingface_hub import HfApi, get_token` inside
    # push_cmd to raise ImportError. We patch builtins.__import__ rather
    # than sys.modules because the caller uses a bare `from ... import`.
    real_import = builtins.__import__

    def _raising_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == "huggingface_hub":
            raise ImportError("No module named 'huggingface_hub'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    result = runner.invoke(app, ["push", str(ckpt), "org/model"])

    assert result.exit_code == 1, result.output
    assert "hf" in result.output.lower() or "huggingface" in result.output.lower()


# ---------------------------------------------------------------------------
# Invalid run id / missing path
# ---------------------------------------------------------------------------


def test_push_invalid_run_id_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown run id (no LocalDB row, no literal path) must exit 2 with
    an actionable diagnostic."""
    # Force LocalDB().get_run() to return None.
    stub_cls = _make_local_db_stub(None)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    result = runner.invoke(app, ["push", "does-not-exist-run-id", "org/model"])

    assert result.exit_code == 2, result.output
    assert "no checkpoint found" in result.output


def test_push_invalid_repo_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Repo ids without a ``/`` are rejected before any network call."""
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()

    result = runner.invoke(app, ["push", str(ckpt), "missing-slash"])
    assert result.exit_code == 2, result.output
    assert "invalid repo id" in result.output.lower()


def test_push_missing_repo_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no positional repo, no --repo, and no output_repo on the run
    record, the command must bail with a clear message."""
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()

    # No LocalDB row — but the literal path resolves so we get far enough
    # to hit the repo-resolution gate.
    stub_cls = _make_local_db_stub(None)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    result = runner.invoke(app, ["push", str(ckpt)])
    assert result.exit_code == 2, result.output
    assert "no target repo resolved" in result.output


# ---------------------------------------------------------------------------
# Run id → DB resolution
# ---------------------------------------------------------------------------


def test_push_resolves_run_id_via_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When given a run id, push should look up the local DB row, read
    ``result.output_dir``, and upload that folder via HfApi."""
    ckpt = tmp_path / "carl-grpo-abc"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_text("weights")

    run_row = {
        "id": "abc",
        "model_id": "org/base",
        "mode": "train:grpo",
        "status": "complete",
        "config": {"output_repo": "org/model"},
        "result": {"output_dir": str(ckpt), "phase": "complete"},
    }
    stub_cls = _make_local_db_stub(run_row)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    api_mock = Mock()
    api_mock.create_repo = Mock()
    api_mock.upload_folder = Mock()
    _install_hf_stub(monkeypatch, api_mock=api_mock)

    result = runner.invoke(app, ["push", "abc", "org/model"])

    assert result.exit_code == 0, result.output
    api_mock.create_repo.assert_called_once()
    api_mock.upload_folder.assert_called_once()
    call_kwargs = api_mock.upload_folder.call_args.kwargs
    assert call_kwargs["folder_path"] == str(ckpt)
    assert call_kwargs["repo_id"] == "org/model"
    assert "pushed org/model" in result.output


def test_push_resolves_repo_from_run_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Omitting the positional repo should fall back to config.output_repo."""
    ckpt = tmp_path / "carl-sft-run-xyz"
    ckpt.mkdir()

    run_row = {
        "id": "run-xyz",
        "model_id": "org/base",
        "config": {"output_repo": "derived-org/derived-model"},
        "result": {"output_dir": str(ckpt)},
    }
    stub_cls = _make_local_db_stub(run_row)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    api_mock = Mock()
    api_mock.create_repo = Mock()
    api_mock.upload_folder = Mock()
    _install_hf_stub(monkeypatch, api_mock=api_mock)

    result = runner.invoke(app, ["push", "run-xyz"])
    assert result.exit_code == 0, result.output
    assert "pushed derived-org/derived-model" in result.output


# ---------------------------------------------------------------------------
# Literal path
# ---------------------------------------------------------------------------


def test_push_accepts_literal_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A literal checkpoint path works even when the LocalDB row is absent."""
    ckpt = tmp_path / "literal-adapter"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")

    stub_cls = _make_local_db_stub(None)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    api_mock = Mock()
    api_mock.create_repo = Mock()
    api_mock.upload_folder = Mock()
    _install_hf_stub(monkeypatch, api_mock=api_mock)

    result = runner.invoke(app, ["push", str(ckpt), "org/literal"])

    assert result.exit_code == 0, result.output
    api_mock.upload_folder.assert_called_once()
    assert api_mock.upload_folder.call_args.kwargs["folder_path"] == str(ckpt)


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_push_private_flag_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--private`` / ``--public`` should reach ``create_repo``."""
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()

    stub_cls = _make_local_db_stub(None)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    api_mock = Mock()
    api_mock.create_repo = Mock()
    api_mock.upload_folder = Mock()
    _install_hf_stub(monkeypatch, api_mock=api_mock)

    result = runner.invoke(app, ["push", str(ckpt), "org/pub", "--public"])
    assert result.exit_code == 0, result.output

    kwargs = api_mock.create_repo.call_args.kwargs
    assert kwargs["private"] is False
    assert "private=False" in result.output

    # Also exercise the private side so we cover both legs of the flag.
    api_mock.create_repo.reset_mock()
    api_mock.upload_folder.reset_mock()
    result = runner.invoke(app, ["push", str(ckpt), "org/priv", "--private"])
    assert result.exit_code == 0, result.output
    assert api_mock.create_repo.call_args.kwargs["private"] is True
    assert "private=True" in result.output


def test_push_commit_message_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "adapter"
    ckpt.mkdir()

    stub_cls = _make_local_db_stub(None)
    monkeypatch.setattr("carl_studio.db.LocalDB", stub_cls)

    api_mock = Mock()
    api_mock.create_repo = Mock()
    api_mock.upload_folder = Mock()
    _install_hf_stub(monkeypatch, api_mock=api_mock)

    result = runner.invoke(
        app,
        ["push", str(ckpt), "org/m", "--message", "ship phase 2"],
    )
    assert result.exit_code == 0, result.output
    assert api_mock.upload_folder.call_args.kwargs["commit_message"] == "ship phase 2"


# ---------------------------------------------------------------------------
# Help smoke
# ---------------------------------------------------------------------------


def test_push_help_renders() -> None:
    result = runner.invoke(app, ["push", "--help"])
    assert result.exit_code == 0
    assert "Ship" in result.output or "ship" in result.output
    # Both positional slots must surface in --help so users know the shape.
    assert "RUN_ID_OR_PATH" in result.output.upper() or "run_id" in result.output.lower()
    assert "REPO" in result.output.upper()
