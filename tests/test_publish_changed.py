"""Tests for scripts/publish_changed.py."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_changed.py"
spec = importlib.util.spec_from_file_location("publish_changed", SCRIPT)
assert spec is not None and spec.loader is not None
publish_changed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_changed)


class TestChangedSubpackages:
    def test_detects_package_from_diff(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="packages/carl-core/src/carl_core/math.py\npackages/carl-core/README.md\n",
            stderr="",
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            result = publish_changed._changed_subpackages("HEAD~1")
        assert result == ["carl-core"]

    def test_ignores_non_package_paths(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="src/carl_studio/llm.py\nREADME.md\n", stderr=""
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            result = publish_changed._changed_subpackages("HEAD~1")
        assert result == []

    def test_picks_multiple_packages(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="packages/carl-core/x.py\npackages/carl-training/y.py\npackages/carl-core/z.py\n",
            stderr="",
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            result = publish_changed._changed_subpackages("HEAD~1")
        assert result == ["carl-core", "carl-training"]


class TestRootChanged:
    def test_root_file_changed(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="src/carl_studio/cli/chat.py\n", stderr=""
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            assert publish_changed._root_changed("HEAD~1") is True

    def test_only_package_changes_is_not_root(self) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="packages/carl-core/README.md\n", stderr=""
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            assert publish_changed._root_changed("HEAD~1") is False


class TestMainOutput:
    def test_names_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="packages/carl-core/x.py\n", stderr=""
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            rc = publish_changed.main(["--since", "HEAD~1"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out == "carl-core"

    def test_tags_format_includes_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="packages/carl-core/x.py\n", stderr=""
        )
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            rc = publish_changed.main(["--since", "HEAD~1", "--format", "tags"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        # Format: carl-core@X.Y.Z
        assert out.startswith("carl-core@")
        assert out.count("@") == 1

    def test_no_changes_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch.object(publish_changed.subprocess, "run", return_value=fake):
            rc = publish_changed.main(["--since", "HEAD~1"])
        assert rc == 0
        assert capsys.readouterr().out == ""
