"""Tests for the freshness check system."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from carl_studio.freshness import (
    CHECK_INTERVAL_DAYS,
    FreshnessReport,
    _check_config,
    _check_credentials,
    _check_packages,
    needs_check,
    run_freshness_check,
)


class TestFreshnessReport:
    def test_no_issues(self):
        report = FreshnessReport(checked_at="2026-04-16T00:00:00")
        assert not report.has_issues
        assert report.summary == "all clear"

    def test_stale_packages(self):
        report = FreshnessReport(stale_packages=["trl: installed 0.12, recommended >=1.0"])
        assert report.has_issues
        assert "1 stale package(s)" in report.summary

    def test_config_warnings(self):
        report = FreshnessReport(config_warnings=["naming_prefix is internal"])
        assert report.has_issues
        assert "1 config warning(s)" in report.summary

    def test_credential_warnings(self):
        report = FreshnessReport(credential_warnings=["session expired"])
        assert report.has_issues
        assert "1 credential issue(s)" in report.summary

    def test_multiple_issues(self):
        report = FreshnessReport(
            stale_packages=["a"],
            config_warnings=["b"],
            credential_warnings=["c"],
        )
        assert report.has_issues
        assert "1 stale" in report.summary
        assert "1 config" in report.summary
        assert "1 credential" in report.summary


class TestNeedsCheck:
    def test_no_file(self, tmp_path: Path):
        with patch("carl_studio.freshness.FRESHNESS_FILE", tmp_path / "nope.yaml"):
            assert needs_check()

    def test_recent_check(self, tmp_path: Path):
        import yaml

        f = tmp_path / "freshness.yaml"
        f.write_text(yaml.dump({"last_check": datetime.now().isoformat()}))
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert not needs_check()

    def test_stale_check(self, tmp_path: Path):
        import yaml

        stale = datetime.now() - timedelta(days=CHECK_INTERVAL_DAYS + 1)
        f = tmp_path / "freshness.yaml"
        f.write_text(yaml.dump({"last_check": stale.isoformat()}))
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert needs_check()

    def test_corrupt_file(self, tmp_path: Path):
        f = tmp_path / "freshness.yaml"
        f.write_text("not: [valid: yaml: {{")
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert needs_check()


class TestCheckPackages:
    def test_stale_package_detected(self):
        report = FreshnessReport()
        with patch("importlib.metadata.version", return_value="0.12.0"):
            _check_packages(report)
        assert any("trl" in s for s in report.stale_packages)

    def test_current_package_ok(self):
        report = FreshnessReport()
        with patch("importlib.metadata.version", return_value="99.0.0"):
            _check_packages(report)
        assert len(report.stale_packages) == 0

    def test_missing_package_ignored(self):
        import importlib.metadata

        report = FreshnessReport()
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("nope"),
        ):
            _check_packages(report)
        assert len(report.stale_packages) == 0


class TestCheckConfig:
    def test_deprecated_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("naming_prefix: il-terminals-carl\n")
        report = FreshnessReport()
        _check_config(report)
        assert any("naming_prefix" in w for w in report.config_warnings)

    def test_deprecated_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("hub_namespace: wheattoast11\n")
        report = FreshnessReport()
        _check_config(report)
        assert any("hub_namespace" in w for w in report.config_warnings)

    def test_clean_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("hub_namespace: myuser\n")
        report = FreshnessReport()
        _check_config(report)
        assert len(report.config_warnings) == 0

    def test_no_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        report = FreshnessReport()
        _check_config(report)
        assert len(report.config_warnings) == 0


class TestCheckCredentials:
    """_check_credentials: HuggingFace token and camp session checks."""

    def test_no_hf_token_warns(self):
        """When huggingface_hub.get_token returns None, warn about missing token."""
        report = FreshnessReport()
        with patch("carl_studio.freshness.get_token", return_value=None, create=True):
            # get_token is imported inside _check_credentials via
            # ``from huggingface_hub import get_token``. We need to mock
            # at the point of use -- patch the import so it resolves to our mock.
            import types

            fake_hf = types.ModuleType("huggingface_hub")
            fake_hf.get_token = lambda: None  # type: ignore[attr-defined]
            with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
                _check_credentials(report)
        assert any("HuggingFace" in w for w in report.credential_warnings)

    def test_hf_token_present_no_warning(self):
        """When huggingface_hub.get_token returns a token, no warning."""
        import types

        report = FreshnessReport()
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.get_token = lambda: "hf_abc123"  # type: ignore[attr-defined]
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            patch("carl_studio.db.LocalDB", side_effect=Exception("skip")),
        ):
            _check_credentials(report)
        hf_warnings = [w for w in report.credential_warnings if "HuggingFace" in w]
        assert len(hf_warnings) == 0

    def test_hf_hub_missing_no_error(self):
        """When huggingface_hub is not installed, skip silently."""
        import builtins

        report = FreshnessReport()
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocking_import):
            _check_credentials(report)
        hf_warnings = [w for w in report.credential_warnings if "HuggingFace" in w]
        assert len(hf_warnings) == 0


class TestRunFreshnessCheck:
    def test_full_run(self, tmp_path: Path):
        with (
            patch("carl_studio.freshness.CARL_HOME", tmp_path),
            patch("carl_studio.freshness.FRESHNESS_FILE", tmp_path / "freshness.yaml"),
            patch("importlib.metadata.version", return_value="99.0.0"),
        ):
            report = run_freshness_check()
            assert report.checked_at
            assert (tmp_path / "freshness.yaml").exists()
