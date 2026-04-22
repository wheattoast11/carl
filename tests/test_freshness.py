"""Tests for the freshness check system."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from carl_core.errors import ValidationError

from carl_studio.freshness import (
    CATEGORY_CONFIG,
    CATEGORY_CREDENTIAL,
    CATEGORY_PACKAGE,
    CHECK_INTERVAL_DAYS,
    CODE_CAMP_SESSION_EXPIRED,
    CODE_CONFIG_INTERNAL_DEFAULT,
    CODE_CONFIG_INVALID,
    CODE_MISSING_HF_TOKEN,
    CODE_STALE_PKG,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
    FreshnessIssue,
    FreshnessReport,
    _check_config,
    _check_credentials,
    _check_packages,
    needs_check,
    run_freshness_check,
)


class TestFreshnessIssue:
    def test_happy_path(self):
        issue = FreshnessIssue(
            code=CODE_STALE_PKG,
            severity=SEVERITY_WARN,
            category=CATEGORY_PACKAGE,
            subject="trl",
            detail="trl: installed 0.12, recommended >=1.0",
            remediation="pip install --upgrade 'trl>=1.0'",
        )
        assert issue.code == CODE_STALE_PKG
        assert issue.severity == SEVERITY_WARN
        assert issue.category == CATEGORY_PACKAGE

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            FreshnessIssue(
                code="x", severity="catastrophic", category=CATEGORY_PACKAGE,
                subject="s", detail="d", remediation="r",
            )

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            FreshnessIssue(
                code="x", severity=SEVERITY_WARN, category="mystery",
                subject="s", detail="d", remediation="r",
            )


class TestFreshnessReport:
    def test_no_issues(self):
        report = FreshnessReport()
        assert not report.has_issues
        assert not report.has_errors
        assert not report.has_warnings
        assert report.summary == "all clear"

    def test_has_errors_flag(self):
        issue = FreshnessIssue(
            code="x", severity=SEVERITY_ERROR, category=CATEGORY_CONFIG,
            subject="s", detail="bad", remediation="fix",
        )
        report = FreshnessReport(issues=[issue])
        assert report.has_errors
        assert not report.has_warnings
        assert report.has_issues

    def test_has_warnings_flag(self):
        issue = FreshnessIssue(
            code="x", severity=SEVERITY_WARN, category=CATEGORY_PACKAGE,
            subject="s", detail="stale", remediation="upgrade",
        )
        report = FreshnessReport(issues=[issue])
        assert not report.has_errors
        assert report.has_warnings
        assert report.has_issues

    def test_info_does_not_trigger_has_issues(self):
        issue = FreshnessIssue(
            code="x", severity=SEVERITY_INFO, category=CATEGORY_PACKAGE,
            subject="s", detail="fyi", remediation="none",
        )
        report = FreshnessReport(issues=[issue])
        assert not report.has_errors
        assert not report.has_warnings

    def test_legacy_stale_packages_kwarg_still_works(self):
        """Old callers passed ``stale_packages=[...]`` — keep them working."""
        report = FreshnessReport(
            stale_packages=["trl: installed 0.12, recommended >=1.0"]
        )
        assert report.has_issues
        assert "1 stale package(s)" in report.summary
        # Back-compat list view is derived from typed issues:
        assert report.stale_packages == [
            "trl: installed 0.12, recommended >=1.0"
        ]
        assert report.issues[0].category == CATEGORY_PACKAGE
        assert report.issues[0].code == CODE_STALE_PKG

    def test_legacy_config_warnings_kwarg_still_works(self):
        report = FreshnessReport(config_warnings=["naming_prefix is internal"])
        assert report.has_issues
        assert "1 config warning(s)" in report.summary
        assert report.config_warnings == ["naming_prefix is internal"]

    def test_legacy_credential_warnings_kwarg_still_works(self):
        report = FreshnessReport(credential_warnings=["session expired"])
        assert report.has_issues
        assert "1 credential issue(s)" in report.summary
        assert report.credential_warnings == ["session expired"]

    def test_multiple_categories_summarized(self):
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

    def test_stale_check_after_ttl(self, tmp_path: Path):
        import yaml

        stale = datetime.now() - timedelta(days=CHECK_INTERVAL_DAYS + 1)
        f = tmp_path / "freshness.yaml"
        f.write_text(yaml.dump({"last_check": stale.isoformat()}))
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert needs_check()

    def test_ttl_default_is_24h(self):
        """TTL default: 24h per spec."""
        assert CHECK_INTERVAL_DAYS == 1

    def test_force_bypasses_ttl(self, tmp_path: Path):
        """Force always runs, regardless of TTL state."""
        import yaml

        f = tmp_path / "freshness.yaml"
        f.write_text(yaml.dump({"last_check": datetime.now().isoformat()}))
        with (
            patch("carl_studio.freshness.FRESHNESS_FILE", f),
            patch("carl_studio.freshness.CARL_HOME", tmp_path),
            patch("importlib.metadata.version", return_value="99.0.0"),
        ):
            # needs_check would say no, but force runs anyway
            assert not needs_check()
            report = run_freshness_check(force=True)
            assert isinstance(report, FreshnessReport)

    def test_corrupt_file_forces_check(self, tmp_path: Path):
        f = tmp_path / "freshness.yaml"
        f.write_text("not: [valid: yaml: {{")
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert needs_check()

    def test_non_dict_file_forces_check(self, tmp_path: Path):
        f = tmp_path / "freshness.yaml"
        f.write_text("- a list")  # valid YAML but not a dict
        with patch("carl_studio.freshness.FRESHNESS_FILE", f):
            assert needs_check()


class TestCheckPackages:
    def test_stale_package_detected_emits_typed_issue(self):
        report = FreshnessReport()
        with patch("importlib.metadata.version", return_value="0.12.0"):
            _check_packages(report)
        # Per spec: code == "carl.freshness.stale_pkg", severity == warn.
        stale = [i for i in report.issues if i.code == CODE_STALE_PKG]
        assert stale, "expected at least one stale_pkg issue"
        assert any(i.subject == "trl" for i in stale)
        for issue in stale:
            assert issue.severity == SEVERITY_WARN
            assert issue.category == CATEGORY_PACKAGE
            assert "pip install" in issue.remediation

    def test_current_package_ok(self):
        report = FreshnessReport()
        with patch("importlib.metadata.version", return_value="99.0.0"):
            _check_packages(report)
        assert not [i for i in report.issues if i.code == CODE_STALE_PKG]

    def test_missing_package_ignored(self):
        import importlib.metadata

        report = FreshnessReport()
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("nope"),
        ):
            _check_packages(report)
        assert not report.issues

    def test_hf_style_corruption_emits_dep_corrupt_error(self) -> None:
        """The HF scenario: metadata.version() returns None for an importable package.

        Regression test: before the dep-probe rewrite, ``_check_packages``
        silently swallowed this via ``except (ValueError, AttributeError,
        TypeError): continue``. Now it surfaces as a
        ``carl.freshness.dep_corrupt`` error with a concrete repair command.
        """
        from carl_studio.freshness import CODE_DEP_CORRUPT, SEVERITY_ERROR

        report = FreshnessReport()
        with patch("importlib.metadata.version", return_value=None):
            _check_packages(report)

        corrupt_issues = [i for i in report.issues if i.code == CODE_DEP_CORRUPT]
        assert corrupt_issues, "expected dep_corrupt issues for None-returning metadata"
        for issue in corrupt_issues:
            assert issue.severity == SEVERITY_ERROR
            assert "pip install --force-reinstall --no-deps" in issue.remediation


class TestCheckConfig:
    def test_deprecated_prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("naming_prefix: il-terminals-carl\n")
        report = FreshnessReport()
        _check_config(report)
        issues = [i for i in report.issues if i.code == CODE_CONFIG_INTERNAL_DEFAULT]
        assert any(i.subject == "naming_prefix" for i in issues)

    def test_deprecated_namespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("hub_namespace: wheattoast11\n")
        report = FreshnessReport()
        _check_config(report)
        issues = [i for i in report.issues if i.code == CODE_CONFIG_INTERNAL_DEFAULT]
        assert any(i.subject == "hub_namespace" for i in issues)

    def test_clean_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text("hub_namespace: myuser\n")
        report = FreshnessReport()
        _check_config(report)
        assert not report.issues

    def test_no_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        report = FreshnessReport()
        _check_config(report)
        assert not report.issues

    def test_invalid_yaml_is_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "carl.yaml").write_text(": : bad yaml :::")
        report = FreshnessReport()
        _check_config(report)
        errors = [i for i in report.issues if i.code == CODE_CONFIG_INVALID]
        assert errors
        assert errors[0].severity == SEVERITY_ERROR


class TestCheckCredentials:
    """_check_credentials: HuggingFace token and camp session checks."""

    def test_no_hf_token_warns_with_structured_issue(self):
        """Per spec: missing API key → severity warn, category credential,
        remediation mentions (for camp) 'carl camp login' or (for HF) login hint.
        """
        import types

        report = FreshnessReport()
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.get_token = lambda: None  # type: ignore[attr-defined]
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            patch("carl_studio.db.LocalDB", side_effect=Exception("skip")),
        ):
            _check_credentials(report)
        hf_issues = [i for i in report.issues if i.code == CODE_MISSING_HF_TOKEN]
        assert hf_issues, "expected a missing_hf_token issue"
        assert hf_issues[0].severity == SEVERITY_WARN
        assert hf_issues[0].category == CATEGORY_CREDENTIAL
        assert "hf auth login" in hf_issues[0].remediation

    def test_camp_session_expired_remediation_contains_carl_camp_login(self):
        """Per spec: severity warn, category credential, remediation mentions
        ``carl camp login``.
        """
        import types

        class _DB:
            def get_auth(self, key: str) -> str | None:
                # simulate expired: jwt is None but tier survived cache
                return None if key == "jwt" else "PAID"

            def get_config(self, key: str) -> str | None:
                return None

            def close(self) -> None:  # pragma: no cover
                pass

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.get_token = lambda: "hf_abc"  # type: ignore[attr-defined]

        report = FreshnessReport()
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            patch("carl_studio.db.LocalDB", return_value=_DB()),
        ):
            _check_credentials(report)
        camp_issues = [i for i in report.issues if i.code == CODE_CAMP_SESSION_EXPIRED]
        assert camp_issues
        assert camp_issues[0].severity == SEVERITY_WARN
        assert camp_issues[0].category == CATEGORY_CREDENTIAL
        assert "carl camp login" in camp_issues[0].remediation

    def test_hf_token_present_no_warning(self):
        import types

        report = FreshnessReport()
        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.get_token = lambda: "hf_abc123"  # type: ignore[attr-defined]
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            patch("carl_studio.db.LocalDB", side_effect=Exception("skip")),
        ):
            _check_credentials(report)
        hf_issues = [i for i in report.issues if i.code == CODE_MISSING_HF_TOKEN]
        assert not hf_issues

    def test_hf_hub_missing_no_error(self):
        import builtins

        report = FreshnessReport()
        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("No module named 'huggingface_hub'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocking_import):
            _check_credentials(report)
        hf_issues = [i for i in report.issues if i.code == CODE_MISSING_HF_TOKEN]
        assert not hf_issues


class TestRunFreshnessCheck:
    def test_full_run_persists_timestamp(self, tmp_path: Path):
        with (
            patch("carl_studio.freshness.CARL_HOME", tmp_path),
            patch("carl_studio.freshness.FRESHNESS_FILE", tmp_path / "freshness.yaml"),
            patch("importlib.metadata.version", return_value="99.0.0"),
        ):
            report = run_freshness_check()
            assert report.checked_at is not None
            assert (tmp_path / "freshness.yaml").exists()

    def test_force_is_equivalent_to_default(self, tmp_path: Path):
        """force=True and default both invoke the checks; neither short-circuits
        on TTL (TTL lives in needs_check, not here)."""
        with (
            patch("carl_studio.freshness.CARL_HOME", tmp_path),
            patch("carl_studio.freshness.FRESHNESS_FILE", tmp_path / "freshness.yaml"),
            patch("importlib.metadata.version", return_value="99.0.0"),
        ):
            a = run_freshness_check(force=True)
            b = run_freshness_check(force=False)
        assert isinstance(a, FreshnessReport)
        assert isinstance(b, FreshnessReport)
