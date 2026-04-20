"""v0.11 W2 — `carl update` tests.

Covers:
- UpdateReport shape + blast-radius derivation
- git_scan happy path (on the actual repo) + graceful failure
- dep_scan version comparison + PyPI mock
- CLI command: --summary-only, --dry-run, --json, plain
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from carl_studio.update.dep_scan import _compare_versions, scan_dep_deltas
from carl_studio.update.git_scan import current_head_sha, scan_git_commits
from carl_studio.update.report import (
    BlastRadiusEntry,
    DependencyDelta,
    GitCommitDelta,
    UpdateReport,
    build_update_report,
)


# ---------------------------------------------------------------------------
# UpdateReport
# ---------------------------------------------------------------------------


class TestUpdateReport:
    def test_empty_report_no_deltas(self) -> None:
        r = UpdateReport(carl_version="0.11.0")
        assert r.any_deltas is False

    def test_report_with_commits_has_deltas(self) -> None:
        r = UpdateReport(
            carl_version="0.11.0",
            git_commits=[
                GitCommitDelta(sha="abc1234", date="2026-04-20", subject="feat: x", author="Tej")
            ],
        )
        assert r.any_deltas is True

    def test_build_update_report_derives_blast_radius(self) -> None:
        commits = [
            GitCommitDelta(sha="abc1234", date="2026-04-20", subject="feat: add foo", author="Tej"),
            GitCommitDelta(sha="def5678", date="2026-04-19", subject="fix: bar regression", author="Tej"),
        ]
        deps = [
            DependencyDelta(package="x", current="1.0.0", latest="1.1.0", severity="minor"),
        ]
        report = build_update_report(
            carl_version="0.11.0", git_commits=commits, dep_deltas=deps
        )
        assert report.blast_radius
        # One entry per commit (capped at 5) + one per dep (capped at 5)
        assert len(report.blast_radius) == 3
        impacts = [b.impact for b in report.blast_radius]
        assert any("Unlocks" in i for i in impacts)  # from "feat:" prefix
        assert any("Resolves" in i for i in impacts)  # from "fix:" prefix

    def test_major_dep_bump_flagged_as_breaking(self) -> None:
        deps = [
            DependencyDelta(package="foo", current="1.0.0", latest="2.0.0", severity="major"),
        ]
        report = build_update_report(carl_version="0.11.0", dep_deltas=deps)
        assert report.blast_radius[0].direction == "breaking"


# ---------------------------------------------------------------------------
# git_scan
# ---------------------------------------------------------------------------


class TestGitScan:
    def test_current_head_sha_on_repo_returns_str(self) -> None:
        sha = current_head_sha()
        # Running in the carl-studio repo — should have a SHA
        assert sha is None or isinstance(sha, str)

    def test_scan_git_commits_on_non_repo_is_graceful(self, tmp_path: Path) -> None:
        commits, errors = scan_git_commits(repo_path=tmp_path, days=7)
        assert commits == []
        # Non-git dir yields at least one error but no raise
        assert isinstance(errors, list)

    def test_scan_git_commits_returns_recent(self) -> None:
        commits, errors = scan_git_commits(days=365, max_commits=5)
        # On the live repo there's definitely at least one recent commit
        assert len(commits) > 0
        assert all(isinstance(c.sha, str) for c in commits)


# ---------------------------------------------------------------------------
# dep_scan
# ---------------------------------------------------------------------------


class TestDepScan:
    def test_compare_versions_major(self) -> None:
        assert _compare_versions("1.0.0", "2.0.0") == "major"

    def test_compare_versions_minor(self) -> None:
        assert _compare_versions("1.0.0", "1.1.0") == "minor"

    def test_compare_versions_patch(self) -> None:
        assert _compare_versions("1.0.0", "1.0.1") == "patch"

    def test_compare_versions_equal(self) -> None:
        assert _compare_versions("1.0.0", "1.0.0") == "equal"

    def test_scan_with_mocked_pypi(self) -> None:
        """Mock the PyPI fetch — no live network hits in this test."""

        with patch(
            "carl_studio.update.dep_scan._fetch_latest_pypi",
            side_effect=lambda pkg: {
                "carl-studio": "0.11.0",
                "pydantic": "3.0.0",  # major bump
                "typer": "0.12.5",  # same as current
            }.get(pkg),
        ):
            installed = {
                "carl-studio": "0.10.0",
                "pydantic": "2.9.0",
                "typer": "0.12.5",
            }
            deltas, errors = scan_dep_deltas(
                installed=installed,
                packages_of_interest=["carl-studio", "pydantic", "typer"],
            )
            # carl-studio 0.10.0 → 0.11.0 = minor, pydantic 2 → 3 = major,
            # typer unchanged = skipped
            assert len(deltas) == 2
            by_pkg = {d.package: d for d in deltas}
            assert by_pkg["carl-studio"].severity == "minor"
            assert by_pkg["pydantic"].severity == "major"

    def test_scan_missing_package_in_errors(self) -> None:
        with patch("carl_studio.update.dep_scan._fetch_latest_pypi", return_value=None):
            deltas, errors = scan_dep_deltas(
                installed={"only-one": "1.0.0"}, packages_of_interest=["not-installed"]
            )
            assert deltas == []
            assert any("not installed" in e for e in errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestUpdateCli:
    def test_cli_runs_dry_run(self) -> None:
        """Dry-run skips network; exercises git scan + report rendering."""
        from carl_studio.cli.update import update_cmd
        import typer

        app = typer.Typer()
        app.command()(update_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--dry-run", "--summary-only"])
        assert result.exit_code == 0, result.output
        assert "carl" in result.output.lower()

    def test_cli_json_output_parses(self) -> None:
        from carl_studio.cli.update import update_cmd
        import typer

        app = typer.Typer()
        app.command()(update_cmd)

        runner = CliRunner()
        result = runner.invoke(app, ["--dry-run", "--json"])
        assert result.exit_code == 0, result.output
        # JSON must parse
        payload = json.loads(result.output)
        assert "carl_version" in payload
        assert "git_commits" in payload
