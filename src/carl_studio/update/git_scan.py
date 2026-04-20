"""Git-log delta scan for `carl update`.

Stdlib-only (subprocess). Returns an empty list when the repo is not a
git repo or git is unavailable — non-fatal, surfaced in report.errors.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from carl_studio.update.report import GitCommitDelta


_DEFAULT_MAX_COMMITS = 20
_DEFAULT_DAYS = 30


def scan_git_commits(
    *,
    repo_path: Path | None = None,
    max_commits: int = _DEFAULT_MAX_COMMITS,
    days: int = _DEFAULT_DAYS,
) -> tuple[list[GitCommitDelta], list[str]]:
    """Return ``(commits, errors)`` for the given repo.

    Uses ``git log --since=<N>.days.ago --pretty=format:...`` with a
    unit-separator delimiter for safe parsing. Empty list + non-fatal
    error string on any failure.
    """

    errors: list[str] = []
    cwd = str(repo_path) if repo_path is not None else None
    sep = "\x1f"
    fmt = f"%h{sep}%ad{sep}%an{sep}%s"

    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"--since={days}.days.ago",
                f"--max-count={max_commits}",
                "--date=short",
                f"--pretty=format:{fmt}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        errors.append(f"git scan failed: {exc}")
        return [], errors

    if result.returncode != 0:
        errors.append(
            f"git log returned {result.returncode}: {result.stderr.strip()[:200]}"
        )
        return [], errors

    commits: list[GitCommitDelta] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split(sep)
        if len(parts) != 4:
            continue
        sha, date, author, subject = parts
        commits.append(
            GitCommitDelta(sha=sha, date=date, author=author, subject=subject)
        )
    return commits, errors


def current_head_sha(*, repo_path: Path | None = None) -> str | None:
    """Return the short SHA of HEAD, or None if git is unavailable."""

    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None
