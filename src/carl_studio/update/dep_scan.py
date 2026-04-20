"""Dependency-delta scan via PyPI JSON API.

Stdlib-only (urllib). Short timeout per package; non-fatal errors go
into the report.errors list rather than aborting the whole scan.
Consent-gated by ``consent_gate("telemetry")`` at the caller level
since this is the only network egress in ``carl update``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from carl_studio.update.report import DependencyDelta


_PYPI_TIMEOUT_S = 5.0
_PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"


def _fetch_latest_pypi(package: str) -> str | None:
    """Return the latest version string from PyPI, or None on any failure."""
    url = _PYPI_JSON_URL.format(package=package)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_PYPI_TIMEOUT_S) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    info = data.get("info") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return None
    version = info.get("version")
    return version if isinstance(version, str) else None


def _compare_versions(current: str, latest: str) -> str:
    """Coarse severity: 'major' | 'minor' | 'patch' | 'info' | 'equal'.

    No PEP 440 parsing — we only look at the left-anchored numeric
    prefix. False negatives on pre-release suffixes are fine; this is
    a display heuristic, not a dependency resolver.
    """
    if current == latest:
        return "equal"
    c_parts = current.split(".")[:3]
    l_parts = latest.split(".")[:3]

    def _int(s: str) -> int:
        num: list[str] = []
        for ch in s:
            if ch.isdigit():
                num.append(ch)
            else:
                break
        return int("".join(num)) if num else 0

    c_major, c_minor, c_patch = (_int(x) for x in (c_parts + ["0", "0"])[:3])
    l_major, l_minor, l_patch = (_int(x) for x in (l_parts + ["0", "0"])[:3])
    if l_major > c_major:
        return "major"
    if l_minor > c_minor:
        return "minor"
    if l_patch > c_patch:
        return "patch"
    return "info"


def scan_dep_deltas(
    *,
    installed: dict[str, str],
    packages_of_interest: list[str] | None = None,
) -> tuple[list[DependencyDelta], list[str]]:
    """Return ``(deltas, errors)`` for the given installed package map.

    Args:
        installed: mapping of package-name → installed-version string.
            Typically produced by reading pyproject.toml requires or
            ``importlib.metadata``.
        packages_of_interest: optional filter — only query these. When
            ``None``, queries every package in ``installed``. Practical
            usage: focus on ~10 CARL-core packages to bound PyPI load.

    Returns:
        (deltas-where-latest-differs-from-current, non-fatal errors)
    """

    errors: list[str] = []
    deltas: list[DependencyDelta] = []
    targets = packages_of_interest or list(installed.keys())
    for pkg in targets:
        current = installed.get(pkg)
        if current is None:
            errors.append(f"{pkg}: not installed")
            continue
        latest = _fetch_latest_pypi(pkg)
        if latest is None:
            errors.append(f"{pkg}: PyPI lookup failed")
            continue
        severity = _compare_versions(current, latest)
        if severity == "equal":
            continue
        deltas.append(
            DependencyDelta(
                package=pkg,
                current=current,
                latest=latest,
                severity=severity,
                summary=None,
            )
        )
    return deltas, errors


def installed_versions_from_metadata(packages: list[str]) -> dict[str, str]:
    """Read installed versions via importlib.metadata."""
    from importlib.metadata import PackageNotFoundError, version as _v

    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = _v(pkg)
        except PackageNotFoundError:
            continue
    return out
