#!/usr/bin/env python3
"""Resolve and optionally rewrite the package version before publishing.

Policy:
- If the source version or release tag is already higher than the latest PyPI
  version, keep that manual bump.
- Otherwise, bump the latest PyPI version to the next minor release.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PYPROJECT_VERSION_RE = re.compile(r'(?m)^version = "(?P<version>[^"]+)"$')
INIT_VERSION_RE = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')
SEMVER_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str | None) -> "Version | None":
        if raw is None:
            return None
        match = SEMVER_RE.match(raw.strip())
        if match is None:
            return None
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def bump_minor(self) -> "Version":
        return Version(self.major, self.minor + 1, 0)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _extract_version(text: str, pattern: re.Pattern[str], path: Path) -> Version:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Could not find version string in {path}")

    version = Version.parse(match.group("version"))
    if version is None:
        raise RuntimeError(f"Unsupported version format in {path}: {match.group('version')}")
    return version


def read_source_version(root: Path) -> Version:
    pyproject_path = root / "pyproject.toml"
    init_path = root / "src" / "carl_studio" / "__init__.py"

    pyproject_text = pyproject_path.read_text()
    init_text = init_path.read_text()

    pyproject_version = _extract_version(pyproject_text, PYPROJECT_VERSION_RE, pyproject_path)
    init_version = _extract_version(init_text, INIT_VERSION_RE, init_path)

    if pyproject_version != init_version:
        raise RuntimeError(
            "Source version mismatch: "
            f"pyproject.toml has {pyproject_version}, __init__.py has {init_version}"
        )

    return pyproject_version


def fetch_pypi_version(package_name: str, timeout: int = 10) -> Version | None:
    url = f"https://pypi.org/pypi/{package_name}/json"
    request = urllib.request.Request(url, headers={"User-Agent": "carl-studio-release-version"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None

    info = payload.get("info", {})
    return Version.parse(info.get("version"))


def resolve_target_version(
    source_version: Version,
    published_version: Version | None,
    release_tag: Version | None,
) -> tuple[Version, str]:
    manual_version = source_version
    mode = "source"

    if release_tag is not None and release_tag > manual_version:
        manual_version = release_tag
        mode = "manual_tag"

    if published_version is None:
        return manual_version, mode

    if manual_version > published_version:
        if manual_version == source_version:
            return manual_version, "manual_source"
        return manual_version, mode

    return published_version.bump_minor(), "auto_minor"


def _replace_version(path: Path, pattern: re.Pattern[str], version: Version, template: str) -> bool:
    text = path.read_text()
    updated = pattern.sub(template.format(version=version), text, count=1)
    if updated == text:
        return False
    path.write_text(updated)
    return True


def apply_version(root: Path, version: Version) -> list[Path]:
    changed: list[Path] = []

    pyproject_path = root / "pyproject.toml"
    init_path = root / "src" / "carl_studio" / "__init__.py"

    if _replace_version(pyproject_path, PYPROJECT_VERSION_RE, version, 'version = "{version}"'):
        changed.append(pyproject_path)
    if _replace_version(init_path, INIT_VERSION_RE, version, '__version__ = "{version}"'):
        changed.append(init_path)

    return changed


def write_github_output(path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--package-name", default="carl-studio")
    parser.add_argument("--release-tag", default=None)
    parser.add_argument("--published-version", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--github-output", default=None, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = args.root.resolve()

    source_version = read_source_version(root)
    published_version = Version.parse(args.published_version)
    if published_version is None and args.published_version is None:
        published_version = fetch_pypi_version(args.package_name)

    release_tag = Version.parse(args.release_tag)
    target_version, mode = resolve_target_version(source_version, published_version, release_tag)

    changed_paths: list[Path] = []
    if args.apply:
        changed_paths = apply_version(root, target_version)

    outputs = {
        "version": str(target_version),
        "mode": mode,
        "changed": "true" if changed_paths else "false",
        "source_version": str(source_version),
        "published_version": str(published_version) if published_version is not None else "",
        "release_tag": str(release_tag) if release_tag is not None else "",
    }

    if args.github_output is not None:
        write_github_output(args.github_output, outputs)

    print(f"Resolved version: {target_version} ({mode})")
    if published_version is not None:
        print(f"Published version: {published_version}")
    if release_tag is not None:
        print(f"Release tag version: {release_tag}")
    if changed_paths:
        print("Updated files:")
        for path in changed_paths:
            print(f"- {path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
