from __future__ import annotations

from pathlib import Path

from scripts.release_version import (
    Version,
    apply_version,
    read_source_version,
    resolve_target_version,
)


def _write_version_files(root: Path, version: str) -> None:
    (root / "src" / "carl_studio").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (root / "src" / "carl_studio" / "__init__.py").write_text(f'__version__ = "{version}"\n')


def test_manual_source_version_wins_over_pypi(tmp_path: Path):
    _write_version_files(tmp_path, "0.5.0")

    source = read_source_version(tmp_path)
    target, mode = resolve_target_version(
        source_version=source,
        published_version=Version.parse("0.4.0"),
        release_tag=None,
    )

    assert str(target) == "0.5.0"
    assert mode == "manual_source"


def test_auto_minor_bump_when_source_matches_pypi(tmp_path: Path):
    _write_version_files(tmp_path, "0.3.0")

    source = read_source_version(tmp_path)
    target, mode = resolve_target_version(
        source_version=source,
        published_version=Version.parse("0.3.0"),
        release_tag=None,
    )

    assert str(target) == "0.4.0"
    assert mode == "auto_minor"


def test_manual_release_tag_wins_when_higher(tmp_path: Path):
    _write_version_files(tmp_path, "0.3.0")

    source = read_source_version(tmp_path)
    target, mode = resolve_target_version(
        source_version=source,
        published_version=Version.parse("0.3.0"),
        release_tag=Version.parse("0.6.0"),
    )

    assert str(target) == "0.6.0"
    assert mode == "manual_tag"


def test_apply_version_updates_both_files(tmp_path: Path):
    _write_version_files(tmp_path, "0.3.0")

    changed = apply_version(tmp_path, Version.parse("0.4.0"))

    assert {path.name for path in changed} == {"pyproject.toml", "__init__.py"}
    assert 'version = "0.4.0"' in (tmp_path / "pyproject.toml").read_text()
    assert '__version__ = "0.4.0"' in (tmp_path / "src" / "carl_studio" / "__init__.py").read_text()
