from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import yaml

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


def test_release_manifests_match_studio_source_version():
    root = Path(__file__).resolve().parents[1]
    version = str(read_source_version(root))
    for name in (
        "plugin.json",
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        "package.json",
    ):
        manifest = json.loads((root / name).read_text())
        assert manifest["version"] == version, name
    assert json.loads((root / "package.json").read_text())["terminals"]["version"] == version


def test_studio_requires_the_released_core_version():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    core = tomllib.loads((root / "packages/carl-core/pyproject.toml").read_text())["project"]
    tree = ast.parse((root / "packages/carl-core/src/carl_core/__init__.py").read_text())
    declared = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    )
    assert declared == core["version"]
    assert f"carl-core>={core['version']}" in project["dependencies"]


def test_publishing_has_one_tag_trigger_and_manual_build_only():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/publish.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    assert set(workflow["on"]["push"]["tags"]) == {"v*", "carl-*@*"}
    publish = next(
        step
        for step in workflow["jobs"]["publish-root"]["steps"]
        if step.get("name") == "Publish to PyPI"
    )
    assert publish["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
