"""Owned installation and portable/native projection contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from carl_core.errors import CARLError

from carl_studio.plugin import PluginInstaller, project

ROOT = Path(__file__).resolve().parents[1]


def test_projection_contains_self_contained_skill_and_native_identity(tmp_path: Path) -> None:
    record = project(ROOT, tmp_path / "marketplace")
    snapshot = Path(record["snapshot"])
    for manifest in (
        snapshot / "plugin.json",
        snapshot / ".claude-plugin/plugin.json",
        snapshot / ".codex-plugin/plugin.json",
    ):
        assert json.loads(manifest.read_text())["name"] == "carl"
    assert (snapshot / "skills/carl/SKILL.md").read_bytes() == (
        ROOT / "skills/carl/SKILL.md"
    ).read_bytes()
    command = json.loads((snapshot / ".mcp.json").read_text())["mcpServers"]["carl"]
    assert command["args"] == ["-m", "carl_studio.mcp"]


def test_install_repeat_update_and_uninstall_preserve_foreign_files(tmp_path: Path) -> None:
    installer = PluginInstaller(ROOT, home=tmp_path)
    foreign = tmp_path / ".agents/skills/foreign/SKILL.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("foreign content")
    first = installer.install(["codex", "claude"], native=False)
    second = installer.install(["codex", "claude"], native=False)
    assert first == second
    assert first["healthy"]
    assert installer.uninstall(native=False)["retained"] == []
    assert foreign.read_text() == "foreign content"
    assert not (tmp_path / ".agents/skills/carl/SKILL.md").exists()
    assert installer.uninstall(native=False)["retained"] == []


def test_foreign_same_name_skill_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / ".agents/skills/carl/SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("foreign carl")
    with pytest.raises(CARLError, match="Foreign or edited"):
        PluginInstaller(ROOT, home=tmp_path).install(["codex"], native=False)
    assert target.read_text() == "foreign carl"


def test_edited_owned_skill_is_retained_on_uninstall(tmp_path: Path) -> None:
    installer = PluginInstaller(ROOT, home=tmp_path)
    installer.install(["claude"], native=False)
    target = tmp_path / ".claude/skills/carl/SKILL.md"
    target.write_text("operator edit")
    assert str(target) in installer.uninstall(native=False)["retained"]
    assert target.read_text() == "operator edit"


def test_symlinked_state_is_refused_before_projection(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    state = tmp_path / ".carl/plugins/carl"
    state.parent.mkdir(parents=True)
    state.symlink_to(outside, target_is_directory=True)
    with pytest.raises(CARLError):
        PluginInstaller(ROOT, home=tmp_path).install(["codex"], native=False)
    assert list(outside.iterdir()) == []


def test_opencode_merge_and_uninstall_preserve_other_configuration(tmp_path: Path) -> None:
    config = tmp_path / ".config/opencode/opencode.json"
    config.parent.mkdir(parents=True)
    original = {"model": "operator/model", "mcp": {"foreign": {"enabled": True}}}
    config.write_text(json.dumps(original))
    installer = PluginInstaller(ROOT, home=tmp_path)
    installer.install(["opencode"])
    assert json.loads(config.read_text())["mcp"]["carl"]["enabled"]
    installer.uninstall()
    assert json.loads(config.read_text()) == original


def test_predictable_temporary_symlink_is_not_followed(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("unchanged")
    target = tmp_path / ".agents/skills/carl/SKILL.md.carl-tmp"
    target.parent.mkdir(parents=True)
    target.symlink_to(victim)
    PluginInstaller(ROOT, home=tmp_path).install(["codex"], native=False)
    assert victim.read_text() == "unchanged"


def test_doctor_detects_changed_projection(tmp_path: Path) -> None:
    installer = PluginInstaller(ROOT, home=tmp_path)
    installer.install(["codex"], native=False)
    path = tmp_path / ".carl/plugins/carl/marketplace/plugins/carl/.mcp.json"
    path.write_text("{}")
    assert str(path) in installer.doctor()["drift"]


def test_preflight_refuses_second_host_before_first_alias_write(tmp_path: Path) -> None:
    target = tmp_path / ".claude/skills/carl/SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("foreign")
    with pytest.raises(CARLError):
        PluginInstaller(ROOT, home=tmp_path).install(["codex", "claude"], native=False)
    assert not (tmp_path / ".agents/skills/carl/SKILL.md").exists()


def test_opencode_parent_symlink_preserves_outside_configuration(tmp_path: Path) -> None:
    installer = PluginInstaller(ROOT, home=tmp_path)
    installer.install(["opencode"])
    target = tmp_path / ".config/opencode"
    target.rename(tmp_path / "saved")
    outside = tmp_path / "outside"
    outside.mkdir()
    path = outside / "opencode.json"
    path.write_text(json.dumps({"mcp": {"carl": installer._load()["opencode_entry"]}}))
    before = path.read_bytes()
    target.symlink_to(outside, target_is_directory=True)
    assert str(target / "opencode.json") in installer.uninstall()["retained"]
    assert path.read_bytes() == before


def test_projected_reference_symlink_refuses_update(tmp_path: Path) -> None:
    installer = PluginInstaller(ROOT, home=tmp_path)
    installer.install(["codex"], native=False)
    victim = tmp_path / "victim"
    victim.write_text("unchanged")
    reference = (
        tmp_path / ".carl/plugins/carl/marketplace/plugins/carl/skills/carl/references/workflows.md"
    )
    reference.unlink()
    reference.symlink_to(victim)
    with pytest.raises(CARLError):
        installer.install(["codex"], native=False)
    assert victim.read_text() == "unchanged"


@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_update_removes_retired_references_and_preserves_edits(tmp_path, edited, legacy):
    import shutil

    root = tmp_path / "source"
    root.mkdir()
    for name in ("plugin.json", "mcp.json", "pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(ROOT / "skills", root / "skills")
    home = tmp_path / "home"
    installer = PluginInstaller(root, home=home)
    installer.install(["codex"], native=False)
    alias = home / ".agents/skills/carl/references/workflows.md"
    snapshot = installer.state / "marketplace/plugins/carl/skills/carl/references/workflows.md"
    if legacy:
        import hashlib

        projection = installer.state / "marketplace/projection.json"
        data = json.loads(projection.read_text())
        del data["owned_files"]
        projection.write_text(json.dumps(data))
        record = json.loads(installer.marker.read_text())
        record["projection_files"][str(projection)] = hashlib.sha256(
            projection.read_bytes()
        ).hexdigest()
        installer.marker.write_text(json.dumps(record))
    (root / "skills/carl/references/workflows.md").unlink()
    if edited:
        alias.write_text("operator edit")
        with pytest.raises(CARLError, match="Edited retired"):
            installer.install(["codex"], native=False)
        assert alias.read_text() == "operator edit"
        assert snapshot.is_file()
    else:
        assert installer.install(["codex"], native=False)["healthy"]
        assert not alias.exists()
        assert not snapshot.exists()


@pytest.mark.parametrize("kind", ["absent", "local", "git", "foreign"])
def test_codex_marketplace_registration_preserves_source(tmp_path, monkeypatch, kind):
    installer = PluginInstaller(ROOT, home=tmp_path)
    source = {"sourceType": "local", "source": str(installer.state / "marketplace")}
    if kind == "git":
        source = {"sourceType": "git", "source": "https://github.com/wheattoast11/carl.git"}
    elif kind == "foreign":
        source = {"sourceType": "git", "source": "https://example.com/foreign.git"}
    entries = [] if kind == "absent" else [{"name": "carl-local", "marketplaceSource": source}]
    calls = []

    def run(argv):
        calls.append(argv)
        return json.dumps({"marketplaces": entries}).encode()

    monkeypatch.setattr(installer, "_run", run)
    if kind == "foreign":
        with pytest.raises(CARLError, match="Foreign Codex"):
            installer._register_codex_marketplace()
        assert len(calls) == 1
    else:
        installer._register_codex_marketplace()
        if kind == "absent":
            assert calls[-1] == [
                "codex",
                "plugin",
                "marketplace",
                "add",
                str(installer.state / "marketplace"),
            ]
        elif kind == "git":
            assert calls[-1] == ["codex", "plugin", "marketplace", "upgrade", "carl-local"]
        else:
            assert len(calls) == 1
    assert not any("remove" in call for call in calls)
