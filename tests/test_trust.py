from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from carl_studio.db import LocalDB
from carl_studio.trust import TrustRegistry, TrustState


cli_trust_mod = cast(Any, import_module("carl_studio.cli.trust"))


runner = CliRunner()
app: Any = cli_trust_mod.trust_app


def _wired() -> None:
    import carl_studio.cli.wiring as _wiring  # noqa: F401

    assert _wiring is not None


@pytest.fixture
def registry(tmp_path: Path) -> TrustRegistry:
    return TrustRegistry(LocalDB(tmp_path / "trust.db"))


def _registry_factory(registry: TrustRegistry) -> object:
    return lambda: registry


def _root_factory(root: Path | None) -> object:
    return lambda: root


def _confirm_factory(result: bool) -> object:
    def _confirm(*_args: object, **_kwargs: object) -> bool:
        return result

    return _confirm


def _set_registry(monkeypatch: pytest.MonkeyPatch, registry: TrustRegistry) -> None:
    monkeypatch.setattr(cli_trust_mod, "get_trust_registry", _registry_factory(registry))


def _set_detected_root(monkeypatch: pytest.MonkeyPatch, root: Path | None) -> None:
    monkeypatch.setattr(cli_trust_mod, "_detect_project_root", _root_factory(root))


def _set_confirm(monkeypatch: pytest.MonkeyPatch, result: bool) -> None:
    monkeypatch.setattr(cli_trust_mod.ui, "confirm", _confirm_factory(result))


def test_trust_registry_round_trip(tmp_path: Path) -> None:
    db = LocalDB(tmp_path / "trust.db")
    registry = TrustRegistry(db)
    assert registry.get() == TrustState()

    state = registry.trust_root(tmp_path)
    assert state.acknowledged_project_root == str(tmp_path.resolve())
    assert registry.is_trusted_root(tmp_path) is True


def test_trust_registry_disable_enable_reset(tmp_path: Path) -> None:
    db = LocalDB(tmp_path / "trust.db")
    registry = TrustRegistry(db)

    disabled = registry.set_enabled(False)
    assert disabled.enabled is False
    assert registry.is_enabled() is False

    enabled = registry.set_enabled(True)
    assert enabled.enabled is True
    assert registry.is_enabled() is True

    registry.trust_root(tmp_path)
    reset = registry.reset()
    assert reset == TrustState()


def test_trust_status_command_reads_registry(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    registry.trust_root(tmp_path)
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, tmp_path)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Trust status" in result.output
    assert "Acknowledged root" in result.output


def test_trust_disable_force_command_updates_registry(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)

    result = runner.invoke(app, ["disable", "--force"])
    assert result.exit_code == 0
    assert registry.is_enabled() is False


def test_trust_acknowledge_command_defaults_to_detected_project(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, tmp_path)

    result = runner.invoke(app, ["acknowledge"])
    assert result.exit_code == 0
    assert registry.is_trusted_root(tmp_path)


def test_trust_acknowledge_requires_path_outside_project(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, None)

    result = runner.invoke(app, ["acknowledge"])
    assert result.exit_code == 2
    assert "Not inside a CARL project" in result.output


def test_trust_reset_force_clears_acknowledged_root(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    registry.trust_root(tmp_path)
    registry.set_enabled(False)
    _set_registry(monkeypatch, registry)

    result = runner.invoke(app, ["reset", "--force"])
    assert result.exit_code == 0
    assert registry.get() == TrustState()


def test_trust_disable_interactive_cancel_exits_nonzero(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)
    _set_confirm(monkeypatch, False)

    result = runner.invoke(app, ["disable"])
    assert result.exit_code == 1
    assert registry.is_enabled() is True


def test_trust_reset_interactive_cancel_keeps_state(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    registry.trust_root(tmp_path)
    _set_registry(monkeypatch, registry)
    _set_confirm(monkeypatch, False)

    result = runner.invoke(app, ["reset"])
    assert result.exit_code == 1
    assert registry.is_trusted_root(tmp_path)


def test_trust_acknowledge_explicit_path(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    target = tmp_path / "project"
    target.mkdir()
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, None)

    result = runner.invoke(app, ["acknowledge", str(target)])
    assert result.exit_code == 0
    assert registry.is_trusted_root(target)


def test_trust_enable_reenables_registry(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    registry.set_enabled(False)
    _set_registry(monkeypatch, registry)

    result = runner.invoke(app, ["enable"])
    assert result.exit_code == 0
    assert registry.is_enabled() is True


def test_trust_registry_current_acknowledged_root_none_when_unset(
    registry: TrustRegistry,
) -> None:
    assert registry.current_acknowledged_root() is None


def test_trust_registry_handles_non_resolving_path(
    registry: TrustRegistry, tmp_path: Path
) -> None:
    class BrokenPath(Path):
        _flavour = type(Path())._flavour

        def resolve(self, strict: bool = False):
            raise OSError("boom")

    broken = BrokenPath(tmp_path / "broken")
    state = registry.trust_root(broken)
    assert state.acknowledged_project_root is not None


def test_trust_registry_untrusted_when_resolve_fails(
    registry: TrustRegistry, tmp_path: Path
) -> None:
    class BrokenPath(Path):
        _flavour = type(Path())._flavour

        def resolve(self, strict: bool = False):
            raise OSError("boom")

    broken = BrokenPath(tmp_path / "broken")
    assert registry.is_trusted_root(broken) is False


def test_trust_registry_key_is_stable(registry: TrustRegistry) -> None:
    assert registry.key == "carl.trust.state"


def test_trust_registry_persists_across_instances(tmp_path: Path) -> None:
    db = LocalDB(tmp_path / "trust.db")
    first = TrustRegistry(db)
    first.set_enabled(False)
    first.trust_root(tmp_path)

    second = TrustRegistry(db)
    state = second.get()
    assert state.enabled is False
    assert state.acknowledged_project_root == str(tmp_path.resolve())


def test_trust_registry_reset_persists_defaults(tmp_path: Path) -> None:
    db = LocalDB(tmp_path / "trust.db")
    registry = TrustRegistry(db)
    registry.set_enabled(False)
    registry.trust_root(tmp_path)
    registry.reset()

    reloaded = TrustRegistry(db)
    assert reloaded.get() == TrustState()


def test_trust_status_shows_no_project_when_missing(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, None)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "(not in project)" in result.output


def test_trust_disable_interactive_confirm_disables(
    registry: TrustRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    _set_registry(monkeypatch, registry)
    _set_confirm(monkeypatch, True)

    result = runner.invoke(app, ["disable"])
    assert result.exit_code == 0
    assert registry.is_enabled() is False


def test_trust_reset_interactive_confirm_resets(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    registry.trust_root(tmp_path)
    _set_registry(monkeypatch, registry)
    _set_confirm(monkeypatch, True)

    result = runner.invoke(app, ["reset"])
    assert result.exit_code == 0
    assert registry.get() == TrustState()


def test_trust_acknowledge_path_argument_can_be_relative(
    registry: TrustRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wired()
    target = tmp_path / "project"
    target.mkdir()
    monkeypatch.chdir(tmp_path.parent)
    _set_registry(monkeypatch, registry)
    _set_detected_root(monkeypatch, None)

    result = runner.invoke(app, ["acknowledge", str(target.relative_to(tmp_path.parent))])
    assert result.exit_code == 0
    assert registry.is_trusted_root(target)
