"""Tests for live CLI command wiring.

Verifies that:
- ``carl init`` and ``carl camp init`` resolve to the same command.
- ``carl flow`` and ``carl camp flow`` both work.
- ``carl research`` is a top-level canonical route while
  ``carl lab research`` remains wired.
- The old hidden top-level billing aliases (``carl upgrade``,
  ``carl billing``, ``carl subscription``) are no longer registered —
  Typer should respond with a "no such command" exit code.
- The canonical camp-scoped paths still resolve.
"""
from __future__ import annotations

import importlib

from typer.testing import CliRunner

runner = CliRunner()


def _wired():
    import carl_studio.cli.apps as apps_mod
    import carl_studio.cli.wiring as wiring_mod

    importlib.reload(apps_mod)
    importlib.reload(wiring_mod)
    return apps_mod


class TestInitAliases:
    def test_carl_init_shows_help(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["init", "--help"])
        assert result.exit_code == 0
        # Both top-level and camp-scoped init invoke the same function,
        # which is advertised as the one-shot onboarding wizard.
        assert "account" in result.output.lower() or "init" in result.output.lower()

    def test_carl_camp_init_shows_help(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "init", "--help"])
        assert result.exit_code == 0

    def test_init_and_camp_init_bind_same_function(self) -> None:
        """Both routes must resolve to the same underlying callback."""
        apps_mod = _wired()
        root = apps_mod.app
        camp_app = apps_mod.camp_app

        top_names = {c.name for c in root.registered_commands}
        camp_names = {c.name for c in camp_app.registered_commands}
        assert "init" in top_names
        assert "init" in camp_names

        top_init = next(c for c in root.registered_commands if c.name == "init")
        camp_init = next(c for c in camp_app.registered_commands if c.name == "init")
        # Typer stores the underlying callable as `.callback`.
        assert top_init.callback is camp_init.callback


class TestFlowAliases:
    def test_carl_flow_resolves(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["flow", "--list"])
        assert result.exit_code == 0

    def test_carl_camp_flow_resolves(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "flow", "--list"])
        assert result.exit_code == 0
        assert "/echo" in result.output or "echo" in result.output

    def test_flow_and_camp_flow_bind_same_function(self) -> None:
        apps_mod = _wired()
        root = apps_mod.app
        camp_app = apps_mod.camp_app

        top_flow = next(c for c in root.registered_commands if c.name == "flow")
        camp_flow = next(c for c in camp_app.registered_commands if c.name == "flow")
        assert top_flow.callback is camp_flow.callback


class TestResearchWiring:
    def test_carl_research_routes_to_live_subapp(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["research", "search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower()

    def test_carl_lab_research_routes_to_live_subapp(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["lab", "research", "search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower()

    def test_research_and_lab_research_bind_same_typer(self) -> None:
        apps_mod = _wired()
        root = apps_mod.app
        lab_app = apps_mod.lab_app

        top_research = next(group for group in root.registered_groups if str(group.name) == "research")
        lab_research = next(group for group in lab_app.registered_groups if str(group.name) == "research")
        assert top_research.typer_instance is lab_research.typer_instance

    def test_research_is_registered_on_top_level_app(self) -> None:
        apps_mod = _wired()
        root = apps_mod.app

        names = {str(group.name) for group in root.registered_groups}
        assert "research" in names

    def test_research_remains_registered_on_lab_app(self) -> None:
        apps_mod = _wired()
        lab_app = apps_mod.lab_app

        names = {str(group.name) for group in lab_app.registered_groups}
        assert "research" in names

    def test_research_extra_install_hint_available_on_root_stub_path(self, monkeypatch) -> None:
        from importlib import reload
        import builtins
        import sys

        import carl_studio.cli.apps as apps_mod
        import carl_studio.cli.wiring as wiring_mod

        real_import = builtins.__import__
        real_research_module = sys.modules.pop("carl_studio.research._cli", None)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "carl_studio.research._cli":
                raise ImportError("simulated missing research extra")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reload(apps_mod)
        reloaded_wiring = reload(wiring_mod)

        try:
            result = runner.invoke(apps_mod.app, ["research"])
            assert result.exit_code == 1
            assert "Research requires the arxiv package." in result.output
            assert "carl-studio[research]" in result.output
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            if real_research_module is not None:
                sys.modules["carl_studio.research._cli"] = real_research_module
            reload(apps_mod)
            reload(reloaded_wiring)

        _wired()

    def test_research_extra_install_hint_available_on_lab_stub_path(self, monkeypatch) -> None:
        from importlib import reload
        import builtins
        import sys

        import carl_studio.cli.apps as apps_mod
        import carl_studio.cli.wiring as wiring_mod

        real_import = builtins.__import__
        real_research_module = sys.modules.pop("carl_studio.research._cli", None)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "carl_studio.research._cli":
                raise ImportError("simulated missing research extra")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reload(apps_mod)
        reloaded_wiring = reload(wiring_mod)

        try:
            result = runner.invoke(apps_mod.app, ["lab", "research"])
            assert result.exit_code == 1
            assert "Research requires the arxiv package." in result.output
            assert "carl-studio[research]" in result.output
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            if real_research_module is not None:
                sys.modules["carl_studio.research._cli"] = real_research_module
            reload(apps_mod)
            reload(reloaded_wiring)

        _wired()


class TestBillingDedupe:
    """WS-D4: the duplicated, hidden top-level billing aliases are removed."""

    def test_carl_upgrade_is_not_registered(self) -> None:
        apps_mod = _wired()
        # `carl upgrade` should exit non-zero because Typer does not know
        # about the command anymore. Typer emits exit code 2 for unknown
        # subcommands.
        result = runner.invoke(apps_mod.app, ["upgrade"])
        assert result.exit_code != 0

    def test_carl_billing_is_not_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["billing"])
        assert result.exit_code != 0

    def test_carl_subscription_is_not_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["subscription"])
        assert result.exit_code != 0

    def test_carl_camp_upgrade_is_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "upgrade", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_billing_is_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "billing", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_subscription_is_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "subscription", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_account_is_registered(self) -> None:
        apps_mod = _wired()
        result = runner.invoke(apps_mod.app, ["camp", "account", "--help"])
        assert result.exit_code == 0

    def test_no_duplicate_billing_commands_on_top_app(self) -> None:
        """Assert via introspection that billing names are NOT on app."""
        apps_mod = _wired()
        root = apps_mod.app

        names = [c.name for c in root.registered_commands]
        assert names.count("upgrade") == 0
        assert names.count("billing") == 0
        assert names.count("subscription") == 0
