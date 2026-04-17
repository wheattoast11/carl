"""Tests for CLI command wiring (WS-D4).

Verifies that:
- ``carl init`` and ``carl camp init`` resolve to the same command.
- ``carl flow`` and ``carl camp flow`` both work.
- The old hidden top-level billing aliases (``carl upgrade``,
  ``carl billing``, ``carl subscription``) are no longer registered —
  Typer should respond with a "no such command" exit code.
- The canonical camp-scoped paths still resolve.
"""
from __future__ import annotations

from typer.testing import CliRunner

from carl_studio.cli.apps import app

runner = CliRunner()


def _wired() -> None:
    import carl_studio.cli.wiring  # noqa: F401


class TestInitAliases:
    def test_carl_init_shows_help(self) -> None:
        _wired()
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        # Both top-level and camp-scoped init invoke the same function,
        # which is advertised as the one-shot onboarding wizard.
        assert "account" in result.output.lower() or "init" in result.output.lower()

    def test_carl_camp_init_shows_help(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "init", "--help"])
        assert result.exit_code == 0

    def test_init_and_camp_init_bind_same_function(self) -> None:
        """Both routes must resolve to the same underlying callback."""
        _wired()
        from carl_studio.cli.apps import app as root, camp_app

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
        _wired()
        result = runner.invoke(app, ["flow", "--list"])
        assert result.exit_code == 0

    def test_carl_camp_flow_resolves(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "flow", "--list"])
        assert result.exit_code == 0
        assert "/echo" in result.output or "echo" in result.output

    def test_flow_and_camp_flow_bind_same_function(self) -> None:
        _wired()
        from carl_studio.cli.apps import app as root, camp_app

        top_flow = next(c for c in root.registered_commands if c.name == "flow")
        camp_flow = next(c for c in camp_app.registered_commands if c.name == "flow")
        assert top_flow.callback is camp_flow.callback


class TestBillingDedupe:
    """WS-D4: the duplicated, hidden top-level billing aliases are removed."""

    def test_carl_upgrade_is_not_registered(self) -> None:
        _wired()
        # `carl upgrade` should exit non-zero because Typer does not know
        # about the command anymore. Typer emits exit code 2 for unknown
        # subcommands.
        result = runner.invoke(app, ["upgrade"])
        assert result.exit_code != 0

    def test_carl_billing_is_not_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["billing"])
        assert result.exit_code != 0

    def test_carl_subscription_is_not_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["subscription"])
        assert result.exit_code != 0

    def test_carl_camp_upgrade_is_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "upgrade", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_billing_is_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "billing", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_subscription_is_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "subscription", "--help"])
        assert result.exit_code == 0

    def test_carl_camp_account_is_registered(self) -> None:
        _wired()
        result = runner.invoke(app, ["camp", "account", "--help"])
        assert result.exit_code == 0

    def test_no_duplicate_billing_commands_on_top_app(self) -> None:
        """Assert via introspection that billing names are NOT on app."""
        _wired()
        from carl_studio.cli.apps import app as root

        names = [c.name for c in root.registered_commands]
        assert names.count("upgrade") == 0
        assert names.count("billing") == 0
        assert names.count("subscription") == 0
