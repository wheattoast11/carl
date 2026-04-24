"""`carl --version` / `carl -V` must print the installed version and exit 0.

Regression: caught by post-publish smoke test of `carl-studio==0.18.1` from
PyPI — the binary answered "No such option: --version" instead of printing
the version. Every CLI users check reflexively to confirm an install.
"""
from __future__ import annotations

from typer.testing import CliRunner

from carl_studio import __version__
from carl_studio.cli.apps import app


runner = CliRunner()


class TestVersionFlag:
    def test_long_flag_prints_version_and_exits(self) -> None:
        # Ensure wiring is loaded so the callback is registered.
        import carl_studio.cli.wiring as _w  # noqa: F401

        assert _w is not None

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0, result.output
        assert __version__ in result.output

    def test_short_flag_prints_version_and_exits(self) -> None:
        import carl_studio.cli.wiring as _w  # noqa: F401

        assert _w is not None

        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0, result.output
        assert __version__ in result.output

    def test_version_output_shape(self) -> None:
        """Expected shape: `carl-studio <semver>` — machine-readable."""
        import carl_studio.cli.wiring as _w  # noqa: F401

        assert _w is not None

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        stripped = result.output.strip()
        assert stripped.startswith("carl-studio "), stripped
        # Parseable as semver-ish
        version_part = stripped.split(" ", 1)[1]
        assert version_part.count(".") >= 2
