"""Modularized CARL CLI package."""

from __future__ import annotations

from importlib import import_module

from carl_studio.logging_config import configure_logging

from .apps import app, camp_app, lab_app

# Install a single root logging handler once — every subcommand module
# imported below uses ``logging.getLogger(__name__)`` so the handler
# needs to be attached before those modules pick up their loggers.
# Honors CARL_LOG_LEVEL + CARL_LOG_JSON. Idempotent: safe to call
# multiple times (tests that need to reconfigure pass force=True).
configure_logging()

for _module in (
    "shared",
    "startup",
    "training",
    "push",
    "core",
    "observe",
    "project_data",
    "lab",
    "platform",
    "db",
    "wiring",
):
    import_module(f"{__name__}.{_module}")


# Top-level ``carl contract`` mount so the constitutional ledger is reachable
# without the ``camp`` prefix (spec: ``carl contract constitution [...]``).
# ``wiring.py`` has already mounted ``contract_app`` under ``camp_app``; we
# additionally mount it at the root here. Typer apps can be added to multiple
# parents safely — commands are looked up per parent.
try:  # best-effort: missing extras should not break the root CLI
    from .contract import contract_app as _contract_app

    app.add_typer(_contract_app, name="contract")
except ImportError:  # pragma: no cover - only when optional extras missing
    pass


__all__ = ["app", "camp_app", "lab_app"]
