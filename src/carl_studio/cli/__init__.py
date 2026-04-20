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

__all__ = ["app", "camp_app", "lab_app"]
