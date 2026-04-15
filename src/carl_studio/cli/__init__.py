"""Modularized CARL CLI package."""

from __future__ import annotations

from importlib import import_module

from .apps import app, camp_app, lab_app

for _module in (
    "shared",
    "startup",
    "training",
    "core",
    "observe",
    "project_data",
    "lab",
    "platform",
    "wiring",
):
    import_module(f"{__name__}.{_module}")

__all__ = ["app", "camp_app", "lab_app"]
