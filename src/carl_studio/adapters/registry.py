"""Registry mapping adapter names to :class:`UnifiedBackend` instances."""

from __future__ import annotations

from typing import Any

from carl_core.errors import ConfigError

from .protocol import UnifiedBackend


_REGISTRY: dict[str, UnifiedBackend] = {}


def register_adapter(adapter: UnifiedBackend) -> None:
    """Register ``adapter`` under its ``name`` attribute.

    A later call with the same name replaces the prior entry (idempotent
    re-registration is safe — useful in tests and plugin reloads).
    """
    if not isinstance(getattr(adapter, "name", None), str) or not adapter.name:
        raise ConfigError(
            "adapter is missing a non-empty 'name' attribute",
            code="carl.adapter.registry",
            context={"type": type(adapter).__name__},
        )
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> UnifiedBackend:
    """Look up a registered adapter by name.

    Raises:
        ConfigError: if ``name`` is not registered. The error's ``context``
            includes the list of known backends so CLI output can render a
            useful hint.
    """
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(
            "backend name must be a non-empty string",
            code="carl.adapter.unknown",
            context={"known": sorted(_REGISTRY.keys())},
        )
    if name not in _REGISTRY:
        raise ConfigError(
            f"unknown training backend: {name!r}",
            code="carl.adapter.unknown",
            context={"known": sorted(_REGISTRY.keys())},
        )
    return _REGISTRY[name]


def list_adapters() -> list[dict[str, Any]]:
    """Return one dict per registered adapter: ``{name, available}``.

    ``available`` is computed by calling each adapter's ``available()``
    method. Adapters must never raise from ``available()``; any exception
    is coerced to ``False`` defensively.
    """
    out: list[dict[str, Any]] = []
    for adapter in _REGISTRY.values():
        try:
            is_available = bool(adapter.available())
        except Exception:
            is_available = False
        out.append({"name": adapter.name, "available": is_available})
    return sorted(out, key=lambda e: e["name"])


def _unregister(name: str) -> None:
    """Test-only hook: drop ``name`` from the registry if present."""
    _REGISTRY.pop(name, None)


def _clear() -> None:
    """Test-only hook: wipe the registry."""
    _REGISTRY.clear()
