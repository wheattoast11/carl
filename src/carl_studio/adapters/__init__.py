"""Backend adapters that route carl training to external systems.

Each adapter implements :class:`UnifiedBackend`. The module-level registry
maps backend name → adapter instance. Built-in names:

* ``trl`` — default, uses the in-process CARLTrainer (always available).
* ``unsloth`` — launches Unsloth in a subprocess via a generated entrypoint.
* ``axolotl`` — pipes translated YAML to ``axolotl train -``.
* ``tinker`` — translates + records state; submission path pending API lock-in.
* ``atropos`` — shells out to the atropos CLI with a JSON config file.

Registration happens on first import. Override by calling
``register_adapter()`` with a custom implementation.
"""

from __future__ import annotations

from .atropos_adapter import AtroposAdapter
from .axolotl_adapter import AxolotlAdapter
from .protocol import AdapterError, BackendJob, BackendStatus, UnifiedBackend
from .registry import get_adapter, list_adapters, register_adapter
from .tinker_adapter import TinkerAdapter
from .trl_adapter import TRLAdapter
from .unsloth_adapter import UnslothAdapter


_BUILTIN_ADAPTERS: tuple[UnifiedBackend, ...] = (
    TRLAdapter(),
    UnslothAdapter(),
    AxolotlAdapter(),
    TinkerAdapter(),
    AtroposAdapter(),
)


def _register_builtins() -> None:
    for adapter in _BUILTIN_ADAPTERS:
        register_adapter(adapter)


_register_builtins()


__all__ = [
    "AdapterError",
    "AtroposAdapter",
    "AxolotlAdapter",
    "BackendJob",
    "BackendStatus",
    "TRLAdapter",
    "TinkerAdapter",
    "UnifiedBackend",
    "UnslothAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
