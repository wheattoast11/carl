"""Environment registry -- discover and look up CARL environments by name or lane."""

from __future__ import annotations

from typing import Type

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec

_REGISTRY: dict[str, Type[BaseEnvironment]] = {}


def register_environment(cls: Type[BaseEnvironment]) -> Type[BaseEnvironment]:
    """Class decorator to register an environment.

    Usage::

        @register_environment
        class MySandbox(BaseEnvironment):
            spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="my-sandbox", ...)
    """
    if not hasattr(cls, "spec") or not isinstance(cls.spec, EnvironmentSpec):
        raise TypeError(f"{cls.__name__} must have a class-level `spec: EnvironmentSpec`")
    if cls.spec.name in _REGISTRY:
        raise ValueError(f"Environment '{cls.spec.name}' already registered by {_REGISTRY[cls.spec.name].__name__}")
    _REGISTRY[cls.spec.name] = cls
    return cls


def get_environment(name: str) -> Type[BaseEnvironment]:
    """Look up environment class by name.

    Raises KeyError if name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown environment: '{name}'. Available: {available}")
    return _REGISTRY[name]


def get_environment_factory(name: str) -> Type[BaseEnvironment]:
    """Return an environment class suitable for TRL's environment_factory.

    Identical to get_environment but named for clarity when passing to GRPOTrainer.
    """
    return get_environment(name)


def list_environments(lane: EnvironmentLane | None = None) -> list[EnvironmentSpec]:
    """List registered environments, optionally filtered by lane."""
    specs = [cls.spec for cls in _REGISTRY.values()]
    if lane is not None:
        specs = [s for s in specs if s.lane == lane]
    return sorted(specs, key=lambda s: s.name)


def is_registered(name: str) -> bool:
    """Check if an environment name is registered."""
    return name in _REGISTRY


def clear_registry() -> None:
    """Clear all registered environments. For testing only."""
    _REGISTRY.clear()
