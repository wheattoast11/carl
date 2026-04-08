"""Multi-environment factory for TRL's environment_factory.

Routes between registered environments based on a dataset column.
TRL expects a single class for environment_factory — this creates
a meta-class that delegates to the correct environment per sample.

Usage::

    from carl_studio.training.multi_env import make_multi_environment
    from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv
    from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv

    MultiEnv = make_multi_environment({
        "python": CodingSandboxEnv,
        "sql": SQLSandboxEnv,
    })

    trainer = GRPOTrainer(
        ...,
        environment_factory=MultiEnv,
    )

Dataset must have an 'env' column matching one of the registered keys.
"""

from __future__ import annotations

from typing import Any, Type

from carl_studio.environments.protocol import BaseEnvironment


def make_multi_environment(
    environments: dict[str, Type[BaseEnvironment]],
    env_column: str = "env",
    default_env: str | None = None,
) -> type:
    """Create a TRL-compatible meta-environment class.

    Args:
        environments: Mapping from env key to environment class.
        env_column: Dataset column name that routes to the correct env.
        default_env: Default env key if column is missing. If None, uses first key.

    Returns:
        A class suitable for TRL's environment_factory parameter.
    """
    if not environments:
        raise ValueError("At least one environment required")

    _default = default_env or next(iter(environments))
    if _default not in environments:
        raise ValueError(f"default_env '{_default}' not in environments: {list(environments.keys())}")

    # Collect all tools across all environments
    _all_tools: dict[str, tuple[str, Type[BaseEnvironment]]] = {}
    for env_key, env_cls in environments.items():
        for tool_name in env_cls.spec.tools:
            if tool_name in _all_tools:
                # Same tool name in multiple envs — route by active env
                pass
            _all_tools[tool_name] = (env_key, env_cls)

    class MultiEnvironment:
        """Meta-environment that routes to the correct underlying environment."""

        def __init__(self) -> None:
            self._envs = {key: cls() for key, cls in environments.items()}
            self._active_key: str = _default
            self._active: BaseEnvironment = self._envs[_default]
            self.reward: float = 0.0
            self.done: bool = False
            self._execution_attempted: bool = False
            self._execution_succeeded: bool = False

        def reset(self, **kwargs: Any) -> str | None:
            """Route to the correct environment based on dataset column.

            Args:
                **kwargs: Dataset columns. Must include env_column for routing.
            """
            self._active_key = kwargs.pop(env_column, _default)
            if self._active_key not in self._envs:
                raise ValueError(
                    f"Unknown env '{self._active_key}'. Available: {list(environments.keys())}"
                )
            self._active = self._envs[self._active_key]
            result = self._active.reset(**kwargs)
            self.reward = 0.0
            self.done = False
            self._execution_attempted = False
            self._execution_succeeded = False
            return result

        def _sync_state(self) -> None:
            """Pull reward/done from active env."""
            self.reward = self._active.reward
            self.done = self._active.done
            if hasattr(self._active, "_execution_attempted"):
                self._execution_attempted = self._active._execution_attempted
            if hasattr(self._active, "_execution_succeeded"):
                self._execution_succeeded = self._active._execution_succeeded

    # Dynamically add tool methods from all environments
    for tool_name, (env_key, env_cls) in _all_tools.items():
        _add_routed_tool(MultiEnvironment, tool_name, environments)

    MultiEnvironment.__name__ = "MultiEnvironment"
    MultiEnvironment.__qualname__ = "MultiEnvironment"
    return MultiEnvironment


def _add_routed_tool(
    meta_cls: type,
    tool_name: str,
    environments: dict[str, Type[BaseEnvironment]],
) -> None:
    """Add a tool method to the meta-class that routes to the active env."""

    # Find a source method for the docstring and signature
    source_method = None
    for env_cls in environments.values():
        if hasattr(env_cls, tool_name):
            source_method = getattr(env_cls, tool_name)
            break

    if source_method is None:
        return

    import inspect
    sig = inspect.signature(source_method)

    # Build the delegating method
    def make_delegate(name: str):
        def delegate(self, *args, **kwargs):
            method = getattr(self._active, name, None)
            if method is None:
                return f"Error: Tool '{name}' not available in {self._active_key} environment"
            result = method(*args, **kwargs)
            self._sync_state()
            return result

        # Copy metadata for TRL tool discovery
        delegate.__name__ = name
        delegate.__qualname__ = f"MultiEnvironment.{name}"
        delegate.__doc__ = source_method.__doc__
        delegate.__annotations__ = source_method.__annotations__.copy()
        return delegate

    setattr(meta_cls, tool_name, make_delegate(tool_name))
