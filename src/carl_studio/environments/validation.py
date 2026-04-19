"""Validate environment classes for TRL environment_factory compatibility.

TRL auto-discovers tools from public methods. This module checks that
an environment class meets all TRL requirements before a training job
is submitted (fail fast locally, not on a $2.50/hr GPU).
"""

from __future__ import annotations

import inspect
from typing import Type

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentSpec


def validate_environment(cls: Type[BaseEnvironment]) -> list[str]:
    """Check environment class for TRL environment_factory compatibility.

    Returns a list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    # 1. Must have spec
    if not hasattr(cls, "spec") or not isinstance(getattr(cls, "spec", None), EnvironmentSpec):
        errors.append("Missing class attribute `spec: EnvironmentSpec`")

    # 2. __init__ must take no arguments (TRL creates instances with no args)
    init_sig = inspect.signature(cls.__init__)
    params = [p for p in init_sig.parameters.values() if p.name != "self"]
    required = [p for p in params if p.default is inspect.Parameter.empty and p.kind not in (
        inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
    )]
    if required:
        errors.append(f"__init__ has required parameters beyond self: {[p.name for p in required]}. TRL passes no args.")

    # 3. Must have reset()
    if not hasattr(cls, "reset") or not callable(getattr(cls, "reset")):
        errors.append("Missing reset() method")
    else:
        reset_sig = inspect.signature(cls.reset)
        if "kwargs" not in str(reset_sig):
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in reset_sig.parameters.values()
            )
            if not has_var_keyword:
                errors.append("reset() must accept **kwargs (TRL passes dataset columns)")

    # 4. Must have at least one public tool method
    tool_methods = _get_tool_methods(cls)
    if not tool_methods:
        errors.append("No public tool methods found (methods not starting with '_', excluding 'reset')")

    # 5. Each tool must have type hints and docstring with Args:
    for name, method in tool_methods:
        if not method.__doc__:
            errors.append(f"Tool '{name}' has no docstring (TRL uses it for tool description)")
        elif "Args:" not in method.__doc__:
            errors.append(f"Tool '{name}' docstring missing 'Args:' section (TRL parses it)")

        sig = inspect.signature(method)
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.annotation is inspect.Parameter.empty:
                errors.append(f"Tool '{name}' parameter '{pname}' has no type hint (TRL needs it)")

    # 6. If spec exists, check declared tools match actual tools
    if hasattr(cls, "spec") and isinstance(cls.spec, EnvironmentSpec):
        declared = set(cls.spec.tools)
        actual = {name for name, _ in tool_methods}
        missing = declared - actual
        extra = actual - declared
        if missing:
            errors.append(f"Spec declares tools not found as methods: {missing}")
        if extra:
            errors.append(f"Public methods not declared in spec.tools: {extra}")

    return errors


def _get_tool_methods(cls: type) -> list[tuple[str, object]]:
    """Get public methods that TRL would discover as tools."""
    # BaseEnvironment contract surface.
    env_excluded = {"reset", "turn_count", "history", "done", "reward", "spec"}
    # EnvironmentConnection connection surface — these are lifecycle
    # / FSM / telemetry helpers inherited from the connection primitive,
    # not TRL tool methods.
    conn_excluded = {
        "open",
        "close",
        "transact",
        "require_ready",
        "state",
        "stats",
        "connection_id",
        "connection_spec",
        "chain",
        "attach_chain",
        "to_dict",
    }
    excluded = env_excluded | conn_excluded
    methods = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if name in excluded:
            continue
        attr = getattr(cls, name)
        if callable(attr) and not isinstance(attr, property):
            methods.append((name, attr))
    return methods
