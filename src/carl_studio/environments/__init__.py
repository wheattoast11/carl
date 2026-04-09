"""RL environment adapters for CARL training.

Public API::

    from carl_studio.environments import (
        BaseEnvironment,
        EnvironmentSpec,
        EnvironmentLane,
        register_environment,
        get_environment,
        list_environments,
        validate_environment,
        CodingSandboxEnv,
        SQLSandboxEnv,
    )
"""

from carl_studio.environments.protocol import (
    BaseEnvironment,
    EnvironmentLane,
    EnvironmentSpec,
)
from carl_studio.environments.registry import (
    clear_registry,
    get_environment,
    list_environments,
    register_environment,
)
from carl_studio.environments.validation import validate_environment

# Import builtins to trigger @register_environment decorators.
# This must come AFTER registry imports to avoid circular imports.
from carl_studio.environments.builtins import CodingSandboxEnv, SQLSandboxEnv  # noqa: F401

__all__ = [
    "BaseEnvironment",
    "EnvironmentLane",
    "EnvironmentSpec",
    "register_environment",
    "get_environment",
    "list_environments",
    "clear_registry",
    "validate_environment",
    "CodingSandboxEnv",
    "SQLSandboxEnv",
]
