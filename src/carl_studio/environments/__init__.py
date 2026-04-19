"""RL environment adapters for CARL training.

Public API::

    from carl_studio.environments import (
        # Core protocol
        BaseEnvironment,
        EnvironmentConnection,
        EnvironmentSpec,
        EnvironmentLane,
        # Registry
        register_environment,
        get_environment,
        list_environments,
        from_hub,
        publish_to_hub,
        # Validation
        validate_environment,
        # OpenEnv bridge
        OpenEnvAction,
        OpenEnvObservation,
        OpenEnvServer,
        OpenEnvClient,
        # Verifiers bridge
        VerifiersAdapter,
        from_prime_verifier,
        # OpenReward managed reward API
        OpenRewardClient,
        # Builtins
        CodingSandboxEnv,
        SQLSandboxEnv,
    )
"""

from __future__ import annotations

from carl_studio.environments.connection import (
    DEFAULT_ENVIRONMENT_CONNECTION_SPEC,
    EnvironmentConnection,
)
from carl_studio.environments.openenv import (
    OpenEnvAction,
    OpenEnvClient,
    OpenEnvObservation,
    OpenEnvServer,
)
from carl_studio.environments.openreward import OpenRewardClient
from carl_studio.environments.protocol import (
    BaseEnvironment,
    EnvironmentLane,
    EnvironmentSpec,
)
from carl_studio.environments.registry import (
    clear_registry,
    from_hub,
    get_environment,
    list_environments,
    publish_to_hub,
    register_environment,
)
from carl_studio.environments.validation import validate_environment
from carl_studio.environments.verifiers import (
    VerifiersAdapter,
    from_prime_verifier,
)

# Import builtins to trigger @register_environment decorators.
# This must come AFTER registry imports to avoid circular imports.
# Use submodule paths directly to keep pyright's reportPrivateImportUsage happy.
from carl_studio.environments.builtins.code_sandbox import CodingSandboxEnv  # noqa: F401
from carl_studio.environments.builtins.sql_sandbox import SQLSandboxEnv  # noqa: F401

__all__ = [
    # Core protocol
    "BaseEnvironment",
    "EnvironmentConnection",
    "EnvironmentLane",
    "EnvironmentSpec",
    "DEFAULT_ENVIRONMENT_CONNECTION_SPEC",
    # Registry
    "register_environment",
    "get_environment",
    "list_environments",
    "clear_registry",
    "from_hub",
    "publish_to_hub",
    # Validation
    "validate_environment",
    # OpenEnv bridge
    "OpenEnvAction",
    "OpenEnvObservation",
    "OpenEnvServer",
    "OpenEnvClient",
    # Verifiers bridge
    "VerifiersAdapter",
    "from_prime_verifier",
    # OpenReward managed reward API
    "OpenRewardClient",
    # Builtins
    "CodingSandboxEnv",
    "SQLSandboxEnv",
]
