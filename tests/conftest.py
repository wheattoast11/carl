"""Test configuration for carl-studio.

Patches around the carl_studio.__init__ issue where _get_gate() triggers
transformers import. Tests that need settings/tier can run without torch/transformers.
"""

import sys
import types


def _stub_carl_studio_init():
    """Pre-load carl_studio package stub before __init__.py triggers transformers.

    Only applies if carl_studio hasn't been successfully imported yet.
    """
    if "carl_studio" in sys.modules:
        # Check if it loaded successfully (has __version__)
        mod = sys.modules["carl_studio"]
        if hasattr(mod, "__version__"):
            return  # Already loaded fine

    # Create stub package
    pkg = types.ModuleType("carl_studio")
    pkg.__path__ = ["src/carl_studio"]
    pkg.__version__ = "0.3.0"
    pkg.__all__ = []
    sys.modules["carl_studio"] = pkg

    # Create stub types subpackage
    types_pkg = types.ModuleType("carl_studio.types")
    types_pkg.__path__ = ["src/carl_studio/types"]
    sys.modules["carl_studio.types"] = types_pkg

    # Load real submodules that don't need transformers
    import importlib.util

    # Stub intermediate subpackages so their children can be imported
    _subpackages = {
        "carl_studio.primitives": "src/carl_studio/primitives",
        "carl_studio.environments": "src/carl_studio/environments",
        "carl_studio.environments.builtins": "src/carl_studio/environments/builtins",
        "carl_studio.compute": "src/carl_studio/compute",
        "carl_studio.observe": "src/carl_studio/observe",
        "carl_studio.eval": "src/carl_studio/eval",
        "carl_studio.data": "src/carl_studio/data",
        "carl_studio.data.adapters": "src/carl_studio/data/adapters",
        "carl_studio.training": "src/carl_studio/training",
        "carl_studio.training.rewards": "src/carl_studio/training/rewards",
    }
    for name, path in _subpackages.items():
        if name not in sys.modules:
            sub = types.ModuleType(name)
            sub.__path__ = [path]
            sys.modules[name] = sub

    _light_modules = {
        "carl_studio.types.config": "src/carl_studio/types/config.py",
        "carl_studio.types.reward": "src/carl_studio/types/reward.py",
        "carl_studio.types.run": "src/carl_studio/types/run.py",
        "carl_studio.tier": "src/carl_studio/tier.py",
        "carl_studio.settings": "src/carl_studio/settings.py",
        "carl_studio.theme": "src/carl_studio/theme.py",
        "carl_studio.environments.protocol": "src/carl_studio/environments/protocol.py",
        "carl_studio.environments.registry": "src/carl_studio/environments/registry.py",
        "carl_studio.environments.validation": "src/carl_studio/environments/validation.py",
        "carl_studio.environments.builtins.code_sandbox": "src/carl_studio/environments/builtins/code_sandbox.py",
        "carl_studio.environments.builtins.sql_sandbox": "src/carl_studio/environments/builtins/sql_sandbox.py",
        "carl_studio.compute.protocol": "src/carl_studio/compute/protocol.py",
        "carl_studio.compute.hf_jobs": "src/carl_studio/compute/hf_jobs.py",
        "carl_studio.observe.data_source": "src/carl_studio/observe/data_source.py",
        "carl_studio.eval.runner": "src/carl_studio/eval/runner.py",
    }

    for name, path in _light_modules.items():
        if name not in sys.modules:
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[name] = mod
                    spec.loader.exec_module(mod)
            except Exception:
                pass  # Module may not exist or have other deps

    # Load __init__.py for subpackages that export public API
    # (e.g. carl_studio.compute.get_backend lives in __init__.py)
    _init_modules = {
        "carl_studio.compute": "src/carl_studio/compute/__init__.py",
        "carl_studio.observe": "src/carl_studio/observe/__init__.py",
    }
    for name, path in _init_modules.items():
        if name in sys.modules:
            try:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec and spec.loader:
                    existing = sys.modules[name]
                    spec.loader.exec_module(existing)
            except Exception:
                pass


# Run stub before any test imports
_stub_carl_studio_init()
