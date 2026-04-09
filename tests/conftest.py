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

    _light_modules = {
        "carl_studio.types.config": "src/carl_studio/types/config.py",
        "carl_studio.types.reward": "src/carl_studio/types/reward.py",
        "carl_studio.types.run": "src/carl_studio/types/run.py",
        "carl_studio.tier": "src/carl_studio/tier.py",
        "carl_studio.settings": "src/carl_studio/settings.py",
        "carl_studio.theme": "src/carl_studio/theme.py",
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


# Run stub before any test imports
_stub_carl_studio_init()
