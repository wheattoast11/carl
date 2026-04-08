"""Compute backends for CARL training.

All backends implement the ComputeBackend protocol. BYOK backends (hf_jobs, runpod,
tinker, prime, ssh, local) ship with carl-studio and require the user's own credentials.

The "camp" backend is the carl.camp managed service — it handles compute provisioning,
monitoring, and continuous alignment. It requires a carl.camp account and is loaded
from an optional dependency. The camp backend adds value through managed infrastructure,
not through different training code — the same CARL rewards, cascade, and environment
run on all backends.
"""

from carl_studio.compute.protocol import ComputeBackend

# BYOK backends — always available, user provides credentials
_BYOK_BACKENDS = {
    "hf_jobs": ("carl_studio.compute.hf_jobs", "HFJobsBackend"),
    "runpod": ("carl_studio.compute.runpod", "RunPodBackend"),
    "tinker": ("carl_studio.compute.tinker", "TinkerBackend"),
    "prime": ("carl_studio.compute.prime", "PrimeBackend"),
    "ssh": ("carl_studio.compute.ssh", "SSHBackend"),
    "local": ("carl_studio.compute.local", "LocalBackend"),
}


def get_backend(name: str) -> ComputeBackend:
    """Get a compute backend by name. Lazy imports.

    BYOK backends (hf_jobs, runpod, tinker, prime, ssh, local) are always available.
    The "camp" backend requires a carl.camp account and the carlcamp package.
    """
    if name in _BYOK_BACKENDS:
        module_path, class_name = _BYOK_BACKENDS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()

    if name == "camp":
        try:
            from carlcamp.compute import CampBackend  # type: ignore[import-not-found]
            return CampBackend()
        except ImportError:
            raise ImportError(
                "The 'camp' backend requires a carl.camp account.\n"
                "  Install: pip install carlcamp\n"
                "  Sign up: https://carl.camp\n\n"
                "For self-hosted training, use any BYOK backend:\n"
                f"  Available: {', '.join(_BYOK_BACKENDS)}"
            ) from None

    available = list(_BYOK_BACKENDS) + ["camp"]
    raise ValueError(f"Unknown backend: {name}. Available: {', '.join(available)}")


__all__ = ["ComputeBackend", "get_backend"]
