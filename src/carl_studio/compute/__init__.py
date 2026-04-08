"""Compute backends for CARL training."""

from carl_studio.compute.protocol import ComputeBackend


def get_backend(name: str) -> ComputeBackend:
    """Get a compute backend by name. Lazy imports."""
    if name == "hf_jobs":
        from carl_studio.compute.hf_jobs import HFJobsBackend

        return HFJobsBackend()
    elif name == "runpod":
        from carl_studio.compute.runpod import RunPodBackend

        return RunPodBackend()
    elif name == "tinker":
        from carl_studio.compute.tinker import TinkerBackend

        return TinkerBackend()
    elif name == "prime":
        from carl_studio.compute.prime import PrimeBackend

        return PrimeBackend()
    elif name == "ssh":
        from carl_studio.compute.ssh import SSHBackend

        return SSHBackend()
    elif name == "local":
        from carl_studio.compute.local import LocalBackend

        return LocalBackend()
    else:
        raise ValueError(f"Unknown backend: {name}. Available: hf_jobs, runpod, tinker, prime, ssh, local")


__all__ = ["ComputeBackend", "get_backend"]
