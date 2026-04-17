# Compute

Backend abstraction for provisioning and executing CARL training jobs on any compute provider.

## Key Abstractions

- `ComputeBackend` -- Runtime-checkable Protocol that all backends implement: `provision()`, `execute()`, `status()`, `logs()`, `stop()`, `teardown()`.
- `get_backend(name)` -- Factory with lazy imports. Returns a backend instance by name.
- `LocalBackend` -- Runs training scripts locally via subprocess.
- `HFJobsBackend` -- Submits to Hugging Face Jobs (user provides HF token).
- `RunPodBackend` -- Provisions RunPod serverless GPU pods.
- `TinkerBackend` -- Tinker compute integration.
- `PrimeBackend` -- Prime Intellect compute.
- `SSHBackend` -- Remote execution over SSH.

All BYOK (bring-your-own-key) backends ship with carl-studio. The `camp` backend is an optional dependency (`pip install carlcamp`) for carl.camp managed infrastructure.

## Architecture

The trainer (`CARLTrainer`) receives a `ComputeTarget` from config, maps it to a backend name, and calls `get_backend()`. The same training config, rewards, and cascade logic run identically on all backends -- only provisioning and job submission differ.

## Usage

```python
from carl_studio.compute import get_backend

backend = get_backend("local")
job_id = await backend.execute(script)
status = await backend.status(job_id)
```
