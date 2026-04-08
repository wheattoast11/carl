"""CARL Studio MCP Server -- exposes training tools for AI agents."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("carl-studio")


@mcp.tool()
async def start_training(config_yaml: str) -> str:
    """Start a CARL training run from a YAML config string.
    Returns JSON with run_id and initial status.
    """
    import json
    import yaml
    from carl_studio.types.config import TrainingConfig
    from carl_studio.training.trainer import CARLTrainer

    raw = yaml.safe_load(config_yaml) or {}
    config = TrainingConfig(**raw)
    trainer = CARLTrainer(config)
    run = await trainer.train()

    return json.dumps({
        "run_id": run.id,
        "phase": run.phase.value,
        "hub_job_id": run.hub_job_id,
        "model": config.base_model,
        "method": config.method.value,
    })


@mcp.tool()
async def get_run_status(run_id: str, backend: str = "hf_jobs") -> str:
    """Get current status of a training run.
    Returns JSON with phase, status.
    """
    import json
    from carl_studio.compute import get_backend

    b = get_backend(backend)
    status = await b.status(run_id)
    return json.dumps({"run_id": run_id, "status": status})


@mcp.tool()
async def get_logs(run_id: str, backend: str = "hf_jobs", tail: int = 30) -> str:
    """Get recent log lines from a training run."""
    import json
    from carl_studio.compute import get_backend

    b = get_backend(backend)
    lines = await b.logs(run_id, tail=tail)
    return json.dumps({"run_id": run_id, "lines": lines})


@mcp.tool()
async def observe_now(run_id: str) -> str:
    """Force a CoherenceObserver assessment on the current training state.
    Requires ANTHROPIC_API_KEY in environment.
    Returns JSON with status, diagnosis, signals, recommendations.
    """
    import json
    from carl_studio.primitives import CoherenceObserver

    observer = CoherenceObserver()
    assessment = observer.force_observe()
    return json.dumps(assessment)


@mcp.tool()
async def get_coherence_metrics(logits_summary: str) -> str:
    """Compute coherence metrics from a logits summary.
    Input: JSON with vocab_size, n_tokens.
    Returns JSON with KAPPA, SIGMA, T_STAR for the given embedding dimension.
    """
    import json
    from carl_studio.primitives.constants import KAPPA, SIGMA, T_STAR

    data = json.loads(logits_summary)
    d = data.get("embedding_dim", 3072)
    return json.dumps({
        "kappa": KAPPA,
        "sigma": SIGMA,
        "kappa_x_sigma": KAPPA * SIGMA,
        "t_star": T_STAR(d),
        "embedding_dim": d,
    })


@mcp.tool()
async def stop_training(run_id: str, backend: str = "hf_jobs") -> str:
    """Cancel a running training job."""
    import json
    from carl_studio.compute import get_backend

    b = get_backend(backend)
    await b.stop(run_id)
    return json.dumps({"run_id": run_id, "status": "stopped"})


@mcp.tool()
async def list_backends() -> str:
    """List available compute backends and their status."""
    import json

    backends = [
        {"name": "hf_jobs", "type": "managed", "description": "HuggingFace Jobs (PEP 723 UV scripts)"},
        {"name": "runpod", "type": "pods", "description": "RunPod GPU pods"},
        {"name": "tinker", "type": "managed", "description": "Tinker managed training API"},
        {"name": "prime", "type": "marketplace", "description": "Prime Intellect GPU marketplace"},
        {"name": "ssh", "type": "remote", "description": "SSH to remote machine"},
        {"name": "local", "type": "direct", "description": "Local GPU"},
    ]
    return json.dumps({"backends": backends})


@mcp.tool()
async def validate_config(config_yaml: str) -> str:
    """Validate a CARL training config without starting training.
    Returns JSON with valid=true/false and any validation errors.
    """
    import json
    import yaml
    from carl_studio.types.config import TrainingConfig

    try:
        raw = yaml.safe_load(config_yaml) or {}
        config = TrainingConfig(**raw)
        return json.dumps({
            "valid": True,
            "model": config.base_model,
            "method": config.method.value,
            "compute": config.compute_target.value,
        })
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


@mcp.tool()
async def generate_bundle(config_yaml: str) -> str:
    """Generate a self-contained training script for HF Jobs from a config.
    Returns the Python script as a string.
    """
    import yaml
    from carl_studio.types.config import TrainingConfig
    from carl_studio.bundler import Bundler

    raw = yaml.safe_load(config_yaml) or {}
    config = TrainingConfig(**raw)
    bundler = Bundler()
    script = bundler.generate(config)
    return script
