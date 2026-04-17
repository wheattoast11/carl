"""CARL Studio MCP Server -- exposes training tools for AI agents."""

from __future__ import annotations

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("carl-studio")

# ---------------------------------------------------------------------------
# Session state — module-level, persists for the MCP server process lifetime
# ---------------------------------------------------------------------------

_session: dict[str, str] = {"jwt": "", "tier": "free", "user_id": ""}
_SESSION_MAX_AGE = 3600  # 1 hour — re-authenticate after this

# ---------------------------------------------------------------------------
# Skills availability — graceful degradation if optional deps missing
# ---------------------------------------------------------------------------

try:
    from carl_studio.skills.runner import SkillRunner, SkillRegistry
    from carl_studio.skills.builtins import BUILTIN_SKILLS

    _skills_available = True
except ImportError:
    _skills_available = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_tier(feature: str) -> bool:
    """Return True if current session tier allows the feature.

    Also enforces session age — sessions older than ``_SESSION_MAX_AGE``
    seconds are treated as unauthenticated (free tier).
    """
    import time

    from carl_studio.tier import Tier, tier_allows

    # Expire stale sessions — force re-authentication
    auth_at = _session.get("authenticated_at")
    if auth_at:
        try:
            if time.time() - int(auth_at) > _SESSION_MAX_AGE:
                _session["tier"] = "free"
                _session["jwt"] = ""
                _session["authenticated_at"] = ""
        except (ValueError, TypeError):
            pass

    t = Tier(_session.get("tier", "free"))
    return tier_allows(t, feature)


def _tier_error(feature: str) -> str:
    """Return standard JSON error for tier gate failures."""
    return json.dumps(
        {
            "error": f"Feature '{feature}' requires CARL Paid tier.",
            "current_tier": _session.get("tier", "free"),
            "upgrade": "https://carl.camp/pricing",
            "hint": "Call authenticate(jwt) first, or upgrade at carl.camp",
        }
    )


def _build_skill_runner() -> "SkillRunner":
    """Build a SkillRunner with all builtins registered."""
    runner = SkillRunner()  # type: ignore[possibly-undefined]
    for skill in BUILTIN_SKILLS:  # type: ignore[possibly-undefined]
        runner.register(skill)
    return runner


# ---------------------------------------------------------------------------
# Original 8 tools — unchanged
# ---------------------------------------------------------------------------


@mcp.tool()
async def start_training(config_yaml: str) -> str:
    """Start a CARL training run from a YAML config string.
    Returns JSON with run_id and initial status.
    """
    import yaml
    from carl_studio.types.config import TrainingConfig
    from carl_studio.training.trainer import CARLTrainer

    raw = yaml.safe_load(config_yaml) or {}
    config = TrainingConfig(**raw)
    trainer = CARLTrainer(config)
    run = await trainer.train()

    return json.dumps(
        {
            "run_id": run.id,
            "phase": run.phase.value,
            "hub_job_id": run.hub_job_id,
            "model": config.base_model,
            "method": config.method.value,
        }
    )


@mcp.tool()
async def get_run_status(run_id: str, backend: str = "hf_jobs") -> str:
    """Get current status of a training run.
    Returns JSON with phase, status.
    """
    from carl_studio.compute import get_backend

    b = get_backend(backend)
    status = await b.status(run_id)
    return json.dumps({"run_id": run_id, "status": status})


@mcp.tool()
async def get_logs(run_id: str, backend: str = "hf_jobs", tail: int = 30) -> str:
    """Get recent log lines from a training run."""
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
    from carl_studio.primitives.constants import KAPPA, SIGMA, T_STAR

    data = json.loads(logits_summary)
    d = data.get("embedding_dim", 3072)
    return json.dumps(
        {
            "kappa": KAPPA,
            "sigma": SIGMA,
            "kappa_x_sigma": KAPPA * SIGMA,
            "t_star": T_STAR(d),
            "embedding_dim": d,
        }
    )


@mcp.tool()
async def stop_training(run_id: str, backend: str = "hf_jobs") -> str:
    """Cancel a running training job."""
    from carl_studio.compute import get_backend

    b = get_backend(backend)
    await b.stop(run_id)
    return json.dumps({"run_id": run_id, "status": "stopped"})


@mcp.tool()
async def list_backends() -> str:
    """List available compute backends and their status."""
    backends = [
        {
            "name": "hf_jobs",
            "type": "managed",
            "description": "HuggingFace Jobs (PEP 723 UV scripts)",
        },
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
    import yaml
    from carl_studio.types.config import TrainingConfig

    try:
        raw = yaml.safe_load(config_yaml) or {}
        config = TrainingConfig(**raw)
        return json.dumps(
            {
                "valid": True,
                "model": config.base_model,
                "method": config.method.value,
                "compute": config.compute_target.value,
            }
        )
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


# ---------------------------------------------------------------------------
# New Tool 1: authenticate
# ---------------------------------------------------------------------------


@mcp.tool()
async def authenticate(jwt: str) -> str:
    """Authenticate this MCP session with a carl.camp JWT.

    Call this first before using PAID features.
    The JWT is verified server-side via the carl.camp Supabase Edge Function.
    Returns: JSON with {tier, user_id, features_available: []}.
    """
    global _session
    import time

    if not jwt or not isinstance(jwt, str):
        return json.dumps({"error": "jwt must be a non-empty string", "authenticated": False})

    try:
        # Server-side JWT verification via Supabase Edge Function
        from carl_studio.camp import fetch_camp_profile, DEFAULT_CARL_CAMP_SUPABASE_URL
        from carl_studio.settings import CARLSettings

        settings = CARLSettings.load()
        supabase_url = settings.supabase_url or DEFAULT_CARL_CAMP_SUPABASE_URL

        profile = fetch_camp_profile(jwt, supabase_url)

        # CampProfile.tier is a plain str — normalize to canonical values
        tier_raw = profile.tier
        tier = "paid" if tier_raw in ("paid", "pro", "enterprise") else "free"
        user_id = profile.user_id or ""

        _session["jwt"] = jwt
        _session["tier"] = tier
        _session["user_id"] = str(user_id)
        _session["authenticated_at"] = str(int(time.time()))

        from carl_studio.tier import Tier, FEATURE_TIERS, tier_allows

        t = Tier(tier)
        features = [f for f, _req in FEATURE_TIERS.items() if tier_allows(t, f)]

        return json.dumps(
            {
                "tier": tier,
                "user_id": str(user_id),
                "features_available": features,
                "authenticated": True,
            }
        )
    except Exception as e:
        # Fail closed — do NOT store the unverified JWT or grant paid access
        _session["jwt"] = ""
        _session["tier"] = "free"
        _session["user_id"] = ""
        return json.dumps({
            "error": f"Server verification failed: {e}",
            "authenticated": False,
            "hint": "Ensure carl.camp is reachable, or check your JWT with 'carl camp account'",
        })


# ---------------------------------------------------------------------------
# New Tool 2: get_tier_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_tier_status() -> str:
    """Get current session tier and which features are accessible.

    Returns: JSON with {tier, authenticated, features: {name: allowed}}.
    Shows 'free' if not authenticated — no JWT required.
    """
    from carl_studio.tier import Tier, FEATURE_TIERS, tier_allows

    tier = _session.get("tier", "free")
    authenticated = bool(_session.get("jwt", ""))
    t = Tier(tier)
    features = {f: tier_allows(t, f) for f in FEATURE_TIERS}

    return json.dumps(
        {
            "tier": tier,
            "authenticated": authenticated,
            "user_id": _session.get("user_id", ""),
            "features": features,
        }
    )


# ---------------------------------------------------------------------------
# New Tool 3: get_project_state
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_project_state(config_path: str = "carl.yaml") -> str:
    """Get full project state: config, last run, last eval, pending sync count.

    Reads carl.yaml from the current directory (or config_path).
    Returns: JSON with {project, last_run, last_eval, pending_sync}.
    No tier required.
    """
    from pathlib import Path

    project_data: dict = {}
    try:
        from carl_studio.project import load_project

        project = load_project(Path(config_path))
        project_data = project.model_dump()
    except FileNotFoundError:
        project_data = {"error": f"Config file not found: {config_path}"}
    except Exception as e:
        project_data = {"error": str(e)}

    last_run: dict | None = None
    last_eval: dict | None = None
    pending_sync_count: int = 0
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        runs = db.list_runs(limit=1)
        last_run = runs[0] if runs else None
        # Eval is stored as a run with mode="eval"
        eval_runs = db.list_runs(limit=1, status=None)
        for r in eval_runs:
            if r.get("mode") == "eval":
                last_eval = r
                break
        pending = db.get_pending_sync()
        pending_sync_count = len(pending)
        db.close()
    except Exception as e:
        last_run = {"error": str(e)}

    return json.dumps(
        {
            "project": project_data,
            "last_run": last_run,
            "last_eval": last_eval,
            "pending_sync": pending_sync_count,
        },
        default=str,
    )


# ---------------------------------------------------------------------------
# New Tool 4: run_skill
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_skill(skill_name: str, inputs_json: str = "{}") -> str:
    """Run a CARL skill by name and return the result.

    Available skills: observer, grader, trainer, synthesizer, deployer.
    inputs_json: JSON string of keyword arguments for the skill.

    Requires PAID tier for grader (full eval) and trainer (job submission).
    Returns: JSON SkillResult.
    """
    if not _skills_available:
        return json.dumps({"error": "Skills not installed. pip install carl-studio[skills]"})

    try:
        inputs = json.loads(inputs_json) if inputs_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid inputs_json: {e}"})

    runner = _build_skill_runner()
    skill = runner.get(skill_name)
    if skill is None:
        available = [s.name for s in runner.list_skills()]
        runner.close()
        return json.dumps(
            {
                "error": f"Skill '{skill_name}' not found.",
                "available": available,
            }
        )

    # Tier gate for paid skills
    requires_tier = getattr(skill, "requires_tier", "free")
    if requires_tier == "paid" and not _require_tier("experiment"):
        runner.close()
        return _tier_error(f"skill:{skill_name}")

    try:
        result = runner.run(skill_name, **inputs)
        runner.close()
        return result.model_dump_json()
    except Exception as e:
        runner.close()
        return json.dumps({"error": str(e), "skill_name": skill_name, "success": False})


# ---------------------------------------------------------------------------
# New Tool 5: list_skills
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_skills() -> str:
    """List all available CARL skills with their descriptions and tier requirements.

    Returns: JSON with {skills: [{name, badge, description, requires_tier}]}.
    """
    if not _skills_available:
        return json.dumps({"error": "Skills not installed. pip install carl-studio[skills]"})

    runner = _build_skill_runner()
    skills_list = [
        {
            "name": s.name,
            "badge": s.badge,
            "description": s.description,
            "requires_tier": s.requires_tier,
        }
        for s in runner.list_skills()
    ]
    runner.close()
    return json.dumps({"skills": skills_list})


# ---------------------------------------------------------------------------
# New Tool 6: get_agent_card
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_agent_card() -> str:
    """Get the CARL Agent Card — capability manifest for A2A integration.

    Returns: JSON CARLAgentCard describing this agent's capabilities,
    skills, tier, and endpoint.
    """
    from carl_studio.a2a.agent_card import CARLAgentCard

    card = CARLAgentCard.current()
    return card.to_json()


# ---------------------------------------------------------------------------
# New Tool 7: dispatch_a2a_task
# ---------------------------------------------------------------------------


@mcp.tool()
async def dispatch_a2a_task(
    skill_name: str,
    inputs_json: str = "{}",
    sender: str = "external",
) -> str:
    """Dispatch an A2A task to the CARL agent's local bus.

    The task will be executed when 'carl agent run' is called locally.
    Requires PAID tier.
    Returns: JSON with {task_id, status: "pending"}.
    """
    if not _require_tier("orchestration"):
        return _tier_error("orchestration")

    try:
        inputs = json.loads(inputs_json) if inputs_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid inputs_json: {e}"})

    try:
        from carl_studio.a2a.bus import LocalBus
        from carl_studio.a2a.task import A2ATask
        import uuid

        task = A2ATask(
            id=str(uuid.uuid4()),
            skill=skill_name,
            inputs=inputs,
            sender=sender,
            receiver="carl-studio",
        )
        bus = LocalBus()
        task_id = bus.post(task)
        bus.close()

        return json.dumps(
            {
                "task_id": task_id,
                "status": "pending",
                "skill": skill_name,
                "sender": sender,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e), "task_id": None, "status": "failed"})


# ---------------------------------------------------------------------------
# New Tool 8: sync_data
# ---------------------------------------------------------------------------


@mcp.tool()
async def sync_data(direction: str = "push", entity_types: str = "runs") -> str:
    """Sync local data with carl.camp. direction: push | pull.

    Requires PAID tier and carl camp login (JWT in session or local DB).
    Returns: JSON with sync results.
    """
    if not _require_tier("sync.cloud"):
        return _tier_error("sync.cloud")

    if direction not in ("push", "pull"):
        return json.dumps({"error": f"Invalid direction '{direction}'. Use 'push' or 'pull'."})

    # Parse entity_types — comma-separated or single
    types_list = [t.strip() for t in entity_types.split(",") if t.strip()] or ["runs"]

    # Inject the session JWT into LocalDB auth cache if present
    jwt = _session.get("jwt", "")
    if jwt:
        try:
            from carl_studio.db import LocalDB

            db = LocalDB()
            db.set_auth("jwt", jwt, ttl_hours=24)
            db.close()
        except Exception:
            pass  # Best-effort — sync.py will read from DB directly

    try:
        from carl_studio.sync import push, pull

        if direction == "push":
            results = push(entity_types=types_list)
        else:
            results = pull()

        return json.dumps(
            {
                "direction": direction,
                "entity_types": types_list,
                "results": results,
                "status": "ok",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "direction": direction,
                "entity_types": types_list,
                "error": str(e),
                "status": "failed",
            }
        )
