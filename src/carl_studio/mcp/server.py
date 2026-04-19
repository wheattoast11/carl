"""CARL Studio MCP Server -- exposes training tools for AI agents.

Tenancy model (UAT-049)
-----------------------
This MCP server is **single-tenant by design**. The ``_session`` dict below
(JWT, tier, user_id, authenticated_at) is a process-wide singleton — it is
*not* keyed per-client. If two MCP clients connect to the same server
process they will share the same auth state, and a JWT posted by one
client will be visible to every other client.

Operators MUST run **one server process per tenant/user**. Do not share a
single ``carl mcp serve`` process across multiple human or automation
identities. Multi-tenant deployments should provision a dedicated
process (or container) per principal and front them with their own
transport — the MCP SDK's ``Context`` object is the correct seam for
per-request state but retrofitting it across all nine authenticated
tools (plus the nine legacy tools already out in the wild) would be a
breaking change that we defer until the MCP 2025-11-25 migration.

The module emits a ``logger.warning`` on import so this constraint is
surfaced every time the server boots, regardless of log level defaults.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from carl_studio.mcp.connection import MCPServerConnection

logger = logging.getLogger(__name__)

mcp = FastMCP("carl-studio")

# ---------------------------------------------------------------------------
# Session state — module-level, persists for the MCP server process lifetime
# ---------------------------------------------------------------------------
#
# UAT-049: this dict is **single-tenant** — see the module docstring.
# Every authenticated tool (`authenticate`, `get_tier_status`,
# `dispatch_a2a_task`, `sync_data`, …) reads/writes this singleton.
# Attempting to serve multiple tenants from one process will leak auth
# state across clients. We emit a banner below so the constraint is
# visible at startup.

_session: dict[str, str] = {"jwt": "", "tier": "free", "user_id": ""}
_SESSION_MAX_AGE = 3600  # 1 hour — re-authenticate after this

logger.warning(
    "carl-studio MCP server loaded: session state is SINGLE-TENANT "
    "(module-level _session dict). Run one process per tenant — "
    "do not share this server across users or automations."
)

# ---------------------------------------------------------------------------
# Connection-primitive binding (Phase 1 port of the MCP surface)
# ---------------------------------------------------------------------------
#
# ``_connection`` is populated by :func:`bind_connection` once a
# MCPServerConnection has been ``open()``-ed. Every @mcp.tool()-decorated
# function below funnels success/failure telemetry through
# :func:`_record_tool_event`, which is a cheap no-op when nothing is bound.
# This keeps the existing in-process behavior identical while giving the
# Connection primitive real visibility into tool-call lifetimes.

_connection: "MCPServerConnection | None" = None

_ToolResult = TypeVar("_ToolResult")


def bind_connection(conn: "MCPServerConnection | None") -> None:
    """Attach (or detach, with ``None``) an MCPServerConnection for telemetry.

    Called by :class:`carl_studio.mcp.connection.MCPServerConnection` after
    ``open()``; the module-level ``_connection`` reference is then read by
    :func:`_record_tool_event` inside every tool body and by the
    elicitation/sampling helpers so callers do not need to pass
    ``connection=`` on every call.
    """
    global _connection
    _connection = conn


def get_bound_connection() -> "MCPServerConnection | None":
    """Return the currently-bound MCPServerConnection, or ``None``.

    Used by :mod:`carl_studio.mcp.elicitation` and
    :mod:`carl_studio.mcp.sampling` to reach the active connection from
    inside a tool body without threading the instance through every
    function signature.
    """
    return _connection


def _record_tool_event(
    tool_name: str,
    *,
    success: bool,
    duration_ms: float | None,
    error: str | None = None,
) -> None:
    """Emit a ``connection.tool.<tool_name>`` event on the bound chain.

    Cheap no-op when no connection is bound or the connection has no
    interaction chain attached.
    """
    if _connection is None:
        return
    ctx: dict[str, str] = {}
    if error is not None:
        ctx["error"] = error
    _connection.record_tool_event(
        tool_name,
        success=success,
        duration_ms=duration_ms,
        **ctx,
    )


async def _run_tool(
    tool_name: str,
    body: Callable[[], Awaitable[_ToolResult]],
) -> _ToolResult:
    """Run a tool body and record a telemetry event on the bound chain.

    Tools return JSON strings where the JSON may itself encode an error
    condition — we still count that as a successful *call* at the
    transport layer. Only uncaught exceptions from the body count as
    transport-level failures; tool bodies do their own error handling
    and already emit ``{"error": ...}`` payloads.
    """
    start = time.monotonic()
    try:
        result = await body()
    except BaseException as exc:
        duration_ms = (time.monotonic() - start) * 1000.0
        _record_tool_event(
            tool_name,
            success=False,
            duration_ms=duration_ms,
            error=repr(exc),
        )
        raise
    duration_ms = (time.monotonic() - start) * 1000.0
    _record_tool_event(tool_name, success=True, duration_ms=duration_ms)
    return result

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

    async def _body() -> str:
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

    return await _run_tool("start_training", _body)


@mcp.tool()
async def get_run_status(run_id: str, backend: str = "hf_jobs") -> str:
    """Get current status of a training run.
    Returns JSON with phase, status.
    """

    async def _body() -> str:
        from carl_studio.compute import get_backend

        b = get_backend(backend)
        status = await b.status(run_id)
        return json.dumps({"run_id": run_id, "status": status})

    return await _run_tool("get_run_status", _body)


@mcp.tool()
async def get_logs(run_id: str, backend: str = "hf_jobs", tail: int = 30) -> str:
    """Get recent log lines from a training run."""

    async def _body() -> str:
        from carl_studio.compute import get_backend

        b = get_backend(backend)
        lines = await b.logs(run_id, tail=tail)
        return json.dumps({"run_id": run_id, "lines": lines})

    return await _run_tool("get_logs", _body)


@mcp.tool()
async def observe_now(run_id: str) -> str:
    """Force a CoherenceObserver assessment on the current training state.
    Requires ANTHROPIC_API_KEY in environment.
    Returns JSON with status, diagnosis, signals, recommendations.
    """

    async def _body() -> str:
        from carl_core import CoherenceObserver

        observer = CoherenceObserver()
        assessment = observer.force_observe()
        return json.dumps(assessment)

    return await _run_tool("observe_now", _body)


@mcp.tool()
async def get_coherence_metrics(logits_summary: str) -> str:
    """Compute coherence metrics from a logits summary.
    Input: JSON with vocab_size, n_tokens.
    Returns JSON with KAPPA, SIGMA, T_STAR for the given embedding dimension.
    """

    async def _body() -> str:
        from carl_core.constants import KAPPA, SIGMA, T_STAR

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

    return await _run_tool("get_coherence_metrics", _body)


@mcp.tool()
async def stop_training(run_id: str, backend: str = "hf_jobs") -> str:
    """Cancel a running training job."""

    async def _body() -> str:
        from carl_studio.compute import get_backend

        b = get_backend(backend)
        await b.stop(run_id)
        return json.dumps({"run_id": run_id, "status": "stopped"})

    return await _run_tool("stop_training", _body)


@mcp.tool()
async def list_backends() -> str:
    """List available compute backends and their status."""

    async def _body() -> str:
        backends = [
            {
                "name": "hf_jobs",
                "type": "managed",
                "description": "HuggingFace Jobs (PEP 723 UV scripts)",
            },
            {"name": "runpod", "type": "pods", "description": "RunPod GPU pods"},
            {"name": "tinker", "type": "managed", "description": "Tinker managed training API"},
            {
                "name": "prime",
                "type": "marketplace",
                "description": "Prime Intellect GPU marketplace",
            },
            {"name": "ssh", "type": "remote", "description": "SSH to remote machine"},
            {"name": "local", "type": "direct", "description": "Local GPU"},
        ]
        return json.dumps({"backends": backends})

    return await _run_tool("list_backends", _body)


@mcp.tool()
async def validate_config(config_yaml: str) -> str:
    """Validate a CARL training config without starting training.
    Returns JSON with valid=true/false and any validation errors.
    """

    async def _body() -> str:
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

    return await _run_tool("validate_config", _body)


@mcp.tool()
async def generate_bundle(config_yaml: str) -> str:
    """Generate a self-contained training script for HF Jobs from a config.
    Returns the Python script as a string.
    """

    async def _body() -> str:
        import yaml
        from carl_studio.types.config import TrainingConfig
        from carl_studio.bundler import Bundler

        raw = yaml.safe_load(config_yaml) or {}
        config = TrainingConfig(**raw)
        bundler = Bundler()
        script = bundler.generate(config)
        return script

    return await _run_tool("generate_bundle", _body)


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

    async def _body() -> str:
        global _session

        if not jwt or not isinstance(jwt, str):
            return json.dumps(
                {"error": "jwt must be a non-empty string", "authenticated": False}
            )

        try:
            # Server-side JWT verification via Supabase Edge Function
            from carl_studio.camp import (
                DEFAULT_CARL_CAMP_SUPABASE_URL,
                fetch_camp_profile,
            )
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

            from carl_studio.tier import FEATURE_TIERS, Tier, tier_allows

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
            return json.dumps(
                {
                    "error": f"Server verification failed: {e}",
                    "authenticated": False,
                    "hint": (
                        "Ensure carl.camp is reachable, or check your JWT "
                        "with 'carl camp account'"
                    ),
                }
            )

    return await _run_tool("authenticate", _body)


# ---------------------------------------------------------------------------
# New Tool 2: get_tier_status
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_tier_status() -> str:
    """Get current session tier and which features are accessible.

    Returns: JSON with {tier, authenticated, features: {name: allowed}}.
    Shows 'free' if not authenticated — no JWT required.
    """

    async def _body() -> str:
        from carl_studio.tier import FEATURE_TIERS, Tier, tier_allows

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

    return await _run_tool("get_tier_status", _body)


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

    async def _body() -> str:
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

    return await _run_tool("get_project_state", _body)


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

    async def _body() -> str:
        if not _skills_available:
            return json.dumps(
                {"error": "Skills not installed. pip install carl-studio[skills]"}
            )

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
            return json.dumps(
                {"error": str(e), "skill_name": skill_name, "success": False}
            )

    return await _run_tool("run_skill", _body)


# ---------------------------------------------------------------------------
# New Tool 5: list_skills
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_skills() -> str:
    """List all available CARL skills with their descriptions and tier requirements.

    Returns: JSON with {skills: [{name, badge, description, requires_tier}]}.
    """

    async def _body() -> str:
        if not _skills_available:
            return json.dumps(
                {"error": "Skills not installed. pip install carl-studio[skills]"}
            )

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

    return await _run_tool("list_skills", _body)


# ---------------------------------------------------------------------------
# New Tool 6: get_agent_card
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_agent_card() -> str:
    """Get the CARL Agent Card — capability manifest for A2A integration.

    Returns: JSON CARLAgentCard describing this agent's capabilities,
    skills, tier, and endpoint.
    """

    async def _body() -> str:
        from carl_studio.a2a.agent_card import CARLAgentCard

        card = CARLAgentCard.current()
        return card.to_json()

    return await _run_tool("get_agent_card", _body)


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

    async def _body() -> str:
        if not _require_tier("orchestration"):
            return _tier_error("orchestration")

        try:
            inputs = json.loads(inputs_json) if inputs_json else {}
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid inputs_json: {e}"})

        try:
            import uuid

            from carl_studio.a2a.bus import LocalBus
            from carl_studio.a2a.task import A2ATask

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

    return await _run_tool("dispatch_a2a_task", _body)


# ---------------------------------------------------------------------------
# New Tool 8: sync_data
# ---------------------------------------------------------------------------


@mcp.tool()
async def sync_data(direction: str = "push", entity_types: str = "runs") -> str:
    """Sync local data with carl.camp. direction: push | pull.

    Requires PAID tier and carl camp login (JWT in session or local DB).
    Returns: JSON with sync results.
    """

    async def _body() -> str:
        if not _require_tier("sync.cloud"):
            return _tier_error("sync.cloud")

        if direction not in ("push", "pull"):
            return json.dumps(
                {"error": f"Invalid direction '{direction}'. Use 'push' or 'pull'."}
            )

        # Parse entity_types — comma-separated or single
        types_list = [t.strip() for t in entity_types.split(",") if t.strip()] or ["runs"]

        # Inject the session JWT into LocalDB auth cache if present.
        # UAT-051: silent ``pass`` used to hide permission/disk errors.
        # Keep the "never break sync" semantics but leave operators a
        # breadcrumb — ``sync.py`` still falls back to reading the JWT
        # directly from the DB, so the outer sync call can continue.
        jwt = _session.get("jwt", "")
        if jwt:
            try:
                from carl_studio.db import LocalDB

                db = LocalDB()
                db.set_auth("jwt", jwt, ttl_hours=24)
                db.close()
            except Exception as exc:
                logger.debug("JWT cache-write failed: %s", exc)

        try:
            from carl_studio.sync import pull, push

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

    return await _run_tool("sync_data", _body)


# ---------------------------------------------------------------------------
# MCP 2025-11-25 — async task surface
# ---------------------------------------------------------------------------
#
# ``submit_async_training`` mirrors the sync ``start_training`` body but
# returns a task handle immediately. The handle is pollable via the
# ``tasks_get`` tool and cancellable via ``tasks_cancel`` — both of which
# are registered at import time by ``register_task_tools`` below.

from carl_studio.mcp.tasks import async_task, register_task_tools  # noqa: E402
from carl_studio.mcp.output_schemas import register_output_schemas  # noqa: E402


async def _start_training_body(config_yaml: str) -> dict[str, object]:
    """Shared training body — reused by sync ``start_training`` and the async version."""
    import yaml
    from carl_studio.types.config import TrainingConfig
    from carl_studio.training.trainer import CARLTrainer

    raw: dict[str, object] = yaml.safe_load(config_yaml) or {}
    config = TrainingConfig(**raw)  # type: ignore[arg-type]
    trainer = CARLTrainer(config)
    run = await trainer.train()
    return {
        "run_id": run.id,
        "phase": run.phase.value,
        "hub_job_id": run.hub_job_id,
        "model": config.base_model,
        "method": config.method.value,
    }


@mcp.tool()
@async_task("submit_async_training")
async def submit_async_training(config_yaml: str) -> dict[str, object]:
    """Submit a CARL training run asynchronously.

    Returns a task handle immediately. Poll ``tasks_get(task_id)`` to
    observe progress; the completed task's ``result`` matches the sync
    ``start_training`` shape (``run_id``, ``phase``, ``hub_job_id``,
    ``model``, ``method``).
    """
    return await _start_training_body(config_yaml)


# Register the MCP 2025-11-25 task polling/cancellation tools on this
# FastMCP instance. Legacy synchronous tools keep their original
# decorators; the new tools live alongside them in the same registry.
register_task_tools(mcp)

# Attach output schemas to every registered tool (legacy + new).
register_output_schemas(mcp)
