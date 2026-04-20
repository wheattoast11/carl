"""CARL Studio MCP Server -- exposes training tools for AI agents.

Session ownership (H2, v0.7.1)
------------------------------
Auth state (JWT, tier, user_id, authenticated_at) is **per-connection**.
Each :class:`~carl_studio.mcp.connection.MCPServerConnection` owns an
immutable :class:`~carl_studio.mcp.session.MCPSession`; tools read it via
:func:`_get_session` and mutate it by building a new snapshot and
assigning it back through the connection's :attr:`session` setter.

Before H2 this module owned a process-wide ``_session: dict[str, str]``
global (UAT-049). That singleton leaked auth across clients whenever two
callers shared the same ``carl mcp serve`` process. The per-connection
routing below closes that leak. We still rely on the operator running the
server inside a single-tenant transport if they want isolation at the
process boundary, but a bound connection no longer pollutes siblings.

Tools that consume session state accept an optional
``ctx: Context | None = None`` parameter — FastMCP auto-injects the active
:class:`~mcp.server.fastmcp.Context` when the type hint is present. When
``ctx`` is provided and carries a ``state.session`` attribute we prefer
it; otherwise we fall back to the bound :class:`MCPServerConnection`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

from mcp.server.fastmcp import Context as _FastMCPContext, FastMCP

from carl_studio.mcp.session import SESSION_MAX_AGE, MCPSession

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from carl_studio.mcp.connection import MCPServerConnection


# FastMCP's ``Context`` is generic over ``(ServerSessionT, LifespanContextT,
# RequestT)``. Our tools never consume those specifics — the injected
# Context is used purely as a carrier for optional per-request session
# state. Exporting a concrete ``Any``-parameterised alias lets tool
# signatures read ``ctx: Context | None = None`` without drowning pyright
# in ``reportMissingTypeArgument`` noise on every tool.
Context = _FastMCPContext[Any, Any, Any]

logger = logging.getLogger(__name__)

mcp = FastMCP("carl-studio")

logger.info(
    "carl-studio MCP server loaded: per-request session state via "
    "MCPServerConnection (H2, v0.7.1)."
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


def _get_session(ctx: "Context | None" = None) -> MCPSession:
    """Return the active :class:`MCPSession` for the current tool call.

    Resolution order:

    1. ``ctx.state.session`` when FastMCP has injected a :class:`Context`
       that exposes a ``state.session`` attribute. This is the
       future-ready seam for multi-request scoping; FastMCP 1.10 does not
       populate ``state.session`` itself but third-party middleware can.
    2. The bound :class:`MCPServerConnection` (set by
       :func:`bind_connection`), which holds the per-connection session.
    3. A transient empty :class:`MCPSession` — covers legacy test paths
       that exercise tools without an open connection. Production FastMCP
       always has a bound connection by the time a tool fires.
    """
    if ctx is not None:
        state = getattr(ctx, "state", None)
        if state is not None:
            maybe_session = getattr(state, "session", None)
            if isinstance(maybe_session, MCPSession):
                return maybe_session
    conn = get_bound_connection()
    if conn is None:
        return MCPSession()
    return conn.session


def _set_session(session: MCPSession) -> None:
    """Atomically replace the bound connection's session snapshot.

    No-op when no connection is bound — tools called outside a real FastMCP
    serve loop (e.g. unit tests that construct tools directly) don't have a
    durable seat to write to. Callers that need strong assertions should
    bind a connection first.
    """
    conn = get_bound_connection()
    if conn is None:
        return
    conn.session = session


def _expire_if_stale(session: MCPSession) -> MCPSession:
    """Return an unauthenticated session when ``session`` is past its TTL.

    Preserves the legacy ``_SESSION_MAX_AGE`` staleness behaviour: a
    successful ``authenticate`` stamps ``authenticated_at``, and any
    subsequent tool call older than :data:`SESSION_MAX_AGE` seconds is
    treated as ``tier="free"`` / empty JWT.
    """
    if session.authenticated_at is None:
        return session
    age = time.time() - session.authenticated_at.timestamp()
    if age > SESSION_MAX_AGE:
        return MCPSession()
    return session


def _require_tier(feature: str, ctx: "Context | None" = None) -> bool:
    """Return True if the current session's tier allows ``feature``.

    Also enforces session age — sessions older than
    :data:`SESSION_MAX_AGE` seconds are treated as unauthenticated and
    downgraded to the free tier (the downgrade is persisted back onto
    the bound connection so subsequent calls see the expiry).
    """
    from carl_studio.tier import Tier, tier_allows

    session = _get_session(ctx)
    expired = _expire_if_stale(session)
    if expired is not session:
        _set_session(expired)
        session = expired

    t = Tier(session.tier or "free")
    return tier_allows(t, feature)


def _tier_error(feature: str, ctx: "Context | None" = None) -> str:
    """Return standard JSON error for tier gate failures."""
    session = _get_session(ctx)
    return json.dumps(
        {
            "error": f"Feature '{feature}' requires CARL Paid tier.",
            "current_tier": session.tier or "free",
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
async def authenticate(jwt: str, ctx: Context | None = None) -> str:
    """Authenticate this MCP session with a carl.camp JWT.

    Call this first before using PAID features.
    The JWT is verified server-side via the carl.camp Supabase Edge Function.
    Returns: JSON with {tier, user_id, features_available: []}.

    ``ctx`` is auto-injected by FastMCP when the server is running inside a
    live request context; it lets per-request state supersede the bound
    :class:`MCPServerConnection` when present. Callers outside a FastMCP
    context may omit it — the tool falls back to the bound connection.
    """

    async def _body() -> str:
        if not jwt or not isinstance(jwt, str):
            return json.dumps(
                {"error": "jwt must be a non-empty string", "authenticated": False}
            )

        try:
            # Server-side JWT verification via Supabase Edge Function
            from datetime import datetime, timezone

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
            user_id = str(profile.user_id or "")

            current = _get_session(ctx)
            new_session = current.with_(
                jwt=jwt,
                tier=tier,
                user_id=user_id,
                authenticated_at=datetime.fromtimestamp(
                    int(time.time()), tz=timezone.utc
                ),
            )
            _set_session(new_session)

            from carl_studio.tier import FEATURE_TIERS, Tier, tier_allows

            t = Tier(tier)
            features = [f for f, _req in FEATURE_TIERS.items() if tier_allows(t, f)]

            return json.dumps(
                {
                    "tier": tier,
                    "user_id": user_id,
                    "features_available": features,
                    "authenticated": True,
                }
            )
        except Exception as e:
            # Fail closed — do NOT store the unverified JWT or grant paid access
            _set_session(MCPSession())
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
async def get_tier_status(ctx: Context | None = None) -> str:
    """Get current session tier and which features are accessible.

    Returns: JSON with {tier, authenticated, features: {name: allowed}}.
    Shows 'free' if not authenticated — no JWT required.

    ``ctx`` is FastMCP-injected when available; see :func:`authenticate`.
    """

    async def _body() -> str:
        from carl_studio.tier import FEATURE_TIERS, Tier, tier_allows

        session = _get_session(ctx)
        tier = session.tier or "free"
        t = Tier(tier)
        features = {f: tier_allows(t, f) for f in FEATURE_TIERS}

        return json.dumps(
            {
                "tier": tier,
                "authenticated": session.is_authenticated,
                "user_id": session.user_id,
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
async def run_skill(
    skill_name: str,
    inputs_json: str = "{}",
    ctx: Context | None = None,
) -> str:
    """Run a CARL skill by name and return the result.

    Available skills: observer, grader, trainer, synthesizer, deployer.
    inputs_json: JSON string of keyword arguments for the skill.

    Requires PAID tier for grader (full eval) and trainer (job submission).
    Returns: JSON SkillResult.

    ``ctx`` is FastMCP-injected when available; see :func:`authenticate`.
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
        if requires_tier == "paid" and not _require_tier("experiment", ctx):
            runner.close()
            return _tier_error(f"skill:{skill_name}", ctx)

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
    ctx: Context | None = None,
) -> str:
    """Dispatch an A2A task to the CARL agent's local bus.

    The task will be executed when 'carl agent run' is called locally.
    Requires PAID tier.
    Returns: JSON with {task_id, status: "pending"}.

    ``ctx`` is FastMCP-injected when available; see :func:`authenticate`.
    """

    async def _body() -> str:
        if not _require_tier("orchestration", ctx):
            return _tier_error("orchestration", ctx)

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
async def sync_data(
    direction: str = "push",
    entity_types: str = "runs",
    ctx: Context | None = None,
) -> str:
    """Sync local data with carl.camp. direction: push | pull.

    Requires PAID tier and carl camp login (JWT in session or local DB).
    Returns: JSON with sync results.

    ``ctx`` is FastMCP-injected when available; see :func:`authenticate`.
    """

    async def _body() -> str:
        if not _require_tier("sync.cloud", ctx):
            return _tier_error("sync.cloud", ctx)

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
        jwt = _get_session(ctx).jwt
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
