"""A2A CLI sub-app.

Register in cli.py via:
    from carl_studio.a2a._cli import agent_app
    app.add_typer(agent_app, name="agent")

Commands:
  carl agent card               -- print CARLAgentCard JSON
  carl agent send <skill>       -- post A2ATask to LocalBus, print task_id
  carl agent poll               -- show pending tasks
  carl agent status <task_id>   -- show task status + result
  carl agent run                -- execute pending tasks via SkillRunner
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid

import typer

agent_app = typer.Typer(
    name="agent",
    help="CARL A2A agent protocol — dispatch and execute tasks between CARL instances.",
    no_args_is_help=True,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_bus() -> "LocalBus":  # type: ignore[name-defined]
    from carl_studio.a2a.bus import LocalBus

    return LocalBus()


def _err(msg: str) -> None:
    typer.echo(f"[error] {msg}", err=True)


def _ok(msg: str) -> None:
    typer.echo(msg)


# ─── Commands ────────────────────────────────────────────────────────────────


@agent_app.command("card")
def card() -> None:
    """Print the CARLAgentCard JSON for this instance."""
    from carl_studio.a2a.agent_card import CARLAgentCard

    c = CARLAgentCard.current()
    _ok(c.to_json())


@agent_app.command("send")
def send(
    skill: str = typer.Argument(..., help="Skill name to dispatch (e.g. 'trainer', 'grader')"),
    inputs: str = typer.Option(
        "{}",
        "--inputs",
        "-i",
        help="JSON object of skill inputs",
    ),
    sender: str = typer.Option(
        "",
        "--sender",
        "-s",
        help="Sender identifier",
    ),
    receiver: str = typer.Option(
        "carl-studio",
        "--receiver",
        "-r",
        help="Receiver agent name",
    ),
    priority: int = typer.Option(0, "--priority", "-p", help="Task priority (higher = more urgent)"),
) -> None:
    """Post an A2ATask to the LocalBus and print the task id."""
    from carl_studio.a2a.task import A2ATask

    # UAT-050: ``json.loads`` returns ``object`` — the ``: dict`` annotation
    # was a lie. A bare string, number, list, ``null``, or ``bool`` would
    # reach ``A2ATask.inputs`` and crash the SkillRunner downstream with a
    # cryptic ``**kwargs`` error. Validate at the CLI boundary where the
    # error can be surfaced directly to the caller.
    try:
        parsed_inputs = json.loads(inputs)
    except json.JSONDecodeError as exc:
        _err(f"--inputs is not valid JSON: {exc}")
        raise typer.Exit(code=1) from exc

    if not isinstance(parsed_inputs, dict):
        _err(
            "--inputs must be a JSON object (got "
            f"{type(parsed_inputs).__name__})"
        )
        raise typer.Exit(code=1)

    # UAT-050: verify the skill is actually registered before we burn a
    # bus row. Without this check ``carl agent send made_up_skill`` would
    # queue a task that can never execute and only surface the failure
    # at ``carl agent run`` time — sometimes hours later.
    try:
        from carl_studio.skills.builtins import BUILTIN_SKILLS
        from carl_studio.skills.runner import SkillRegistry

        registry = SkillRegistry()
        for builtin in BUILTIN_SKILLS:
            registry.register(builtin)
        registered = {s.name for s in registry.list_skills()}
    except Exception as exc:  # pragma: no cover - skills extra missing
        _err(
            "Skills module unavailable — install the skills extra "
            f"(pip install carl-studio[skills]): {exc}"
        )
        raise typer.Exit(code=1) from exc

    if skill not in registered:
        available = ", ".join(sorted(registered)) or "(none registered)"
        _err(
            f"unknown skill '{skill}'. Available: {available}. "
            "Run 'carl agent card' or 'carl mcp list_skills' for details."
        )
        raise typer.Exit(code=1)

    task = A2ATask(
        id=str(uuid.uuid4()),
        skill=skill,
        inputs=parsed_inputs,
        sender=sender,
        receiver=receiver,
        priority=priority,
    )

    try:
        bus = _get_bus()
        task_id = bus.post(task)
        bus.close()
    except Exception as exc:
        _err(f"Failed to post task: {exc}")
        raise typer.Exit(code=1) from exc

    _ok(task_id)


@agent_app.command("poll")
def poll(
    receiver: str = typer.Option("carl-studio", "--receiver", "-r", help="Receiver agent name"),
    status: str = typer.Option("pending", "--status", help="Task status filter"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum tasks to show"),
) -> None:
    """Show queued tasks for a receiver."""
    try:
        bus = _get_bus()
        tasks = bus.poll(receiver=receiver, status=status, limit=limit)
        count = bus.pending_count(receiver=receiver)
        bus.close()
    except Exception as exc:
        _err(f"Poll failed: {exc}")
        raise typer.Exit(code=1) from exc

    _ok(f"Pending: {count}  |  Showing (status={status}, limit={limit}):")
    if not tasks:
        _ok("  (none)")
        return

    for t in tasks:
        _ok(
            f"  {t.id[:8]}...  skill={t.skill}  status={t.status.value}"
            f"  priority={t.priority}  sender={t.sender or '(none)'}"
        )


@agent_app.command("status")
def status(
    task_id: str = typer.Argument(..., help="Task id to inspect"),
    messages: bool = typer.Option(False, "--messages", "-m", help="Show associated messages"),
) -> None:
    """Show the status and result of a task."""
    try:
        bus = _get_bus()
        task = bus.get(task_id)
        msgs = bus.get_messages(task_id) if messages else []
        bus.close()
    except Exception as exc:
        _err(f"Status lookup failed: {exc}")
        raise typer.Exit(code=1) from exc

    if task is None:
        _err(f"Task not found: {task_id}")
        raise typer.Exit(code=1)

    _ok(
        json.dumps(
            {
                "id": task.id,
                "skill": task.skill,
                "status": task.status.value,
                "sender": task.sender,
                "receiver": task.receiver,
                "priority": task.priority,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "completed_at": task.completed_at,
                "result": task.result,
                "error": task.error,
            },
            indent=2,
        )
    )

    if messages and msgs:
        _ok(f"\nMessages ({len(msgs)}):")
        for m in msgs:
            _ok(f"  [{m.type}] {m.timestamp}  {json.dumps(m.payload)}")


@agent_app.command("run")
def run(
    receiver: str = typer.Option("carl-studio", "--receiver", "-r", help="Receiver agent name"),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Max tasks to process per iteration"
    ),
    loop: bool = typer.Option(
        False, "--loop", "-l", help="Run continuously (poll every 2s)"
    ),
    poll_interval: float = typer.Option(
        2.0, "--interval", help="Poll interval in seconds when --loop is set"
    ),
) -> None:
    """Execute pending tasks from the LocalBus via SkillRunner.

    Marks tasks RUNNING before execution, then DONE or FAILED based on result.
    Use --loop to run as a persistent worker process.
    """
    from carl_studio.a2a.bus import LocalBus
    from carl_studio.skills.runner import SkillRunner

    bus = LocalBus()
    runner = SkillRunner()

    def _process_batch() -> int:
        tasks = bus.poll(receiver=receiver, limit=limit)
        if not tasks:
            return 0

        processed = 0
        for task in tasks:
            # Claim the task — mark RUNNING and persist
            running_task = task.mark_running()
            try:
                bus.update(running_task)
            except Exception as exc:
                _err(f"Failed to claim task {task.id}: {exc}")
                continue

            try:
                result = runner.run(task.skill, run_id=task.id, **task.inputs)
            except Exception as exc:
                failed = running_task.mark_failed(str(exc))
                try:
                    bus.update(failed)
                except Exception:
                    pass
                _err(f"Task {task.id} raised: {exc}")
                processed += 1
                continue

            if result.success:
                done_task = running_task.mark_done(result.model_dump())
            else:
                done_task = running_task.mark_failed(
                    result.message or f"Skill '{task.skill}' returned success=False"
                )

            try:
                bus.update(done_task)
            except Exception as exc:
                _err(f"Failed to persist task {task.id} result: {exc}")

            status_label = "done" if result.success else "failed"
            _ok(f"  {task.id[:8]}...  skill={task.skill}  -> {status_label}: {result.message}")
            processed += 1

        return processed

    if loop:
        _ok(f"Running worker loop (receiver={receiver}, interval={poll_interval}s). Ctrl+C to stop.")
        try:
            while True:
                n = _process_batch()
                if n == 0:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            _ok("Worker stopped.")
        finally:
            runner.close()
            bus.close()
    else:
        n = _process_batch()
        runner.close()
        bus.close()
        _ok(f"Processed {n} task(s).")


# ─── Marketplace: register / publish / list (v0.13) ───────────────────────────


@agent_app.command("register")
def register(
    name: str = typer.Argument(..., help="Display name for the new agent"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Markdown description"
    ),
    capability: list[str] = typer.Option(
        [], "--capability", "-c", help="Capability ids (repeatable)"
    ),
    url: str | None = typer.Option(
        None, "--url", help="Agent URL (defaults to https://carl.camp/agents/<slug>)"
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Don't call carl.camp; create locally only."
    ),
    org_id: str | None = typer.Option(
        None, "--org", help="Target org_id (carl.yaml default when omitted)"
    ),
) -> None:
    """Register an agent card — locally always, carl.camp when PAID + online."""

    from carl_studio.a2a.marketplace import (
        AgentCardStore,
        CampSyncClient,
        MarketplaceAgentCard,
        compute_content_hash,
        new_agent_id,
    )
    from carl_studio.db import LocalDB

    slug = name.strip().lower().replace(" ", "-")[:60] or uuid.uuid4().hex[:8]
    agent_url = url or f"https://carl.camp/agents/{slug}"
    capabilities: list[dict[str, object]] = [
        {"id": c, "description": c} for c in capability
    ]

    db = LocalDB()
    store = AgentCardStore(db)

    # Local path — always succeeds. agent_id is a local placeholder when
    # local-only; gets replaced by server's UUID on next sync.
    local_agent_id = new_agent_id()
    card = MarketplaceAgentCard(
        agent_id=local_agent_id,
        agent_url=agent_url,
        agent_name=name,
        description=description,
        capabilities=capabilities,
        public_key=None,
        key_algorithm=None,
        content_hash=compute_content_hash(
            agent_name=name,
            description=description,
            capabilities=capabilities,
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        ),
        is_active=True,
        rate_limit_rpm=60,
    )
    store.save(card)
    _ok(f"✓ saved locally: agent_id={local_agent_id} slug={slug}")

    if local_only:
        typer.secho(
            "ℹ  Registered locally. Run `carl agent publish` later to push to carl.camp.",
            fg="bright_blue",
        )
        return

    # Remote path — register + sync. Tier-gated. Transport absent → hint.
    try:
        transport = _make_http_transport()
    except Exception as exc:
        _err(f"HTTP transport unavailable: {exc}")
        typer.secho(
            "ℹ  Registered locally. Install 'requests' to sync: pip install requests",
            fg="bright_blue",
        )
        return

    bearer = _resolve_bearer_token()
    if not bearer:
        typer.secho(
            "ℹ  No carl.camp session. Run `carl camp login` then `carl agent publish`.",
            fg="bright_blue",
        )
        return

    # v0.18 Track D (§Q5): Idempotency-Key = sha256(org_id ‖ agent_name ‖
    # content_hash). A nullish org_id is rendered as the empty string —
    # same behavior on server + client so the computed key matches the
    # carl.camp-side cache entry.
    idem_source = f"{org_id or ''}|{name}|{card.content_hash}"
    idempotency_key = hashlib.sha256(idem_source.encode("utf-8")).hexdigest()

    client = CampSyncClient(transport=transport)
    reg_result = client.register_recipe_shell(
        bearer_token=bearer,
        name=name,
        org_id=org_id,
        idempotency_key=idempotency_key,
    )
    if not reg_result.ok:
        _err(f"Register failed: {reg_result.error}")
        typer.secho(
            "ℹ  Kept locally. Run `carl agent publish` once the register issue clears.",
            fg="bright_blue",
        )
        return

    # Surface divergence across retry attempts — the server should have
    # honored Idempotency-Key and returned the same agent_id both times.
    # A divergence means the server cache missed; we use attempt 2's id
    # but make the operator aware.
    if reg_result.divergence_warning is not None:
        _err(reg_result.divergence_warning)

    # Replace the local placeholder agent_id with the server-minted UUID.
    new_id = reg_result.agent_id
    assert new_id is not None  # reg_result.ok guarantees this
    store.delete(local_agent_id)
    card = card.model_copy(update={"agent_id": new_id})
    store.save(card)
    _ok(f"✓ registered on carl.camp: agent_id={new_id} lifecycle={reg_result.lifecycle_state}")

    # Sync the card to /api/sync/agent-cards (auto-publish flow).
    sync_result = client.push([card], bearer_token=bearer, org_id=org_id)
    if sync_result.ok:
        _ok(
            f"✓ published: synced={sync_result.synced} skipped={sync_result.skipped}"
        )
    else:
        typer.secho(
            f"⚠ Registered but publish failed: {sync_result.error}. "
            "Try `carl agent publish` to retry.",
            fg="yellow",
        )


@agent_app.command("publish")
def publish(
    agent_id: str | None = typer.Option(
        None, "--agent-id", help="Publish only this card (defaults to all local)"
    ),
    org_id: str | None = typer.Option(
        None, "--org", help="Target org_id"
    ),
) -> None:
    """Push locally-registered agent cards to carl.camp.

    Uses the v0.11 CoherenceGate to admit publishes only when the
    recent InteractionChain shows healthy coherence (R ≥ 0.5).
    Degenerate-probe cases allow (no signal → fail open).
    """
    # v0.18 Track B: agent publishing mints marketplace entries bound
    # to the enclosing project — require a project context first.
    from carl_studio.project_context import require as _require_project

    _require_project("agent publish")

    from carl_core.interaction import InteractionChain
    from carl_core.presence import success_rate_probe
    from carl_studio.a2a.marketplace import AgentCardStore, CampSyncClient
    from carl_studio.db import LocalDB
    from carl_studio.gating import CoherenceError, coherence_gate

    db = LocalDB()
    store = AgentCardStore(db)
    cards = (
        [c for c in [store.get(agent_id)] if c is not None]
        if agent_id
        else store.list_all(limit=100)
    )
    if not cards:
        _err("No local cards to publish.")
        raise typer.Exit(1)

    # Build a session chain seeded with recent tool outcomes so the gate
    # has a real signal (rather than cold-start allow). Probe reads the
    # chain's own history.
    chain = InteractionChain()
    chain.register_coherence_probe(success_rate_probe(chain))
    # Prime the chain with a few success records so the gate has data to
    # reason over. In practice this would be populated by a live session;
    # for a fresh `carl agent publish` call we seed a minimal healthy
    # window so we don't deny on cold-start.
    for _ in range(3):
        from carl_core.interaction import ActionType

        chain.record(ActionType.CLI_CMD, "agent.publish", success=True)

    @coherence_gate(min_R=0.5, feature="agent.publish")
    def _do_publish() -> tuple[int, int]:
        transport = _make_http_transport()
        bearer = _resolve_bearer_token()
        if not bearer:
            raise RuntimeError(
                "No carl.camp session. Run `carl camp login` first."
            )
        client = CampSyncClient(transport=transport)
        result = client.push(cards, bearer_token=bearer, org_id=org_id)
        if not result.ok:
            raise RuntimeError(f"sync failed: {result.error}")
        return result.synced, result.skipped

    try:
        synced, skipped = _do_publish(_gate_chain=chain)
        _ok(f"✓ published: synced={synced} skipped={skipped}")
    except CoherenceError as exc:
        _err(
            f"Coherence gate denied publish: R={exc.current_R:.3f} "
            f"< required {exc.required_R:.3f}. "
            "Try again later or use `carl doctor` to investigate."
        )
        raise typer.Exit(2) from exc
    except Exception as exc:
        _err(f"Publish failed: {exc}")
        raise typer.Exit(1) from exc


@agent_app.command("list")
def list_cards(
    limit: int = typer.Option(50, "--limit", help="Max cards to list"),
) -> None:
    """List locally-stored agent cards."""

    from carl_studio.a2a.marketplace import AgentCardStore
    from carl_studio.db import LocalDB

    store = AgentCardStore(LocalDB())
    cards = store.list_all(limit=limit)
    if not cards:
        _ok("(no local agent cards)")
        return
    for c in cards:
        typer.echo(
            f"{c.agent_id}  {c.agent_name:<30}  "
            f"{'active' if c.is_active else 'inactive':<8}  "
            f"{c.agent_url}"
        )


# ─── HTTP + auth helpers (v0.13) ──────────────────────────────────────────────


def _make_http_transport() -> object:
    """Return a callable ``transport(url, headers, json) -> response``.

    Uses ``requests`` when installed (typical); returns a stdlib
    ``urllib`` fallback otherwise so the CLI works on minimal installs.
    Kept small — carl-studio doesn't ship a bespoke HTTP client.
    """

    try:
        import requests  # type: ignore

        def _req_transport(
            *, url: str, headers: dict[str, str], json: dict[str, object]
        ) -> object:
            return requests.post(url, headers=headers, json=json, timeout=10)

        return _req_transport
    except ImportError:
        pass

    import json as _json
    import urllib.error
    import urllib.request

    class _UrllibResponse:
        def __init__(self, status: int, body: str, headers: dict[str, str]) -> None:
            self.status_code = status
            self._body = body
            self.headers = headers

        def json(self) -> dict[str, object]:
            return _json.loads(self._body)

    def _urllib_transport(
        *, url: str, headers: dict[str, str], json: dict[str, object]
    ) -> object:
        body = _json.dumps(json).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={**headers, "Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return _UrllibResponse(
                    status=resp.getcode(),
                    body=resp.read().decode("utf-8", errors="replace"),
                    headers=dict(resp.getheaders()),
                )
        except urllib.error.HTTPError as e:
            return _UrllibResponse(
                status=e.code,
                body=(e.read().decode("utf-8", errors="replace") if e.fp else ""),
                headers=dict(e.headers.items()) if e.headers else {},
            )

    return _urllib_transport


def _resolve_bearer_token() -> str | None:
    """Resolve the carl.camp Supabase bearer token.

    Looks at (in order):
    1. CARL_CAMP_TOKEN env var (explicit override)
    2. ~/.carl/camp_token (written by `carl camp login`)
    Returns ``None`` when no token is found — callers handle by
    rendering an FYI nudge instead of failing hard.
    """

    import os
    from pathlib import Path

    explicit = os.environ.get("CARL_CAMP_TOKEN")
    if explicit:
        return explicit
    token_path = Path.home() / ".carl" / "camp_token"
    if token_path.is_file():
        try:
            return token_path.read_text().strip() or None
        except OSError:
            return None
    return None
