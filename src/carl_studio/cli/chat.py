"""Chat CLI — ``carl chat``.

Primary natural language interface to CARL. Launches an agentic chat loop
with tool use, frame awareness, and proactive guidance.

Session persistence: --session <id> resumes, auto-saves on exit.
Session listing: --sessions lists all saved sessions.
Cost tracking: --budget <usd> sets a hard spending cap.
"""

from __future__ import annotations

from typing import Any

import typer

from carl_studio.console import get_console


def _list_sessions() -> None:
    """List saved chat sessions as a Rich table and exit."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    console = Console()
    sessions_dir = Path.home() / ".carl" / "sessions"

    if not sessions_dir.exists():
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Chat Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Model", style="dim")
    table.add_column("Turns", justify="right")
    table.add_column("Cost", justify="right", style="green")
    table.add_column("Last Modified", style="dim")

    for path in sorted(
        sessions_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(path.read_text())
            turn_count = data.get("turn_count", len(data.get("messages", [])))
            cost = data.get("total_cost_usd", 0.0)
            title = data.get("title", "")
            model = data.get("model", "")
            updated = data.get("updated_at", "")
            if not updated:
                mtime = path.stat().st_mtime
                updated = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
            else:
                # Trim ISO suffix for display
                updated = updated[:16].replace("T", " ")
            table.add_row(
                path.stem,
                title[:30],
                model,
                str(turn_count),
                f"${cost:.4f}" if cost else "-",
                updated,
            )
        except Exception:
            table.add_row(path.stem, "?", "?", "?", "?", "?")

    if table.row_count == 0:
        console.print("[dim]No sessions found.[/dim]")
    else:
        console.print(table)


def run_one_shot_agent(
    prompt: str,
    *,
    model: str = "",
    config: str = "carl.yaml",
    api_key: str | None = None,
    frame_source: str = "current",
    budget: float = 0.0,
) -> None:
    """Fire a single agent turn with ``prompt`` and print the streamed reply.

    Used for ``carl ask "train a model on gsm8k"`` style invocations.
    Does not enter the interactive loop; returns when the turn is done.
    """
    c = get_console()

    frame = None
    if frame_source and frame_source != "none":
        from carl_studio.frame import WorkFrame

        if frame_source == "current":
            frame = WorkFrame.from_project(config)
        else:
            frame = WorkFrame.load(frame_source)

    from carl_studio.chat_agent import CARLAgent

    agent = CARLAgent(
        api_key=api_key or "",
        model=model,
        frame=frame,
        max_budget_usd=budget,
    )

    info = agent.provider_info
    c.blank()
    c.kv("Model", info["model"])
    c.kv("Prompt", prompt[:120] + ("..." if len(prompt) > 120 else ""))
    c.blank()

    streaming_text = False
    try:
        for event in agent.chat(prompt):
            if event.kind == "text_delta":
                if not streaming_text:
                    c.print("carl> ", end="")
                    streaming_text = True
                c.print(event.content, end="")
            elif event.kind == "text":
                c.blank()
                from rich.markdown import Markdown

                c.print(Markdown(f"**carl>** {event.content}"))
            elif event.kind == "tool_call":
                if streaming_text:
                    c.blank()
                    streaming_text = False
                c.info(f"[{event.tool_name}] {_format_args(event.tool_args)}")
            elif event.kind == "tool_result":
                preview = event.content[:200]
                if len(event.content) > 200:
                    preview += "..."
                c.info(f"  -> {preview}")
            elif event.kind == "done":
                if streaming_text:
                    c.blank()
                    streaming_text = False
            elif event.kind == "error":
                c.error(event.content)
    except Exception as exc:
        c.error(str(exc))

    cost = agent.cost_summary
    if cost["turn_count"] > 0:
        c.blank()
        c.kv("Cost", f"${cost['total_cost_usd']:.4f}", key_width=10)


def ask_cmd(
    prompt: str = typer.Argument(..., help="What to ask or do"),
    model: str = typer.Option("", "--model", "-m", help="Claude model"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config"),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"
    ),
    frame_source: str = typer.Option("current", "--frame", help="WorkFrame"),
    budget: float = typer.Option(0.0, "--budget", help="Max spend in USD"),
) -> None:
    """Ask Carl a single question or give a single instruction."""
    run_one_shot_agent(
        prompt,
        model=model,
        config=config,
        api_key=api_key,
        frame_source=frame_source,
        budget=budget,
    )


def chat_cmd(
    model: str = typer.Option(
        "", "--model", "-m", help="Claude model for agent (default: from settings)"
    ),
    config: str = typer.Option(
        "carl.yaml", "--config", "-c", help="Project config for context"
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"
    ),
    frame_source: str = typer.Option(
        "current", "--frame", help="WorkFrame: 'current', 'none', or path to YAML"
    ),
    session: str = typer.Option(
        "", "--session", "-s", help="Resume a saved session by ID"
    ),
    budget: float = typer.Option(
        0.0, "--budget", help="Max spend in USD (0 = unlimited)"
    ),
    voice: bool = typer.Option(
        False, "--voice", help="Voice mode (requires faster-whisper + kokoro)"
    ),
    local: bool = typer.Option(
        False, "--local", help="Use local model via LM Studio (localhost:1234)"
    ),
    sessions: bool = typer.Option(
        False, "--sessions", help="List saved chat sessions and exit"
    ),
) -> None:
    """Chat with CARL. Proactive agent with tool use, file ingestion, and analysis."""
    c = get_console()

    if sessions:
        _list_sessions()
        raise typer.Exit(0)

    if voice:
        c.warn("Voice mode coming soon. Requires: pip install faster-whisper kokoro pyaudio")
        c.info("For now, use text chat. The agent loop is identical — voice adds I/O wrapping.")
        raise typer.Exit(0)

    if local:
        c.warn("Local model mode coming soon. Requires LM Studio running at localhost:1234.")
        c.info("Use: carl chat --model claude-sonnet-4-6  for cloud inference.")
        raise typer.Exit(0)

    # First-run nudge: if the user has never initialized Carl, offer it now.
    # The chat agent needs an API key at minimum; walking through init first
    # beats hitting a mystery credential prompt mid-conversation.
    try:
        from carl_studio.cli.init import _first_run_complete, init_cmd
    except ImportError:
        pass
    else:
        if not _first_run_complete():
            c.blank()
            c.info("Looks like this is your first run. Let's set up in one minute.")
            if typer.confirm("Run carl init now?", default=True):
                try:
                    init_cmd(skip_extras=False, skip_project=False, force=False, json_output=False)
                except typer.Exit:
                    pass
                c.blank()

    # Load frame
    frame = None
    if frame_source and frame_source != "none":
        from carl_studio.frame import WorkFrame

        if frame_source == "current":
            frame = WorkFrame.from_project(config)
        else:
            frame = WorkFrame.load(frame_source)

    # Create agent — api_key and model resolve from CARLSettings when not passed
    from carl_studio.chat_agent import CARLAgent

    agent = CARLAgent(
        api_key=api_key or "",
        model=model,
        frame=frame,
        max_budget_usd=budget,
        session_id=session,
    )

    # Resume session if requested
    if session:
        resumed = agent.load_session(session)
        # _last_load_quarantined becomes True when the underlying session
        # file was corrupt and got moved into the quarantine directory.
        # The user needs to see this — silent quarantine is a footgun that
        # masquerades as a fresh session start.
        if getattr(agent, "_last_load_quarantined", False):
            quarantine_dir = (
                __import__("pathlib").Path.home()
                / ".carl" / "sessions" / ".quarantine"
            )
            c.warn(
                f"Session '{session}' was corrupted and has been quarantined "
                f"to {quarantine_dir}. Starting fresh."
            )
        if resumed:
            c.ok(f"Resumed session: {session}")
        else:
            c.info(f"New session: {session}")

    info = agent.provider_info
    director = c.theme.persona.value.upper()
    c.blank()
    c.header(f"Chat with {director}")
    c.kv("Model", info["model"])
    c.kv("Key", info["api_key_source"])
    if frame and frame.active:
        c.kv("Frame", f"{frame.domain}/{frame.function}/{frame.role}")
    if session:
        c.kv("Session", session)
    if budget > 0:
        c.kv("Budget", f"${budget:.2f}")
    c.info("Type 'quit' or Ctrl+C to exit.")
    c.blank()

    # Env-baked intro — rendered ONCE before the first input() for a
    # zero-latency greeting. Resumed sessions skip the intro because the
    # user is mid-conversation; a fresh "/s"-style priming would be noise.
    from carl_studio.cli.intro import parse_intro_selection, render_intro
    from carl_studio.jit_context import JITContext, extract

    bootstrap_done = bool(session)  # resumed sessions skip bootstrap priming
    if not bootstrap_done:
        render_intro(c)

    while True:
        try:
            user_input = input("  you> ").strip()
        except (KeyboardInterrupt, EOFError):
            c.blank()
            _exit_session(c, agent)
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            _exit_session(c, agent)
            break

        # Bootstrap phase: on the FIRST turn of a fresh session, classify
        # the input against the intro moves and prime the agent. Subsequent
        # turns fall through to the normal path.
        primed_input = user_input
        bootstrap_ctx: JITContext | None = None
        if not bootstrap_done:
            bootstrap_done = True
            move_key = parse_intro_selection(user_input)

            # "/s" or "s" short-circuits into the sticky queue — the user
            # signalled "queue this for the heartbeat loop", so we do not
            # spin up the agent at all for this turn. The raw input is
            # treated as the note content (when the user typed just "s" or
            # "sticky" with no body, we prompt once for the note).
            if move_key == "s":
                try:
                    from carl_studio.db import LocalDB
                    from carl_studio.sticky import StickyQueue

                    note_body = user_input
                    # Strip the bare selection token so "s" alone does
                    # not become the note content.
                    stripped = user_input.strip().lower()
                    if stripped in ("s", "sticky", "/s", "/sticky"):
                        try:
                            note_body = input("  note> ").strip()
                        except (KeyboardInterrupt, EOFError):
                            note_body = ""
                    if note_body:
                        queue = StickyQueue(LocalDB())
                        queue.append(note_body, session_id=session or None)
                        c.ok("queued")
                    else:
                        c.info("nothing queued (empty note)")
                except Exception as exc:
                    c.error(f"sticky append failed: {exc}")
                continue

            # Extract JIT context and prime the agent's next system prompt.
            try:
                bootstrap_ctx = extract(user_input, move_key=move_key)
                prompt_ext = bootstrap_ctx.system_prompt_extension()
                if prompt_ext and hasattr(agent, "_system_prompt_extension"):
                    # CARLAgent exposes `_system_prompt_extension` explicitly
                    # as the public primer surface for CLI bootstrap paths.
                    agent._system_prompt_extension = prompt_ext  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:
                # JIT extraction is best-effort; never block the chat on it.
                c.warn(f"intro priming skipped: {exc}")
                bootstrap_ctx = None

        try:
            streaming_text = False
            for event in agent.chat(primed_input):
                if event.kind == "text_delta":
                    if not streaming_text:
                        c.blank()
                        c.print("carl> ", end="")
                        streaming_text = True
                    c.print(event.content, end="")
                elif event.kind == "text":
                    c.blank()
                    from rich.markdown import Markdown

                    c.print(Markdown(f"**carl>** {event.content}"))
                    c.blank()
                elif event.kind == "tool_call":
                    if streaming_text:
                        c.blank()
                        streaming_text = False
                    c.info(f"[{event.tool_name}] {_format_args(event.tool_args)}")
                elif event.kind == "tool_blocked":
                    c.warn(f"[{event.tool_name}] {event.content}")
                elif event.kind == "tool_result":
                    result_preview = event.content[:200]
                    if len(event.content) > 200:
                        result_preview += "..."
                    c.info(f"  -> {result_preview}")
                elif event.kind == "done":
                    if streaming_text:
                        c.blank()
                        streaming_text = False
                    c.blank()
                elif event.kind == "error":
                    c.error(event.content)
        except Exception as exc:
            c.error(str(exc))

        # Bootstrap phase — apply the JIT frame patch AFTER turn-1 completes.
        # We only do this when the agent has no existing frame (or an empty
        # one); an explicit pre-existing WorkFrame wins over a JIT guess.
        if bootstrap_ctx is not None and bootstrap_ctx.frame_patch:
            try:
                from carl_studio.frame import WorkFrame

                current = getattr(agent, "_frame", None) or WorkFrame()
                # Only apply the patch if the frame is currently inactive —
                # respect an explicit frame set via --frame or set_frame().
                if not getattr(current, "active", False):
                    # CARLAgent intentionally exposes these attributes for CLI
                    # bootstrap priming; the leading underscore is a module
                    # convention, not a hard access gate.
                    agent._frame = current.model_copy(update=bootstrap_ctx.frame_patch)  # pyright: ignore[reportPrivateUsage]
                    # Invalidate any cached constitution prompt so
                    # frame-topic rule selection picks up the new domain.
                    if hasattr(agent, "_constitution_prompt"):
                        agent._constitution_prompt = None  # pyright: ignore[reportPrivateUsage]
            except Exception as exc:
                c.warn(f"frame bootstrap skipped: {exc}")
        bootstrap_ctx = None  # one-shot — never re-apply on later turns


def _exit_session(c: Any, agent: Any) -> None:
    """Save session and show cost summary on exit.

    Also offers to promote 1-3 durable rules derived from the session into
    the user's constitution when the session is non-trivial (>=3 turns) and
    the agent has a constitution loaded.
    """
    # Auto-save if session has content
    if agent._messages:
        sid = agent.save_session(title="carl chat")
        c.info(f"Session saved: {sid}")
        c.info(f"Resume with: carl chat --session {sid}")

    # Offer learnings promotion — only when the session produced real content
    # and we actually have a constitution to append to.
    cost = agent.cost_summary
    turn_count = int(cost.get("turn_count", 0) or 0)
    constitution = getattr(agent, "_constitution", None)
    if turn_count >= 3 and constitution is not None:
        _maybe_promote_learnings(c, agent, constitution)

    # Show cost summary
    if turn_count > 0:
        c.kv("Cost", f"${cost['total_cost_usd']:.4f}", key_width=10)
        c.kv("Tokens", f"{cost['total_input_tokens']}in / {cost['total_output_tokens']}out", key_width=10)
        c.kv("Turns", str(turn_count), key_width=10)

    c.voice("farewell")


def _maybe_promote_learnings(c: Any, agent: Any, constitution: Any) -> None:
    """Prompt the user to codify durable learnings into the constitution."""
    try:
        prompt_ok = typer.confirm(
            "Promote any learnings from this session?", default=False,
        )
    except Exception:
        return
    if not prompt_ok:
        return

    try:
        proposed = agent.suggest_learnings() or []
    except Exception as exc:
        c.warn(f"Could not propose learnings: {exc}")
        return

    if not proposed:
        c.info("No durable learnings proposed.")
        return

    for rule in proposed:
        preview = f"[{rule.id} | p{rule.priority}] {rule.text[:160]}"
        try:
            keep = typer.confirm(f"Add rule? {preview}", default=False)
        except Exception:
            keep = False
        if not keep:
            continue
        try:
            constitution.append(rule)
            c.ok(f"Rule codified: {rule.id}")
        except Exception as exc:
            c.warn(f"Failed to append rule {rule.id}: {exc}")


def _format_args(args: dict) -> str:
    """Compact display of tool arguments."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)
