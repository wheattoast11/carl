"""Chat CLI — ``carl chat``.

Primary natural language interface to CARL. Launches an agentic chat loop
with tool use, frame awareness, and proactive guidance.
"""

from __future__ import annotations

import typer

from carl_studio.console import get_console


def chat_cmd(
    model: str = typer.Option(
        "claude-sonnet-4-6", "--model", "-m", help="Claude model for agent"
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
    voice: bool = typer.Option(
        False, "--voice", help="Voice mode (requires faster-whisper + kokoro)"
    ),
    local: bool = typer.Option(
        False, "--local", help="Use local model via LM Studio (localhost:1234)"
    ),
) -> None:
    """Chat with CARL. Proactive agent with tool use, file ingestion, and analysis."""
    c = get_console()

    if voice:
        c.warn("Voice mode coming soon. Requires: pip install faster-whisper kokoro pyaudio")
        c.info("For now, use text chat. The agent loop is identical — voice adds I/O wrapping.")
        raise typer.Exit(0)

    if local:
        c.warn("Local model mode coming soon. Requires LM Studio running at localhost:1234.")
        c.info("Use: carl chat --model claude-sonnet-4-6  for cloud inference.")
        raise typer.Exit(0)

    if not api_key:
        c.error("ANTHROPIC_API_KEY required for carl chat")
        c.info("Set via: export ANTHROPIC_API_KEY=sk-ant-...")
        c.info("Or pass: carl chat --api-key sk-ant-...")
        raise typer.Exit(1)

    # Load frame
    frame = None
    if frame_source and frame_source != "none":
        from carl_studio.frame import WorkFrame

        if frame_source == "current":
            frame = WorkFrame.from_project(config)
        else:
            frame = WorkFrame.load(frame_source)

    # Create agent
    from carl_studio.chat_agent import CARLAgent

    agent = CARLAgent(api_key=api_key, model=model, frame=frame)

    director = c.theme.persona.value.upper()
    c.blank()
    c.header(f"Chat with {director}")
    c.kv("Model", model)
    if frame and frame.active:
        c.kv("Frame", f"{frame.domain}/{frame.function}/{frame.role}")
    c.info("Type 'quit' or Ctrl+C to exit.")
    c.blank()

    while True:
        try:
            user_input = input("  you> ").strip()
        except (KeyboardInterrupt, EOFError):
            c.blank()
            c.voice("farewell")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            c.voice("farewell")
            break

        try:
            for event in agent.chat(user_input):
                if event.kind == "text":
                    from rich.markdown import Markdown

                    c.blank()
                    c.print(Markdown(f"**carl>** {event.content}"))
                    c.blank()
                elif event.kind == "tool_call":
                    c.info(f"[{event.tool_name}] {_format_args(event.tool_args)}")
                elif event.kind == "tool_result":
                    result_preview = event.content[:200]
                    if len(event.content) > 200:
                        result_preview += "..."
                    c.info(f"  -> {result_preview}")
                elif event.kind == "error":
                    c.error(event.content)
        except Exception as exc:
            c.error(str(exc))


def _format_args(args: dict) -> str:
    """Compact display of tool arguments."""
    parts = []
    for k, v in args.items():
        val = str(v)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)
