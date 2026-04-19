"""`carl commit` -- promote a learning to long-term constitutional memory."""
from __future__ import annotations

import json
import re
from hashlib import sha1
from pathlib import Path
from typing import Any

import typer

from carl_studio.console import get_console


def commit_cmd(
    learning: str = typer.Argument(None, help="Rule/learning in plain English"),
    rule_id: str = typer.Option("", "--id", help="Stable dot-id; auto-generated if empty"),
    priority: int = typer.Option(60, "--priority", "-p", min=0, max=100),
    tags: str = typer.Option("", "--tags", help="Comma-separated topic tags"),
    from_session: str = typer.Option("", "--from-session", help="Extract learnings from a saved session"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Codify a learning into ~/.carl/constitution.yaml."""
    c = get_console()

    stmt = (learning or "").strip()
    sess = (from_session or "").strip()

    if sess and stmt:
        c.error_with_hint(
            "Choose one",
            detail="Use either LEARNING argument or --from-session, not both.",
            code="carl.commit.ambiguous",
        )
        raise typer.Exit(2)

    if not sess and not stmt:
        c.error_with_hint(
            "Nothing to commit",
            detail="Pass a learning as argument or --from-session <id>.",
            hint='Example: carl commit "never trust transient state" --tags caching',
            code="carl.commit.empty",
        )
        raise typer.Exit(2)

    from carl_studio.constitution import Constitution, ConstitutionalRule

    to_append: list[ConstitutionalRule] = []

    if sess:
        to_append.extend(_learnings_from_session(sess))
        if not to_append:
            c.info("No durable learnings extracted from session.")
            return
    else:
        rid = rule_id.strip() or _auto_id(stmt)
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        to_append.append(
            ConstitutionalRule(
                id=rid,
                text=stmt,
                priority=priority,
                tags=tag_list,
                source="user",
            )
        )

    if dry_run:
        for rule in to_append:
            c.print(
                f"[dim]would commit:[/] [bold]{rule.id}[/] (p{rule.priority}) -- {rule.text}"
            )
        c.info("--dry-run: nothing written")
        return

    constitution = Constitution.load()
    for rule in to_append:
        constitution.append(rule)
        c.ok(f"Committed: {rule.id} (p{rule.priority})")


def _auto_id(text: str) -> str:
    """Derive a stable id from the learning text."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)][:4]
    base = "user." + ".".join(words or ["rule"])
    digest = sha1(text.encode("utf-8")).hexdigest()[:6]
    return f"{base}.{digest}"


def _learnings_from_session(session_id: str) -> list[Any]:
    """Ask CARLAgent to summarize a saved session's key learnings as rule candidates."""
    from carl_studio.chat_agent import CARLAgent
    from carl_studio.constitution import ConstitutionalRule

    session_path = Path.home() / ".carl" / "sessions" / f"{session_id}.json"
    if not session_path.exists():
        raise typer.BadParameter(f"Session not found: {session_path}")

    try:
        raw = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Session file is not valid JSON: {session_path}") from exc
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"Session file malformed: {session_path}")

    transcript = _summarize_transcript(raw)
    if not transcript.strip():
        return []

    prompt = (
        "Review this conversation transcript and extract up to 3 durable lessons "
        "that should guide future sessions. Respond as JSON: "
        '[{"id": "...", "text": "...", "priority": 50, "tags": ["..."]}]. '
        "Each rule should be generalizable, not session-specific. "
        "If nothing is worth committing, return []."
        + "\n\nTRANSCRIPT:\n"
        + transcript
    )

    try:
        agent = CARLAgent(api_key=None, frame=None, max_budget_usd=0.30)
    except ImportError:
        return []
    agent._tools = []  # type: ignore[attr-defined]

    out: list[str] = []
    try:
        for event in agent.chat(prompt):
            kind = getattr(event, "kind", "") or ""
            if kind in ("text", "text_delta"):
                out.append(getattr(event, "content", "") or "")
    except Exception:
        return []

    blob = "".join(out).strip()
    start = blob.find("[")
    end = blob.rfind("]")
    if start < 0 or end < 0 or end < start:
        return []
    try:
        records = json.loads(blob[start : end + 1])
    except json.JSONDecodeError:
        return []

    rules: list[ConstitutionalRule] = []
    if not isinstance(records, list):
        return []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        data = dict(rec)
        data.pop("source", None)
        if not data.get("id") or not data.get("text"):
            continue
        try:
            rules.append(ConstitutionalRule(source="user", **data))
        except Exception:
            continue
    return rules


def _summarize_transcript(raw: dict[str, Any]) -> str:
    """Flatten the last few turns of a session into a compact transcript string."""
    messages_raw: Any = raw.get("messages")
    if not isinstance(messages_raw, list):
        return ""
    messages: list[Any] = list(messages_raw)
    tail: list[Any] = messages[-40:] if len(messages) > 40 else messages
    lines: list[str] = []
    for m in tail:
        if not isinstance(m, dict):
            continue
        msg: dict[str, Any] = m
        role = str(msg.get("role", "?"))
        content_raw: Any = msg.get("content", "")
        if isinstance(content_raw, list):
            parts: list[str] = []
            content_list: list[Any] = list(content_raw)
            for c in content_list:
                if isinstance(c, dict):
                    c_dict: dict[str, Any] = c
                    text = c_dict.get("text") or c_dict.get("content") or ""
                    if text:
                        parts.append(str(text))
            content_text = " ".join(parts)
        else:
            content_text = str(content_raw)
        text_line = content_text[:400]
        lines.append(f"[{role}] {text_line}")
    return "\n".join(lines)


__all__ = ["commit_cmd"]
