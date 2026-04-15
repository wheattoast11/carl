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

import json
import sys
import time
import uuid
from typing import Optional

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

    try:
        parsed_inputs: dict = json.loads(inputs)
    except json.JSONDecodeError as exc:
        _err(f"--inputs is not valid JSON: {exc}")
        raise typer.Exit(code=1) from exc

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
