"""``carl flow`` — execute chained slash-prefixed operations.

Usage::

    carl flow "/doctor /freshness"
    carl flow "/ask 'train a small model on gsm8k'"
    carl flow "/echo hello /echo world"
    carl flow "/doctor /freshness" --json
    carl flow "/nope /echo after" --no-continue-on-failure

Parsing rules:
  - Tokens starting with ``/`` are operation names.
  - Everything between one op token and the next is that op's argument list.
  - A missing op name or unknown op is a hard error (no silent drops).

Each executed step appends to an ``InteractionChain``. On completion the
chain is persisted to ``~/.carl/interactions/<chain_id>.jsonl`` so it can
be replayed, inspected, or fed back as training signal.

Exit codes:
  - 0 if every step in the chain recorded ``success=True``.
  - 1 if any step failed (unknown ops, ops that raise ``SystemExit(!=0)``, or
    ops that raise any other exception). Chain execution continues through
    remaining ops unless ``--no-continue-on-failure`` is passed.
"""
from __future__ import annotations

import json as _json
import shlex
from pathlib import Path

import typer

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.console import get_console
from carl_studio.settings import CARL_HOME

from .operations import get_operation, list_operations

INTERACTIONS_DIR = CARL_HOME / "interactions"


def flow_cmd(
    chain: str = typer.Argument("", help='Slash-separated ops, e.g. "/doctor /freshness"'),
    list_ops: bool = typer.Option(False, "--list", help="List registered operations and exit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and print the plan, don't execute"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Don't write the chain trace to disk"),
    json_output: bool = typer.Option(False, "--json", help="Emit chain + results as JSON"),
    no_continue_on_failure: bool = typer.Option(
        False,
        "--no-continue-on-failure",
        help="Abort the chain on first failure instead of running remaining ops",
    ),
) -> None:
    """Chain operations in sequence. Each step appends to a shared trace."""
    c = get_console()

    if list_ops:
        if json_output:
            typer.echo(_json.dumps({"ops": list_operations()}, indent=2))
        else:
            c.blank()
            c.header("Registered flow ops")
            for name in list_operations():
                c.info(f"  /{name}")
            c.blank()
        raise typer.Exit(0)

    if not chain:
        if json_output:
            typer.echo(
                _json.dumps(
                    {
                        "error": "missing chain",
                        "hint": "See registered ops with: carl flow --list",
                    },
                    indent=2,
                )
            )
        else:
            c.error("Missing chain. Example: carl flow '/doctor /freshness'")
            c.info("See registered ops with: carl flow --list")
        raise typer.Exit(1)

    try:
        steps = parse_chain(chain)
    except ValueError as exc:
        if json_output:
            typer.echo(_json.dumps({"error": str(exc)}, indent=2))
        else:
            c.error(str(exc))
        raise typer.Exit(1) from exc

    if not steps:
        if json_output:
            typer.echo(_json.dumps({"ops": [], "note": "no operations to run"}, indent=2))
        else:
            c.warn("No operations to run.")
        raise typer.Exit(0)

    if dry_run:
        if json_output:
            typer.echo(
                _json.dumps(
                    {"plan": [{"op": op, "args": list(args)} for op, args in steps]},
                    indent=2,
                )
            )
        else:
            c.blank()
            c.header("flow plan")
            for op, args in steps:
                arg_preview = " ".join(args) if args else ""
                c.info(f"  /{op} {arg_preview}".rstrip())
            c.blank()
        raise typer.Exit(0)

    trace = InteractionChain()
    trace.context["flow"] = chain
    aborted = False

    if not json_output:
        c.blank()

    for op, args in steps:
        handler = get_operation(op)
        if handler is None:
            if not json_output:
                c.error(f"Unknown op: /{op}")
            trace.record(
                ActionType.CLI_CMD,
                f"unknown:/{op}",
                input={"args": args},
                success=False,
                output={"error": f"no such op: /{op}"},
            )
            if no_continue_on_failure:
                aborted = True
                break
            continue

        if not json_output:
            c.info(f"[flow] /{op}")

        try:
            handler(trace, args)
        except BaseException as exc:  # noqa: BLE001 — cross-op isolation
            # Well-behaved ops never escape with an exception: ``_run`` wraps
            # invocation and records failure on the chain. This branch covers
            # ops that bypass ``_run`` (e.g. the inline-echo op).
            if not json_output:
                c.error(f"  /{op} raised: {exc}")
            trace.record(
                ActionType.CLI_CMD,
                f"error:/{op}",
                input={"args": args},
                output={
                    "error": str(exc),
                    "type": exc.__class__.__name__,
                },
                success=False,
            )

        last = trace.last()
        if last is not None and not last.success and no_continue_on_failure:
            aborted = True
            break

    if not json_output:
        c.blank()

    saved_path: Path | None = None
    if not no_persist:
        saved_path = _persist_trace(trace)

    failed_steps = sum(1 for s in trace.steps if not s.success)
    exit_code = 0 if failed_steps == 0 else 1

    if json_output:
        payload = trace.to_dict()
        payload["summary"] = {
            "total": len(trace),
            "failed": failed_steps,
            "success_rate": trace.success_rate(),
            "aborted": aborted,
            "exit_code": exit_code,
        }
        if saved_path is not None:
            payload["trace_path"] = str(saved_path)
        typer.echo(_json.dumps(payload, indent=2, default=str))
    else:
        if saved_path is not None:
            c.info(f"Trace saved: {saved_path}")
        if failed_steps > 0:
            c.warn(f"{failed_steps}/{len(trace)} step(s) failed.")

    if exit_code != 0:
        raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_chain(raw: str) -> list[tuple[str, list[str]]]:
    """Parse a flow string into ``[(op, args), ...]``.

    Examples::

        parse_chain("/doctor")                      -> [("doctor", [])]
        parse_chain("/ask 'train a model'")         -> [("ask", ["train a model"])]
        parse_chain("/echo hello /echo world")      -> [("echo", ["hello"]), ("echo", ["world"])]

    Raises ``ValueError`` on:
      - empty input
      - a trailing arg with no preceding op
      - malformed quoting (passed through from ``shlex``)
    """
    tokens = shlex.split(raw)
    if not tokens:
        raise ValueError("flow string is empty")

    steps: list[tuple[str, list[str]]] = []
    current_op: str | None = None
    current_args: list[str] = []

    for tok in tokens:
        if tok.startswith("/") and len(tok) > 1:
            if current_op is not None:
                steps.append((current_op, current_args))
            current_op = tok[1:]
            current_args = []
        else:
            if current_op is None:
                raise ValueError(f"Argument {tok!r} appears before any /op")
            current_args.append(tok)

    if current_op is not None:
        steps.append((current_op, current_args))

    return steps


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_trace(chain: InteractionChain) -> Path:
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERACTIONS_DIR / f"{chain.chain_id}.jsonl"
    path.write_text(chain.to_jsonl())
    return path
