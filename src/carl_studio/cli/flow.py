"""``carl flow`` — execute chained slash-prefixed operations.

Usage::

    carl flow "/doctor /freshness"
    carl flow "/ask 'train a small model on gsm8k'"
    carl flow "/echo hello /echo world"

Parsing rules:
  - Tokens starting with ``/`` are operation names.
  - Everything between one op token and the next is that op's argument list.
  - A missing op name or unknown op is a hard error (no silent drops).

Each executed step appends to an ``InteractionChain``. On completion the
chain is persisted to ``~/.carl/interactions/<chain_id>.jsonl`` so it can
be replayed, inspected, or fed back as training signal.
"""
from __future__ import annotations

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
) -> None:
    """Chain operations in sequence. Each step appends to a shared trace."""
    c = get_console()

    if list_ops:
        c.blank()
        c.header("Registered flow ops")
        for name in list_operations():
            c.info(f"  /{name}")
        c.blank()
        raise typer.Exit(0)

    if not chain:
        c.error("Missing chain. Example: carl flow '/doctor /freshness'")
        c.info("See registered ops with: carl flow --list")
        raise typer.Exit(1)

    try:
        steps = parse_chain(chain)
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(1) from exc

    if not steps:
        c.warn("No operations to run.")
        raise typer.Exit(0)

    if dry_run:
        c.blank()
        c.header("flow plan")
        for op, args in steps:
            arg_preview = " ".join(args) if args else ""
            c.info(f"  /{op} {arg_preview}".rstrip())
        c.blank()
        raise typer.Exit(0)

    trace = InteractionChain()
    trace.context["flow"] = chain

    c.blank()
    for op, args in steps:
        handler = get_operation(op)
        if handler is None:
            c.error(f"Unknown op: /{op}")
            trace.record(ActionType.CLI_CMD, f"unknown:/{op}", input={"args": args}, success=False)
            continue
        c.info(f"[flow] /{op}")
        try:
            handler(trace, args)
        except Exception as exc:  # pragma: no cover - defensive
            c.error(f"  /{op} raised: {exc}")
            trace.record(ActionType.CLI_CMD, f"error:/{op}", input={"args": args},
                         output={"error": str(exc)}, success=False)
    c.blank()

    if not no_persist:
        saved = _persist_trace(trace)
        c.info(f"Trace saved: {saved}")


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
