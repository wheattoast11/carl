"""Operation registry for ``carl flow`` slash-prefixed chains.

Each operation is ``(chain, args) -> chain``. Operations look up their
dependencies lazily — they may wrap existing Typer commands, Carl tools,
or pure functions. Keep them small and well-named; this registry is the
public surface for chainable actions.

Adding an operation:

    def simplify_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
        chain.record(ActionType.CLI_CMD, "simplify", input={"args": args})
        # ... do the thing ...
        return chain

    OPERATIONS["simplify"] = simplify_op
"""
from __future__ import annotations

from typing import Callable

from carl_core.interaction import ActionType, InteractionChain

Operation = Callable[[InteractionChain, list[str]], InteractionChain]

__all__ = ["OPERATIONS", "Operation", "get_operation", "list_operations"]


# ---------------------------------------------------------------------------
# Built-in ops — thin wrappers over existing commands
# ---------------------------------------------------------------------------


def _doctor_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    chain.record(ActionType.CLI_CMD, "carl doctor", input={"args": args})
    try:
        from carl_studio.cli.startup import doctor as doctor_cmd

        doctor_cmd(check_freshness=False, json_output=False)
    except SystemExit:
        pass
    return chain


def _start_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    chain.record(ActionType.CLI_CMD, "carl start", input={"args": args})
    try:
        from carl_studio.cli.startup import start as start_cmd

        start_cmd(inventory=False, json_output=False)
    except SystemExit:
        pass
    return chain


def _init_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    chain.record(ActionType.CLI_CMD, "carl init", input={"args": args})
    try:
        from carl_studio.cli.init import init_cmd

        init_cmd(skip_extras=False, skip_project=False, force=False, json_output=False)
    except SystemExit:
        pass
    return chain


def _ask_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    prompt = " ".join(args).strip() or "hello"
    chain.record(ActionType.CLI_CMD, "carl ask", input={"prompt": prompt[:120]})
    try:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)
    except SystemExit:
        pass
    return chain


def _freshness_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    chain.record(ActionType.CLI_CMD, "freshness", input={"args": args})
    try:
        from carl_studio.freshness import run_freshness_check

        report = run_freshness_check(force=True)
        chain.context["freshness"] = report.summary
    except Exception as exc:  # pragma: no cover - defensive
        chain.record(ActionType.CLI_CMD, "freshness_failed", output={"error": str(exc)}, success=False)
    return chain


def _echo_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Trivial pass-through — useful for tests and debugging chains."""
    from carl_studio.console import get_console

    msg = " ".join(args)
    get_console().info(f"[flow:echo] {msg}")
    chain.record(ActionType.CLI_CMD, "echo", input={"args": args}, output={"message": msg})
    return chain


OPERATIONS: dict[str, Operation] = {
    "doctor": _doctor_op,
    "start": _start_op,
    "init": _init_op,
    "ask": _ask_op,
    "freshness": _freshness_op,
    "echo": _echo_op,
}


def get_operation(name: str) -> Operation | None:
    """Return the op registered under ``name`` or None if unknown."""
    return OPERATIONS.get(name)


def list_operations() -> list[str]:
    """Return all registered op names, sorted."""
    return sorted(OPERATIONS.keys())
