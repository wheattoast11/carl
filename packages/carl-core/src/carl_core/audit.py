"""Audit decorator + ``InteractionChain`` contextvar.

The imperative form for emitting an audit step today is::

    chain.record(
        ActionType.TRAINING_STEP,
        "pipeline.train_step",
        input=batch_meta(batch),
        output={"loss": loss, "lr": lr},
        duration_ms=dur_ms,
        success=True,
    )

Every call site carries the chain reference + the `ActionType` + the naming
logic + start-time bookkeeping + the try/except + the `record` call itself.
That's ~5 lines per site and miss-one-and-you-get-an-audit-hole.

This module flips the emission to declarative:

    @audited(ActionType.TRAINING_STEP, name_fn=lambda self, batch: f"step.{batch.id}")
    def train_step(self, batch: Batch) -> TrainResult: ...

The decorator:

1. Pulls the active chain from :data:`CURRENT_CHAIN` (``contextvars.ContextVar``).
2. Times the call via ``time.perf_counter`` — delta becomes ``duration_ms``.
3. Records success + result on return, failure + exception-repr on raise.
4. Skips emission cleanly when no chain is bound (library consumers in
   unit tests don't get unwanted audit noise).

Bind a chain for a scope via :func:`chain_context`::

    with chain_context(agent.chain):
        result = train_step(batch)  # step lands in agent.chain

The contextvar carries across ``asyncio`` boundaries as :mod:`contextvars`
is designed for. Sync callers inside the same process also inherit it
naturally via stack scope.

Why a contextvar vs. a thread-local: ``asyncio.run`` + structured
concurrency need the current chain to propagate into spawned tasks,
which thread-locals don't give us.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, ParamSpec, TypeVar

from carl_core.interaction import ActionType, InteractionChain


__all__ = [
    "CURRENT_CHAIN",
    "audited",
    "chain_context",
    "get_current_chain",
    "set_current_chain",
]


P = ParamSpec("P")
R = TypeVar("R")


# The active chain for the current task/thread. ``None`` when unbound —
# decorated calls skip audit emission cleanly in that case so library
# consumers (unit tests, one-off scripts) don't accumulate noise.
CURRENT_CHAIN: ContextVar[InteractionChain | None] = ContextVar(
    "carl_core.audit.current_chain", default=None,
)


def get_current_chain() -> InteractionChain | None:
    """Read the chain bound in the current context, or ``None``."""
    return CURRENT_CHAIN.get()


def set_current_chain(chain: InteractionChain | None) -> Token[InteractionChain | None]:
    """Set the contextvar directly and return the reset token.

    Prefer :func:`chain_context` which handles the reset in a ``finally``.
    Use this helper when the chain lifecycle outlasts a single scope
    (e.g. long-running daemons setting once at startup).
    """
    return CURRENT_CHAIN.set(chain)


@contextmanager
def chain_context(chain: InteractionChain) -> Iterator[InteractionChain]:
    """Bind ``chain`` as the active one for the duration of the ``with`` block.

    Reset to the prior value on exit — supports nesting + exceptions.

    Example::

        with chain_context(agent.chain):
            result = train_step(batch)   # @audited step lands in agent.chain
            with chain_context(sub_chain):
                sub_op()                 # lands in sub_chain
            another_op()                 # back to agent.chain
    """
    token = CURRENT_CHAIN.set(chain)
    try:
        yield chain
    finally:
        CURRENT_CHAIN.reset(token)


def audited(
    action: ActionType,
    *,
    name: str | None = None,
    name_fn: Callable[..., str] | None = None,
    input_fn: Callable[..., Any] | None = None,
    output_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Emit an :class:`~carl_core.interaction.Step` around the wrapped call.

    When the contextvar has no bound chain, the wrapper is a no-op pass-through —
    the wrapped function runs normally and returns its result without any
    audit side effect. This keeps tests + library consumers clean.

    Parameters
    ----------
    action
        The :class:`ActionType` the emitted step takes.
    name
        A static step name. Mutually exclusive with ``name_fn``.
    name_fn
        ``(*args, **kwargs) → str`` — called with the wrapped function's
        arguments to produce a per-call step name. Overrides ``name``.
    input_fn
        ``(*args, **kwargs) → Any`` — produces the step's ``input``. Defaults
        to ``None`` (no input captured beyond args count).
    output_fn
        ``(result) → Any`` — produces the step's ``output``. Defaults to
        ``repr(result)`` truncated to 200 chars. Caller should override
        when the return value contains secrets or is too large.

    Failure emits a step with ``success=False`` and the exception's
    ``type(exc).__name__: exc`` string as output. The exception re-raises;
    decorator never swallows.
    """
    if name is not None and name_fn is not None:
        raise ValueError("audited: pass one of `name` or `name_fn`, not both")

    def _decorate(fn: Callable[P, R]) -> Callable[P, R]:
        fn_name = name if name is not None else fn.__name__

        @functools.wraps(fn)
        def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            chain = CURRENT_CHAIN.get()
            if chain is None:
                return fn(*args, **kwargs)

            step_name = name_fn(*args, **kwargs) if name_fn is not None else fn_name
            step_input: Any = None
            if input_fn is not None:
                try:
                    step_input = input_fn(*args, **kwargs)
                except Exception as exc:
                    step_input = {"__input_fn_error__": f"{type(exc).__name__}: {exc}"}

            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                chain.record(
                    action,
                    step_name,
                    input=step_input,
                    output=f"{type(exc).__name__}: {exc}",
                    duration_ms=duration_ms,
                    success=False,
                )
                raise

            duration_ms = (time.perf_counter() - started) * 1000.0
            step_output: Any
            if output_fn is not None:
                try:
                    step_output = output_fn(result)
                except Exception as exc:
                    step_output = {"__output_fn_error__": f"{type(exc).__name__}: {exc}"}
            else:
                try:
                    step_output = repr(result)[:200]
                except Exception:
                    step_output = f"<{type(result).__name__} repr failed>"

            chain.record(
                action,
                step_name,
                input=step_input,
                output=step_output,
                duration_ms=duration_ms,
                success=True,
            )
            return result

        return _wrapped

    return _decorate
