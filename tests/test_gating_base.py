"""Tests for :class:`carl_studio.gating.BaseGate` — the shared gate loop.

These tests pin the generic mechanics (instantiation, enforce/decorator
surfaces, sync + async wrapping, ``functools.wraps`` preservation,
empty-chain no-op, multi-line detail passthrough). The semantics-specific
behavior (consent `code`, tier `feature/required/current` fields) lives in
``test_gating.py``, ``test_consent.py``, ``test_tier_gate.py``.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any

import pytest

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain

from carl_studio.gating import BaseGate, GatingPredicate


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _AllowPredicate:
    """Predicate that always allows. Implements :class:`GatingPredicate`."""

    @property
    def name(self) -> str:
        return "test:allow"

    def check(self) -> tuple[bool, str]:
        return True, "ok"


class _DenyPredicate:
    """Predicate that always denies with a configurable detail string."""

    def __init__(self, detail: str = "nope") -> None:
        self._detail = detail

    @property
    def name(self) -> str:
        return "test:deny"

    def check(self) -> tuple[bool, str]:
        return False, self._detail


class _MyError(CARLError):
    """Domain-specific error used to verify ``error_type`` routing."""

    code = "carl.test.denied"


GATE_TEST_DENIED = "carl.gate.test_denied"


# ---------------------------------------------------------------------------
# Generic instantiation
# ---------------------------------------------------------------------------


def test_base_gate_is_generic() -> None:
    """``BaseGate`` is parameterized by the predicate type."""
    assert hasattr(BaseGate, "__class_getitem__")
    # Subscript the generic — must not raise.
    specialized = BaseGate[_DenyPredicate]
    # The bound parameter surfaces on the specialized alias.
    assert specialized is not None


def test_predicates_satisfy_protocol() -> None:
    """Both test predicates structurally conform to :class:`GatingPredicate`."""
    assert isinstance(_AllowPredicate(), GatingPredicate)
    assert isinstance(_DenyPredicate(), GatingPredicate)


# ---------------------------------------------------------------------------
# ``enforce`` (call-site surface, used by ``consent_gate``)
# ---------------------------------------------------------------------------


def test_enforce_allow_does_not_raise() -> None:
    """Allow path returns ``None`` — no exception, no side-effects."""
    result = BaseGate.enforce(
        _AllowPredicate(),
        chain=None,
        error_factory=lambda detail: _MyError(detail, code="carl.test.denied"),
        gate_code=GATE_TEST_DENIED,
    )
    assert result is None


def test_enforce_deny_raises_error_type_with_context() -> None:
    """Deny path raises ``error_type`` with ``gate_code`` + ``gate_name``."""
    predicate = _DenyPredicate("bad state")
    with pytest.raises(_MyError) as excinfo:
        BaseGate.enforce(
            predicate,
            chain=None,
            error_factory=lambda detail: _MyError(
                detail, code="carl.test.denied"
            ),
            gate_code=GATE_TEST_DENIED,
        )
    err = excinfo.value
    # Original error ``code`` preserved (unchanged by BaseGate).
    assert err.code == "carl.test.denied"
    # Shared gate context layered on.
    assert err.context.get("gate_code") == GATE_TEST_DENIED
    assert err.context.get("gate_name") == "test:deny"
    assert err.context.get("gate_detail") == "bad state"


def test_enforce_emits_event_on_deny_path() -> None:
    """Deny path appends exactly one ``GATE_CHECK`` step to the chain."""
    chain = InteractionChain()
    with pytest.raises(_MyError):
        BaseGate.enforce(
            _DenyPredicate("bad"),
            chain=chain,
            error_factory=lambda detail: _MyError(detail),
            gate_code=GATE_TEST_DENIED,
        )
    assert len(chain) == 1
    step = chain.steps[0]
    assert step.action is ActionType.GATE_CHECK
    assert step.name == "test:deny"
    assert step.success is False


def test_enforce_emits_event_on_allow_path() -> None:
    """Allow path still appends a GATE_CHECK step (audit every check)."""
    chain = InteractionChain()
    BaseGate.enforce(
        _AllowPredicate(),
        chain=chain,
        error_factory=lambda detail: _MyError(detail),
        gate_code=GATE_TEST_DENIED,
    )
    assert len(chain) == 1
    assert chain.steps[0].success is True


def test_enforce_empty_chain_is_silent_noop() -> None:
    """``chain=None`` produces no event and no raise on allow."""
    # Just verifying it does not raise — there's no chain to inspect.
    BaseGate.enforce(
        _AllowPredicate(),
        chain=None,
        error_factory=lambda detail: _MyError(detail),
        gate_code=GATE_TEST_DENIED,
    )


def test_enforce_merges_extra_context_without_clobbering_gate_code() -> None:
    """``extra_context`` landed on error.context; gate_code still wins."""
    with pytest.raises(_MyError) as excinfo:
        BaseGate.enforce(
            _DenyPredicate("bad"),
            chain=None,
            error_factory=lambda detail: _MyError(
                detail, context={"flag": "telemetry"}
            ),
            gate_code=GATE_TEST_DENIED,
            extra_context={"my_extra": "x"},
        )
    err = excinfo.value
    # Constructor context preserved.
    assert err.context["flag"] == "telemetry"
    # Gate code added.
    assert err.context["gate_code"] == GATE_TEST_DENIED
    # Extra context folded in.
    assert err.context["my_extra"] == "x"


def test_enforce_multiline_detail_lands_verbatim() -> None:
    """A multi-line deny detail flows through to ``gate_detail`` unchanged."""
    detail = "line one\nline two\nline three"
    with pytest.raises(_MyError) as excinfo:
        BaseGate.enforce(
            _DenyPredicate(detail),
            chain=None,
            error_factory=lambda d: _MyError(d),
            gate_code=GATE_TEST_DENIED,
        )
    assert excinfo.value.context["gate_detail"] == detail


# ---------------------------------------------------------------------------
# ``for_predicate`` (decorator surface, used by ``tier_gate``)
# ---------------------------------------------------------------------------


def test_for_predicate_returns_decorator_factory() -> None:
    """``for_predicate`` yields a factory producing decorators."""
    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )
    decorator = engine()
    assert callable(decorator)


def test_for_predicate_allow_calls_wrapped_function() -> None:
    """When the predicate allows, the wrapped function runs normally."""
    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine()
    def do_it(x: int, y: int) -> int:
        return x + y

    assert do_it(2, 3) == 5


def test_for_predicate_deny_raises_without_calling_wrapped() -> None:
    """Deny path raises ``error_type`` and never invokes the wrapped fn."""
    calls: list[int] = []

    engine = BaseGate[_DenyPredicate].for_predicate(
        predicate_factory=_DenyPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine()
    def side_effect() -> None:
        calls.append(1)

    with pytest.raises(_MyError):
        side_effect()
    assert calls == []


def test_for_predicate_preserves_functools_wraps_metadata() -> None:
    """``functools.wraps`` is applied — ``__wrapped__`` + ``__name__`` flow."""
    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    def original(x: int) -> int:
        """Original docstring."""
        return x * 2

    wrapped = engine()(original)

    # functools.wraps copies these.
    assert wrapped.__name__ == "original"
    assert wrapped.__doc__ == "Original docstring."
    # And sets __wrapped__ for introspection.
    assert getattr(wrapped, "__wrapped__") is original


def test_for_predicate_async_support() -> None:
    """Async functions are detected and preserved as coroutines."""
    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine()
    async def do_async(x: int) -> int:
        return x + 1

    assert inspect.iscoroutinefunction(do_async)
    result = asyncio.run(do_async(3))
    assert result == 4


def test_for_predicate_async_deny_raises() -> None:
    """Async wrapper deny path raises before awaiting the wrapped coro."""
    engine = BaseGate[_DenyPredicate].for_predicate(
        predicate_factory=_DenyPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine()
    async def do_async() -> str:
        return "should-not-run"

    with pytest.raises(_MyError):
        asyncio.run(do_async())


def test_for_predicate_gate_chain_kwarg_threaded_and_consumed() -> None:
    """``_gate_chain`` is popped before the wrapped fn is called."""
    seen_kwargs: dict[str, Any] = {}

    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine()
    def observe(**kwargs: Any) -> None:
        seen_kwargs.update(kwargs)

    chain = InteractionChain()
    observe(_gate_chain=chain, x=1)

    # Wrapped fn never saw ``_gate_chain`` (it was consumed by the gate).
    assert "_gate_chain" not in seen_kwargs
    assert seen_kwargs == {"x": 1}
    # And the chain was used for the allow event.
    assert len(chain) == 1
    assert chain.steps[0].name == "test:allow"
    assert chain.steps[0].success is True


def test_for_predicate_factory_args_passed_through() -> None:
    """Args passed to the outer factory reach ``predicate_factory``."""
    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def spy_factory(*args: Any, **kwargs: Any) -> _AllowPredicate:
        captured.append((args, kwargs))
        return _AllowPredicate()

    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=spy_factory,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    @engine("arg_a", mode="b")
    def do_it() -> str:
        return "ran"

    assert do_it() == "ran"
    assert captured == [(("arg_a",), {"mode": "b"})]


def test_for_predicate_preserves_nested_wraps_across_chains() -> None:
    """A user's own ``functools.wraps`` decorator stays visible beneath."""
    engine = BaseGate[_AllowPredicate].for_predicate(
        predicate_factory=_AllowPredicate,
        error_type=_MyError,
        gate_code=GATE_TEST_DENIED,
    )

    def user_decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def inner(*a: Any, **k: Any) -> Any:
            return fn(*a, **k)

        return inner

    @engine()
    @user_decorator
    def target() -> str:
        """target doc"""
        return "ok"

    assert target.__name__ == "target"
    assert target.__doc__ == "target doc"
    assert target() == "ok"
