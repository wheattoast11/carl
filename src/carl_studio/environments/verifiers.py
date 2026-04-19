"""Prime Intellect Verifiers compatibility shim.

The ``verifiers`` package defines two primitives — ``Parser`` and
``Rubric`` — that together define a reward function. This module adapts
that pair to carl-studio's reward surface so environments can borrow
Verifiers rubrics without forking them.

Duck-typed protocols
--------------------

We do not import from ``verifiers`` at module load time. Instead, both
protocols accept any object that implements the stated methods. This
keeps the adapter usable with:

* the real ``verifiers`` package (when installed),
* user-defined parser/rubric pairs that predate the package,
* test doubles.

Gating
------

:func:`from_prime_verifier` is the only entrypoint that actually imports
``verifiers``. If the package is missing it raises
:class:`ConnectionUnavailableError` with a clear install hint.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from carl_core.connection import ConnectionUnavailableError


@runtime_checkable
class VerifierParser(Protocol):
    """Duck-typed Parser contract.

    A valid parser turns a raw model response into a structured dict and
    decides whether that dict is well-formed.
    """

    def parse(self, response: str) -> dict[str, Any]:
        """Structure a raw response. MUST NOT raise on garbage input —
        return an empty or minimal dict instead so the rubric can score 0.
        """
        ...

    def validate(self, parsed: dict[str, Any]) -> bool:
        """Return True if the parsed output satisfies the format contract."""
        ...


@runtime_checkable
class VerifierRubric(Protocol):
    """Duck-typed Rubric contract.

    Produces a reward in ``[0.0, 1.0]`` from a parsed response. The
    ``target`` kwarg is optional so rubrics can be reference-free
    (e.g. self-consistency checks) or reference-based (e.g. exact match).
    """

    def score(
        self,
        parsed: dict[str, Any],
        *,
        target: dict[str, Any] | None = None,
    ) -> float:
        ...


class VerifiersAdapter:
    """Wrap a ``(parser, rubric)`` pair as a carl-studio reward callable.

    The adapter clamps to ``[0.0, 1.0]``, catches rubric errors, and
    refuses to score responses the parser rejects. It is intentionally
    tiny — no state, no I/O — so it can be mixed into ``training/rewards``
    without ceremony.
    """

    def __init__(
        self,
        parser: VerifierParser,
        rubric: VerifierRubric,
        *,
        reject_score: float = 0.0,
        error_score: float = 0.0,
    ) -> None:
        if not _duck_check(parser, ("parse", "validate")):
            raise TypeError(
                "VerifiersAdapter parser must define parse() and validate() methods",
            )
        if not _duck_check(rubric, ("score",)):
            raise TypeError("VerifiersAdapter rubric must define a score() method")
        self._parser = parser
        self._rubric = rubric
        self._reject_score = float(reject_score)
        self._error_score = float(error_score)

    @property
    def parser(self) -> VerifierParser:
        return self._parser

    @property
    def rubric(self) -> VerifierRubric:
        return self._rubric

    def compute_reward(
        self,
        response: str,
        target: dict[str, Any] | None = None,
    ) -> float:
        """Run parser + rubric; clamp to [0.0, 1.0].

        * If the parser rejects the output, return ``reject_score`` (0.0).
        * If the rubric raises, swallow and return ``error_score`` (0.0).
        """
        if not isinstance(response, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self._error_score
        try:
            parsed_raw: Any = self._parser.parse(response)
        except Exception:  # noqa: BLE001 — parsers must not crash us
            return self._error_score
        if not isinstance(parsed_raw, dict):
            return self._reject_score
        parsed: dict[str, Any] = parsed_raw  # type: ignore[assignment]
        try:
            if not self._parser.validate(parsed):
                return self._reject_score
        except Exception:  # noqa: BLE001
            return self._error_score
        try:
            raw: Any = self._rubric.score(parsed, target=target)
        except Exception:  # noqa: BLE001
            return self._error_score
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self._error_score
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    # Enable ``adapter(response, target)`` as well as
    # ``adapter.compute_reward(response, target)`` so existing
    # training/rewards callsites that expect a plain callable work.
    def __call__(
        self,
        response: str,
        target: dict[str, Any] | None = None,
    ) -> float:
        return self.compute_reward(response, target)


# ---------------------------------------------------------------------------
# Prime Verifiers registry pull
# ---------------------------------------------------------------------------


def from_prime_verifier(
    verifier_name: str,
    *,
    token: str | None = None,  # noqa: ARG001 — reserved for future registry auth
) -> VerifiersAdapter:
    """Load a Prime Intellect Verifier by name and wrap it.

    Gates on the ``verifiers`` package being installed; raises
    :class:`ConnectionUnavailableError` with an install hint otherwise.

    The verifiers package exposes rubrics either as ``(parser, rubric)``
    tuples or as a combined ``Environment``-like object. We handle both
    shapes.
    """
    if not isinstance(verifier_name, str) or not verifier_name:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("verifier_name must be a non-empty string")

    try:
        import verifiers as _verifiers  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ConnectionUnavailableError(
            "The 'verifiers' package is not installed. "
            "Install with: pip install verifiers",
            context={"package": "verifiers", "verifier_name": verifier_name},
            cause=exc,
        ) from exc

    # Prime Verifiers exposes ``load_environment`` in modern versions and
    # ``load_verifier`` in older ones. Try both; give up gracefully.
    loader: Any = getattr(_verifiers, "load_environment", None) or getattr(
        _verifiers, "load_verifier", None,
    )
    if loader is None or not callable(loader):
        raise ConnectionUnavailableError(
            "verifiers package is installed but does not expose a "
            "load_environment/load_verifier entrypoint",
            context={
                "package": "verifiers",
                "verifier_name": verifier_name,
                "has": sorted(n for n in dir(_verifiers) if not n.startswith("_"))[:32],
            },
        )

    try:
        loaded: Any = loader(verifier_name)
    except Exception as exc:  # noqa: BLE001
        raise ConnectionUnavailableError(
            f"Failed to load verifier '{verifier_name}'",
            context={"verifier_name": verifier_name, "error": str(exc)[:500]},
            cause=exc,
        ) from exc

    parser, rubric = _unpack_prime_result(loaded, verifier_name)
    return VerifiersAdapter(parser, rubric)


def _unpack_prime_result(
    loaded: Any,
    verifier_name: str,
) -> tuple[VerifierParser, VerifierRubric]:
    """Unpack the variety of shapes Prime Verifiers returns."""
    # Shape 1: already a (parser, rubric) tuple.
    if isinstance(loaded, tuple):
        tup: tuple[Any, ...] = cast(tuple[Any, ...], loaded)
        if len(tup) == 2:
            parser_candidate: Any = tup[0]
            rubric_candidate: Any = tup[1]
            if _duck_check(parser_candidate, ("parse", "validate")) and _duck_check(
                rubric_candidate, ("score",),
            ):
                return parser_candidate, rubric_candidate

    # Shape 2: an Environment-like object with .parser and .rubric attrs.
    container: Any = cast(Any, loaded)
    parser_attr: Any = getattr(container, "parser", None)
    rubric_attr: Any = getattr(container, "rubric", None)
    if parser_attr is not None and rubric_attr is not None:
        if _duck_check(parser_attr, ("parse", "validate")) and _duck_check(
            rubric_attr, ("score",),
        ):
            return parser_attr, rubric_attr

    raise ConnectionUnavailableError(
        f"Verifier '{verifier_name}' does not expose a (parser, rubric) pair "
        f"that matches the VerifiersAdapter protocol",
        context={
            "verifier_name": verifier_name,
            "loaded_type": type(container).__name__,
        },
    )


def _duck_check(obj: Any, methods: tuple[str, ...]) -> bool:
    """Return True if ``obj`` has all named methods as callables."""
    if obj is None:
        return False
    for name in methods:
        attr = getattr(obj, name, None)
        if attr is None or not callable(attr):
            return False
    return True


__all__ = [
    "VerifierParser",
    "VerifierRubric",
    "VerifiersAdapter",
    "from_prime_verifier",
]
