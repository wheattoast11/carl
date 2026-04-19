"""Tests for the Prime Verifiers compatibility shim."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from carl_core.connection import ConnectionUnavailableError

from carl_studio.environments import verifiers as verifiers_mod
from carl_studio.environments.verifiers import (
    VerifiersAdapter,
    from_prime_verifier,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _EchoParser:
    """Echoes the response as a dict."""

    def parse(self, response: str) -> dict[str, Any]:
        return {"answer": response.strip()}

    def validate(self, parsed: dict[str, Any]) -> bool:
        return "answer" in parsed and isinstance(parsed["answer"], str)


class _ExactMatchRubric:
    def score(self, parsed: dict[str, Any], *, target: dict[str, Any] | None = None) -> float:
        if target is None:
            return 0.0
        return 1.0 if parsed.get("answer") == target.get("answer") else 0.0


class _RaisingParser:
    def parse(self, response: str) -> dict[str, Any]:
        raise RuntimeError("boom")

    def validate(self, parsed: dict[str, Any]) -> bool:
        return True


class _RejectingParser:
    def parse(self, response: str) -> dict[str, Any]:
        return {"answer": response.strip()}

    def validate(self, parsed: dict[str, Any]) -> bool:
        return False


class _OutOfRangeRubric:
    def score(self, parsed: dict[str, Any], *, target: dict[str, Any] | None = None) -> float:  # noqa: ARG002
        return 42.0


class _RaisingRubric:
    def score(self, parsed: dict[str, Any], *, target: dict[str, Any] | None = None) -> float:  # noqa: ARG002
        raise RuntimeError("rubric crash")


# ---------------------------------------------------------------------------
# VerifiersAdapter
# ---------------------------------------------------------------------------


class TestVerifiersAdapter:
    def test_requires_parser_with_parse_and_validate(self):
        with pytest.raises(TypeError):
            VerifiersAdapter(object(), _ExactMatchRubric())  # type: ignore[arg-type]

    def test_requires_rubric_with_score(self):
        with pytest.raises(TypeError):
            VerifiersAdapter(_EchoParser(), object())  # type: ignore[arg-type]

    def test_exact_match(self):
        adapter = VerifiersAdapter(_EchoParser(), _ExactMatchRubric())
        reward = adapter.compute_reward("42", target={"answer": "42"})
        assert reward == 1.0

    def test_mismatch(self):
        adapter = VerifiersAdapter(_EchoParser(), _ExactMatchRubric())
        reward = adapter.compute_reward("42", target={"answer": "99"})
        assert reward == 0.0

    def test_parser_rejection_yields_reject_score(self):
        adapter = VerifiersAdapter(_RejectingParser(), _ExactMatchRubric(), reject_score=0.0)
        reward = adapter.compute_reward("anything", target={"answer": "anything"})
        assert reward == 0.0

    def test_parser_exception_yields_error_score(self):
        adapter = VerifiersAdapter(_RaisingParser(), _ExactMatchRubric())
        reward = adapter.compute_reward("x", target={"answer": "x"})
        assert reward == 0.0

    def test_rubric_exception_yields_error_score(self):
        adapter = VerifiersAdapter(_EchoParser(), _RaisingRubric())
        reward = adapter.compute_reward("x", target={"answer": "x"})
        assert reward == 0.0

    def test_clamps_out_of_range_to_upper_bound(self):
        adapter = VerifiersAdapter(_EchoParser(), _OutOfRangeRubric())
        reward = adapter.compute_reward("x", target={"answer": "x"})
        assert reward == 1.0

    def test_non_string_response_yields_error_score(self):
        adapter = VerifiersAdapter(_EchoParser(), _ExactMatchRubric())
        reward = adapter.compute_reward(42)  # type: ignore[arg-type]
        assert reward == 0.0

    def test_callable_wrapper(self):
        adapter = VerifiersAdapter(_EchoParser(), _ExactMatchRubric())
        assert adapter("7", target={"answer": "7"}) == 1.0


# ---------------------------------------------------------------------------
# from_prime_verifier
# ---------------------------------------------------------------------------


class TestFromPrimeVerifier:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            from_prime_verifier("")

    def test_missing_package_raises_connection_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Ensure any real "verifiers" module is hidden.
        monkeypatch.setitem(sys.modules, "verifiers", None)  # triggers ImportError
        with pytest.raises(ConnectionUnavailableError, match="verifiers"):
            from_prime_verifier("any-name")

    def test_loads_environment_shape(self, monkeypatch: pytest.MonkeyPatch):
        # Build a stub ``verifiers`` module exposing load_environment.
        fake = types.ModuleType("verifiers")
        parser = _EchoParser()
        rubric = _ExactMatchRubric()

        class _Env:
            pass

        env = _Env()
        env.parser = parser  # type: ignore[attr-defined]
        env.rubric = rubric  # type: ignore[attr-defined]

        def _load(name: str) -> Any:
            assert name == "demo-verifier"
            return env

        fake.load_environment = _load  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "verifiers", fake)

        adapter = from_prime_verifier("demo-verifier")
        assert adapter.compute_reward("x", target={"answer": "x"}) == 1.0

    def test_loads_tuple_shape(self, monkeypatch: pytest.MonkeyPatch):
        fake = types.ModuleType("verifiers")

        def _load(name: str) -> Any:  # noqa: ARG001
            return (_EchoParser(), _ExactMatchRubric())

        fake.load_verifier = _load  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "verifiers", fake)

        adapter = from_prime_verifier("demo-verifier-tuple")
        assert adapter.compute_reward("y", target={"answer": "y"}) == 1.0

    def test_invalid_shape_raises_connection_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        fake = types.ModuleType("verifiers")

        def _load(name: str) -> Any:  # noqa: ARG001
            return object()

        fake.load_environment = _load  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "verifiers", fake)

        with pytest.raises(ConnectionUnavailableError, match="parser, rubric"):
            from_prime_verifier("weird-shape")

    def test_no_loader_attribute_raises(self, monkeypatch: pytest.MonkeyPatch):
        fake = types.ModuleType("verifiers")
        # Neither load_environment nor load_verifier on the module.
        monkeypatch.setitem(sys.modules, "verifiers", fake)
        with pytest.raises(ConnectionUnavailableError, match="entrypoint"):
            from_prime_verifier("nope")

    def test_loader_exception_wraps(self, monkeypatch: pytest.MonkeyPatch):
        fake = types.ModuleType("verifiers")

        def _load(name: str) -> Any:  # noqa: ARG001
            raise RuntimeError("registry down")

        fake.load_environment = _load  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "verifiers", fake)
        with pytest.raises(ConnectionUnavailableError, match="Failed to load"):
            from_prime_verifier("failing-verifier")


# ---------------------------------------------------------------------------
# Protocol contracts
# ---------------------------------------------------------------------------


class TestProtocolRuntimeCheck:
    def test_echo_parser_satisfies_protocol(self):
        parser = _EchoParser()
        assert isinstance(parser, verifiers_mod.VerifierParser)

    def test_exact_match_rubric_satisfies_protocol(self):
        rubric = _ExactMatchRubric()
        assert isinstance(rubric, verifiers_mod.VerifierRubric)
