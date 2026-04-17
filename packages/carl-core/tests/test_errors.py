"""Tests for carl_core.errors — typed error hierarchy with redaction."""
from __future__ import annotations

import pytest

from carl_core.errors import (
    BudgetError,
    CARLError,
    CARLTimeoutError,
    ConfigError,
    CredentialError,
    NetworkError,
    PermissionError,
    ValidationError,
)


def test_carl_error_default_code() -> None:
    err = CARLError("boom")
    assert err.code == "carl.error"
    assert str(err) == "boom"
    assert err.context == {}


def test_subclass_codes() -> None:
    assert ConfigError("x").code == "carl.config"
    assert ValidationError("x").code == "carl.validation"
    assert CredentialError("x").code == "carl.credential"
    assert NetworkError("x").code == "carl.network"
    assert BudgetError("x").code == "carl.budget"
    assert PermissionError("x").code == "carl.permission"
    assert CARLTimeoutError("x").code == "carl.timeout"


def test_custom_code_overrides_class_default() -> None:
    err = ConfigError("missing", code="carl.config.missing_field")
    assert err.code == "carl.config.missing_field"
    assert ConfigError.code == "carl.config"


def test_context_passes_through() -> None:
    ctx = {"host": "api.example.com", "attempt": 3}
    err = NetworkError("timeout", context=ctx)
    assert err.context == ctx
    assert err.context is not ctx  # defensive copy


def test_cause_chain_from_raise_from() -> None:
    original = ValueError("bad input")
    try:
        try:
            raise original
        except ValueError as exc:
            raise ValidationError("invalid shape", cause=exc) from exc
    except ValidationError as err:
        assert err.__cause__ is original
        d = err.to_dict()
        assert d["cause"]["type"] == "ValueError"
        assert d["cause"]["message"] == "bad input"


def test_cause_via_kwarg_only() -> None:
    original = RuntimeError("upstream fail")
    err = NetworkError("downstream fail", cause=original)
    assert err.__cause__ is original


def test_to_dict_basic_shape() -> None:
    err = ConfigError("bad value", context={"field": "timeout_ms"})
    d = err.to_dict()
    assert d["code"] == "carl.config"
    assert d["message"] == "bad value"
    assert d["type"] == "ConfigError"
    assert d["context"] == {"field": "timeout_ms"}
    assert "cause" not in d
    assert "traceback" not in d


def test_to_dict_with_traceback() -> None:
    try:
        raise CARLError("kaboom")
    except CARLError as err:
        d = err.to_dict(include_traceback=True)
        assert "traceback" in d
        assert isinstance(d["traceback"], list)
        assert any("kaboom" in line for line in d["traceback"])


def test_sensitive_keys_redacted() -> None:
    err = CredentialError(
        "auth fail",
        context={
            "api_key": "sk-123",
            "auth_token": "bearer-xyz",
            "password": "hunter2",
            "secret": "shh",
            "Authorization": "Basic abc",
            "user_id": "u_42",
        },
    )
    d = err.to_dict()
    assert d["context"]["api_key"] == "***REDACTED***"
    assert d["context"]["auth_token"] == "***REDACTED***"
    assert d["context"]["password"] == "***REDACTED***"
    assert d["context"]["secret"] == "***REDACTED***"
    assert d["context"]["Authorization"] == "***REDACTED***"
    assert d["context"]["user_id"] == "u_42"


def test_nested_redaction_dict_in_dict() -> None:
    err = CARLError(
        "nested",
        context={
            "request": {
                "headers": {"authorization": "Bearer xyz"},
                "path": "/v1/run",
            },
            "public": "safe-value",
        },
    )
    d = err.to_dict()
    assert d["context"]["request"]["headers"]["authorization"] == "***REDACTED***"
    assert d["context"]["request"]["path"] == "/v1/run"
    assert d["context"]["public"] == "safe-value"


def test_nested_redaction_list_of_dicts() -> None:
    err = CARLError(
        "list",
        context={
            "calls": [
                {"api_key": "a", "ok": True},
                {"bearer_token": "b", "ok": False},
            ],
        },
    )
    d = err.to_dict()
    assert d["context"]["calls"][0]["api_key"] == "***REDACTED***"
    assert d["context"]["calls"][0]["ok"] is True
    assert d["context"]["calls"][1]["bearer_token"] == "***REDACTED***"
    assert d["context"]["calls"][1]["ok"] is False


def test_isinstance_exception() -> None:
    err = BudgetError("over cap")
    assert isinstance(err, Exception)
    assert isinstance(err, CARLError)
    with pytest.raises(Exception):
        raise err


def test_catchable_as_carlerror() -> None:
    with pytest.raises(CARLError) as excinfo:
        raise NetworkError("disconnect", context={"url": "https://x"})
    assert excinfo.value.code == "carl.network"
    assert excinfo.value.context["url"] == "https://x"


def test_original_context_not_mutated_by_to_dict() -> None:
    ctx = {"api_key": "sk-123", "fine": "ok"}
    err = CARLError("x", context=ctx)
    _ = err.to_dict()
    assert err.context == {"api_key": "sk-123", "fine": "ok"}


def test_permission_error_does_not_shadow_builtin() -> None:
    # CARL's PermissionError is distinct from the stdlib
    import builtins
    assert PermissionError is not builtins.PermissionError
    assert issubclass(PermissionError, CARLError)


def test_timeout_error_is_carlerror_not_stdlib() -> None:
    import builtins
    assert CARLTimeoutError is not builtins.TimeoutError
    assert issubclass(CARLTimeoutError, CARLError)
