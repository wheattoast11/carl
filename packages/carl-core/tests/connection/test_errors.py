"""Tests for carl_core.connection.errors — error hierarchy + redaction."""
from __future__ import annotations

from carl_core.connection.errors import (
    CARLConnectionError,
    ConnectionAuthError,
    ConnectionClosedError,
    ConnectionPolicyError,
    ConnectionTransitionError,
    ConnectionUnavailableError,
)
from carl_core.errors import CARLError


def test_connection_error_inherits_carl_error() -> None:
    """All connection errors descend from CARLError so existing
    carl.* catch sites pick them up."""
    assert issubclass(CARLConnectionError, CARLError)


def test_base_code() -> None:
    assert CARLConnectionError("boom").code == "carl.connection"


def test_subclass_codes() -> None:
    assert ConnectionUnavailableError("x").code == "carl.connection.unavailable"
    assert ConnectionAuthError("x").code == "carl.connection.auth"
    assert ConnectionTransitionError("x").code == "carl.connection.transition"
    assert ConnectionClosedError("x").code == "carl.connection.closed"
    assert ConnectionPolicyError("x").code == "carl.connection.policy"


def test_context_propagates() -> None:
    err = ConnectionTransitionError(
        "invalid",
        context={"from": "init", "to": "ready", "allowed": ["connecting"]},
    )
    assert err.context["from"] == "init"
    assert err.context["to"] == "ready"
    d = err.to_dict()
    assert d["code"] == "carl.connection.transition"
    assert d["context"]["from"] == "init"


def test_context_redacts_secrets() -> None:
    """Secret-shaped keys in context dicts are masked by CARLError.to_dict."""
    err = ConnectionAuthError(
        "bad creds",
        context={"api_key": "sk-xxx", "bearer_token": "abc", "host": "api.example.com"},
    )
    d = err.to_dict()
    assert d["context"]["api_key"] == "***REDACTED***"
    assert d["context"]["bearer_token"] == "***REDACTED***"
    assert d["context"]["host"] == "api.example.com"


def test_cause_chain_preserved() -> None:
    root = RuntimeError("root cause")
    err = ConnectionUnavailableError("cannot reach remote", cause=root)
    assert err.__cause__ is root
    d = err.to_dict()
    assert d["cause"]["type"] == "RuntimeError"
    assert d["cause"]["message"] == "root cause"
