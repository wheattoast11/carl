"""Tests for carl_core.connection.spec — pure data model."""
from __future__ import annotations

import pytest

from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)


def _spec(**over: object) -> ConnectionSpec:
    """Helper: build a spec with sensible defaults, overridable per-test."""
    base: dict[str, object] = {
        "name": "demo.echo",
        "scope": ConnectionScope.ONE_P,
        "kind": ConnectionKind.UTILITY,
        "direction": ConnectionDirection.BIDIRECTIONAL,
        "transport": ConnectionTransport.IN_PROCESS,
        "trust": ConnectionTrust.PUBLIC,
    }
    base.update(over)
    return ConnectionSpec(**base)  # type: ignore[arg-type]


def test_spec_frozen() -> None:
    spec = _spec()
    with pytest.raises(Exception):
        spec.name = "mutated"  # type: ignore[misc]


def test_spec_defaults() -> None:
    spec = _spec()
    assert spec.version == "0"
    assert spec.endpoint is None
    assert spec.metadata == {}


def test_spec_to_dict_contains_all_fields() -> None:
    spec = _spec(
        name="mcp.server.carl-studio",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.PROTOCOL,
        direction=ConnectionDirection.INGRESS,
        transport=ConnectionTransport.STDIO,
        trust=ConnectionTrust.AUTHENTICATED,
        version="2025-11-25",
        endpoint="stdio://",
        metadata={"extra": "data"},
    )
    d = spec.to_dict()
    assert d == {
        "name": "mcp.server.carl-studio",
        "scope": "3p",
        "kind": "protocol",
        "direction": "ingress",
        "transport": "stdio",
        "trust": "authenticated",
        "version": "2025-11-25",
        "endpoint": "stdio://",
        "metadata": {"extra": "data"},
    }


def test_spec_to_dict_copies_metadata() -> None:
    """Mutating the returned dict's metadata must not affect the spec."""
    md = {"a": 1}
    spec = _spec(metadata=md)
    out = spec.to_dict()
    assert isinstance(out["metadata"], dict)
    out["metadata"]["a"] = 999
    assert spec.metadata == {"a": 1}


def test_scope_values() -> None:
    assert ConnectionScope.ONE_P.value == "1p"
    assert ConnectionScope.THREE_P.value == "3p"


def test_trust_ordering() -> None:
    """Trust levels form a total order."""
    assert ConnectionTrust.PUBLIC < ConnectionTrust.AUTHENTICATED
    assert ConnectionTrust.AUTHENTICATED < ConnectionTrust.SIGNED
    assert ConnectionTrust.SIGNED < ConnectionTrust.METERED
    assert ConnectionTrust.METERED >= ConnectionTrust.PUBLIC
    assert ConnectionTrust.PUBLIC <= ConnectionTrust.METERED
    assert not (ConnectionTrust.PUBLIC > ConnectionTrust.SIGNED)


def test_trust_ordering_rejects_non_trust() -> None:
    """Comparisons against non-Trust values raise or return NotImplemented."""
    with pytest.raises(TypeError):
        _ = ConnectionTrust.PUBLIC < "authenticated"  # type: ignore[operator]


def test_kind_values_cover_expected_domains() -> None:
    kinds = {k.value for k in ConnectionKind}
    for expected in {
        "training",
        "environment",
        "protocol",
        "payment",
        "skill",
        "memory",
        "model",
        "utility",
    }:
        assert expected in kinds


def test_transport_values() -> None:
    # Every transport must have a non-empty lowercase value
    for t in ConnectionTransport:
        assert t.value
        assert t.value == t.value.lower()
