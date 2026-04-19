"""Tests for carl_core.connection.coherence — the ChannelCoherence primitive
that gives the Connection isomorphism claim a falsifiable observable.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from carl_core.connection.base import BaseConnection
from carl_core.connection.coherence import (
    ChannelCoherence,
    channel_coherence_diff,
    channel_coherence_distance,
)
from carl_core.connection.registry import reset_registry
from carl_core.connection.spec import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)


_SPEC = ConnectionSpec(
    name="demo.coherence",
    scope=ConnectionScope.ONE_P,
    kind=ConnectionKind.UTILITY,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.IN_PROCESS,
    trust=ConnectionTrust.PUBLIC,
)


class _CCConn(BaseConnection):
    spec = _SPEC

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)

    def _connect(self) -> None:
        pass

    def _close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    reset_registry()


def test_channel_coherence_empty() -> None:
    cc = ChannelCoherence.empty()
    assert cc.phi_mean == 0.0
    assert cc.cloud_quality == 0.0
    assert cc.success_rate == 0.0
    assert cc.latency_ms == 0.0
    assert cc.as_dict() == {
        "phi_mean": 0.0,
        "cloud_quality": 0.0,
        "success_rate": 0.0,
        "latency_ms": 0.0,
    }


def test_channel_coherence_from_mapping() -> None:
    cc = ChannelCoherence.from_mapping(
        {
            "phi_mean": 0.7,
            "cloud_quality": 0.4,
            "success_rate": 0.95,
            "latency_ms": 120.0,
        }
    )
    assert cc.phi_mean == pytest.approx(0.7)
    assert cc.cloud_quality == pytest.approx(0.4)
    assert cc.success_rate == pytest.approx(0.95)
    assert cc.latency_ms == pytest.approx(120.0)

    # Missing keys default to 0.0; values get coerced to float.
    cc2 = ChannelCoherence.from_mapping({"phi_mean": 1})
    assert cc2.phi_mean == 1.0
    assert cc2.cloud_quality == 0.0
    assert cc2.success_rate == 0.0
    assert cc2.latency_ms == 0.0


def test_channel_coherence_diff_pointwise() -> None:
    a = ChannelCoherence(
        phi_mean=0.8, cloud_quality=0.5, success_rate=1.0, latency_ms=100.0
    )
    b = ChannelCoherence(
        phi_mean=0.6, cloud_quality=0.5, success_rate=0.9, latency_ms=160.0
    )
    diff = channel_coherence_diff(a, b)
    assert diff["phi_mean"] == pytest.approx(0.2)
    assert diff["cloud_quality"] == pytest.approx(0.0)
    assert diff["success_rate"] == pytest.approx(0.1)
    assert diff["latency_ms"] == pytest.approx(60.0)


def test_channel_coherence_distance_is_euclidean_on_first_three() -> None:
    a = ChannelCoherence(
        phi_mean=0.8, cloud_quality=0.5, success_rate=1.0, latency_ms=100.0
    )
    b = ChannelCoherence(
        phi_mean=0.6, cloud_quality=0.5, success_rate=0.9, latency_ms=900.0
    )
    # sqrt(0.2^2 + 0.0^2 + 0.1^2) == sqrt(0.05); latency difference ignored.
    expected = math.sqrt(0.04 + 0.0 + 0.01)
    assert channel_coherence_distance(a, b) == pytest.approx(expected)

    # Identical observations -> zero distance.
    assert channel_coherence_distance(a, a) == pytest.approx(0.0)


def test_channel_coherence_nonfinite_yields_infinity() -> None:
    a = ChannelCoherence(phi_mean=float("nan"))
    b = ChannelCoherence(phi_mean=0.5)
    diff = channel_coherence_diff(a, b)
    assert diff["phi_mean"] == float("inf")
    assert diff["cloud_quality"] == 0.0

    # Also triggers the distance path (squaring inf stays inf).
    a_inf = ChannelCoherence(
        phi_mean=float("inf"), cloud_quality=0.0, success_rate=0.0
    )
    b_zero = ChannelCoherence()
    assert channel_coherence_distance(a_inf, b_zero) == float("inf")


def test_connection_publishes_and_reads_channel_coherence() -> None:
    conn = _CCConn()
    # Default is empty — makes the initial state observable & falsifiable.
    assert conn.channel_coherence() == ChannelCoherence.empty()

    new_cc = ChannelCoherence(
        phi_mean=0.42,
        cloud_quality=0.31,
        success_rate=0.9,
        latency_ms=75.0,
    )
    conn.publish_channel_coherence(new_cc)
    assert conn.channel_coherence() == new_cc

    # to_dict() surfaces the snapshot for logs / CLI.
    payload = conn.to_dict()
    assert payload["channel_coherence"] == new_cc.as_dict()
