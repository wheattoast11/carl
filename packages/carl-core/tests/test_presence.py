"""v0.10 W12 — compose_presence_report() tests.

Peer-review verdict P2-1 (2026-04-20): "carl-sense" was a proposed new
primitive; review said it's 80% thin composition. This module implements
it as a composition helper, and these tests pin the contract.
"""

from __future__ import annotations

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_core.presence import PresenceReport, compose_presence_report


def _chain_with(samples: list[tuple[float, bool]]) -> InteractionChain:
    """samples = [(kuramoto_r, success), ...]"""
    chain = InteractionChain()
    for v, s in samples:
        chain.record(
            ActionType.LLM_REPLY,
            "test",
            input={},
            output={},
            success=s,
            kuramoto_r=v,
        )
    return chain


class TestColdStart:
    def test_none_chain_returns_cold_start(self) -> None:
        report = compose_presence_report(None)
        assert report.window_size == 0
        assert report.R == 1.0
        assert report.constructive is False
        assert "cold-start" in report.note

    def test_empty_chain_returns_cold_start(self) -> None:
        report = compose_presence_report(InteractionChain())
        assert report.window_size == 0
        assert report.constructive is False

    def test_chain_without_kuramoto_r_is_cold_start(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.CLI_CMD, "test", input={}, output={}, success=True)
        report = compose_presence_report(chain)
        assert report.window_size == 0


class TestPopulatedChain:
    def test_mean_R_over_window(self) -> None:
        chain = _chain_with([(0.2, True), (0.4, True), (0.6, True)])
        report = compose_presence_report(chain, window=8)
        assert report.window_size == 3
        assert report.R == pytest.approx((0.2 + 0.4 + 0.6) / 3)
        assert report.crystallization == pytest.approx(report.R)

    def test_constructive_true_when_last_step_succeeded(self) -> None:
        chain = _chain_with([(0.5, False), (0.5, True)])
        assert compose_presence_report(chain).constructive is True

    def test_constructive_false_when_last_step_failed(self) -> None:
        chain = _chain_with([(0.5, True), (0.5, False)])
        assert compose_presence_report(chain).constructive is False

    def test_recent_action_types_captured(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.CLI_CMD, "a", input={}, output={}, success=True, kuramoto_r=0.5)
        chain.record(ActionType.TOOL_CALL, "b", input={}, output={}, success=True, kuramoto_r=0.5)
        chain.record(ActionType.LLM_REPLY, "c", input={}, output={}, success=True, kuramoto_r=0.5)
        report = compose_presence_report(chain)
        # Action types are serialized as their string values
        assert report.recent_action_types == ["cli_cmd", "tool_call", "llm_reply"]

    def test_note_bucket_by_R(self) -> None:
        # diffuse (R < 0.3)
        diffuse = compose_presence_report(_chain_with([(0.1, True)]))
        assert "diffuse" in diffuse.note
        # liquid (0.3 <= R < 0.7)
        liquid = compose_presence_report(_chain_with([(0.5, True)]))
        assert "liquid" in liquid.note
        # stable (R >= 0.7)
        stable = compose_presence_report(_chain_with([(0.9, True)]))
        assert "stable" in stable.note

    def test_window_truncation(self) -> None:
        chain = _chain_with([(0.1, True)] * 5 + [(0.9, True)] * 2)
        # Only the last 2 should count
        report = compose_presence_report(chain, window=2)
        assert report.window_size == 2
        assert report.R == pytest.approx(0.9)


class TestSerialization:
    def test_report_is_frozen_dataclass(self) -> None:
        report = compose_presence_report(None)
        # All fields primitives + list[str] — safe to serialize
        assert isinstance(report, PresenceReport)
        # Frozen — cannot mutate
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            report.R = 0.0  # type: ignore[misc]

    def test_phase_populated_when_given(self) -> None:
        report = compose_presence_report(None, phase=0.5)
        assert report.has_phase is True
        assert report.psi == 0.5

    def test_phase_absent_by_default(self) -> None:
        report = compose_presence_report(None)
        assert report.has_phase is False
        assert report.psi == 0.0
