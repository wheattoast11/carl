"""v0.10 Fano-consensus peer-review fixes (2026-04-20).

Pins the four P1 actions coalesced from the K_7 complete-graph peer
review:

- V1 boundedness: AgentCardStore.list_all(limit, offset) pagination +
  MarketplaceAgentCard.capabilities max_length=100 cap.
- V4 contrastive: CoherenceSnapshot.is_degenerate + variance field.
- V7 gate realization: end-to-end integration test showing a coherence
  probe feeding kuramoto_r into Step records that then drive a
  coherence_gate decision — proves the primitive is usable, not just
  library code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.a2a.marketplace import (
    AgentCardStore,
    MarketplaceAgentCard,
    compute_content_hash,
    new_agent_id,
)
from carl_studio.gating import (
    CoherenceError,
    CoherenceGatePredicate,
    CoherenceSnapshot,
    coherence_gate,
    read_chain_coherence,
)


# ---------------------------------------------------------------------------
# V1 boundedness — AgentCardStore pagination
# ---------------------------------------------------------------------------


class _TinyDB:
    def __init__(self, path: Path) -> None:
        self._path = path

    def connect(self) -> Any:
        import sqlite3

        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn


def _make_card(n: int) -> MarketplaceAgentCard:
    caps = [{"id": f"cap{n}", "description": f"#{n}"}]
    return MarketplaceAgentCard(
        agent_id=new_agent_id(),
        agent_url=f"https://carl.camp/agents/agent-{n}",
        agent_name=f"Agent {n}",
        description=None,
        capabilities=caps,
        public_key=None,
        content_hash=compute_content_hash(
            agent_name=f"Agent {n}",
            description=None,
            capabilities=caps,
            public_key=None,
            is_active=True,
            rate_limit_rpm=60,
        ),
        is_active=True,
        rate_limit_rpm=60,
    )


class TestListPagination:
    def test_limit_bounds_row_count(self, tmp_path: Path) -> None:
        store = AgentCardStore(_TinyDB(tmp_path / "p.db"))
        for i in range(12):
            store.save(_make_card(i))
        page = store.list_all(limit=5, offset=0)
        assert len(page) == 5

    def test_offset_skips(self, tmp_path: Path) -> None:
        store = AgentCardStore(_TinyDB(tmp_path / "p.db"))
        for i in range(12):
            store.save(_make_card(i))
        first_5 = store.list_all(limit=5, offset=0)
        next_5 = store.list_all(limit=5, offset=5)
        first_ids = {c.agent_id for c in first_5}
        next_ids = {c.agent_id for c in next_5}
        assert first_ids.isdisjoint(next_ids)

    def test_negative_limit_raises(self, tmp_path: Path) -> None:
        store = AgentCardStore(_TinyDB(tmp_path / "p.db"))
        with pytest.raises(ValueError):
            store.list_all(limit=-1)
        with pytest.raises(ValueError):
            store.list_all(offset=-1)

    def test_count_cheap_enumeration(self, tmp_path: Path) -> None:
        store = AgentCardStore(_TinyDB(tmp_path / "p.db"))
        for i in range(7):
            store.save(_make_card(i))
        assert store.count() == 7


# ---------------------------------------------------------------------------
# V1 boundedness — MarketplaceAgentCard.capabilities size cap
# ---------------------------------------------------------------------------


class TestCapabilitiesCap:
    def test_under_cap_accepted(self) -> None:
        caps = [{"id": f"c{i}"} for i in range(99)]
        _ = MarketplaceAgentCard(
            agent_id=new_agent_id(),
            agent_url="https://x",
            agent_name="n",
            capabilities=caps,
            content_hash="h",
        )  # no raise

    def test_over_cap_rejected(self) -> None:
        caps = [{"id": f"c{i}"} for i in range(101)]
        with pytest.raises(ValidationError):
            MarketplaceAgentCard(
                agent_id=new_agent_id(),
                agent_url="https://x",
                agent_name="n",
                capabilities=caps,
                content_hash="h",
            )


# ---------------------------------------------------------------------------
# V4 contrastive — CoherenceSnapshot.is_degenerate + variance
# ---------------------------------------------------------------------------


class TestDegeneracy:
    def test_no_data_is_degenerate(self) -> None:
        snap = CoherenceSnapshot(R=1.0, window_size=0)
        assert snap.is_degenerate is True
        assert snap.has_data is False

    def test_single_sample_not_degenerate(self) -> None:
        # One sample is unknown-but-not-flagged
        snap = read_chain_coherence(_chain_with([0.5]))
        assert snap.window_size == 1
        assert snap.is_degenerate is False

    def test_constant_multi_sample_is_degenerate(self) -> None:
        snap = read_chain_coherence(_chain_with([0.5, 0.5, 0.5, 0.5]))
        assert snap.window_size == 4
        assert snap.variance == pytest.approx(0.0)
        assert snap.is_degenerate is True

    def test_varying_multi_sample_not_degenerate(self) -> None:
        snap = read_chain_coherence(_chain_with([0.1, 0.5, 0.9]))
        assert snap.variance > 0
        assert snap.is_degenerate is False

    def test_near_constant_still_degenerate(self) -> None:
        # Below 1e-6 variance threshold
        snap = read_chain_coherence(_chain_with([0.5, 0.5 + 1e-8, 0.5 - 1e-8]))
        assert snap.is_degenerate is True


def _chain_with(values: list[float]) -> InteractionChain:
    chain = InteractionChain()
    for v in values:
        chain.record(
            ActionType.LLM_REPLY,
            "t",
            input={},
            output={},
            success=True,
            kuramoto_r=v,
        )
    return chain


# ---------------------------------------------------------------------------
# V7 end-to-end — probe + auto-attach + CoherenceGate in one scenario
# ---------------------------------------------------------------------------


class TestEndToEndGateRealization:
    def test_probe_drives_gate_deny(self) -> None:
        """Probe reports low R → gate denies.

        This is the traceable "G realized" scenario the peer review
        asked for. A caller registers a coherence probe on the chain,
        records several LLM_REPLY steps (which auto-populate kuramoto_r
        from the probe), then invokes a @coherence_gate-protected
        function. The function denies because recent R is below
        threshold.
        """
        chain = InteractionChain()
        # Probe schedule chosen so the tail-window mean is unambiguously
        # below the gate threshold of 0.5.
        schedule = iter([0.3, 0.2, 0.1, 0.2, 0.1])

        def _probe(**kw: Any) -> dict[str, float]:
            try:
                return {"kuramoto_r": next(schedule)}
            except StopIteration:
                return {"kuramoto_r": 0.1}

        chain.register_coherence_probe(_probe)

        # Generate 5 auto-attach-eligible steps — each invokes the probe.
        for i in range(5):
            chain.record(ActionType.LLM_REPLY, f"step-{i}", success=True)

        # Confirm kuramoto_r was populated from the probe
        kr = [s.kuramoto_r for s in chain.steps if s.kuramoto_r is not None]
        assert len(kr) == 5
        # Mean ≈ 0.18 — clearly below 0.5

        @coherence_gate(min_R=0.5, feature="training.update")
        def _gated() -> str:
            return "allowed"

        with pytest.raises(CoherenceError) as exc_info:
            _gated(_gate_chain=chain)
        assert exc_info.value.current_R < 0.5

    def test_probe_drives_gate_allow(self) -> None:
        """Probe reports high R → gate allows."""
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"kuramoto_r": 0.85})

        # A handful of steps
        for i in range(4):
            chain.record(ActionType.LLM_REPLY, f"s-{i}", success=True)

        @coherence_gate(min_R=0.5)
        def _gated() -> str:
            return "allowed"

        assert _gated(_gate_chain=chain) == "allowed"

    def test_degenerate_probe_surfaces_but_does_not_force_deny(self) -> None:
        """Probe returns constant R=1.0 → is_degenerate=True BUT gate allows.

        Pins two behaviors simultaneously:
        1. read_chain_coherence flags the constant probe as degenerate
           (variance == 0 over multi-sample window).
        2. The gate's allow/deny logic is still based on R vs min_R;
           degeneracy is an INFORMATIONAL signal, not a blocker. A
           constant R=1.0 probe passes any threshold < 1.0.

        Consumers that want to downweight/disable admission under
        degenerate signal must query ``snap.is_degenerate`` explicitly —
        the gate does not do this implicitly (intentional, so existing
        call sites don't change behavior under probe swap).
        """
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"kuramoto_r": 1.0})
        for i in range(5):
            chain.record(ActionType.LLM_REPLY, f"d-{i}", success=True)

        snap = read_chain_coherence(chain)
        assert snap.is_degenerate is True  # variance=0 over 5 samples
        assert snap.R == pytest.approx(1.0)

        @coherence_gate(min_R=0.5)
        def _gated() -> str:
            return "allowed"

        # Gate allows — R=1.0 >= 0.5, regardless of degeneracy.
        assert _gated(_gate_chain=chain) == "allowed"
