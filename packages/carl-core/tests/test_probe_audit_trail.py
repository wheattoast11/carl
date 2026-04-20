"""v0.11 Fano V5 — probe audit trail.

Closes the witnessability gap: a coherence probe that populates
phi/kuramoto_r/channel_coherence on a Step now leaves a Step.probe_call
fingerprint (probe_name + inputs_sha256 + output_sha256 + populated
field list). Digests preserve BITC axiom 1 bounded-support.
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain


class TestProbeCallPopulation:
    def test_no_probe_leaves_probe_call_none(self) -> None:
        chain = InteractionChain()
        step = chain.record(ActionType.LLM_REPLY, "no-probe")
        assert step.probe_call is None

    def test_probe_returning_kuramoto_r_populates_probe_call(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"kuramoto_r": 0.77})
        step = chain.record(ActionType.LLM_REPLY, "t")
        assert step.probe_call is not None
        assert step.probe_call["populated"] == ["kuramoto_r"]
        assert isinstance(step.probe_call["probe_name"], str)
        assert len(step.probe_call["inputs_sha256"]) == 12
        assert len(step.probe_call["output_sha256"]) == 12

    def test_probe_populates_multiple_fields(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(
            lambda **kw: {"phi": 0.1, "kuramoto_r": 0.5, "channel_coherence": {"x": 0.9}}
        )
        step = chain.record(ActionType.TRAINING_STEP, "t")
        assert step.probe_call is not None
        assert set(step.probe_call["populated"]) == {
            "phi",
            "kuramoto_r",
            "channel_coherence",
        }

    def test_probe_returning_non_coherence_keys_does_not_populate(self) -> None:
        chain = InteractionChain()
        # Probe returns unrelated keys → no coherence fields set, no probe_call
        chain.register_coherence_probe(lambda **kw: {"unrelated": "value"})
        step = chain.record(ActionType.LLM_REPLY, "t")
        assert step.phi is None
        assert step.probe_call is None

    def test_probe_exception_leaves_probe_call_none(self) -> None:
        chain = InteractionChain()

        def _boom(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("probe failed")

        chain.register_coherence_probe(_boom)
        step = chain.record(ActionType.LLM_REPLY, "t")
        assert step.phi is None
        assert step.probe_call is None

    def test_explicit_coherence_kwargs_skip_probe_and_audit(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"kuramoto_r": 0.2})
        step = chain.record(ActionType.LLM_REPLY, "t", kuramoto_r=0.9)
        # Explicit kwarg wins; probe not invoked; no audit entry
        assert step.kuramoto_r == pytest.approx(0.9)
        assert step.probe_call is None


class TestProbeCallSerialization:
    def test_to_dict_includes_probe_call(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"phi": 0.5})
        step = chain.record(ActionType.LLM_REPLY, "t")
        d = step.to_dict()
        assert "probe_call" in d
        assert d["probe_call"] is not None
        assert d["probe_call"]["populated"] == ["phi"]

    def test_to_dict_probe_call_none_when_no_probe(self) -> None:
        chain = InteractionChain()
        step = chain.record(ActionType.CLI_CMD, "t")
        d = step.to_dict()
        assert d["probe_call"] is None


class TestProbeCallBoundedness:
    def test_digest_fields_are_bounded_length(self) -> None:
        """Fano V1 boundedness: digests are fixed-length, not full payloads."""
        chain = InteractionChain()
        chain.register_coherence_probe(
            lambda **kw: {"kuramoto_r": 0.5, "payload": "x" * 10_000}
        )
        step = chain.record(
            ActionType.LLM_REPLY,
            "big",
            input={"huge": "y" * 100_000},
        )
        assert step.probe_call is not None
        # Digests are exactly 12 hex chars regardless of input size
        assert len(step.probe_call["inputs_sha256"]) == 12
        assert len(step.probe_call["output_sha256"]) == 12

    def test_probe_call_determinism(self) -> None:
        """Same inputs → same digest. Reproducibility check."""

        def _probe(**kw: Any) -> dict[str, Any]:
            return {"kuramoto_r": 0.5}

        c1 = InteractionChain()
        c1.register_coherence_probe(_probe)
        s1 = c1.record(ActionType.LLM_REPLY, "same-name")

        c2 = InteractionChain()
        c2.register_coherence_probe(_probe)
        s2 = c2.record(ActionType.LLM_REPLY, "same-name")

        # Same action + name + input/output + probe output → same digests
        assert s1.probe_call is not None and s2.probe_call is not None
        assert s1.probe_call["inputs_sha256"] == s2.probe_call["inputs_sha256"]
        assert s1.probe_call["output_sha256"] == s2.probe_call["output_sha256"]
