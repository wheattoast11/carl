"""T3 — Step.eml_tree audit-hook round-trip tests.

Pins down the contract on the v0.9 ``Step.eml_tree`` field:

1. A Step without ``eml_tree`` serializes identically to the pre-v0.9
   format (the ``eml_tree`` key is absent, not present-but-None).
2. A Step with a populated ``eml_tree`` survives
   ``to_dict -> from_dict`` round-trip byte-for-byte.
3. ``to_jsonl -> parse`` preserves the ``eml_tree`` through a full
   serialize/deserialize cycle.
4. Existing callsite contracts are preserved — every pre-existing
   Step construction still works unchanged.
5. The canonical content hash of a chain is sensitive to the eml_tree
   payload, so any mutation to the audit tree changes the chain's
   witness fingerprint.
"""
from __future__ import annotations

import json

from carl_core.hashing import content_hash
from carl_core.interaction import ActionType, InteractionChain, Step


# ---------------------------------------------------------------------------
# 1. Default shape unchanged — legacy serializations stay byte-identical.
# ---------------------------------------------------------------------------


class TestDefaultShapeUnchanged:
    def test_step_without_eml_tree_defaults_to_none(self) -> None:
        step = Step(action=ActionType.CLI_CMD, name="legacy.call")
        assert step.eml_tree is None

    def test_to_dict_omits_eml_tree_key_when_none(self) -> None:
        # Critical: the key is ABSENT (not present-but-None) so that the
        # wire format is byte-identical to pre-v0.9 persistence.
        step = Step(action=ActionType.CLI_CMD, name="legacy.call")
        d = step.to_dict()
        assert "eml_tree" not in d

    def test_record_defaults_preserve_legacy_shape(self) -> None:
        chain = InteractionChain()
        s = chain.record(ActionType.TOOL_CALL, "tool.call")
        assert s.eml_tree is None
        d = s.to_dict()
        assert "eml_tree" not in d


# ---------------------------------------------------------------------------
# 2. Populated eml_tree round-trips byte-identically.
# ---------------------------------------------------------------------------


class TestEmlTreeRoundTrip:
    def test_dict_roundtrip_preserves_tree(self) -> None:
        tree: dict[str, object] = {
            "op": "eml",
            "children": [
                {"op": "leaf", "x": 0.5, "y": 0.3},
                {"op": "leaf", "x": 0.7, "y": 0.2},
            ],
            "score": 1.21,
            "depth": 2,
        }
        step = Step(action=ActionType.REWARD, name="reward.eml", eml_tree=tree)
        d = step.to_dict()
        assert d["eml_tree"] == tree

        # Chain round-trip carries it through.
        chain = InteractionChain()
        chain.steps.append(step)
        revived = InteractionChain.from_dict(chain.to_dict())
        assert len(revived.steps) == 1
        assert revived.steps[0].eml_tree == tree

    def test_jsonl_roundtrip_preserves_tree(self) -> None:
        tree: dict[str, object] = {
            "op": "eml",
            "children": [{"op": "leaf", "x": 0.1, "y": 0.9}],
        }
        chain = InteractionChain()
        chain.record(
            ActionType.TRAINING_STEP,
            name="train.eml.step",
            input={"batch": 0},
            output={"loss": 0.42},
        )
        # Attach tree to the freshly-recorded step (record() does not
        # currently thread eml_tree — callers mutate the returned step).
        chain.steps[-1].eml_tree = tree

        blob = chain.to_jsonl()
        lines = blob.split("\n")
        assert len(lines) == 2  # header + 1 step
        header = json.loads(lines[0])
        row = json.loads(lines[1])
        assert row["eml_tree"] == tree

        # And reconstructing from the dict form yields the same tree.
        revived = InteractionChain.from_dict(
            {
                "chain_id": header["chain_id"],
                "started_at": header["started_at"],
                "steps": [row],
                "context": header.get("context", {}),
            }
        )
        assert revived.steps[0].eml_tree == tree

    def test_roundtrip_idempotent(self) -> None:
        tree = {"op": "eml", "x": 0.5, "y": 0.3, "children": []}
        step = Step(action=ActionType.EVAL_PHASE, name="eval.eml", eml_tree=tree)
        chain = InteractionChain()
        chain.steps.append(step)

        once = chain.to_dict()
        twice = InteractionChain.from_dict(once).to_dict()
        # Steps payload should match byte-for-byte after a second roundtrip.
        assert once["steps"] == twice["steps"]


# ---------------------------------------------------------------------------
# 3. Legacy fields still work — every pre-existing Step attribute keeps
#    its prior behaviour.
# ---------------------------------------------------------------------------


class TestLegacyFieldsPreserved:
    def test_phi_kuramoto_channel_coherence_still_work(self) -> None:
        cc = {"phi_mean": 0.42, "latency_ms": 75.0}
        step = Step(
            action=ActionType.EXTERNAL,
            name="legacy.combined",
            phi=0.42,
            kuramoto_r=0.77,
            channel_coherence=cc,
            eml_tree={"op": "eml", "score": 1.1},
        )
        chain = InteractionChain()
        chain.steps.append(step)
        revived = InteractionChain.from_dict(chain.to_dict())
        r = revived.steps[0]
        assert r.phi == 0.42
        assert r.kuramoto_r == 0.77
        assert r.channel_coherence == cc
        assert r.eml_tree == {"op": "eml", "score": 1.1}

    def test_success_rate_unaffected_by_eml_tree(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.CLI_CMD, "ok", success=True)
        chain.record(ActionType.CLI_CMD, "fail", success=False)
        chain.steps[-1].eml_tree = {"op": "eml", "score": 0.01}
        assert chain.success_rate() == 0.5


# ---------------------------------------------------------------------------
# 4. Chain hash sensitivity — adding / mutating the tree changes the hash.
# ---------------------------------------------------------------------------


class TestChainHashSensitivity:
    def _chain_with_step(self, tree: dict | None) -> InteractionChain:
        chain = InteractionChain(chain_id="fixed_chain_id")
        step = Step(
            action=ActionType.REWARD,
            name="hash.probe",
            step_id="fixed_step_id",
            started_at=_fixed_time(),
            eml_tree=tree,
        )
        chain.steps.append(step)
        chain.started_at = _fixed_time()
        return chain

    def test_absent_tree_hash_stable(self) -> None:
        h1 = content_hash(self._chain_with_step(None).to_dict())
        h2 = content_hash(self._chain_with_step(None).to_dict())
        assert h1 == h2

    def test_populated_tree_changes_hash(self) -> None:
        baseline = content_hash(self._chain_with_step(None).to_dict())
        with_tree = content_hash(
            self._chain_with_step({"op": "eml", "score": 1.0}).to_dict()
        )
        assert baseline != with_tree

    def test_different_trees_have_different_hashes(self) -> None:
        a = content_hash(
            self._chain_with_step({"op": "eml", "score": 1.0}).to_dict()
        )
        b = content_hash(
            self._chain_with_step({"op": "eml", "score": 2.0}).to_dict()
        )
        assert a != b


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------


def _fixed_time():
    from datetime import datetime, timezone

    return datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)
