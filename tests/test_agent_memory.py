"""Tests for CARLAgent constitution + memory integration (WS-M5).

Covers:
  * Backward-compat construction (no memory / no constitution).
  * Constitutional system-prompt injection (single compile, topic filtering).
  * Memory recall on user turns (layer filter, score threshold, chain steps).
  * ``remember()`` writes to the store AND records MEMORY_WRITE into the chain.
  * Auto-remember on strong-signal phrases ("always", "never", "remember", "note:").
  * ``suggest_learnings()`` parses JSON array replies; returns [] on malformed.

The Anthropic SDK is mocked throughout — no network calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock

import pytest

from carl_core.interaction import ActionType
from carl_core.memory import MemoryLayer, MemoryStore
from carl_studio.chat_agent import CARLAgent
from carl_studio.constitution import Constitution, ConstitutionalRule


# =========================================================================
# SDK stream stubs (subset of test_chat_agent_robustness helpers)
# =========================================================================


class _Delta:
    def __init__(self, text: str) -> None:
        self.type = "text_delta"
        self.text = text


class _Event:
    def __init__(self, text: str) -> None:
        self.type = "content_block_delta"
        self.delta = _Delta(text)


class _Usage:
    def __init__(
        self,
        *,
        input_tokens: int = 50,
        output_tokens: int = 10,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(
        self,
        content: Iterable[Any],
        *,
        stop_reason: str = "end_turn",
        usage: _Usage | None = None,
    ) -> None:
        self.content = list(content)
        self.stop_reason = stop_reason
        self.usage = usage or _Usage()


class _StubStream:
    def __init__(
        self,
        events: Iterable[Any] | None = None,
        final: _Response | None = None,
    ) -> None:
        self._events = list(events or [])
        self._final = final

    def __enter__(self) -> _StubStream:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:
        yield from self._events

    def get_final_message(self) -> _Response:
        if self._final is None:
            return _Response([_TextBlock("ok")], usage=_Usage())
        return self._final


def _make_client(stream: _StubStream) -> MagicMock:
    client = MagicMock()
    client.messages.stream = MagicMock(return_value=stream)
    return client


def _stub_constitution(rules: list[ConstitutionalRule]) -> Constitution:
    """Build a Constitution with a known rule set, bypassing disk."""
    return Constitution(rules=rules)


# =========================================================================
# 1. Backward compat
# =========================================================================


class TestBackwardCompat:
    def test_agent_builds_without_memory_or_constitution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing store + failed Constitution.load() must not break init."""
        monkeypatch.setattr(
            "carl_studio.constitution.Constitution.load",
            classmethod(lambda cls, **_: (_ for _ in ()).throw(RuntimeError("no disk"))),
        )

        class _BustedStore:
            def __init__(self, *a: Any, **kw: Any) -> None:
                raise RuntimeError("no perms")

        monkeypatch.setattr("carl_core.memory.MemoryStore", _BustedStore)

        agent = CARLAgent(model="test-m", _client=MagicMock())
        assert agent._constitution is None
        assert agent._memory is None
        # Still functional: provider_info still returns, cost summary still works.
        assert agent.provider_info["model"] == "test-m"
        assert agent.cost_summary["turn_count"] == 0


# =========================================================================
# 2. Constitution injection into system prompt
# =========================================================================


class TestConstitutionInjection:
    def test_rule_text_appears_in_system_prompt(self, tmp_path: Path) -> None:
        rule = ConstitutionalRule(
            id="test.no_emoji",
            text="Do not emit emoji in output.",
            priority=95,
            tags=["style"],
        )
        constitution = _stub_constitution([rule])
        agent = CARLAgent(
            model="test-m",
            _client=MagicMock(),
            constitution=constitution,
            memory_store=MemoryStore(tmp_path / "mem"),
        )
        prompt = agent._build_system_prompt()
        assert "Do not emit emoji in output." in prompt
        assert "test.no_emoji" in prompt

    def test_high_priority_rules_present_even_when_topic_filter(
        self, tmp_path: Path
    ) -> None:
        """Priority>=90 rules bypass topic filtering."""
        hi = ConstitutionalRule(
            id="ops.never_leak_secrets",
            text="Never log secrets.",
            priority=100,
            tags=["security"],
        )
        lo = ConstitutionalRule(
            id="style.bullet_first",
            text="Lead with a bullet.",
            priority=40,
            tags=["style"],
        )
        constitution = _stub_constitution([hi, lo])

        class _Frame:
            active = True
            domain = "pharma"
            function = "forecasting"
            role = "analyst"
            objectives: list[str] = []

            def attention_query(self) -> str:
                return "pharma / forecasting"

        agent = CARLAgent(
            model="test-m",
            _client=MagicMock(),
            constitution=constitution,
            memory_store=MemoryStore(tmp_path / "mem"),
            frame=_Frame(),
        )
        prompt = agent._build_system_prompt()
        assert "Never log secrets." in prompt
        # Style rule isn't tagged with the frame's topics and isn't priority>=90,
        # so it should NOT appear.
        assert "Lead with a bullet." not in prompt

    def test_prompt_compile_is_cached_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rule = ConstitutionalRule(
            id="cache.hit",
            text="cached rule text",
            priority=80,
        )
        constitution = _stub_constitution([rule])
        agent = CARLAgent(
            model="test-m",
            _client=MagicMock(),
            constitution=constitution,
            memory_store=MemoryStore(tmp_path / "mem"),
        )

        # Spy on Constitution.compile_system_prompt at the class level —
        # Pydantic locks instance assignment for declared methods.
        calls = {"n": 0}
        orig = Constitution.compile_system_prompt

        def _spy(self: Constitution, *args: Any, **kwargs: Any) -> str:
            calls["n"] += 1
            return orig(self, *args, **kwargs)

        monkeypatch.setattr(Constitution, "compile_system_prompt", _spy)
        agent._build_system_prompt()
        agent._build_system_prompt()
        agent._build_system_prompt()
        assert calls["n"] == 1


# =========================================================================
# 3. Memory recall on user turn
# =========================================================================


class TestMemoryRecall:
    def _seed_memory(self, store: MemoryStore) -> None:
        store.write(
            "We decided to use gradient checkpointing for large models.",
            layer=MemoryLayer.LONG,
            tags={"training"},
        )
        store.write(
            "Past session: bucket territories by revenue quartile.",
            layer=MemoryLayer.WORKING,
            tags={"sales"},
        )

    def test_recall_injects_note_into_user_turn(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path / "mem")
        self._seed_memory(store)

        stream = _StubStream(final=_Response([_TextBlock("ack")]))
        agent = CARLAgent(
            model="test-m",
            _client=_make_client(stream),
            memory_store=store,
            constitution=None,
        )
        list(agent.chat("How should I bucket territories?"))
        # The stored prompt for the user turn should include the recall block.
        user_turns = [m for m in agent._messages if m.get("role") == "user"]
        assert user_turns
        injected = str(user_turns[0]["content"])
        assert "# Recalled context" in injected
        assert "territories" in injected.lower() or "territorie" in injected.lower()

    def test_no_recall_when_below_threshold(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path / "mem")
        store.write(
            "completely unrelated xyzzy notes about gardening",
            layer=MemoryLayer.LONG,
            tags={"hobby"},
        )
        stream = _StubStream(final=_Response([_TextBlock("ok")]))
        agent = CARLAgent(
            model="test-m",
            _client=_make_client(stream),
            memory_store=store,
            constitution=None,
        )
        list(agent.chat("train an llm for math"))
        user_turns = [m for m in agent._messages if m.get("role") == "user"]
        assert user_turns
        # Recall block must not appear since similarity is ~0.
        assert "# Recalled context" not in str(user_turns[0]["content"])

    def test_memory_read_step_recorded_per_item(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path / "mem")
        # Seed two matching items.
        store.write(
            "Token budgets for training: pick 64k as max_tokens per turn.",
            layer=MemoryLayer.LONG,
            tags={"training"},
        )
        store.write(
            "Training checkpoint cadence: every 500 steps for long runs.",
            layer=MemoryLayer.WORKING,
            tags={"training"},
        )

        stream = _StubStream(final=_Response([_TextBlock("ok")]))
        agent = CARLAgent(
            model="test-m",
            _client=_make_client(stream),
            memory_store=store,
        )
        # Force the chain to exist eagerly.
        chain = agent._get_chain()
        assert chain is not None

        list(agent.chat("How should we configure training checkpoints and token budgets?"))

        reads = [s for s in chain.steps if s.action == ActionType.MEMORY_READ]
        assert len(reads) >= 1
        # Each read step should carry the recalled content in its output.
        assert all(r.output for r in reads)


# =========================================================================
# 4. remember() — MEMORY_WRITE recording
# =========================================================================


class TestRemember:
    def test_remember_writes_and_records_step(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path / "mem")
        agent = CARLAgent(
            model="test-m",
            _client=MagicMock(),
            memory_store=store,
            constitution=None,
        )
        chain = agent._get_chain()
        assert chain is not None

        item = agent.remember(
            "Use BF16 for Hopper GPUs.",
            layer=MemoryLayer.LONG,
            tags={"hardware"},
        )
        assert item is not None
        assert item.content == "Use BF16 for Hopper GPUs."
        persisted = store.list_layer(MemoryLayer.LONG)
        assert any(p.content == "Use BF16 for Hopper GPUs." for p in persisted)

        writes = [s for s in chain.steps if s.action == ActionType.MEMORY_WRITE]
        assert writes, "expected MEMORY_WRITE step"
        assert writes[-1].name == item.id

    def test_remember_returns_none_when_store_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force bootstrap failure.
        monkeypatch.setattr(
            "carl_core.memory.MemoryStore",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no disk")),
        )
        agent = CARLAgent(model="test-m", _client=MagicMock(), constitution=None)
        assert agent._memory is None
        assert agent.remember("anything") is None


# =========================================================================
# 5. Auto-remember on strong-signal phrases
# =========================================================================


class TestAutoRemember:
    @pytest.mark.parametrize(
        "utterance",
        [
            "Always prefer BF16 over FP32 on H100s.",
            "Never log the ANTHROPIC_API_KEY.",
            "Please remember: use rank 16 for LoRA.",
            "note: batch size must stay under 64 on A100.",
        ],
    )
    def test_signal_phrases_trigger_auto_remember(
        self, tmp_path: Path, utterance: str,
    ) -> None:
        store = MemoryStore(tmp_path / "mem")
        stream = _StubStream(final=_Response([_TextBlock("ack")]))
        agent = CARLAgent(
            model="test-m",
            _client=_make_client(stream),
            memory_store=store,
            constitution=None,
        )
        list(agent.chat(utterance))
        # SHORT is the default auto-remember layer.
        short_items = store.list_layer(MemoryLayer.SHORT)
        assert any(
            si.content == utterance and "auto_remember" in si.tags
            for si in short_items
        ), f"expected auto-remember for: {utterance}"

    def test_neutral_utterance_does_not_auto_remember(
        self, tmp_path: Path,
    ) -> None:
        store = MemoryStore(tmp_path / "mem")
        stream = _StubStream(final=_Response([_TextBlock("ok")]))
        agent = CARLAgent(
            model="test-m",
            _client=_make_client(stream),
            memory_store=store,
            constitution=None,
        )
        list(agent.chat("What's the weather?"))
        assert store.list_layer(MemoryLayer.SHORT) == []


# =========================================================================
# 6. suggest_learnings()
# =========================================================================


class TestSuggestLearnings:
    def _agent_with_fake_reply(
        self,
        tmp_path: Path,
        reply_text: str,
        constitution: Constitution | None = None,
    ) -> CARLAgent:
        client = MagicMock()
        # suggest_learnings uses .messages.create, not .stream.
        fake_response = _Response([_TextBlock(reply_text)])
        client.messages.create = MagicMock(return_value=fake_response)
        # Populate the stream path too so the chat() call path stays clean.
        client.messages.stream = MagicMock(
            return_value=_StubStream(final=_Response([_TextBlock("ok")]))
        )
        agent = CARLAgent(
            model="test-m",
            _client=client,
            memory_store=MemoryStore(tmp_path / "mem"),
            constitution=constitution,
        )
        # Seed three turns so the guard in cli/_exit_session would fire too.
        agent._messages = [
            {"role": "user", "content": "train an llm"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok. what dataset?"}]},
            {"role": "user", "content": "always use rank 16 for LoRA."},
            {"role": "assistant", "content": [{"type": "text", "text": "noted."}]},
        ]
        agent._turn_count = 2
        return agent

    def test_returns_parsed_rules_from_json(self, tmp_path: Path) -> None:
        reply = json.dumps([
            {
                "id": "training.rank_16",
                "text": "Use LoRA rank 16 for small models by default.",
                "priority": 70,
                "tags": ["training", "lora"],
            },
            {
                "id": "ops.never_log_key",
                "text": "Never log the API key.",
                "priority": 95,
                "tags": ["security"],
            },
        ])
        agent = self._agent_with_fake_reply(tmp_path, reply)
        rules = agent.suggest_learnings()
        assert len(rules) == 2
        assert {r.id for r in rules} == {"training.rank_16", "ops.never_log_key"}

    def test_returns_empty_list_on_malformed_json(self, tmp_path: Path) -> None:
        agent = self._agent_with_fake_reply(tmp_path, "not json at all {{{")
        rules = agent.suggest_learnings()
        assert rules == []

    def test_extracts_first_json_array_when_surrounded_by_prose(
        self, tmp_path: Path,
    ) -> None:
        reply = (
            "Here are my suggestions:\n\n"
            + json.dumps([
                {
                    "id": "workflow.stage_first",
                    "text": "Declare stages upfront for complex tasks.",
                    "priority": 60,
                    "tags": ["workflow"],
                }
            ])
            + "\n\nLet me know which to keep."
        )
        agent = self._agent_with_fake_reply(tmp_path, reply)
        rules = agent.suggest_learnings()
        assert len(rules) == 1
        assert rules[0].id == "workflow.stage_first"

    def test_skips_entries_missing_required_fields(self, tmp_path: Path) -> None:
        reply = json.dumps([
            {"id": "", "text": "no id"},
            {"id": "ok.one", "text": "good entry", "priority": 50, "tags": []},
            {"text": "also missing id"},
        ])
        agent = self._agent_with_fake_reply(tmp_path, reply)
        rules = agent.suggest_learnings()
        assert len(rules) == 1
        assert rules[0].id == "ok.one"

    def test_empty_when_no_message_history(self, tmp_path: Path) -> None:
        reply = "[]"
        agent = self._agent_with_fake_reply(tmp_path, reply)
        agent._messages = []
        assert agent.suggest_learnings() == []
