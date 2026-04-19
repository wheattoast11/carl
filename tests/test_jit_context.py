"""Tests for ``carl_studio.jit_context``."""
from __future__ import annotations

import json

import pytest

from carl_studio.frame import WorkFrame
from carl_studio.jit_context import JITContext, TaskIntent, extract


def test_move_key_e_yields_explore_intent() -> None:
    ctx = extract("show me stuff", move_key="e")
    assert ctx.intent is TaskIntent.EXPLORE
    assert ctx.frame_patch.get("function") == "exploration"


def test_move_key_t_yields_train_intent() -> None:
    ctx = extract("whatever", move_key="t")
    assert ctx.intent is TaskIntent.TRAIN
    assert ctx.frame_patch.get("function") == "training"


def test_move_key_v_yields_eval_intent() -> None:
    ctx = extract("anything", move_key="v")
    assert ctx.intent is TaskIntent.EVAL
    assert ctx.frame_patch.get("function") == "evaluation"


def test_move_key_s_yields_sticky_intent() -> None:
    ctx = extract("queue this", move_key="s")
    assert ctx.intent is TaskIntent.STICKY
    assert ctx.frame_patch.get("function") == "task-management"


def test_move_key_uppercase_is_case_insensitive() -> None:
    ctx = extract("scan the repo", move_key="E")
    assert ctx.intent is TaskIntent.EXPLORE


def test_free_form_regex_classification_explore() -> None:
    ctx = extract("explore the repo for me")
    assert ctx.intent is TaskIntent.EXPLORE


def test_free_form_regex_classification_train() -> None:
    ctx = extract("please fine-tune llama on my dataset")
    assert ctx.intent is TaskIntent.TRAIN
    # Contains the word "dataset" → task_type pinned to "dataset".
    assert ctx.task_type == "dataset"


def test_free_form_regex_classification_eval() -> None:
    ctx = extract("benchmark the latest checkpoint")
    assert ctx.intent is TaskIntent.EVAL
    assert ctx.task_type == "checkpoint"


def test_free_form_regex_classification_sticky() -> None:
    ctx = extract("/sticky remind me to ship")
    assert ctx.intent is TaskIntent.STICKY


def test_extract_without_match_returns_free() -> None:
    ctx = extract("hello there friend")
    assert ctx.intent is TaskIntent.FREE
    assert ctx.frame_patch.get("function") == "general"


def test_extract_empty_string_returns_free() -> None:
    ctx = extract("")
    assert ctx.intent is TaskIntent.FREE
    assert ctx.domain == ""
    assert ctx.frame_patch == {"function": "general", "objectives": ["free:free"]}


def test_extract_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        extract(42)  # type: ignore[arg-type]


def test_primer_references_raw_input() -> None:
    raw = "train a lora on wiki_small"
    ctx = extract(raw)
    assert raw in ctx.primer
    assert "train" in ctx.primer


def test_primer_used_by_system_prompt_extension() -> None:
    ctx = extract("train a lora on wiki_small")
    ext = ctx.system_prompt_extension()
    assert ext == ctx.primer
    assert "train" in ext


def test_system_prompt_extension_fallback_when_primer_empty() -> None:
    # Build a context directly to bypass extract() populating primer.
    ctx = JITContext(
        intent=TaskIntent.EXPLORE,
        domain="my/repo",
        task_type="dataset",
        raw_input="",
        primer="",
    )
    ext = ctx.system_prompt_extension()
    assert "[jit-context]" in ext
    assert "intent=explore" in ext
    assert "domain=my/repo" in ext
    assert "task_type=dataset" in ext


def test_frame_patch_matches_WorkFrame_fields() -> None:
    ctx = extract("train a tiny model on trivial dataset")
    base = WorkFrame()
    updated = base.model_copy(update=ctx.frame_patch)
    # None of our patch keys should be foreign to WorkFrame.
    assert set(ctx.frame_patch).issubset(set(WorkFrame.model_fields))
    # function is always set; domain and objectives depend on inputs.
    assert updated.function == "training"
    if "domain" in ctx.frame_patch:
        assert updated.domain == ctx.frame_patch["domain"]
    if "objectives" in ctx.frame_patch:
        assert updated.objectives == ctx.frame_patch["objectives"]


def test_frame_patch_skips_empty_domain() -> None:
    # Whitespace-only input yields no tokens, so domain stays empty.
    ctx = extract("    ")
    assert "domain" not in ctx.frame_patch


def test_jit_context_serializes_cleanly() -> None:
    ctx = extract("evaluate my checkpoint on wiki_small", move_key="v")
    payload = ctx.model_dump(mode="json")
    # Must round-trip through json.
    raw = json.dumps(payload)
    back = json.loads(raw)
    assert back["intent"] == "eval"
    assert back["raw_input"] == "evaluate my checkpoint on wiki_small"
    assert back["frame_patch"]["function"] == "evaluation"
    assert isinstance(back["extracted_at"], str)


def test_task_type_falls_back_to_intent_when_no_keyword() -> None:
    ctx = extract("explore things", move_key="e")
    assert ctx.task_type == "explore"


def test_extracted_at_is_iso8601_utc() -> None:
    ctx = extract("anything")
    assert ctx.extracted_at.endswith("Z")
    # Length check: YYYY-MM-DDTHH:MM:SSZ = 20 chars
    assert len(ctx.extracted_at) == 20
