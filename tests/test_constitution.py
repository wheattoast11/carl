"""Tests for the constitutional rules compiler (CRYSTAL layer).

Covers:
- Loading + merging from repo markdown, packaged defaults, and user overlay.
- Corrupt user overlay raises a typed ConfigError with a stable code.
- System-prompt compilation: empty, topic filter, priority >=90 always included,
  max_rules cap.
- Overlay persistence: append, overwrite-in-place, parent dir creation.
- Markdown parser: scoped to ## Rules / ## Policy / ## Do NOT sections only.
- Deterministic sort order: priority DESC, id ASC.
- topics() returns the union of all tags.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carl_core.errors import ConfigError
from carl_studio.constitution import (
    Constitution,
    ConstitutionalRule,
    PACKAGED_DEFAULTS,
    _parse_markdown_rules,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_markdown_repo(root: Path, claude_md: str = "", agents_md: str | None = None) -> Path:
    """Create an isolated fake repo root with CLAUDE.md / AGENTS.md files."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "CLAUDE.md").write_text(claude_md, encoding="utf-8")
    if agents_md is not None:
        (root / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------


def test_load_uses_real_repo_claude_md(tmp_path: Path) -> None:
    """Loading from the actual repo root should return at least the packaged rules."""
    repo_root = Path(__file__).resolve().parents[1]
    missing_overlay = tmp_path / "nonexistent.yaml"

    const = Constitution.load(repo_root=repo_root, user_overlay=missing_overlay)

    assert const.rules, "expected at least packaged rules from the real repo"
    # The packaged default rule must be present.
    ids = {r.id for r in const.rules}
    assert "security.no_secret_logging" in ids


def test_load_missing_user_overlay_uses_other_sources(tmp_path: Path) -> None:
    """Missing user overlay should not error; rules come from repo + packaged defaults."""
    repo_root = _write_markdown_repo(tmp_path / "repo", claude_md="")
    missing_overlay = tmp_path / "does_not_exist.yaml"

    const = Constitution.load(repo_root=repo_root, user_overlay=missing_overlay)

    # At minimum the packaged defaults should be present.
    ids = {r.id for r in const.rules}
    assert "security.no_secret_logging" in ids
    assert "reliability.typed_errors" in ids


def test_load_corrupt_user_overlay_raises_config_error(tmp_path: Path) -> None:
    """A corrupt user overlay YAML must surface as ConfigError with stable code."""
    repo_root = _write_markdown_repo(tmp_path / "repo", claude_md="")
    overlay = tmp_path / "bad.yaml"
    overlay.write_text("rules: {this is: not a list\n- broken", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        Constitution.load(repo_root=repo_root, user_overlay=overlay)

    assert excinfo.value.code == "carl.constitution.bad_user_overlay"


# ---------------------------------------------------------------------------
# compile_system_prompt
# ---------------------------------------------------------------------------


def test_compile_empty_rules_returns_empty_string() -> None:
    const = Constitution(rules=[])
    assert const.compile_system_prompt() == ""
    assert const.compile_system_prompt(topics={"security"}) == ""


def test_compile_includes_high_priority_regardless_of_topic() -> None:
    """priority >=90 rules are injected even if topics filter would exclude them."""
    rules = [
        ConstitutionalRule(
            id="security.critical",
            text="Never leak secrets.",
            priority=100,
            tags=["security"],
        ),
        ConstitutionalRule(
            id="ux.nice",
            text="Be friendly.",
            priority=50,
            tags=["ux"],
        ),
    ]
    const = Constitution(rules=rules)

    prompt = const.compile_system_prompt(topics={"training"})

    assert "security.critical" in prompt
    assert "ux.nice" not in prompt


def test_compile_filters_by_topic() -> None:
    rules = [
        ConstitutionalRule(id="a.one", text="alpha", priority=50, tags=["training"]),
        ConstitutionalRule(id="b.two", text="beta", priority=50, tags=["eval"]),
        ConstitutionalRule(id="c.three", text="gamma", priority=50, tags=[]),
    ]
    const = Constitution(rules=rules)

    prompt = const.compile_system_prompt(topics={"training"})

    assert "a.one" in prompt
    assert "b.two" not in prompt
    assert "c.three" not in prompt


def test_compile_respects_max_rules_cap() -> None:
    rules = [
        ConstitutionalRule(
            id=f"cat.rule{i:03d}",
            text=f"rule {i}",
            priority=80 - i,  # strictly descending priority
            tags=["shared"],
        )
        for i in range(10)
    ]
    const = Constitution(rules=sorted(rules, key=lambda r: (-r.priority, r.id)))

    prompt = const.compile_system_prompt(max_rules=3)

    # Header line + 3 rule lines
    assert prompt.count("\n- [") == 3


# ---------------------------------------------------------------------------
# Overlay persistence
# ---------------------------------------------------------------------------


def test_append_writes_to_overlay_and_is_visible_on_reload(tmp_path: Path) -> None:
    repo_root = _write_markdown_repo(tmp_path / "repo", claude_md="")
    overlay = tmp_path / "overlay.yaml"

    const = Constitution.load(repo_root=repo_root, user_overlay=overlay)
    new_rule = ConstitutionalRule(
        id="user.custom.test",
        text="Custom user rule.",
        priority=75,
        tags=["custom"],
        source="user",
    )
    const.append(new_rule, overlay=overlay)

    # File exists and parses correctly.
    assert overlay.exists()
    data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    ids = [r["id"] for r in data["rules"]]
    assert "user.custom.test" in ids

    # Reloading picks it up.
    reloaded = Constitution.load(repo_root=repo_root, user_overlay=overlay)
    reloaded_ids = {r.id for r in reloaded.rules}
    assert "user.custom.test" in reloaded_ids


def test_append_overwrites_existing_rule_with_same_id(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    const = Constitution(rules=[])

    const.append(
        ConstitutionalRule(id="dup.rule", text="first", priority=50, tags=["a"]),
        overlay=overlay,
    )
    const.append(
        ConstitutionalRule(id="dup.rule", text="second", priority=80, tags=["b"]),
        overlay=overlay,
    )

    data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    matching = [r for r in data["rules"] if r["id"] == "dup.rule"]
    assert len(matching) == 1, "append must overwrite, not duplicate"
    assert matching[0]["text"] == "second"
    assert matching[0]["priority"] == 80


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    overlay = tmp_path / "deep" / "nested" / "overlay.yaml"
    assert not overlay.parent.exists()

    const = Constitution(rules=[])
    const.append(
        ConstitutionalRule(id="nest.test", text="hi", priority=50, tags=[]),
        overlay=overlay,
    )

    assert overlay.exists()
    assert overlay.parent.is_dir()


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------


def test_markdown_parser_extracts_rules_from_rules_and_policy_sections() -> None:
    md = """
# Heading

Ignore this paragraph.

## Rules

- security.no_leaks: Don't leak secrets.
- **style.clean**: Keep code tidy.

## Policy

- release.signed: Sign all release artifacts.

## Do NOT

- avoid.foo: Do not foo in production.
""".strip()

    rules = _parse_markdown_rules(md, source="claude_md")
    ids = {r.id for r in rules}

    # Dot-form ids get preserved; bare ids get namespaced with the source.
    assert "security.no_leaks" in ids
    assert "style.clean" in ids
    assert "release.signed" in ids
    assert "avoid.foo" in ids

    texts = {r.text for r in rules}
    assert "Don't leak secrets." in texts
    assert "Sign all release artifacts." in texts


def test_markdown_parser_ignores_bullets_outside_scoped_sections() -> None:
    md = """
## Overview

- this.should.not: be parsed.
- not_a_rule: either

## Rules

- foo.bar: Actually a rule.
""".strip()

    rules = _parse_markdown_rules(md, source="claude_md")
    ids = {r.id for r in rules}

    assert "foo.bar" in ids
    assert "this.should.not" not in ids
    # The "not_a_rule" bullet lives outside the section too, so it must be absent.
    assert not any(r.text == "either" for r in rules)


# ---------------------------------------------------------------------------
# Sorting / topics
# ---------------------------------------------------------------------------


def test_rules_sorted_by_priority_desc_then_id_asc(tmp_path: Path) -> None:
    repo_root = _write_markdown_repo(tmp_path / "repo", claude_md="")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {"id": "zzz.low", "text": "low", "priority": 10, "tags": []},
                    {"id": "aaa.high", "text": "high", "priority": 95, "tags": []},
                    {"id": "mmm.mid", "text": "mid", "priority": 95, "tags": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    const = Constitution.load(repo_root=repo_root, user_overlay=overlay)

    # Extract just the three user-added rules and verify their relative order.
    user_rules = [r for r in const.rules if r.source == "user"]
    ordered_ids = [r.id for r in user_rules]
    # aaa.high and mmm.mid both have priority 95; aaa.high comes first by id.
    assert ordered_ids.index("aaa.high") < ordered_ids.index("mmm.mid")
    assert ordered_ids.index("mmm.mid") < ordered_ids.index("zzz.low")

    # Priorities must be non-increasing across the full sorted list.
    priorities = [r.priority for r in const.rules]
    assert priorities == sorted(priorities, reverse=True)


def test_topics_returns_union_of_all_tags() -> None:
    const = Constitution(
        rules=[
            ConstitutionalRule(id="a.one", text="x", priority=50, tags=["security", "logging"]),
            ConstitutionalRule(id="b.two", text="y", priority=50, tags=["logging", "ux"]),
            ConstitutionalRule(id="c.three", text="z", priority=50, tags=[]),
        ]
    )

    assert const.topics() == {"security", "logging", "ux"}


# ---------------------------------------------------------------------------
# Sanity: packaged defaults file parses
# ---------------------------------------------------------------------------


def test_packaged_defaults_parses_cleanly() -> None:
    """The shipped rules.yaml must itself load without error."""
    assert PACKAGED_DEFAULTS.exists(), "rules.yaml must ship alongside constitution.py"
    data = yaml.safe_load(PACKAGED_DEFAULTS.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data["rules"], list)
    assert data["rules"], "packaged rules must not be empty"
