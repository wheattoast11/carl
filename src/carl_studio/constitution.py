"""Constitutional rules: the CRYSTAL layer of CARL's memory.

Rules compile from three sources in precedence order (later overrides earlier):
1. Repo CLAUDE.md + AGENTS.md (parsed from ## Rules / ## Policy sections)
2. Packaged defaults at src/carl_studio/rules.yaml
3. User overlay at ~/.carl/constitution.yaml (editable via `carl commit`)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from carl_core.errors import ConfigError, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_OVERLAY = Path.home() / ".carl" / "constitution.yaml"
PACKAGED_DEFAULTS = Path(__file__).parent / "rules.yaml"


class ConstitutionalRule(BaseModel):
    """A single compiled rule with priority and topic tags."""

    id: str = Field(
        description="Stable dot-separated identifier, e.g. 'security.no_secret_logging'"
    )
    text: str = Field(
        description="The rule text in natural language; appears in system prompt"
    )
    priority: int = Field(
        default=50, ge=0, le=100, description="0=overridable ... 100=inviolable"
    )
    tags: list[str] = Field(
        default_factory=list, description="Topic tags for resonance retrieval"
    )
    source: str = Field(
        default="user",
        description="Where this rule came from: claude_md|agents_md|default|user",
    )


class Constitution(BaseModel):
    """Compiled constitutional rules merged from repo, package, and user sources."""

    rules: list[ConstitutionalRule] = Field(default_factory=list)

    @classmethod
    def load(
        cls,
        *,
        repo_root: Path | None = None,
        user_overlay: Path | None = None,
    ) -> Constitution:
        """Load + merge rules from all sources.

        Later sources override earlier ones by rule id:
          1. CLAUDE.md / AGENTS.md in the repo root
          2. Packaged defaults (src/carl_studio/rules.yaml)
          3. User overlay YAML (~/.carl/constitution.yaml)
        """
        root = repo_root or REPO_ROOT
        overlay = user_overlay if user_overlay is not None else USER_OVERLAY
        merged: dict[str, ConstitutionalRule] = {}

        # 1. Parse repo markdown
        for md_name, source_name in (("CLAUDE.md", "claude_md"), ("AGENTS.md", "agents_md")):
            md_path = root / md_name
            if md_path.exists():
                try:
                    md_text = md_path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise ConfigError(
                        f"failed to read repo markdown at {md_path}",
                        code="carl.constitution.bad_markdown",
                        context={"path": str(md_path)},
                    ) from exc
                for rule in _parse_markdown_rules(md_text, source=source_name):
                    merged[rule.id] = rule

        # 2. Packaged defaults
        if PACKAGED_DEFAULTS.exists():
            for rule in _load_yaml_rules(PACKAGED_DEFAULTS, source="default"):
                merged[rule.id] = rule

        # 3. User overlay — corrupt YAML here must surface as a ConfigError with
        # the dedicated user-overlay code so callers can distinguish it from
        # packaged-default errors.
        if overlay.exists():
            try:
                for rule in _load_yaml_rules(overlay, source="user"):
                    merged[rule.id] = rule
            except Exception as exc:
                raise ConfigError(
                    f"failed to load user constitution at {overlay}",
                    code="carl.constitution.bad_user_overlay",
                    context={"path": str(overlay)},
                ) from exc

        return cls(
            rules=sorted(merged.values(), key=lambda r: (-r.priority, r.id))
        )

    def compile_system_prompt(
        self,
        *,
        topics: set[str] | None = None,
        max_rules: int = 40,
    ) -> str:
        """Render rules as a system-prompt snippet.

        If ``topics`` is given, only rules whose tags intersect with ``topics``
        are included, plus all high-priority rules (priority >= 90) regardless
        of topic. If ``topics`` is None, all rules are included up to ``max_rules``.
        """
        if max_rules <= 0:
            return ""

        selected: list[ConstitutionalRule] = []
        for rule in self.rules:
            if rule.priority >= 90 or topics is None:
                selected.append(rule)
            elif rule.tags and set(rule.tags) & topics:
                selected.append(rule)
            if len(selected) >= max_rules:
                break

        if not selected:
            return ""

        lines = [
            "# Carl constitution (inviolable unless superseded by user instruction)"
        ]
        for rule in selected:
            lines.append(f"- [{rule.id} | p{rule.priority}] {rule.text}")
        return "\n".join(lines)

    def append(
        self,
        rule: ConstitutionalRule,
        *,
        overlay: Path | None = None,
    ) -> None:
        """Persist a new or updated rule to the user overlay YAML.

        If a rule with the same id already exists, it is replaced (not duplicated).
        Parent directories are created if missing.
        """
        path = overlay if overlay is not None else USER_OVERLAY
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {"rules": []}
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict) and isinstance(loaded.get("rules"), list):
                    data = loaded
                elif isinstance(loaded, dict):
                    # Dict without a valid 'rules' list: reset.
                    data = {"rules": []}
                else:
                    # Non-mapping root is corrupt.
                    raise ConfigError(
                        f"user constitution is corrupt: {path}",
                        code="carl.constitution.corrupt_user_overlay",
                        context={"path": str(path)},
                    )
            except ConfigError:
                raise
            except Exception as exc:
                raise ConfigError(
                    f"user constitution is corrupt: {path}",
                    code="carl.constitution.corrupt_user_overlay",
                    context={"path": str(path)},
                ) from exc

        existing = [
            r
            for r in data["rules"]
            if isinstance(r, dict) and r.get("id") != rule.id
        ]
        existing.append(rule.model_dump(exclude={"source"}))
        data["rules"] = existing
        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")

    def topics(self) -> set[str]:
        """Union of all tags across rules."""
        out: set[str] = set()
        for r in self.rules:
            out.update(r.tags)
        return out


_MD_RULE_RE = re.compile(
    r"^\s*[-*]\s*(?:\*\*|__)?(?P<id>[a-z0-9][a-z0-9._-]+?)(?:\*\*|__)?\s*[:\-]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)

_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")

_SECTION_KEYWORDS = ("rules", "policy", "do not", "keep")

_HEADING_STOPWORDS = frozenset(
    {
        "rules",
        "policy",
        "do",
        "not",
        "the",
        "a",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "without",
    }
)


def _parse_markdown_rules(md: str, *, source: str) -> list[ConstitutionalRule]:
    """Extract rule-like list items from ## Rules/## Policy/## Do NOT sections.

    Accepts these patterns inside ``##`` sections whose title matches
    /rules?|policy|do not|keep/i:
      - id.path: rule text
      - **id.path**: rule text
      - plain bullet text (autoid assigned)
    """
    rules: list[ConstitutionalRule] = []
    in_section = False
    idx = 0
    tags_from_heading: list[str] = []

    for line in md.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip().lower()
            in_section = any(k in heading for k in _SECTION_KEYWORDS)
            tags_from_heading = [
                t
                for t in re.findall(r"[a-z0-9]+", heading)
                if t not in _HEADING_STOPWORDS
            ]
            continue
        if line.startswith("# "):
            # Top-level heading resets section tracking
            in_section = False
            tags_from_heading = []
            continue
        if not in_section or not line.strip():
            continue

        m = _MD_RULE_RE.match(line)
        if not m:
            bullet = _BULLET_RE.match(line)
            if bullet:
                idx += 1
                rules.append(
                    ConstitutionalRule(
                        id=f"{source}.{idx:03d}",
                        text=bullet.group(1).strip(),
                        priority=50,
                        tags=list(tags_from_heading),
                        source=source,
                    )
                )
            continue

        rid_raw = m.group("id").strip().lower()
        text = m.group("text").strip()

        # Sanity-check the extracted id — reject short / word-y "ids" so we don't
        # turn normal prose bullets into synthetic rules.
        if not _looks_like_rule_id(rid_raw):
            idx += 1
            rules.append(
                ConstitutionalRule(
                    id=f"{source}.{idx:03d}",
                    text=f"{rid_raw}: {text}",
                    priority=50,
                    tags=list(tags_from_heading),
                    source=source,
                )
            )
            continue

        rid = f"{source}.{rid_raw}" if "." not in rid_raw else rid_raw
        rules.append(
            ConstitutionalRule(
                id=rid,
                text=text,
                priority=60,
                tags=list(tags_from_heading),
                source=source,
            )
        )

    return rules


def _looks_like_rule_id(candidate: str) -> bool:
    """Return True if ``candidate`` looks like a dot-separated rule id.

    Avoids hijacking normal prose bullets like "- Command: do X" by requiring
    either a dot, an underscore, or a dash within the id itself.
    """
    if not candidate:
        return False
    if any(sep in candidate for sep in (".", "_", "-")):
        # Heuristic: reject very short segments that look like plain words.
        return len(candidate) >= 3
    return False


def _load_yaml_rules(path: Path, *, source: str) -> list[ConstitutionalRule]:
    """Parse a YAML file with a top-level ``rules:`` list into ConstitutionalRule objects."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"failed to read {path}",
            code="carl.constitution.bad_yaml",
            context={"path": str(path)},
        ) from exc

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValidationError(
            f"invalid YAML in {path}: {exc}",
            code="carl.constitution.bad_yaml",
            context={"path": str(path)},
        ) from exc

    if not isinstance(data, dict):
        raise ValidationError(
            f"constitution YAML root must be a mapping with 'rules': {path}",
            code="carl.constitution.bad_yaml",
            context={"path": str(path)},
        )
    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        raise ValidationError(
            f"'rules' must be a list in {path}",
            code="carl.constitution.bad_yaml",
            context={"path": str(path)},
        )

    out: list[ConstitutionalRule] = []
    for rr in raw_rules:
        if not isinstance(rr, dict):
            continue
        rr = dict(rr)  # copy so we don't mutate caller-visible data
        rr.setdefault("source", source)
        try:
            out.append(ConstitutionalRule(**rr))
        except Exception as exc:
            raise ValidationError(
                f"invalid rule entry in {path}: {exc}",
                code="carl.constitution.bad_rule",
                context={"path": str(path), "entry": rr},
            ) from exc
    return out


__all__ = [
    "Constitution",
    "ConstitutionalRule",
    "PACKAGED_DEFAULTS",
    "REPO_ROOT",
    "USER_OVERLAY",
]
