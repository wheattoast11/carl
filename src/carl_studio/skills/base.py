"""Base skill contract for CARL Studio merit badges.

Every CARL skill is a composable workflow that can earn a merit badge.
Skills are pure: execute() takes kwargs, returns SkillResult. No side effects
beyond what the skill explicitly does. Side effects (badge award, DB recording)
happen in SkillRunner, not here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class SkillResult(BaseModel):
    """Output from a skill execution."""

    skill_name: str
    success: bool
    badge_earned: bool = False
    metrics: dict[str, Any] = {}
    message: str = ""
    artifact: dict[str, Any] = {}  # optional output artifact


class SkillRun(BaseModel):
    """A historical record of a skill execution (hydrated from DB row)."""

    id: int
    skill_name: str
    run_id: str | None = None
    inputs: dict[str, object] = {}
    result: dict[str, object] = {}
    badge_earned: bool = False
    success: bool = False
    message: str = ""
    started_at: str = ""
    completed_at: str | None = None


class BaseSkill(ABC):
    """Abstract base for all CARL skills (merit badges).

    Skills are composable CARL workflows. Each skill maps to a summer camp
    merit badge. Earn a badge by running a skill that gates PASS.

    Subclasses MUST define all ClassVar attributes as class-level annotations
    with values — they are inspected by SkillRunner and the CLI.
    """

    name: ClassVar[str]          # e.g. "observer"
    badge: ClassVar[str]         # e.g. "Observer Badge"
    description: ClassVar[str]   # used for LLM routing + MCP tool description
    requires_tier: ClassVar[str] = "free"  # "free" or "paid"

    @abstractmethod
    def execute(self, **kwargs: Any) -> SkillResult:
        """Execute the skill. Returns SkillResult with badge_earned=True on PASS."""
        ...

    def to_mcp_schema(self) -> dict[str, Any]:
        """Return MCP tool schema for this skill."""
        return {
            "name": f"skill_{self.name}",
            "description": f"[Badge: {self.badge}] {self.description}",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }
