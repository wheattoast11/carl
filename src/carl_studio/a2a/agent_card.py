"""CARLAgentCard — capability manifest for this CARL instance.

Serves as the discovery manifest for A2A task dispatch.
Other agents query this to know what tasks to send.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CARLAgentCard:
    """Describes what this CARL agent can do."""

    name: str = "carl-studio"
    version: str = "0.3.0"
    tier: str = "free"          # "free" | "paid"
    capabilities: list[str] = field(default_factory=lambda: [
        "train", "eval", "observe", "push", "bundle", "bench", "align", "learn",
    ])
    skills: list[str] = field(default_factory=list)
    endpoint: str = "stdio"     # "stdio" | "http://host:port"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def current(cls) -> "CARLAgentCard":
        """Build AgentCard from current settings + installed skills."""
        from carl_studio.settings import CARLSettings

        try:
            settings = CARLSettings.load()
            tier = settings.get_effective_tier().value
        except Exception:
            tier = "free"

        skills: list[str] = []
        try:
            from carl_studio.skills.builtins import BUILTIN_SKILLS  # type: ignore[import]
            from carl_studio.skills.runner import SkillRunner

            runner = SkillRunner()
            for s in BUILTIN_SKILLS:
                runner.register(s)
            skills = [s.name for s in runner.list_skills()]
            runner.close()
        except Exception:
            pass

        try:
            from carl_studio import __version__ as _ver
            version = _ver
        except Exception:
            version = "0.3.0"

        return cls(tier=tier, skills=skills, version=version)

    def to_a2a_spec(self) -> dict[str, Any]:
        """Return A2A protocol spec-compliant agent card."""
        from carl_studio.a2a.spec import agent_card_to_spec

        return agent_card_to_spec(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "CARLAgentCard":
        """Deserialize from JSON string."""
        parsed = json.loads(data)
        return cls(**parsed)
